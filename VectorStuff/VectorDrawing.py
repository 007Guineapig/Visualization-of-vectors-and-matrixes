"""Vykresľovanie vektorov v 2D a 3D — extrahované z Application.py"""

import math
import numpy as np
from OpenGL.GL import *


class VectorDrawingMixin:
    """Mixin trieda pre vykresľovanie vektorov.
    Predpokladá self má: camera, vector_renderer, vector_manager, ui_renderer,
    vectors_as_points, background_dark, clock
    """

    # =========================================================================
    # HELPER METÓDY
    # =========================================================================

    def _get_arrow_size_2d(self):
        return max(0.15, min(0.12 * self.camera.ortho_scale, 2.0))

    def _get_point_radius_2d(self):
        return 0.07 * self.camera.ortho_scale / 6.5

    def _get_point_radius_3d(self):
        base = 0.07 * math.sqrt(self.camera.distance / 7.0)
        return max(base, 0.03)

    def _draw_single_arrow_2d(self, ox, oy, vx, vy, arrow_size, v_color, v_alpha, z=0.3):
        """Vykreslí jeden 2D vektor so šípkou"""
        vec_length = math.sqrt(vx * vx + vy * vy)
        if vec_length <= 0.1:
            return

        angle = math.atan2(vy, vx)
        line_end_x = ox + vx - arrow_size * 0.35 * math.cos(angle)
        line_end_y = oy + vy - arrow_size * 0.35 * math.sin(angle)

        glLineWidth(6)
        if v_alpha < 1.0:
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glColor4f(v_color[0], v_color[1], v_color[2], v_alpha)
        else:
            glColor3f(*v_color)

        glBegin(GL_LINES)
        glVertex3f(ox, oy, z)
        glVertex3f(line_end_x, line_end_y, z)
        glEnd()

        self.vector_renderer.draw_triangle_arrowhead_2d(
            ox + vx, oy + vy, angle, arrow_size, v_color, v_alpha, z=z
        )

        if v_alpha < 1.0:
            glDisable(GL_BLEND)

    def _draw_single_point_2d(self, x, y, radius, v_color, v_alpha, z=0.2):
        """Vykreslí jeden 2D bod"""
        self.vector_renderer.draw_circle_2d([x, y, z], radius=radius, color=v_color, alpha=v_alpha)

    def _draw_single_vector_3d(self, vec, offset, v_color, v_alpha, cam_pos):
        """Vykreslí jeden 3D vektor so šípkou"""
        glPushMatrix()
        glTranslatef(offset[0], offset[1], offset[2])

        end_point = (vec[0], vec[1] if len(vec) > 1 else 0, vec[2] if len(vec) > 2 else 0)
        vec_array = np.array(end_point)
        length = np.linalg.norm(vec_array)
        delta = 0.15

        if length > delta:
            vec_short = vec_array * (length - delta) / length
        else:
            vec_short = vec_array

        glLineWidth(6)
        if v_alpha < 1.0:
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glColor4f(v_color[0], v_color[1], v_color[2], v_alpha)
        else:
            glColor3f(*v_color)

        glBegin(GL_LINES)
        glVertex3f(0, 0, 0)
        glVertex3f(vec_short[0], vec_short[1], vec_short[2])
        glEnd()

        if v_alpha < 1.0:
            glDisable(GL_BLEND)

        self.vector_renderer.draw_conehead_3d(
            (0, 0, 0), end_point,
            color=v_color, size=0.7,
            camera_pos=cam_pos, radius=0.25,
            alpha=v_alpha
        )
        glPopMatrix()

    def _draw_single_point_3d(self, pos, radius, v_color, v_alpha):
        """Vykreslí jeden 3D bod"""
        self.vector_renderer.draw_sphere(pos, radius=radius, color=v_color, alpha=v_alpha)

    def _extract_vec_data(self, v, default_color=(1, 1, 1)):
        """Extrahuje bežné dáta z vektora"""
        vec = v['vec']
        offset = v.get('offset', [0, 0, 0])
        v_color = v.get('color', default_color)
        v_alpha = v.get('alpha', 1.0)
        return vec, offset, v_color, v_alpha

    def _iter_matrix_rows(self, vec, offset, row_offsets=None):
        """Iteruje cez riadky matice a vracia (x, y, z, ox, oy, oz)"""
        for i, row in enumerate(vec):
            x = row[0] if len(row) > 0 else 0
            y = row[1] if len(row) > 1 else 0
            z = row[2] if len(row) > 2 else 0

            if row_offsets and i < len(row_offsets):
                ro = row_offsets[i]
                ox = ro[0]
                oy = ro[1] if len(ro) > 1 else 0
                oz = ro[2] if len(ro) > 2 else 0
            else:
                ox = offset[0]
                oy = offset[1] if len(offset) > 1 else 0
                oz = offset[2] if len(offset) > 2 else 0

            yield x, y, z, ox, oy, oz

    # =========================================================================
    # 2D VYKRESĽOVANIE
    # =========================================================================

    def _draw_vectors_as_points_2d(self, vectors, default_color, z=0.2):
        """Vykreslí zoznam vektorov ako body v 2D"""
        radius = self._get_point_radius_2d()
        for v in vectors:
            vec, offset, v_color, v_alpha = self._extract_vec_data(v, default_color)

            if isinstance(vec[0], (int, float)):
                pos_x = vec[0] + offset[0]
                pos_y = (vec[1] if len(vec) > 1 else 0) + (offset[1] if len(offset) > 1 else 0)
                self._draw_single_point_2d(pos_x, pos_y, radius, v_color, v_alpha, z)
            else:
                row_offsets = v.get('row_offsets', None)
                for x, y, _, ox, oy, _ in self._iter_matrix_rows(vec, offset, row_offsets):
                    self._draw_single_point_2d(x + ox, y + oy, radius, v_color, v_alpha, z)

    def _draw_vectors_as_arrows_2d(self, vectors, default_color, z=0.3):
        """Vykreslí zoznam vektorov ako šípky v 2D"""
        arrow_size = self._get_arrow_size_2d()
        for v in vectors:
            vec, offset, v_color, v_alpha = self._extract_vec_data(v, default_color)

            if v_alpha < 1.0:
                glDepthMask(GL_FALSE)

            if isinstance(vec[0], (int, float)):
                ox = offset[0]
                oy = offset[1] if len(offset) > 1 else 0
                vx = vec[0]
                vy = vec[1] if len(vec) > 1 else 0
                self._draw_single_arrow_2d(ox, oy, vx, vy, arrow_size, v_color, v_alpha, z)
            else:
                row_offsets = v.get('row_offsets', None)
                for x, y, _, ox, oy, _ in self._iter_matrix_rows(vec, offset, row_offsets):
                    self._draw_single_arrow_2d(ox, oy, x, y, arrow_size, v_color, v_alpha, z)

            if v_alpha < 1.0:
                glDepthMask(GL_TRUE)

    def draw_vectors_2d(self, color):
        """Vykreslí vektory v 2D - S PODPOROU PRE SPAN"""
        from VectorStuff.VectorRenderer import VectorRenderer

        arrow_size = self._get_arrow_size_2d()
        # ===== ROZKLAD DO BÁZY MÁ NAJVYŠŠIU PRIORITU =====    <-- NOVÝ BLOK
        if self.vector_manager.decomposition_controller.active:
            vectors_to_draw = self.vector_manager.decomposition_controller.get_vectors_to_draw()
            if self.vectors_as_points:
                self._draw_vectors_as_points_2d(vectors_to_draw, color, z=0.2)
            else:
                self._draw_vectors_as_arrows_2d(vectors_to_draw, color, z=0.3)
            self.draw_vector_labels_2d(vectors_to_draw)
            return
        # ===== KONIEC NOVÉHO BLOKU =====
        # ===== SPAN MÁ PRIORITU =====
        if self.vector_manager.span_controller.active:
            vectors_to_draw = self.vector_manager.span_controller.get_current_vectors()
            persistent = [v for v in vectors_to_draw if v.get('is_persistent', False)]
            current = [v for v in vectors_to_draw if not v.get('is_persistent', False)]

            if self.vectors_as_points:
                self._draw_vectors_as_points_2d(persistent, (0.6, 0.2, 0.6), z=0.05)
                self._draw_vectors_as_points_2d(current, (1, 1, 1), z=0.3)
            else:
                # Perzistentné — čiary (nízke Z), potom šípky (vyššie Z)
                self._draw_span_persistent_arrows(persistent, arrow_size)
                # Aktuálne
                self._draw_vectors_as_arrows_2d(current, (1, 1, 1), z=0.2)

            self.draw_vector_labels_2d(current)
            return

        # ===== ANIMATION CONTROLLER =====
        if self.vector_manager.animation_controller.current_operation:
            vectors_to_draw = self.vector_manager.animation_controller.get_vectors_to_draw()
            vectors_sorted = sorted(vectors_to_draw, key=lambda v: v.get('alpha', 1.0))

            if self.vectors_as_points:
                for v in vectors_sorted:
                    vec, offset, v_color, v_alpha = self._extract_vec_data(v, color)
                    if v_alpha < 1.0:
                        glDepthMask(GL_FALSE)
                    self._draw_vectors_as_points_2d([v], color, z=0.2)
                    if v_alpha < 1.0:
                        glDepthMask(GL_TRUE)
            else:
                self._draw_vectors_as_arrows_2d(vectors_sorted, color, z=0.3)
            return

        # ===== PÔVODNÉ VEKTORY =====
        if self.vectors_as_points:
            self._draw_vectors_as_points_2d(self.vector_manager.animated_vectors, color)
        else:
            VectorRenderer.draw_vectors_2d_animated(
                self.vector_manager.animated_vectors,
                self.clock.get_time() / 1000.0,
                ortho_scale=self.camera.ortho_scale,
                color=color
            )

    def _draw_span_persistent_arrows(self, persistent, arrow_size):
        """Vykreslí perzistentné span vektory — najprv čiary, potom šípky"""
        # Zoradiť podľa dĺžky
        sorted_p = sorted(persistent,
                          key=lambda v: v['vec'][0] ** 2 + (v['vec'][1] if len(v['vec']) > 1 else 0) ** 2)

        # 1. Čiary (nízke Z)
        for v in sorted_p:
            vec, offset, v_color, v_alpha = self._extract_vec_data(v, (0.6, 0.2, 0.6))
            x = vec[0]
            y = vec[1] if len(vec) > 1 else 0
            ox = offset[0]
            oy = offset[1] if len(offset) > 1 else 0

            vec_length = math.sqrt(x * x + y * y)
            if vec_length > 0.1:
                angle = math.atan2(y, x)
                line_end_x = ox + x - arrow_size * 0.35 * math.cos(angle)
                line_end_y = oy + y - arrow_size * 0.35 * math.sin(angle)

                glLineWidth(4)
                glEnable(GL_BLEND)
                glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
                glColor4f(v_color[0], v_color[1], v_color[2], v_alpha)
                glBegin(GL_LINES)
                glVertex3f(ox, oy, 0.01)
                glVertex3f(line_end_x, line_end_y, 0.01)
                glEnd()
                glDisable(GL_BLEND)

        # 2. Šípky (vyššie Z, plná alpha)
        for v in persistent:
            vec, offset, v_color, _ = self._extract_vec_data(v, (0.6, 0.2, 0.6))
            x = vec[0]
            y = vec[1] if len(vec) > 1 else 0
            ox = offset[0]
            oy = offset[1] if len(offset) > 1 else 0

            vec_length = math.sqrt(x * x + y * y)
            if vec_length > 0.1:
                angle = math.atan2(y, x)
                self.vector_renderer.draw_triangle_arrowhead_2d(
                    ox + x, oy + y, angle, arrow_size, v_color, 1.0, z=0.05
                )

    # =========================================================================
    # 2D LABELY
    # =========================================================================

    def draw_vector_labels_2d(self, vectors):
        """Vykreslí labely pri vektoroch v 2D"""
        if not vectors:
            return

        width, height = self._get_window_size()
        left, right, bottom, top = self._get_visible_area_2d()

        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, width, height, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        text_color = (1, 1, 1) if self.background_dark else (0, 0, 0)

        for v in vectors:
            if not v.get('show_label', False):
                continue
            label = v.get('label', '')
            if not label:
                continue

            vec = v['vec']
            offset = v.get('offset', [0, 0])

            mid_x = offset[0] + vec[0] * 0.5
            mid_y = offset[1] + (vec[1] * 0.5 if len(vec) > 1 else 0)

            screen_x, screen_y = self._world_to_screen(mid_x, mid_y, left, right, bottom, top, width, height)
            self.ui_renderer.draw_text_2d(label, (screen_x + 10, screen_y - 10),
                                          color=text_color, font_size=18)

        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)

    # =========================================================================
    # 3D VYKRESĽOVANIE
    # =========================================================================

    def draw_vectors_3d(self, color):
        """Vykreslí vektory v 3D"""
        cam_pos = self.camera.get_position()

        if self.vector_manager.animation_controller.current_operation:
            vectors_to_draw = self.vector_manager.animation_controller.get_vectors_to_draw()
            opaque = [v for v in vectors_to_draw if v.get('alpha', 1.0) >= 1.0]
            transparent = [v for v in vectors_to_draw if v.get('alpha', 1.0) < 1.0]

            if self.vectors_as_points:
                self._draw_3d_points_list(opaque, color, cam_pos)
                glDepthMask(GL_FALSE)
                self._draw_3d_points_list(transparent, color, cam_pos)
                glDepthMask(GL_TRUE)
            else:
                self._draw_3d_arrows_list(opaque, color, cam_pos)
                glDepthMask(GL_FALSE)
                self._draw_3d_arrows_list(transparent, color, cam_pos)
                glDepthMask(GL_TRUE)
        else:
            if self.vectors_as_points:
                self._draw_3d_points_list(self.vector_manager.animated_vectors, color, cam_pos)
            else:
                self._draw_3d_arrows_list(self.vector_manager.animated_vectors, color, cam_pos)

    def _draw_3d_points_list(self, vectors, default_color, cam_pos):
        """Vykreslí zoznam 3D bodov"""
        radius = self._get_point_radius_3d()
        for v in vectors:
            vec, offset, v_color, v_alpha = self._extract_vec_data(v, default_color)

            if isinstance(vec[0], (int, float)):
                vec3d = [
                    (vec[i] if i < len(vec) else 0) + (offset[i] if i < len(offset) else 0)
                    for i in range(3)
                ]
                self._draw_single_point_3d(vec3d, radius, v_color, v_alpha)
            else:
                row_offsets = v.get('row_offsets', None)
                for x, y, z, ox, oy, oz in self._iter_matrix_rows(vec, offset, row_offsets):
                    self._draw_single_point_3d([x + ox, y + oy, z + oz], radius, v_color, v_alpha)

    def _draw_3d_arrows_list(self, vectors, default_color, cam_pos):
        """Vykreslí zoznam 3D šípok"""
        for v in vectors:
            vec, offset, v_color, v_alpha = self._extract_vec_data(v, default_color)

            if isinstance(vec[0], (int, float)):
                offset_3d = [offset[i] if i < len(offset) else 0 for i in range(3)]
                self._draw_single_vector_3d(vec, offset_3d, v_color, v_alpha, cam_pos)
            else:
                row_offsets = v.get('row_offsets', None)
                for i, row in enumerate(vec):
                    if row_offsets and i < len(row_offsets):
                        ro = list(row_offsets[i]) + [0] * (3 - len(row_offsets[i]))
                    else:
                        ro = [offset[j] if j < len(offset) else 0 for j in range(3)]
                    self._draw_single_vector_3d(row, ro, v_color, v_alpha, cam_pos)

    # =========================================================================
    # SPOLOČNÉ POMOCNÉ METÓDY
    # =========================================================================

    def _get_window_size(self):
        """Vráti šírku a výšku okna"""
        import pygame
        return pygame.display.get_window_size()

    def _get_visible_area_2d(self):
        """Vráti (left, right, bottom, top) viditeľnej oblasti"""
        import pygame
        width, height = pygame.display.get_window_size()
        aspect = width / max(height, 1)
        ortho = self.camera.ortho_scale
        pan_x = self.camera.pan_offset_x
        pan_y = self.camera.pan_offset_y

        if aspect >= 1.0:
            left = -ortho * aspect + pan_x
            right = ortho * aspect + pan_x
            bottom = -ortho + pan_y
            top = ortho + pan_y
        else:
            left = -ortho + pan_x
            right = ortho + pan_x
            bottom = -ortho / aspect + pan_y
            top = ortho / aspect + pan_y

        return left, right, bottom, top

    @staticmethod
    def _world_to_screen(wx, wy, left, right, bottom, top, width, height):
        """Konverzia world -> screen súradníc"""
        sx = (wx - left) / (right - left) * width
        sy = height - (wy - bottom) / (top - bottom) * height
        return sx, sy
