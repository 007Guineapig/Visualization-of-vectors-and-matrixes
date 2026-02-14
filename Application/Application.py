"""Hlavná aplikácia — refaktorovaná verzia.

Veľké bloky kódu sú extrahované do mixin tried:
- BazaInputMixin  (Application/BazaInputScreen.py)  — run_baza_input, ~960 riadkov
- VectorDrawingMixin (Application/VectorDrawing.py)  — draw_vectors_2d/3d, ~720 riadkov
- MathDisplayMixin (Application/MathDisplay.py)      — render_math_display, ~350 riadkov
"""

from Configurations.Colors import Colors
from VectorStuff.VectorManager import VectorManager
from VectorStuff.VectorRenderer import VectorRenderer
from Handler.InputHandler import InputHandler
from Cache.TextureCache import TextureCache
from Rendering.UIRenderer import UIRenderer
from Rendering.EnhancedUIRenderer import EnhancedUIRenderer
from Configurations.Config import Config
from Camera.Camera import Camera
from Rendering.GridRenderer import GridRenderer
from Rendering.AxesRenderer import AxesRenderer
from Controllers.TransformationController import TransformationController

from Application.BazaInputScreen import BazaInputMixin
from VectorStuff.VectorDrawing import VectorDrawingMixin
from Application.MathDisplay import MathDisplayMixin

import math
import pygame
from pygame.locals import *
from OpenGL.GL import *


class Application(BazaInputMixin, VectorDrawingMixin, MathDisplayMixin):
    """Hlavná aplikácia — orchestrácia"""

    def __init__(self):
        pygame.init()

        pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLEBUFFERS, 1)
        pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLESAMPLES, 4)
        self._last_render_time = 0
        self._min_frame_time = 1.0 / 60.0
        self.RESULT_COLOR = (0.5, 0, 0.5)
        self.screen = pygame.display.set_mode((900, 700), DOUBLEBUF | OPENGL | RESIZABLE)

        self.width, self.height = pygame.display.get_window_size()
        pygame.display.set_caption("XYZ axes — OOP Version")

        self.setup_opengl()

        # Components
        self.camera = Camera(self.width, self.height)
        self.vector_renderer = VectorRenderer()
        self.ui_renderer = UIRenderer()
        self.grid_renderer = GridRenderer()
        self.axes_renderer = AxesRenderer()

        self.vector_manager = VectorManager()
        self.input_handler = InputHandler()
        self.enhanced_ui = EnhancedUIRenderer()
        self.transformation_controller = TransformationController()

        # State
        self.clock = pygame.time.Clock()
        self.running = True
        self.startup_screen = True
        self.is_not_baza = True
        self.view_2d_mode = True
        self.background_dark = True
        self._update_background_color()
        self.grid_mode = 0
        self.plane_grid_mode = 1
        self.show_axes = True
        self.vectors_as_points = True
        self.show_helper_lines = False

        # Vectors
        self.saved_baza = []
        self.pending_vector = None

        # UI State
        self.matrix_inputs = [["" for _ in range(3)] for _ in range(3)]
        self.matrix_active_cell = (-1, -1)
        self.show_matrix_input = True

        pygame.font.init()

    def setup_opengl(self):
        """Inicializuje OpenGL"""
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_MULTISAMPLE)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_LINE_SMOOTH)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
        glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST)
        glClearColor(0.0, 0.0, 0.0, 1.0)

    def _update_background_color(self):
        if self.background_dark:
            glClearColor(*Colors.DARK_BG, 1.0)
        else:
            glClearColor(*Colors.LIGHT_BG, 1.0)

    def _move_camera_to_operation_plane(self):
        """Presunie kameru na rovinu aktuálnej operácie (len 3D)"""
        if self.view_2d_mode:
            return

        ctrl = self.vector_manager.animation_controller
        plane = ctrl.current_plane
        if not plane:
            return

        max_val = self.get_max_from_vectors()
        distance = max(25.0, max_val * 3.0 + 5)

        if plane == "3D":
            optimal_view = (ctrl.compute_optimal_view_for_current_step()
                            or ctrl.compute_optimal_view_for_3d())
            if optimal_view:
                self.camera.move_to_plane(plane, distance, custom_view=optimal_view)
            else:
                self.camera.move_to_plane(plane, distance)
        else:
            self.camera.move_to_plane(plane, distance)

    # =========================================================================
    # STARTUP SCREEN
    # =========================================================================

    def run_startup_screen(self):
        """Moderný startup screen"""
        while self.startup_screen and self.running:
            dt = self.clock.tick(Config.FPS) / 1000.0
            mx, my = pygame.mouse.get_pos()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    self.startup_screen = False
                    return

                elif event.type == pygame.VIDEORESIZE:
                    self.width, self.height = event.w, event.h
                    self.screen = pygame.display.set_mode((self.width, self.height),
                                                          DOUBLEBUF | OPENGL | RESIZABLE)
                    glViewport(0, 0, self.width, self.height)

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    cx, cy = self.width // 2, self.height // 2
                    btn_w, btn_h = 180, 52
                    gap = 24

                    btn_2d = pygame.Rect(cx - btn_w - gap // 2, cy + 30, btn_w, btn_h)
                    btn_3d = pygame.Rect(cx + gap // 2, cy + 30, btn_w, btn_h)
                    toggle = pygame.Rect(30, self.height - 60, 130, 38)

                    if toggle.collidepoint(mx, my):
                        self.background_dark = not self.background_dark
                    elif btn_2d.collidepoint(mx, my):
                        self.view_2d_mode = True
                        self.startup_screen = False
                    elif btn_3d.collidepoint(mx, my):
                        self.view_2d_mode = False
                        self.startup_screen = False

            # Render
            if self.background_dark:
                glClearColor(*Colors.DARK_BG, 1.0)
            else:
                glClearColor(*Colors.LIGHT_BG, 1.0)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            glMatrixMode(GL_PROJECTION)
            glLoadIdentity()
            glOrtho(0, self.width, self.height, 0, -1, 1)
            glMatrixMode(GL_MODELVIEW)
            glLoadIdentity()

            title_col = Colors.TEXT_DARK if self.background_dark else Colors.TEXT_LIGHT
            sub_col = Colors.TEXT_DARK_SEC if self.background_dark else Colors.TEXT_LIGHT_SEC

            cx, cy = self.width // 2, self.height // 2

            self.ui_renderer.draw_text_2d("Linear Algebra Visualizer",
                                          (cx - 160, cy - 100), title_col, 28)
            self.ui_renderer.draw_text_2d("Interaktívna vizualizácia vektorových operácií",
                                          (cx - 170, cy - 50), sub_col, 16)
            self.ui_renderer.draw_text_2d("Vyberte režim zobrazenia:",
                                          (cx - 95, cy - 10), sub_col, 15)

            btn_w, btn_h = 180, 52
            gap = 24
            btn_2d = pygame.Rect(cx - btn_w - gap // 2, cy + 30, btn_w, btn_h)
            btn_3d = pygame.Rect(cx + gap // 2, cy + 30, btn_w, btn_h)
            toggle = pygame.Rect(30, self.height - 60, 130, 38)

            hover_2d = btn_2d.collidepoint(mx, my)
            hover_3d = btn_3d.collidepoint(mx, my)
            hover_toggle = toggle.collidepoint(mx, my)

            self.ui_renderer.draw_button_2d(btn_2d.x, btn_2d.y, btn_2d.w, btn_2d.h,
                                            "2D Režim", is_dark=self.background_dark,
                                            hover=hover_2d, primary=True)
            self.ui_renderer.draw_button_2d(btn_3d.x, btn_3d.y, btn_3d.w, btn_3d.h,
                                            "3D Režim", is_dark=self.background_dark,
                                            hover=hover_3d, primary=True)

            theme_label = "Tmavý režim" if self.background_dark else "Svetlý režim"
            self.ui_renderer.draw_button_2d(toggle.x, toggle.y, toggle.w, toggle.h,
                                            theme_label, is_dark=self.background_dark,
                                            hover=hover_toggle)

            self.ui_renderer.draw_text_2d("v2.0", (self.width - 50, self.height - 30), sub_col, 12)
            pygame.display.flip()

    # =========================================================================
    # NAVIGÁCIA
    # =========================================================================

    def go_back_to_menu(self):
        """Vráti sa späť do hlavného menu"""
        self.vector_manager.animation_controller.clear()
        self.vector_manager.animated_vectors.clear()
        self.saved_baza.clear()
        self.camera.reset()
        self.vector_manager.span_controller.clear()
        self.transformation_controller.clear()

        self.startup_screen = True
        self.is_not_baza = True

        rows = 2 if self.view_2d_mode else 3
        cols = 2 if self.view_2d_mode else 3
        self.matrix_inputs = [["" for _ in range(cols)] for _ in range(rows)]
        self.matrix_active_cell = (-1, -1)

    # =========================================================================
    # EVENT HANDLING
    # =========================================================================

    def handle_events(self):
        mx, my = pygame.mouse.get_pos()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.VIDEORESIZE:
                self.width, self.height = event.w, event.h
                self.screen = pygame.display.set_mode((self.width, self.height),
                                                      DOUBLEBUF | OPENGL | RESIZABLE)
                glViewport(0, 0, self.width, self.height)
                self.camera.width = self.width
                self.camera.height = self.height
            elif event.type == pygame.KEYDOWN:
                self.handle_keypress(event)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                self.handle_mouse_down(event, mx, my)
            elif event.type == pygame.MOUSEBUTTONUP:
                self.handle_mouse_up(event)
            elif event.type == pygame.MOUSEMOTION:
                self.handle_mouse_motion(event)

    def handle_keypress(self, event):
        """Spracuje stlačenie klávesy"""
        # Span krokovanie
        if self.vector_manager.span_controller.active:
            if event.key == pygame.K_o:
                self.vector_manager.span_controller.show_all_combinations()
                if self.vector_manager.span_controller.locked_ortho_scale:
                    self.camera.ortho_scale = self.vector_manager.span_controller.locked_ortho_scale
                    self.camera.pan_offset_x = 0
                    self.camera.pan_offset_y = 0
                return
            if event.key == pygame.K_f:
                span_ctrl = self.vector_manager.span_controller
                span_ctrl.color_scheme = (span_ctrl.color_scheme + 1) % 2
                if span_ctrl.show_all_mode:
                    span_ctrl.show_all_combinations()
                return
            if event.key == pygame.K_p:
                self.vector_manager.span_controller.auto_play = not self.vector_manager.span_controller.auto_play
                return
            if event.key == pygame.K_SPACE:
                self.vector_manager.span_controller.next_step()
                return
            elif event.key == pygame.K_BACKSPACE:
                if not self._is_text_input_active():
                    self.vector_manager.span_controller.prev_step()
                    return
            elif event.key == pygame.K_c:
                self.vector_manager.span_controller.clear()
                return

        # Transformácia krokovanie
        if self.transformation_controller.active:
            if event.key == pygame.K_SPACE:
                self.transformation_controller.next_step()
                return
            elif event.key == pygame.K_BACKSPACE:
                if not self._is_text_input_active():
                    self.transformation_controller.prev_step()
                    return
            elif event.key == pygame.K_c:
                self.transformation_controller.clear()
                return

        # Animácia krokovanie
        if event.key == pygame.K_SPACE:
            if self.vector_manager.animation_controller.current_operation:
                self.vector_manager.animation_controller.next_step()
                self._move_camera_to_operation_plane()
            return

        if event.key == pygame.K_BACKSPACE:
            if not self._is_text_input_active():
                if self.vector_manager.animation_controller.current_operation:
                    self.vector_manager.animation_controller.prev_step()
                    self._move_camera_to_operation_plane()
                    return

        if event.key == pygame.K_c:
            self.vector_manager.animation_controller.clear()
            return
        if event.key == pygame.K_v:
            if self.vector_manager.animation_controller.current_operation:
                self._move_camera_to_operation_plane()
            return
        if event.key == pygame.K_b:
            self.plane_grid_mode = (self.plane_grid_mode + 1) % 6
            return
        if event.key == pygame.K_ESCAPE:
            self.go_back_to_menu()
            self.run_startup_screen()
            if self.running:
                self.run_baza_input()
            return
        elif event.key == pygame.K_g:
            self.grid_mode = (self.grid_mode + 1) % 4
        elif event.key == pygame.K_a:
            self.show_axes = not self.show_axes
        elif event.key == pygame.K_r:
            self.camera.reset()
            self.vector_manager.reset_animations()
        elif event.key == pygame.K_m:
            self.show_helper_lines = not self.show_helper_lines
        elif event.key == pygame.K_i:
            self.input_handler.show_input_active = True

    def _is_text_input_active(self):
        """Skontroluje či je aktívny nejaký textový input"""
        return any([
            self.input_handler.show_input_active,
            self.input_handler.show_multiplication_active,
            self.input_handler.show_matrix_size_active,
            self.input_handler.show_random_range_active,
            self.matrix_active_cell != (-1, -1)
        ])

    def handle_mouse_down(self, event, mx, my):
        back_button_rect = pygame.Rect(20, 10, 120, 40)
        if back_button_rect.collidepoint(mx, my):
            self.go_back_to_menu()
            self.run_startup_screen()
            if self.running:
                self.run_baza_input()
            return

        toggle_bg_rect = pygame.Rect(20, self.height - 50, 120, 35)
        if toggle_bg_rect.collidepoint(mx, my):
            self.background_dark = not self.background_dark
            if self.background_dark:
                glClearColor(0.0, 0.0, 0.0, 1.0)
            else:
                glClearColor(1.0, 1.0, 1.0, 1.0)
            return

        draw_button_rect = pygame.Rect(self.width - 160, 120, 140, 35)
        if draw_button_rect.collidepoint(mx, my):
            self.vectors_as_points = not self.vectors_as_points
            if not self.vectors_as_points:
                self.vector_manager.reset_animations()
            return

        if event.button == 1:
            if not self.vector_manager.span_controller.show_all_mode:
                self.camera.rotating = not self.view_2d_mode
                self.camera.panning = self.view_2d_mode
            self.camera.last_mouse = event.pos
        elif event.button == 3:
            if not self.vector_manager.span_controller.show_all_mode:
                self.camera.panning = True
            self.camera.last_mouse = event.pos
        elif event.button in (4, 5):
            self._handle_scroll(event.button)

    def _handle_scroll(self, button):
        """Spracuje scroll kolieskom"""
        span_ctrl = self.vector_manager.span_controller
        zoom_in = (button == 4)
        factor = 1.1

        if span_ctrl.show_all_mode and span_ctrl.locked_ortho_scale:
            new_scale = self.camera.ortho_scale / factor if zoom_in else self.camera.ortho_scale * factor
            min_scale = span_ctrl.locked_ortho_scale * 0.5
            max_scale = span_ctrl.locked_ortho_scale * 1.5
            if min_scale <= new_scale <= max_scale:
                self.camera.ortho_scale = new_scale
        else:
            if zoom_in:
                self.camera.zoom_in(self.view_2d_mode)
            else:
                self.camera.zoom_out(self.view_2d_mode)

    def handle_mouse_up(self, event):
        if event.button in (1, 3):
            self.camera.rotating = False
            self.camera.panning = False

    def handle_mouse_motion(self, event):
        dx = event.pos[0] - self.camera.last_mouse[0]
        dy = event.pos[1] - self.camera.last_mouse[1]
        self.camera.last_mouse = event.pos

        if self.vector_manager.span_controller.show_all_mode:
            return

        if self.view_2d_mode and self.camera.panning:
            self.camera.handle_panning_2d(dx, dy)
        elif not self.view_2d_mode:
            if self.camera.rotating:
                self.camera.handle_rotation(dx, dy)
            if self.camera.panning:
                self.camera.handle_panning_3d(dx, dy)

    # =========================================================================
    # UPDATE & RENDER
    # =========================================================================

    def update(self, dt):
        self.vector_manager.update_animations(dt)
        if self.transformation_controller.active:
            self.transformation_controller.update(dt)
        if not self.view_2d_mode and self.camera.animating_to_plane:
            self.camera.animate_to_plane(
                self.vector_manager.animation_controller.current_plane, dt
            )

    def render(self):
        current_time = pygame.time.get_ticks() / 1000.0
        if current_time - self._last_render_time < self._min_frame_time:
            return
        self._last_render_time = current_time

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        if self.view_2d_mode:
            self.render_2d()
        else:
            self.render_3d()

        self.render_ui()
        pygame.display.flip()

    def render_2d(self):
        self.camera.setup_2d_projection()

        if self.grid_mode % 2 == 0:
            self.grid_renderer.draw_grid_2d(self.camera.ortho_scale,
                                            self.camera.pan_offset_x,
                                            self.camera.pan_offset_y,
                                            self.width, self.height)

        if self.transformation_controller.active:
            self.draw_transformation_grid_2d()
            self.draw_transformation_basis_vectors_2d()

        if self.show_axes:
            tc = self.transformation_controller
            if tc.active and tc.t > 0.01:
                dim_alpha = max(0.15, 1.0 - tc.t * 0.8)
                # self._draw_axes_2d_dimmed(dim_alpha)
            else:
                self.axes_renderer.draw_axes_2d(self.camera.ortho_scale,
                                                self.camera.pan_offset_x,
                                                self.camera.pan_offset_y,
                                                self.width, self.height,
                                                self.ui_renderer, self.background_dark)

        if self.vector_manager.span_controller.active:
            self.draw_span_circle_points()

        color = (0, 0, 0) if not self.background_dark else (1, 1, 1)
        self.draw_vectors_2d(color)

        if self.show_helper_lines:
            self.draw_helper_lines_2d()

    # =========================================================================
    # TRANSFORMÁCIA BÁZY — RENDERING
    # =========================================================================

    def draw_transformation_grid_2d(self):
        """Vykreslí transformovanú mriežku"""
        tc = self.transformation_controller
        if not tc.active or tc.t < 0.01:
            return

        M = tc.get_interpolated_matrix()
        m00, m10 = float(M[0, 0]), float(M[1, 0])
        m01, m11 = float(M[0, 1]), float(M[1, 1])

        left, right, bottom, top = self._get_visible_area_2d()

        # Inverzná matica
        det = m00 * m11 - m01 * m10
        if abs(det) < 1e-10:
            return

        inv_det = 1.0 / det
        i00, i01 = m11 * inv_det, -m01 * inv_det
        i10, i11 = -m10 * inv_det, m00 * inv_det

        corners = [(left, bottom), (right, bottom), (right, top), (left, top)]
        orig_xs = [i00 * sx + i01 * sy for sx, sy in corners]
        orig_ys = [i10 * sx + i11 * sy for sx, sy in corners]

        orig_left, orig_right = min(orig_xs), max(orig_xs)
        orig_bottom, orig_top = min(orig_ys), max(orig_ys)
        orig_visible = max(orig_right - orig_left, orig_top - orig_bottom)

        # Dynamický step
        step = tc.grid_step
        display_step = step
        while orig_visible / display_step > 100:
            display_step *= 2

        start_u = math.floor(orig_left / display_step) * display_step
        end_u = math.ceil(orig_right / display_step) * display_step
        start_v = math.floor(orig_bottom / display_step) * display_step
        end_v = math.ceil(orig_top / display_step) * display_step

        # Mriežka
        alpha = 0.4 + 0.4 * tc.t
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glLineWidth(1.5)
        glColor4f(0.4 * tc.t, 0.6 * tc.t, 0.2 * tc.t, alpha)

        glBegin(GL_LINES)
        u = start_u
        while u <= end_u:
            glVertex3f(m00 * u + m01 * start_v, m10 * u + m11 * start_v, 0.05)
            glVertex3f(m00 * u + m01 * end_v, m10 * u + m11 * end_v, 0.05)
            u += display_step

        v = start_v
        while v <= end_v:
            glVertex3f(m00 * start_u + m01 * v, m10 * start_u + m11 * v, 0.05)
            glVertex3f(m00 * end_u + m01 * v, m10 * end_u + m11 * v, 0.05)
            v += display_step
        glEnd()
        glDisable(GL_BLEND)

        # Transformované osi
        axis_extent_u = max(abs(start_u), abs(end_u))
        axis_extent_v = max(abs(start_v), abs(end_v))
        axis_color = (1, 1, 1) if self.background_dark else (0, 0, 0)

        glEnable(GL_BLEND)
        glLineWidth(3.0)

        glColor3f(*axis_color)
        glBegin(GL_LINES)
        glVertex3f(m00 * (-axis_extent_u), m10 * (-axis_extent_u), 0.06)
        glVertex3f(m00 * axis_extent_u, m10 * axis_extent_u, 0.06)
        glEnd()

        glColor3f(*axis_color)
        glBegin(GL_LINES)
        glVertex3f(m01 * (-axis_extent_v), m11 * (-axis_extent_v), 0.06)
        glVertex3f(m01 * axis_extent_v, m11 * axis_extent_v, 0.06)
        glEnd()

        glDisable(GL_BLEND)

    def draw_transformation_basis_vectors_2d(self):
        """Vykreslí animované bázové vektory pri transformácii"""
        tc = self.transformation_controller
        if not tc.active:
            return

        i_hat, j_hat = tc.get_basis_vectors_2d()
        arrow_size = max(0.15, min(0.12 * self.camera.ortho_scale, 2.0))

        # î vektor
        ix, iy = i_hat[0], i_hat[1]
        self._draw_basis_vector_2d(ix, iy, arrow_size, tc.color_i)

        # ĵ vektor
        jx, jy = j_hat[0], j_hat[1]
        self._draw_basis_vector_2d(jx, jy, arrow_size, tc.color_j)

        # Labely
        width, height = self._get_window_size()
        left, right, bottom, top = self._get_visible_area_2d()

        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, width, height, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        sx, sy = self._world_to_screen(ix, iy, left, right, bottom, top, width, height)
        self.ui_renderer.draw_text_2d(f"i ({ix:.1f}, {iy:.1f})", (sx + 10, sy - 20),
                                       color=tc.color_i, font_size=16)

        sx, sy = self._world_to_screen(jx, jy, left, right, bottom, top, width, height)
        self.ui_renderer.draw_text_2d(f"j ({jx:.1f}, {jy:.1f})", (sx + 10, sy - 20),
                                       color=tc.color_j, font_size=16)

        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)

    def _draw_basis_vector_2d(self, vx, vy, arrow_size, color):
        """Vykreslí jeden bázový vektor"""
        vec_length = math.sqrt(vx * vx + vy * vy)
        if vec_length <= 0.05:
            return

        angle = math.atan2(vy, vx)
        line_end_x = vx - arrow_size * 0.35 * math.cos(angle)
        line_end_y = vy - arrow_size * 0.35 * math.sin(angle)

        glLineWidth(5)
        glColor3f(*color)
        glBegin(GL_LINES)
        glVertex3f(0, 0, 0.4)
        glVertex3f(line_end_x, line_end_y, 0.4)
        glEnd()

        self.vector_renderer.draw_triangle_arrowhead_2d(
            vx, vy, angle, arrow_size, color, 1.0, z=0.4
        )

    # =========================================================================
    # HELPER LINES (komponenty)
    # =========================================================================

    def draw_helper_lines_2d(self):
        """Vykreslí pomocné čiary (rozklad na komponenty)"""
        vectors_to_draw = []

        if self.vector_manager.span_controller.active:
            all_vecs = self.vector_manager.span_controller.get_current_vectors()
            vectors_to_draw = [v for v in all_vecs if not v.get('is_persistent', False)]
        elif self.vector_manager.animation_controller.current_operation:
            vectors_to_draw = self.vector_manager.animation_controller.get_vectors_to_draw()
        else:
            vectors_to_draw = self.vector_manager.animated_vectors

        if not vectors_to_draw:
            return

        width, height = self._get_window_size()
        left, right, bottom, top = self._get_visible_area_2d()

        for v in vectors_to_draw:
            vec = v.get('vec', None)
            if vec is None:
                continue

            offset = v.get('offset', [0, 0])
            v_alpha = v.get('alpha', 1.0)
            if v_alpha < 0.3:
                continue

            if isinstance(vec[0], (int, float)):
                self._draw_single_helper_lines(vec, offset, v_alpha, left, right, bottom, top, width, height)
            else:
                row_offsets = v.get('row_offsets', None)
                for i, row in enumerate(vec):
                    if row_offsets and i < len(row_offsets):
                        ro = row_offsets[i]
                    else:
                        ro = offset
                    self._draw_single_helper_lines(row, ro, v_alpha, left, right, bottom, top, width, height,
                                                    show_labels=False)

    def _draw_single_helper_lines(self, vec, offset, v_alpha, left, right, bottom, top,
                                   width, height, show_labels=True):
        """Vykreslí helper lines pre jeden vektor"""
        ox = offset[0]
        oy = offset[1] if len(offset) > 1 else 0
        vx = vec[0]
        vy = vec[1] if len(vec) > 1 else 0

        if abs(vx) < 0.01 and abs(vy) < 0.01:
            return

        end_x = ox + vx
        end_y = oy + vy
        line_alpha = min(v_alpha, 0.7)

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glLineWidth(8.0)

        dash_length = 0.15 * self.camera.ortho_scale / 6.0

        # X-komponent — červená
        if abs(vx) > 0.01:
            glColor4f(0.85, 0.2, 0.2, line_alpha)
            self._draw_dashed_line_2d(ox, oy, end_x, oy, dash_length, z=0.15)

        # Y-komponent — zelená
        if abs(vy) > 0.01:
            glColor4f(0.2, 0.85, 0.2, line_alpha)
            self._draw_dashed_line_2d(end_x, oy, end_x, end_y, dash_length, z=0.15)

        glDisable(GL_BLEND)

        if not show_labels:
            return

        # Labely
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, width, height, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        if abs(vx) > 0.01:
            mid_sx, mid_sy = self._world_to_screen(ox + vx / 2, oy, left, right, bottom, top, width, height)
            label_x = f"{vx:.1f}" if isinstance(vx, float) else str(int(vx))
            y_off = 15 if vy >= 0 else -20
            self.ui_renderer.draw_text_2d(label_x, (mid_sx - 10, mid_sy + y_off),
                                          color=(0.85, 0.2, 0.2), font_size=18)

        if abs(vy) > 0.01:
            mid_sx, mid_sy = self._world_to_screen(end_x, oy + vy / 2, left, right, bottom, top, width, height)
            label_y = f"{vy:.1f}" if isinstance(vy, float) else str(int(vy))
            x_off = 10 if vx >= 0 else -35
            self.ui_renderer.draw_text_2d(label_y, (mid_sx + x_off, mid_sy - 8),
                                          color=(0.2, 0.85, 0.2), font_size=18)

        # Vektor label
        vec_sx, vec_sy = self._world_to_screen(end_x, end_y, left, right, bottom, top, width, height)
        vx_str = f"{vx:.1f}" if isinstance(vx, float) else str(int(vx))
        vy_str = f"{vy:.1f}" if isinstance(vy, float) else str(int(vy))
        text_color = (1, 1, 1) if self.background_dark else (0, 0, 0)
        self.ui_renderer.draw_text_2d(f"[{vx_str}]", (vec_sx + 15, vec_sy - 25),
                                       color=text_color, font_size=16)
        self.ui_renderer.draw_text_2d(f"[{vy_str}]", (vec_sx + 15, vec_sy - 5),
                                       color=text_color, font_size=16)

        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)

    def _draw_dashed_line_2d(self, x1, y1, x2, y2, dash_length=0.2, z=0.15):
        dx = x2 - x1
        dy = y2 - y1
        length = math.sqrt(dx * dx + dy * dy)
        if length < 0.001:
            return

        ndx, ndy = dx / length, dy / length
        num_dashes = max(1, int(length / (dash_length * 2)))
        actual_dash = length / (num_dashes * 2)

        glBegin(GL_LINES)
        for i in range(num_dashes):
            start_t = i * 2 * actual_dash
            end_t = start_t + actual_dash
            glVertex3f(x1 + ndx * start_t, y1 + ndy * start_t, z)
            glVertex3f(x1 + ndx * end_t, y1 + ndy * end_t, z)
        glEnd()

    # =========================================================================
    # SPAN VIZUALIZÁCIA
    # =========================================================================

    def draw_span_circle_points(self):
        """Vykreslí body kruhu pre span vizualizáciu"""
        span_ctrl = self.vector_manager.span_controller
        if not span_ctrl.circle_points or not span_ctrl.basis_vectors:
            return

        v1, v2 = span_ctrl.basis_vectors
        radius = 0.08 * self.camera.ortho_scale / 6.5

        inactive_color = (0.5, 0.5, 0.5)
        active_color = (1.0, 0.8, 0.0)
        past_color = (0.3, 0.8, 0.3)

        for i, (c1, c2) in enumerate(span_ctrl.circle_points):
            point_x = c1 * v1[0] + c2 * v2[0]
            point_y = c1 * v1[1] + c2 * v2[1]

            if i == span_ctrl.current_circle_index and span_ctrl.current_step >= 2:
                color = active_color
                point_radius = radius * 1.5
            elif span_ctrl.current_step >= 2 and i < span_ctrl.current_circle_index:
                color = past_color
                point_radius = radius
            else:
                color = inactive_color
                point_radius = radius * 0.8

    # =========================================================================
    # 3D RENDERING
    # =========================================================================

    def render_3d(self):
        self.camera.setup_3d_projection()

        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LESS)
        glDepthMask(GL_TRUE)

        length_xyz = max(10, self.get_max_from_vectors())
        if length_xyz > 10:
            length_xyz += 3

        ctrl = self.vector_manager.animation_controller
        has_operation = ctrl.current_operation is not None
        plane_active = has_operation and self.plane_grid_mode > 0

        # 1. Výplň roviny
        if plane_active:
            plane = ctrl.current_plane
            normal = ctrl.operation_plane_normal

            if plane and normal and self.plane_grid_mode in (4, 5, 6, 7):
                fill_color = (0.6, 0.6, 0) if self.background_dark else (0.8, 0.8, 0.4)
                is_transparent = self.plane_grid_mode in (4, 6)

                if is_transparent:
                    glEnable(GL_BLEND)
                    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
                    alpha = 0.25
                else:
                    glDisable(GL_BLEND)
                    alpha = 1.0

                glEnable(GL_POLYGON_OFFSET_FILL)
                glPolygonOffset(1.0, 1.0)
                self.grid_renderer.draw_filled_plane(
                    normal=normal, center=[0, 0, 0], size=length_xyz,
                    color=fill_color, alpha=alpha, transparent=is_transparent
                )
                glDisable(GL_POLYGON_OFFSET_FILL)

        # 2. Globálne mriežky
        if self.grid_mode == 1:
            self.grid_renderer.draw_planes_3d(length_xyz)
        elif self.grid_mode == 2:
            self.grid_renderer.draw_planes_3d(length_xyz, colored=True)
        elif self.grid_mode == 3:
            self.grid_renderer.draw_grid_3d(length_xyz)

        # 3. Mriežka v rovine operácie
        if plane_active:
            plane = ctrl.current_plane
            normal = ctrl.operation_plane_normal

            if plane and normal and self.plane_grid_mode in (1, 2, 3, 4, 5):
                grid_configs = {
                    1: ((0.6, 0.6, 0) if self.background_dark else (0.4, 0.4, 0), 1.0),
                    2: ((0, 0.6, 0.6) if self.background_dark else (0, 0.4, 0.4), 0.5),
                    3: ((0.6, 0, 0.6) if self.background_dark else (0.4, 0, 0.4), 2.0),
                }
                grid_color, step = grid_configs.get(self.plane_grid_mode,
                    ((0.2, 0.2, 0.2) if self.background_dark else (0.5, 0.5, 0.5), 1.0))

                self.grid_renderer.draw_grid_in_plane(
                    normal=normal, center=[0, 0, 0], size=length_xyz,
                    step=step, color=grid_color
                )

        # 4. Osi
        if self.show_axes:
            cam_pos = self.camera.get_position()
            self.axes_renderer.draw_axes_3d(length_xyz, cam_pos, self.vector_renderer)

        # 5. Vektory
        if self.grid_mode in (1, 2):
            v_color = (0, 0, 0)
        else:
            v_color = (0, 0, 0) if not self.background_dark else (1, 1, 1)

        self.draw_vectors_3d(v_color)

    # =========================================================================
    # UI OVERLAY
    # =========================================================================

    def render_ui(self):
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, self.width, self.height, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        glDisable(GL_DEPTH_TEST)

        text_color = (1, 1, 1) if self.background_dark else (0, 0, 0)

        # Tlačidlá
        toggle_bg_rect = pygame.Rect(20, self.height - 50, 120, 35)
        label = "Dark Mode" if self.background_dark else "Light Mode"
        self.ui_renderer.draw_button_2d(toggle_bg_rect.x, toggle_bg_rect.y,
                                        toggle_bg_rect.width, toggle_bg_rect.height,
                                        label, active=True)

        back_button_rect = pygame.Rect(20, 10, 120, 40)
        self.ui_renderer.draw_button_2d(back_button_rect.x, back_button_rect.y,
                                        back_button_rect.width, back_button_rect.height,
                                        "← Menu", active=True)

        self.ui_renderer.draw_text_2d("ESC = späť do menu", (150, 20),
                                      color=text_color, font_size=14)

        if self.view_2d_mode:
            helper_status = "M: Pomocné čiary [ZAP]" if self.show_helper_lines else "M: Pomocné čiary [VYP]"
            helper_color = (0.3, 0.9, 0.3) if self.show_helper_lines else text_color
            self.ui_renderer.draw_text_2d(helper_status, (150, 40), color=helper_color, font_size=14)

        draw_button_rect = pygame.Rect(self.width - 160, 120, 140, 35)
        button_label = "As Points" if self.vectors_as_points else "As Vectors"
        self.ui_renderer.draw_button_2d(draw_button_rect.x, draw_button_rect.y,
                                        draw_button_rect.width, draw_button_rect.height,
                                        button_label, active=True)

        if self.saved_baza and not self.view_2d_mode:
            self.enhanced_ui.draw_basis_matrix_top_right(
                self.saved_baza[0], self.width, self.height, font_size=20,
                color=(0, 0, 0) if not self.background_dark else (1, 1, 1)
            )

        # Operácia indikátor
        if self.vector_manager.animation_controller.current_operation:
            self._render_operation_ui(text_color)

        # Transformácia indikátor
        if self.transformation_controller.active:
            self._render_transformation_ui(text_color)

        # Span indikátor
        if self.vector_manager.span_controller.active:
            self._render_span_ui(text_color)

        glEnable(GL_DEPTH_TEST)
        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)

    def _render_operation_ui(self, text_color):
        """Vykreslí UI pre operáciu"""
        ctrl = self.vector_manager.animation_controller
        math_info = ctrl.get_math_display_info()
        if math_info:
            self.render_math_display(math_info)

        is_matrix = ctrl._is_matrix(ctrl.operands[0]) if ctrl.operands else False
        op_names = {
            'add': 'Sčítanie matíc' if is_matrix else 'Sčítanie vektorov',
            'subtract': 'Odčítanie matíc' if is_matrix else 'Odčítanie vektorov',
            'scalar_mult': 'Násobenie matice konštantou' if is_matrix else 'Násobenie vektora konštantou',
            'linear_combination': 'Lineárna kombinácia vektorov'
        }
        op_name = op_names.get(ctrl.operation_type, ctrl.operation_type)
        step_text = f"{op_name} - Krok {ctrl.current_step}/{ctrl.max_steps}"

        if is_matrix:
            descriptions = {
                'add': ["Zobrazenie matíc A a B", "Vektory matice B posunuté na konce A",
                        "Výsledná matica A + B", "Komutatívnosť: A + B = B + A"],
                'subtract': ["Zobrazenie matíc A a B", "Negácia matice B → -B",
                              "Vektory -B posunuté na konce A", "Výsledná matica A - B"],
                'scalar_mult': ["Pôvodná matica A",
                                f"Výsledná matica {ctrl.constant}·A" if ctrl.constant else "Výsledná matica"],
            }
        else:
            descriptions = {
                'add': ["Zobrazenie vektorov v1 a v2", "Vektor v2 posunutý na koniec v1",
                        "Výsledný vektor v1 + v2", "Komutatívnosť: v1 + v2 = v2 + v1"],
                'subtract': ["Zobrazenie vektorov v1 a v2", "Negácia vektora v2 → -v2",
                              "Vektor -v2 posunutý na koniec v1", "Výsledný vektor v1 - v2"],
                'scalar_mult': ["Pôvodný vektor v",
                                f"Výsledný vektor {ctrl.constant}·v" if ctrl.constant else "Výsledný vektor"],
                'linear_combination': ["Zobrazenie vektorov v1 a v2", "Vykreslenie c1·v1",
                                        "Vykreslenie c2·v2", "Pridanie c2·v2",
                                        "Výsledný vektor c1·v1 + c2·v2"],
            }

        desc_list = descriptions.get(ctrl.operation_type, [""])
        desc = desc_list[ctrl.current_step] if ctrl.current_step < len(desc_list) else ""

        cx = self.width // 2
        self.ui_renderer.draw_text_2d(step_text, (cx - 150, 20), color=text_color, font_size=24)
        self.ui_renderer.draw_text_2d(desc, (cx - 200, 50), color=text_color, font_size=18)
        self.ui_renderer.draw_text_2d("SPACE=ďalší | BACKSPACE=späť | C=zrušiť",
                                      (cx - 180, 75), color=text_color, font_size=16)

        # 3D info
        plane = ctrl.current_plane
        normal = ctrl.operation_plane_normal
        if plane and not self.view_2d_mode:
            self.ui_renderer.draw_text_2d(f"Rovina: {plane}", (cx - 180, 95),
                                          color=text_color, font_size=16)
            if normal:
                self.ui_renderer.draw_text_2d(
                    f"Normála: ({normal[0]:.2f}, {normal[1]:.2f}, {normal[2]:.2f})",
                    (cx - 180, 115), color=text_color, font_size=14)
            self.ui_renderer.draw_text_2d(
                f"Kamera: az={self.camera.azimuth:.1f}° el={self.camera.elevation:.1f}°",
                (cx - 180, 135), color=text_color, font_size=14)

            grid_modes = ["vypnutý", "grid", "grid jemný", "grid hrubý", "vyplnená + grid", "vyplnená"]
            self.ui_renderer.draw_text_2d(f"Plane grid (B): {grid_modes[self.plane_grid_mode]}",
                                          (cx - 180, 155), color=text_color, font_size=14)

            if not self.camera.animating_to_plane:
                self.ui_renderer.draw_text_2d("V=zobraz rovinu", (cx - 180, 175),
                                              color=text_color, font_size=14)

    def _render_transformation_ui(self, text_color):
        """Vykreslí UI pre transformáciu bázy"""
        tc = self.transformation_controller
        desc = tc.get_step_description()
        cx = self.width // 2

        self.ui_renderer.draw_text_2d(f"Transformácia bázy - Krok {tc.current_step}/{tc.max_steps}",
                                      (cx - 180, 20), color=text_color, font_size=24)
        self.ui_renderer.draw_text_2d(desc, (cx - 200, 50), color=text_color, font_size=18)
        self.ui_renderer.draw_text_2d("SPACE=ďalší | BACKSPACE=späť | C=zrušiť",
                                      (cx - 180, 75), color=text_color, font_size=16)

        mat = tc.get_matrix_display()
        if mat:
            self.ui_renderer.draw_text_2d("Matica:", (20, self.height - 120),
                                          color=text_color, font_size=16)
            self.ui_renderer.draw_text_2d(f"[{mat[0][0]:.1f}  {mat[0][1]:.1f}]",
                                          (20, self.height - 100), color=text_color, font_size=18)
            self.ui_renderer.draw_text_2d(f"[{mat[1][0]:.1f}  {mat[1][1]:.1f}]",
                                          (20, self.height - 78), color=text_color, font_size=18)
            self.ui_renderer.draw_text_2d(f"det = {tc.determinant:.2f}",
                                          (20, self.height - 55), color=text_color, font_size=14)

    def _render_span_ui(self, text_color):
        """Vykreslí UI pre span"""
        span_ctrl = self.vector_manager.span_controller
        cx = self.width // 2

        self.ui_renderer.draw_text_2d(f"Span - Krok {span_ctrl.current_step + 1}",
                                      (cx - 150, 20), color=text_color, font_size=24)

        comb = span_ctrl.combinations[span_ctrl.current_step]
        desc = f"c1·v1 + c2·v2 kde c1={comb['c1']:.2f}, c2={comb['c2']:.2f}"
        self.ui_renderer.draw_text_2d(desc, (cx - 200, 50), color=text_color, font_size=18)

        if span_ctrl.auto_play:
            control_text = "P=pauza | BACKSPACE=späť | C=zrušiť"
            play_status = "▶ AUTO-PLAY AKTÍVNY"
        else:
            control_text = "P=play | SPACE=ďalší | BACKSPACE=späť | C=zrušiť"
            play_status = "⏸ Manuálny režim"

        self.ui_renderer.draw_text_2d(control_text, (cx - 200, 75), color=text_color, font_size=16)
        self.ui_renderer.draw_text_2d(play_status, (cx - 120, 95), color=text_color, font_size=14)
        self.ui_renderer.draw_text_2d("(Nekonečné krokovanie - nové kombinácie sa generujú priebežne)",
                                      (cx - 240, 115), color=text_color, font_size=14)

    # =========================================================================
    # UTILITIES
    # =========================================================================

    def get_max_from_vectors(self):
        """Vráti najväčšiu hodnotu zo všetkých vektorov"""
        max_val = 10.0

        if self.vector_manager.animation_controller.current_operation:
            ctrl = self.vector_manager.animation_controller
            all_points = [[0, 0, 0]]

            for operand in ctrl.operands:
                if isinstance(operand[0], (int, float)):
                    all_points.append(list(operand) + [0] * (3 - len(operand)))
                else:
                    for row in operand:
                        all_points.append(list(row) + [0] * (3 - len(row)))

            if ctrl.result:
                if isinstance(ctrl.result[0], (int, float)):
                    all_points.append(list(ctrl.result) + [0] * (3 - len(ctrl.result)))
                else:
                    for row in ctrl.result:
                        all_points.append(list(row) + [0] * (3 - len(row)))

            if ctrl.operation_type in ['add', 'subtract'] and len(ctrl.operands) >= 2:
                offset = ctrl.operands[0]
                second = ctrl.operands[1]

                if isinstance(offset[0], (int, float)) and isinstance(second[0], (int, float)):
                    combined = [
                        (offset[i] if i < len(offset) else 0) + (second[i] if i < len(second) else 0)
                        for i in range(3)
                    ]
                    all_points.append(combined)
                elif not isinstance(offset[0], (int, float)) and not isinstance(second[0], (int, float)):
                    for i, row1 in enumerate(offset):
                        if i < len(second):
                            row2 = second[i]
                            combined = [
                                (row1[j] if j < len(row1) else 0) + (row2[j] if j < len(row2) else 0)
                                for j in range(3)
                            ]
                            all_points.append(combined)

            if all_points:
                max_val = max(max(abs(coord) for coord in point) for point in all_points)

        if self.vector_manager.animated_vectors:
            for v in self.vector_manager.animated_vectors:
                vec = v['vec']
                if isinstance(vec[0], (int, float)):
                    current_max = max(abs(x) for x in vec)
                else:
                    current_max = max(max(abs(x) for x in row) for row in vec)
                max_val = max(max_val, current_max)

        return max_val

    # =========================================================================
    # HLAVNÁ SLUČKA
    # =========================================================================

    def run(self):
        self.run_startup_screen()

        if not self.running:
            TextureCache.clear()
            pygame.quit()
            return

        self.run_baza_input()

        if not self.running:
            TextureCache.clear()
            pygame.quit()
            return

        while self.running:
            dt = self.clock.tick(60) / 1000.0
            self.handle_events()
            self.update(dt)
            self.render()

        TextureCache.clear()
        pygame.quit()
