import math

import numpy as np
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
class GridRenderer:
    """Optimalizovaný renderer pre mriežky a plochy"""

    # Cache pre opakované výpočty
    _plane_tangents_cache = {}

    @staticmethod
    def draw_filled_plane(normal, center=(0, 0, 0), size=10.0, color=(0.5, 0.5, 0.0), alpha=1.0, transparent=False):
        """Nakreslí vyplnenú rovinu definovanú normálou"""
        normal = np.array(normal, dtype=np.float32)
        norm_length = np.linalg.norm(normal)
        if norm_length < 1e-6:
            return
        normal = normal / norm_length

        # Nájdi dva kolmé vektory v rovine
        if abs(normal[2]) < 0.9:
            ref = np.array([0, 0, 1], dtype=np.float32)
        else:
            ref = np.array([1, 0, 0], dtype=np.float32)

        tangent1 = np.cross(normal, ref)
        tangent1 = tangent1 / np.linalg.norm(tangent1)
        tangent2 = np.cross(normal, tangent1)
        tangent2 = tangent2 / np.linalg.norm(tangent2)

        center = np.array(center, dtype=np.float32)

        # Vypočítaj 4 rohy roviny
        corner1 = center - tangent1 * size - tangent2 * size
        corner2 = center + tangent1 * size - tangent2 * size
        corner3 = center + tangent1 * size + tangent2 * size
        corner4 = center - tangent1 * size + tangent2 * size

        # Posuň rovinu mierne dozadu aby vektory boli vždy viditeľné
        glEnable(GL_POLYGON_OFFSET_FILL)
        glPolygonOffset(1.0, 1.0)

        if transparent:
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glColor4f(color[0], color[1], color[2], alpha)
        else:
            glColor3f(color[0], color[1], color[2])

        glBegin(GL_QUADS)
        glVertex3f(corner1[0], corner1[1], corner1[2])
        glVertex3f(corner2[0], corner2[1], corner2[2])
        glVertex3f(corner3[0], corner3[1], corner3[2])
        glVertex3f(corner4[0], corner4[1], corner4[2])
        glEnd()

        glDisable(GL_POLYGON_OFFSET_FILL)

        if transparent:
            glDisable(GL_BLEND)

    @staticmethod
    def draw_grid_2d(ortho_scale, pan_x, pan_y, width, height, step=1.0, z=-0.1, max_lines=100):
        """Nakreslí 2D mriežku - PLYNULÝ ZOOM (fade medzi úrovňami)"""
        width, height = pygame.display.get_window_size()
        aspect = width / height

        if aspect >= 1.0:
            left = -ortho_scale * aspect + pan_x
            right = ortho_scale * aspect + pan_x
            bottom = -ortho_scale + pan_y
            top = ortho_scale + pan_y
        else:
            left = -ortho_scale + pan_x
            right = ortho_scale + pan_x
            bottom = -ortho_scale / aspect + pan_y
            top = ortho_scale / aspect + pan_y

        visible_range = max(right - left, top - bottom)

        # Nájdi "hrubý" step (vždy viditeľný) a "jemný" step (fade out)
        coarse_step = step
        while visible_range / coarse_step > max_lines:
            coarse_step *= 2

        fine_step = coarse_step / 2.0

        # Pomer: koľko jemných čiar by bolo viditeľných
        # Keď sa blíži k max_lines, jemné čiary miznú
        fine_line_count = visible_range / fine_step
        # fade: 1.0 keď je málo čiar, 0.0 keď ich je max_lines
        fade = max(0.0, min(1.0, 2.0 - fine_line_count / max_lines))

        # Hrubé čiary — vždy plné
        glLineWidth(1.0)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        start_x = math.floor(left / coarse_step) * coarse_step
        end_x = math.ceil(right / coarse_step) * coarse_step
        start_y = math.floor(bottom / coarse_step) * coarse_step
        end_y = math.ceil(top / coarse_step) * coarse_step

        glColor4f(0.5, 0.5, 0.5, 1.0)
        glBegin(GL_LINES)
        x = start_x
        while x <= end_x:
            glVertex3f(x, bottom, z)
            glVertex3f(x, top, z)
            x += coarse_step
        y = start_y
        while y <= end_y:
            glVertex3f(left, y, z)
            glVertex3f(right, y, z)
            y += coarse_step
        glEnd()

        # Jemné čiary — fade out pri zoom out
        if fade > 0.01 and fine_step >= step:
            start_x_fine = math.floor(left / fine_step) * fine_step
            end_x_fine = math.ceil(right / fine_step) * fine_step
            start_y_fine = math.floor(bottom / fine_step) * fine_step
            end_y_fine = math.ceil(top / fine_step) * fine_step

            glColor4f(0.8, 0.8, 0.8, fade * 0.6)
            glBegin(GL_LINES)
            x = start_x_fine
            while x <= end_x_fine:
                # Preskočiť čiary na pozíciách hrubého stepu (tie sú už nakreslené)
                if abs(x / coarse_step - round(x / coarse_step)) > 0.01:
                    glVertex3f(x, bottom, z)
                    glVertex3f(x, top, z)
                x += fine_step
            y = start_y_fine
            while y <= end_y_fine:
                if abs(y / coarse_step - round(y / coarse_step)) > 0.01:
                    glVertex3f(left, y, z)
                    glVertex3f(right, y, z)
                y += fine_step
            glEnd()

        glDisable(GL_BLEND)

    @staticmethod
    def draw_grid_3d(size=10.0, step=1.0):
        step = 2.0
        """Nakreslí 3D mriežku - len čiary bez výplne"""
        # XY rovina (Z=0) - zelená
        GridRenderer.draw_grid_in_plane(
            normal=[0, 0, 1], center=[0, 0, 0], size=size, step=step,
            color=(0.2, 0.7, 0.2)
        )
        # XZ rovina (Y=0) - červená
        GridRenderer.draw_grid_in_plane(
            normal=[0, 1, 0], center=[0, 0, 0], size=size, step=step,
            color=(0.7, 0.2, 0.2)
        )
        # YZ rovina (X=0) - modrá
        GridRenderer.draw_grid_in_plane(
            normal=[1, 0, 0], center=[0, 0, 0], size=size, step=step,
            color=(0.2, 0.2, 0.7)
        )

    @staticmethod
    def draw_planes_3d(size=10.0, step=1.0, colored=False):
        """Nakreslí 3D plochy s výplňou a gridom"""

        if colored:
            # Farebné plochy - nepriehľadné
            alpha = 1.0
            transparent = False
            xy_color = (0.2, 0.8, 0.2)  # Zelená pre XY
            xz_color = (0.8, 0.2, 0.2)  # Červená pre XZ
            yz_color = (0.2, 0.2, 0.8)  # Modrá pre YZ
        else:
            # Neutrálne sivé plochy - priesvitné
            alpha = 1.0
            transparent = False
            xy_color = (0.5, 0.5, 0.5)
            xz_color = (0.5, 0.5, 0.5)
            yz_color = (0.5, 0.5, 0.5)

        # XY rovina (Z=0) - zelená
        GridRenderer.draw_filled_plane(
            normal=[0, 0, 1], center=[0, 0, 0], size=size,
            color=xy_color, alpha=alpha, transparent=transparent
        )
        GridRenderer.draw_grid_in_plane(
            normal=[0, 0, 1], center=[0, 0, 0], size=size, step=step,
            color=(0.1, 0.4, 0.1) if colored else (0.3, 0.3, 0.3)
        )

        # XZ rovina (Y=0) - červená
        GridRenderer.draw_filled_plane(
            normal=[0, 1, 0], center=[0, 0, 0], size=size,
            color=xz_color, alpha=alpha, transparent=transparent
        )
        GridRenderer.draw_grid_in_plane(
            normal=[0, 1, 0], center=[0, 0, 0], size=size, step=step,
            color=(0.4, 0.1, 0.1) if colored else (0.3, 0.3, 0.3)
        )

        # YZ rovina (X=0) - modrá
        GridRenderer.draw_filled_plane(
            normal=[1, 0, 0], center=[0, 0, 0], size=size,
            color=yz_color, alpha=alpha, transparent=transparent
        )
        GridRenderer.draw_grid_in_plane(
            normal=[1, 0, 0], center=[0, 0, 0], size=size, step=step,
            color=(0.1, 0.1, 0.4) if colored else (0.3, 0.3, 0.3)
        )



    @staticmethod
    def draw_grid_in_plane(normal, center=(0, 0, 0), size=10.0, step=1.0, color=(0.4, 0.4, 0.4)):
        """Nakreslí grid v ľubovoľnej rovine definovanej normálou - OPTIMALIZOVANÉ"""
        # Normalizuj normálu
        normal = np.array(normal, dtype=np.float32)
        norm_length = np.linalg.norm(normal)
        if norm_length < 1e-6:
            return
        normal = normal / norm_length

        # Cache key pre tangent vektory
        normal_key = (round(normal[0], 4), round(normal[1], 4), round(normal[2], 4))

        if normal_key in GridRenderer._plane_tangents_cache:
            tangent1, tangent2 = GridRenderer._plane_tangents_cache[normal_key]
        else:
            # Nájdi dva kolmé vektory v rovine
            if abs(normal[2]) < 0.9:
                ref = np.array([0, 0, 1], dtype=np.float32)
            else:
                ref = np.array([1, 0, 0], dtype=np.float32)

            tangent1 = np.cross(normal, ref)
            tangent1 = tangent1 / np.linalg.norm(tangent1)
            tangent2 = np.cross(normal, tangent1)
            tangent2 = tangent2 / np.linalg.norm(tangent2)

            # Ulož do cache
            GridRenderer._plane_tangents_cache[normal_key] = (tangent1.copy(), tangent2.copy())

        center = np.array(center, dtype=np.float32)

        # Nakresli grid s polygon offset
        glPushAttrib(GL_LINE_BIT | GL_CURRENT_BIT | GL_DEPTH_BUFFER_BIT)
        glDepthFunc(GL_LEQUAL)
        glEnable(GL_POLYGON_OFFSET_LINE)
        glPolygonOffset(1.0, 1.0)

        glLineWidth(1.5)
        glColor3f(color[0], color[1], color[2])
        glBegin(GL_LINES)

        size_int = int(size)
        step_int = max(1, int(step))

        # Čiary pozdĺž tangent1
        for i in range(-size_int, size_int + 1, step_int):
            t = float(i)
            p1 = center + tangent1 * (-size) + tangent2 * t
            p2 = center + tangent1 * size + tangent2 * t
            glVertex3f(p1[0], p1[1], p1[2])
            glVertex3f(p2[0], p2[1], p2[2])

        # Čiary pozdĺž tangent2
        for i in range(-size_int, size_int + 1, step_int):
            t = float(i)
            p1 = center + tangent1 * t + tangent2 * (-size)
            p2 = center + tangent1 * t + tangent2 * size
            glVertex3f(p1[0], p1[1], p1[2])
            glVertex3f(p2[0], p2[1], p2[2])

        glEnd()

        glDisable(GL_POLYGON_OFFSET_LINE)
        glPopAttrib()

    @classmethod
    def clear_cache(cls):
        """Vymaže cache - volať pri zmene scény"""
        cls._plane_tangents_cache.clear()