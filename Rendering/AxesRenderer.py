import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
class AxesRenderer:
    """Renderer pre súradnicové osi"""

    @staticmethod
    def draw_axes_2d(ortho_scale, pan_x, pan_y, width, height, ui_renderer):
        """Nakreslí 2D osi s aspect ratio korekciou"""
        width, height = pygame.display.get_window_size()

        # OPRAVA: Vypočítaj aspect ratio a správne hranice
        aspect = width / height

        if aspect >= 1.0:  # Široké okno
            left = -ortho_scale * aspect + pan_x
            right = ortho_scale * aspect + pan_x
            bottom = -ortho_scale + pan_y
            top = ortho_scale + pan_y
        else:  # Vysoké okno
            left = -ortho_scale + pan_x
            right = ortho_scale + pan_x
            bottom = -ortho_scale / aspect + pan_y
            top = ortho_scale / aspect + pan_y

        # Draw axis lines
        glLineWidth(3.0)
        glBegin(GL_LINES)
        glColor3f(1, 0, 0)  # X axis
        glVertex2f(left, 0)
        glVertex2f(right, 0)
        glColor3f(0, 1, 0)  # Y axis
        glVertex2f(0, bottom)
        glVertex2f(0, top)
        glEnd()

        # Draw labels
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, width, height, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        def world_to_screen(wx, wy, color=(1, 1, 1), label=""):
            sx = (wx - left) / (right - left) * width
            sy = height - (wy - bottom) / (top - bottom) * height

            if label in ["X1+", "X1-"]:
                sx += -30 if label == "X1+" else 10
                sy += 5
            if label in ["X2+", "X2-"]:
                sy += -30 if label == "X2-" else 10
                sx += 5

            ui_renderer.draw_text_2d(label, (sx, sy), color=color, font_size=20)

        world_to_screen(right, 0, color=(1, 0, 0), label="X1+")
        world_to_screen(left, 0, color=(1, 0, 0), label="X1-")
        world_to_screen(0, top, color=(0, 1, 0), label="X2+")
        world_to_screen(0, bottom, color=(0, 1, 0), label="X2-")

        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)

    @staticmethod
    def draw_axes_3d(length, camera_pos, vector_renderer):
        """Nakreslí 3D osi"""
        # Draw axis lines
        length_line = length - 0.5
        glLineWidth(5.0)
        glBegin(GL_LINES)
        # X axis (red)
        glColor3f(1, 0, 0)
        glVertex3f(-length_line, 0, 0)
        glVertex3f(length_line, 0, 0)
        # Y axis (green)
        glColor3f(0, 1, 0)
        glVertex3f(0, -length_line, 0)
        glVertex3f(0, length_line, 0)
        # Z axis (blue)
        glColor3f(0, 0, 1)
        glVertex3f(0, 0, -length_line)
        glVertex3f(0, 0, length_line)
        glEnd()
        glLineWidth(2.0)

        # Draw arrowheads
        arrow_size = 1
        vector_renderer.draw_conehead_3d((0, 0, 0), (length, 0, 0),
                                         color=(1, 0, 0), size=arrow_size,
                                         camera_pos=camera_pos)
        vector_renderer.draw_conehead_3d((0, 0, 0), (0, length, 0),
                                         color=(0, 1, 0), size=arrow_size,
                                         camera_pos=camera_pos)
        vector_renderer.draw_conehead_3d((0, 0, 0), (0, 0, length),
                                         color=(0, 0, 1), size=arrow_size,
                                         camera_pos=camera_pos)
        vector_renderer.draw_conehead_3d((0, 0, 0), (-length, 0, 0),
                                         color=(1, 0, 0), size=arrow_size,
                                         camera_pos=camera_pos)
        vector_renderer.draw_conehead_3d((0, 0, 0), (0, -length, 0),
                                         color=(0, 1, 0), size=arrow_size,
                                         camera_pos=camera_pos)
        vector_renderer.draw_conehead_3d((0, 0, 0), (0, 0, -length),
                                         color=(0, 0, 1), size=arrow_size,
                                         camera_pos=camera_pos)