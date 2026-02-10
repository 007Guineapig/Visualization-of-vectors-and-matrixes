import pygame

from Rendering.UIRenderer import UIRenderer
from OpenGL.GL import *


class EnhancedUIRenderer(UIRenderer):
    """Extended UI renderer with additional components"""
    @staticmethod
    def draw_basis_matrix_top_right(matrix, width, height, font_size=24, color=(1, 1, 1)):
        """Draw basis matrix in top right corner"""
        width, height = pygame.display.get_window_size()

        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, width, height, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        glDisable(GL_DEPTH_TEST)

        rows = len(matrix)
        line_height = font_size + 5
        total_height = line_height * rows
        total_width = 0
        for row in matrix:
            row_text = "  ".join(str(c) for c in row)
            total_width = max(total_width, len(row_text) * (font_size * 0.6))

        top_right_x = width - 50
        top_right_y = 20

        # Draw parentheses
        EnhancedUIRenderer.draw_parenthesis(top_right_x - total_width - 10, top_right_y - 10,
                                            total_height, left=True, color=color, thickness=3)
        EnhancedUIRenderer.draw_parenthesis(top_right_x + 15, top_right_y - 10,
                                            total_height, left=False, color=color, thickness=3)

        # Draw matrix text
        for i, row in enumerate(matrix):
            row_text = " ".join(f"{c:>5}" for c in row)
            UIRenderer.draw_text_2d(row_text,
                                    (top_right_x - total_width - 10, top_right_y - 10 + i * line_height),
                                    color=color, font_size=font_size)

        glEnable(GL_DEPTH_TEST)
        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)

    @staticmethod
    def draw_parenthesis(x, y, height, left=True, color=(1, 1, 1), thickness=3):
        """Draw smooth parenthesis using Bezier curve"""
        glDisable(GL_DEPTH_TEST)
        glLineWidth(thickness)
        glColor3f(*color)

        radius = height * 0.15
        top_y = y
        bottom_y = y + height

        if left:
            p0 = (x, bottom_y)
            p1 = (x - radius, bottom_y - height * 0.15)
            p2 = (x - radius, top_y + height * 0.15)
            p3 = (x, top_y)
        else:
            p0 = (x, bottom_y)
            p1 = (x + radius, bottom_y - height * 0.15)
            p2 = (x + radius, top_y + height * 0.15)
            p3 = (x, top_y)

        glBegin(GL_LINE_STRIP)
        segments = 50
        for i in range(segments + 1):
            t = i / segments
            vx = ((1 - t) ** 3 * p0[0] + 3 * (1 - t) ** 2 * t * p1[0] +
                  3 * (1 - t) * t ** 2 * p2[0] + t ** 3 * p3[0])
            vy = ((1 - t) ** 3 * p0[1] + 3 * (1 - t) ** 2 * t * p1[1] +
                  3 * (1 - t) * t ** 2 * p2[1] + t ** 3 * p3[1])
            glVertex2f(vx, vy)
        glEnd()
        glEnable(GL_DEPTH_TEST)