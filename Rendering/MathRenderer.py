import pygame

from Rendering.EnhancedUIRenderer import EnhancedUIRenderer
from Rendering.UIRenderer import UIRenderer
from OpenGL.GL import *


class MathRenderer:
    """Renderer pre matematické výrazy - S ANIMÁCIOU"""

    @staticmethod
    def draw_vector_math(x, y, vec, color=(1, 1, 1), font_size=24, highlight=False, alpha=1.0):
        """Nakreslí vektor v matematickom zápise"""
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        width, height = pygame.display.get_window_size()
        glOrtho(0, width, height, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        glDisable(GL_DEPTH_TEST)

        # Výška vektora
        line_height = font_size + 5
        vec_height = len(vec) * line_height

        # Farba s alpha
        if alpha < 1.0:
            draw_color = (color[0], color[1], color[2], alpha)
        else:
            draw_color = color

        # Zátvorky
        bracket_color = color if highlight else (0.7, 0.7, 0.7)
        thickness = 3 if highlight else 2
        EnhancedUIRenderer.draw_parenthesis(x - 5, y, vec_height, left=True,
                                            color=bracket_color, thickness=thickness)
        EnhancedUIRenderer.draw_parenthesis(x + 45, y, vec_height, left=False,
                                            color=bracket_color, thickness=thickness)

        # Zložky vektora
        text_color = color if highlight else (0.8, 0.8, 0.8)
        for i, val in enumerate(vec):
            if isinstance(val, float):
                text = f"{val:.1f}"
            else:
                text = str(val)
            UIRenderer.draw_text_2d(text, (x + 5, y + i * line_height),
                                    color=text_color, font_size=font_size)

        glEnable(GL_DEPTH_TEST)
        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)

        return x + 55

    @staticmethod
    def draw_animated_vector_math(x, y, vec_from, vec_to, progress, color=(1, 1, 1),
                                  font_size=24, highlight=False):
        """Nakreslí animovaný vektor - interpoluje hodnoty medzi vec_from a vec_to"""
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        width, height = pygame.display.get_window_size()
        glOrtho(0, width, height, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        glDisable(GL_DEPTH_TEST)

        line_height = font_size + 5
        vec_height = len(vec_to) * line_height

        bracket_color = color if highlight else (0.7, 0.7, 0.7)
        thickness = 3 if highlight else 2
        EnhancedUIRenderer.draw_parenthesis(x - 5, y, vec_height, left=True,
                                            color=bracket_color, thickness=thickness)
        EnhancedUIRenderer.draw_parenthesis(x + 45, y, vec_height, left=False,
                                            color=bracket_color, thickness=thickness)

        text_color = color if highlight else (0.8, 0.8, 0.8)
        for i in range(len(vec_to)):
            # Interpoluj hodnotu
            if vec_from is not None and i < len(vec_from):
                val = vec_from[i] + (vec_to[i] - vec_from[i]) * progress
            else:
                val = vec_to[i] * progress

            text = f"{val:.1f}"
            UIRenderer.draw_text_2d(text, (x + 5, y + i * line_height),
                                    color=text_color, font_size=font_size)

        glEnable(GL_DEPTH_TEST)
        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)

        return x + 55

    @staticmethod
    def draw_component_operation_animated(x, y, vec1, vec2, op, progress,
                                          color=(1, 1, 1), font_size=20):
        """Nakreslí animovanú operáciu medzi komponentmi (3 + 2 → 5)"""
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        width, height = pygame.display.get_window_size()
        glOrtho(0, width, height, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        glDisable(GL_DEPTH_TEST)

        line_height = font_size + 5
        vec_height = len(vec1) * line_height

        EnhancedUIRenderer.draw_parenthesis(x - 5, y, vec_height, left=True,
                                            color=color, thickness=3)
        EnhancedUIRenderer.draw_parenthesis(x + 100, y, vec_height, left=False,
                                            color=color, thickness=3)

        for i in range(len(vec1)):
            v1 = vec1[i]
            v2 = vec2[i]

            if op == "+":
                result = v1 + v2
            elif op == "-":
                result = v1 - v2
            else:
                result = v1 + v2

            # Animovaný text: na začiatku "3 + 2", na konci "5"
            if progress < 0.5:
                # Zobraz operáciu
                v1_str = f"{v1:.1f}" if isinstance(v1, float) else str(v1)
                v2_str = f"{v2:.1f}" if isinstance(v2, float) else str(v2)
                text = f"{v1_str} {op} {v2_str}"
                # Fade out
                alpha = 1.0 - (progress * 2)
            else:
                # Zobraz výsledok
                text = f"{result:.1f}" if isinstance(result, float) else str(result)
                # Fade in
                alpha = (progress - 0.5) * 2

            # Farba s alpha simuláciou (svetlejšia = viac priesvitná)
            adj_color = (
                color[0] * alpha + (1 - alpha) * 0.3,
                color[1] * alpha + (1 - alpha) * 0.3,
                color[2] * alpha + (1 - alpha) * 0.3
            )

            UIRenderer.draw_text_2d(text, (x + 5, y + i * line_height),
                                    color=adj_color, font_size=font_size)

        glEnable(GL_DEPTH_TEST)
        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)

        return x + 110

    @staticmethod
    def draw_vector_component_operation(x, y, vec1, vec2, op="+", color=(1, 1, 1), font_size=20):
        """Nakreslí vektor s operáciami medzi zložkami (napr. 2+3, 4+7)"""
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        width, height = pygame.display.get_window_size()
        glOrtho(0, width, height, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        glDisable(GL_DEPTH_TEST)

        line_height = font_size + 5
        vec_height = len(vec1) * line_height

        EnhancedUIRenderer.draw_parenthesis(x - 5, y, vec_height, left=True,
                                            color=color, thickness=3)
        EnhancedUIRenderer.draw_parenthesis(x + 80, y, vec_height, left=False,
                                            color=color, thickness=3)

        for i in range(len(vec1)):
            v1 = f"{vec1[i]:.1f}" if isinstance(vec1[i], float) else str(vec1[i])
            v2 = f"{vec2[i]:.1f}" if isinstance(vec2[i], float) else str(vec2[i])
            text = f"{v1} {op} {v2}"
            UIRenderer.draw_text_2d(text, (x + 5, y + i * line_height),
                                    color=color, font_size=font_size)

        glEnable(GL_DEPTH_TEST)
        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)

        return x + 90

    @staticmethod
    def draw_operator(x, y, op, color=(1, 1, 1), font_size=32):
        """Nakreslí operátor (+, -, ×)"""
        UIRenderer.draw_text_2d(op, (x, y + 10), color=color, font_size=font_size)
        return x + 30

    @staticmethod
    def draw_equals(x, y, color=(1, 1, 1), font_size=32):
        """Nakreslí rovná sa"""
        UIRenderer.draw_text_2d("=", (x, y + 10), color=color, font_size=font_size)
        return x + 30

    @staticmethod
    def draw_equals_animated(x, y, progress, color=(1, 1, 1), font_size=32):
        """Nakreslí animované rovná sa (fade in)"""
        # Fade in efekt
        alpha = progress
        adj_color = (
            color[0] * alpha + (1 - alpha) * 0.2,
            color[1] * alpha + (1 - alpha) * 0.2,
            color[2] * alpha + (1 - alpha) * 0.2
        )
        UIRenderer.draw_text_2d("=", (x, y + 10), color=adj_color, font_size=font_size)
        return x + 30

    @staticmethod
    def draw_scalar(x, y, scalar, color=(1, 1, 1), font_size=28):
        """Nakreslí skalár"""
        text = f"{scalar:.1f}" if isinstance(scalar, float) else str(scalar)
        UIRenderer.draw_text_2d(text, (x, y + 12), color=color, font_size=font_size)
        return x + 30
