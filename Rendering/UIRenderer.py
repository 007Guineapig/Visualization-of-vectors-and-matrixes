import math

import pygame
from OpenGL.GL import *
from Cache.TextureCache import TextureCache

from Configurations.Colors import Colors


class UIRenderer:
    """Moderný UI renderer s vylepšeným dizajnom"""

    @staticmethod
    def draw_rounded_rect(x, y, w, h, radius, color, alpha=1.0):
        """Nakreslí zaoblený obdĺžnik"""
        if alpha < 1.0:
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glColor4f(color[0], color[1], color[2], alpha)
        else:
            glColor3f(*color)

        radius = min(radius, w / 2, h / 2)
        segments = 8

        # Hlavné obdĺžniky
        glBegin(GL_QUADS)
        # Stred
        glVertex2f(x + radius, y)
        glVertex2f(x + w - radius, y)
        glVertex2f(x + w - radius, y + h)
        glVertex2f(x + radius, y + h)
        # Ľavý
        glVertex2f(x, y + radius)
        glVertex2f(x + radius, y + radius)
        glVertex2f(x + radius, y + h - radius)
        glVertex2f(x, y + h - radius)
        # Pravý
        glVertex2f(x + w - radius, y + radius)
        glVertex2f(x + w, y + radius)
        glVertex2f(x + w, y + h - radius)
        glVertex2f(x + w - radius, y + h - radius)
        # Horný
        glVertex2f(x + radius, y)
        glVertex2f(x + w - radius, y)
        glVertex2f(x + w - radius, y + radius)
        glVertex2f(x + radius, y + radius)
        # Dolný
        glVertex2f(x + radius, y + h - radius)
        glVertex2f(x + w - radius, y + h - radius)
        glVertex2f(x + w - radius, y + h)
        glVertex2f(x + radius, y + h)
        glEnd()

        # Rohové oblúky
        corners = [
            (x + radius, y + radius, 180, 270),
            (x + w - radius, y + radius, 270, 360),
            (x + w - radius, y + h - radius, 0, 90),
            (x + radius, y + h - radius, 90, 180),
        ]

        for cx, cy, start, end in corners:
            glBegin(GL_TRIANGLE_FAN)
            glVertex2f(cx, cy)
            for i in range(segments + 1):
                angle = math.radians(start + (end - start) * i / segments)
                glVertex2f(cx + radius * math.cos(angle), cy + radius * math.sin(angle))
            glEnd()

        if alpha < 1.0:
            glDisable(GL_BLEND)

    @staticmethod
    def draw_text_2d(text, position, color=(1, 1, 1), font_size=20):
        """Nakreslí text v 2D"""
        if not text:
            return 0

        result = TextureCache.get_texture(text, "Arial", font_size, color)
        if result[0] is None:
            return 0

        tex_id, w, h = result

        glEnable(GL_TEXTURE_2D)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glDisable(GL_DEPTH_TEST)

        x, y = position
        glColor3f(1, 1, 1)
        glBindTexture(GL_TEXTURE_2D, tex_id)

        glBegin(GL_QUADS)
        glTexCoord2f(0, 0);
        glVertex2f(x, y)
        glTexCoord2f(1, 0);
        glVertex2f(x + w, y)
        glTexCoord2f(1, 1);
        glVertex2f(x + w, y + h)
        glTexCoord2f(0, 1);
        glVertex2f(x, y + h)
        glEnd()

        glEnable(GL_DEPTH_TEST)
        glDisable(GL_TEXTURE_2D)
        glDisable(GL_BLEND)

        return w

    @staticmethod
    def draw_button_2d(x, y, w, h, label, active=False, is_dark=True, hover=False, primary=False):
        """Nakreslí moderné tlačidlo"""
        WIDTH, HEIGHT = pygame.display.get_window_size()

        glDisable(GL_DEPTH_TEST)
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, WIDTH, HEIGHT, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        radius = 8

        # Farby podľa stavu
        if primary:
            bg_color = Colors.PRIMARY_HOVER if hover else Colors.PRIMARY
            text_color = (1, 1, 1)
            # Tieň pre primary
            shadow = (Colors.PRIMARY[0] * 0.4, Colors.PRIMARY[1] * 0.4, Colors.PRIMARY[2] * 0.4)
            UIRenderer.draw_rounded_rect(x + 1, y + 2, w, h, radius, shadow, 0.3)
        else:
            if is_dark:
                bg_color = Colors.DARK_HOVER if hover else (Colors.DARK_CARD if active else Colors.DARK_SURFACE)
                text_color = Colors.TEXT_DARK
                border_color = Colors.BORDER_DARK
            else:
                bg_color = Colors.LIGHT_HOVER if hover else (Colors.LIGHT_CARD if active else Colors.LIGHT_SURFACE)
                text_color = Colors.TEXT_LIGHT
                border_color = Colors.BORDER_LIGHT

        # Pozadie
        UIRenderer.draw_rounded_rect(x, y, w, h, radius, bg_color)

        # Okraj pre ne-primary tlačidlá
        if not primary:
            glLineWidth(1.0)
            glColor3f(*border_color)
            # Zjednodušený okraj
            glBegin(GL_LINE_LOOP)
            glVertex2f(x + radius, y)
            glVertex2f(x + w - radius, y)
            glVertex2f(x + w, y + radius)
            glVertex2f(x + w, y + h - radius)
            glVertex2f(x + w - radius, y + h)
            glVertex2f(x + radius, y + h)
            glVertex2f(x, y + h - radius)
            glVertex2f(x, y + radius)
            glEnd()

        # Text
        if label:
            result = TextureCache.get_texture(label, "Arial", 15, text_color)
            if result[0] is not None:
                tex_id, tw, th = result
                glEnable(GL_TEXTURE_2D)
                glEnable(GL_BLEND)
                glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
                tx = x + (w - tw) / 2
                ty = y + (h - th) / 2
                glColor3f(1, 1, 1)
                glBindTexture(GL_TEXTURE_2D, tex_id)
                glBegin(GL_QUADS)
                glTexCoord2f(0, 0);
                glVertex2f(tx, ty)
                glTexCoord2f(1, 0);
                glVertex2f(tx + tw, ty)
                glTexCoord2f(1, 1);
                glVertex2f(tx + tw, ty + th)
                glTexCoord2f(0, 1);
                glVertex2f(tx, ty + th)
                glEnd()
                glDisable(GL_TEXTURE_2D)
                glDisable(GL_BLEND)

        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glEnable(GL_DEPTH_TEST)

    @staticmethod
    def draw_input_box_3d(x, y, w, h, text, active=False, fill_color=None,
                          fill_color_outline=None, is_dark=True):
        """Nakreslí moderný input box"""
        WIDTH, HEIGHT = pygame.display.get_window_size()

        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, WIDTH, HEIGHT, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        glDisable(GL_DEPTH_TEST)

        radius = 6

        # Farby
        if is_dark:
            bg_color = Colors.DARK_CARD if active else Colors.DARK_SURFACE
            border_color = Colors.PRIMARY if active else Colors.BORDER_DARK
            text_color = Colors.TEXT_DARK
        else:
            bg_color = Colors.LIGHT_SURFACE
            border_color = Colors.PRIMARY if active else Colors.BORDER_LIGHT
            text_color = Colors.TEXT_LIGHT

        # Ak je zadaná vlastná farba okraja, použi ju
        if fill_color_outline and not active:
            border_color = fill_color_outline

        # Pozadie
        UIRenderer.draw_rounded_rect(x, y, w, h, radius, bg_color)

        # Okraj
        glLineWidth(2 if active else 1)
        glColor3f(*border_color)
        glBegin(GL_LINE_LOOP)
        glVertex2f(x + radius, y)
        glVertex2f(x + w - radius, y)
        glVertex2f(x + w, y + radius)
        glVertex2f(x + w, y + h - radius)
        glVertex2f(x + w - radius, y + h)
        glVertex2f(x + radius, y + h)
        glVertex2f(x, y + h - radius)
        glVertex2f(x, y + radius)
        glEnd()

        # Text
        if text:
            result = TextureCache.get_texture(text, "Arial", 15, text_color)
            if result[0] is not None:
                tex_id, tw, th = result
                glEnable(GL_TEXTURE_2D)
                glEnable(GL_BLEND)
                glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
                tx = x + (w - tw) / 2
                ty = y + (h - th) / 2
                glColor3f(1, 1, 1)
                glBindTexture(GL_TEXTURE_2D, tex_id)
                glBegin(GL_QUADS)
                glTexCoord2f(0, 0);
                glVertex2f(tx, ty)
                glTexCoord2f(1, 0);
                glVertex2f(tx + tw, ty)
                glTexCoord2f(1, 1);
                glVertex2f(tx + tw, ty + th)
                glTexCoord2f(0, 1);
                glVertex2f(tx, ty + th)
                glEnd()
                glDisable(GL_TEXTURE_2D)
                glDisable(GL_BLEND)

        # Kurzor keď je aktívny
        if active and int(pygame.time.get_ticks() / 500) % 2 == 0:
            cursor_x = x + w / 2
            if text:
                result = TextureCache.get_texture(text, "Arial", 15, text_color)
                if result[0]:
                    cursor_x = x + (w + result[1]) / 2 + 2
            glColor3f(*Colors.PRIMARY)
            glLineWidth(2)
            glBegin(GL_LINES)
            glVertex2f(cursor_x, y + 6)
            glVertex2f(cursor_x, y + h - 6)
            glEnd()

        glEnable(GL_DEPTH_TEST)
        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)

    @staticmethod
    def draw_card(x, y, w, h, is_dark=True, title=None):
        """Nakreslí kartu/kontajner"""
        WIDTH, HEIGHT = pygame.display.get_window_size()

        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, WIDTH, HEIGHT, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        glDisable(GL_DEPTH_TEST)

        radius = 12

        if is_dark:
            bg_color = Colors.DARK_CARD
            title_color = Colors.TEXT_DARK
        else:
            bg_color = Colors.LIGHT_CARD
            title_color = Colors.TEXT_LIGHT

        # Tieň
        UIRenderer.draw_rounded_rect(x + 2, y + 3, w, h, radius, (0, 0, 0), 0.15)

        # Pozadie
        UIRenderer.draw_rounded_rect(x, y, w, h, radius, bg_color)

        # Titulok
        if title:
            result = TextureCache.get_texture(title, "Arial", 16, title_color)
            if result[0] is not None:
                tex_id, tw, th = result
                glEnable(GL_TEXTURE_2D)
                glEnable(GL_BLEND)
                glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
                glColor3f(1, 1, 1)
                glBindTexture(GL_TEXTURE_2D, tex_id)
                tx, ty = x + 16, y + 14
                glBegin(GL_QUADS)
                glTexCoord2f(0, 0);
                glVertex2f(tx, ty)
                glTexCoord2f(1, 0);
                glVertex2f(tx + tw, ty)
                glTexCoord2f(1, 1);
                glVertex2f(tx + tw, ty + th)
                glTexCoord2f(0, 1);
                glVertex2f(tx, ty + th)
                glEnd()
                glDisable(GL_TEXTURE_2D)
                glDisable(GL_BLEND)

        glEnable(GL_DEPTH_TEST)
        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)