
import pygame
from pygame.locals import *
from OpenGL.GL import *
from Cache.FontCache import FontCache
from OpenGL.GLU import *

class TextureCache:
    """Cache pre text textúry - vytvorí textúru len raz pre každý unikátny text"""
    _cache = {}
    _max_size = 500
    _access_order = []

    @classmethod
    def get_texture(cls, text, font_name, font_size, color):
        """Vráti (tex_id, width, height) pre daný text"""
        if not text:
            return None, 0, 0

        # Normalizuj color na tuple
        color_key = tuple(int(c * 255) for c in color[:3])
        key = (text, font_name, font_size, color_key)

        if key in cls._cache:
            # Update LRU
            if key in cls._access_order:
                cls._access_order.remove(key)
            cls._access_order.append(key)
            return cls._cache[key]

        # Vytvor novú textúru
        font = FontCache.get_font(font_name, font_size)
        text_surface = font.render(text, True, color_key)
        text_data = pygame.image.tostring(text_surface, "RGBA", False)  # OPRAVA: False namiesto True
        w, h = text_surface.get_size()

        tex_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tex_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, text_data)

        # Ulož do cache
        cls._cache[key] = (tex_id, w, h)
        cls._access_order.append(key)

        # LRU eviction ak je cache plný
        while len(cls._cache) > cls._max_size:
            old_key = cls._access_order.pop(0)
            if old_key in cls._cache:
                old_tex_id, _, _ = cls._cache[old_key]
                glDeleteTextures([old_tex_id])
                del cls._cache[old_key]

        return cls._cache[key]

    @classmethod
    def clear(cls):
        """Vymaže všetky textúry z cache"""
        for tex_id, _, _ in cls._cache.values():
            try:
                glDeleteTextures([tex_id])
            except:
                pass
        cls._cache.clear()
        cls._access_order.clear()