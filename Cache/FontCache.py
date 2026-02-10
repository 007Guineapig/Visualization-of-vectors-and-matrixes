import pygame


class FontCache:
    """Cache pre fonty - načíta font len raz"""
    _fonts = {}

    @classmethod
    def get_font(cls, name, size, bold=True):
        key = (name, size, bold)
        if key not in cls._fonts:
            cls._fonts[key] = pygame.font.SysFont(name, size, bold)
        return cls._fonts[key]