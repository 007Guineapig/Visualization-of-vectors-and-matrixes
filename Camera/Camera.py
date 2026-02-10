import math

import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

class Camera:
    """Správa kamery a viewportu"""

    def _lerp_angle(self, start, end, t):
        """Interpolácia uhlov - vždy ide kratšou cestou"""
        diff = end - start

        # Normalizuj rozdiel na rozsah -180 až 180
        while diff > 180:
            diff -= 360
        while diff < -180:
            diff += 360

        return start + diff * t

    def __init__(self, width, height):
        self.width = width

        self.height = height

        self.distance = 7.0
        self.azimuth = 45.0
        self.elevation = 25.0
        self.target = [0.0, 0.0, 0.0]
        self.pan_offset_x = 0.0
        self.pan_offset_y = 0.0
        self.ortho_scale = 6.0

        # Camera movement
        self.rotating = False
        self.panning = False
        self.last_mouse = (0, 0)
        self.yaw_speed = 0.3
        self.pitch_speed = 0.3
        self.pan_speed = 0.01
        self.zoom_speed = 1.5

        self.animating_to_plane = False
        self.animation_progress = 0.0
        self.animation_speed = 1.5
        self.start_distance = self.distance
        self.start_azimuth = self.azimuth
        self.start_elevation = self.elevation
        self.target_distance = self.distance
        self.target_azimuth = self.azimuth
        self.target_elevation = self.elevation


    # V triede Camera - opravená metóda move_to_plane pre lepšie uhly
    # V triede Camera - opravená metóda move_to_plane
    def move_to_plane(self, plane_type, distance=10.0, custom_view=None):
        """Nastaví cieľovú pozíciu kamery pre danú rovinu"""
        self.start_distance = self.distance
        self.start_azimuth = self.azimuth
        self.start_elevation = self.elevation
        self.target_distance = distance

        if custom_view:  # Pre 3D s vlastným výpočtom
            self.target_azimuth = custom_view['azimuth']
            self.target_elevation = custom_view['elevation']
        elif plane_type == "XY":  # Z=0, pozeráme SPREDU (v smere osi Z)
            self.target_azimuth = 0.0
            self.target_elevation = 0.0
        elif plane_type == "XZ":  # Y=0, pozeráme ZHORA (v smere osi Y)
            self.target_azimuth = 0.0
            self.target_elevation = 90.0  # OPRAVA: presne 90°
        elif plane_type == "YZ":  # X=0, pozeráme ZBOKU (v smere osi X)
            self.target_azimuth = 90.0
            self.target_elevation = 0.0
        else:  # Všeobecný 3D pohľad
            self.target_azimuth = 45.0
            self.target_elevation = 25.0

        self.animating_to_plane = True
        self.animation_progress = 0.0

    def animate_to_plane(self, plane_type, dt):
        """Plynulo animuje kameru do roviny (XY, XZ, YZ)"""
        if not self.animating_to_plane:
            return

        self.animation_progress += dt * self.animation_speed

        if self.animation_progress >= 1.0:
            self.animation_progress = 1.0
            self.animating_to_plane = False
            self.distance = self.target_distance
            self.azimuth = self.target_azimuth
            self.elevation = self.target_elevation
        else:
            # Smooth interpolation (ease-in-out)
            t = self.animation_progress
            smooth_t = t * t * (3.0 - 2.0 * t)

            self.distance = self.start_distance + (self.target_distance - self.start_distance) * smooth_t

            # OPRAVA: Použiť lerp_angle pre azimut
            self.azimuth = self._lerp_angle(self.start_azimuth, self.target_azimuth, smooth_t)

            # Elevácia nepotrebuje špeciálnu interpoláciu (je obmedzená na -90 až 90)
            self.elevation = self.start_elevation + (self.target_elevation - self.start_elevation) * smooth_t

    def get_position(self):
        """Vráti pozíciu kamery v 3D priestore"""
        self.width, self.height = pygame.display.get_window_size()
        rad_az = math.radians(self.azimuth)
        rad_el = math.radians(self.elevation)
        cam_x = self.target[0] + self.distance * math.cos(rad_el) * math.sin(rad_az)
        cam_y = self.target[1] + self.distance * math.sin(rad_el)
        cam_z = self.target[2] + self.distance * math.cos(rad_el) * math.cos(rad_az)
        return (cam_x, cam_y, cam_z)

    def setup_3d_projection(self):
        """Nastaví 3D projekciu"""
        self.width, self.height = pygame.display.get_window_size()
        cam_x, cam_y, cam_z = self.get_position()

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        #gluPerspective(45, (self.width / self.height), 0.1, 500.0)
        gluPerspective(45, self.width / self.height, 1.0, 100.0)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        # Pri pohľade presne zhora/zdola použijem iný "up" vektor
        if abs(self.elevation) > 89.0:
            # Pozeráme zhora alebo zdola - up vektor musí byť v XZ rovine
            up_x = -math.sin(math.radians(self.azimuth))
            up_y = 0
            up_z = -math.cos(math.radians(self.azimuth))
        else:
            up_x, up_y, up_z = 0, 1, 0

        gluLookAt(cam_x, cam_y, cam_z,
                  self.target[0], self.target[1], self.target[2],
                  up_x, up_y, up_z)

    def setup_2d_projection(self):
        """Nastaví 2D ortografickú projekciu s aspect ratio"""
        self.width, self.height = pygame.display.get_window_size()

        # OPRAVA: Pridaj aspect ratio do projekcie
        aspect = self.width / self.height

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()

        if aspect >= 1.0:  # Široké okno
            glOrtho(-self.ortho_scale * aspect + self.pan_offset_x,
                    self.ortho_scale * aspect + self.pan_offset_x,
                    -self.ortho_scale + self.pan_offset_y,
                    self.ortho_scale + self.pan_offset_y,
                    -1, 1)
        else:  # Vysoké okno
            glOrtho(-self.ortho_scale + self.pan_offset_x,
                    self.ortho_scale + self.pan_offset_x,
                    -self.ortho_scale / aspect + self.pan_offset_y,
                    self.ortho_scale / aspect + self.pan_offset_y,
                    -1, 1)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

    def handle_rotation(self, dx, dy):
        """Spracuje rotáciu kamery"""
        self.azimuth -= dx * self.yaw_speed
        self.elevation = max(-89.9, min(89.9, self.elevation + dy * self.pitch_speed))

    def handle_panning_3d(self, dx, dy):
        """Spracuje panning v 3D"""
        rad_az = math.radians(self.azimuth)
        rad_el = math.radians(self.elevation)
        fx = math.cos(rad_el) * math.sin(rad_az)
        fy = math.sin(rad_el)
        fz = math.cos(rad_el) * math.cos(rad_az)
        rx, ry, rz = fz, 0.0, -fx
        ux = ry * fz - rz * fy
        uy = rz * fx - rx * fz
        uz = rx * fy - ry * fx
        self.target[0] -= (rx * dx + ux * dy) * self.pan_speed
        self.target[1] -= (ry * dx + uy * dy) * self.pan_speed
        self.target[2] -= (rz * dx + uz * dy) * self.pan_speed

    def handle_panning_2d(self, dx, dy):
        """Spracuje panning v 2D"""
        self.pan_offset_x -= dx * 0.002 * self.ortho_scale
        self.pan_offset_y += dy * 0.002 * self.ortho_scale

    def zoom_in(self, is_2d, locked_scale=None):
        """Priblíženie"""
        if is_2d:
            new_scale = self.ortho_scale / 1.1
            # Ak je locked_scale, nepriblížuj viac ako je povolené
            if locked_scale is not None:
                new_scale = max(new_scale, locked_scale * 0.5)  # Povoľ priblížiť max 2x
            self.ortho_scale = new_scale
        else:
            self.distance = max(1.0, self.distance - self.zoom_speed)

    def zoom_out(self, is_2d, locked_scale=None):
        """Oddialenie"""
        if is_2d:
            new_scale = self.ortho_scale * 1.1
            # Ak je locked_scale, neodďaľuj viac ako je povolené
            if locked_scale is not None:
                new_scale = min(new_scale, locked_scale * 1.5)  # Povoľ oddialiť max 1.5x
            self.ortho_scale = new_scale
        else:
            self.distance += self.zoom_speed

    def reset(self):
        """Reset kamery na východzie hodnoty"""
        self.distance = 7.0
        self.azimuth = 45.0
        self.elevation = 25.0
        self.target = [0.0, 0.0, 0.0]
        self.pan_offset_x = 0.0
        self.pan_offset_y = 0.0
        self.ortho_scale = 6.0