from main import *

import math
import pygame
import numpy as np
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from numpy import array, cross, linalg
from math import sin, cos, pi
import ast
import random



def draw_vectors_2d_animated(vectors, dt, line_width=6, arrow_fraction=0.15, speed=2.0, ortho_scale=6.0, color = (1,1,1)):
    glPushAttrib(GL_LINE_BIT | GL_CURRENT_BIT)
    glLineWidth(line_width)

    zoom_scale = ortho_scale / 6.0

    for v in vectors:
        vec = v['vec']

        # Single vector case
        if isinstance(vec[0], (int, float)):
            v['progress'] = min(v.get('progress', 0.0) + dt * speed, 1.0)
            x, y = vec[0], vec[1]
            px, py = x * v['progress'], y * v['progress']

            # Draw line
            glColor3f(*color)
            glBegin(GL_LINES)
            glVertex2f(0, 0)
            glVertex2f(px, py)
            glEnd()

            # Draw arrowhead
            angle = math.atan2(y, x)
            vec_len = math.hypot(x, y) * v['progress']
            base_arrow_size = line_width * arrow_fraction * zoom_scale
            arrow_size = min(base_arrow_size, vec_len / 5.0)
            arrow_angle = 0.3
            glBegin(GL_LINES)
            glColor3f(*color)
            # Left side
            glVertex2f(px, py)
            glVertex2f(px - arrow_size * math.cos(angle - arrow_angle),
                       py - arrow_size * math.sin(angle - arrow_angle))
            # Right side
            glVertex2f(px, py)
            glVertex2f(px - arrow_size * math.cos(angle + arrow_angle),
                       py - arrow_size * math.sin(angle + arrow_angle))
            glEnd()

        # Matrix of vectors case
        else:
            # Initialize per-row progress if not exists
            if 'row_progress' not in v:
                v['row_progress'] = [0.0] * len(vec)

            for i, row in enumerate(vec):
                x, y = row[0], row[1]
                v['row_progress'][i] = min(v['row_progress'][i] + dt * speed, 1.0)
                px, py = x * v['row_progress'][i], y * v['row_progress'][i]

                glColor3f(*color)
                glBegin(GL_LINES)
                glVertex2f(0, 0)
                glVertex2f(px, py)
                glEnd()

                angle = math.atan2(y, x)
                vec_len = math.hypot(x, y) * v['row_progress'][i]
                base_arrow_size = line_width * arrow_fraction * zoom_scale
                arrow_size = min(base_arrow_size, vec_len / 3.0)
                arrow_angle = 0.3
                glBegin(GL_LINES)
                glColor3f(*color)
                glVertex2f(px, py)
                glVertex2f(px - arrow_size * math.cos(angle - arrow_angle),
                           py - arrow_size * math.sin(angle - arrow_angle))
                glVertex2f(px, py)
                glVertex2f(px - arrow_size * math.cos(angle + arrow_angle),
                           py - arrow_size * math.sin(angle + arrow_angle))
                glEnd()
    glPopAttrib()

def draw_vectors_3d_animated(vectors, dt, line_width=6, base_arrow_size=0.15,
                             segment_speed=0.3, final_vector_speed=0.2,
                             camera_pos=None, color=(1,1,1)):
    """
    Bezpečná verzia:
    - Podporuje 2D aj 3D vektory (chýbajúce Z sa doplní 0).
    - Stabilné porovnávanie segmentov (body konvertované na tuple).
    - Korektné inicializácie v['progress'], v['segments'], v['row_progress'], v['row_segments'].
    """

    # --- Voliteľné: ochráň dt, aby prvý frame po pauze nepreskočil segmentovú fázu ---
    if dt is None:
        dt = 0.0
    else:
        try:
            dt = float(dt)
        except Exception:
            dt = 0.0
    max_dt = 1.0 / 30.0
    dt = min(max(dt, 0.0), max_dt)

    # OpenGL štýl vykreslenia
    glPushAttrib(GL_LINE_BIT | GL_CURRENT_BIT)
    glLineWidth(line_width)

    reference_distance = 2.0
    base_max_radius = 0.02
    min_radius = 0.005

    def to_pt(p):
        """Normalize to tuples for stable comparisons."""
        # Bezpečne prevedie bod (list/tuple/numpy-like) na 3-prvkový tuple
        try:
            # rozbal čo ide...
            x = float(p[0]) if len(p) > 0 else 0.0
            y = float(p[1]) if len(p) > 1 else 0.0
            z = float(p[2]) if len(p) > 2 else 0.0
        except Exception:
            x, y, z = 0.0, 0.0, 0.0
        return (x, y, z)

    def vec3(v):
        """Coerce any vector-like (x[,y[,z]]) to (x,y,z)."""
        try:
            x = float(v[0]) if len(v) > 0 else 0.0
            y = float(v[1]) if len(v) > 1 else 0.0
            z = float(v[2]) if len(v) > 2 else 0.0
        except Exception:
            # ak by bol v = číslo alebo čokoľvek iné, hrajme na istotu
            try:
                x = float(v)
            except Exception:
                x = 0.0
            y = 0.0
            z = 0.0
        return x, y, z

    def is_number(x):
        try:
            float(x)
            return True
        except Exception:
            return False

    for v in (vectors or []):
        # Vytiahni farbu pre daný vektor (ak má vlastnú), inak default
        v_color = v.get('color', color)

        # Rozlíš, či ide o single vector alebo maticu vektorov
        vec_field = v.get('vec', None)
        if vec_field is None:
            # nič na kreslenie
            continue

        is_single_vector = False
        if isinstance(vec_field, (list, tuple)):
            if len(vec_field) > 0 and is_number(vec_field[0]):
                # napr. [x, y] alebo [x, y, z]
                is_single_vector = True
            # inak predpokladáme maticu vektorov (list of list/tuple)
        else:
            # nie štandardná štruktúra -> preskoč
            continue

        # --- Single vector ---
        if is_single_vector:
            # Safe init stavov
            if 'progress' not in v or not isinstance(v['progress'], (int, float)):
                v['progress'] = 0.0
            if 'segments' not in v or not isinstance(v['segments'], list):
                v['segments'] = []

            # Posuň progress
            if v['progress'] < 0.75:
                v['progress'] = min(v['progress'] + dt * segment_speed, 0.75)
            else:
                v['progress'] = min(v['progress'] + dt * final_vector_speed, 1.0)

            ex, ey, ez = vec3(vec_field)
            segment_points = [to_pt((ex, 0, 0)), to_pt((ex, ey, 0)), to_pt((ex, ey, ez))]

            # Dokresli už hotové segmenty (helpers)
            for s, e in v['segments']:
                radius = compute_radius(s, e, base_max_radius, min_radius, camera_pos, reference_distance)
                draw_oval_line(s, e, radius=radius, color=v_color,alpha =0.5)

            # Aktuálny segment (X/Y/Z)
            if v['progress'] < 0.75:
                local_p = v['progress'] / 0.75
                phase = local_p * 3.0
                seg_index = int(min(phase, 2.999))  # 0,1,2
                seg_progress = phase - seg_index

                seg_start = to_pt((0, 0, 0)) if seg_index == 0 else segment_points[seg_index - 1]
                seg_end = segment_points[seg_index]
                px, py, pz = [s + (e - s) * seg_progress for s, e in zip(seg_start, seg_end)]

                radius = compute_radius(seg_start, (px, py, pz), base_max_radius, min_radius, camera_pos, reference_distance)
                glDepthMask(GL_FALSE)
                draw_oval_line(seg_start, (px, py, pz), radius=radius, color=v_color,alpha =0.5)
                glDepthMask(GL_TRUE)
                # Ulož hotové segmenty pred aktuálnym
                for i in range(seg_index):
                    s = to_pt((0, 0, 0)) if i == 0 else segment_points[i - 1]
                    e = segment_points[i]
                    if (s, e) not in v['segments']:
                        v['segments'].append((s, e))

            # Finálna fáza – kresli celý vektor z (0,0,0) -> (ex,ey,ez)
            else:
                # Uisti sa, že posledný Z helper je evidovaný len vtedy, ak má zmysel (tu môže byť ez == 0, nevadí)
                z_seg = (segment_points[1], segment_points[2])
                if z_seg not in v['segments']:
                    v['segments'].append(z_seg)

                t = (v['progress'] - 0.75) / 0.25  # 0→1
                px, py, pz = ex * t, ey * t, ez * t

                radius = compute_radius((0, 0, 0), (px, py, pz), base_max_radius, min_radius, camera_pos, reference_distance)
                draw_oval_line((0, 0, 0), (px, py, pz), radius=radius, color=v_color)
                if v['progress'] >= 1.0:
                    # Šípka sa dá kresliť aj v 2D (z=0) – stále je to 3D šípka v rovine Z=0
                    draw_conehead_3d((0, 0, 0), (ex, ey, ez),
                                     color=(1.0, 0.4, 0.7),
                                     size=max(0.9, radius * 8),
                                     camera_pos=camera_pos,
                                     radius=0.3)

        # --- Matrix of vectors ---
        else:
            rows = vec_field if isinstance(vec_field, (list, tuple)) else []
            n = len(rows)

            # Safe init radových polí s korektnou dĺžkou
            if 'row_progress' not in v or not isinstance(v['row_progress'], list) or len(v['row_progress']) != n:
                v['row_progress'] = [0.0] * n
            if 'row_segments' not in v or not isinstance(v['row_segments'], list) or len(v['row_segments']) != n:
                v['row_segments'] = [[] for _ in range(n)]

            for i, row in enumerate(rows):
                ex, ey, ez = vec3(row)

                # Posúvanie progressu po riadkoch
                if v['row_progress'][i] < 0.75:
                    v['row_progress'][i] = min(v['row_progress'][i] + dt * segment_speed, 0.75)
                else:
                    v['row_progress'][i] = min(v['row_progress'][i] + dt * final_vector_speed, 1.0)

                segment_points = [to_pt((ex, 0, 0)), to_pt((ex, ey, 0)), to_pt((ex, ey, ez))]

                # Už hotové segmenty
                for s, e in v['row_segments'][i]:
                    radius = compute_radius(s, e, base_max_radius, min_radius, camera_pos, reference_distance)
                    draw_oval_line(s, e, radius=radius, color=v_color,alpha =0.5)

                # Aktuálny segment
                if v['row_progress'][i] < 0.75:
                    local_p = v['row_progress'][i] / 0.75
                    phase = local_p * 3.0
                    seg_index = int(min(phase, 2.999))
                    seg_progress = phase - seg_index

                    seg_start = to_pt((0, 0, 0)) if seg_index == 0 else segment_points[seg_index - 1]
                    seg_end = segment_points[seg_index]
                    px, py, pz = [s + (e - s) * seg_progress for s, e in zip(seg_start, seg_end)]

                    radius = compute_radius(seg_start, (px, py, pz), base_max_radius, min_radius, camera_pos, reference_distance)
                    draw_oval_line(seg_start, (px, py, pz), radius=radius, color=v_color,alpha =0.5)

                    # Ulož hotové pred aktuálnym
                    for j in range(seg_index):
                        s = to_pt((0, 0, 0)) if j == 0 else segment_points[j - 1]
                        e = segment_points[j]
                        if (s, e) not in v['row_segments'][i]:
                            v['row_segments'][i].append((s, e))

                # Finálna fáza
                else:
                    # ponechaj posledný Z helper (ak ez=0, stále ok – je to rovnaký bod)
                    z_seg = (segment_points[1], segment_points[2])
                    if z_seg not in v['row_segments'][i]:
                        v['row_segments'][i].append(z_seg)

                    t = (v['row_progress'][i] - 0.75) / 0.25
                    px, py, pz = ex * t, ey * t, ez * t

                    # pôvodný bod finálnej čiary
                    vec = np.array([ex * t, ey * t, ez * t])
                    length = np.linalg.norm(vec)
                    delta = 0.5  # o koľko skrátime

                    if length > delta:
                        vec_short = vec * (length - delta) / length
                    else:
                        vec_short = np.zeros(3)

                    px, py, pz = vec_short


                    radius = compute_radius((0, 0, 0), (px, py, pz), base_max_radius, min_radius, camera_pos, reference_distance)
                    draw_oval_line((0, 0, 0), (px, py, pz), radius=radius, color=v_color)

                    if v['row_progress'][i] >= 1.0:
                        draw_conehead_3d((0, 0, 0), (ex, ey, ez),
                                         color=(1.0, 0.4, 0.7),
                                         size=max(0.9, radius * 8),
                                         camera_pos=camera_pos,
                                         radius=0.3)

    glPopAttrib()
