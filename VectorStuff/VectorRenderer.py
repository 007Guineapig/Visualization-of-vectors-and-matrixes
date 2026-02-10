import math

import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *

class VectorRenderer:
    """Optimalizovaný renderer pre vektory a matice"""

    # Pre-computed tabuľky pre kruhy/cylindre
    _circle_cache = {}
    _quadric = None

    @staticmethod
    def draw_triangle_arrowhead_2d(tip_x, tip_y, angle, size, color, alpha=1.0, z=0.0):
        # size = 0.6
        """Nakreslí vyplnený trojuholníkový hrot šípky v 2D"""
        spread_angle = 0.30

        back_x1 = tip_x - size * math.cos(angle - spread_angle)
        back_y1 = tip_y - size * math.sin(angle - spread_angle)
        back_x2 = tip_x - size * math.cos(angle + spread_angle)
        back_y2 = tip_y - size * math.sin(angle + spread_angle)

        if alpha < 1.0:
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glColor4f(color[0], color[1], color[2], alpha)
        else:
            glColor3f(color[0], color[1], color[2])

        glBegin(GL_TRIANGLES)
        glVertex3f(tip_x, tip_y, z)
        glVertex3f(back_x1, back_y1, z)
        glVertex3f(back_x2, back_y2, z)
        glEnd()

        if alpha < 1.0:
            glDisable(GL_BLEND)

    @classmethod
    def _get_circle_points(cls, segments=16):
        """Vráti pre-computed body pre kruh"""
        if segments not in cls._circle_cache:
            points = []
            for i in range(segments + 1):
                angle = 2.0 * math.pi * i / segments
                points.append((math.cos(angle), math.sin(angle)))
            cls._circle_cache[segments] = points
        return cls._circle_cache[segments]

    @classmethod
    def _get_quadric(cls):
        """Vráti reusable quadric objekt"""
        if cls._quadric is None:
            cls._quadric = gluNewQuadric()
            gluQuadricNormals(cls._quadric, GLU_SMOOTH)
        return cls._quadric

    @staticmethod
    def compute_radius(start, end, base_max_radius, min_radius, camera_pos, reference_distance):
        """Vypočíta polomer pre oval line - OPTIMALIZOVANÉ"""
        if camera_pos is None or reference_distance is None or reference_distance <= 0:
            return max(float(base_max_radius), float(min_radius))

        # Použijem jednoduchší výpočet
        mid_x = (start[0] + end[0]) * 0.5
        mid_y = (start[1] + end[1]) * 0.5
        mid_z = (start[2] + end[2]) * 0.5

        dx = camera_pos[0] - mid_x
        dy = camera_pos[1] - mid_y
        dz = camera_pos[2] - mid_z

        cam_dist = math.sqrt(dx * dx + dy * dy + dz * dz)
        cam_dist = max(cam_dist, 1e-6)

        scale = cam_dist / reference_distance
        scale = max(0.5, min(2.0, scale))

        radius = float(base_max_radius) * scale
        return max(radius, float(min_radius))

    @staticmethod
    def draw_vectors_2d_animated(vectors, dt, line_width=6, arrow_fraction=0.15,
                                 speed=0.5, ortho_scale=6.0, color=(1, 1, 1)):
        """Animované kreslenie 2D vektorov s trojuholníkovými šípkami"""
        if not vectors:
            return

        glPushAttrib(GL_LINE_BIT | GL_CURRENT_BIT)
        glLineWidth(line_width)

        zoom_scale = ortho_scale / 6.0
        base_arrow_size = line_width * 0.08 * zoom_scale

        for v in vectors:
            vec = v['vec']
            v_color = v.get('color', color)
            v_alpha = v.get('alpha', 1.0)

            if isinstance(vec[0], (int, float)):
                progress = v.get('progress', 0.0)
                progress = min(progress + dt * speed, 1.0)
                v['progress'] = progress

                x, y = vec[0], vec[1] if len(vec) > 1 else 0
                px, py = x * progress, y * progress

                vec_len = math.hypot(px, py)
                if vec_len > 0.01:
                    angle = math.atan2(y, x)
                    arrow_size = min(base_arrow_size, vec_len * 0.25)

                    # Skráť čiaru
                    line_end_x = px - arrow_size * 0.7 * math.cos(angle)
                    line_end_y = py - arrow_size * 0.7 * math.sin(angle)

                    if v_alpha < 1.0:
                        glEnable(GL_BLEND)
                        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
                        glColor4f(v_color[0], v_color[1], v_color[2], v_alpha)
                    else:
                        glColor3f(*v_color)

                    glBegin(GL_LINES)
                    glVertex2f(0, 0)
                    glVertex2f(line_end_x, line_end_y)
                    glEnd()

                    # Trojuholníková šípka
                    VectorRenderer.draw_triangle_arrowhead_2d(
                        px, py, angle, arrow_size, v_color, v_alpha
                    )

                    if v_alpha < 1.0:
                        glDisable(GL_BLEND)

            else:
                n = len(vec)
                if 'row_progress' not in v:
                    v['row_progress'] = [0.0] * n

                for i, row in enumerate(vec):
                    x, y = row[0], row[1] if len(row) > 1 else 0

                    progress = v['row_progress'][i]
                    progress = min(progress + dt * speed, 1.0)
                    v['row_progress'][i] = progress

                    px, py = x * progress, y * progress

                    vec_len = math.hypot(px, py)
                    if vec_len > 0.01:
                        angle = math.atan2(y, x)
                        arrow_size = min(base_arrow_size, vec_len * 0.25)

                        line_end_x = px - arrow_size * 0.7 * math.cos(angle)
                        line_end_y = py - arrow_size * 0.7 * math.sin(angle)

                        if v_alpha < 1.0:
                            glEnable(GL_BLEND)
                            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
                            glColor4f(v_color[0], v_color[1], v_color[2], v_alpha)
                        else:
                            glColor3f(*v_color)

                        glBegin(GL_LINES)
                        glVertex2f(0, 0)
                        glVertex2f(line_end_x, line_end_y)
                        glEnd()

                        VectorRenderer.draw_triangle_arrowhead_2d(
                            px, py, angle, arrow_size, v_color, v_alpha
                        )

                        if v_alpha < 1.0:
                            glDisable(GL_BLEND)

        glPopAttrib()

    @staticmethod
    def draw_vectors_3d_animated(vectors, dt, line_width=6, base_arrow_size=0.15,
                                 segment_speed=0.3, final_vector_speed=0.2,
                                 camera_pos=None, color=(1, 1, 1)):
        """Animované kreslenie 3D vektorov - OPTIMALIZOVANÉ"""
        if not vectors:
            return

        # Sanitize dt
        if dt is None or not isinstance(dt, (int, float)):
            dt = 0.0
        dt = min(max(float(dt), 0.0), 1.0 / 30.0)

        glPushAttrib(GL_LINE_BIT | GL_CURRENT_BIT)
        glLineWidth(line_width)

        reference_distance = 2.0
        base_max_radius = 0.02
        min_radius = 0.005

        for v in vectors:
            if not isinstance(v, dict) or 'vec' not in v:
                continue

            v_color = v.get('color', color)
            vec = v['vec']

            # Skontroluj typ vektora
            if not isinstance(vec, (list, tuple)) or len(vec) == 0:
                continue

            is_single = isinstance(vec[0], (int, float))

            if is_single:
                # Single vector
                if 'progress' not in v:
                    v['progress'] = 0.0
                if 'segments' not in v:
                    v['segments'] = []

                # Update progress
                if v['progress'] < 0.75:
                    v['progress'] = min(v['progress'] + dt * segment_speed, 0.75)
                else:
                    v['progress'] = min(v['progress'] + dt * final_vector_speed, 1.0)

                ex = float(vec[0]) if len(vec) > 0 else 0.0
                ey = float(vec[1]) if len(vec) > 1 else 0.0
                ez = float(vec[2]) if len(vec) > 2 else 0.0

                # Segment points
                seg_pts = [(ex, 0, 0), (ex, ey, 0), (ex, ey, ez)]

                # Draw completed segments
                for s, e in v['segments']:
                    radius = VectorRenderer.compute_radius(s, e, base_max_radius, min_radius,
                                                           camera_pos, reference_distance)
                    VectorRenderer.draw_oval_line(s, e, radius=radius, color=v_color, alpha=0.5)

                # Current animation phase
                if v['progress'] < 0.75:
                    local_p = v['progress'] / 0.75
                    phase = local_p * 3.0
                    seg_idx = int(min(phase, 2.999))
                    seg_prog = phase - seg_idx

                    if seg_idx == 0:
                        seg_start = (0, 0, 0)
                    else:
                        seg_start = seg_pts[seg_idx - 1]
                    seg_end = seg_pts[seg_idx]

                    px = seg_start[0] + (seg_end[0] - seg_start[0]) * seg_prog
                    py = seg_start[1] + (seg_end[1] - seg_start[1]) * seg_prog
                    pz = seg_start[2] + (seg_end[2] - seg_start[2]) * seg_prog

                    radius = VectorRenderer.compute_radius(seg_start, (px, py, pz),
                                                           base_max_radius, min_radius,
                                                           camera_pos, reference_distance)
                    glDepthMask(GL_FALSE)
                    VectorRenderer.draw_oval_line(seg_start, (px, py, pz),
                                                  radius=radius, color=v_color, alpha=0.5)
                    glDepthMask(GL_TRUE)

                    # Add completed segments
                    for i in range(seg_idx):
                        s = (0, 0, 0) if i == 0 else seg_pts[i - 1]
                        e = seg_pts[i]
                        if (s, e) not in v['segments']:
                            v['segments'].append((s, e))

                else:
                    # Final phase - draw main vector
                    z_seg = (seg_pts[1], seg_pts[2])
                    if z_seg not in v['segments']:
                        v['segments'].append(z_seg)

                    t = (v['progress'] - 0.75) / 0.25
                    vec_full = np.array([ex * t, ey * t, ez * t], dtype=np.float32)
                    length = np.linalg.norm(vec_full)
                    delta = 0.15

                    if length > delta:
                        vec_short = vec_full * (length - delta) / length
                    else:
                        vec_short = vec_full

                    radius = VectorRenderer.compute_radius((0, 0, 0), tuple(vec_short),
                                                           base_max_radius, min_radius,
                                                           camera_pos, reference_distance)
                    VectorRenderer.draw_oval_line((0, 0, 0), tuple(vec_short),
                                                  radius=radius, color=v_color)

                    if v['progress'] >= 1.0:
                        VectorRenderer.draw_conehead_3d((0, 0, 0), (ex, ey, ez),
                                                        color=(1.0, 0.4, 0.7),
                                                        size=max(0.9, radius * 8),
                                                        camera_pos=camera_pos,
                                                        radius=0.3)

            else:
                # Matrix of vectors
                n = len(vec)
                if 'row_progress' not in v or len(v['row_progress']) != n:
                    v['row_progress'] = [0.0] * n
                if 'row_segments' not in v or len(v['row_segments']) != n:
                    v['row_segments'] = [[] for _ in range(n)]

                for i, row in enumerate(vec):
                    ex = float(row[0]) if len(row) > 0 else 0.0
                    ey = float(row[1]) if len(row) > 1 else 0.0
                    ez = float(row[2]) if len(row) > 2 else 0.0

                    # Update progress
                    if v['row_progress'][i] < 0.75:
                        v['row_progress'][i] = min(v['row_progress'][i] + dt * segment_speed, 0.75)
                    else:
                        v['row_progress'][i] = min(v['row_progress'][i] + dt * final_vector_speed, 1.0)

                    seg_pts = [(ex, 0, 0), (ex, ey, 0), (ex, ey, ez)]

                    # Draw completed segments
                    for s, e in v['row_segments'][i]:
                        radius = VectorRenderer.compute_radius(s, e, base_max_radius, min_radius,
                                                               camera_pos, reference_distance)
                        VectorRenderer.draw_oval_line(s, e, radius=radius, color=v_color, alpha=0.5)

                    if v['row_progress'][i] < 0.75:
                        local_p = v['row_progress'][i] / 0.75
                        phase = local_p * 3.0
                        seg_idx = int(min(phase, 2.999))
                        seg_prog = phase - seg_idx

                        if seg_idx == 0:
                            seg_start = (0, 0, 0)
                        else:
                            seg_start = seg_pts[seg_idx - 1]
                        seg_end = seg_pts[seg_idx]

                        px = seg_start[0] + (seg_end[0] - seg_start[0]) * seg_prog
                        py = seg_start[1] + (seg_end[1] - seg_start[1]) * seg_prog
                        pz = seg_start[2] + (seg_end[2] - seg_start[2]) * seg_prog

                        radius = VectorRenderer.compute_radius(seg_start, (px, py, pz),
                                                               base_max_radius, min_radius,
                                                               camera_pos, reference_distance)
                        VectorRenderer.draw_oval_line(seg_start, (px, py, pz),
                                                      radius=radius, color=v_color, alpha=0.5)

                        for j in range(seg_idx):
                            s = (0, 0, 0) if j == 0 else seg_pts[j - 1]
                            e = seg_pts[j]
                            if (s, e) not in v['row_segments'][i]:
                                v['row_segments'][i].append((s, e))

                    else:
                        z_seg = (seg_pts[1], seg_pts[2])
                        if z_seg not in v['row_segments'][i]:
                            v['row_segments'][i].append(z_seg)

                        t = (v['row_progress'][i] - 0.75) / 0.25
                        vec_full = np.array([ex * t, ey * t, ez * t], dtype=np.float32)
                        length = np.linalg.norm(vec_full)
                        delta = 0.15

                        if length > delta:
                            vec_short = vec_full * (length - delta) / length
                        else:
                            vec_short = vec_full

                        radius = VectorRenderer.compute_radius((0, 0, 0), tuple(vec_short),
                                                               base_max_radius, min_radius,
                                                               camera_pos, reference_distance)
                        VectorRenderer.draw_oval_line((0, 0, 0), tuple(vec_short),
                                                      radius=radius, color=v_color)

                        if v['row_progress'][i] >= 1.0:
                            VectorRenderer.draw_conehead_3d((0, 0, 0), (ex, ey, ez),
                                                            color=(1.0, 0.4, 0.7),
                                                            size=max(0.9, radius * 8),
                                                            camera_pos=camera_pos,
                                                            radius=0.3)

        glPopAttrib()

    @staticmethod
    def draw_oval_line(start, end, radius=0.005, segments=8, color=(1, 1, 0), alpha=1.0):
        """Nakreslí 3D čiaru ako oval/tubu - OPTIMALIZOVANÉ"""
        # Konverzia na numpy
        start = np.array(start, dtype=np.float32)
        end = np.array(end, dtype=np.float32)
        dir_vec = end - start
        length = np.linalg.norm(dir_vec)

        if length < 1e-6:
            return

        dir_vec /= length

        # Výpočet kolmých vektorov
        if abs(dir_vec[0]) < 0.001 and abs(dir_vec[1]) < 0.001:
            up = np.array([0, 1, 0], dtype=np.float32)
        else:
            up = np.array([0, 0, 1], dtype=np.float32)

        side = np.cross(dir_vec, up)
        side_len = np.linalg.norm(side)
        if side_len < 1e-6:
            return
        side /= side_len

        up_vec = np.cross(side, dir_vec)

        # Kreslenie
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glColor4f(color[0], color[1], color[2], alpha)

        # Použijem cached circle points
        circle_pts = VectorRenderer._get_circle_points(segments)

        glBegin(GL_TRIANGLE_STRIP)
        for cos_t, sin_t in circle_pts:
            offset = radius * cos_t * side + radius * sin_t * up_vec
            s_pt = start + offset
            e_pt = end + offset
            glVertex3f(s_pt[0], s_pt[1], s_pt[2])
            glVertex3f(e_pt[0], e_pt[1], e_pt[2])
        glEnd()

        glDisable(GL_BLEND)

    @staticmethod
    def draw_conehead_3d(start, end, color=(1.0, 0.0, 1.0), size=None,
                         camera_pos=None, radius=0.3, alpha=1.0):
        """Nakreslí kužeľovú šípku na konci vektora - OPTIMALIZOVANÉ"""
        start = np.array(start, dtype=np.float32)
        end = np.array(end, dtype=np.float32)

        dir_vec = end - start
        length = np.linalg.norm(dir_vec)

        if length < 1e-6:
            return

        # Výpočet size vektora
        if size is None:
            size_vec = 0.2 * dir_vec
        elif np.isscalar(size):
            size_vec = dir_vec / length * size
        else:
            size_vec = np.array(size, dtype=np.float32)

        size_len = np.linalg.norm(size_vec)
        if size_len < 1e-6:
            return

        cone_base_center = end - size_vec
        dir_norm = size_vec / size_len

        # Nastavenie farby a blendingu
        if alpha < 1.0:
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glColor4f(color[0], color[1], color[2], alpha)
        else:
            glColor3f(*color)

        glPushMatrix()
        glTranslatef(cone_base_center[0], cone_base_center[1], cone_base_center[2])

        # Rotácia kužeľa
        up = np.array([0, 0, 1], dtype=np.float32)
        axis = np.cross(up, dir_norm)
        axis_len = np.linalg.norm(axis)

        if axis_len > 1e-6:
            axis /= axis_len
            dot = np.clip(np.dot(up, dir_norm), -1.0, 1.0)
            angle = math.degrees(np.arccos(dot))
            glRotatef(angle, axis[0], axis[1], axis[2])
        elif dir_norm[2] < 0:
            glRotatef(180, 1, 0, 0)

        # Kreslenie kužeľa
        quad = VectorRenderer._get_quadric()
        gluCylinder(quad, radius, 0.0, size_len, 12, 2)

        glPopMatrix()

        if alpha < 1.0:
            glDisable(GL_BLEND)

    @staticmethod
    def draw_sphere(position, radius=0.1, slices=8, stacks=6, color=(1, 0, 1), alpha=1.0):
        """Nakreslí guľu na pozícii - OPTIMALIZOVANÉ (menej polygónov)"""
        glPushMatrix()
        glTranslatef(position[0], position[1], position[2])

        if alpha < 1.0:
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glColor4f(color[0], color[1], color[2], alpha)
        else:
            glColor3f(color[0], color[1], color[2])

        quad = VectorRenderer._get_quadric()
        gluSphere(quad, radius, slices, stacks)

        if alpha < 1.0:
            glDisable(GL_BLEND)

        glPopMatrix()

    @staticmethod
    def draw_circle_2d(position, radius=0.1, segments=16, color=(1, 1, 0), alpha=1.0):
        """Nakreslí kruh v 2D - OPTIMALIZOVANÉ s cached points"""
        circle_pts = VectorRenderer._get_circle_points(segments)

        if alpha < 1.0:
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glColor4f(color[0], color[1], color[2], alpha)
        else:
            glColor3f(color[0], color[1], color[2])

        px, py = position[0], position[1]
        pz = position[2] if len(position) > 2 else 0

        glBegin(GL_TRIANGLE_FAN)
        glVertex3f(px, py, pz)
        for cos_a, sin_a in circle_pts:
            glVertex3f(px + cos_a * radius, py + sin_a * radius, pz)
        glEnd()

        if alpha < 1.0:
            glDisable(GL_BLEND)

    @classmethod
    def cleanup(cls):
        """Uvoľní resources - volať pri ukončení aplikácie"""
        if cls._quadric is not None:
            gluDeleteQuadric(cls._quadric)
            cls._quadric = None
        cls._circle_cache.clear()