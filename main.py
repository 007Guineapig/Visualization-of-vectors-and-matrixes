
from numpy import array, cross, linalg
from math import sin, cos, pi



class VectorUtils:
    """Utility functions for vector operations"""

    @staticmethod
    def format_number(x):
        """Format number nicely — remove .0 when not needed"""
        if float(x).is_integer():
            return str(int(x))
        else:
            return f"{x:.2f}".rstrip('0').rstrip('.')

    @staticmethod
    def format_vector(vec):
        """Format vectors and matrices cleanly for display"""
        if isinstance(vec[0], (int, float)):
            return "(" + ", ".join(VectorUtils.format_number(x) for x in vec) + ")"
        else:
            return "(" + ", ".join(
                "(" + ", ".join(VectorUtils.format_number(x) for x in row) + ")" for row in vec) + ")"

    @staticmethod
    def snap_number(x, precision=6):
        """Round to nearest number; convert to int if whole, else keep float."""
        x = round(x, precision)
        if x.is_integer():
            return int(x)
        return x

    @staticmethod
    def max_from_vectors(vectors):
        """Return max value from all vectors/matrices"""
        max_val = None
        for v in vectors:
            vec = v['vec'] if isinstance(v, dict) and 'vec' in v else v
            if isinstance(vec[0], (int, float)):
                current_max = max(vec)
            else:
                current_max = max(max(row) for row in vec)
            if max_val is None or current_max > max_val:
                max_val = current_max
        return max_val


class InputHandler:
    """Handles user input for vector/matrix entry"""

    def __init__(self):
        self.input_text = ""
        self.show_input_active = False
        self.multiplication_input = ""
        self.show_multiplication_active = False
        self.matrix_size_input = ""
        self.random_range_input = ""
        self.show_matrix_size_active = False
        self.show_random_range_active = False

    def handle_text_input(self, event, active_field='input'):
        """Handle keyboard input for text fields"""
        if active_field == 'input' and self.show_input_active:
            if event.key == K_BACKSPACE:
                self.input_text = self.input_text[:-1]
            else:
                self.input_text += event.unicode

        elif active_field == 'multiplication' and self.show_multiplication_active:
            if event.key == K_BACKSPACE:
                self.multiplication_input = self.multiplication_input[:-1]
            else:
                self.multiplication_input += event.unicode

        elif active_field == 'matrix_size' and self.show_matrix_size_active:
            if event.key == K_BACKSPACE:
                self.matrix_size_input = self.matrix_size_input[:-1]
            elif event.unicode.isdigit() or event.unicode == ',':
                self.matrix_size_input += event.unicode

        elif active_field == 'random_range' and self.show_random_range_active:
            if event.key == K_BACKSPACE:
                self.random_range_input = self.random_range_input[:-1]
            elif event.unicode.isdigit() or event.unicode in ",.-":
                self.random_range_input += event.unicode

    def parse_input(self):
        """Parse and return vector from input_text"""
        try:
            val = ast.literal_eval(self.input_text)
            if isinstance(val, (list, tuple)):
                if all(isinstance(row, (list, tuple)) and len(row) in (2, 3) for row in val):
                    return val  # Matrix
                elif len(val) in (2, 3) and all(isinstance(c, (int, float)) for c in val):
                    return val  # Single vector
            return None
        except:
            return None

    def parse_multiplication(self):
        """Parse multiplication input"""
        try:
            return ast.literal_eval(self.multiplication_input)
        except:
            return None




class UIRenderer:
    """Renderer pre UI elementy"""

    @staticmethod
    def draw_text_2d(text, position, color=(1, 1, 1), font_size=20):
        """Nakreslí text v 2D"""
        font = pygame.font.SysFont("Arial", font_size, True)
        text_surface = font.render(text, True,
                                   (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)))
        text_data = pygame.image.tostring(text_surface, "RGBA", True)
        w, h = text_surface.get_size()

        tex_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tex_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, text_data)

        glEnable(GL_TEXTURE_2D)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glDisable(GL_DEPTH_TEST)

        x, y = position
        glColor3f(1, 1, 1)
        glBindTexture(GL_TEXTURE_2D, tex_id)
        glBegin(GL_QUADS)
        glTexCoord2f(0, 0)
        glVertex2f(x, y + h)
        glTexCoord2f(1, 0)
        glVertex2f(x + w, y + h)
        glTexCoord2f(1, 1)
        glVertex2f(x + w, y)
        glTexCoord2f(0, 1)
        glVertex2f(x, y)
        glEnd()

        glEnable(GL_DEPTH_TEST)
        glDisable(GL_TEXTURE_2D)
        glDisable(GL_BLEND)
        glDeleteTextures([tex_id])

    @staticmethod
    def draw_button_2d(x, y, w, h, label, active=False):
        """Nakreslí tlačidlo"""

        WIDTH,HEIGHT = pygame.display.get_window_size()


        glDisable(GL_DEPTH_TEST)
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, WIDTH, HEIGHT, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        # Rectangle
        glColor3f(0.5, 0.9, 0.5) if active else glColor3f(0.8, 0.8, 0.8)
        glBegin(GL_QUADS)
        glVertex2f(x, y)
        glVertex2f(x + w, y)
        glVertex2f(x + w, y + h)
        glVertex2f(x, y + h)
        glEnd()

        # Border
        glColor3f(0, 0, 0)
        glBegin(GL_LINE_LOOP)
        glVertex2f(x, y)
        glVertex2f(x + w, y)
        glVertex2f(x + w, y + h)
        glVertex2f(x, y + h)
        glEnd()

        # Text
        font = pygame.font.SysFont("Arial", 18, True)
        text_surface = font.render(label, True, (0, 0, 0))
        tw, th = text_surface.get_size()
        text_data = pygame.image.tostring(text_surface, "RGBA", False)

        tex_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tex_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, tw, th, 0, GL_RGBA, GL_UNSIGNED_BYTE, text_data)

        glEnable(GL_TEXTURE_2D)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        tx = x + (w - tw) / 2
        ty = y + (h - th) / 2

        glBindTexture(GL_TEXTURE_2D, tex_id)
        glBegin(GL_QUADS)
        glTexCoord2f(0, 0)
        glVertex2f(tx, ty)
        glTexCoord2f(1, 0)
        glVertex2f(tx + tw, ty)
        glTexCoord2f(1, 1)
        glVertex2f(tx + tw, ty + th)
        glTexCoord2f(0, 1)
        glVertex2f(tx, ty + th)
        glEnd()

        glDisable(GL_TEXTURE_2D)
        glDisable(GL_BLEND)
        glDeleteTextures([tex_id])

        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glEnable(GL_DEPTH_TEST)

    @staticmethod
    def draw_input_box_3d(x, y, w, h, text, active=False,
                          fill_color=(0.7, 0.7, 0.7),
                          fill_color_outline=(0, 0, 0)):
        """Nakreslí input box"""
        WIDTH, HEIGHT = pygame.display.get_window_size()

        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, WIDTH, HEIGHT, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        glDisable(GL_DEPTH_TEST)

        r, g, b = fill_color
        if active:
            r, g, b = min(r + 0.2, 1.0), min(g + 0.2, 1.0), min(b + 0.2, 1.0)
        glColor3f(r, g, b)
        glBegin(GL_QUADS)
        glVertex2f(x, y)
        glVertex2f(x + w, y)
        glVertex2f(x + w, y + h)
        glVertex2f(x, y + h)
        glEnd()

        r, g, b = fill_color_outline
        glLineWidth(4.0)
        glColor3f(r, g, b)
        glBegin(GL_LINE_LOOP)
        glVertex2f(x, y)
        glVertex2f(x + w, y)
        glVertex2f(x + w, y + h)
        glVertex2f(x, y + h)
        glEnd()

        # Text
        font = pygame.font.SysFont("Arial", 18, True)
        text_surface = font.render(text, True, (0, 0, 0))
        text_data = pygame.image.tostring(text_surface, "RGBA", True)
        tw, th = text_surface.get_size()

        tex_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tex_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, tw, th, 0, GL_RGBA, GL_UNSIGNED_BYTE, text_data)

        glEnable(GL_TEXTURE_2D)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        glBindTexture(GL_TEXTURE_2D, tex_id)
        glBegin(GL_QUADS)
        glTexCoord2f(0, 0)
        glVertex2f(x + 5, y + 5 + th)
        glTexCoord2f(1, 0)
        glVertex2f(x + 5 + tw, y + 5 + th)
        glTexCoord2f(1, 1)
        glVertex2f(x + 5 + tw, y + 5)
        glTexCoord2f(0, 1)
        glVertex2f(x + 5, y + 5)
        glEnd()

        glDisable(GL_TEXTURE_2D)
        glDisable(GL_BLEND)
        glDeleteTextures([tex_id])

        glEnable(GL_DEPTH_TEST)

        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)


class EnhancedUIRenderer(UIRenderer):
    """Extended UI renderer with additional components"""

    @staticmethod
    def draw_dropdown(vectors, selected_index, rect, open=False, width=None, height=None):
        """Draw dropdown for vector selection"""
        if width is None:
            width = pygame.display.get_surface().get_width()
        if height is None:
            height = pygame.display.get_surface().get_height()

        width, height = pygame.display.get_window_size()

        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, width, height, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        # Main rectangle
        glColor3f(0.8, 0.8, 0.8)
        glBegin(GL_QUADS)
        glVertex2f(rect.x, rect.y)
        glVertex2f(rect.x + rect.width, rect.y)
        glVertex2f(rect.x + rect.width, rect.y + rect.height)
        glVertex2f(rect.x, rect.y + rect.height)
        glEnd()

        glColor3f(0, 0, 0)
        glBegin(GL_LINE_LOOP)
        glVertex2f(rect.x, rect.y)
        glVertex2f(rect.x + rect.width, rect.y)
        glVertex2f(rect.x + rect.width, rect.y + rect.height)
        glVertex2f(rect.x, rect.y + rect.height)
        glEnd()

        # Selected text
        if selected_index == -1:
            label = "Select Vector"
        else:
            label = VectorUtils.format_vector(vectors[selected_index]['vec'])
        UIRenderer.draw_text_2d(label, (rect.x + 5, rect.y + 5), color=(0, 0, 0), font_size=18)

        # Dropdown items
        if open:
            dropdown_item_height = 25
            for i, v in enumerate(vectors):
                y = rect.y + rect.height + i * dropdown_item_height
                glColor3f(0.9, 0.9, 0.9) if i != selected_index else glColor3f(0.7, 0.7, 1)
                glBegin(GL_QUADS)
                glVertex2f(rect.x, y)
                glVertex2f(rect.x + rect.width, y)
                glVertex2f(rect.x + rect.width, y + dropdown_item_height)
                glVertex2f(rect.x, y + dropdown_item_height)
                glEnd()

                glColor3f(0, 0, 0)
                glBegin(GL_LINE_LOOP)
                glVertex2f(rect.x, y)
                glVertex2f(rect.x + rect.width, y)
                glVertex2f(rect.x + rect.width, y + dropdown_item_height)
                glVertex2f(rect.x, y + dropdown_item_height)
                glEnd()

                UIRenderer.draw_text_2d(VectorUtils.format_vector(v['vec']),
                                        (rect.x + 5, y + 5), color=(0, 0, 0), font_size=16)

        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)

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
# ============= VECTOR MANAGER =============


class VectorManager:
    """Manages vector operations and animations"""

    def __init__(self):
        self.animated_vectors = []
        self.pending_vector = None
        self.selected_vector_index = -1
        self.animation_controller = AnimationController()

    def add_vector(self, vec, progress=0.0):
        """Add a new vector"""
        self.animated_vectors.append({'vec': vec, 'progress': progress})

    def reset_animations(self):
        """Reset all vector animations"""
        for v in self.animated_vectors:
            v['progress'] = 0.0
            if 'segments' in v:
                v['segments'].clear()
            if 'row_progress' in v:
                v['row_progress'] = [0.0] * len(v['row_progress'])
            if 'row_segments' in v:
                for lst in v['row_segments']:
                    lst.clear()

    def multiply_selected_vector(self, multiplier):
        """Multiply selected vector by a value (scalar or matrix)"""
        if self.selected_vector_index == -1:
            return False

        selected = self.animated_vectors[self.selected_vector_index]
        selected_np = np.array(selected['vec'])
        val_np = np.array(multiplier)

        # Determine multiplication type
        if selected_np.ndim == 1 and val_np.ndim == 1:
            result = np.dot(selected_np, val_np)
        else:
            result = selected_np @ val_np

        # Animate to new position
        frames = 30
        if isinstance(result, np.ndarray):
            if result.ndim == 1:
                target = result.tolist()
                current = selected['vec']
                selected['delta'] = [(t - c) / frames for c, t in zip(current, target)]
                selected['frames_left'] = frames
            else:
                target_vectors = result.tolist()
                current_vectors = selected['vec']
                selected['delta'] = [
                    [(t - c) / frames for c, t in zip(curr_vec, targ_vec)]
                    for curr_vec, targ_vec in zip(current_vectors, target_vectors)
                ]
                selected['frames_left'] = frames
        return True

    def update_animations(self, dt):
        """Update frame-based animations"""
        # Update animation controller
        self.animation_controller.update(dt)

        # Pôvodná logika pre iné animácie
        for v in self.animated_vectors:
            if 'frames_left' in v and v['frames_left'] > 0:
                if isinstance(v['vec'][0], (list, tuple)):  # matrix
                    v['vec'] = [
                        [c + d for c, d in zip(list(curr), delt)]
                        for curr, delt in zip(v['vec'], v['delta'])
                    ]
                else:  # single vector
                    v['vec'] = [c + d for c, d in zip(list(v['vec']), v['delta'])]

                v['frames_left'] -= 1

                # Snap to clean numbers when finished
                if v['frames_left'] == 0:
                    if isinstance(v['vec'][0], (list, tuple)):
                        v['vec'] = [[VectorUtils.snap_number(c) for c in row] for row in v['vec']]
                    else:
                        v['vec'] = [VectorUtils.snap_number(c) for c in v['vec']]

class VectorRenderer:
    """Renderer pre vektory a matice"""

    @staticmethod
    def compute_radius(start, end, base_max_radius, min_radius, camera_pos, reference_distance):
        """Vypočíta polomer pre oval line"""
        radius = float(base_max_radius)

        if camera_pos and reference_distance and reference_distance > 0:
            mid = ((start[0] + end[0]) * 0.5,
                   (start[1] + end[1]) * 0.5,
                   (start[2] + end[2]) * 0.5)
            cam_dist = math.sqrt((camera_pos[0] - mid[0]) ** 2 +
                                 (camera_pos[1] - mid[1]) ** 2 +
                                 (camera_pos[2] - mid[2]) ** 2)
            cam_dist = max(cam_dist, 1e-6)
            scale = cam_dist / reference_distance
            scale = max(0.5, min(2.0, scale))
            radius *= scale

        return max(radius, float(min_radius))

    @staticmethod
    def draw_vectors_2d_animated(vectors, dt, line_width=6, arrow_fraction=0.15,
                                 speed=0.5, ortho_scale=6.0, color=(1, 1, 1)):
        """Animované kreslenie 2D vektorov"""
        glPushAttrib(GL_LINE_BIT | GL_CURRENT_BIT)
        glLineWidth(line_width)

        zoom_scale = ortho_scale / 6.0

        for v in vectors:
            vec = v['vec']

            # Single vector
            if isinstance(vec[0], (int, float)):
                v['progress'] = min(v.get('progress', 0.0) + dt * speed, 1.0)
                x, y = vec[0], vec[1]
                px, py = x * v['progress'], y * v['progress']

                glColor3f(*color)
                glBegin(GL_LINES)
                glVertex2f(0, 0)
                glVertex2f(px, py)
                glEnd()

                # Arrowhead
                angle = math.atan2(y, x)
                vec_len = math.hypot(x, y) * v['progress']
                base_arrow_size = line_width * arrow_fraction * zoom_scale
                arrow_size = min(base_arrow_size, vec_len / 5.0)
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

            # Matrix of vectors
            else:
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

    @staticmethod
    def draw_vectors_3d_animated(vectors, dt, line_width=6, base_arrow_size=0.15,
                                 segment_speed=0.3, final_vector_speed=0.2,
                                 camera_pos=None, color=(1, 1, 1)):
        """Animované kreslenie 3D vektorov s postupným X→Y→Z"""
        # OPRAVA: Lepšie ošetrenie dt parametra
        if dt is None or not isinstance(dt, (int, float)):
            dt = 0.0
        else:
            dt = float(dt)

        max_dt = 1.0 / 30.0
        dt = min(max(dt, 0.0), max_dt)

        # OPRAVA: Kontrola či vectors existuje a nie je None
        if vectors is None or not isinstance(vectors, list):
            return

        glPushAttrib(GL_LINE_BIT | GL_CURRENT_BIT)
        glLineWidth(line_width)

        reference_distance = 2.0
        base_max_radius = 0.02
        min_radius = 0.005

        def to_pt(p):
            try:
                x = float(p[0]) if len(p) > 0 else 0.0
                y = float(p[1]) if len(p) > 1 else 0.0
                z = float(p[2]) if len(p) > 2 else 0.0
            except:
                x, y, z = 0.0, 0.0, 0.0
            return (x, y, z)

        def vec3(v):
            try:
                x = float(v[0]) if len(v) > 0 else 0.0
                y = float(v[1]) if len(v) > 1 else 0.0
                z = float(v[2]) if len(v) > 2 else 0.0
            except:
                try:
                    x = float(v)
                except:
                    x = 0.0
                y = 0.0
                z = 0.0
            return x, y, z

        def is_number(x):
            try:
                float(x)
                return True
            except:
                return False

        for v in (vectors or []):
            v_color = v.get('color', color)
            vec_field = v.get('vec', None)
            if vec_field is None:
                continue

            is_single_vector = False
            if isinstance(vec_field, (list, tuple)):
                if len(vec_field) > 0 and is_number(vec_field[0]):
                    is_single_vector = True
            else:
                continue

            # Single vector
            if is_single_vector:
                if 'progress' not in v or not isinstance(v['progress'], (int, float)):
                    v['progress'] = 0.0
                if 'segments' not in v or not isinstance(v['segments'], list):
                    v['segments'] = []

                if v['progress'] < 0.75:
                    v['progress'] = min(v['progress'] + dt * segment_speed, 0.75)
                else:
                    v['progress'] = min(v['progress'] + dt * final_vector_speed, 1.0)

                ex, ey, ez = vec3(vec_field)
                segment_points = [to_pt((ex, 0, 0)), to_pt((ex, ey, 0)), to_pt((ex, ey, ez))]

                # Draw completed segments
                for s, e in v['segments']:
                    radius = VectorRenderer.compute_radius(s, e, base_max_radius, min_radius,
                                                           camera_pos, reference_distance)
                    VectorRenderer.draw_oval_line(s, e, radius=radius, color=v_color, alpha=0.5)

                # Current segment
                if v['progress'] < 0.75:
                    local_p = v['progress'] / 0.75
                    phase = local_p * 3.0
                    seg_index = int(min(phase, 2.999))
                    seg_progress = phase - seg_index

                    seg_start = to_pt((0, 0, 0)) if seg_index == 0 else segment_points[seg_index - 1]
                    seg_end = segment_points[seg_index]
                    px, py, pz = [s + (e - s) * seg_progress for s, e in zip(seg_start, seg_end)]

                    radius = VectorRenderer.compute_radius(seg_start, (px, py, pz), base_max_radius,
                                                           min_radius, camera_pos, reference_distance)
                    glDepthMask(GL_FALSE)
                    VectorRenderer.draw_oval_line(seg_start, (px, py, pz), radius=radius,
                                                  color=v_color, alpha=0.5)
                    glDepthMask(GL_TRUE)

                    for i in range(seg_index):
                        s = to_pt((0, 0, 0)) if i == 0 else segment_points[i - 1]
                        e = segment_points[i]
                        if (s, e) not in v['segments']:
                            v['segments'].append((s, e))

                # Final phase
                else:
                    z_seg = (segment_points[1], segment_points[2])
                    if z_seg not in v['segments']:
                        v['segments'].append(z_seg)

                    t = (v['progress'] - 0.75) / 0.25
                    # Skráť vektor aby čiara nezasahovala do šípky
                    vec = np.array([ex * t, ey * t, ez * t])
                    length = np.linalg.norm(vec)
                    delta = 0.15

                    if length > delta:
                        vec_short = vec * (length - delta) / length
                    else:
                        vec_short = vec

                    px, py, pz = vec_short

                    radius = VectorRenderer.compute_radius((0, 0, 0), (px, py, pz), base_max_radius,
                                                           min_radius, camera_pos, reference_distance)
                    VectorRenderer.draw_oval_line((0, 0, 0), (px, py, pz), radius=radius, color=v_color)

                    if v['progress'] >= 1.0:
                        VectorRenderer.draw_conehead_3d((0, 0, 0), (ex, ey, ez),
                                                        color=(1.0, 0.4, 0.7),
                                                        size=max(0.9, radius * 8),
                                                        camera_pos=camera_pos,
                                                        radius=0.3)

            # Matrix of vectors
            else:
                rows = vec_field if isinstance(vec_field, (list, tuple)) else []
                n = len(rows)

                if 'row_progress' not in v or not isinstance(v['row_progress'], list) or len(v['row_progress']) != n:
                    v['row_progress'] = [0.0] * n
                if 'row_segments' not in v or not isinstance(v['row_segments'], list) or len(v['row_segments']) != n:
                    v['row_segments'] = [[] for _ in range(n)]

                for i, row in enumerate(rows):
                    ex, ey, ez = vec3(row)

                    if v['row_progress'][i] < 0.75:
                        v['row_progress'][i] = min(v['row_progress'][i] + dt * segment_speed, 0.75)
                    else:
                        v['row_progress'][i] = min(v['row_progress'][i] + dt * final_vector_speed, 1.0)

                    segment_points = [to_pt((ex, 0, 0)), to_pt((ex, ey, 0)), to_pt((ex, ey, ez))]

                    for s, e in v['row_segments'][i]:
                        radius = VectorRenderer.compute_radius(s, e, base_max_radius, min_radius,
                                                               camera_pos, reference_distance)
                        VectorRenderer.draw_oval_line(s, e, radius=radius, color=v_color, alpha=0.5)

                    if v['row_progress'][i] < 0.75:
                        local_p = v['row_progress'][i] / 0.75
                        phase = local_p * 3.0
                        seg_index = int(min(phase, 2.999))
                        seg_progress = phase - seg_index

                        seg_start = to_pt((0, 0, 0)) if seg_index == 0 else segment_points[seg_index - 1]
                        seg_end = segment_points[seg_index]
                        px, py, pz = [s + (e - s) * seg_progress for s, e in zip(seg_start, seg_end)]

                        radius = VectorRenderer.compute_radius(seg_start, (px, py, pz), base_max_radius,
                                                               min_radius, camera_pos, reference_distance)
                        VectorRenderer.draw_oval_line(seg_start, (px, py, pz), radius=radius,
                                                      color=v_color, alpha=0.5)

                        for j in range(seg_index):
                            s = to_pt((0, 0, 0)) if j == 0 else segment_points[j - 1]
                            e = segment_points[j]
                            if (s, e) not in v['row_segments'][i]:
                                v['row_segments'][i].append((s, e))

                    else:
                        z_seg = (segment_points[1], segment_points[2])
                        if z_seg not in v['row_segments'][i]:
                            v['row_segments'][i].append(z_seg)

                        t = (v['row_progress'][i] - 0.75) / 0.25
                        vec = np.array([ex * t, ey * t, ez * t])
                        length = np.linalg.norm(vec)
                        delta = 0.15

                        if length > delta:
                            vec_short = vec * (length - delta) / length
                        else:
                            vec_short = np.zeros(3)

                        px, py, pz = vec_short

                        radius = VectorRenderer.compute_radius((0, 0, 0), (px, py, pz), base_max_radius,
                                                               min_radius, camera_pos, reference_distance)
                        VectorRenderer.draw_oval_line((0, 0, 0), (px, py, pz), radius=radius, color=v_color)

                        if v['row_progress'][i] >= 1.0:
                            VectorRenderer.draw_conehead_3d((0, 0, 0), (ex, ey, ez),
                                                            color=(1.0, 0.4, 0.7),
                                                            size=max(0.9, radius * 8),
                                                            camera_pos=camera_pos,
                                                            radius=0.3)

        glPopAttrib()

    @staticmethod
    def draw_oval_line(start, end, radius=0.005, segments=12, color=(1, 1, 0), alpha=1.0):
        """Nakreslí 3D čiaru ako oval/tubu"""
        start = array(start, dtype=float)
        end = array(end, dtype=float)
        dir_vec = end - start
        length = linalg.norm(dir_vec)
        if length == 0:
            return

        dir_vec /= length

        if abs(dir_vec[0]) < 0.0001 and abs(dir_vec[1]) < 0.0001:
            up = array([0, 1, 0])
        else:
            up = array([0, 0, 1])
        side = cross(dir_vec, up)
        side /= linalg.norm(side)
        up_vec = cross(side, dir_vec)
        up_vec /= linalg.norm(up_vec)

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glColor4f(color[0], color[1], color[2], alpha)

        glBegin(GL_TRIANGLE_STRIP)
        for i in range(segments + 1):
            theta = 2 * pi * i / segments
            offset = radius * cos(theta) * side + radius * sin(theta) * up_vec
            glVertex3fv(start + offset)
            glVertex3fv(end + offset)
        glEnd()

    @staticmethod
    @staticmethod
    def draw_conehead_3d(start, end, color=(1.0, 0.0, 1.0), size=None, camera_pos=None, radius=0.3, alpha=1.0):
        """Nakreslí kužeľovú šípku na konci vektora s podporou pre alpha"""
        start = np.array(start, dtype=float)
        end = np.array(end, dtype=float)

        if size is None:
            dir_vec = end - start
            length = np.linalg.norm(dir_vec)
            if length == 0:
                return
            size_vec = 0.2 * dir_vec
        elif np.isscalar(size):
            dir_vec = end - start
            length = np.linalg.norm(dir_vec)
            if length == 0:
                return
            size_vec = dir_vec / length * size
        else:
            size_vec = np.array(size, dtype=float)

        cone_base_center = end - size_vec
        dir_vec = size_vec / np.linalg.norm(size_vec)

        # Ak je alpha < 1.0, zapni blending
        if alpha < 1.0:
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glColor4f(color[0], color[1], color[2], alpha)
        else:
            glColor3f(*color)

        glPushMatrix()
        glTranslatef(*cone_base_center)

        up = np.array([0, 0, 1])
        axis = np.cross(up, dir_vec)
        axis_len = np.linalg.norm(axis)
        if axis_len > 1e-6:
            axis /= axis_len
            angle = math.degrees(np.arccos(np.clip(np.dot(up, dir_vec), -1.0, 1.0)))
            glRotatef(angle, *axis)
        elif dir_vec[2] < 0:
            glRotatef(180, 1, 0, 0)

        height = np.linalg.norm(size_vec)

        quad = gluNewQuadric()
        gluQuadricNormals(quad, GLU_SMOOTH)
        gluCylinder(quad, radius, 0.0, height, 20, 5)
        gluDeleteQuadric(quad)
        glPopMatrix()

        if alpha < 1.0:
            glDisable(GL_BLEND)

    @staticmethod
    def draw_sphere(position, radius=0.1, slices=12, stacks=12, color=(1, 0, 1), alpha=1.0):
        """Nakreslí guľu na pozícii s podporou pre alpha"""
        glPushMatrix()
        glTranslatef(*position)

        # Ak je alpha < 1.0, zapni blending
        if alpha < 1.0:
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glColor4f(color[0], color[1], color[2], alpha)
        else:
            glColor3fv(color)

        quad = gluNewQuadric()
        gluSphere(quad, radius, slices, stacks)
        gluDeleteQuadric(quad)

        if alpha < 1.0:
            glDisable(GL_BLEND)

        glPopMatrix()

    @staticmethod
    def draw_circle_2d(position, radius=0.1, segments=24, color=(1, 1, 0), alpha=1.0):
        """Nakreslí kruh v 2D s podporou pre alpha"""
        if alpha < 1.0:
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glColor4f(color[0], color[1], color[2], alpha)
        else:
            glColor3fv(color)

        glBegin(GL_TRIANGLE_FAN)
        glVertex3f(position[0], position[1], position[2])
        for i in range(segments + 1):
            angle = 2 * math.pi * i / segments
            x = position[0] + math.cos(angle) * radius
            y = position[1] + math.sin(angle) * radius
            glVertex3f(x, y, position[2])
        glEnd()

        if alpha < 1.0:
            glDisable(GL_BLEND)


class MathRenderer:
    """Renderer pre matematické výrazy v ľavom hornom rohu"""

    @staticmethod
    def draw_vector_math(x, y, vec, color=(1, 1, 1), font_size=24, highlight=False):
        """Nakreslí vektor v matematickom zápise"""
        # Hlavná zátvorka
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
        vec_height = len(vec) * (font_size + 5)

        # Zátvorky
        if highlight:
            EnhancedUIRenderer.draw_parenthesis(x - 5, y, vec_height, left=True,
                                                color=color, thickness=3)
            EnhancedUIRenderer.draw_parenthesis(x + 40, y, vec_height, left=False,
                                                color=color, thickness=3)
        else:
            EnhancedUIRenderer.draw_parenthesis(x - 5, y, vec_height, left=True,
                                                color=(0.7, 0.7, 0.7), thickness=2)
            EnhancedUIRenderer.draw_parenthesis(x + 40, y, vec_height, left=False,
                                                color=(0.7, 0.7, 0.7), thickness=2)

        # Zložky vektora
        for i, val in enumerate(vec):
            text_color = color if highlight else (0.8, 0.8, 0.8)
            text = f"{val:.1f}" if isinstance(val, float) else str(val)
            UIRenderer.draw_text_2d(text, (x + 5, y + i * (font_size + 5)),
                                    color=text_color, font_size=font_size)

        glEnable(GL_DEPTH_TEST)
        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)

        return x + 50  # Vráť pozíciu za vektorom

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

        vec_height = len(vec1) * (font_size + 5)

        # Zátvorky
        EnhancedUIRenderer.draw_parenthesis(x - 5, y, vec_height, left=True,
                                            color=color, thickness=3)
        EnhancedUIRenderer.draw_parenthesis(x + 80, y, vec_height, left=False,
                                            color=color, thickness=3)

        # Operácie medzi zložkami
        for i in range(len(vec1)):
            v1 = f"{vec1[i]:.1f}" if isinstance(vec1[i], float) else str(vec1[i])
            v2 = f"{vec2[i]:.1f}" if isinstance(vec2[i], float) else str(vec2[i])
            text = f"{v1} {op} {v2}"
            UIRenderer.draw_text_2d(text, (x + 5, y + i * (font_size + 5)),
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
    def draw_scalar(x, y, scalar, color=(1, 1, 1), font_size=28):
        """Nakreslí skalár"""
        text = f"{scalar:.1f}" if isinstance(scalar, float) else str(scalar)
        UIRenderer.draw_text_2d(text, (x, y + 12), color=color, font_size=font_size)
        return x + 30


class Config:
    """Globálna konfigurácia aplikácie"""
    FPS = 60
    LENGTH_XYZ = 10.0

    # UI dimensions
    MATRIX_CELL_W = 40
    MATRIX_CELL_H = 30
    MATRIX_GAP = 5
    BUTTON_W = 200
    BUTTON_H = 60
    DROPDOWN_ITEM_HEIGHT = 25

    # Colors
    COL_COLORS_OUTLINE = [(1.0, 0.0, 1.0), (0.0, 1.0, 1.0), (1.0, 1.0, 0.0)]


class Camera:
    """Správa kamery a viewportu"""

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

    def move_to_plane(self, plane_type, distance=10.0, custom_view=None):
        """Nastaví cieľovú pozíciu kamery pre danú rovinu"""
        self.start_distance = self.distance
        self.start_azimuth = self.azimuth
        self.start_elevation = self.elevation
        self.target_distance = distance

        if custom_view:  # Pre 3D s vlastným výpočtom
            self.target_azimuth = custom_view['azimuth']
            self.target_elevation = custom_view['elevation']
        elif plane_type == "XY":  # Z=0
            self.target_azimuth = 0.0
            self.target_elevation = 89.0
        elif plane_type == "XZ":  # Y=0
            self.target_azimuth = 0.0
            self.target_elevation = 0.1
        elif plane_type == "YZ":  # X=0
            self.target_azimuth = 90.0
            self.target_elevation = 0.1
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
            self.azimuth = self.start_azimuth + (self.target_azimuth - self.start_azimuth) * smooth_t
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
        gluPerspective(45, (self.width / self.height), 0.1, 500.0)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(cam_x, cam_y, cam_z,
                  self.target[0], self.target[1], self.target[2],
                  0, 1, 0)

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

    def zoom_in(self, is_2d):
        """Priblíženie"""
        if is_2d:
            self.ortho_scale /= 1.1
        else:
            self.distance = max(1.0, self.distance - self.zoom_speed)

    def zoom_out(self, is_2d):
        """Oddialenie"""
        if is_2d:
            self.ortho_scale *= 1.1
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


class GridRenderer:
    """Renderer pre mriežky a plochy"""

    @staticmethod
    def draw_grid_2d(ortho_scale, pan_x, pan_y, width, height, step=1.0, z=-0.1, max_lines=200):
        """Nakreslí 2D mriežku s aspect ratio korekciou"""
        width, height = pygame.display.get_window_size()

        # OPRAVA: Vypočítaj aspect ratio a uprav viditeľnú oblasť
        aspect = width / height

        if aspect >= 1.0:  # Okno je širšie ako vysoké
            left = -ortho_scale * aspect + pan_x
            right = ortho_scale * aspect + pan_x
            bottom = -ortho_scale + pan_y
            top = ortho_scale + pan_y
        else:  # Okno je vyššie ako široké
            left = -ortho_scale + pan_x
            right = ortho_scale + pan_x
            bottom = -ortho_scale / aspect + pan_y
            top = ortho_scale / aspect + pan_y

        visible_range = max(right - left, top - bottom)
        display_step = step
        if visible_range > 50:
            display_step *= int(visible_range / 50)

        start_x = max(math.floor(left / display_step) * display_step,
                      pan_x - max_lines * display_step)
        end_x = min(math.ceil(right / display_step) * display_step,
                    pan_x + max_lines * display_step)
        start_y = max(math.floor(bottom / display_step) * display_step,
                      pan_y - max_lines * display_step)
        end_y = min(math.ceil(top / display_step) * display_step,
                    pan_y + max_lines * display_step)

        glColor3f(0.5, 0.5, 0.5)
        glLineWidth(1.0)
        glBegin(GL_LINES)

        x = start_x
        while x <= end_x:
            glVertex3f(x, bottom, z)
            glVertex3f(x, top, z)
            x += display_step

        y = start_y
        while y <= end_y:
            glVertex3f(left, y, z)
            glVertex3f(right, y, z)
            y += display_step

        glEnd()

    @staticmethod
    def draw_grid_3d(size=10.0, step=1.0):
        """Nakreslí 3D mriežku - VYLEPŠENÉ"""
        # Uložíme pôvodný depth func
        glPushAttrib(GL_DEPTH_BUFFER_BIT)

        # Použijeme LEQUAL aby sa čiary vždy vykreslili
        glDepthFunc(GL_LEQUAL)

        glEnable(GL_POLYGON_OFFSET_LINE)
        glPolygonOffset(-2.0, -2.0)  # Väčší offset

        glLineWidth(2.0)  # Hrubšie čiary
        glBegin(GL_LINES)

        offset = 0.02  # Väčší offset od rovín

        for i in range(-int(size), int(size) + 1, int(step)):
            # XZ plane (červená) - na Y=0
            glColor3f(0.5, 0.0, 0.0)  # Svetlejšia
            glVertex3f(-size, offset, i)
            glVertex3f(size, offset, i)
            glVertex3f(i, offset, -size)
            glVertex3f(i, offset, size)

            # XY plane (zelená) - na Z=0
            glColor3f(0.0, 0.5, 0.0)  # Svetlejšia
            glVertex3f(-size, i, offset)
            glVertex3f(size, i, offset)
            glVertex3f(i, -size, offset)
            glVertex3f(i, size, offset)

            # YZ plane (modrá) - na X=0
            glColor3f(0.0, 0.0, 0.5)  # Svetlejšia
            glVertex3f(offset, -size, i)
            glVertex3f(offset, size, i)
            glVertex3f(offset, i, -size)
            glVertex3f(offset, i, size)

        glEnd()

        glDisable(GL_POLYGON_OFFSET_LINE)
        glPopAttrib()  # Obnoví depth func

    @staticmethod
    def draw_planes_3d(size=2.0, step=1.0, colored=False):
        """Nakreslí 3D plochy - VYLEPŠENÉ"""
        # Najprv nakresli plochy
        # XZ plane (Y=0)
        if colored:
            glColor3f(0.8, 0.8, 0.8)
        else:
            glColor3f(1.0, 0, 0)
        glBegin(GL_QUADS)
        glVertex3f(-size, 0, -size)
        glVertex3f(size, 0, -size)
        glVertex3f(size, 0, size)
        glVertex3f(-size, 0, size)
        glEnd()

        # XY plane (Z=0)
        if colored:
            glColor3f(0.8, 0.8, 0.8)
        else:
            glColor3f(0, 1, 0)
        glBegin(GL_QUADS)
        glVertex3f(-size, -size, 0)
        glVertex3f(size, -size, 0)
        glVertex3f(size, size, 0)
        glVertex3f(-size, size, 0)
        glEnd()

        # YZ plane (X=0)
        if colored:
            glColor3f(0.8, 0.8, 0.8)
        else:
            glColor3f(0, 0, 1)
        glBegin(GL_QUADS)
        glVertex3f(0, -size, -size)
        glVertex3f(0, size, -size)
        glVertex3f(0, size, size)
        glVertex3f(0, -size, size)
        glEnd()

        # Grid čiary s agresívnejším offsetom
        glPushAttrib(GL_DEPTH_BUFFER_BIT)
        glDepthFunc(GL_LEQUAL)

        glEnable(GL_POLYGON_OFFSET_LINE)
        glPolygonOffset(-3.0, -3.0)  # Ešte väčší offset

        glColor3f(0, 0, 0)
        glLineWidth(1.5)
        glBegin(GL_LINES)

        offset = 0.01

        # XZ plane grid (na Y=0)
        for x in range(-int(size), int(size) + 1, int(step)):
            glVertex3f(x, offset, -size)
            glVertex3f(x, offset, size)
        for z in range(-int(size), int(size) + 1, int(step)):
            glVertex3f(-size, offset, z)
            glVertex3f(size, offset, z)

        # XY plane grid (na Z=0)
        for x in range(-int(size), int(size) + 1, int(step)):
            glVertex3f(x, -size, offset)
            glVertex3f(x, size, offset)
        for y in range(-int(size), int(size) + 1, int(step)):
            glVertex3f(-size, y, offset)
            glVertex3f(size, y, offset)

        # YZ plane grid (na X=0)
        for y in range(-int(size), int(size) + 1, int(step)):
            glVertex3f(offset, y, -size)
            glVertex3f(offset, y, size)
        for z in range(-int(size), int(size) + 1, int(step)):
            glVertex3f(offset, -size, z)
            glVertex3f(offset, size, z)

        glEnd()

        glDisable(GL_POLYGON_OFFSET_LINE)
        glPopAttrib()
class AnimationController:
    """Controls step-by-step animation of vector operations - S PLYNULÝMI PRECHODMI"""

    def __init__(self):
        self.current_operation = None
        self.current_step = 0
        self.max_steps = 0
        self.operands = []
        self.result = None
        self.operation_type = None
        self.constant = None
        self.math_display_step = 0

        self.RESULT_COLOR = (0.5, 0, 0.5)
        # Animačný stav
        self.animating = False
        self.animation_progress = 0.0
        self.animation_speed = 0.6  # Rýchlosť animácie (vyššie = rýchlejšie)
        self.source_vectors = []  # Vektory na začiatku animácie
        self.target_vectors = []  # Cieľové vektory

        self.current_plane = None  # None, "XY", "XZ", "YZ", alebo "3D"

    def compute_optimal_view_for_current_step(self):
        """Vypočíta optimálnu pozíciu kamery pre aktuálny krok animácie"""
        if not self.current_operation:
            return None

        # Získaj vektory pre aktuálny krok
        current_vectors = self._get_vectors_for_step(self.current_step)

        if not current_vectors:
            return None

        # Zozbieraj všetky viditeľné body (s offsetmi)
        points = []

        for v in current_vectors:
            vec = v['vec']
            offset = v.get('offset', [0, 0, 0])
            alpha = v.get('alpha', 1.0)

            # Ignoruj úplne priesvitné vektory (alpha < 0.2)
            if alpha < 0.2:
                continue

            if isinstance(vec[0], (int, float)):
                # Jednoduchý vektor
                vec3d = list(vec) + [0] * (3 - len(vec))
                vec3d = vec3d[:3]
                # Pridaj aj koncový bod vektora s offsetom
                end_point = [vec3d[i] + offset[i] for i in range(3)]
                points.append(end_point)
                # Pridaj aj origin ak je offset nenulový
                if any(abs(o) > 1e-6 for o in offset):
                    points.append(offset[:3])
            else:
                # Matica
                row_offsets = v.get('row_offsets', None)
                for i, row in enumerate(vec):
                    vec3d = list(row) + [0] * (3 - len(row))
                    vec3d = vec3d[:3]

                    if row_offsets and i < len(row_offsets):
                        row_offset = row_offsets[i][:3]
                    else:
                        row_offset = offset[:3]

                    # Koncový bod
                    end_point = [vec3d[j] + row_offset[j] for j in range(3)]
                    points.append(end_point)
                    # Origin
                    if any(abs(o) > 1e-6 for o in row_offset):
                        points.append(row_offset)

        # Pridaj origin
        points.append([0, 0, 0])

        if len(points) < 2:
            return None

        # Pre jednoduché prípady (2-3 body) použi cross product
        if len(points) <= 4 and len(current_vectors) <= 2:
            # Skús nájsť 2 nenulové vektory
            vectors_for_cross = []
            for v in current_vectors:
                vec = v['vec']
                if isinstance(vec[0], (int, float)):
                    vec3d = list(vec) + [0] * (3 - len(vec))
                    if sum(abs(x) for x in vec3d[:3]) > 1e-6:
                        vectors_for_cross.append(vec3d[:3])
                else:
                    for row in vec:
                        vec3d = list(row) + [0] * (3 - len(row))
                        if sum(abs(x) for x in vec3d[:3]) > 1e-6:
                            vectors_for_cross.append(vec3d[:3])

                if len(vectors_for_cross) >= 2:
                    break

            if len(vectors_for_cross) >= 2:
                v1 = np.array(vectors_for_cross[0])
                v2 = np.array(vectors_for_cross[1])

                normal = np.cross(v1, v2)

                if np.linalg.norm(normal) > 1e-6:
                    normal = normal / np.linalg.norm(normal)

                    # Zabezpeč že normála smeruje "hore"
                    if normal[1] < 0:
                        normal = -normal

                    azimuth = np.degrees(np.arctan2(normal[0], normal[2]))
                    elevation = np.degrees(np.arcsin(np.clip(normal[1], -1.0, 1.0)))
                    elevation = max(15, min(75, abs(elevation)))

                    return {
                        'azimuth': azimuth,
                        'elevation': elevation,
                        'normal': normal.tolist()
                    }

        # PCA pre zložitejšie prípady
        points_np = np.array(points)
        centroid = np.mean(points_np, axis=0)
        centered = points_np - centroid

        cov_matrix = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

        idx = eigenvalues.argsort()[::-1]
        eigenvectors = eigenvectors[:, idx]

        normal = eigenvectors[:, 2]

        # Zabezpeč že normála smeruje "hore"
        if normal[1] < 0:
            normal = -normal

        azimuth = np.degrees(np.arctan2(normal[0], normal[2]))
        elevation = np.degrees(np.arcsin(np.clip(normal[1] / np.linalg.norm(normal), -1.0, 1.0)))
        elevation = max(15, min(75, abs(elevation)))

        return {
            'azimuth': azimuth,
            'elevation': elevation,
            'normal': normal.tolist()
        }

    def compute_simple_view_for_vectors(self):
        """Jednoduchší výpočet optimálneho pohľadu pre 2 vektory"""
        if not self.operands or len(self.operands) < 2:
            return None

        # Získaj prvé 2 vektory
        v1 = self.operands[0]
        v2 = self.operands[1]

        # Konvertuj na numpy 3D vektory
        if isinstance(v1[0], (list, tuple)):
            # Matica - vezmi prvý riadok
            v1 = list(v1[0]) + [0] * (3 - len(v1[0]))
        else:
            v1 = list(v1) + [0] * (3 - len(v1))

        if isinstance(v2[0], (list, tuple)):
            # Matica - vezmi prvý riadok
            v2 = list(v2[0]) + [0] * (3 - len(v2[0]))
        else:
            v2 = list(v2) + [0] * (3 - len(v2))

        v1 = np.array(v1[:3])
        v2 = np.array(v2[:3])

        # Normálový vektor roviny = cross product
        normal = np.cross(v1, v2)

        if np.linalg.norm(normal) < 1e-6:
            # Vektory sú paralelné - použijem default view
            return None

        normal = normal / np.linalg.norm(normal)

        # Vypočítaj azimuth a elevation z normály
        # Normála smeruje von z roviny, chceme sa pozerať proti nej
        azimuth = np.degrees(np.arctan2(normal[0], normal[2]))
        elevation = np.degrees(np.arcsin(np.clip(normal[1], -1.0, 1.0)))

        # Prispôsob uhly pre lepší pohľad
        elevation = max(15, min(75, abs(elevation)))

        return {
            'azimuth': azimuth,
            'elevation': elevation,
            'normal': normal.tolist()
        }

    def compute_optimal_view_for_3d(self):
        """Vypočíta optimálny uhol pohľadu pre 3D vektory pomocou PCA"""
        if not self.operands:
            return None

        # Pre jednoduché prípady (2 vektory) použi cross product
        if len(self.operands) == 2:
            simple_view = self.compute_simple_view_for_vectors()
            if simple_view:
                return simple_view

        # Zozbieraj všetky body
        points = []
        for op in self.operands:
            if isinstance(op[0], (list, tuple)):  # Matica
                for row in op:
                    vec3d = list(row) + [0] * (3 - len(row))
                    points.append(vec3d[:3])
            else:  # Vektor
                vec3d = list(op) + [0] * (3 - len(op))
                points.append(vec3d[:3])

        # Pridaj aj výsledok ak existuje
        if self.result:
            if isinstance(self.result[0], (list, tuple)):
                for row in self.result:
                    vec3d = list(row) + [0] * (3 - len(row))
                    points.append(vec3d[:3])
            else:
                vec3d = list(self.result) + [0] * (3 - len(self.result))
                points.append(vec3d[:3])

        if len(points) < 2:
            return None

        # Vypočítaj centroid
        points_np = np.array(points)
        centroid = np.mean(points_np, axis=0)

        # Centroid bod do originu
        centered = points_np - centroid

        # PCA - nájdi hlavné smery
        cov_matrix = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

        # Zoraď podľa eigenvalues (najväčší = hlavný smer)
        idx = eigenvalues.argsort()[::-1]
        eigenvectors = eigenvectors[:, idx]

        # Normálový vektor roviny = tretí eigenvektor (najmenší)
        normal = eigenvectors[:, 2]

        # Zabezpeč že normála smeruje "hore"
        if normal[1] < 0:
            normal = -normal

        # Vypočítaj azimuth a elevation z normály
        azimuth = np.degrees(np.arctan2(normal[0], normal[2]))
        elevation = np.degrees(np.arcsin(np.clip(normal[1] / np.linalg.norm(normal), -1.0, 1.0)))

        # Prispôsob uhly pre lepší pohľad
        elevation = max(15, min(75, abs(elevation)))

        return {
            'azimuth': azimuth,
            'elevation': elevation,
            'normal': normal.tolist()
        }

    def detect_operation_plane(self, tolerance=1e-6):
        """Zistí v akej rovine sa operácia vykonáva"""
        if not self.operands:
            return "3D"

        # Skontroluj prvý operand
        vec = self.operands[0]
        is_matrix = isinstance(vec[0], (list, tuple))

        if is_matrix:
            # Matica - skontroluj všetky vektory
            all_z_zero = all(abs(row[2] if len(row) > 2 else 0) < tolerance for row in vec)
            all_y_zero = all(abs(row[1] if len(row) > 1 else 0) < tolerance for row in vec)
            all_x_zero = all(abs(row[0]) < tolerance for row in vec)
        else:
            # Jednoduchý vektor
            all_z_zero = abs(vec[2] if len(vec) > 2 else 0) < tolerance
            all_y_zero = abs(vec[1] if len(vec) > 1 else 0) < tolerance
            all_x_zero = abs(vec[0]) < tolerance

        # Kontrola druhého operandu ak existuje
        if len(self.operands) > 1:
            vec2 = self.operands[1]
            is_matrix2 = isinstance(vec2[0], (list, tuple))

            if is_matrix2:
                all_z_zero = all_z_zero and all(abs(row[2] if len(row) > 2 else 0) < tolerance for row in vec2)
                all_y_zero = all_y_zero and all(abs(row[1] if len(row) > 1 else 0) < tolerance for row in vec2)
                all_x_zero = all_x_zero and all(abs(row[0]) < tolerance for row in vec2)
            else:
                all_z_zero = all_z_zero and (abs(vec2[2] if len(vec2) > 2 else 0) < tolerance)
                all_y_zero = all_y_zero and (abs(vec2[1] if len(vec2) > 1 else 0) < tolerance)
                all_x_zero = all_x_zero and (abs(vec2[0]) < tolerance)

        # Urči rovinu
        if all_z_zero:
            return "XY"
        elif all_y_zero:
            return "XZ"
        elif all_x_zero:
            return "YZ"
        else:
            return "3D"

    def setup_operation(self, operation_type, operands, result, constant=None):
        """Setup a new operation for step-by-step visualization"""
        self.operation_type = operation_type
        self.operands = operands
        self.result = result
        self.constant = constant
        self.current_step = 0

        # Define number of steps based on operation
        if operation_type == 'add':
            self.max_steps = 2
        elif operation_type == 'subtract':
            self.max_steps = 3
        elif operation_type == 'scalar_mult':
            self.max_steps = 1
        elif operation_type == 'linear_combination':
            self.max_steps = 4
        else:
            self.max_steps = 1

        self.current_operation = operation_type

        # Nastav počiatočné vektory
        self.source_vectors = self._get_vectors_for_step(0)
        self.target_vectors = self.source_vectors
        self.animation_progress = 1.0

        self.current_plane = self.detect_operation_plane()

    def next_step(self):
        """Move to next step with animation"""
        if self.current_step < self.max_steps:
            self.source_vectors = self._get_vectors_for_step(self.current_step)
            self.current_step += 1
            self.target_vectors = self._get_vectors_for_step(self.current_step)
            self.animating = True
            self.animation_progress = 0.0
            return True
        return False

    def prev_step(self):
        """Move to previous step with animation"""
        if self.current_step > 0:
            self.source_vectors = self._get_vectors_for_step(self.current_step)
            self.current_step -= 1
            self.target_vectors = self._get_vectors_for_step(self.current_step)
            self.animating = True
            self.animation_progress = 0.0
            return True
        return False

    def update(self, dt):
        """Update animation progress"""
        if self.animating:
            self.animation_progress += dt * self.animation_speed
            if self.animation_progress >= 1.0:
                self.animation_progress = 1.0
                self.animating = False

    def reset(self):
        """Reset animation"""
        self.current_step = 0
        self.animation_progress = 1.0
        self.animating = False
        if self.current_operation:
            self.source_vectors = self._get_vectors_for_step(0)
            self.target_vectors = self.source_vectors

    def clear(self):
        """Clear current operation"""
        self.current_operation = None
        self.current_step = 0
        self.max_steps = 0
        self.operands = []
        self.result = None
        self.constant = None
        self.animating = False
        self.animation_progress = 0.0
        self.source_vectors = []
        self.target_vectors = []

    def _is_matrix(self, data):
        """Check if data is a matrix (list of lists)"""
        if not isinstance(data, (list, tuple)):
            return False
        if len(data) == 0:
            return False
        return isinstance(data[0], (list, tuple))

    def _interpolate_vector(self, source, target, t):
        """Interpoluje medzi dvoma vektormi - OPRAVENÉ PRE row_offsets"""
        result = source.copy()

        # Interpoluj vektor
        if isinstance(source['vec'][0], (int, float)):
            # Jednoduchý vektor
            result['vec'] = [s + (tv - s) * self.animation_progress
                             for s, tv in zip(source['vec'], target['vec'])]
        else:
            # Matica
            result['vec'] = [[s + (tv - s) * self.animation_progress
                              for s, tv in zip(s_row, t_row)]
                             for s_row, t_row in zip(source['vec'], target['vec'])]

        # Interpoluj offset ak existuje
        if 'offset' in source and 'offset' in target:
            result['offset'] = [s + (to - s) * self.animation_progress
                                for s, to in zip(source['offset'], target['offset'])]
        elif 'offset' in target:
            # Ak source nemá offset, začni od [0,0,0]
            src_offset = source.get('offset', [0, 0, 0])
            result['offset'] = [s + (to - s) * self.animation_progress
                                for s, to in zip(src_offset, target['offset'])]

        # OPRAVA: Správna interpolácia row_offsets
        if 'row_offsets' in target:
            if 'row_offsets' in source:
                # Interpoluj medzi existujúcimi row_offsets
                result['row_offsets'] = [
                    [s + (to - s) * self.animation_progress
                     for s, to in zip(s_off, t_off)]
                    for s_off, t_off in zip(source['row_offsets'], target['row_offsets'])
                ]
            else:
                # Source nemá row_offsets, začni od [0,0,0] pre každý riadok
                src_offset = source.get('offset', [0, 0, 0])
                result['row_offsets'] = [
                    [src_offset[j] + (t_off[j] - src_offset[j]) * self.animation_progress
                     for j in range(len(t_off))]
                    for t_off in target['row_offsets']
                ]
        elif 'row_offsets' in source:
            # Target nemá row_offsets, interpoluj späť k [0,0,0]
            tgt_offset = target.get('offset', [0, 0, 0])
            result['row_offsets'] = [
                [s + (tgt_offset[j] - s) * self.animation_progress
                 for j, s in enumerate(s_off)]
                for s_off in source['row_offsets']
            ]

        # Interpoluj alpha
        source_alpha = source.get('alpha', 1.0)
        target_alpha = target.get('alpha', 1.0)
        result['alpha'] = source_alpha + (target_alpha - source_alpha) * self.animation_progress

        # Interpoluj color ak sa mení
        if 'color' in source and 'color' in target:
            result['color'] = tuple(
                s + (t - s) * self.animation_progress
                for s, t in zip(source['color'], target['color'])
            )
        elif 'color' in target:
            result['color'] = target['color']

        # Skopíruj label
        if 'label' in target:
            result['label'] = target['label']

        return result

    def _get_vectors_for_step(self, step):
        """Získa vektory pre daný krok (bez animácie)"""
        if not self.current_operation:
            return []

        vectors = []
        is_matrix_op = self._is_matrix(self.operands[0]) if self.operands else False

        if self.operation_type == 'add':
            if step == 0:
                vectors.append({
                    'vec': self.operands[0],
                    'offset': [0, 0, 0],
                    'color': (1, 0.5, 0),
                    'label': 'A' if is_matrix_op else 'v1'
                })
                vectors.append({
                    'vec': self.operands[1],
                    'offset': [0, 0, 0],
                    'color': (0, 0.5, 1),
                    'label': 'B' if is_matrix_op else 'v2'
                })
            elif step == 1:
                if is_matrix_op:
                    vectors.append({
                        'vec': self.operands[0],
                        'offset': [0, 0, 0],
                        'color': (1, 0.5, 0),
                        'label': 'A'
                    })
                    row_offsets = []
                    for row in self.operands[0]:
                        offset = list(row) + [0] * (3 - len(row))
                        row_offsets.append(offset[:3])
                    vectors.append({
                        'vec': self.operands[1],
                        'offset': [0, 0, 0],
                        'row_offsets': row_offsets,
                        'color': (0, 0.5, 1),
                        'label': 'B'
                    })
                else:
                    vectors.append({
                        'vec': self.operands[0],
                        'offset': [0, 0, 0],
                        'color': (1, 0.5, 0),
                        'label': 'v1'
                    })
                    offset = list(self.operands[0]) + [0] * (3 - len(self.operands[0]))
                    vectors.append({
                        'vec': self.operands[1],
                        'offset': offset[:3],
                        'color': (0, 0.5, 1),
                        'label': 'v2'
                    })
            elif step == 2:
                vectors.append({
                    'vec': self.operands[0],
                    'offset': [0, 0, 0],
                    'color': (1, 0.5, 0),
                    'alpha': 0.3,
                    'label': 'A' if is_matrix_op else 'v1'
                })
                if is_matrix_op:
                    row_offsets = []
                    for row in self.operands[0]:
                        offset = list(row) + [0] * (3 - len(row))
                        row_offsets.append(offset[:3])
                    vectors.append({
                        'vec': self.operands[1],
                        'offset': [0, 0, 0],
                        'row_offsets': row_offsets,
                        'color': (0, 0.5, 1),
                        'alpha': 0.3,
                        'label': 'B'
                    })
                else:
                    offset = list(self.operands[0]) + [0] * (3 - len(self.operands[0]))
                    vectors.append({
                        'vec': self.operands[1],
                        'offset': offset[:3],
                        'color': (0, 0.5, 1),
                        'alpha': 0.3,
                        'label': 'v2'
                    })
                vectors.append({
                    'vec': self.result,
                    'offset': [0, 0, 0],
                    'color': self.RESULT_COLOR,
                    'alpha': 1.0,
                    'label': 'A + B' if is_matrix_op else 'v1 + v2'
                })

        elif self.operation_type == 'subtract':
            if step == 0:
                vectors.append({
                    'vec': self.operands[0],
                    'offset': [0, 0, 0],
                    'color': (1, 0.5, 0),
                    'label': 'A' if is_matrix_op else 'v1'
                })
                vectors.append({
                    'vec': self.operands[1],
                    'offset': [0, 0, 0],
                    'color': (0, 0.5, 1),
                    'label': 'B' if is_matrix_op else 'v2'
                })
            elif step == 1:
                vectors.append({
                    'vec': self.operands[0],
                    'offset': [0, 0, 0],
                    'color': (1, 0.5, 0),
                    'label': 'A' if is_matrix_op else 'v1'
                })
                if is_matrix_op:
                    negated = [[-x for x in row] for row in self.operands[1]]
                    vectors.append({
                        'vec': negated,
                        'offset': [0, 0, 0],
                        'color': (1, 0, 0.5),
                        'label': '-B'
                    })
                else:
                    negated = [-x for x in self.operands[1]]
                    vectors.append({
                        'vec': negated,
                        'offset': [0, 0, 0],
                        'color': (1, 0, 0.5),
                        'label': '-v2'
                    })
            elif step == 2:
                vectors.append({
                    'vec': self.operands[0],
                    'offset': [0, 0, 0],
                    'color': (1, 0.5, 0),
                    'label': 'A' if is_matrix_op else 'v1'
                })
                if is_matrix_op:
                    negated = [[-x for x in row] for row in self.operands[1]]
                    row_offsets = []
                    for row in self.operands[0]:
                        offset = list(row) + [0] * (3 - len(row))
                        row_offsets.append(offset[:3])
                    vectors.append({
                        'vec': negated,
                        'offset': [0, 0, 0],
                        'row_offsets': row_offsets,
                        'color': (1, 0, 0.5),
                        'label': '-B'
                    })
                else:
                    negated = [-x for x in self.operands[1]]
                    offset = list(self.operands[0]) + [0] * (3 - len(self.operands[0]))
                    vectors.append({
                        'vec': negated,
                        'offset': offset[:3],
                        'color': (1, 0, 0.5),
                        'label': '-v2'
                    })
            elif step == 3:
                vectors.append({
                    'vec': self.operands[0],
                    'offset': [0, 0, 0],
                    'color': (1, 0.5, 0),
                    'alpha': 0.3,
                    'label': 'A' if is_matrix_op else 'v1'
                })
                if is_matrix_op:
                    negated = [[-x for x in row] for row in self.operands[1]]
                    row_offsets = []
                    for row in self.operands[0]:
                        offset = list(row) + [0] * (3 - len(row))
                        row_offsets.append(offset[:3])
                    vectors.append({
                        'vec': negated,
                        'offset': [0, 0, 0],
                        'row_offsets': row_offsets,
                        'color': (1, 0, 0.5),
                        'alpha': 0.3,
                        'label': '-B'
                    })
                else:
                    negated = [-x for x in self.operands[1]]
                    offset = list(self.operands[0]) + [0] * (3 - len(self.operands[0]))
                    vectors.append({
                        'vec': negated,
                        'offset': offset[:3],
                        'color': (1, 0, 0.5),
                        'alpha': 0.3,
                        'label': '-v2'
                    })
                vectors.append({
                    'vec': self.result,
                    'offset': [0, 0, 0],
                    'color': self.RESULT_COLOR,
                    'alpha': 1.0,
                    'label': 'A - B' if is_matrix_op else 'v1 - v2'
                })

        elif self.operation_type == 'scalar_mult':
            if step == 0:
                vectors.append({
                    'vec': self.operands[0],
                    'offset': [0, 0, 0],
                    'color': (1, 0.5, 0),
                    'label': 'A' if is_matrix_op else 'v'
                })
            elif step == 1:
                label = f'{self.constant}·A' if is_matrix_op else f'{self.constant}·v'
                vectors.append({
                    'vec': self.result,
                    'offset': [0, 0, 0],
                    'color': self.RESULT_COLOR,
                    'label': label
                })
        # V _get_vectors_for_step pridaj (pred_return vectors):
        elif self.operation_type == 'linear_combination':
            c1, c2 = self.constant if self.constant else [1, 1]

            if step == 0:
                # Zobraz oba pôvodné vektory
                vectors.append({
                    'vec': self.operands[0],
                    'offset': [0, 0, 0],
                    'color': (1, 0.5, 0),
                    'label': 'v1'
                })
                vectors.append({
                    'vec': self.operands[1],
                    'offset': [0, 0, 0],
                    'color': (0, 0.5, 1),
                    'label': 'v2'
                })

            elif step == 1:
                # Zobraz c1*v1 (zvýrazni), v2 priesvitne
                if is_matrix_op:
                    scaled_v1 = [[c1 * x for x in row] for row in self.operands[0]]
                    vectors.append({
                        'vec': scaled_v1,
                        'offset': [0, 0, 0],
                        'color': (1, 0.5, 0),
                        'label': f'{c1}·v1'
                    })
                else:
                    scaled_v1 = [c1 * x for x in self.operands[0]]
                    vectors.append({
                        'vec': scaled_v1,
                        'offset': [0, 0, 0],
                        'color': (1, 0.5, 0),
                        'label': f'{c1}·v1'
                    })
                vectors.append({
                    'vec': self.operands[1],
                    'offset': [0, 0, 0],
                    'color': (0, 0.5, 1),
                    'alpha': 0.3,
                    'label': 'v2'
                })

            elif step == 2:
                # Zobraz c2*v2 (zvýrazni), v1 priesvitne
                if is_matrix_op:
                    scaled_v1 = [[c1 * x for x in row] for row in self.operands[0]]
                    scaled_v2 = [[c2 * x for x in row] for row in self.operands[1]]
                    vectors.append({
                        'vec': scaled_v1,
                        'offset': [0, 0, 0],
                        'color': (1, 0.5, 0),
                        'alpha': 0.3,
                        'label': 'v1'
                    })
                else:
                    scaled_v1 = [c1 * x for x in self.operands[0]]
                    scaled_v2 = [c2 * x for x in self.operands[1]]
                    vectors.append({
                        'vec': scaled_v1,
                        'offset': [0, 0, 0],
                        'color': (1, 0.5, 0),
                        'alpha': 0.3,
                        'label': 'v1'
                    })

                vectors.append({
                    'vec': scaled_v2,
                    'offset': [0, 0, 0],
                    'color': (0, 0.5, 1),
                    'label': f'{c2}·v2'
                })

            elif step == 3:
                # Zobraz c1*v1 + c2*v2 (oba vektory viditeľné)
                if is_matrix_op:
                    scaled_v1 = [[c1 * x for x in row] for row in self.operands[0]]
                    scaled_v2 = [[c2 * x for x in row] for row in self.operands[1]]
                    vectors.append({
                        'vec': scaled_v1,
                        'offset': [0, 0, 0],
                        'color': (1, 0.5, 0),
                        'label': f'{c1}·v1'
                    })
                    row_offsets = []
                    for row in scaled_v1:
                        offset = list(row) + [0] * (3 - len(row))
                        row_offsets.append(offset[:3])

                    vectors.append({
                        'vec': scaled_v2,
                        'offset': [0, 0, 0],
                        'row_offsets': row_offsets,
                        'color': (0, 0.5, 1),
                        'label': f'{c2}·v2'
                    })

                else:
                    scaled_v1 = [c1 * x for x in self.operands[0]]
                    scaled_v2 = [c2 * x for x in self.operands[1]]
                    vectors.append({
                        'vec': scaled_v1,
                        'offset': [0, 0, 0],
                        'color': (1, 0.5, 0),
                        'label': f'{c1}·v1'
                    })
                    offset = list(scaled_v1) + [0] * (3 - len(scaled_v1))
                    vectors.append({
                        'vec': scaled_v2,
                        'offset': offset[:3],
                        'color': (0, 0.5, 1),
                        'label': f'{c2}·v2'
                    })

            elif step == 4:
                # Výsledok (predošlé vektory priesvitné)
                if is_matrix_op:
                    scaled_v1 = [[c1 * x for x in row] for row in self.operands[0]]
                    scaled_v2 = [[c2 * x for x in row] for row in self.operands[1]]
                    vectors.append({
                        'vec': scaled_v1,
                        'offset': [0, 0, 0],
                        'color': (1, 0.5, 0),
                        'alpha': 0.3,
                        'label': f'{c1}·v1'
                    })

                    row_offsets = []
                    for row in scaled_v1:
                        offset = list(row) + [0] * (3 - len(row))
                        row_offsets.append(offset[:3])

                    vectors.append({
                        'vec': scaled_v2,
                        'offset': [0, 0, 0],
                        'row_offsets': row_offsets,
                        'color': (0, 0.5, 1),
                        'alpha': 0.3,
                        'label': f'{c2}·v2'
                    })

                else:
                    scaled_v1 = [c1 * x for x in self.operands[0]]
                    scaled_v2 = [c2 * x for x in self.operands[1]]
                    vectors.append({
                        'vec': scaled_v1,
                        'offset': [0, 0, 0],
                        'color': (1, 0.5, 0),
                        'alpha': 0.3,
                        'label': f'{c1}·v1'
                    })

                    offset = list(scaled_v1) + [0] * (3 - len(scaled_v1))
                    vectors.append({
                        'vec': scaled_v2,
                        'offset': offset[:3],
                        'color': (0, 0.5, 1),
                        'alpha': 0.3,
                        'label': f'{c2}·v2'
                    })

                vectors.append({
                    'vec': self.result,
                    'offset': [0, 0, 0],
                    'color': self.RESULT_COLOR,
                    'label': f'{c1}·v1 + {c2}·v2'
                })


        return vectors

    def get_vectors_to_draw(self):
        """Get interpolated vectors based on animation progress"""
        if not self.current_operation:
            return []

        # Ak neanimujeme, vráť target vektory
        if not self.animating or self.animation_progress >= 1.0:
            return self.target_vectors

        interpolated = []

        # Interpoluj existujúce vektory (ktoré sú v oboch)
        min_count = min(len(self.source_vectors), len(self.target_vectors))
        for i in range(min_count):
            interpolated.append(self._interpolate_vector(
                self.source_vectors[i],
                self.target_vectors[i],
                self.animation_progress
            ))

        # Pridaj nové vektory (ktoré sú len v target) - animuj ich od progress=0
        if len(self.target_vectors) > len(self.source_vectors):
            for i in range(len(self.source_vectors), len(self.target_vectors)):
                new_vec = self.target_vectors[i].copy()

                # Animuj vektor od nuly do plnej veľkosti
                if isinstance(new_vec['vec'][0], (int, float)):
                    new_vec['vec'] = [v * self.animation_progress for v in new_vec['vec']]
                    new_vec['progress'] = self.animation_progress
                else:
                    new_vec['vec'] = [[v * self.animation_progress for v in row]
                                      for row in new_vec['vec']]
                    if 'row_progress' not in new_vec:
                        new_vec['row_progress'] = [self.animation_progress] * len(new_vec['vec'])
                    else:
                        new_vec['row_progress'] = [self.animation_progress] * len(new_vec['row_progress'])

                interpolated.append(new_vec)

        # Fade-out staré vektory (ktoré sú len v source)
        elif len(self.source_vectors) > len(self.target_vectors):
            for i in range(len(self.target_vectors), len(self.source_vectors)):
                old_vec = self.source_vectors[i].copy()
                old_vec['alpha'] = 1.0 - self.animation_progress
                interpolated.append(old_vec)

        return interpolated

    def get_math_display_info(self):
        """Vráti informácie pre matematické zobrazenie aktuálneho kroku"""
        if not self.current_operation:
            return None

        info = {
            'operation': self.operation_type,
            'step': self.current_step,
            'operands': self.operands,
            'result': self.result,
            'constant': self.constant
        }

        return info


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


import math
import pygame
import numpy as np
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import ast
import random


class Application:
    """Hlavná aplikácia - KOMPLETNE AKTUALIZOVANÁ"""

    def __init__(self):
        pygame.init()

        # Enable hardware multisampling
        pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLEBUFFERS, 1)
        pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLESAMPLES, 4)

        self.RESULT_COLOR = (0.5, 0, 0.5)
        self.screen = pygame.display.set_mode((900, 700), DOUBLEBUF | OPENGL | RESIZABLE)
        #self.width = self.screen.get_width()
        #self.height = self.screen.get_height()

        self.width,self.height = pygame.display.get_window_size()

        pygame.display.set_caption("XYZ axes — OOP Version")

        # Setup OpenGL
        self.setup_opengl()

        # Components
        self.camera = Camera(self.width, self.height)
        self.vector_renderer = VectorRenderer()
        self.ui_renderer = UIRenderer()
        self.grid_renderer = GridRenderer()
        self.axes_renderer = AxesRenderer()

        # NEW: Enhanced components
        self.vector_manager = VectorManager()
        self.input_handler = InputHandler()
        self.enhanced_ui = EnhancedUIRenderer()

        # State
        self.clock = pygame.time.Clock()
        self.running = True
        self.startup_screen = True
        self.is_not_baza = True
        self.view_2d_mode = True
        self.background_dark = True
        self.grid_mode = 0
        self.show_axes = True
        self.vectors_as_points = True

        # Vectors - OPRAVA: už nepoužívame priamo animated_vectors
        self.saved_baza = []
        self.pending_vector = None

        # UI State
        self.matrix_inputs = [["" for _ in range(3)] for _ in range(3)]
        self.matrix_active_cell = (-1, -1)
        self.show_matrix_input = True

        # Dropdown
        self.dropdown_open = False
        self.dropdown_rect = pygame.Rect(10, 10, 150, 30)

        # PRIDAJ: Inicializuj pygame font
        pygame.font.init()

    def setup_opengl(self):
        """Inicializuje OpenGL"""
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_MULTISAMPLE)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_LINE_SMOOTH)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
        glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST)
        glClearColor(0.0, 0.0, 0.0, 1.0)

    def run_startup_screen(self):
        """Spustí úvodný startup screen"""
        while self.startup_screen and self.running:
            dt = self.clock.tick(Config.FPS) / 1000.0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    self.startup_screen = False
                    return

                elif event.type == pygame.VIDEORESIZE:
                    self.width, self.height = event.w, event.h
                    self.screen = pygame.display.set_mode((self.width, self.height),
                                                          DOUBLEBUF | OPENGL | RESIZABLE)
                    glViewport(0, 0, self.width, self.height)

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mx, my = event.pos
                    button_w, button_h = 200, 60
                    button_2d_rect = pygame.Rect(self.width // 2 - button_w - 20,
                                                 self.height // 2 - button_h // 2,
                                                 button_w, button_h)
                    button_3d_rect = pygame.Rect(self.width // 2 + 20,
                                                 self.height // 2 - button_h // 2,
                                                 button_w, button_h)
                    toggle_bg_rect = pygame.Rect(20, self.height - 50, 120, 35)

                    if toggle_bg_rect.collidepoint(mx, my):
                        self.background_dark = not self.background_dark
                    elif button_2d_rect.collidepoint(mx, my):
                        self.view_2d_mode = True
                        self.startup_screen = False
                    elif button_3d_rect.collidepoint(mx, my):
                        self.view_2d_mode = False
                        self.startup_screen = False

            # Render startup screen
            glMatrixMode(GL_PROJECTION)
            glLoadIdentity()
            glOrtho(0, self.width, self.height, 0, -1, 1)
            glMatrixMode(GL_MODELVIEW)
            glLoadIdentity()

            if self.background_dark:
                glClearColor(0.0, 0.0, 0.0, 1.0)
            else:
                glClearColor(1.0, 1.0, 1.0, 1.0)

            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            button_w, button_h = 200, 60
            button_2d_rect = pygame.Rect(self.width // 2 - button_w - 20,
                                         self.height // 2 - button_h // 2,
                                         button_w, button_h)
            button_3d_rect = pygame.Rect(self.width // 2 + 20,
                                         self.height // 2 - button_h // 2,
                                         button_w, button_h)
            toggle_bg_rect = pygame.Rect(20, self.height - 50, 120, 35)

            self.ui_renderer.draw_button_2d(button_2d_rect.x, button_2d_rect.y,
                                            button_2d_rect.w, button_2d_rect.h,
                                            "2D režim", active=True)
            self.ui_renderer.draw_button_2d(button_3d_rect.x, button_3d_rect.y,
                                            button_3d_rect.w, button_3d_rect.h,
                                            "3D režim", active=True)
            label = "Dark Mode" if self.background_dark else "Light Mode"
            self.ui_renderer.draw_button_2d(toggle_bg_rect.x, toggle_bg_rect.y,
                                            toggle_bg_rect.width, toggle_bg_rect.height,
                                            label, active=True)

            pygame.display.flip()

    def run_baza_input(self):
        """Spustí obrazovku pre zadanie bázy"""
        pending_input_panel = None
        button_w, button_h = 200, 60

        while self.is_not_baza and self.running:
            dt = self.clock.tick(Config.FPS) / 1000.0

            rows = 2 if self.view_2d_mode else 3
            cols = 2 if self.view_2d_mode else 3

            if len(self.matrix_inputs) != rows or len(self.matrix_inputs[0]) != cols:
                self.matrix_inputs = [["" for _ in range(cols)] for _ in range(rows)]

            mx, my = pygame.mouse.get_pos()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    self.is_not_baza = False
                    return

                elif event.type == pygame.VIDEORESIZE:
                    self.width, self.height = event.w, event.h
                    self.screen = pygame.display.set_mode((self.width, self.height),
                                                          DOUBLEBUF | OPENGL | RESIZABLE)
                    glViewport(0, 0, self.width, self.height)

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    toggle_bg_rect = pygame.Rect(20, self.height - 50, 120, 35)
                    if toggle_bg_rect.collidepoint(mx, my):
                        self.background_dark = not self.background_dark

                    # --- VEKTOROVÉ OPERÁCIE TLAČIDLÁ ---
                    vop_buttons = [
                        ("Sčítania", pygame.Rect(35, 60, button_w, button_h)),
                        ("Odčitanie", pygame.Rect(35, 60 + button_h + 5, button_w, button_h)),
                        ("Násobenie Konštantou", pygame.Rect(35, 60 + button_h * 2 + 10, button_w, button_h)),
                        ("Lineárna kombinácia", pygame.Rect(35, 60 + button_h * 3 + 15, button_w, button_h))
                    ]

                    for name, rect in vop_buttons:
                        if rect.collidepoint(mx, my):
                            rows_panel = 2 if self.view_2d_mode else 3

                            if name in ["Sčítania", "Odčitanie"]:
                                symbol = "+" if name == "Sčítania" else "-"
                                num_panels = 2
                                has_constant = False
                            elif name == "Násobenie Konštantou":
                                symbol = "×"
                                num_panels = 1
                                has_constant = True
                            elif name == "Lineárna kombinácia":
                                symbol = None
                                num_panels = 4  # c1, v1, c2, v2
                                has_constant = False

                                # Špeciálne rozloženie: konštanta - vektor - konštanta - vektor
                                constant_width = 60
                                vector_panel_width = Config.MATRIX_CELL_W + Config.MATRIX_GAP
                                symbol_width = 30

                                total_width = (constant_width + vector_panel_width + symbol_width) * 2
                                start_x = self.width // 2 - total_width // 2

                                panels = []
                                current_x = start_x

                                # c1
                                panels.append({
                                    "type": "constant",
                                    "rows": 1,
                                    "cols": 1,
                                    "values": [[""]],
                                    "active_cell": (0, 0),
                                    "x": current_x,
                                    "y": self.height // 2 - Config.MATRIX_CELL_H // 2,
                                    "label": "c1"
                                })
                                current_x += constant_width

                                # Symbol ×
                                current_x += 10

                                # v1
                                panels.append({
                                    "type": "vector",
                                    "rows": rows_panel,
                                    "cols": 1,
                                    "values": [["" for _ in range(1)] for _ in range(rows_panel)],
                                    "active_cell": (-1, -1),
                                    "x": current_x,
                                    "y": self.height // 2 - (
                                                rows_panel * (Config.MATRIX_CELL_H + Config.MATRIX_GAP)) // 2,
                                    "label": "v1"
                                })
                                current_x += vector_panel_width + 20

                                # Symbol +
                                current_x += 10

                                # c2
                                panels.append({
                                    "type": "constant",
                                    "rows": 1,
                                    "cols": 1,
                                    "values": [[""]],
                                    "active_cell": (-1, -1),
                                    "x": current_x,
                                    "y": self.height // 2 - Config.MATRIX_CELL_H // 2,
                                    "label": "c2"
                                })
                                current_x += constant_width

                                # Symbol ×
                                current_x += 10

                                # v2
                                panels.append({
                                    "type": "vector",
                                    "rows": rows_panel,
                                    "cols": 1,
                                    "values": [["" for _ in range(1)] for _ in range(rows_panel)],
                                    "active_cell": (-1, -1),
                                    "x": current_x,
                                    "y": self.height // 2 - (
                                                rows_panel * (Config.MATRIX_CELL_H + Config.MATRIX_GAP)) // 2,
                                    "label": "v2"
                                })

                                pending_input_panel = {
                                    "type": "vector",
                                    "operation": name,
                                    "symbol": None,  # Budeme kresliť manuálne
                                    "num_panels": len(panels),
                                    "has_constant": False,
                                    "panels": panels,
                                    "active_panel": 0,
                                    "is_linear_combination": True
                                }

                                break

                            else:  # Lineárna kombinácia
                                symbol = None
                                num_panels = 1
                                has_constant = False


                            panel_width = Config.MATRIX_CELL_W + Config.MATRIX_GAP
                            constant_width = 60 if has_constant else 0
                            symbol_width = 40 if (symbol and not has_constant) else (30 if has_constant else 0)
                            total_width = num_panels * panel_width + symbol_width + constant_width
                            start_x = self.width // 2 - total_width // 2

                            panels = []
                            if has_constant:
                                # Constant input panel
                                panels.append({
                                    "type": "constant",
                                    "rows": 1,
                                    "cols": 1,
                                    "values": [[""]],
                                    "active_cell": (0, 0),
                                    "x": start_x,
                                    "y": self.height // 2 - Config.MATRIX_CELL_H // 2
                                })
                                # Vector panel
                                panels.append({
                                    "type": "vector",
                                    "rows": rows_panel,
                                    "cols": 1,
                                    "values": [["" for _ in range(1)] for _ in range(rows_panel)],
                                    "active_cell": (-1, -1),
                                    "x": start_x + constant_width + symbol_width,
                                    "y": self.height // 2 - (
                                            rows_panel * (Config.MATRIX_CELL_H + Config.MATRIX_GAP)) // 2
                                })
                            else:
                                for i in range(num_panels):
                                    panels.append({
                                        "type": "vector",
                                        "rows": rows_panel,
                                        "cols": 1,
                                        "values": [["" for _ in range(1)] for _ in range(rows_panel)],
                                        "active_cell": (0, 0) if i == 0 else (-1, -1),
                                        "x": start_x + i * (panel_width + symbol_width),
                                        "y": self.height // 2 - (
                                                rows_panel * (Config.MATRIX_CELL_H + Config.MATRIX_GAP)) // 2
                                    })

                            pending_input_panel = {
                                "type": "vector",
                                "operation": name,
                                "symbol": symbol,
                                "num_panels": len(panels),
                                "has_constant": has_constant,
                                "panels": panels,
                                "active_panel": 0
                            }
                            break

                    # --- MATICOVÉ OPERÁCIE TLAČIDLÁ ---
                    mop_buttons = [
                        ("Sčítania", pygame.Rect(35, 380, button_w, button_h)),
                        ("Odčitanie", pygame.Rect(35, 380 + button_h + 5, button_w, button_h)),
                        ("Násobenie Konštantou", pygame.Rect(35, 380 + button_h * 2 + 10, button_w, button_h)),
                        ("Lineárna kombinácia", pygame.Rect(35, 380 + button_h * 3 + 15, button_w, button_h))
                    ]

                    for name, rect in mop_buttons:
                        if rect.collidepoint(mx, my):
                            rows_panel = cols_panel = 2 if self.view_2d_mode else 3

                            if name in ["Sčítania", "Odčitanie"]:
                                symbol = "+" if name == "Sčítania" else "-"
                                num_panels = 2
                                has_constant = False
                            elif name == "Násobenie Konštantou":
                                symbol = "×"
                                num_panels = 1
                                has_constant = True


                            elif name == "Lineárna kombinácia":
                                symbol = None
                                num_panels = 4  # c1, v1, c2, v2
                                has_constant = False

                                # Špeciálne rozloženie: konštanta - vektor - konštanta - vektor
                                constant_width = 60
                                vector_panel_width = Config.MATRIX_CELL_W + Config.MATRIX_GAP
                                symbol_width = 30

                                total_width = (constant_width + vector_panel_width + symbol_width) * 2
                                start_x = self.width // 2 - total_width // 2

                                panels = []
                                current_x = start_x

                                # c1
                                panels.append({
                                    "type": "constant",
                                    "rows": 1,
                                    "cols": 1,
                                    "values": [[""]],
                                    "active_cell": (0, 0),
                                    "x": current_x,
                                    "y": self.height // 2 - Config.MATRIX_CELL_H // 2,
                                    "label": "c1"
                                })
                                current_x += constant_width

                                # Symbol ×
                                current_x += 10

                                # v1
                                panels.append({
                                    "type": "matrix",
                                    "rows": rows_panel,
                                    "cols": cols_panel,
                                    "values": [["" for _ in range(cols_panel)] for _ in range(rows_panel)],
                                    "active_cell": (-1, -1),
                                    "x": current_x,
                                    "y": self.height // 2 - (
                                                rows_panel * (Config.MATRIX_CELL_H + Config.MATRIX_GAP)) // 2,
                                    "label": "mat1"
                                })
                                current_x += vector_panel_width + 20

                                # Symbol +
                                current_x += 10

                                # c2
                                panels.append({
                                    "type": "constant",
                                    "rows": 1,
                                    "cols": 1,
                                    "values": [[""]],
                                    "active_cell": (-1, -1),
                                    "x": current_x,
                                    "y": self.height // 2 - Config.MATRIX_CELL_H // 2,
                                    "label": "c2"
                                })
                                current_x += constant_width

                                # Symbol ×
                                current_x += 10

                                # v2
                                panels.append({
                                    "type": "matrix",
                                    "rows": rows_panel,
                                    "cols": cols_panel,
                                    "values": [["" for _ in range(cols_panel)] for _ in range(rows_panel)],
                                    "active_cell": (-1, -1),
                                    "x": current_x,
                                    "y": self.height // 2 - (
                                                rows_panel * (Config.MATRIX_CELL_H + Config.MATRIX_GAP)) // 2,
                                    "label": "mat2"
                                })

                                pending_input_panel = {
                                    "type": "matrix",
                                    "operation": name,
                                    "symbol": None,  # Budeme kresliť manuálne
                                    "num_panels": len(panels),
                                    "has_constant": False,
                                    "panels": panels,
                                    "active_panel": 0,
                                    "is_linear_combination": True
                                }

                                break

                            else:
                                symbol = None
                                num_panels = 1
                                has_constant = False

                            panel_width = cols_panel * (Config.MATRIX_CELL_W + Config.MATRIX_GAP)
                            constant_width = 60 if has_constant else 0
                            symbol_width = 40 if (symbol and not has_constant) else (30 if has_constant else 0)
                            total_width = num_panels * panel_width + symbol_width + constant_width
                            start_x = self.width // 2 - total_width // 2

                            panels = []
                            if has_constant:
                                panels.append({
                                    "type": "constant",
                                    "rows": 1,
                                    "cols": 1,
                                    "values": [[""]],
                                    "active_cell": (0, 0),
                                    "x": start_x,
                                    "y": self.height // 2 - Config.MATRIX_CELL_H // 2
                                })
                                panels.append({
                                    "type": "matrix",
                                    "rows": rows_panel,
                                    "cols": cols_panel,
                                    "values": [["" for _ in range(cols_panel)] for _ in range(rows_panel)],
                                    "active_cell": (-1, -1),
                                    "x": start_x + constant_width + symbol_width,
                                    "y": self.height // 2 - (
                                            rows_panel * (Config.MATRIX_CELL_H + Config.MATRIX_GAP)) // 2
                                })
                            else:
                                for i in range(num_panels):
                                    panels.append({
                                        "type": "matrix",
                                        "rows": rows_panel,
                                        "cols": cols_panel,
                                        "values": [["" for _ in range(cols_panel)] for _ in range(rows_panel)],
                                        "active_cell": (0, 0) if i == 0 else (-1, -1),
                                        "x": start_x + i * (panel_width + symbol_width),
                                        "y": self.height // 2 - (
                                                rows_panel * (Config.MATRIX_CELL_H + Config.MATRIX_GAP)) // 2
                                    })

                            pending_input_panel = {
                                "type": "matrix",
                                "operation": name,
                                "symbol": symbol,
                                "num_panels": len(panels),
                                "has_constant": has_constant,
                                "panels": panels,
                                "active_panel": 0
                            }
                            break

                    # --- Kliknutie na input panel ---
                    if pending_input_panel:
                        clicked_panel = False
                        for panel_idx, panel in enumerate(pending_input_panel["panels"]):
                            if panel.get("type") == "constant":
                                x = panel["x"]
                                y = panel["y"]
                                rect = pygame.Rect(x, y, 50, Config.MATRIX_CELL_H)
                                if rect.collidepoint(mx, my):
                                    for p in pending_input_panel["panels"]:
                                        p["active_cell"] = (-1, -1)
                                    pending_input_panel["active_panel"] = panel_idx
                                    panel["active_cell"] = (0, 0)
                                    self.matrix_active_cell = (-1, -1)
                                    clicked_panel = True
                                    break
                            else:
                                for r in range(panel["rows"]):
                                    for c in range(panel["cols"]):
                                        x = panel["x"] + c * (Config.MATRIX_CELL_W + Config.MATRIX_GAP)
                                        y = panel["y"] + r * (Config.MATRIX_CELL_H + Config.MATRIX_GAP)
                                        rect = pygame.Rect(x, y, Config.MATRIX_CELL_W, Config.MATRIX_CELL_H)
                                        if rect.collidepoint(mx, my):
                                            for p in pending_input_panel["panels"]:
                                                p["active_cell"] = (-1, -1)
                                            pending_input_panel["active_panel"] = panel_idx
                                            panel["active_cell"] = (r, c)
                                            self.matrix_active_cell = (-1, -1)
                                            clicked_panel = True
                                            break
                                    if clicked_panel:
                                        break
                            if clicked_panel:
                                break

                    # Check matrix cell clicks (original báza input)
                    if not pending_input_panel:
                        matrix_start_x = self.width - 155
                        matrix_start_y = 170
                        for r in range(rows):
                            for c in range(cols):
                                x = matrix_start_x + c * (Config.MATRIX_CELL_W + Config.MATRIX_GAP)
                                y = matrix_start_y + r * (Config.MATRIX_CELL_H + Config.MATRIX_GAP)
                                rect = pygame.Rect(x, y, Config.MATRIX_CELL_W, Config.MATRIX_CELL_H)
                                if rect.collidepoint(mx, my):
                                    self.matrix_active_cell = (r, c)
                                    break

                elif event.type == pygame.KEYDOWN:
                    # --- Spracovanie pending panel inputu ---
                    pending_handled = False
                    if pending_input_panel:
                        active_panel_idx = pending_input_panel["active_panel"]
                        panel = pending_input_panel["panels"][active_panel_idx]
                        r, c = panel["active_cell"]

                        if r != -1 and c != -1:
                            pending_handled = True
                            if event.key == pygame.K_RETURN:
                                # Skontroluj či sú všetky polia vyplnené
                                all_filled = True
                                for p in pending_input_panel["panels"]:
                                    for row in p["values"]:
                                        for val in row:
                                            if not val.strip():
                                                all_filled = False
                                                break

                                if all_filled:
                                    # Extrahuj konštantu ak existuje
                                    constant = None
                                    data_panels = []
                                    for p in pending_input_panel["panels"]:
                                        if p.get("type") == "constant":
                                            if constant is None:
                                                constant = []
                                            constant.append(float(p["values"][0][0]))
                                        else:
                                            data_panels.append(p)

                                    operation = pending_input_panel["operation"]
                                    result = None
                                    operands = []

                                    # VEKTOROVÉ OPERÁCIE
                                    if pending_input_panel["type"] == "vector":
                                        if operation == "Sčítania":
                                            vec1 = [float(v[0]) for v in data_panels[0]["values"]]
                                            vec2 = [float(v[0]) for v in data_panels[1]["values"]]
                                            result = [a + b for a, b in zip(vec1, vec2)]
                                            operands = [vec1, vec2]
                                            # NASTAVENIE ANIMÁCIE
                                            self.vector_manager.animation_controller.setup_operation(
                                                'add', operands, result
                                            )

                                        elif operation == "Odčitanie":
                                            vec1 = [float(v[0]) for v in data_panels[0]["values"]]
                                            vec2 = [float(v[0]) for v in data_panels[1]["values"]]
                                            result = [a - b for a, b in zip(vec1, vec2)]
                                            operands = [vec1, vec2]
                                            # NASTAVENIE ANIMÁCIE
                                            self.vector_manager.animation_controller.setup_operation(
                                                'subtract', operands, result
                                            )

                                        elif operation == "Násobenie Konštantou":
                                            vec = [float(v[0]) for v in data_panels[0]["values"]]
                                            result = [constant[0] * v for v in vec]
                                            operands = [vec]
                                            # NASTAVENIE ANIMÁCIE
                                            self.vector_manager.animation_controller.setup_operation(
                                                'scalar_mult', operands, result, constant[0]
                                            )

                                        elif operation == "Lineárna kombinácia":
                                            vec1 = [float(v[0]) for v in data_panels[0]["values"]]
                                            vec2 = [float(v[0]) for v in data_panels[1]["values"]]
                                            c1, c2 = constant[0], constant[1]
                                            result = [c1 * v1 + c2 * v2 for v1, v2 in zip(vec1, vec2)]
                                            operands = [vec1, vec2]
                                            self.vector_manager.animation_controller.setup_operation(
                                                'linear_combination', operands, result, [c1, c2]
                                            )

                                    # MATICOVÉ OPERÁCIE
                                    else:
                                        if operation == "Sčítania":
                                            mat1 = [[float(v) for v in row] for row in data_panels[0]["values"]]
                                            mat2 = [[float(v) for v in row] for row in data_panels[1]["values"]]
                                            result = [[a + b for a, b in zip(row1, row2)] for row1, row2 in
                                                      zip(mat1, mat2)]
                                            operands = [mat1, mat2]
                                            self.vector_manager.animation_controller.setup_operation(
                                                'add', operands, result
                                            )

                                        elif operation == "Odčitanie":
                                            mat1 = [[float(v) for v in row] for row in data_panels[0]["values"]]
                                            mat2 = [[float(v) for v in row] for row in data_panels[1]["values"]]
                                            result = [[a - b for a, b in zip(row1, row2)] for row1, row2 in
                                                      zip(mat1, mat2)]
                                            operands = [mat1, mat2]
                                            self.vector_manager.animation_controller.setup_operation(
                                                'subtract', operands, result
                                            )

                                        elif operation == "Násobenie Konštantou":
                                            mat = [[float(v) for v in row] for row in data_panels[0]["values"]]
                                            result = [[constant[0] * v for v in row] for row in mat]
                                            operands = [mat]
                                            self.vector_manager.animation_controller.setup_operation(
                                                'scalar_mult', operands, result, constant[0]
                                            )

                                        elif operation == "Lineárna kombinácia":
                                            mat1 = [[float(v) for v in row] for row in data_panels[0]["values"]]
                                            mat2 = [[float(v) for v in row] for row in data_panels[1]["values"]]
                                            c1, c2 = constant[0], constant[1]
                                            result = [[c1 * v1 + c2 * v2 for v1, v2 in zip(vec1, vec2)] for vec1, vec2
                                                      in zip(mat1, mat2)]
                                            operands = [mat1, mat2]
                                            self.vector_manager.animation_controller.setup_operation(
                                                'linear_combination', operands, result, [c1, c2]
                                            )

                                    # ✅ IHNEĎ PO NASTAVENÍ OPERÁCIE PRESUŇ KAMERU
                                    if not self.view_2d_mode:
                                        plane = self.vector_manager.animation_controller.current_plane
                                        if plane:
                                            max_val = self.get_max_from_vectors()
                                            distance = max(15.0, max_val * 3.5)

                                            if plane == "3D":
                                                optimal_view = self.vector_manager.animation_controller.compute_optimal_view_for_3d()
                                                if optimal_view:
                                                    self.camera.move_to_plane(plane, distance, custom_view=optimal_view)
                                                else:
                                                    self.camera.move_to_plane(plane, distance)
                                            else:
                                                self.camera.move_to_plane(plane, distance)

                                    pending_input_panel = None

                            elif event.key == pygame.K_BACKSPACE:
                                panel["values"][r][c] = panel["values"][r][c][:-1]
                            elif event.unicode.isdigit() or event.unicode in ".-":
                                panel["values"][r][c] += event.unicode

                    # --- Spracovanie originálnej bázy ---
                    if not pending_handled and self.matrix_active_cell != (-1, -1):
                        r, c = self.matrix_active_cell
                        if event.key == pygame.K_RETURN:
                            try:
                                matrix_values = []
                                for row in self.matrix_inputs:
                                    filled = [float(cell) for cell in row if cell.strip()]
                                    if filled:
                                        matrix_values.append(tuple(filled))

                                if matrix_values:
                                    mat_np = np.array(matrix_values).T
                                    if self.view_2d_mode:
                                        if mat_np.shape != (2, 2) or np.linalg.det(mat_np) == 0:
                                            print("Zadané vektory NEtvoria bázu 2D!")
                                            continue
                                    else:
                                        if mat_np.shape != (3, 3) or np.linalg.matrix_rank(mat_np) < 3:
                                            print("Zadané vektory NEtvoria bázu 3D!")
                                            continue

                                    self.saved_baza.append(tuple(matrix_values))
                                    print(f"Baza uložená: {tuple(matrix_values)}")
                                    self.matrix_inputs = [["" for _ in range(cols)] for _ in range(rows)]
                                    self.matrix_active_cell = (-1, -1)
                                    self.is_not_baza = False
                            except Exception as e:
                                print("Invalid matrix input:", e)

                        elif event.key == pygame.K_BACKSPACE:
                            self.matrix_inputs[r][c] = self.matrix_inputs[r][c][:-1]
                        elif event.unicode.isdigit() or event.unicode in ".-":
                            self.matrix_inputs[r][c] += event.unicode

            # === RENDER ===
            glMatrixMode(GL_PROJECTION)
            glLoadIdentity()
            glOrtho(0, self.width, self.height, 0, -1, 1)
            glMatrixMode(GL_MODELVIEW)
            glLoadIdentity()

            if self.background_dark:
                glClearColor(0.0, 0.0, 0.0, 1.0)
            else:
                glClearColor(1.0, 1.0, 1.0, 1.0)

            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            # Draw matrix input boxes (original báza)
            matrix_start_x = self.width - 155
            matrix_start_y = 170
            for r in range(rows):
                for c in range(cols):
                    x = matrix_start_x + c * (Config.MATRIX_CELL_W + Config.MATRIX_GAP)
                    y = matrix_start_y + r * (Config.MATRIX_CELL_H + Config.MATRIX_GAP)
                    active = (r, c) == self.matrix_active_cell
                    col_color_outline = Config.COL_COLORS_OUTLINE[c] if c < len(Config.COL_COLORS_OUTLINE) else (0.7,
                                                                                                                 0.7,
                                                                                                                 0.7)
                    self.ui_renderer.draw_input_box_3d(x, y, Config.MATRIX_CELL_W, Config.MATRIX_CELL_H,
                                                       self.matrix_inputs[r][c], active,
                                                       fill_color=(0.7, 0.7, 0.7),
                                                       fill_color_outline=col_color_outline)

            # --- Draw TLAČIDLÁ ---
            button_placement = pygame.Rect(35, 60, button_w, button_h)
            color1 = (1, 1, 1) if self.background_dark else (0, 0, 0)
            self.ui_renderer.draw_text_2d("Vektorove operácie: ", (button_placement.x - 15, button_placement.y - 40),
                                          color=color1, font_size=30)
            self.ui_renderer.draw_button_2d(button_placement.x, button_placement.y, button_placement.w,
                                            button_placement.h, "Sčítania", active=True)
            self.ui_renderer.draw_button_2d(button_placement.x, button_placement.y + button_h + 5, button_placement.w,
                                            button_placement.h, "Odčitanie", active=True)
            self.ui_renderer.draw_button_2d(button_placement.x, button_placement.y + button_h * 2 + 10,
                                            button_placement.w, button_placement.h, "Násobenie Konštantou", active=True)
            self.ui_renderer.draw_button_2d(button_placement.x, button_placement.y + button_h * 3 + 15,
                                            button_placement.w, button_placement.h, "Lineárna kombinácia", active=True)

            button_placement = pygame.Rect(35, 380, button_w, button_h)
            self.ui_renderer.draw_text_2d("Maticove operácie: ", (button_placement.x - 15, button_placement.y - 40),
                                          color=color1, font_size=30)
            self.ui_renderer.draw_button_2d(button_placement.x, button_placement.y, button_placement.w,
                                            button_placement.h, "Sčítania", active=True)
            self.ui_renderer.draw_button_2d(button_placement.x, button_placement.y + button_h + 5, button_placement.w,
                                            button_placement.h, "Odčitanie", active=True)
            self.ui_renderer.draw_button_2d(button_placement.x, button_placement.y + button_h * 2 + 10,
                                            button_placement.w, button_placement.h, "Násobenie Konštantou", active=True)
            self.ui_renderer.draw_button_2d(button_placement.x, button_placement.y + button_h * 3 + 15,
                                            button_placement.w, button_placement.h, "Lineárna kombinácia", active=True)

            # Draw toggle button
            toggle_bg_rect = pygame.Rect(20, self.height - 50, 120, 35)
            label = "Dark Mode" if self.background_dark else "Light Mode"
            self.ui_renderer.draw_button_2d(toggle_bg_rect.x, toggle_bg_rect.y,
                                            toggle_bg_rect.width, toggle_bg_rect.height,
                                            label, active=True)

            # --- Draw PENDING INPUT PANEL ---
            if pending_input_panel:

                # Špeciálne vykreslenie pre lineárnu kombináciu
                if pending_input_panel.get("is_linear_combination"):
                    color_symbol = (1, 1, 1) if self.background_dark else (0, 0, 0)

                    for panel_idx, panel in enumerate(pending_input_panel["panels"]):
                        is_active_panel = (panel_idx == pending_input_panel["active_panel"])

                        if panel.get("type") == "constant":
                            x = panel["x"]
                            y = panel["y"]
                            active = is_active_panel and panel["active_cell"] == (0, 0)
                            self.ui_renderer.draw_input_box_3d(x, y, 50, Config.MATRIX_CELL_H,
                                                               panel["values"][0][0], active)
                            # Label nad konštantou
                            label = panel.get("label", "")
                            if label:
                                self.ui_renderer.draw_text_2d(label, (x + 15, y - 25),
                                                              color=color_symbol, font_size=18)
                        else:
                            # Vektor
                            for r in range(panel["rows"]):
                                for c in range(panel["cols"]):
                                    x = panel["x"] + c * (Config.MATRIX_CELL_W + Config.MATRIX_GAP)
                                    y = panel["y"] + r * (Config.MATRIX_CELL_H + Config.MATRIX_GAP)
                                    active = is_active_panel and (r, c) == panel["active_cell"]
                                    self.ui_renderer.draw_input_box_3d(x, y, Config.MATRIX_CELL_W,
                                                                       Config.MATRIX_CELL_H,
                                                                       panel["values"][r][c], active)
                            # Label nad vektorom
                            label = panel.get("label", "")
                            if label:
                                x = panel["x"]
                                y = panel["y"] - 25
                                self.ui_renderer.draw_text_2d(label, (x + 10, y),
                                                              color=color_symbol, font_size=18)

                        # Symboly medzi panelmi
                        if panel_idx == 0:  # Po c1
                            symbol_x = panel["x"] + 50
                            symbol_y = panel["y"] + 5
                            self.ui_renderer.draw_text_2d("×", (symbol_x, symbol_y),
                                                          color=color_symbol, font_size=30)
                        elif panel_idx == 1:  # Po v1
                            symbol_x = panel["x"] + Config.MATRIX_CELL_W + 10
                            symbol_y = panel["y"] + (panel["rows"] * Config.MATRIX_CELL_H) // 2 - 10
                            self.ui_renderer.draw_text_2d("+", (symbol_x, symbol_y),
                                                          color=color_symbol, font_size=35)
                        elif panel_idx == 2:  # Po c2
                            symbol_x = panel["x"] + 50
                            symbol_y = panel["y"] + 5
                            self.ui_renderer.draw_text_2d("×", (symbol_x, symbol_y),
                                                          color=color_symbol, font_size=30)
                else:

                    for panel_idx, panel in enumerate(pending_input_panel["panels"]):
                        is_active_panel = (panel_idx == pending_input_panel["active_panel"])

                        if panel.get("type") == "constant":
                            x = panel["x"]
                            y = panel["y"]
                            active = is_active_panel and panel["active_cell"] == (0, 0)
                            self.ui_renderer.draw_input_box_3d(x, y, 50, Config.MATRIX_CELL_H, panel["values"][0][0],
                                                               active)
                        else:
                            for r in range(panel["rows"]):
                                for c in range(panel["cols"]):
                                    x = panel["x"] + c * (Config.MATRIX_CELL_W + Config.MATRIX_GAP)
                                    y = panel["y"] + r * (Config.MATRIX_CELL_H + Config.MATRIX_GAP)
                                    active = is_active_panel and (r, c) == panel["active_cell"]
                                    self.ui_renderer.draw_input_box_3d(x, y, Config.MATRIX_CELL_W, Config.MATRIX_CELL_H,
                                                                       panel["values"][r][c], active)

                        # Draw symbol between panels
                        if pending_input_panel["symbol"]:
                            if pending_input_panel.get("has_constant") and panel_idx == 0:
                                symbol_x = panel["x"] + 60
                                symbol_y = panel["y"]
                                color_symbol = (1, 1, 1) if self.background_dark else (0, 0, 0)
                                self.ui_renderer.draw_text_2d(pending_input_panel["symbol"], (symbol_x, symbol_y),
                                                              color=color_symbol, font_size=40)
                            elif not pending_input_panel.get("has_constant") and panel_idx < len(
                                    pending_input_panel["panels"]) - 1:
                                symbol_x = panel["x"] + panel["cols"] * (Config.MATRIX_CELL_W + Config.MATRIX_GAP) + 10
                                symbol_y = panel["y"] + (panel["rows"] * (Config.MATRIX_CELL_H + Config.MATRIX_GAP)) // 2
                                color_symbol = (1, 1, 1) if self.background_dark else (0, 0, 0)
                                self.ui_renderer.draw_text_2d(pending_input_panel["symbol"], (symbol_x, symbol_y),
                                                              color=color_symbol, font_size=40)

            pygame.display.flip()

    def handle_events(self):
        """Spracuje všetky eventy"""
        mx, my = pygame.mouse.get_pos()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            elif event.type == pygame.VIDEORESIZE:
                self.width, self.height = event.w, event.h
                self.screen = pygame.display.set_mode((self.width, self.height),
                                                      DOUBLEBUF | OPENGL | RESIZABLE)
                glViewport(0, 0, self.width, self.height)
                self.camera.width = self.width
                self.camera.height = self.height

            elif event.type == pygame.KEYDOWN:
                self.handle_keypress(event)

            elif event.type == pygame.MOUSEBUTTONDOWN:
                self.handle_mouse_down(event, mx, my)

            elif event.type == pygame.MOUSEBUTTONUP:
                self.handle_mouse_up(event)

            elif event.type == pygame.MOUSEMOTION:
                self.handle_mouse_motion(event)

    def handle_keypress(self, event):
        """Spracuje stlačenie klávesy - ROZŠÍRENÉ"""
        # KROKOVANIE ANIMÁCIE - má prioritu
        if event.key == pygame.K_SPACE:
            if self.vector_manager.animation_controller.current_operation:
                # ✅ NAJPRV KROKUJ
                self.vector_manager.animation_controller.next_step()

                # ✅ POTOM PRESUŇ KAMERU PRE NOVÝ KROK
                if not self.view_2d_mode:
                    plane = self.vector_manager.animation_controller.current_plane
                    if plane:
                        max_val = self.get_max_from_vectors()
                        distance = max(15.0, max_val * 3.5)

                        if plane == "3D":
                            # Vypočítaj optimálny pohľad pre aktuálny krok
                            optimal_view = self.vector_manager.animation_controller.compute_optimal_view_for_current_step()
                            if optimal_view:
                                self.camera.move_to_plane(plane, distance, custom_view=optimal_view)
                            else:
                                # Fallback na pôvodnú metódu
                                optimal_view = self.vector_manager.animation_controller.compute_optimal_view_for_3d()
                                if optimal_view:
                                    self.camera.move_to_plane(plane, distance, custom_view=optimal_view)
                                else:
                                    self.camera.move_to_plane(plane, distance)
                        else:
                            # Pre XY, XZ, YZ roviny použijem štandardný pohľad
                            self.camera.move_to_plane(plane, distance)
            return

        # BACKSPACE má špeciálnu logiku - ak nie je aktívny žiadny input, použije sa na krokovanie
        if event.key == pygame.K_BACKSPACE:
            # Kontrola či je aktívny nejaký input field
            if not any([
                self.input_handler.show_input_active,
                self.input_handler.show_multiplication_active,
                self.input_handler.show_matrix_size_active,
                self.input_handler.show_random_range_active,
                self.matrix_active_cell != (-1, -1)
            ]):
                # Ak nie je aktívny input, použije sa na krokovanie
                if self.vector_manager.animation_controller.current_operation:
                    # ✅ NAJPRV KROKUJ SPÄŤ
                    self.vector_manager.animation_controller.prev_step()

                    # ✅ POTOM PRESUŇ KAMERU PRE NOVÝ KROK
                    if not self.view_2d_mode:
                        plane = self.vector_manager.animation_controller.current_plane
                        if plane:
                            max_val = self.get_max_from_vectors()
                            distance = max(15.0, max_val * 3.5)

                            if plane == "3D":
                                optimal_view = self.vector_manager.animation_controller.compute_optimal_view_for_current_step()
                                if optimal_view:
                                    self.camera.move_to_plane(plane, distance, custom_view=optimal_view)
                                else:
                                    optimal_view = self.vector_manager.animation_controller.compute_optimal_view_for_3d()
                                    if optimal_view:
                                        self.camera.move_to_plane(plane, distance, custom_view=optimal_view)
                                    else:
                                        self.camera.move_to_plane(plane, distance)
                            else:
                                self.camera.move_to_plane(plane, distance)
                    return

        if event.key == pygame.K_c:
            self.vector_manager.animation_controller.clear()
            return

        if event.key == pygame.K_v:
            if self.vector_manager.animation_controller.current_operation:
                plane = self.vector_manager.animation_controller.current_plane
                if plane and not self.view_2d_mode:
                    max_val = self.get_max_from_vectors()
                    distance = max(15.0, max_val * 3.5)

                    # ✅ POUŽIJEM AKTUÁLNY KROK
                    if plane == "3D":
                        optimal_view = self.vector_manager.animation_controller.compute_optimal_view_for_current_step()
                        if optimal_view:
                            self.camera.move_to_plane(plane, distance, custom_view=optimal_view)
                        else:
                            self.camera.move_to_plane(plane, distance)
                    else:
                        self.camera.move_to_plane(plane, distance)
            return

        # Potom existujúce checky...
        if event.key == pygame.K_ESCAPE:
            self.running = False
        elif event.key == pygame.K_g:
            self.grid_mode = (self.grid_mode + 1) % 4
        elif event.key == pygame.K_a:
            self.show_axes = not self.show_axes
        elif event.key == pygame.K_r:
            self.camera.reset()
            self.vector_manager.reset_animations()
        elif event.key == pygame.K_m:
            self.show_matrix_input = not self.show_matrix_input
        elif event.key == pygame.K_i:
            self.input_handler.show_input_active = True

        # NEW: Vector input handling
        if self.input_handler.show_input_active:
            if event.key == pygame.K_RETURN:
                vec = self.input_handler.parse_input()
                if vec:
                    self.vector_manager.add_vector(vec)
                self.input_handler.input_text = ""
                self.input_handler.show_input_active = False
            elif event.key == pygame.K_BACKSPACE:
                self.input_handler.input_text = self.input_handler.input_text[:-1]
            else:
                self.input_handler.input_text += event.unicode
            return

        # NEW: Multiplication input handling
        if self.input_handler.show_multiplication_active:
            if event.key == pygame.K_RETURN:
                multiplier = self.input_handler.parse_multiplication()
                if multiplier:
                    self.vector_manager.multiply_selected_vector(multiplier)
                self.input_handler.multiplication_input = ""
                self.input_handler.show_multiplication_active = False
            elif event.key == pygame.K_BACKSPACE:
                self.input_handler.multiplication_input = self.input_handler.multiplication_input[:-1]
            else:
                self.input_handler.multiplication_input += event.unicode
            return

        # Matrix size input
        if self.input_handler.show_matrix_size_active:
            if event.key == pygame.K_RETURN:
                self.input_handler.show_matrix_size_active = False
            else:
                self.input_handler.handle_text_input(event, 'matrix_size')
            return

        # Random range input
        if self.input_handler.show_random_range_active:
            if event.key == pygame.K_RETURN:
                self.input_handler.show_random_range_active = False
            else:
                self.input_handler.handle_text_input(event, 'random_range')
            return

        # Matrix input
        if self.show_matrix_input and self.matrix_active_cell != (-1, -1):
            r, c = self.matrix_active_cell
            if event.key == pygame.K_RETURN:
                # Try to add matrix as vector
                try:
                    rows = 2 if self.view_2d_mode else 3
                    cols = 2 if self.view_2d_mode else 3
                    matrix_values = []
                    for c_idx in range(cols):
                        column_cells = [self.matrix_inputs[r_idx][c_idx] for r_idx in range(rows)
                                        if self.matrix_inputs[r_idx][c_idx].strip()]
                        if column_cells:
                            float_col = [float(cell) for cell in column_cells]
                            matrix_values.append(tuple(float_col))

                    if matrix_values:
                        self.vector_manager.add_vector(tuple(matrix_values))
                        self.matrix_inputs = [["" for _ in range(cols)] for _ in range(rows)]
                        self.matrix_active_cell = (-1, -1)
                except Exception as e:
                    print("Invalid matrix input:", e)

            elif event.key == pygame.K_BACKSPACE:
                self.matrix_inputs[r][c] = self.matrix_inputs[r][c][:-1]
            elif event.unicode.isdigit() or event.unicode in ".-":
                self.matrix_inputs[r][c] += event.unicode

    def handle_mouse_down(self, event, mx, my):
        """Spracuje stlačenie myši - ROZŠÍRENÉ"""
        # Toggle background
        toggle_bg_rect = pygame.Rect(20, self.height - 50, 120, 35)
        if toggle_bg_rect.collidepoint(mx, my):
            self.background_dark = not self.background_dark
            if self.background_dark:
                glClearColor(0.0, 0.0, 0.0, 1.0)
            else:
                glClearColor(1.0, 1.0, 1.0, 1.0)
            return

        # NEW: Dropdown click
        if self.dropdown_rect.collidepoint(mx, my):
            self.dropdown_open = not self.dropdown_open
            return
        elif self.dropdown_open:
            dropdown_item_height = 25
            for i in range(len(self.vector_manager.animated_vectors)):
                item_rect = pygame.Rect(
                    self.dropdown_rect.x,
                    self.dropdown_rect.y + self.dropdown_rect.height + i * dropdown_item_height,
                    self.dropdown_rect.width,
                    dropdown_item_height
                )
                if item_rect.collidepoint(mx, my):
                    self.vector_manager.selected_vector_index = i
                    self.dropdown_open = False
                    return
            self.dropdown_open = False

        # NEW: Multiplication input click
        multiplication_rect = pygame.Rect(self.width - 160, 330, 140, 30)
        if multiplication_rect.collidepoint(mx, my):
            self.input_handler.show_multiplication_active = True
            self.input_handler.show_input_active = False
            return

        # NEW: Matrix size input
        matrix_size_rect = pygame.Rect(self.width - 210, 290, 60, 30)
        if matrix_size_rect.collidepoint(mx, my):
            self.input_handler.show_matrix_size_active = True
            self.input_handler.show_random_range_active = False
            return

        # NEW: Random range input
        random_range_rect = pygame.Rect(self.width - 140, 290, 60, 30)
        if random_range_rect.collidepoint(mx, my):
            self.input_handler.show_random_range_active = True
            self.input_handler.show_matrix_size_active = False
            return

        # NEW: Random button
        random_button_rect = pygame.Rect(self.width - 70, 290, 60, 30)
        if random_button_rect.collidepoint(mx, my):
            self.generate_random_matrix()
            return

        # NEW: Draw Vector button
        draw_button_rect = pygame.Rect(self.width - 160, 120, 140, 35)
        if draw_button_rect.collidepoint(mx, my):
            self.vectors_as_points = not self.vectors_as_points
            if not self.vectors_as_points:
                self.vector_manager.reset_animations()
            return

        # Matrix input clicks
        if self.show_matrix_input:
            rows = 2 if self.view_2d_mode else 3
            cols = 2 if self.view_2d_mode else 3
            matrix_start_x = self.width - 155
            matrix_start_y = 170
            for r in range(rows):
                for c in range(cols):
                    x = matrix_start_x + c * (Config.MATRIX_CELL_W + Config.MATRIX_GAP)
                    y = matrix_start_y + r * (Config.MATRIX_CELL_H + Config.MATRIX_GAP)
                    rect = pygame.Rect(x, y, Config.MATRIX_CELL_W, Config.MATRIX_CELL_H)
                    if rect.collidepoint(mx, my):
                        self.matrix_active_cell = (r, c)
                        return

        # Camera controls
        if event.button == 1:
            self.camera.rotating = not self.view_2d_mode
            self.camera.panning = self.view_2d_mode
            self.camera.last_mouse = event.pos
        elif event.button == 3:
            self.camera.panning = True
            self.camera.last_mouse = event.pos
        elif event.button == 4:  # scroll up
            self.camera.zoom_in(self.view_2d_mode)
        elif event.button == 5:  # scroll down
            self.camera.zoom_out(self.view_2d_mode)

    def generate_random_matrix(self):
        """Generuje náhodnú maticu"""
        try:
            # OPRAVA: Pridaj validáciu vstupu
            if not self.input_handler.matrix_size_input or ',' not in self.input_handler.matrix_size_input:
                print("Please enter size as: rows,cols (e.g., 2,2 or 3,3)")
                return

            rows, cols = map(int, self.input_handler.matrix_size_input.split(','))
            if not (1 <= rows <= 3 and 1 <= cols <= 3):
                print("Only up to 3x3 supported")
                return

            low, high = (0, 9)
            if self.input_handler.random_range_input and ',' in self.input_handler.random_range_input:
                parts = self.input_handler.random_range_input.split(',')
                if len(parts) == 2:
                    try:
                        low, high = map(float, parts)
                    except ValueError:
                        print("Invalid range format")
                        return

            self.matrix_inputs = [[str(random.randint(int(low), int(high))) for _ in range(cols)]
                                  for _ in range(rows)]

            # Fill empty cells
            while len(self.matrix_inputs) < 3:
                self.matrix_inputs.append([""] * 3)
            for r in range(len(self.matrix_inputs)):
                while len(self.matrix_inputs[r]) < 3:
                    self.matrix_inputs[r].append("")

            print(f"Generated random {rows}x{cols} matrix.")
        except Exception as e:
            print("Error generating matrix:", e)

    def handle_mouse_up(self, event):
        """Spracuje uvoľnenie myši"""
        if event.button in (1, 3):
            self.camera.rotating = False
            self.camera.panning = False

    def handle_mouse_motion(self, event):
        """Spracuje pohyb myši"""
        dx = event.pos[0] - self.camera.last_mouse[0]
        dy = event.pos[1] - self.camera.last_mouse[1]
        self.camera.last_mouse = event.pos

        if self.view_2d_mode and self.camera.panning:
            self.camera.handle_panning_2d(dx, dy)
        elif not self.view_2d_mode:
            if self.camera.rotating:
                self.camera.handle_rotation(dx, dy)
            if self.camera.panning:
                self.camera.handle_panning_3d(dx, dy)

    def update(self, dt):
        """Aktualizuje stav aplikácie - ROZŠÍRENÉ"""
        # NEW: Update vector animations
        self.vector_manager.update_animations(dt)

        # NOVÉ: Animuj kameru ak je v 3D režime
        if not self.view_2d_mode and self.camera.animating_to_plane:
            self.camera.animate_to_plane(
                self.vector_manager.animation_controller.current_plane,
                dt
            )

    def render(self):
        """Vykreslí scénu"""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        if self.view_2d_mode:
            self.render_2d()
        else:
            self.render_3d()

        # Draw UI overlay
        self.render_ui()

        pygame.display.flip()

    def render_2d(self):
        """Vykreslí 2D scénu"""
        self.camera.setup_2d_projection()

        # Draw grid
        if self.grid_mode % 2 == 0:
            self.grid_renderer.draw_grid_2d(self.camera.ortho_scale,
                                            self.camera.pan_offset_x,
                                            self.camera.pan_offset_y,
                                            self.width, self.height)

        # Draw axes
        if self.show_axes:
            self.axes_renderer.draw_axes_2d(self.camera.ortho_scale,
                                            self.camera.pan_offset_x,
                                            self.camera.pan_offset_y,
                                            self.width, self.height,
                                            self.ui_renderer)

        # Draw vectors
        color = (0, 0, 0) if not self.background_dark else (1, 1, 1)
        self.draw_vectors_2d(color)

    def render_3d(self):
        """Vykreslí 3D scénu"""
        self.camera.setup_3d_projection()

        length_xyz = max(10, self.get_max_from_vectors())

        # Draw grid/planes
        if self.grid_mode == 1:
            self.grid_renderer.draw_planes_3d(length_xyz)
        elif self.grid_mode == 2:
            self.grid_renderer.draw_planes_3d(length_xyz, colored=True)
        elif self.grid_mode == 3:
            self.grid_renderer.draw_grid_3d(length_xyz)

        # Draw axes
        if self.show_axes:
            cam_pos = self.camera.get_position()
            self.axes_renderer.draw_axes_3d(length_xyz, cam_pos, self.vector_renderer)

        # Draw vectors
        if self.grid_mode in (1, 2):
            color = (0, 0, 0)
        else:
            color = (0, 0, 0) if not self.background_dark else (1, 1, 1)
        self.draw_vectors_3d(color)

    def draw_vectors_2d(self, color):
        """Vykreslí vektory v 2D - POUŽÍVA animation_controller"""
        if self.vector_manager.animation_controller.current_operation:
            vectors_to_draw = self.vector_manager.animation_controller.get_vectors_to_draw()

            # OPRAVA: Zoraď vektory - najprv priesvitné (alpha < 1.0), potom plne viditeľné
            vectors_sorted = sorted(vectors_to_draw, key=lambda v: v.get('alpha', 1.0))

            if self.vectors_as_points:
                for v in vectors_sorted:
                    vec = v['vec']
                    offset = v.get('offset', [0, 0])
                    v_color = v.get('color', color)
                    v_alpha = v.get('alpha', 1.0)
                    base_radius = 0.07
                    radius = base_radius * self.camera.ortho_scale / 6.5

                    # Vypni depth write pre priesvitné
                    if v_alpha < 1.0:
                        glDepthMask(GL_FALSE)

                    if isinstance(vec[0], (int, float)):
                        pos = [vec[0] + offset[0], vec[1] + offset[1] if len(vec) > 1 else offset[1], 0.2]
                        self.vector_renderer.draw_circle_2d(pos, radius=radius, color=v_color, alpha=v_alpha)
                    else:
                        # Matica - každý riadok je bod
                        row_offsets = v.get('row_offsets', None)

                        for i, row in enumerate(vec):
                            x = row[0]
                            y = row[1] if len(row) > 1 else 0

                            if row_offsets and i < len(row_offsets):
                                ox = row_offsets[i][0]
                                oy = row_offsets[i][1] if len(row_offsets[i]) > 1 else 0
                            else:
                                ox = offset[0]
                                oy = offset[1] if len(offset) > 1 else 0

                            pos = [x + ox, y + oy, 0.2]
                            self.vector_renderer.draw_circle_2d(pos, radius=radius, color=v_color, alpha=v_alpha)

                    # Zapni depth write späť
                    if v_alpha < 1.0:
                        glDepthMask(GL_TRUE)

            else:
                # Vykresli vektory s offsetom
                for v in vectors_sorted:
                    vec = v['vec']
                    offset = v.get('offset', [0, 0])
                    v_color = v.get('color', color)
                    v_alpha = v.get('alpha', 1.0)

                    # Vypni depth write pre priesvitné
                    if v_alpha < 1.0:
                        glDepthMask(GL_FALSE)

                    if isinstance(vec[0], (int, float)):
                        # Jednoduchý vektor
                        x = vec[0]
                        y = vec[1] if len(vec) > 1 else 0
                        ox = offset[0]
                        oy = offset[1] if len(offset) > 1 else 0

                        glLineWidth(6)

                        # Podpora pre alpha v 2D
                        if v_alpha < 1.0:
                            glEnable(GL_BLEND)
                            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
                            glColor4f(v_color[0], v_color[1], v_color[2], v_alpha)
                        else:
                            glColor3f(*v_color)

                        glBegin(GL_LINES)
                        glVertex2f(ox, oy)
                        glVertex2f(ox + x, oy + y)
                        glEnd()

                        # Šípka len ak je vektor dosť veľký
                        vec_length = math.sqrt(x * x + y * y)
                        if vec_length > 0.1:
                            angle = math.atan2(y, x)
                            arrow_size = 0.3
                            px, py = ox + x, oy + y
                            glBegin(GL_LINES)
                            glVertex2f(px, py)
                            glVertex2f(px - arrow_size * math.cos(angle - 0.3),
                                       py - arrow_size * math.sin(angle - 0.3))
                            glVertex2f(px, py)
                            glVertex2f(px - arrow_size * math.cos(angle + 0.3),
                                       py - arrow_size * math.sin(angle + 0.3))
                            glEnd()

                        if v_alpha < 1.0:
                            glDisable(GL_BLEND)

                    else:
                        # Matica - každý riadok je vektor
                        row_offsets = v.get('row_offsets', None)

                        for i, row in enumerate(vec):
                            x = row[0]
                            y = row[1] if len(row) > 1 else 0

                            # Použiť row_offsets ak existujú
                            if row_offsets and i < len(row_offsets):
                                ox = row_offsets[i][0]
                                oy = row_offsets[i][1] if len(row_offsets[i]) > 1 else 0
                            else:
                                ox = offset[0]
                                oy = offset[1] if len(offset) > 1 else 0

                            glLineWidth(6)

                            if v_alpha < 1.0:
                                glEnable(GL_BLEND)
                                glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
                                glColor4f(v_color[0], v_color[1], v_color[2], v_alpha)
                            else:
                                glColor3f(*v_color)

                            glBegin(GL_LINES)
                            glVertex2f(ox, oy)
                            glVertex2f(ox + x, oy + y)
                            glEnd()

                            # Šípka len ak je vektor dosť veľký
                            vec_length = math.sqrt(x * x + y * y)
                            if vec_length > 0.1:
                                angle = math.atan2(y, x)
                                arrow_size = 0.3
                                px, py = ox + x, oy + y
                                glBegin(GL_LINES)
                                glVertex2f(px, py)
                                glVertex2f(px - arrow_size * math.cos(angle - 0.3),
                                           py - arrow_size * math.sin(angle - 0.3))
                                glVertex2f(px, py)
                                glVertex2f(px - arrow_size * math.cos(angle + 0.3),
                                           py - arrow_size * math.sin(angle + 0.3))
                                glEnd()

                            if v_alpha < 1.0:
                                glDisable(GL_BLEND)

                    # Zapni depth write späť
                    if v_alpha < 1.0:
                        glDepthMask(GL_TRUE)

        else:
            # Pôvodná logika
            if self.vectors_as_points:
                for v in self.vector_manager.animated_vectors:
                    vec = v['vec']
                    base_radius = 0.07
                    radius = base_radius * self.camera.ortho_scale / 6.5

                    if isinstance(vec[0], (int, float)):
                        self.vector_renderer.draw_circle_2d([vec[0], vec[1] if len(vec) > 1 else 0, 0.2],
                                                            radius=radius, color=color)
                    else:
                        for row in vec:
                            self.vector_renderer.draw_circle_2d([row[0], row[1] if len(row) > 1 else 0, 0.2],
                                                                radius=radius, color=color)
            else:
                VectorRenderer.draw_vectors_2d_animated(
                    self.vector_manager.animated_vectors,
                    self.clock.get_time() / 1000.0,
                    ortho_scale=self.camera.ortho_scale,
                    color=color
                )

    def draw_vectors_3d(self, color):
        """Vykreslí vektory v 3D - SPRÁVNE PORADIE PRE PRIESVITNOSŤ"""
        # Ak je aktívna animácia, použij vektory z controllera
        if self.vector_manager.animation_controller.current_operation:
            vectors_to_draw = self.vector_manager.animation_controller.get_vectors_to_draw()
            cam_pos = self.camera.get_position()

            if self.vectors_as_points:
                # OPRAVA: Rozdeľ vektory na plne viditeľné a priesvitné
                opaque_vectors = [v for v in vectors_to_draw if v.get('alpha', 1.0) >= 1.0]
                transparent_vectors = [v for v in vectors_to_draw if v.get('alpha', 1.0) < 1.0]

                distance = self.camera.distance

                # Najprv vykresli plne viditeľné vektory
                for v in opaque_vectors:
                    vec = v['vec']
                    offset = v.get('offset', [0, 0, 0])
                    v_color = v.get('color', color)
                    v_alpha = v.get('alpha', 1.0)

                    base_radius = 0.07
                    radius = base_radius * math.sqrt(distance / 7.0)
                    radius = max(radius, 0.03)

                    if isinstance(vec[0], (int, float)):
                        vec3d = [vec[i] + offset[i] if i < len(vec) else offset[i] for i in range(3)]
                        self.vector_renderer.draw_sphere(vec3d, radius=radius, color=v_color, alpha=v_alpha)
                    else:
                        row_offsets = v.get('row_offsets', None)
                        for i, row in enumerate(vec):
                            base = [row[j] if j < len(row) else 0 for j in range(3)]
                            if row_offsets:
                                ro = row_offsets[i]
                                vec3d = [base[j] + ro[j] for j in range(3)]
                            else:
                                vec3d = [base[j] + offset[j] for j in range(3)]
                            self.vector_renderer.draw_sphere(vec3d, radius=radius, color=v_color, alpha=v_alpha)

                # Potom vykresli priesvitné vektory s vypnutým depth write
                glDepthMask(GL_FALSE)
                for v in transparent_vectors:
                    vec = v['vec']
                    offset = v.get('offset', [0, 0, 0])
                    v_color = v.get('color', color)
                    v_alpha = v.get('alpha', 1.0)

                    base_radius = 0.07
                    radius = base_radius * math.sqrt(distance / 7.0)
                    radius = max(radius, 0.03)

                    if isinstance(vec[0], (int, float)):
                        vec3d = [vec[i] + offset[i] if i < len(vec) else offset[i] for i in range(3)]
                        self.vector_renderer.draw_sphere(vec3d, radius=radius, color=v_color, alpha=v_alpha)
                    else:
                        row_offsets = v.get('row_offsets', None)
                        for i, row in enumerate(vec):
                            base = [row[j] if j < len(row) else 0 for j in range(3)]
                            if row_offsets:
                                ro = row_offsets[i]
                                vec3d = [base[j] + ro[j] for j in range(3)]
                            else:
                                vec3d = [base[j] + offset[j] for j in range(3)]
                            self.vector_renderer.draw_sphere(vec3d, radius=radius, color=v_color, alpha=v_alpha)
                glDepthMask(GL_TRUE)

            else:
                # OPRAVA: Rovnaká logika pre šípky
                opaque_vectors = [v for v in vectors_to_draw if v.get('alpha', 1.0) >= 1.0]
                transparent_vectors = [v for v in vectors_to_draw if v.get('alpha', 1.0) < 1.0]

                # Najprv plne viditeľné
                for v in opaque_vectors:
                    vec = v['vec']
                    offset = v.get('offset', [0, 0, 0])
                    v_color = v.get('color', color)
                    v_alpha = v.get('alpha', 1.0)

                    if isinstance(vec[0], (int, float)):
                        glPushMatrix()
                        glTranslatef(offset[0], offset[1], offset[2])

                        # Skráť vektor aby čiara nezasahovala do šípky
                        end_point = (vec[0], vec[1] if len(vec) > 1 else 0, vec[2] if len(vec) > 2 else 0)
                        vec_array = np.array(end_point)
                        length = np.linalg.norm(vec_array)
                        delta = 0.15

                        if length > delta:
                            vec_short = vec_array * (length - delta) / length
                        else:
                            vec_short = vec_array

                        glLineWidth(6)
                        glColor3f(*v_color)
                        glBegin(GL_LINES)
                        glVertex3f(0, 0, 0)
                        glVertex3f(vec_short[0], vec_short[1], vec_short[2])
                        glEnd()

                        self.vector_renderer.draw_conehead_3d((0, 0, 0), end_point,
                                                              color=v_color, size=0.7,
                                                              camera_pos=cam_pos, radius=0.25,
                                                              alpha=v_alpha)
                        glPopMatrix()
                    else:
                        row_offsets = v.get('row_offsets', None)
                        for i, row in enumerate(vec):
                            glPushMatrix()
                            if row_offsets:
                                ro = row_offsets[i]
                                glTranslatef(ro[0], ro[1], ro[2])
                            else:
                                glTranslatef(offset[0], offset[1], offset[2])

                            # Skráť vektor aby čiara nezasahovala do šípky
                            end_point = (row[0] if len(row) > 0 else 0,
                                         row[1] if len(row) > 1 else 0,
                                         row[2] if len(row) > 2 else 0)
                            vec_array = np.array(end_point)
                            length = np.linalg.norm(vec_array)
                            delta = 0.15

                            if length > delta:
                                vec_short = vec_array * (length - delta) / length
                            else:
                                vec_short = vec_array

                            glLineWidth(6)
                            glColor3f(*v_color)
                            glBegin(GL_LINES)
                            glVertex3f(0, 0, 0)
                            glVertex3f(vec_short[0], vec_short[1], vec_short[2])
                            glEnd()

                            self.vector_renderer.draw_conehead_3d((0, 0, 0), end_point,
                                                                  color=v_color, size=0.7,
                                                                  camera_pos=cam_pos, radius=0.25,
                                                                  alpha=v_alpha)
                            glPopMatrix()

                # Potom priesvitné s vypnutým depth write
                glDepthMask(GL_FALSE)
                for v in transparent_vectors:
                    vec = v['vec']
                    offset = v.get('offset', [0, 0, 0])
                    v_color = v.get('color', color)
                    v_alpha = v.get('alpha', 1.0)

                    if isinstance(vec[0], (int, float)):
                        glPushMatrix()
                        glTranslatef(offset[0], offset[1], offset[2])

                        # Skráť vektor aby čiara nezasahovala do šípky
                        end_point = (vec[0], vec[1] if len(vec) > 1 else 0, vec[2] if len(vec) > 2 else 0)
                        vec_array = np.array(end_point)
                        length = np.linalg.norm(vec_array)
                        delta = 0.15

                        if length > delta:
                            vec_short = vec_array * (length - delta) / length
                        else:
                            vec_short = vec_array

                        glLineWidth(6)
                        glEnable(GL_BLEND)
                        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
                        glColor4f(v_color[0], v_color[1], v_color[2], v_alpha)
                        glBegin(GL_LINES)
                        glVertex3f(0, 0, 0)
                        glVertex3f(vec_short[0], vec_short[1], vec_short[2])
                        glEnd()
                        glDisable(GL_BLEND)

                        self.vector_renderer.draw_conehead_3d((0, 0, 0), end_point,
                                                              color=v_color, size=0.7,
                                                              camera_pos=cam_pos, radius=0.25,
                                                              alpha=v_alpha)
                        glPopMatrix()
                    else:
                        row_offsets = v.get('row_offsets', None)
                        for i, row in enumerate(vec):
                            glPushMatrix()
                            if row_offsets:
                                ro = row_offsets[i]
                                glTranslatef(ro[0], ro[1], ro[2])
                            else:
                                glTranslatef(offset[0], offset[1], offset[2])

                            # Skráť vektor aby čiara nezasahovala do šípky
                            end_point = (row[0] if len(row) > 0 else 0,
                                         row[1] if len(row) > 1 else 0,
                                         row[2] if len(row) > 2 else 0)
                            vec_array = np.array(end_point)
                            length = np.linalg.norm(vec_array)
                            delta = 0.15

                            if length > delta:
                                vec_short = vec_array * (length - delta) / length
                            else:
                                vec_short = vec_array

                            glLineWidth(6)
                            glEnable(GL_BLEND)
                            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
                            glColor4f(v_color[0], v_color[1], v_color[2], v_alpha)
                            glBegin(GL_LINES)
                            glVertex3f(0, 0, 0)
                            glVertex3f(vec_short[0], vec_short[1], vec_short[2])
                            glEnd()
                            glDisable(GL_BLEND)

                            self.vector_renderer.draw_conehead_3d((0, 0, 0), end_point,
                                                                  color=v_color, size=0.7,
                                                                  camera_pos=cam_pos, radius=0.25,
                                                                  alpha=v_alpha)
                            glPopMatrix()
                glDepthMask(GL_TRUE)

        else:
            # Pôvodná logika pre normálne vektory (bez animácie)
            if self.vectors_as_points:
                cam_pos = self.camera.get_position()
                distance = self.camera.distance

                for v in self.vector_manager.animated_vectors:
                    vec = v['vec']
                    v_color = v.get('color', color)
                    v_alpha = v.get('alpha', 1.0)
                    base_radius = 0.07
                    radius = base_radius * math.sqrt(distance / 7.0)
                    radius = max(radius, 0.03)

                    if isinstance(vec[0], (int, float)):
                        vec3d = list(vec) + [0] * (3 - len(vec))
                        self.vector_renderer.draw_sphere(vec3d, radius=radius, color=v_color, alpha=v_alpha)
                    else:
                        row_offsets = v.get("row_offsets", None)
                        offset = v.get("offset", [0, 0, 0])

                        for i, row in enumerate(vec):
                            base = list(row) + [0] * (3 - len(row))
                            if row_offsets:
                                ro = row_offsets[i]
                                vec3d = [base[j] + ro[j] for j in range(3)]
                            else:
                                vec3d = [base[j] + offset[j] for j in range(3)]
                            self.vector_renderer.draw_sphere(vec3d, radius=radius, color=v_color, alpha=v_alpha)
            else:
                # Nešanimované vektory so šípkami
                cam_pos = self.camera.get_position()

                for v in self.vector_manager.animated_vectors:
                    vec = v['vec']
                    v_color = v.get('color', color)
                    v_alpha = v.get('alpha', 1.0)
                    offset = v.get('offset', [0, 0, 0])

                    if isinstance(vec[0], (int, float)):
                        glPushMatrix()
                        glTranslatef(offset[0], offset[1], offset[2])

                        # Skráť vektor aby čiara nezasahovala do šípky
                        end_point = (vec[0], vec[1] if len(vec) > 1 else 0, vec[2] if len(vec) > 2 else 0)
                        vec_array = np.array(end_point)
                        length = np.linalg.norm(vec_array)
                        delta = 0.15

                        if length > delta:
                            vec_short = vec_array * (length - delta) / length
                        else:
                            vec_short = vec_array

                        glLineWidth(6)
                        if v_alpha < 1.0:
                            glEnable(GL_BLEND)
                            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
                            glColor4f(v_color[0], v_color[1], v_color[2], v_alpha)
                        else:
                            glColor3f(*v_color)

                        glBegin(GL_LINES)
                        glVertex3f(0, 0, 0)
                        glVertex3f(vec_short[0], vec_short[1], vec_short[2])
                        glEnd()

                        if v_alpha < 1.0:
                            glDisable(GL_BLEND)

                        self.vector_renderer.draw_conehead_3d((0, 0, 0), end_point,
                                                              color=v_color, size=0.7,
                                                              camera_pos=cam_pos, radius=0.25,
                                                              alpha=v_alpha)
                        glPopMatrix()
                    else:
                        row_offsets = v.get('row_offsets', None)
                        for i, row in enumerate(vec):
                            glPushMatrix()
                            if row_offsets:
                                ro = row_offsets[i]
                                glTranslatef(ro[0], ro[1], ro[2])
                            else:
                                glTranslatef(offset[0], offset[1], offset[2])

                            # Skráť vektor aby čiara nezasahovala do šípky
                            end_point = (row[0] if len(row) > 0 else 0,
                                         row[1] if len(row) > 1 else 0,
                                         row[2] if len(row) > 2 else 0)
                            vec_array = np.array(end_point)
                            length = np.linalg.norm(vec_array)
                            delta = 0.15

                            if length > delta:
                                vec_short = vec_array * (length - delta) / length
                            else:
                                vec_short = vec_array

                            glLineWidth(6)
                            if v_alpha < 1.0:
                                glEnable(GL_BLEND)
                                glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
                                glColor4f(v_color[0], v_color[1], v_color[2], v_alpha)
                            else:
                                glColor3f(*v_color)

                            glBegin(GL_LINES)
                            glVertex3f(0, 0, 0)
                            glVertex3f(vec_short[0], vec_short[1], vec_short[2])
                            glEnd()

                            if v_alpha < 1.0:
                                glDisable(GL_BLEND)

                            self.vector_renderer.draw_conehead_3d((0, 0, 0), end_point,
                                                                  color=v_color, size=0.7,
                                                                  camera_pos=cam_pos, radius=0.25,
                                                                  alpha=v_alpha)
                            glPopMatrix()

    def render_ui(self):
        """Vykreslí UI overlay - ROZŠÍRENÉ S LEPŠÍMI POPISMI"""
        # Setup 2D projection for UI
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, self.width, self.height, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        glDisable(GL_DEPTH_TEST)

        # Draw toggle background button
        toggle_bg_rect = pygame.Rect(20, self.height - 50, 120, 35)
        label = "Dark Mode" if self.background_dark else "Light Mode"
        self.ui_renderer.draw_button_2d(toggle_bg_rect.x, toggle_bg_rect.y,
                                        toggle_bg_rect.width, toggle_bg_rect.height,
                                        label, active=True)

        # Draw vector input field
        if self.input_handler.show_input_active:
            self.ui_renderer.draw_input_box_3d(
                10, 10, 200, 30,
                self.input_handler.input_text,
                active=True
            )

        # Draw dropdown for vector selection
        if self.vector_manager.animated_vectors:
            self.enhanced_ui.draw_dropdown(
                self.vector_manager.animated_vectors,
                self.vector_manager.selected_vector_index,
                self.dropdown_rect,
                open=self.dropdown_open,
                width=self.width,
                height=self.height
            )

        # Draw multiplication input
        if self.input_handler.show_multiplication_active:
            multiplication_rect = pygame.Rect(self.width - 160, 330, 140, 30)
            self.ui_renderer.draw_input_box_3d(
                multiplication_rect.x, multiplication_rect.y,
                multiplication_rect.w, multiplication_rect.h,
                self.input_handler.multiplication_input,
                active=True
            )

        # Draw matrix size input
        if self.input_handler.show_matrix_size_active:
            matrix_size_rect = pygame.Rect(self.width - 210, 290, 60, 30)
            self.ui_renderer.draw_input_box_3d(
                matrix_size_rect.x, matrix_size_rect.y,
                matrix_size_rect.width, matrix_size_rect.height,
                self.input_handler.matrix_size_input,
                active=True
            )

        # Draw random range input
        if self.input_handler.show_random_range_active:
            random_range_rect = pygame.Rect(self.width - 140, 290, 60, 30)
            self.ui_renderer.draw_input_box_3d(
                random_range_rect.x, random_range_rect.y,
                random_range_rect.width, random_range_rect.height,
                self.input_handler.random_range_input,
                active=True
            )

        # Draw random button
        random_button_rect = pygame.Rect(self.width - 70, 290, 60, 30)
        self.ui_renderer.draw_button_2d(
            random_button_rect.x, random_button_rect.y,
            random_button_rect.width, random_button_rect.height,
            "Random", active=True
        )

        # Draw vector toggle button
        draw_button_rect = pygame.Rect(self.width - 160, 120, 140, 35)
        button_label = "As Points" if self.vectors_as_points else "As Vectors"
        self.ui_renderer.draw_button_2d(
            draw_button_rect.x, draw_button_rect.y,
            draw_button_rect.width, draw_button_rect.height,
            button_label, active=True
        )

        # Draw matrix input if enabled
        if self.show_matrix_input:
            rows = 2 if self.view_2d_mode else 3
            cols = 2 if self.view_2d_mode else 3
            matrix_start_x = self.width - 155
            matrix_start_y = 170

            for r in range(rows):
                for c in range(cols):
                    x = matrix_start_x + c * (Config.MATRIX_CELL_W + Config.MATRIX_GAP)
                    y = matrix_start_y + r * (Config.MATRIX_CELL_H + Config.MATRIX_GAP)
                    active = (r, c) == self.matrix_active_cell
                    col_color_outline = Config.COL_COLORS_OUTLINE[c] if c < len(Config.COL_COLORS_OUTLINE) else (0.7,
                                                                                                                 0.7,
                                                                                                                 0.7)

                    self.ui_renderer.draw_input_box_3d(
                        x, y, Config.MATRIX_CELL_W, Config.MATRIX_CELL_H,
                        self.matrix_inputs[r][c], active,
                        fill_color=(0.7, 0.7, 0.7),
                        fill_color_outline=col_color_outline
                    )

        # Draw basis matrix if exists
        if self.saved_baza and not self.view_2d_mode:
            self.enhanced_ui.draw_basis_matrix_top_right(
                self.saved_baza[0],
                self.width, self.height,
                font_size=20,
                color=(0, 0, 0) if not self.background_dark else (1, 1, 1)
            )

        # INDIKÁTOR KROKOVANIA - OPRAVENÉ POPISY PRE MATICE
        if self.vector_manager.animation_controller.current_operation:
            ctrl = self.vector_manager.animation_controller
            math_info = ctrl.get_math_display_info()

            if math_info:
                self.render_math_display(math_info)


            is_matrix = ctrl._is_matrix(ctrl.operands[0]) if ctrl.operands else False

            op_names = {
                'add': 'Sčítanie matíc' if is_matrix else 'Sčítanie vektorov',
                'subtract': 'Odčítanie matíc' if is_matrix else 'Odčítanie vektorov',
                'scalar_mult': 'Násobenie matice konštantou' if is_matrix else 'Násobenie vektora konštantou',
                'linear_combination': 'Lineárna kombinácia vektorov'
            }
            op_name = op_names.get(ctrl.operation_type, ctrl.operation_type)

            step_text = f"{op_name} - Krok {ctrl.current_step}/{ctrl.max_steps}"

            # Popisy krokov
            if is_matrix:
                descriptions = {
                    'add': [
                        "Zobrazenie matíc A (oranžová) a B (modrá)",
                        "Vektory matice B posunuté na konce vektorov matice A",
                        "Výsledná matica A + B (zelená)"
                    ],
                    'subtract': [
                        "Zobrazenie matíc A (oranžová) a B (modrá)",
                        "Negácia matice B → -B (ružová)",
                        "Vektory matice -B posunuté na konce vektorov matice A",
                        "Výsledná matica A - B (zelená)"
                    ],
                    'scalar_mult': [
                        "Pôvodná matica A (oranžová)",
                        f"Výsledná matica {ctrl.constant}·A (zelená)" if ctrl.constant else "Výsledná matica (zelená)"
                    ]
                }
            else:
                descriptions = {
                    'add': [
                        "Zobrazenie vektorov v1 (oranžová) a v2 (modrá)",
                        "Vektor v2 posunutý na koniec v1",
                        "Výsledný vektor v1 + v2 (zelená)"
                    ],
                    'subtract': [
                        "Zobrazenie vektorov v1 (oranžová) a v2 (modrá)",
                        "Negácia vektora v2 → -v2 (ružová)",
                        "Vektor -v2 posunutý na koniec v1",
                        "Výsledný vektor v1 - v2 (zelená)"
                    ],
                    'scalar_mult': [
                        "Pôvodný vektor v (oranžová)",
                        f"Výsledný vektor {ctrl.constant}·v (zelená)" if ctrl.constant else "Výsledný vektor (zelená)"
                    ],
                    'linear_combination': [
                        "Zobrazenie vektorov v1 a v2",
                        f"Vykreslenie c1·v1",
                        f"Vykreslenie c2·v2",
                        f"Pridanie c2·v2",
                        "Výsledný vektor c1.v1 + c2.v2 (zelená)"
                    ]
                }

            desc_list = descriptions.get(ctrl.operation_type, [""])
            desc = desc_list[ctrl.current_step] if ctrl.current_step < len(desc_list) else ""

            # Vykreslí indikátor
            text_color = (1, 1, 1) if self.background_dark else (0, 0, 0)
            self.ui_renderer.draw_text_2d(step_text, (self.width // 2 - 150, 20),
                                          color=text_color, font_size=24)
            self.ui_renderer.draw_text_2d(desc, (self.width // 2 - 200, 50),
                                          color=text_color, font_size=18)
            self.ui_renderer.draw_text_2d("SPACE=ďalší | BACKSPACE=späť | C=zrušiť",
                                          (self.width // 2 - 180, 75),
                                          color=text_color, font_size=16)
        if self.vector_manager.animation_controller.current_operation:
            plane = self.vector_manager.animation_controller.current_plane
            if plane and not self.view_2d_mode:
                plane_text = f"Rovina: {plane}"
                text_color = (1, 1, 1) if self.background_dark else (0, 0, 0)
                self.ui_renderer.draw_text_2d(plane_text, (self.width // 2 - 180, 95),
                                              color=text_color, font_size=16)

                if not self.camera.animating_to_plane:
                    self.ui_renderer.draw_text_2d("V=zobraz rovinu",
                                                  (self.width // 2 - 180, 115),
                                                  color=text_color, font_size=14)

        glEnable(GL_DEPTH_TEST)

        # Restore projection
        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)

    def _is_matrix(self, data):
        """Check if data is a matrix (list of lists)"""
        if not isinstance(data, (list, tuple)):
            return False
        if len(data) == 0:
            return False
        return isinstance(data[0], (list, tuple))

    def render_math_display(self, math_info):
        """Vykreslí matematické zobrazenie operácie v ľavom hornom rohu"""
        math_renderer = MathRenderer()

        start_x = 20
        start_y = 80
        current_x = start_x

        op_type = math_info['operation']
        step = math_info['step']
        operands = math_info['operands']
        result = math_info['result']
        constant = math_info['constant']

        is_matrix_op = self._is_matrix(operands[0]) if operands else False

        text_color = (1, 1, 1) if self.background_dark else (0, 0, 0)

        # SČÍTANIE
        if op_type == 'add':
            if step == 0:
                # v1 + v2
                current_x = math_renderer.draw_vector_math(current_x, start_y, operands[0],
                                                           color=(0.8, 0.8, 0.8), font_size=22)
                current_x = math_renderer.draw_operator(current_x, start_y, "+",
                                                        color=text_color, font_size=28)
                current_x = math_renderer.draw_vector_math(current_x, start_y, operands[1],
                                                           color=(0.8, 0.8, 0.8), font_size=22)

            elif step == 1:
                # v1 (highlight) + v2
                current_x = math_renderer.draw_vector_math(current_x, start_y, operands[0],
                                                           color=(1, 0.5, 0), font_size=22, highlight=True)
                current_x = math_renderer.draw_operator(current_x, start_y, "+",
                                                        color=text_color, font_size=28)
                current_x = math_renderer.draw_vector_math(current_x, start_y, operands[1],
                                                           color=(0, 0.5, 1), font_size=22, highlight=True)

                # Druhý riadok: rozklad operácie
                current_x = start_x
                start_y += len(operands[0]) * 27 + 20
                current_x = math_renderer.draw_vector_component_operation(
                    current_x, start_y, operands[0], operands[1], "+",
                    color=(0.8, 0.8, 0.8), font_size=20)

            elif step == 2:
                # v1 + v2 = result
                current_x = math_renderer.draw_vector_math(current_x, start_y, operands[0],
                                                           color=(1, 0.5, 0), font_size=22, highlight=True)
                current_x = math_renderer.draw_operator(current_x, start_y, "+",
                                                        color=text_color, font_size=28)
                current_x = math_renderer.draw_vector_math(current_x, start_y, operands[1],
                                                           color=(0, 0.5, 1), font_size=22, highlight=True)
                current_x = math_renderer.draw_equals(current_x, start_y,
                                                      color=text_color, font_size=28)
                current_x = math_renderer.draw_vector_math(current_x, start_y, result,
                                                           color= self.RESULT_COLOR, font_size=22, highlight=True)

        # ODČÍTANIE
        elif op_type == 'subtract':
            if step == 0:
                current_x = math_renderer.draw_vector_math(current_x, start_y, operands[0],
                                                           color=(0.8, 0.8, 0.8), font_size=22)
                current_x = math_renderer.draw_operator(current_x, start_y, "-",
                                                        color=text_color, font_size=28)
                current_x = math_renderer.draw_vector_math(current_x, start_y, operands[1],
                                                           color=(0.8, 0.8, 0.8), font_size=22)

            elif step == 1:
                # v1 - v2 = v1 + (-v2)
                if is_matrix_op:
                    negated = [[-x for x in row] for row in operands[1]]
                else:
                    negated = [-x for x in operands[1]]
                current_x = math_renderer.draw_vector_math(current_x, start_y, operands[0],
                                                           color=(1, 0.5, 0), font_size=22, highlight=True)
                current_x = math_renderer.draw_operator(current_x, start_y, "+",
                                                        color=text_color, font_size=28)
                current_x = math_renderer.draw_vector_math(current_x, start_y, negated,
                                                           color=(1, 0, 0.5), font_size=22, highlight=True)

            elif step == 2:
                if is_matrix_op:
                    negated = [[-x for x in row] for row in operands[1]]
                else:
                    negated = [-x for x in operands[1]]
                current_x = math_renderer.draw_vector_math(current_x, start_y, operands[0],
                                                           color=(1, 0.5, 0), font_size=22, highlight=True)
                current_x = math_renderer.draw_operator(current_x, start_y, "+",
                                                        color=text_color, font_size=28)
                current_x = math_renderer.draw_vector_math(current_x, start_y, negated,
                                                           color=(1, 0, 0.5), font_size=22, highlight=True)

                # Druhý riadok
                current_x = start_x
                start_y += len(operands[0]) * 27 + 20
                current_x = math_renderer.draw_vector_component_operation(
                    current_x, start_y, operands[0], negated, "+",
                    color=(0.8, 0.8, 0.8), font_size=20)

            elif step == 3:
                if is_matrix_op:
                    negated = [[-x for x in row] for row in operands[1]]
                else:
                    negated = [-x for x in operands[1]]
                current_x = math_renderer.draw_vector_math(current_x, start_y, operands[0],
                                                           color=(1, 0.5, 0), font_size=22, highlight=True)
                current_x = math_renderer.draw_operator(current_x, start_y, "+",
                                                        color=text_color, font_size=28)
                current_x = math_renderer.draw_vector_math(current_x, start_y, negated,
                                                           color=(1, 0, 0.5), font_size=22, highlight=True)
                current_x = math_renderer.draw_equals(current_x, start_y,
                                                      color=text_color, font_size=28)
                current_x = math_renderer.draw_vector_math(current_x, start_y, result,
                                                           color= self.RESULT_COLOR, font_size=22, highlight=True)

        # NÁSOBENIE KONŠTANTOU
        elif op_type == 'scalar_mult':
            if step == 0:
                current_x = math_renderer.draw_scalar(current_x, start_y, constant,
                                                      color=text_color, font_size=26)
                current_x = math_renderer.draw_operator(current_x - 10, start_y, "·",
                                                        color=text_color, font_size=28)
                current_x = math_renderer.draw_vector_math(current_x, start_y, operands[0],
                                                           color=(1, 0.5, 0), font_size=22, highlight=True)

            elif step == 1:
                current_x = math_renderer.draw_scalar(current_x, start_y, constant,
                                                      color=text_color, font_size=26)
                current_x = math_renderer.draw_operator(current_x - 10, start_y, "·",
                                                        color=text_color, font_size=28)
                current_x = math_renderer.draw_vector_math(current_x, start_y, operands[0],
                                                           color=(1, 0.5, 0), font_size=22, highlight=True)
                current_x = math_renderer.draw_equals(current_x, start_y,
                                                      color=text_color, font_size=28)
                current_x = math_renderer.draw_vector_math(current_x, start_y, result,
                                                           color= self.RESULT_COLOR, font_size=22, highlight=True)

        # LINEÁRNA KOMBINÁCIA
        elif op_type == 'linear_combination':
            c1, c2 = constant if constant else [1, 1]

            if step == 0:
                # c1·v1 + c2·v2
                current_x = math_renderer.draw_scalar(current_x, start_y, c1,
                                                      color=text_color, font_size=24)
                current_x = math_renderer.draw_operator(current_x - 10, start_y, "·",
                                                        color=text_color, font_size=26)
                current_x = math_renderer.draw_vector_math(current_x, start_y, operands[0],
                                                           color=(0.8, 0.8, 0.8), font_size=20)
                current_x = math_renderer.draw_operator(current_x, start_y, "+",
                                                        color=text_color, font_size=26)
                current_x = math_renderer.draw_scalar(current_x, start_y, c2,
                                                      color=text_color, font_size=24)
                current_x = math_renderer.draw_operator(current_x - 10, start_y, "·",
                                                        color=text_color, font_size=26)
                current_x = math_renderer.draw_vector_math(current_x, start_y, operands[1],
                                                           color=(0.8, 0.8, 0.8), font_size=20)

            elif step == 1:
                # Highlight c1·v1
                if is_matrix_op:
                    scaled_v1 = [[c1 * x for x in row] for row in operands[0]]
                else:
                    scaled_v1 = [c1 * x for x in operands[0]]
                current_x = math_renderer.draw_vector_math(current_x, start_y, scaled_v1,
                                                           color=(1, 0.5, 0), font_size=20, highlight=True)
                current_x = math_renderer.draw_operator(current_x, start_y, "+",
                                                        color=text_color, font_size=26)
                current_x = math_renderer.draw_scalar(current_x, start_y, c2,
                                                      color=text_color, font_size=24)
                current_x = math_renderer.draw_operator(current_x - 10, start_y, "·",
                                                        color=text_color, font_size=26)
                current_x = math_renderer.draw_vector_math(current_x, start_y, operands[1],
                                                           color=(0.8, 0.8, 0.8), font_size=20)

            elif step == 2:
                # Oba vektory
                if is_matrix_op:
                    scaled_v1 = [[c1 * x for x in row] for row in operands[0]]
                    scaled_v2 = [[c2 * x for x in row] for row in operands[1]]
                else:
                    scaled_v1 = [c1 * x for x in operands[0]]
                    scaled_v2 = [c2 * x for x in operands[1]]

                current_x = math_renderer.draw_vector_math(current_x, start_y, scaled_v1,
                                                           color=(1, 0.5, 0), font_size=20, highlight=True)
                current_x = math_renderer.draw_operator(current_x, start_y, "+",
                                                        color=text_color, font_size=26)
                current_x = math_renderer.draw_vector_math(current_x, start_y, scaled_v2,
                                                           color=(0, 0.5, 1), font_size=20, highlight=True)

            elif step == 3:
                # c1.v1 (highlight) + c2.v2
                if is_matrix_op:
                    scaled_v1 = [[c1 * x for x in row] for row in operands[0]]
                    scaled_v2 = [[c2 * x for x in row] for row in operands[1]]
                else:
                    scaled_v1 = [c1 * x for x in operands[0]]
                    scaled_v2 = [c2 * x for x in operands[1]]
                current_x = math_renderer.draw_vector_math(current_x, start_y, scaled_v1,
                                                           color=(1, 0.5, 0), font_size=20, highlight=True)
                current_x = math_renderer.draw_operator(current_x, start_y, "+",
                                                        color=text_color, font_size=26)
                current_x = math_renderer.draw_vector_math(current_x, start_y, scaled_v2,
                                                           color=(0, 0.5, 1), font_size=20, highlight=True)

                # Druhý riadok: rozklad operácie
                current_x = start_x
                start_y += len(scaled_v1) * 27 + 20
                current_x = math_renderer.draw_vector_component_operation(
                    current_x, start_y, scaled_v1, scaled_v2, "+",
                    color=(0.8, 0.8, 0.8), font_size=20)


            elif step == 4:
                # Výsledok
                if is_matrix_op:
                    scaled_v1 = [[c1 * x for x in row] for row in operands[0]]
                    scaled_v2 = [[c2 * x for x in row] for row in operands[1]]
                else:
                    scaled_v1 = [c1 * x for x in operands[0]]
                    scaled_v2 = [c2 * x for x in operands[1]]
                current_x = math_renderer.draw_vector_math(current_x, start_y, scaled_v1,
                                                           color=(1, 0.5, 0), font_size=20, highlight=True)
                current_x = math_renderer.draw_operator(current_x, start_y, "+",
                                                        color=text_color, font_size=26)
                current_x = math_renderer.draw_vector_math(current_x, start_y, scaled_v2,
                                                           color=(0, 0.5, 1), font_size=20, highlight=True)
                current_x = math_renderer.draw_equals(current_x, start_y,
                                                      color=text_color, font_size=26)
                current_x = math_renderer.draw_vector_math(current_x, start_y, result,
                                                           color=(0, 1, 0), font_size=20, highlight=True)

    def get_max_from_vectors(self):
        """Vráti najväčšiu hodnotu zo všetkých vektorov - S OFFSETMI"""
        max_val = 10.0

        # Skontroluj vektory z animation controllera (operácie)
        if self.vector_manager.animation_controller.current_operation:
            ctrl = self.vector_manager.animation_controller

            # Všetky body ktoré sa v animácii objavia
            all_points = []

            # Pridaj origin
            all_points.append([0, 0, 0])

            # Všetky operandy a ich možné pozície
            for operand in ctrl.operands:
                if isinstance(operand[0], (int, float)):
                    all_points.append(list(operand) + [0] * (3 - len(operand)))
                else:
                    for row in operand:
                        all_points.append(list(row) + [0] * (3 - len(row)))

            # Výsledok
            if ctrl.result:
                if isinstance(ctrl.result[0], (int, float)):
                    all_points.append(list(ctrl.result) + [0] * (3 - len(ctrl.result)))
                else:
                    for row in ctrl.result:
                        all_points.append(list(row) + [0] * (3 - len(row)))

            # Pri sčítaní/odčítaní druhý vektor môže byť posunutý
            if ctrl.operation_type in ['add', 'subtract'] and len(ctrl.operands) >= 2:
                offset = ctrl.operands[0]
                second = ctrl.operands[1]

                if isinstance(offset[0], (int, float)) and isinstance(second[0], (int, float)):
                    # Koncový bod druhého vektora posunutého na koniec prvého
                    combined = [offset[i] + second[i] if i < len(offset) and i < len(second) else
                                (offset[i] if i < len(offset) else second[i] if i < len(second) else 0)
                                for i in range(3)]
                    all_points.append(combined)
                elif not isinstance(offset[0], (int, float)) and not isinstance(second[0], (int, float)):
                    # Maticová operácia - každý riadok druhej matice posunutý
                    for i, row1 in enumerate(offset):
                        if i < len(second):
                            row2 = second[i]
                            combined = [row1[j] + row2[j] if j < len(row1) and j < len(row2) else
                                        (row1[j] if j < len(row1) else row2[j] if j < len(row2) else 0)
                                        for j in range(3)]
                            all_points.append(combined)

            # Nájdi maximum zo všetkých súradníc
            if all_points:
                max_val = max(max(abs(coord) for coord in point) for point in all_points)

        # Skontroluj aj bežné vektory
        if self.vector_manager.animated_vectors:
            for v in self.vector_manager.animated_vectors:
                vec = v['vec']
                if isinstance(vec[0], (int, float)):
                    current_max = max(abs(x) for x in vec)
                else:
                    current_max = max(max(abs(x) for x in row) for row in vec)
                max_val = max(max_val, current_max)

        return max_val

    def run(self):
        """Hlavná slučka aplikácie"""
        # Run startup screen
        self.run_startup_screen()

        if not self.running:
            pygame.quit()
            return

        # Run baza input
        self.run_baza_input()

        if not self.running:
            pygame.quit()
            return

        # Main game loop
        while self.running:
            dt = self.clock.tick(Config.FPS) / 1000.0

            self.handle_events()
            self.update(dt)
            self.render()

        pygame.quit()


def main():
    """Hlavná funkcia"""
    app = Application()
    app.run()


if __name__ == "__main__":
    main()