
from numpy import array, cross, linalg
from math import sin, cos, pi


class Colors:
    """Moderná farebná paleta"""
    # Dark theme
    DARK_BG = (0.06, 0.07, 0.10)
    DARK_SURFACE = (0.11, 0.12, 0.16)
    DARK_CARD = (0.14, 0.16, 0.20)
    DARK_HOVER = (0.20, 0.22, 0.28)

    # Light theme
    LIGHT_BG = (0.94, 0.95, 0.97)
    LIGHT_SURFACE = (1.0, 1.0, 1.0)
    LIGHT_CARD = (0.98, 0.98, 0.99)
    LIGHT_HOVER = (0.90, 0.92, 0.95)

    # Accent
    PRIMARY = (0.30, 0.55, 0.95)
    PRIMARY_HOVER = (0.40, 0.65, 1.0)
    SECONDARY = (0.60, 0.35, 0.90)

    # Vectors
    VECTOR_A = (1.0, 0.55, 0.15)
    VECTOR_B = (0.20, 0.60, 0.95)
    VECTOR_RESULT = (0.70, 0.25, 0.85)
    VECTOR_NEG = (0.95, 0.30, 0.55)

    # Text
    TEXT_DARK = (0.92, 0.94, 0.96)
    TEXT_DARK_SEC = (0.55, 0.60, 0.68)
    TEXT_LIGHT = (0.10, 0.12, 0.16)
    TEXT_LIGHT_SEC = (0.40, 0.45, 0.52)

    # Borders
    BORDER_DARK = (0.25, 0.28, 0.34)
    BORDER_LIGHT = (0.80, 0.82, 0.86)

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
        self.span_controller = SpanAnimationController()

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


        # NOVÉ: Update span controller
        self.span_controller.update(dt)

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
    def draw_vectors_2d_animated2(vectors, dt, line_width=6, arrow_fraction=0.15,
                                 speed=0.5, ortho_scale=6.0, color=(1, 1, 1)):
        """Animované kreslenie 2D vektorov - OPTIMALIZOVANÉ"""
        if not vectors:
            return

        glPushAttrib(GL_LINE_BIT | GL_CURRENT_BIT)
        glLineWidth(line_width)

        zoom_scale = ortho_scale / 6.0
        base_arrow_size = line_width * arrow_fraction * zoom_scale
        arrow_angle = 0.3

        for v in vectors:
            vec = v['vec']
            v_color = v.get('color', color)

            # Single vector
            if isinstance(vec[0], (int, float)):
                progress = v.get('progress', 0.0)
                progress = min(progress + dt * speed, 1.0)
                v['progress'] = progress

                x, y = vec[0], vec[1] if len(vec) > 1 else 0
                px, py = x * progress, y * progress

                glColor3f(*v_color)
                glBegin(GL_LINES)
                glVertex2f(0, 0)
                glVertex2f(px, py)
                glEnd()

                # Arrowhead
                vec_len = math.hypot(px, py)
                if vec_len > 0.01:
                    angle = math.atan2(y, x)
                    arrow_size = min(base_arrow_size, vec_len * 0.2)

                    glBegin(GL_LINES)
                    glVertex2f(px, py)
                    glVertex2f(px - arrow_size * math.cos(angle - arrow_angle),
                               py - arrow_size * math.sin(angle - arrow_angle))
                    glVertex2f(px, py)
                    glVertex2f(px - arrow_size * math.cos(angle + arrow_angle),
                               py - arrow_size * math.sin(angle + arrow_angle))
                    glEnd()

            # Matrix of vectors
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

                    glColor3f(*v_color)
                    glBegin(GL_LINES)
                    glVertex2f(0, 0)
                    glVertex2f(px, py)
                    glEnd()

                    vec_len = math.hypot(px, py)
                    if vec_len > 0.01:
                        angle = math.atan2(y, x)
                        arrow_size = min(base_arrow_size, vec_len * 0.2)

                        glBegin(GL_LINES)
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
    COL_COLORS_OUTLINE = [Colors.VECTOR_A, Colors.VECTOR_B, Colors.SECONDARY]


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


# ============= OPTIMALIZOVANÁ GRID RENDERER =============
class GridRenderer:
    """Optimalizovaný renderer pre mriežky a plochy"""

    # Cache pre opakované výpočty
    _plane_tangents_cache = {}

    @staticmethod
    def draw_filled_plane(normal, center=(0, 0, 0), size=10.0, color=(0.5, 0.5, 0.0), alpha=1.0, transparent=False):
        """Nakreslí vyplnenú rovinu definovanú normálou"""
        normal = np.array(normal, dtype=np.float32)
        norm_length = np.linalg.norm(normal)
        if norm_length < 1e-6:
            return
        normal = normal / norm_length

        # Nájdi dva kolmé vektory v rovine
        if abs(normal[2]) < 0.9:
            ref = np.array([0, 0, 1], dtype=np.float32)
        else:
            ref = np.array([1, 0, 0], dtype=np.float32)

        tangent1 = np.cross(normal, ref)
        tangent1 = tangent1 / np.linalg.norm(tangent1)
        tangent2 = np.cross(normal, tangent1)
        tangent2 = tangent2 / np.linalg.norm(tangent2)

        center = np.array(center, dtype=np.float32)

        # Vypočítaj 4 rohy roviny
        corner1 = center - tangent1 * size - tangent2 * size
        corner2 = center + tangent1 * size - tangent2 * size
        corner3 = center + tangent1 * size + tangent2 * size
        corner4 = center - tangent1 * size + tangent2 * size

        # Posuň rovinu mierne dozadu aby vektory boli vždy viditeľné
        glEnable(GL_POLYGON_OFFSET_FILL)
        glPolygonOffset(1.0, 1.0)

        if transparent:
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glColor4f(color[0], color[1], color[2], alpha)
        else:
            glColor3f(color[0], color[1], color[2])

        glBegin(GL_QUADS)
        glVertex3f(corner1[0], corner1[1], corner1[2])
        glVertex3f(corner2[0], corner2[1], corner2[2])
        glVertex3f(corner3[0], corner3[1], corner3[2])
        glVertex3f(corner4[0], corner4[1], corner4[2])
        glEnd()

        glDisable(GL_POLYGON_OFFSET_FILL)

        if transparent:
            glDisable(GL_BLEND)

    @staticmethod
    def draw_grid_2d(ortho_scale, pan_x, pan_y, width, height, step=1.0, z=-0.1, max_lines=100):
        """Nakreslí 2D mriežku - OPTIMALIZOVANÉ"""
        width, height = pygame.display.get_window_size()
        aspect = width / height

        if aspect >= 1.0:
            left = -ortho_scale * aspect + pan_x
            right = ortho_scale * aspect + pan_x
            bottom = -ortho_scale + pan_y
            top = ortho_scale + pan_y
        else:
            left = -ortho_scale + pan_x
            right = ortho_scale + pan_x
            bottom = -ortho_scale / aspect + pan_y
            top = ortho_scale / aspect + pan_y

        visible_range = max(right - left, top - bottom)

        # Dynamicky prispôsob step podľa zoom levelu
        display_step = step
        while visible_range / display_step > max_lines:
            display_step *= 2

        start_x = math.floor(left / display_step) * display_step
        end_x = math.ceil(right / display_step) * display_step
        start_y = math.floor(bottom / display_step) * display_step
        end_y = math.ceil(top / display_step) * display_step

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
    def draw_grid_in_plane(normal, center=(0, 0, 0), size=10.0, step=1.0, color=(0.4, 0.4, 0.4)):
        """Nakreslí grid v ľubovoľnej rovine definovanej normálou - OPTIMALIZOVANÉ"""
        # Normalizuj normálu
        normal = np.array(normal, dtype=np.float32)
        norm_length = np.linalg.norm(normal)
        if norm_length < 1e-6:
            return
        normal = normal / norm_length

        # Cache key pre tangent vektory
        normal_key = (round(normal[0], 4), round(normal[1], 4), round(normal[2], 4))

        if normal_key in GridRenderer._plane_tangents_cache:
            tangent1, tangent2 = GridRenderer._plane_tangents_cache[normal_key]
        else:
            # Nájdi dva kolmé vektory v rovine
            if abs(normal[2]) < 0.9:
                ref = np.array([0, 0, 1], dtype=np.float32)
            else:
                ref = np.array([1, 0, 0], dtype=np.float32)

            tangent1 = np.cross(normal, ref)
            tangent1 = tangent1 / np.linalg.norm(tangent1)
            tangent2 = np.cross(normal, tangent1)
            tangent2 = tangent2 / np.linalg.norm(tangent2)

            # Ulož do cache
            GridRenderer._plane_tangents_cache[normal_key] = (tangent1.copy(), tangent2.copy())

        center = np.array(center, dtype=np.float32)

        # Nakresli grid s polygon offset
        glPushAttrib(GL_LINE_BIT | GL_CURRENT_BIT | GL_DEPTH_BUFFER_BIT)
        glDepthFunc(GL_LEQUAL)
        glEnable(GL_POLYGON_OFFSET_LINE)
        glPolygonOffset(-1.0, -1.0)

        glLineWidth(1.5)
        glColor3f(color[0], color[1], color[2])
        glBegin(GL_LINES)

        size_int = int(size)
        step_int = max(1, int(step))

        # Čiary pozdĺž tangent1
        for i in range(-size_int, size_int + 1, step_int):
            t = float(i)
            p1 = center + tangent1 * (-size) + tangent2 * t
            p2 = center + tangent1 * size + tangent2 * t
            glVertex3f(p1[0], p1[1], p1[2])
            glVertex3f(p2[0], p2[1], p2[2])

        # Čiary pozdĺž tangent2
        for i in range(-size_int, size_int + 1, step_int):
            t = float(i)
            p1 = center + tangent1 * t + tangent2 * (-size)
            p2 = center + tangent1 * t + tangent2 * size
            glVertex3f(p1[0], p1[1], p1[2])
            glVertex3f(p2[0], p2[1], p2[2])

        glEnd()

        glDisable(GL_POLYGON_OFFSET_LINE)
        glPopAttrib()
    @staticmethod
    def draw_grid_3d(size=10.0, step=1.0):
        step = 2.0
        """Nakreslí 3D mriežku - len čiary bez výplne"""
        # XY rovina (Z=0) - zelená
        GridRenderer.draw_grid_in_plane(
            normal=[0, 0, 1], center=[0, 0, 0], size=size, step=step,
            color=(0.2, 0.7, 0.2)
        )
        # XZ rovina (Y=0) - červená
        GridRenderer.draw_grid_in_plane(
            normal=[0, 1, 0], center=[0, 0, 0], size=size, step=step,
            color=(0.7, 0.2, 0.2)
        )
        # YZ rovina (X=0) - modrá
        GridRenderer.draw_grid_in_plane(
            normal=[1, 0, 0], center=[0, 0, 0], size=size, step=step,
            color=(0.2, 0.2, 0.7)
        )

    @staticmethod
    def draw_planes_3d(size=10.0, step=1.0, colored=False):
        """Nakreslí 3D plochy s výplňou a gridom"""

        if colored:
            # Farebné plochy - nepriehľadné
            alpha = 1.0
            transparent = False
            xy_color = (0.2, 0.8, 0.2)  # Zelená pre XY
            xz_color = (0.8, 0.2, 0.2)  # Červená pre XZ
            yz_color = (0.2, 0.2, 0.8)  # Modrá pre YZ
        else:
            # Neutrálne sivé plochy - priesvitné
            alpha = 1.0
            transparent = False
            xy_color = (0.5, 0.5, 0.5)
            xz_color = (0.5, 0.5, 0.5)
            yz_color = (0.5, 0.5, 0.5)

        # XY rovina (Z=0) - zelená
        GridRenderer.draw_filled_plane(
            normal=[0, 0, 1], center=[0, 0, 0], size=size,
            color=xy_color, alpha=alpha, transparent=transparent
        )
        GridRenderer.draw_grid_in_plane(
            normal=[0, 0, 1], center=[0, 0, 0], size=size, step=step,
            color=(0.1, 0.4, 0.1) if colored else (0.3, 0.3, 0.3)
        )

        # XZ rovina (Y=0) - červená
        GridRenderer.draw_filled_plane(
            normal=[0, 1, 0], center=[0, 0, 0], size=size,
            color=xz_color, alpha=alpha, transparent=transparent
        )
        GridRenderer.draw_grid_in_plane(
            normal=[0, 1, 0], center=[0, 0, 0], size=size, step=step,
            color=(0.4, 0.1, 0.1) if colored else (0.3, 0.3, 0.3)
        )

        # YZ rovina (X=0) - modrá
        GridRenderer.draw_filled_plane(
            normal=[1, 0, 0], center=[0, 0, 0], size=size,
            color=yz_color, alpha=alpha, transparent=transparent
        )
        GridRenderer.draw_grid_in_plane(
            normal=[1, 0, 0], center=[0, 0, 0], size=size, step=step,
            color=(0.1, 0.1, 0.4) if colored else (0.3, 0.3, 0.3)
        )



    @staticmethod
    def draw_grid_in_plane(normal, center=(0, 0, 0), size=10.0, step=1.0, color=(0.4, 0.4, 0.4)):
        """Nakreslí grid v ľubovoľnej rovine definovanej normálou - OPTIMALIZOVANÉ"""
        # Normalizuj normálu
        normal = np.array(normal, dtype=np.float32)
        norm_length = np.linalg.norm(normal)
        if norm_length < 1e-6:
            return
        normal = normal / norm_length

        # Cache key pre tangent vektory
        normal_key = (round(normal[0], 4), round(normal[1], 4), round(normal[2], 4))

        if normal_key in GridRenderer._plane_tangents_cache:
            tangent1, tangent2 = GridRenderer._plane_tangents_cache[normal_key]
        else:
            # Nájdi dva kolmé vektory v rovine
            if abs(normal[2]) < 0.9:
                ref = np.array([0, 0, 1], dtype=np.float32)
            else:
                ref = np.array([1, 0, 0], dtype=np.float32)

            tangent1 = np.cross(normal, ref)
            tangent1 = tangent1 / np.linalg.norm(tangent1)
            tangent2 = np.cross(normal, tangent1)
            tangent2 = tangent2 / np.linalg.norm(tangent2)

            # Ulož do cache
            GridRenderer._plane_tangents_cache[normal_key] = (tangent1.copy(), tangent2.copy())

        center = np.array(center, dtype=np.float32)

        # Nakresli grid s polygon offset
        glPushAttrib(GL_LINE_BIT | GL_CURRENT_BIT | GL_DEPTH_BUFFER_BIT)
        glDepthFunc(GL_LEQUAL)
        glEnable(GL_POLYGON_OFFSET_LINE)
        glPolygonOffset(1.0, 1.0)

        glLineWidth(1.5)
        glColor3f(color[0], color[1], color[2])
        glBegin(GL_LINES)

        size_int = int(size)
        step_int = max(1, int(step))

        # Čiary pozdĺž tangent1
        for i in range(-size_int, size_int + 1, step_int):
            t = float(i)
            p1 = center + tangent1 * (-size) + tangent2 * t
            p2 = center + tangent1 * size + tangent2 * t
            glVertex3f(p1[0], p1[1], p1[2])
            glVertex3f(p2[0], p2[1], p2[2])

        # Čiary pozdĺž tangent2
        for i in range(-size_int, size_int + 1, step_int):
            t = float(i)
            p1 = center + tangent1 * t + tangent2 * (-size)
            p2 = center + tangent1 * t + tangent2 * size
            glVertex3f(p1[0], p1[1], p1[2])
            glVertex3f(p2[0], p2[1], p2[2])

        glEnd()

        glDisable(GL_POLYGON_OFFSET_LINE)
        glPopAttrib()






    @classmethod
    def clear_cache(cls):
        """Vymaže cache - volať pri zmene scény"""
        cls._plane_tangents_cache.clear()
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

        self.operation_plane_normal = None  # PRIDAJ TOTO

    def compute_optimal_view_for_3d(self):
        """Vypočíta optimálny uhol pohľadu pre 3D vektory pomocou PCA"""
        if not self.operands:
            return None

        simple_view = self.compute_simple_view_for_vectors()
        if simple_view:
            return simple_view

        # Fallback na PCA
        points = []
        for op in self.operands:
            if isinstance(op[0], (list, tuple)):
                for row in op:
                    vec3d = list(row) + [0] * (3 - len(row))
                    points.append(vec3d[:3])
            else:
                vec3d = list(op) + [0] * (3 - len(op))
                points.append(vec3d[:3])

        if self.result:
            if isinstance(self.result[0], (list, tuple)):
                for row in self.result:
                    vec3d = list(row) + [0] * (3 - len(row))
                    points.append(vec3d[:3])
            else:
                vec3d = list(self.result) + [0] * (3 - len(self.result))
                points.append(vec3d[:3])

        points.append([0, 0, 0])

        if len(points) < 3:
            return None

        points_np = np.array(points, dtype=np.float64)
        centroid = np.mean(points_np, axis=0)
        centered = points_np - centroid

        cov_matrix = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

        idx = eigenvalues.argsort()[::-1]
        eigenvectors = eigenvectors[:, idx]

        normal = np.real(eigenvectors[:, 2])
        normal = normal / np.linalg.norm(normal)

        if normal[1] > 0:
            normal = -normal

        elevation = np.degrees(np.arcsin(np.clip(-normal[1], -1.0, 1.0)))
        azimuth = np.degrees(np.arctan2(-normal[0], -normal[2]))

        if elevation > 89.5:
            elevation = 89.5

        return {
            'azimuth': azimuth,
            'elevation': elevation,
            'normal': normal.tolist()
        }

    def compute_simple_view_for_vectors(self):
        """Vypočíta optimálny uhol pohľadu pre vektory v rovine"""
        if not self.operands or len(self.operands) < 1:
            return None

        # Zozbieraj všetky vektory
        vectors_3d = []

        for operand in self.operands:
            if isinstance(operand[0], (list, tuple)):
                for row in operand:
                    vec3d = list(row) + [0] * (3 - len(row))
                    vec3d = vec3d[:3]
                    if sum(abs(x) for x in vec3d) > 1e-6:
                        vectors_3d.append(np.array(vec3d, dtype=np.float64))
            else:
                vec3d = list(operand) + [0] * (3 - len(operand))
                vec3d = vec3d[:3]
                if sum(abs(x) for x in vec3d) > 1e-6:
                    vectors_3d.append(np.array(vec3d, dtype=np.float64))

        if len(vectors_3d) < 2:
            return None

        # Nájdi 2 nelineárne závislé vektory pre cross product
        best_normal = None
        best_norm = 0

        for i in range(len(vectors_3d)):
            for j in range(i + 1, len(vectors_3d)):
                v1 = vectors_3d[i]
                v2 = vectors_3d[j]

                normal = np.cross(v1, v2)
                norm = np.linalg.norm(normal)

                if norm > best_norm:
                    best_norm = norm
                    best_normal = normal

        if best_normal is None or best_norm < 1e-6:
            return None

        normal = best_normal / best_norm

        # Orientácia - preferujeme pohľad "spredu/zhora"
        if normal[1] > 0:
            normal = -normal

        # Výpočet uhlov
        elevation = np.degrees(np.arcsin(np.clip(-normal[1], -1.0, 1.0)))
        azimuth = np.degrees(np.arctan2(-normal[0], -normal[2]))

        # Žiadne umelé limity - ponechaj presné hodnoty
        if elevation > 89.5:
            elevation = 89.5

        return {
            'azimuth': azimuth,
            'elevation': elevation,
            'normal': normal.tolist()
        }
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

    # V triede AnimationController - opravená metóda compute_simple_view_for_vectors
    def compute_simple_view_for_vectors(self):
        """Vypočíta optimálny uhol pohľadu pre vektory v rovine"""
        if not self.operands or len(self.operands) < 1:
            return None

        # Zozbieraj všetky vektory
        vectors_3d = []

        for operand in self.operands:
            if isinstance(operand[0], (list, tuple)):
                for row in operand:
                    vec3d = list(row) + [0] * (3 - len(row))
                    vec3d = vec3d[:3]
                    if sum(abs(x) for x in vec3d) > 1e-6:
                        vectors_3d.append(np.array(vec3d, dtype=np.float64))
            else:
                vec3d = list(operand) + [0] * (3 - len(operand))
                vec3d = vec3d[:3]
                if sum(abs(x) for x in vec3d) > 1e-6:
                    vectors_3d.append(np.array(vec3d, dtype=np.float64))

        if len(vectors_3d) < 2:
            return None

        # Nájdi 2 nelineárne závislé vektory pre cross product
        best_normal = None
        best_norm = 0

        for i in range(len(vectors_3d)):
            for j in range(i + 1, len(vectors_3d)):
                v1 = vectors_3d[i]
                v2 = vectors_3d[j]

                normal = np.cross(v1, v2)
                norm = np.linalg.norm(normal)

                if norm > best_norm:
                    best_norm = norm
                    best_normal = normal

        if best_normal is None or best_norm < 1e-6:
            return None

        normal = best_normal / best_norm

        # Orientácia - preferujeme pohľad "spredu/zhora"
        if normal[1] > 0:
            normal = -normal

        # Výpočet uhlov
        elevation = np.degrees(np.arcsin(np.clip(-normal[1], -1.0, 1.0)))
        azimuth = np.degrees(np.arctan2(-normal[0], -normal[2]))

        # OPRAVA: Žiadne umelé limity - ponechaj presné hodnoty
        # Len extrémny limit pre numerickú stabilitu gluLookAt
        if elevation > 89.5:
            elevation = 89.5

        return {
            'azimuth': azimuth,
            'elevation': elevation,
            'normal': normal.tolist()
        }

    def compute_simple_view_for_vectors(self):
        """Vypočíta optimálny uhol pohľadu pre vektory v rovine"""
        if not self.operands or len(self.operands) < 1:
            return None

        # Zozbieraj všetky vektory
        vectors_3d = []

        for operand in self.operands:
            if isinstance(operand[0], (list, tuple)):
                for row in operand:
                    vec3d = list(row) + [0] * (3 - len(row))
                    vec3d = vec3d[:3]
                    if sum(abs(x) for x in vec3d) > 1e-6:
                        vectors_3d.append(np.array(vec3d, dtype=np.float64))
            else:
                vec3d = list(operand) + [0] * (3 - len(operand))
                vec3d = vec3d[:3]
                if sum(abs(x) for x in vec3d) > 1e-6:
                    vectors_3d.append(np.array(vec3d, dtype=np.float64))

        if len(vectors_3d) < 2:
            return None

        # Nájdi 2 nelineárne závislé vektory pre cross product
        best_normal = None
        best_norm = 0

        for i in range(len(vectors_3d)):
            for j in range(i + 1, len(vectors_3d)):
                v1 = vectors_3d[i]
                v2 = vectors_3d[j]

                normal = np.cross(v1, v2)
                norm = np.linalg.norm(normal)

                if norm > best_norm:
                    best_norm = norm
                    best_normal = normal

        if best_normal is None or best_norm < 1e-6:
            return None

        normal = best_normal / best_norm

        # Orientácia - preferujeme pohľad "spredu/zhora"
        if normal[1] > 0:
            normal = -normal

        # Výpočet uhlov
        elevation = np.degrees(np.arcsin(np.clip(-normal[1], -1.0, 1.0)))
        azimuth = np.degrees(np.arctan2(-normal[0], -normal[2]))

        # OPRAVA: Žiadne umelé limity - ponechaj presné hodnoty
        # Len extrémny limit pre numerickú stabilitu gluLookAt
        if elevation > 89.5:
            elevation = 89.5

        return {
            'azimuth': azimuth,
            'elevation': elevation,
            'normal': normal.tolist()
        }
    # V triede AnimationController - opravená metóda detect_operation_plane

    def detect_operation_plane(self, tolerance=1e-4):  # Zvýšená tolerancia z 1e-6
        """Zistí v akej rovine sa operácia vykonáva"""
        if not self.operands:
            return "3D"

        # Zozbieraj všetky súradnice zo všetkých operandov
        all_x = []
        all_y = []
        all_z = []

        for operand in self.operands:
            if isinstance(operand[0], (list, tuple)):  # Matica
                for row in operand:
                    all_x.append(row[0] if len(row) > 0 else 0)
                    all_y.append(row[1] if len(row) > 1 else 0)
                    all_z.append(row[2] if len(row) > 2 else 0)
            else:  # Vektor
                all_x.append(operand[0] if len(operand) > 0 else 0)
                all_y.append(operand[1] if len(operand) > 1 else 0)
                all_z.append(operand[2] if len(operand) > 2 else 0)

        # Skontroluj či sú všetky hodnoty blízko nule
        all_x_zero = all(abs(x) < tolerance for x in all_x)
        all_y_zero = all(abs(y) < tolerance for y in all_y)
        all_z_zero = all(abs(z) < tolerance for z in all_z)

        # Urči rovinu
        if all_z_zero and not all_x_zero and not all_y_zero:
            return "XY"
        elif all_y_zero and not all_x_zero and not all_z_zero:
            return "XZ"
        elif all_x_zero and not all_y_zero and not all_z_zero:
            return "YZ"
        else:
            return "3D"

    # V triede AnimationController - uprav metódu setup_operation
    def setup_operation(self, operation_type, operands, result, constant=None):
        """Setup a new operation for step-by-step visualization"""
        self.operation_type = operation_type
        self.operands = operands
        self.result = result
        self.constant = constant
        self.current_step = 0

        # Define number of steps based on operation
        if operation_type == 'add':
            self.max_steps = 3  # ZMENENÉ z 2 na 3 (pridaný krok pre komutatívnosť)
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
        self.operation_plane_normal = self._compute_plane_normal()

    # V triede AnimationController - opravená metóda _compute_plane_normal

    def _compute_plane_normal(self):
        """Vypočíta normálu roviny pre celú operáciu (volá sa raz na začiatku)"""
        if self.current_plane == "XY":
            return [0, 0, 1]
        elif self.current_plane == "XZ":
            return [0, 1, 0]
        elif self.current_plane == "YZ":
            return [1, 0, 0]
        elif self.current_plane == "3D":
            # Pre všeobecnú 3D rovinu použijem vypočítanú normálu
            optimal_view = self.compute_simple_view_for_vectors()
            if optimal_view and 'normal' in optimal_view:
                return optimal_view['normal']
            # Fallback - skús PCA
            optimal_view = self.compute_optimal_view_for_3d()
            if optimal_view and 'normal' in optimal_view:
                return optimal_view['normal']
        return [0, 0, 1]  # Default - XY rovina

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
        self.operation_plane_normal = None

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

    # V triede AnimationController - uprav metódu _get_vectors_for_step
    def _get_vectors_for_step(self, step):
        """Získa vektory pre daný krok (bez animácie)"""
        if not self.current_operation:
            return []

        vectors = []
        is_matrix_op = self._is_matrix(self.operands[0]) if self.operands else False

        if self.operation_type == 'add':
            if step == 0:
                # Krok 0: Oba vektory z originu
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
                # Krok 1: v2 posunutý na koniec v1
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
                # Krok 2: Výsledok + predošlé priesvitné
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


            # V triede AnimationController - uprav len krok 3 v metóde _get_vectors_for_step

            # V triede AnimationController - uprav krok 3 v metóde _get_vectors_for_step

            # Pridaj 'fade_in': True pre komutatívne vektory

            elif step == 3:

                # Krok 3: KOMUTATÍVNOSŤ - pôvodné zostávajú, pridajú sa nové

                if is_matrix_op:

                    # === PÔVODNÉ VEKTORY (presne ako v kroku 2) ===

                    vectors.append({

                        'vec': self.operands[0],

                        'offset': [0, 0, 0],

                        'color': (1, 0.5, 0),

                        'alpha': 0.3,

                        'label': 'A'

                    })

                    row_offsets_v1 = []

                    for row in self.operands[0]:
                        offset = list(row) + [0] * (3 - len(row))

                        row_offsets_v1.append(offset[:3])

                    vectors.append({

                        'vec': self.operands[1],

                        'offset': [0, 0, 0],

                        'row_offsets': row_offsets_v1,

                        'color': (0, 0.5, 1),

                        'alpha': 0.3,

                        'label': 'B'

                    })

                    vectors.append({

                        'vec': self.result,

                        'offset': [0, 0, 0],

                        'color': self.RESULT_COLOR,

                        'alpha': 1.0,

                        'label': 'A + B'

                    })

                    # === NOVÉ VEKTORY PRE OPAČNÉ SČÍTANIE (s fade_in) ===

                    vectors.append({

                        'vec': self.operands[1],

                        'offset': [0, 0, 0],

                        'color': (0.3, 0.8, 1),

                        'alpha': 0.7,

                        'label': 'B (nové)',

                        'fade_in': True  # PRIDANÉ

                    })

                    row_offsets_v2 = []

                    for row in self.operands[1]:
                        offset = list(row) + [0] * (3 - len(row))

                        row_offsets_v2.append(offset[:3])

                    vectors.append({

                        'vec': self.operands[0],

                        'offset': [0, 0, 0],

                        'row_offsets': row_offsets_v2,

                        'color': (1, 0.8, 0.4),

                        'alpha': 0.7,

                        'label': 'A (nové)',

                        'fade_in': True  # PRIDANÉ

                    })


                else:

                    # === PÔVODNÉ VEKTORY (presne ako v kroku 2) ===

                    vectors.append({

                        'vec': self.operands[0],

                        'offset': [0, 0, 0],

                        'color': (1, 0.5, 0),

                        'alpha': 0.3,

                        'label': 'v1'

                    })

                    offset_v1 = list(self.operands[0]) + [0] * (3 - len(self.operands[0]))

                    vectors.append({

                        'vec': self.operands[1],

                        'offset': offset_v1[:3],

                        'color': (0, 0.5, 1),

                        'alpha': 0.3,

                        'label': 'v2'

                    })

                    vectors.append({

                        'vec': self.result,

                        'offset': [0, 0, 0],

                        'color': self.RESULT_COLOR,

                        'alpha': 1.0,

                        'label': 'v1 + v2'

                    })

                    # === NOVÉ VEKTORY PRE OPAČNÉ SČÍTANIE (s fade_in) ===

                    vectors.append({

                        'vec': self.operands[1],

                        'offset': [0, 0, 0],

                        'color': (0.3, 0.8, 1),

                        'alpha': 0.7,

                        'label': 'v2 (nové)',

                        'fade_in': True  # PRIDANÉ

                    })

                    offset_v2 = list(self.operands[1]) + [0] * (3 - len(self.operands[1]))

                    vectors.append({

                        'vec': self.operands[0],

                        'offset': offset_v2[:3],

                        'color': (1, 0.8, 0.4),

                        'alpha': 0.7,

                        'label': 'v1 (nové)',

                        'fade_in': True  # PRIDANÉ

                    })
        elif self.operation_type == 'subtract':
            # ... zvyšok subtract ostáva rovnaký ...
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

        elif self.operation_type == 'linear_combination':
            c1, c2 = self.constant if self.constant else [1, 1]

            if step == 0:
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

    # V triede AnimationController - uprav metódu get_vectors_to_draw

    # V triede AnimationController - uprav metódu get_vectors_to_draw

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

        # Pridaj nové vektory (ktoré sú len v target)
        if len(self.target_vectors) > len(self.source_vectors):
            for i in range(len(self.source_vectors), len(self.target_vectors)):
                target_vec = self.target_vectors[i]
                new_vec = {}

                # Skontroluj či má fade_in flag
                if target_vec.get('fade_in', False):
                    # FADE-IN: vektor v plnej veľkosti, len alpha sa mení
                    new_vec['vec'] = target_vec['vec']

                    if 'offset' in target_vec:
                        new_vec['offset'] = target_vec['offset']

                    if 'row_offsets' in target_vec:
                        new_vec['row_offsets'] = target_vec['row_offsets']

                    if 'color' in target_vec:
                        new_vec['color'] = target_vec['color']

                    if 'label' in target_vec:
                        new_vec['label'] = target_vec['label']

                    # Alpha sa postupne zvyšuje od 0 do cieľovej hodnoty
                    target_alpha = target_vec.get('alpha', 1.0)
                    new_vec['alpha'] = target_alpha * self.animation_progress

                    # Progress na 1.0 aby sa vektor vykreslil celý
                    new_vec['progress'] = 1.0
                    if not isinstance(new_vec['vec'][0], (int, float)):
                        new_vec['row_progress'] = [1.0] * len(new_vec['vec'])

                else:
                    # NORMÁLNA ANIMÁCIA: vektor rastie od nuly
                    new_vec = target_vec.copy()

                    if isinstance(target_vec['vec'][0], (int, float)):
                        new_vec['vec'] = [v * self.animation_progress for v in target_vec['vec']]
                        new_vec['progress'] = self.animation_progress
                    else:
                        new_vec['vec'] = [[v * self.animation_progress for v in row]
                                          for row in target_vec['vec']]
                        new_vec['row_progress'] = [self.animation_progress] * len(target_vec['vec'])

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


class FontCache:
    """Cache pre fonty - načíta font len raz"""
    _fonts = {}

    @classmethod
    def get_font(cls, name, size, bold=True):
        key = (name, size, bold)
        if key not in cls._fonts:
            cls._fonts[key] = pygame.font.SysFont(name, size, bold)
        return cls._fonts[key]


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


class SpanAnimationController:
    """Ovládač animácie pre vizualizáciu span (lineárneho obalu)"""

    def __init__(self):
        self.active = False
        self.basis_vectors = []
        self.current_step = 0
        self.combinations = []
        self.animating = False
        self.animation_progress = 0.0
        self.animation_speed = 1.0

        # Predgenerované body kruhu
        self.circle_radius = 2.5
        self.num_circle_points = 20
        self.circle_points = []
        self.variation_radius = 0.8

        self.current_circle_index = 0
        self.last_target_point = None

        # Zoznam zachovaných vektorov
        self.persistent_vectors = []
        self.persistence_chance = 0.4

        # Automatické prehrávanie
        self.auto_play = False
        self.auto_play_delay = 1.0
        self.auto_play_timer = 0.0

        # Pre závislé vektory (priamka)
        self.line_points = []
        self.line_direction = 1
        self.line_step = 0.5
        self.current_line_position = 0.0

        # NOVÉ: Flag pre "show all" režim a uložené nastavenia kamery
        self.show_all_mode = False
        self.locked_ortho_scale = None
        self.color_scheme = 0

    def are_vectors_dependent(self):
        """Zistí či sú vektory lineárne závislé"""
        if not self.basis_vectors or len(self.basis_vectors) < 2:
            return False

        v1, v2 = self.basis_vectors
        cross_product = v1[0] * v2[1] - v1[1] * v2[0]
        return abs(cross_product) < 1e-6

    def show_all_combinations(self):
        """Zobrazí všetky možné kombinácie - OBMEDZENÝ PRIESTOR"""
        if not self.basis_vectors or len(self.basis_vectors) < 2:
            return

        self.animating = False
        self.auto_play = False
        self.persistent_vectors = []

        # NOVÉ: Aktivuj show_all režim
        self.show_all_mode = True

        v1, v2 = self.basis_vectors
        are_dependent = self.are_vectors_dependent()

        if are_dependent:
            # PRÍPAD 1: ZÁVISLÉ VEKTORY - PRIAMKA
            print("🔴 Závislé vektory - kreslím priamku!")

            v1_length = math.sqrt(v1[0] ** 2 + v1[1] ** 2)

            # Obmedzený rozsah
            c_range = 10
            num_points = 100
            step = (2 * c_range) / num_points

            direction = np.array(v1)
            c_values = np.arange(-c_range, c_range + step, step)

            for c in c_values:
                result = (c * direction / max(v1_length, 0.01)).tolist()
                result = [c * v1[0], c * v1[1]]

                if c < 0:
                    t = min(abs(c) / (c_range * 0.5), 1.0)
                    r = 0.1 + 0.3 * (1 - t)
                    g = 0.3 + 0.4 * (1 - t)
                    b = 1.0
                elif c > 0:
                    t = min(c / (c_range * 0.5), 1.0)
                    r = 1.0
                    g = 0.3 + 0.4 * (1 - t)
                    b = 0.1 + 0.3 * (1 - t)
                else:
                    r, g, b = 1.0, 1.0, 0.5

                distance = abs(c)
                alpha = max(0.4, 1.0 - (distance / c_range) * 0.6)

                self.persistent_vectors.append({
                    'vec': result,
                    'c1': c,
                    'c2': 0.0,
                    'color': (r, g, b),
                    'alpha': alpha,
                    'is_persistent': True
                })

            print(f"✨ Zobrazených {len(self.persistent_vectors)} bodov na priamke")
            self.locked_ortho_scale = 15.0  # Vhodná hodnota pre grid_range=10


        else:

            # PRÍPAD 2: NEZÁVISLÉ VEKTORY - ROVINA

            print("🟢 Nezávislé vektory - kreslím rovinu!")

            grid_range = 10
            grid_range_x = 18

            step = 1.0

            for grid_x in range(-grid_range_x, grid_range_x + 1, int(step)):

                for grid_y in range(-grid_range, grid_range + 1, int(step)):

                    target_x = grid_x + 0.5
                    target_y = grid_y + 0.5
                    det = v1[0] * v2[1] - v1[1] * v2[0]

                    if abs(det) < 1e-6:
                        continue

                    c1 = (target_x * v2[1] - target_y * v2[0]) / det
                    c2 = (v1[0] * target_y - v1[1] * target_x) / det
                    result = [c1 * v1[i] + c2 * v2[i] for i in range(len(v1))]

                    # FAREBNÁ SCHÉMA

                    if self.color_scheme == 1:

                        # RAKÚSKA VLAJKA
                        normalized_y = target_y / grid_range
                        if normalized_y > 0.33:
                            r, g, b = 0.93, 0.11, 0.14
                        elif normalized_y < -0.33:
                            r, g, b = 0.93, 0.11, 0.14
                        else:
                            r, g, b = 1.0, 1.0, 1.0
                    else:
                        # RAINBOW (pôvodný)
                        angle = math.atan2(target_y, target_x)
                        hue = (angle + math.pi) / (2 * math.pi)
                        h = hue * 6
                        i_h = int(h) % 6
                        f = h - int(h)
                        if i_h == 0:
                            r, g, b = 1, f, 0
                        elif i_h == 1:
                            r, g, b = 1 - f, 1, 0
                        elif i_h == 2:
                            r, g, b = 0, 1, f
                        elif i_h == 3:
                            r, g, b = 0, 1 - f, 1
                        elif i_h == 4:
                            r, g, b = f, 0, 1
                        else:
                            r, g, b = 1, 0, 1 - f
                        r = 0.2 + r * 0.8
                        g = 0.2 + g * 0.8
                        b = 0.2 + b * 0.8
                    distance = math.sqrt(target_x ** 2 + target_y ** 2)
                    max_distance = grid_range * math.sqrt(2)
                    alpha = max(0.3, 1.0 - (distance / max_distance) * 0.7)
                    self.persistent_vectors.append({
                        'vec': result,
                        'c1': c1,
                        'c2': c2,
                        'color': (r, g, b),
                        'alpha': alpha,
                        'is_persistent': True

                    })

            print(f"✨ Zobrazených {len(self.persistent_vectors)} kombinácií")
            self.locked_ortho_scale = 6.0  # Vhodná hodnota pre grid_range=10

        # NOVÉ: Ulož ideálny zoom level


    def setup_span(self, vector1, vector2):
        """Nastaví span animáciu pre dva vektory"""
        v1 = np.array(vector1, dtype=float)
        v2 = np.array(vector2, dtype=float)

        # Zisti, či sú vektory lineárne závislé
        cross_product = v1[0] * v2[1] - v1[1] * v2[0]
        are_dependent = abs(cross_product) < 1e-6

        if are_dependent:
            # ZÁVISLÉ VEKTORY - span je priamka
            print("⚠️ Vektory sú lineárne závislé - span je len priamka!")
            self.basis_vectors = [v1.tolist(), v2.tolist()]
            self.circle_radius = 1.5
            self.num_circle_points = 15

            # NOVÉ: Pre závislé vektory - generuj body pozdĺž priamky
            self.line_points = []
            self.line_direction = 1  # 1 = dopredu, -1 = dozadu
            self.line_step = 0.5  # Krok pozdĺž priamky
            self.current_line_position = 0.0
        else:
            # NEZÁVISLÉ VEKTORY - span je rovina
            print("✓ Vektory sú lineárne nezávislé - span je celá rovina!")
            self.basis_vectors = [v1.tolist(), v2.tolist()]
            self.circle_radius = 2.5
            self.num_circle_points = 20

        # Inicializácia stavu
        self.current_step = 0
        self.active = True
        self.animating = False
        self.animation_progress = 1.0
        self.current_circle_index = 0
        self.persistent_vectors = []

        # Generovanie bodov kruhu (pre nezávislé)
        self.circle_points = []
        for i in range(self.num_circle_points):
            angle = (2 * math.pi * i) / self.num_circle_points
            c1 = self.circle_radius * math.cos(angle)
            c2 = self.circle_radius * math.sin(angle)
            self.circle_points.append((c1, c2))

        # Prvý krok
        if are_dependent:
            self.combinations = [{'c1': 0.5, 'c2': 0.0}]  # Začni s malou hodnotou
            self.current_line_position = 0.5
        else:
            self.combinations = [{'c1': 1.0, 'c2': 1.0}]

        print(f"Pôvodné: v1={vector1}, v2={vector2}")
        print(f"Použité: v1={v1.tolist()}, v2={v2.tolist()}")
        print(f"✓ Span nastavený: {self.num_circle_points} bodov, polomer {self.circle_radius}")

    def next_step(self):
        """Prejdi na ďalší krok - s možnosťou zachovania výsledného vektora"""
        are_dependent = self.are_vectors_dependent()

        # ZACHOVANÉ: Pred prechodom na nový krok, s 40% šancou ulož aktuálny výsledný vektor
        if self.current_step >= 1 and random.random() < self.persistence_chance:
            current_comb = self.combinations[self.current_step]
            c1, c2 = current_comb['c1'], current_comb['c2']
            v1, v2 = self.basis_vectors

            # Pre závislé: použij len c1
            if are_dependent:
                result = [c1 * v1[i] for i in range(len(v1))]
                self.persistent_vectors.append({
                    'vec': result,
                    'c1': c1,
                    'c2': 0.0
                })
                print(f"✓ Vektor zachovaný! (c={c1:.2f})")
            else:
                # Pre nezávislé: pôvodná logika
                result = [c1 * v1[i] + c2 * v2[i] for i in range(len(v1))]
                self.persistent_vectors.append({
                    'vec': result,
                    'c1': c1,
                    'c2': c2
                })
                print(f"✓ Vektor zachovaný! (c1={c1:.2f}, c2={c2:.2f})")

        self.current_step += 1

        if self.current_step >= len(self.combinations):
            if not self.basis_vectors or len(self.basis_vectors) < 2:
                c1 = random.uniform(-2, 2)
                c2 = random.uniform(-2, 2)
                self.combinations.append({'c1': c1, 'c2': c2})
            else:
                v1 = self.basis_vectors[0]
                v2 = self.basis_vectors[1]

                if are_dependent:
                    # ========================================
                    # OPRAVENÉ: PLYNULÝ POHYB PRE ZÁVISLÉ VEKTORY
                    # ========================================

                    # Pridaj malý krok k aktuálnej pozícii
                    step_size = random.uniform(0.3, 0.8)  # Variabilný krok
                    self.current_line_position += step_size * self.line_direction

                    # Ak sme príliš ďaleko, otoč smer
                    max_distance = 4.0
                    if abs(self.current_line_position) > max_distance:
                        self.line_direction *= -1
                        self.current_line_position += step_size * self.line_direction * 2

                    # Občas (10%) zmeň smer pre zaujímavejšiu animáciu
                    if random.random() < 0.1:
                        self.line_direction *= -1

                    c1 = self.current_line_position
                    c2 = 0.0

                    result_x = c1 * v1[0]
                    result_y = c1 * v1[1]

                    self.last_target_point = (result_x, result_y)
                    self.combinations.append({'c1': c1, 'c2': c2})

                    print(f"Krok {self.current_step}: Závislé vektory, c={c1:.2f} "
                          f"→ ({result_x:.2f}, {result_y:.2f})")
                else:
                    # PRE NEZÁVISLÉ: Pôvodná logika
                    if not self.circle_points:
                        print("CHYBA: Kruh nebol vygenerovaný!")
                        return False

                    self.current_circle_index = (self.current_circle_index + 1) % len(self.circle_points)
                    circle_point = self.circle_points[self.current_circle_index]
                    base_c1, base_c2 = circle_point

                    random_radius_factor = random.uniform(0.3, 1.2)
                    scaled_c1 = base_c1 * random_radius_factor
                    scaled_c2 = base_c2 * random_radius_factor

                    angle_variation = random.uniform(-0.2, 0.2)
                    angle = math.atan2(scaled_c2, scaled_c1) + angle_variation
                    radius = math.sqrt(scaled_c1 ** 2 + scaled_c2 ** 2)

                    c1 = radius * math.cos(angle)
                    c2 = radius * math.sin(angle)

                    result_x = c1 * v1[0] + c2 * v2[0]
                    result_y = c1 * v1[1] + c2 * v2[1]

                    self.last_target_point = (result_x, result_y)
                    self.combinations.append({'c1': c1, 'c2': c2})

                    print(f"Krok {self.current_step}: Polomer {radius:.2f} "
                          f"→ ({result_x:.2f}, {result_y:.2f}), c1={c1:.2f}, c2={c2:.2f}")

        self.animating = True
        self.animation_progress = 0.0
        return True

    def prev_step(self):
        """Vráť sa na predošlý krok"""
        if self.current_step > 0:
            # Odstráň posledný zachovaný vektor ak existuje
            if self.persistent_vectors:
                last_persistent = self.persistent_vectors[-1]
                current_comb = self.combinations[self.current_step]

                if (abs(last_persistent['c1'] - current_comb['c1']) < 0.01 and
                        abs(last_persistent['c2'] - current_comb['c2']) < 0.01):
                    self.persistent_vectors.pop()
                    print("✗ Zachovaný vektor odstránený")

            self.current_step -= 1

            if self.current_step >= 2:
                steps_after_basis = self.current_step - 2
                self.current_circle_index = steps_after_basis % len(self.circle_points)
            else:
                self.current_circle_index = 0

            self.animating = True
            self.animation_progress = 0.0
            return True
        return False

    def update(self, dt):
        """Aktualizuj animáciu"""
        if self.animating:
            self.animation_progress += dt * self.animation_speed
            if self.animation_progress >= 1.0:
                self.animation_progress = 1.0
                self.animating = False

                # Ak je auto_play aktívne, resetuj timer
                if self.auto_play:
                    self.auto_play_timer = 0.0

        # Automatické krokovanie
        if self.auto_play and not self.animating:
            self.auto_play_timer += dt
            if self.auto_play_timer >= self.auto_play_delay:
                self.next_step()
                self.auto_play_timer = 0.0

    def get_current_vectors(self):
        """Vráti aktuálne vektory na vykreslenie s interpoláciou"""
        if not self.active or not self.basis_vectors:
            return []

        # Ak máme plno perzistentných vektorov (režim "show all"), zobraz len tie
        if len(self.persistent_vectors) > 50:
            vectors = []
            for pv in self.persistent_vectors:
                vectors.append({
                    'vec': pv['vec'],
                    'offset': [0, 0, 0],
                    'color': pv.get('color', (0.6, 0.2, 0.6)),
                    'alpha': pv.get('alpha', 0.4),
                    'label': f"c1={pv['c1']:.1f}, c2={pv['c2']:.1f}",
                    'show_label': False,
                    'is_persistent': True
                })
            return vectors

        v1, v2 = self.basis_vectors
        are_dependent = self.are_vectors_dependent()

        current_comb = self.combinations[self.current_step]
        c1_target = current_comb['c1']
        c2_target = current_comb['c2']

        # Interpolácia koeficientov
        if self.animating and self.current_step > 0:
            prev_comb = self.combinations[self.current_step - 1]
            t = self.animation_progress
            t = t * t * (3.0 - 2.0 * t)  # Smooth interpolation

            c1 = prev_comb['c1'] + (c1_target - prev_comb['c1']) * t
            c2 = prev_comb['c2'] + (c2_target - prev_comb['c2']) * t
        else:
            c1, c2 = c1_target, c2_target

        vectors = []

        # ========================================
        # PRÍPAD 1: ZÁVISLÉ VEKTORY - PRIAMKA
        # ========================================
        if are_dependent:
            scaled_v1 = [c1 * x for x in v1]

            # ZACHOVANÉ VEKTORY - fialové stopy (NAJPRV - VZADU)
            for persistent in self.persistent_vectors:
                vectors.append({
                    'vec': persistent['vec'],
                    'offset': [0, 0, 0],
                    'color': (0.6, 0.2, 0.6),
                    'alpha': 0.4,
                    'label': f"c={persistent['c1']:.1f}",
                    'show_label': False,
                    'is_persistent': True  # PRIDANÉ
                })

            # AKTUÁLNY VEKTOR - oranžový (NAKONIEC - VPREDU)
            vectors.append({
                'vec': scaled_v1,
                'offset': [0, 0, 0],
                'color': (1.0, 0.5, 0.2),
                'alpha': 1.0,
                'label': f'{c1:.2f}·v',
                'show_label': True,
                'is_persistent': False  # PRIDANÉ
            })

            return vectors

        # ========================================
        # PRÍPAD 2: NEZÁVISLÉ VEKTORY - ROVINA
        # ========================================
        scaled_v1 = [c1 * x for x in v1]
        scaled_v2 = [c2 * x for x in v2]

        # Oranžový vektor (vždy od originu)
        vectors.append({
            'vec': scaled_v1,
            'offset': [0, 0, 0],
            'color': (1.0, 0.5, 0.2),
            'alpha': 1.0,
            'label': f'{c1:.2f}·v1',
            'show_label': True,
            'is_persistent': False  # PRIDANÉ
        })

        # ANIMÁCIA OFFSETU modrého vektora
        if self.current_step == 0:
            blue_offset = [0, 0, 0]
        elif self.current_step == 1 and self.animating:
            t = self.animation_progress
            t = t * t * (3.0 - 2.0 * t)
            blue_offset = [scaled_v1[i] * t for i in range(len(scaled_v1))]
        else:
            blue_offset = scaled_v1

        # Modrý vektor
        vectors.append({
            'vec': scaled_v2,
            'offset': blue_offset,
            'color': (0.2, 0.5, 1.0),
            'alpha': 1.0,
            'label': f'{c2:.2f}·v2',
            'show_label': True,
            'is_persistent': False  # PRIDANÉ
        })

        # Od kroku 1 ďalej zobraz fialový výsledok a zachované vektory
        if self.current_step >= 1:
            # ZACHOVANÉ VEKTORY - bez labelov (VZADU)
            for persistent in self.persistent_vectors:
                vectors.append({
                    'vec': persistent['vec'],
                    'offset': [0, 0, 0],
                    'color': (0.6, 0.2, 0.6),
                    'alpha': 0.4,
                    'label': f"c1={persistent['c1']:.1f}, c2={persistent['c2']:.1f}",
                    'show_label': False,
                    'is_persistent': True  # PRIDANÉ
                })

            # Výsledná kombinácia - aktuálna (tmavo fialová)
            result = [scaled_v1[i] + scaled_v2[i] for i in range(len(v1))]

            if self.current_step == 1 and self.animating:
                purple_alpha = self.animation_progress
            else:
                purple_alpha = 1.0

            vectors.append({
                'vec': result,
                'offset': [0, 0, 0],
                'color': (0.5, 0.0, 0.5),
                'alpha': purple_alpha,
                'label': f'{c1:.2f}·v1 + {c2:.2f}·v2',
                'show_label': False,
                'is_persistent': False  # PRIDANÉ
            })

        return vectors

    def clear(self):
        """Vyčisti span animáciu"""
        self.active = False
        self.basis_vectors = []
        self.combinations = []
        self.current_step = 0
        self.circle_points = []
        self.current_circle_index = 0
        self.persistent_vectors = []
        self.animating = False
        self.animation_progress = 0.0
        self.auto_play = False
        self.auto_play_timer = 0.0
        self.line_points = []
        self.line_direction = 1
        self.line_step = 0.5
        self.current_line_position = 0.0

        # NOVÉ: Reset show_all režimu
        self.show_all_mode = False
        self.locked_ortho_scale = None

class Application:
    """Hlavná aplikácia - KOMPLETNE AKTUALIZOVANÁ"""

    def __init__(self):
        pygame.init()

        # Enable hardware multisampling
        pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLEBUFFERS, 1)
        pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLESAMPLES, 4)
        self._last_render_time = 0
        self._min_frame_time = 1.0 / 60.0  # Max 60 FPS
        self.RESULT_COLOR = (0.5, 0, 0.5)
        self.screen = pygame.display.set_mode((900, 700), DOUBLEBUF | OPENGL | RESIZABLE)

        self.width, self.height = pygame.display.get_window_size()

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
        self._update_background_color()
        self.grid_mode = 0
        self.plane_grid_mode = 1  # NOVÉ: 0=vypnutý, 1=grid, 2=jemný, 3=hrubý, 4=vyplnená+grid, 5=vyplnená
        self.show_axes = True
        self.vectors_as_points = True

        # Vectors
        self.saved_baza = []
        self.pending_vector = None

        # UI State
        self.matrix_inputs = [["" for _ in range(3)] for _ in range(3)]
        self.matrix_active_cell = (-1, -1)
        self.show_matrix_input = True

        # Dropdown
        self.dropdown_open = False
        self.dropdown_rect = pygame.Rect(10, 10, 150, 30)

        # Inicializuj pygame font
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
        """Moderný startup screen"""
        while self.startup_screen and self.running:
            dt = self.clock.tick(Config.FPS) / 1000.0
            mx, my = pygame.mouse.get_pos()

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
                    # Definície tlačidiel
                    cx, cy = self.width // 2, self.height // 2
                    btn_w, btn_h = 180, 52
                    gap = 24

                    btn_2d = pygame.Rect(cx - btn_w - gap // 2, cy + 30, btn_w, btn_h)
                    btn_3d = pygame.Rect(cx + gap // 2, cy + 30, btn_w, btn_h)
                    toggle = pygame.Rect(30, self.height - 60, 130, 38)

                    if toggle.collidepoint(mx, my):
                        self.background_dark = not self.background_dark
                    elif btn_2d.collidepoint(mx, my):
                        self.view_2d_mode = True
                        self.startup_screen = False
                    elif btn_3d.collidepoint(mx, my):
                        self.view_2d_mode = False
                        self.startup_screen = False

            # === RENDER ===
            if self.background_dark:
                glClearColor(*Colors.DARK_BG, 1.0)
            else:
                glClearColor(*Colors.LIGHT_BG, 1.0)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            glMatrixMode(GL_PROJECTION)
            glLoadIdentity()
            glOrtho(0, self.width, self.height, 0, -1, 1)
            glMatrixMode(GL_MODELVIEW)
            glLoadIdentity()

            # Farby textu
            title_col = Colors.TEXT_DARK if self.background_dark else Colors.TEXT_LIGHT
            sub_col = Colors.TEXT_DARK_SEC if self.background_dark else Colors.TEXT_LIGHT_SEC

            cx, cy = self.width // 2, self.height // 2

            # Titulok
            self.ui_renderer.draw_text_2d("Linear Algebra Visualizer",
                                          (cx - 160, cy - 100), title_col, 28)
            self.ui_renderer.draw_text_2d("Interaktívna vizualizácia vektorových operácií",
                                          (cx - 170, cy - 50), sub_col, 16)
            self.ui_renderer.draw_text_2d("Vyberte režim zobrazenia:",
                                          (cx - 95, cy - 10), sub_col, 15)

            # Tlačidlá
            btn_w, btn_h = 180, 52
            gap = 24

            btn_2d = pygame.Rect(cx - btn_w - gap // 2, cy + 30, btn_w, btn_h)
            btn_3d = pygame.Rect(cx + gap // 2, cy + 30, btn_w, btn_h)
            toggle = pygame.Rect(30, self.height - 60, 130, 38)

            hover_2d = btn_2d.collidepoint(mx, my)
            hover_3d = btn_3d.collidepoint(mx, my)
            hover_toggle = toggle.collidepoint(mx, my)

            self.ui_renderer.draw_button_2d(btn_2d.x, btn_2d.y, btn_2d.w, btn_2d.h,
                                            "2D Režim", is_dark=self.background_dark,
                                            hover=hover_2d, primary=True)
            self.ui_renderer.draw_button_2d(btn_3d.x, btn_3d.y, btn_3d.w, btn_3d.h,
                                            "3D Režim", is_dark=self.background_dark,
                                            hover=hover_3d, primary=True)

            theme_label = "Tmavý režim" if self.background_dark else "Svetlý režim"
            self.ui_renderer.draw_button_2d(toggle.x, toggle.y, toggle.w, toggle.h,
                                            theme_label, is_dark=self.background_dark,
                                            hover=hover_toggle)

            # Verzia
            self.ui_renderer.draw_text_2d("v2.0", (self.width - 50, self.height - 30), sub_col, 12)

            pygame.display.flip()

    def run_baza_input(self):
        """Spustí obrazovku pre zadanie bázy - KOMPLETNÁ OPRAVENÁ VERZIA"""
        pending_input_panel = None
        span_input_panel = None

        while self.is_not_baza and self.running:
            dt = self.clock.tick(Config.FPS) / 1000.0

            rows = 2 if self.view_2d_mode else 3
            cols = 2 if self.view_2d_mode else 3

            # Bezpečnejšia kontrola matrix_inputs
            needs_reinit = len(self.matrix_inputs) != rows
            if not needs_reinit and len(self.matrix_inputs) > 0:
                needs_reinit = len(self.matrix_inputs[0]) != cols
            if needs_reinit:
                self.matrix_inputs = [["" for _ in range(cols)] for _ in range(rows)]

            mx, my = pygame.mouse.get_pos()

            # =================================================================
            # DEFINUJ VŠETKY RECT-y TU - PRED EVENT HANDLING AJ RENDEROVANÍM
            # =================================================================
            back_rect = pygame.Rect(20, 15, 90, 36)
            toggle_rect = pygame.Rect(30, self.height - 55, 130, 38)

            basis_x = self.width - 210
            basis_y = 70
            std_basis_rect = pygame.Rect(basis_x, basis_y, 190, 44)
            span_rect = pygame.Rect(basis_x, basis_y + 55, 190, 44)

            card_x, card_y = 25, 70
            card_w, card_h = 200, 290
            card2_x = card_x + card_w + 20

            btn_w = card_w - 30
            btn_h = 42
            btn_gap = 8
            btn_y_start = card_y + 45

            vec_op_names = ["Sčítania", "Odčitanie", "Násobenie Konštantou", "Lineárna kombinácia"]
            vec_op_rects = []
            for i in range(4):
                rect = pygame.Rect(card_x + 15, btn_y_start + i * (btn_h + btn_gap), btn_w, btn_h)
                vec_op_rects.append(rect)

            mat_op_rects = []
            for i in range(4):
                rect = pygame.Rect(card2_x + 15, btn_y_start + i * (btn_h + btn_gap), btn_w, btn_h)
                mat_op_rects.append(rect)

            matrix_x = basis_x + 25
            matrix_y = basis_y + 165
            cell_w, cell_h = 45, 34
            cell_gap = 5

            matrix_cell_rects = []
            for r in range(rows):
                row_rects = []
                for c in range(cols):
                    rect = pygame.Rect(matrix_x + c * (cell_w + cell_gap),
                                       matrix_y + r * (cell_h + cell_gap), cell_w, cell_h)
                    row_rects.append(rect)
                matrix_cell_rects.append(row_rects)

            # =================================================================
            # EVENT HANDLING
            # =================================================================
            restart_loop = False

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
                    # === Späť ===
                    if back_rect.collidepoint(mx, my):
                        self.startup_screen = True
                        self.is_not_baza = True
                        pending_input_panel = None
                        span_input_panel = None
                        self.matrix_active_cell = (-1, -1)
                        self.run_startup_screen()
                        if not self.running:
                            return
                        new_rows = 2 if self.view_2d_mode else 3
                        new_cols = 2 if self.view_2d_mode else 3
                        self.matrix_inputs = [["" for _ in range(new_cols)] for _ in range(new_rows)]
                        restart_loop = True
                        break

                    # === Štandardná báza ===
                    if std_basis_rect.collidepoint(mx, my):
                        if self.view_2d_mode:
                            self.saved_baza.append(((1, 0), (0, 1)))
                        else:
                            self.saved_baza.append(((1, 0, 0), (0, 1, 0), (0, 0, 1)))
                        print("Použitá štandardná báza (identitná matica)")
                        self.is_not_baza = False
                        continue

                    # === SPAN TLAČIDLO ===
                    if span_rect.collidepoint(mx, my) and self.view_2d_mode:
                        panel_width = 2 * (Config.MATRIX_CELL_W + Config.MATRIX_GAP)
                        panel_height = 2 * (Config.MATRIX_CELL_H + Config.MATRIX_GAP)

                        start_x = self.width // 2 - panel_width // 2
                        start_y = self.height // 2 - panel_height // 2

                        span_input_panel = {
                            'values': [["" for _ in range(2)] for _ in range(2)],
                            'active_cell': (0, 0),
                            'x': start_x,
                            'y': start_y
                        }
                        pending_input_panel = None
                        self.matrix_active_cell = (-1, -1)
                        continue

                    # === Theme toggle ===
                    if toggle_rect.collidepoint(mx, my):
                        self.background_dark = not self.background_dark

                    # === VEKTOROVÉ OPERÁCIE ===
                    for i, rect in enumerate(vec_op_rects):
                        if rect.collidepoint(mx, my):
                            name = vec_op_names[i]
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
                                num_panels = 4
                                has_constant = False

                                constant_width = 60
                                vector_panel_width = Config.MATRIX_CELL_W + Config.MATRIX_GAP
                                symbol_width = 30

                                total_width = (constant_width + vector_panel_width + symbol_width) * 2
                                start_x = self.width // 2 - total_width // 2

                                panels = []
                                current_x = start_x

                                panels.append({
                                    "type": "constant", "rows": 1, "cols": 1,
                                    "values": [[""]], "active_cell": (0, 0),
                                    "x": current_x,
                                    "y": self.height // 2 - Config.MATRIX_CELL_H // 2,
                                    "label": "c1"
                                })
                                current_x += constant_width + 10

                                panels.append({
                                    "type": "vector", "rows": rows_panel, "cols": 1,
                                    "values": [["" for _ in range(1)] for _ in range(rows_panel)],
                                    "active_cell": (-1, -1),
                                    "x": current_x,
                                    "y": self.height // 2 - (
                                                rows_panel * (Config.MATRIX_CELL_H + Config.MATRIX_GAP)) // 2,
                                    "label": "v1"
                                })
                                current_x += vector_panel_width + 30

                                panels.append({
                                    "type": "constant", "rows": 1, "cols": 1,
                                    "values": [[""]], "active_cell": (-1, -1),
                                    "x": current_x,
                                    "y": self.height // 2 - Config.MATRIX_CELL_H // 2,
                                    "label": "c2"
                                })
                                current_x += constant_width + 10

                                panels.append({
                                    "type": "vector", "rows": rows_panel, "cols": 1,
                                    "values": [["" for _ in range(1)] for _ in range(rows_panel)],
                                    "active_cell": (-1, -1),
                                    "x": current_x,
                                    "y": self.height // 2 - (
                                                rows_panel * (Config.MATRIX_CELL_H + Config.MATRIX_GAP)) // 2,
                                    "label": "v2"
                                })

                                pending_input_panel = {
                                    "type": "vector", "operation": name, "symbol": None,
                                    "num_panels": len(panels), "has_constant": False,
                                    "panels": panels, "active_panel": 0,
                                    "is_linear_combination": True
                                }
                                break
                            else:
                                symbol = None
                                num_panels = 1
                                has_constant = False

                            if name != "Lineárna kombinácia":
                                panel_width = Config.MATRIX_CELL_W + Config.MATRIX_GAP
                                constant_width = 60 if has_constant else 0
                                symbol_width = 40 if (symbol and not has_constant) else (30 if has_constant else 0)
                                total_width = num_panels * panel_width + symbol_width + constant_width
                                start_x = self.width // 2 - total_width // 2

                                panels = []
                                if has_constant:
                                    panels.append({
                                        "type": "constant", "rows": 1, "cols": 1,
                                        "values": [[""]], "active_cell": (0, 0),
                                        "x": start_x,
                                        "y": self.height // 2 - Config.MATRIX_CELL_H // 2
                                    })
                                    panels.append({
                                        "type": "vector", "rows": rows_panel, "cols": 1,
                                        "values": [["" for _ in range(1)] for _ in range(rows_panel)],
                                        "active_cell": (-1, -1),
                                        "x": start_x + constant_width + symbol_width,
                                        "y": self.height // 2 - (
                                                    rows_panel * (Config.MATRIX_CELL_H + Config.MATRIX_GAP)) // 2
                                    })
                                else:
                                    for j in range(num_panels):
                                        panels.append({
                                            "type": "vector", "rows": rows_panel, "cols": 1,
                                            "values": [["" for _ in range(1)] for _ in range(rows_panel)],
                                            "active_cell": (0, 0) if j == 0 else (-1, -1),
                                            "x": start_x + j * (panel_width + symbol_width),
                                            "y": self.height // 2 - (
                                                        rows_panel * (Config.MATRIX_CELL_H + Config.MATRIX_GAP)) // 2
                                        })

                                pending_input_panel = {
                                    "type": "vector", "operation": name, "symbol": symbol,
                                    "num_panels": len(panels), "has_constant": has_constant,
                                    "panels": panels, "active_panel": 0
                                }
                            break

                    # === MATICOVÉ OPERÁCIE ===
                    for i, rect in enumerate(mat_op_rects):
                        if rect.collidepoint(mx, my):
                            name = vec_op_names[i]
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
                                num_panels = 4
                                has_constant = False

                                constant_width = 60
                                matrix_panel_width = cols_panel * (Config.MATRIX_CELL_W + Config.MATRIX_GAP)
                                symbol_width = 30

                                total_width = (constant_width + matrix_panel_width + symbol_width) * 2
                                start_x = self.width // 2 - total_width // 2

                                panels = []
                                current_x = start_x

                                panels.append({
                                    "type": "constant", "rows": 1, "cols": 1,
                                    "values": [[""]], "active_cell": (0, 0),
                                    "x": current_x,
                                    "y": self.height // 2 - Config.MATRIX_CELL_H // 2,
                                    "label": "c1"
                                })
                                current_x += constant_width + 10

                                panels.append({
                                    "type": "matrix", "rows": rows_panel, "cols": cols_panel,
                                    "values": [["" for _ in range(cols_panel)] for _ in range(rows_panel)],
                                    "active_cell": (-1, -1),
                                    "x": current_x,
                                    "y": self.height // 2 - (
                                                rows_panel * (Config.MATRIX_CELL_H + Config.MATRIX_GAP)) // 2,
                                    "label": "mat1"
                                })
                                current_x += matrix_panel_width + 30

                                panels.append({
                                    "type": "constant", "rows": 1, "cols": 1,
                                    "values": [[""]], "active_cell": (-1, -1),
                                    "x": current_x,
                                    "y": self.height // 2 - Config.MATRIX_CELL_H // 2,
                                    "label": "c2"
                                })
                                current_x += constant_width + 10

                                panels.append({
                                    "type": "matrix", "rows": rows_panel, "cols": cols_panel,
                                    "values": [["" for _ in range(cols_panel)] for _ in range(rows_panel)],
                                    "active_cell": (-1, -1),
                                    "x": current_x,
                                    "y": self.height // 2 - (
                                                rows_panel * (Config.MATRIX_CELL_H + Config.MATRIX_GAP)) // 2,
                                    "label": "mat2"
                                })

                                pending_input_panel = {
                                    "type": "matrix", "operation": name, "symbol": None,
                                    "num_panels": len(panels), "has_constant": False,
                                    "panels": panels, "active_panel": 0,
                                    "is_linear_combination": True
                                }
                                break
                            else:
                                symbol = None
                                num_panels = 1
                                has_constant = False

                            if name != "Lineárna kombinácia":
                                panel_width = cols_panel * (Config.MATRIX_CELL_W + Config.MATRIX_GAP)
                                constant_width = 60 if has_constant else 0
                                symbol_width = 40 if (symbol and not has_constant) else (30 if has_constant else 0)
                                total_width = num_panels * panel_width + symbol_width + constant_width
                                start_x = self.width // 2 - total_width // 2

                                panels = []
                                if has_constant:
                                    panels.append({
                                        "type": "constant", "rows": 1, "cols": 1,
                                        "values": [[""]], "active_cell": (0, 0),
                                        "x": start_x,
                                        "y": self.height // 2 - Config.MATRIX_CELL_H // 2
                                    })
                                    panels.append({
                                        "type": "matrix", "rows": rows_panel, "cols": cols_panel,
                                        "values": [["" for _ in range(cols_panel)] for _ in range(rows_panel)],
                                        "active_cell": (-1, -1),
                                        "x": start_x + constant_width + symbol_width,
                                        "y": self.height // 2 - (
                                                    rows_panel * (Config.MATRIX_CELL_H + Config.MATRIX_GAP)) // 2
                                    })
                                else:
                                    for j in range(num_panels):
                                        panels.append({
                                            "type": "matrix", "rows": rows_panel, "cols": cols_panel,
                                            "values": [["" for _ in range(cols_panel)] for _ in range(rows_panel)],
                                            "active_cell": (0, 0) if j == 0 else (-1, -1),
                                            "x": start_x + j * (panel_width + symbol_width),
                                            "y": self.height // 2 - (
                                                        rows_panel * (Config.MATRIX_CELL_H + Config.MATRIX_GAP)) // 2
                                        })

                                pending_input_panel = {
                                    "type": "matrix", "operation": name, "symbol": symbol,
                                    "num_panels": len(panels), "has_constant": has_constant,
                                    "panels": panels, "active_panel": 0
                                }
                            break

                    # === Kliknutie na input panel ===
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

                    # === SPAN PANEL KLIKNUTIA ===
                    if span_input_panel:
                        clicked = False
                        for r in range(2):
                            for c in range(2):
                                x = span_input_panel['x'] + c * (Config.MATRIX_CELL_W + Config.MATRIX_GAP)
                                y = span_input_panel['y'] + r * (Config.MATRIX_CELL_H + Config.MATRIX_GAP)
                                rect = pygame.Rect(x, y, Config.MATRIX_CELL_W, Config.MATRIX_CELL_H)
                                if rect.collidepoint(mx, my):
                                    span_input_panel['active_cell'] = (r, c)
                                    clicked = True
                                    break
                            if clicked:
                                break
                        if clicked:
                            continue

                    # === Matrix cells pre vlastnú bázu ===
                    if not pending_input_panel and not span_input_panel:
                        for r in range(rows):
                            for c in range(cols):
                                if matrix_cell_rects[r][c].collidepoint(mx, my):
                                    self.matrix_active_cell = (r, c)
                                    break

                elif event.type == pygame.KEYDOWN:
                    # === ESC ===
                    if event.key == pygame.K_ESCAPE:
                        if span_input_panel:
                            span_input_panel = None
                            continue
                        if pending_input_panel:
                            pending_input_panel = None
                            continue
                        else:
                            self.startup_screen = True
                            self.is_not_baza = True
                            self.matrix_active_cell = (-1, -1)
                            span_input_panel = None
                            self.run_startup_screen()
                            if not self.running:
                                return
                            new_rows = 2 if self.view_2d_mode else 3
                            new_cols = 2 if self.view_2d_mode else 3
                            self.matrix_inputs = [["" for _ in range(new_cols)] for _ in range(new_rows)]
                            restart_loop = True
                            break

                    # KRITICKÁ OPRAVA: Inicializuj tieto premenné NA ZAČIATKU keydown handlera
                    span_handled = False
                    pending_handled = False

                    # === SPAN PANEL INPUT ===
                    if span_input_panel:
                        r, c = span_input_panel['active_cell']
                        span_handled = True

                        if event.key == pygame.K_RETURN:
                            all_filled = True
                            for row in span_input_panel['values']:
                                for val in row:
                                    if not val.strip():
                                        all_filled = False
                                        break

                            if all_filled:
                                try:
                                    v1 = [float(span_input_panel['values'][0][0]),
                                          float(span_input_panel['values'][1][0])]
                                    v2 = [float(span_input_panel['values'][0][1]),
                                          float(span_input_panel['values'][1][1])]

                                    self.vector_manager.span_controller.setup_span(v1, v2)
                                    self.saved_baza.append(((1, 0), (0, 1)))

                                    span_input_panel = None
                                    self.is_not_baza = False
                                    print(f"Span nastavený pre vektory v1={v1}, v2={v2}")
                                except Exception as e:
                                    print("Chyba pri parsovaní span matice:", e)

                        elif event.key == pygame.K_BACKSPACE:
                            span_input_panel['values'][r][c] = span_input_panel['values'][r][c][:-1]
                        elif event.unicode.isdigit() or event.unicode in ".-":
                            span_input_panel['values'][r][c] += event.unicode

                    # === Pending panel input ===
                    if not span_handled and pending_input_panel:
                        active_panel_idx = pending_input_panel["active_panel"]
                        panel = pending_input_panel["panels"][active_panel_idx]
                        r, c = panel["active_cell"]

                        if r != -1 and c != -1:
                            pending_handled = True
                            if event.key == pygame.K_RETURN:
                                all_filled = True
                                for p in pending_input_panel["panels"]:
                                    for row in p["values"]:
                                        for val in row:
                                            if not val.strip():
                                                all_filled = False
                                                break

                                if all_filled:
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

                                    if pending_input_panel["type"] == "vector":
                                        if operation == "Sčítania":
                                            vec1 = [float(v[0]) for v in data_panels[0]["values"]]
                                            vec2 = [float(v[0]) for v in data_panels[1]["values"]]
                                            result = [a + b for a, b in zip(vec1, vec2)]
                                            operands = [vec1, vec2]
                                            self.vector_manager.animation_controller.setup_operation('add', operands,
                                                                                                     result)
                                        elif operation == "Odčitanie":
                                            vec1 = [float(v[0]) for v in data_panels[0]["values"]]
                                            vec2 = [float(v[0]) for v in data_panels[1]["values"]]
                                            result = [a - b for a, b in zip(vec1, vec2)]
                                            operands = [vec1, vec2]
                                            self.vector_manager.animation_controller.setup_operation('subtract',
                                                                                                     operands, result)
                                        elif operation == "Násobenie Konštantou":
                                            vec = [float(v[0]) for v in data_panels[0]["values"]]
                                            result = [constant[0] * v for v in vec]
                                            operands = [vec]
                                            self.vector_manager.animation_controller.setup_operation('scalar_mult',
                                                                                                     operands, result,
                                                                                                     constant[0])
                                        elif operation == "Lineárna kombinácia":
                                            vec1 = [float(v[0]) for v in data_panels[0]["values"]]
                                            vec2 = [float(v[0]) for v in data_panels[1]["values"]]
                                            c1, c2 = constant[0], constant[1]
                                            result = [c1 * v1 + c2 * v2 for v1, v2 in zip(vec1, vec2)]
                                            operands = [vec1, vec2]
                                            self.vector_manager.animation_controller.setup_operation(
                                                'linear_combination', operands, result, [c1, c2])
                                    else:
                                        if operation == "Sčítania":
                                            mat1 = [[float(v) for v in row] for row in data_panels[0]["values"]]
                                            mat2 = [[float(v) for v in row] for row in data_panels[1]["values"]]
                                            result = [[a + b for a, b in zip(row1, row2)] for row1, row2 in
                                                      zip(mat1, mat2)]
                                            operands = [mat1, mat2]
                                            self.vector_manager.animation_controller.setup_operation('add', operands,
                                                                                                     result)
                                        elif operation == "Odčitanie":
                                            mat1 = [[float(v) for v in row] for row in data_panels[0]["values"]]
                                            mat2 = [[float(v) for v in row] for row in data_panels[1]["values"]]
                                            result = [[a - b for a, b in zip(row1, row2)] for row1, row2 in
                                                      zip(mat1, mat2)]
                                            operands = [mat1, mat2]
                                            self.vector_manager.animation_controller.setup_operation('subtract',
                                                                                                     operands, result)
                                        elif operation == "Násobenie Konštantou":
                                            mat = [[float(v) for v in row] for row in data_panels[0]["values"]]
                                            result = [[constant[0] * v for v in row] for row in mat]
                                            operands = [mat]
                                            self.vector_manager.animation_controller.setup_operation('scalar_mult',
                                                                                                     operands, result,
                                                                                                     constant[0])
                                        elif operation == "Lineárna kombinácia":
                                            mat1 = [[float(v) for v in row] for row in data_panels[0]["values"]]
                                            mat2 = [[float(v) for v in row] for row in data_panels[1]["values"]]
                                            c1, c2 = constant[0], constant[1]
                                            result = [[c1 * v1 + c2 * v2 for v1, v2 in zip(vec1, vec2)] for vec1, vec2
                                                      in zip(mat1, mat2)]
                                            operands = [mat1, mat2]
                                            self.vector_manager.animation_controller.setup_operation(
                                                'linear_combination', operands, result, [c1, c2])

                                    if not self.view_2d_mode:
                                        plane = self.vector_manager.animation_controller.current_plane
                                        if plane:
                                            max_val = self.get_max_from_vectors()
                                            distance = max(25.0, max_val * 3.0 + 5)
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

                    # === Originálna báza input ===
                    if not span_handled and not pending_handled and self.matrix_active_cell != (-1, -1):
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

            if restart_loop:
                continue

            # =================================================================
            # RENDER
            # =================================================================
            if self.background_dark:
                glClearColor(*Colors.DARK_BG, 1.0)
            else:
                glClearColor(*Colors.LIGHT_BG, 1.0)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            glMatrixMode(GL_PROJECTION)
            glLoadIdentity()
            glOrtho(0, self.width, self.height, 0, -1, 1)
            glMatrixMode(GL_MODELVIEW)
            glLoadIdentity()

            text_col = Colors.TEXT_DARK if self.background_dark else Colors.TEXT_LIGHT
            sub_col = Colors.TEXT_DARK_SEC if self.background_dark else Colors.TEXT_LIGHT_SEC

            # === SPÄŤ ===
            hover_back = back_rect.collidepoint(mx, my)
            self.ui_renderer.draw_button_2d(back_rect.x, back_rect.y, back_rect.w, back_rect.h,
                                            "← Späť", is_dark=self.background_dark, hover=hover_back)
            self.ui_renderer.draw_text_2d("ESC", (120, 25), sub_col, 12)

            # === TITULOK ===
            mode_str = "2D" if self.view_2d_mode else "3D"
            self.ui_renderer.draw_text_2d(f"Operácie ({mode_str})", (self.width // 2 - 60, 20), text_col, 22)

            # === VEKTOROVÉ OPERÁCIE ===
            self.ui_renderer.draw_card(card_x, card_y, card_w, card_h, self.background_dark, "Vektorové operácie")
            vec_labels = ["Sčítanie", "Odčítanie", "Násobenie k.", "Lin. kombinácia"]
            for i, rect in enumerate(vec_op_rects):
                hover = rect.collidepoint(mx, my)
                self.ui_renderer.draw_button_2d(rect.x, rect.y, rect.w, rect.h,
                                                vec_labels[i], is_dark=self.background_dark, hover=hover)

            # === MATICOVÉ OPERÁCIE ===
            self.ui_renderer.draw_card(card2_x, card_y, card_w, card_h, self.background_dark, "Maticové operácie")
            for i, rect in enumerate(mat_op_rects):
                hover = rect.collidepoint(mx, my)
                self.ui_renderer.draw_button_2d(rect.x, rect.y, rect.w, rect.h,
                                                vec_labels[i], is_dark=self.background_dark, hover=hover)

            # === ŠTANDARDNÁ BÁZA ===
            hover_std = std_basis_rect.collidepoint(mx, my)
            self.ui_renderer.draw_button_2d(std_basis_rect.x, std_basis_rect.y,
                                            std_basis_rect.w, std_basis_rect.h,
                                            "Štandardná báza", is_dark=self.background_dark,
                                            hover=hover_std, primary=True)
            basis_desc = "Identitná matica I₂" if self.view_2d_mode else "Identitná matica I₃"
            self.ui_renderer.draw_text_2d(basis_desc, (basis_x + 30, basis_y + 52), sub_col, 13)

            # === SPAN TLAČIDLO ===
            if self.view_2d_mode:
                hover_span = span_rect.collidepoint(mx, my)
                self.ui_renderer.draw_button_2d(span_rect.x, span_rect.y,
                                                span_rect.w, span_rect.h,
                                                "Span (Lineárny obal)",
                                                is_dark=self.background_dark,
                                                hover=hover_span, primary=True)
                self.ui_renderer.draw_text_2d("Zadaj 2 vektory (stĺpce)",
                                              (basis_x + 20, span_rect.y + 48),
                                              sub_col, 13)

            # === VLASTNÁ BÁZA ===
            self.ui_renderer.draw_card(basis_x, basis_y + 120, 190, 180, self.background_dark, "Vlastná báza")
            for r in range(rows):
                for c in range(cols):
                    rect = matrix_cell_rects[r][c]
                    active = (r, c) == self.matrix_active_cell
                    col_color = Config.COL_COLORS_OUTLINE[c] if c < len(Config.COL_COLORS_OUTLINE) else None
                    self.ui_renderer.draw_input_box_3d(rect.x, rect.y, rect.w, rect.h,
                                                       self.matrix_inputs[r][c], active,
                                                       fill_color_outline=col_color,
                                                       is_dark=self.background_dark)

            # === THEME TOGGLE ===
            hover_toggle = toggle_rect.collidepoint(mx, my)
            theme_label = "Tmavý režim" if self.background_dark else "Svetlý režim"
            self.ui_renderer.draw_button_2d(toggle_rect.x, toggle_rect.y, toggle_rect.w, toggle_rect.h,
                                            theme_label, is_dark=self.background_dark, hover=hover_toggle)

            # === PENDING INPUT PANEL ===
            if pending_input_panel:
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
                            label = panel.get("label", "")
                            if label:
                                self.ui_renderer.draw_text_2d(label, (x + 15, y - 25), color=color_symbol, font_size=18)
                        else:
                            for r in range(panel["rows"]):
                                for c in range(panel["cols"]):
                                    x = panel["x"] + c * (Config.MATRIX_CELL_W + Config.MATRIX_GAP)
                                    y = panel["y"] + r * (Config.MATRIX_CELL_H + Config.MATRIX_GAP)
                                    active = is_active_panel and (r, c) == panel["active_cell"]
                                    self.ui_renderer.draw_input_box_3d(x, y, Config.MATRIX_CELL_W,
                                                                       Config.MATRIX_CELL_H,
                                                                       panel["values"][r][c], active)
                            label = panel.get("label", "")
                            if label:
                                self.ui_renderer.draw_text_2d(label, (panel["x"] + 10, panel["y"] - 25),
                                                              color=color_symbol, font_size=18)

                        if panel_idx == 0:
                            font_size_sym = 30
                            symbol_x = panel["x"] + 50 + 5
                            symbol_y = panel["y"] + Config.MATRIX_CELL_H // 2 - font_size_sym // 3
                            self.ui_renderer.draw_text_2d("×", (symbol_x, symbol_y), color=color_symbol,
                                                          font_size=font_size_sym)
                        elif panel_idx == 1:
                            font_size_sym = 35
                            symbol_x = panel["x"] + panel["cols"] * (Config.MATRIX_CELL_W + Config.MATRIX_GAP) + 10
                            panel_total_height = panel["rows"] * (
                                        Config.MATRIX_CELL_H + Config.MATRIX_GAP) - Config.MATRIX_GAP
                            symbol_y = panel["y"] + (panel_total_height - font_size_sym) // 2
                            self.ui_renderer.draw_text_2d("+", (symbol_x, symbol_y), color=color_symbol,
                                                          font_size=font_size_sym)
                        elif panel_idx == 2:
                            font_size_sym = 30
                            symbol_x = panel["x"] + 50 + 5
                            symbol_y = panel["y"] + Config.MATRIX_CELL_H // 2 - font_size_sym // 3
                            self.ui_renderer.draw_text_2d("×", (symbol_x, symbol_y), color=color_symbol,
                                                          font_size=font_size_sym)
                else:
                    color_symbol = (1, 1, 1) if self.background_dark else (0, 0, 0)
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

                        if pending_input_panel["symbol"]:
                            if pending_input_panel.get("has_constant") and panel_idx == 0:
                                font_size_sym = 40
                                symbol_x = panel["x"] + 50 + 5
                                symbol_y = panel["y"] + Config.MATRIX_CELL_H // 2 - font_size_sym // 3
                                self.ui_renderer.draw_text_2d(pending_input_panel["symbol"], (symbol_x, symbol_y),
                                                              color=color_symbol, font_size=font_size_sym)
                            elif not pending_input_panel.get("has_constant") and panel_idx < len(
                                    pending_input_panel["panels"]) - 1:
                                font_size_sym = 40
                                symbol_x = panel["x"] + panel["cols"] * (Config.MATRIX_CELL_W + Config.MATRIX_GAP) + 10
                                panel_total_height = panel["rows"] * (
                                            Config.MATRIX_CELL_H + Config.MATRIX_GAP) - Config.MATRIX_GAP
                                symbol_y = panel["y"] + (panel_total_height - font_size_sym) // 2
                                self.ui_renderer.draw_text_2d(pending_input_panel["symbol"], (symbol_x, symbol_y),
                                                              color=color_symbol, font_size=font_size_sym)

            # === SPAN INPUT PANEL ===
            if span_input_panel:
                panel_width = 2 * (Config.MATRIX_CELL_W + Config.MATRIX_GAP) + 40
                panel_height = 2 * (Config.MATRIX_CELL_H + Config.MATRIX_GAP) + 80
                bg_x = span_input_panel['x'] - 20
                bg_y = span_input_panel['y'] - 40

                self.ui_renderer.draw_card(bg_x, bg_y, panel_width, panel_height,
                                           self.background_dark, "Span - Zadaj 2 vektory")

                for r in range(2):
                    for c in range(2):
                        x = span_input_panel['x'] + c * (Config.MATRIX_CELL_W + Config.MATRIX_GAP)
                        y = span_input_panel['y'] + r * (Config.MATRIX_CELL_H + Config.MATRIX_GAP)
                        active = (r, c) == span_input_panel['active_cell']

                        col_color = Config.COL_COLORS_OUTLINE[c] if c < len(Config.COL_COLORS_OUTLINE) else None
                        self.ui_renderer.draw_input_box_3d(x, y, Config.MATRIX_CELL_W, Config.MATRIX_CELL_H,
                                                           span_input_panel['values'][r][c], active,
                                                           fill_color_outline=col_color,
                                                           is_dark=self.background_dark)

                self.ui_renderer.draw_text_2d("v1",
                                              (span_input_panel['x'] + 5, span_input_panel['y'] - 25),
                                              text_col, 16)
                self.ui_renderer.draw_text_2d("v2",
                                              (span_input_panel['x'] + Config.MATRIX_CELL_W + Config.MATRIX_GAP + 5,
                                               span_input_panel['y'] - 25),
                                              text_col, 16)

                help_y = span_input_panel['y'] + 2 * (Config.MATRIX_CELL_H + Config.MATRIX_GAP) + 10
                self.ui_renderer.draw_text_2d("Enter = potvrdiť | ESC = zrušiť",
                                              (span_input_panel['x'] - 50, help_y),
                                              sub_col, 14)

            pygame.display.flip()

    def go_back_to_menu(self):
        """Vráti sa späť do hlavného menu"""
        # Vyčisti aktuálny stav
        self.vector_manager.animation_controller.clear()
        self.vector_manager.animated_vectors.clear()
        self.saved_baza.clear()
        self.camera.reset()
        self.vector_manager.span_controller.clear()

        # Nastav flagy pre reštart
        self.startup_screen = True
        self.is_not_baza = True

        # Vyčisti maticový vstup
        rows = 2 if self.view_2d_mode else 3
        cols = 2 if self.view_2d_mode else 3
        self.matrix_inputs = [["" for _ in range(cols)] for _ in range(rows)]
        self.matrix_active_cell = (-1, -1)

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
        """Spracuje stlačenie klávesy - ROZŠÍRENÉ O SPAN"""

        if self.vector_manager.span_controller.active:

            if event.key == pygame.K_o:
                self.vector_manager.span_controller.show_all_combinations()
                # Nastav kameru na správny zoom
                if self.vector_manager.span_controller.locked_ortho_scale:
                    self.camera.ortho_scale = self.vector_manager.span_controller.locked_ortho_scale
                    self.camera.pan_offset_x = 0
                    self.camera.pan_offset_y = 0
                return

                # NOVÉ: Prepínanie farebnej schémy klávesou F
            if event.key == pygame.K_f:
                span_ctrl = self.vector_manager.span_controller
                span_ctrl.color_scheme = (span_ctrl.color_scheme + 1) % 2
                if span_ctrl.show_all_mode:
                    # Pregeneruj s novou schémou
                    span_ctrl.show_all_combinations()
                scheme_names = ["Rainbow", "Rakúsko"]
                print(f"Farebná schéma: {scheme_names[span_ctrl.color_scheme]}")
                return

            if event.key == pygame.K_p:
                # Toggle auto-play
                self.vector_manager.span_controller.auto_play = not self.vector_manager.span_controller.auto_play
                if self.vector_manager.span_controller.auto_play:
                    print("▶ Auto-play ZAPNUTÝ")
                else:
                    print("⏸ Auto-play VYPNUTÝ")
                return

            if event.key == pygame.K_SPACE:
                self.vector_manager.span_controller.next_step()
                return
            elif event.key == pygame.K_BACKSPACE:
                # Backspace len pre krokovanie, nie mazanie textu
                if not any([
                    self.input_handler.show_input_active,
                    self.input_handler.show_multiplication_active,
                    self.input_handler.show_matrix_size_active,
                    self.input_handler.show_random_range_active,
                    self.matrix_active_cell != (-1, -1)
                ]):
                    self.vector_manager.span_controller.prev_step()
                    return
            elif event.key == pygame.K_c:
                self.vector_manager.span_controller.clear()
                return

        # ===== KROKOVANIE ANIMÁCIE - PÔVODNÝ KÓD =====
        if event.key == pygame.K_SPACE:
            if self.vector_manager.animation_controller.current_operation:
                self.vector_manager.animation_controller.next_step()

                if not self.view_2d_mode:
                    plane = self.vector_manager.animation_controller.current_plane
                    if plane:
                        max_val = self.get_max_from_vectors()
                        distance = max(25.0, max_val * 3.0 + 5)

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

        # BACKSPACE - krokovanie späť alebo mazanie textu
        if event.key == pygame.K_BACKSPACE:
            if not any([
                self.input_handler.show_input_active,
                self.input_handler.show_multiplication_active,
                self.input_handler.show_matrix_size_active,
                self.input_handler.show_random_range_active,
                self.matrix_active_cell != (-1, -1)
            ]):
                if self.vector_manager.animation_controller.current_operation:
                    self.vector_manager.animation_controller.prev_step()

                    if not self.view_2d_mode:
                        plane = self.vector_manager.animation_controller.current_plane
                        if plane:
                            max_val = self.get_max_from_vectors()
                            distance = max(25.0, max_val * 3.0 + 5)

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
                    distance = max(25.0, max_val * 3.0 + 5)

                    if plane == "3D":
                        optimal_view = self.vector_manager.animation_controller.compute_optimal_view_for_current_step()
                        if optimal_view:
                            self.camera.move_to_plane(plane, distance, custom_view=optimal_view)
                        else:
                            self.camera.move_to_plane(plane, distance)
                    else:
                        self.camera.move_to_plane(plane, distance)
            return

        # NOVÉ: Prepínanie plane grid módu
        if event.key == pygame.K_b:
            self.plane_grid_mode = (self.plane_grid_mode + 1) % 6
            return

        if event.key == pygame.K_ESCAPE:
            self.go_back_to_menu()
            self.run_startup_screen()
            if self.running:
                self.run_baza_input()
            return
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

        # Vector input handling
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

        # Multiplication input handling
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
        # === NOVÉ: Tlačidlo späť do menu ===
        back_button_rect = pygame.Rect(20, 10, 120, 40)
        if back_button_rect.collidepoint(mx, my):
            self.go_back_to_menu()
            self.run_startup_screen()
            if self.running:
                self.run_baza_input()
            return

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

        # Camera controls - na konci handle_mouse_down
        if event.button == 1:
            # NOVÉ: Blokuj v show_all režime
            if not self.vector_manager.span_controller.show_all_mode:
                self.camera.rotating = not self.view_2d_mode
                self.camera.panning = self.view_2d_mode
            self.camera.last_mouse = event.pos
        elif event.button == 3:
            # NOVÉ: Blokuj v show_all režime
            if not self.vector_manager.span_controller.show_all_mode:
                self.camera.panning = True
            self.camera.last_mouse = event.pos
        elif event.button == 4:  # scroll up
            span_ctrl = self.vector_manager.span_controller
            if span_ctrl.show_all_mode and span_ctrl.locked_ortho_scale:
                # Obmedzený zoom
                new_scale = self.camera.ortho_scale / 1.1
                min_scale = span_ctrl.locked_ortho_scale * 0.5
                if new_scale >= min_scale:
                    self.camera.ortho_scale = new_scale
            else:
                self.camera.zoom_in(self.view_2d_mode)

        elif event.button == 5:  # scroll down
            span_ctrl = self.vector_manager.span_controller
            if span_ctrl.show_all_mode and span_ctrl.locked_ortho_scale:
                # Obmedzený zoom
                new_scale = self.camera.ortho_scale * 1.1
                max_scale = span_ctrl.locked_ortho_scale * 1.5
                if new_scale <= max_scale:
                    self.camera.ortho_scale = new_scale
            else:
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

        # NOVÉ: Blokuj pohyb kamery v show_all režime
        if self.vector_manager.span_controller.show_all_mode:
            return

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
        """Vykreslí scénu - S FPS LIMITOM"""
        current_time = pygame.time.get_ticks() / 1000.0

        # Skip render ak je príliš skoro
        if current_time - self._last_render_time < self._min_frame_time:
            return

        self._last_render_time = current_time

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        if self.view_2d_mode:
            self.render_2d()
        else:
            self.render_3d()

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

        # NOVÉ: Vykresli body kruhu ak je span aktívny
        # debug purposes
        if self.vector_manager.span_controller.active:
            self.draw_span_circle_points()

        # Draw vectors
        color = (0, 0, 0) if not self.background_dark else (1, 1, 1)
        self.draw_vectors_2d(color)

    def draw_span_circle_points(self):
        """Vykreslí body kruhu pre span vizualizáciu"""
        span_ctrl = self.vector_manager.span_controller

        if not span_ctrl.circle_points or not span_ctrl.basis_vectors:
            return

        v1, v2 = span_ctrl.basis_vectors

        # Farby
        inactive_color = (0.5, 0.5, 0.5)  # Sivá pre neaktívne body
        active_color = (1.0, 0.8, 0.0)  # Žltá pre aktuálny bod
        past_color = (0.3, 0.8, 0.3)  # Zelená pre prejdené body

        # Polomer bodov - škálovaný podľa zoomu
        base_radius = 0.08
        radius = base_radius * self.camera.ortho_scale / 6.5

        # Vykresli všetky body kruhu
        for i, (c1, c2) in enumerate(span_ctrl.circle_points):
            # Vypočítaj pozíciu bodu v priestore
            point_x = c1 * v1[0] + c2 * v2[0]
            point_y = c1 * v1[1] + c2 * v2[1]

            # Urči farbu podľa stavu
            if i == span_ctrl.current_circle_index and span_ctrl.current_step >= 2:
                # Aktuálny bod - žltý
                color = active_color
                point_radius = radius * 1.5  # Väčší
            elif span_ctrl.current_step >= 2 and i < span_ctrl.current_circle_index:
                # Prejdené body - zelené
                color = past_color
                point_radius = radius
            elif span_ctrl.current_step >= 2 and i == 0 and span_ctrl.current_circle_index > 0 and \
                    (span_ctrl.current_step - 2) >= len(span_ctrl.circle_points):
                # Bod 0 po dokončení prvého kola - tiež zelený
                color = past_color
                point_radius = radius
            else:
                # Neprejdené body - sivé
                color = inactive_color
                point_radius = radius * 0.8  # Menší

            # Vykresli bod
            # self.vector_renderer.draw_circle_2d(
                #      [point_x, point_y, 0.1],
                #      radius=point_radius,
                #      color=color,
            #      alpha=0.7
            # )

            # VOLITEĽNE: Vykresli číslo bodu (pre debug)
            if False:  # Zmeň na True ak chceš vidieť čísla
                glMatrixMode(GL_PROJECTION)
                glPushMatrix()
                glLoadIdentity()
                width, height = pygame.display.get_window_size()

                aspect = width / height
                if aspect >= 1.0:
                    left = -self.camera.ortho_scale * aspect + self.camera.pan_offset_x
                    right = self.camera.ortho_scale * aspect + self.camera.pan_offset_x
                    bottom = -self.camera.ortho_scale + self.camera.pan_offset_y
                    top = self.camera.ortho_scale + self.camera.pan_offset_y
                else:
                    left = -self.camera.ortho_scale + self.camera.pan_offset_x
                    right = self.camera.ortho_scale + self.camera.pan_offset_x
                    bottom = -self.camera.ortho_scale / aspect + self.camera.pan_offset_y
                    top = self.camera.ortho_scale / aspect + self.camera.pan_offset_y

                glOrtho(0, width, height, 0, -1, 1)
                glMatrixMode(GL_MODELVIEW)
                glPushMatrix()
                glLoadIdentity()

                # Prepočítaj na screen coordinates
                sx = (point_x - left) / (right - left) * width
                sy = height - (point_y - bottom) / (top - bottom) * height

                text_color = (0, 0, 0) if not self.background_dark else (1, 1, 1)
                self.ui_renderer.draw_text_2d(str(i), (sx - 5, sy - 20),
                                              color=text_color, font_size=14)

                glPopMatrix()
                glMatrixMode(GL_PROJECTION)
                glPopMatrix()
                glMatrixMode(GL_MODELVIEW)


    def render_3d(self):
        """
        Vykreslí 3D scénu so správnym vrstvením:
        1. Výplň roviny (zapisuje do hĺbky, aby skryla veci za sebou)
        2. Mriežky (globálne aj v rovine)
        3. Osi (budú schované za výplňou roviny)
        4. Vektory (vždy navrchu vďaka vyčisteniu hĺbky)
        """
        self.camera.setup_3d_projection()

        # ZÁKLADNÉ NASTAVENIE HĹBKY
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LESS)
        glDepthMask(GL_TRUE)

        length_xyz = max(10, self.get_max_from_vectors())

        if length_xyz > 10:
            length_xyz = length_xyz + 3

        # Získame dáta o operácii pre rovinu
        ctrl = self.vector_manager.animation_controller
        has_operation = ctrl.current_operation is not None
        plane_active = has_operation and self.plane_grid_mode > 0

        # =========================================================
        # 1. KROK: VÝPLŇ ROVINY (Prekážka pre hĺbku)
        # =========================================================
        if plane_active:
            plane = ctrl.current_plane
            normal = ctrl.operation_plane_normal

            if plane and normal and self.plane_grid_mode in (4, 5, 6, 7):
                fill_color = (0.6, 0.6, 0) if self.background_dark else (0.8, 0.8, 0.4)

                # Nastavenie priehľadnosti
                is_transparent = self.plane_grid_mode in (4, 6)
                if is_transparent:
                    glEnable(GL_BLEND)
                    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
                    alpha = 0.25
                else:
                    glDisable(GL_BLEND)
                    alpha = 1.0

                # POLYGON OFFSET: Odtlačíme výplň o kúsok dozadu,
                # aby čiary mriežky, ktoré sú na nej, nepreblikovali (Z-fighting)
                glEnable(GL_POLYGON_OFFSET_FILL)
                glPolygonOffset(1.0, 1.0)

                self.grid_renderer.draw_filled_plane(
                    normal=normal, center=[0, 0, 0], size=length_xyz,
                    color=fill_color, alpha=alpha, transparent=is_transparent
                )

                glDisable(GL_POLYGON_OFFSET_FILL)

        # =========================================================
        # 2. KROK: GLOBÁLNE MRIEŽKY (XYZ PLANES)
        # =========================================================
        if self.grid_mode == 1:
            self.grid_renderer.draw_planes_3d(length_xyz)
        elif self.grid_mode == 2:
            self.grid_renderer.draw_planes_3d(length_xyz, colored=True)
        elif self.grid_mode == 3:
            self.grid_renderer.draw_grid_3d(length_xyz)

        # =========================================================
        # 3. KROK: MRIEŽKA V ROVINE OPERÁCIE
        # =========================================================
        if plane_active:
            plane = ctrl.current_plane
            normal = ctrl.operation_plane_normal

            if plane and normal and self.plane_grid_mode in (1, 2, 3, 4, 5):
                # Nastavenie parametrov mriežky podľa módu
                if self.plane_grid_mode == 1:
                    grid_color = (0.6, 0.6, 0) if self.background_dark else (0.4, 0.4, 0)
                    step = 1.0
                elif self.plane_grid_mode == 2:
                    grid_color = (0, 0.6, 0.6) if self.background_dark else (0, 0.4, 0.4)
                    step = 0.5
                elif self.plane_grid_mode == 3:
                    grid_color = (0.6, 0, 0.6) if self.background_dark else (0.4, 0, 0.4)
                    step = 2.0
                else:  # Mód 4, 5
                    grid_color = (0.2, 0.2, 0.2) if self.background_dark else (0.5, 0.5, 0.5)
                    step = 1.0

                self.grid_renderer.draw_grid_in_plane(
                    normal=normal, center=[0, 0, 0], size=length_xyz,
                    step=step, color=grid_color
                )

        # =========================================================
        # 4. KROK: OSI (Budú schované za rovinou, ak sú "pod" ňou)
        # =========================================================
        if self.show_axes:
            cam_pos = self.camera.get_position()
            # Dôležité: Osi teraz rešpektujú hĺbku zapísanú rovinou v Kroku 1
            self.axes_renderer.draw_axes_3d(length_xyz, cam_pos, self.vector_renderer)

        # =========================================================
        # 5. KROK: VEKTORY (Navrchu)
        # =========================================================
        # Vymažeme hĺbkový buffer. Tým povieme OpenGL: "Zabudni, ako hlboko sú osi a rovina."
        # Všetko, čo nakreslíme teraz, bude navrchu.
        #glClear(GL_DEPTH_BUFFER_BIT)

        if self.grid_mode in (1, 2):
            v_color = (0, 0, 0)
        else:
            v_color = (0, 0, 0) if not self.background_dark else (1, 1, 1)

        self.draw_vectors_3d(v_color)

    def _update_background_color(self):
        """Aktualizuje farbu pozadia podľa témy"""
        if self.background_dark:
            glClearColor(*Colors.DARK_BG, 1.0)
        else:
            glClearColor(*Colors.LIGHT_BG, 1.0)

    def draw_vectors_2d(self, color):
        """Vykreslí vektory v 2D - S PODPOROU PRE SPAN"""
        arrow_size = max(0.15, min(0.1 * self.camera.ortho_scale, 2.0))
        # ===== SPAN MÁ PRIORITU =====
        if self.vector_manager.span_controller.active:
            vectors_to_draw = self.vector_manager.span_controller.get_current_vectors()

            # Rozdeľ na perzistentné (vzadu) a aktuálne (vpredu)
            persistent_vectors = [v for v in vectors_to_draw if v.get('is_persistent', False)]
            current_vectors = [v for v in vectors_to_draw if not v.get('is_persistent', False)]

            if self.vectors_as_points:
                # 1. Najprv kresli perzistentné (vzadu, Z = 0.05)
                for v in persistent_vectors:
                    vec = v['vec']
                    offset = v.get('offset', [0, 0])
                    v_color = v.get('color', (0.6, 0.2, 0.6))
                    v_alpha = v.get('alpha', 0.4)
                    base_radius = 0.07
                    radius = base_radius * self.camera.ortho_scale / 6.5

                    pos = [vec[0] + offset[0], vec[1] + offset[1] if len(vec) > 1 else offset[1], 0.05]
                    self.vector_renderer.draw_circle_2d(pos, radius=radius, color=v_color, alpha=v_alpha)

                # 2. Potom kresli aktuálne (vpredu, Z = 0.3)
                for v in current_vectors:
                    vec = v['vec']
                    offset = v.get('offset', [0, 0])
                    v_color = v.get('color', (1, 1, 1))
                    v_alpha = v.get('alpha', 1.0)
                    base_radius = 0.07
                    radius = base_radius * self.camera.ortho_scale / 6.5

                    pos = [vec[0] + offset[0], vec[1] + offset[1] if len(vec) > 1 else offset[1], 0.3]
                    self.vector_renderer.draw_circle_2d(pos, radius=radius, color=v_color, alpha=v_alpha)

            else:
                # Šípky - rovnaká logika

                # 1. Najprv VŠETKY ČIARY perzistentných (vzadu, nízke Z)
                sorted_persistent = sorted(persistent_vectors,
                                           key=lambda v: v['vec'][0] ** 2 + (
                                               v['vec'][1] if len(v['vec']) > 1 else 0) ** 2,
                                           reverse=False)

                for v in sorted_persistent:
                    vec = v['vec']
                    offset = v.get('offset', [0, 0])
                    v_color = v.get('color', (0.6, 0.2, 0.6))
                    v_alpha = v.get('alpha', 0.4)

                    x = vec[0]
                    y = vec[1] if len(vec) > 1 else 0
                    ox = offset[0]
                    oy = offset[1] if len(offset) > 1 else 0

                    vec_length = math.sqrt(x * x + y * y)
                    if vec_length > 0.1:
                        angle = math.atan2(y, x)

                        # Skráť čiaru o veľkosť šípky
                        line_end_x = ox + x - arrow_size * 0.35 * math.cos(angle)
                        line_end_y = oy + y - arrow_size * 0.35 * math.sin(angle)

                        glLineWidth(4)
                        glEnable(GL_BLEND)
                        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
                        glColor4f(v_color[0], v_color[1], v_color[2], v_alpha)

                        glBegin(GL_LINES)
                        glVertex3f(ox, oy, 0.01)  # Nízke Z pre čiary
                        glVertex3f(line_end_x, line_end_y, 0.01)
                        glEnd()

                        glDisable(GL_BLEND)

                # 2. Potom VŠETKY ŠÍPKY perzistentných (vyššie Z)
                for v in persistent_vectors:
                    vec = v['vec']
                    offset = v.get('offset', [0, 0])
                    v_color = v.get('color', (0.6, 0.2, 0.6))
                    # v_alpha = v.get('alpha', 0.4)  # STARÝ RIADOK

                    x = vec[0]
                    y = vec[1] if len(vec) > 1 else 0
                    ox = offset[0]
                    oy = offset[1] if len(offset) > 1 else 0

                    vec_length = math.sqrt(x * x + y * y)
                    if vec_length > 0.1:
                        angle = math.atan2(y, x)

                        # Šípka na pôvodnom konci vektora - PLNÁ ALPHA
                        self.vector_renderer.draw_triangle_arrowhead_2d(
                            ox + x, oy + y, angle, arrow_size, v_color, 1.0, z=0.05  # Zmenené z v_alpha na 1.0
                        )

                # 3. VŠETKY ČIARY aktuálnych (stredné Z)
                for v in current_vectors:
                    vec = v['vec']
                    offset = v.get('offset', [0, 0])
                    v_color = v.get('color', (1, 1, 1))
                    v_alpha = v.get('alpha', 1.0)

                    x = vec[0]
                    y = vec[1] if len(vec) > 1 else 0
                    ox = offset[0]
                    oy = offset[1] if len(offset) > 1 else 0

                    vec_length = math.sqrt(x * x + y * y)
                    if vec_length > 0.1:
                        angle = math.atan2(y, x)

                        # Skráť čiaru o veľkosť šípky
                        line_end_x = ox + x - arrow_size * 0.35 * math.cos(angle)
                        line_end_y = oy + y - arrow_size * 0.35 * math.sin(angle)

                        glLineWidth(6)

                        if v_alpha < 1.0:
                            glEnable(GL_BLEND)
                            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
                            glColor4f(v_color[0], v_color[1], v_color[2], v_alpha)
                        else:
                            glColor3f(*v_color)

                        glBegin(GL_LINES)
                        glVertex3f(ox, oy, 0.2)  # Stredné Z pre čiary
                        glVertex3f(line_end_x, line_end_y, 0.2)
                        glEnd()

                        if v_alpha < 1.0:
                            glDisable(GL_BLEND)

                # 4. VŠETKY ŠÍPKY aktuálnych (najvyššie Z)
                for v in current_vectors:
                    vec = v['vec']
                    offset = v.get('offset', [0, 0])
                    v_color = v.get('color', (1, 1, 1))
                    v_alpha = v.get('alpha', 1.0)

                    x = vec[0]
                    y = vec[1] if len(vec) > 1 else 0
                    ox = offset[0]
                    oy = offset[1] if len(offset) > 1 else 0

                    vec_length = math.sqrt(x * x + y * y)
                    if vec_length > 0.1:
                        angle = math.atan2(y, x)

                        # Šípka na pôvodnom konci
                        self.vector_renderer.draw_triangle_arrowhead_2d(
                            ox + x, oy + y, angle, arrow_size, v_color, v_alpha, z=0.3
                        )

                # Labely
                self.draw_vector_labels_2d(current_vectors)
                return  # Koniec pre span
        # ===== ANIMATION CONTROLLER =====
        if self.vector_manager.animation_controller.current_operation:
            vectors_to_draw = self.vector_manager.animation_controller.get_vectors_to_draw()

            # Zoraď vektory - najprv priesvitné, potom plne viditeľné
            vectors_sorted = sorted(vectors_to_draw, key=lambda v: v.get('alpha', 1.0))

            if self.vectors_as_points:
                for v in vectors_sorted:
                    vec = v['vec']
                    offset = v.get('offset', [0, 0])
                    v_color = v.get('color', color)
                    v_alpha = v.get('alpha', 1.0)
                    base_radius = 0.07
                    radius = base_radius * self.camera.ortho_scale / 6.5

                    if v_alpha < 1.0:
                        glDepthMask(GL_FALSE)

                    if isinstance(vec[0], (int, float)):
                        pos = [vec[0] + offset[0], vec[1] + offset[1] if len(vec) > 1 else offset[1], 0.2]
                        self.vector_renderer.draw_circle_2d(pos, radius=radius, color=v_color, alpha=v_alpha)
                    else:
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

                    if v_alpha < 1.0:
                        glDepthMask(GL_TRUE)

            else:
                # Vykresli vektory s offsetom
                for v in vectors_sorted:
                    vec = v['vec']
                    offset = v.get('offset', [0, 0])
                    v_color = v.get('color', color)
                    v_alpha = v.get('alpha', 1.0)

                    if v_alpha < 1.0:
                        glDepthMask(GL_FALSE)

                    if isinstance(vec[0], (int, float)):
                        # Jednoduchý vektor
                        x = vec[0]
                        y = vec[1] if len(vec) > 1 else 0
                        ox = offset[0]
                        oy = offset[1] if len(offset) > 1 else 0

                        vec_length = math.sqrt(x * x + y * y)
                        if vec_length > 0.1:
                            angle = math.atan2(y, x)

                            # Skráť čiaru o veľkosť šípky
                            line_end_x = ox + x - arrow_size * 0.35 * math.cos(angle)
                            line_end_y = oy + y - arrow_size * 0.35 * math.sin(angle)

                            glLineWidth(6)

                            if v_alpha < 1.0:
                                glEnable(GL_BLEND)
                                glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
                                glColor4f(v_color[0], v_color[1], v_color[2], v_alpha)
                            else:
                                glColor3f(*v_color)

                            glBegin(GL_LINES)
                            glVertex3f(ox, oy, 0.3)
                            glVertex3f(line_end_x, line_end_y, 0.3)
                            glEnd()

                            # Šípka na konci
                            self.vector_renderer.draw_triangle_arrowhead_2d(
                                ox + x, oy + y, angle, arrow_size, v_color, v_alpha, z=0.3
                            )

                            if v_alpha < 1.0:
                                glDisable(GL_BLEND)

                    else:
                        # Matica - každý riadok je vektor
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

                            vec_length = math.sqrt(x * x + y * y)
                            if vec_length > 0.1:
                                angle = math.atan2(y, x)

                                # Skráť čiaru o veľkosť šípky
                                line_end_x = ox + x - arrow_size * 0.35 * math.cos(angle)
                                line_end_y = oy + y - arrow_size * 0.35 * math.sin(angle)

                                glLineWidth(6)

                                if v_alpha < 1.0:
                                    glEnable(GL_BLEND)
                                    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
                                    glColor4f(v_color[0], v_color[1], v_color[2], v_alpha)
                                else:
                                    glColor3f(*v_color)

                                glBegin(GL_LINES)
                                glVertex3f(ox, oy, 0.3)
                                glVertex3f(line_end_x, line_end_y, 0.3)
                                glEnd()

                                # Šípka na konci
                                self.vector_renderer.draw_triangle_arrowhead_2d(
                                    ox + x, oy + y, angle, arrow_size, v_color, v_alpha, z=0.3
                                )

                                if v_alpha < 1.0:
                                    glDisable(GL_BLEND)

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

    def draw_vector_labels_2d(self, vectors):
        """Vykreslí labely pri vektoroch v 2D"""
        if not vectors:
            return

        width, height = pygame.display.get_window_size()

        # Setup 2D projekcie pre text
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()

        aspect = width / height
        if aspect >= 1.0:
            left = -self.camera.ortho_scale * aspect + self.camera.pan_offset_x
            right = self.camera.ortho_scale * aspect + self.camera.pan_offset_x
            bottom = -self.camera.ortho_scale + self.camera.pan_offset_y
            top = self.camera.ortho_scale + self.camera.pan_offset_y
        else:
            left = -self.camera.ortho_scale + self.camera.pan_offset_x
            right = self.camera.ortho_scale + self.camera.pan_offset_x
            bottom = -self.camera.ortho_scale / aspect + self.camera.pan_offset_y
            top = self.camera.ortho_scale / aspect + self.camera.pan_offset_y

        glOrtho(0, width, height, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        text_color = (1, 1, 1) if self.background_dark else (0, 0, 0)

        for v in vectors:
            # Zobraz len ak má flag show_label
            if not v.get('show_label', False):
                continue

            label = v.get('label', '')
            if not label:
                continue

            vec = v['vec']
            offset = v.get('offset', [0, 0])

            # ZMENA: Vypočítaj pozíciu STREDU vektora (nie konca)
            mid_x = offset[0] + vec[0] * 0.5  # Stred = offset + polovica vektora
            mid_y = offset[1] + (vec[1] * 0.5 if len(vec) > 1 else 0)

            # Konverzia world -> screen coordinates
            screen_x = (mid_x - left) / (right - left) * width
            screen_y = height - (mid_y - bottom) / (top - bottom) * height

            # Offset pre label (mierne vedľa stredu vektora)
            label_x = screen_x + 10
            label_y = screen_y - 10

            # Vykresli label
            self.ui_renderer.draw_text_2d(label, (label_x, label_y),
                                          color=text_color, font_size=18)

        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
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
        """Vykreslí UI overlay"""
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

        # Tlačidlo späť do menu
        back_button_rect = pygame.Rect(20, 10, 120, 40)
        self.ui_renderer.draw_button_2d(back_button_rect.x, back_button_rect.y,
                                        back_button_rect.width, back_button_rect.height,
                                        "← Menu", active=True)

        # Nápoveda
        text_color = (1, 1, 1) if self.background_dark else (0, 0, 0)
        self.ui_renderer.draw_text_2d("ESC = späť do menu", (150, 20),
                                      color=text_color, font_size=14)

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

        # INDIKÁTOR KROKOVANIA
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

            if is_matrix:
                descriptions = {
                    'add': [
                        "Zobrazenie matíc A (oranžová) a B (modrá)",
                        "Vektory matice B posunuté na konce vektorov matice A",
                        "Výsledná matica A + B (fialová)",
                        "Komutatívnosť: A + B = B + A (rovnobežník)"
                    ],
                    'subtract': [
                        "Zobrazenie matíc A (oranžová) a B (modrá)",
                        "Negácia matice B → -B (ružová)",
                        "Vektory matice -B posunuté na konce vektorov matice A",
                        "Výsledná matica A - B (fialová)"
                    ],
                    'scalar_mult': [
                        "Pôvodná matica A (oranžová)",
                        f"Výsledná matica {ctrl.constant}·A (fialová)" if ctrl.constant else "Výsledná matica (fialová)"
                    ]
                }
            else:
                descriptions = {
                    'add': [
                        "Zobrazenie vektorov v1 (oranžová) a v2 (modrá)",
                        "Vektor v2 posunutý na koniec v1",
                        "Výsledný vektor v1 + v2 (fialová)",
                        "Komutatívnosť: v1 + v2 = v2 + v1 (rovnobežník)"
                    ],
                    'subtract': [
                        "Zobrazenie vektorov v1 (oranžová) a v2 (modrá)",
                        "Negácia vektora v2 → -v2 (ružová)",
                        "Vektor -v2 posunutý na koniec v1",
                        "Výsledný vektor v1 - v2 (fialová)"
                    ],
                    'scalar_mult': [
                        "Pôvodný vektor v (oranžová)",
                        f"Výsledný vektor {ctrl.constant}·v (fialová)" if ctrl.constant else "Výsledný vektor (fialová)"
                    ],
                    'linear_combination': [
                        "Zobrazenie vektorov v1 a v2",
                        f"Vykreslenie c1·v1",
                        f"Vykreslenie c2·v2",
                        f"Pridanie c2·v2",
                        "Výsledný vektor c1·v1 + c2·v2 (fialová)"
                    ]
                }

            desc_list = descriptions.get(ctrl.operation_type, [""])
            desc = desc_list[ctrl.current_step] if ctrl.current_step < len(desc_list) else ""

            self.ui_renderer.draw_text_2d(step_text, (self.width // 2 - 150, 20),
                                          color=text_color, font_size=24)
            self.ui_renderer.draw_text_2d(desc, (self.width // 2 - 200, 50),
                                          color=text_color, font_size=18)
            self.ui_renderer.draw_text_2d("SPACE=ďalší | BACKSPACE=späť | C=zrušiť",
                                          (self.width // 2 - 180, 75),
                                          color=text_color, font_size=16)

        # Informácie o rovine a gridu
        if self.vector_manager.animation_controller.current_operation:
            plane = self.vector_manager.animation_controller.current_plane
            normal = self.vector_manager.animation_controller.operation_plane_normal

            if plane and not self.view_2d_mode:
                text_color = (1, 1, 1) if self.background_dark else (0, 0, 0)

                plane_text = f"Rovina: {plane}"
                self.ui_renderer.draw_text_2d(plane_text, (self.width // 2 - 180, 95),
                                              color=text_color, font_size=16)

                if normal:
                    normal_text = f"Normála: ({normal[0]:.2f}, {normal[1]:.2f}, {normal[2]:.2f})"
                    self.ui_renderer.draw_text_2d(normal_text, (self.width // 2 - 180, 115),
                                                  color=text_color, font_size=14)

                cam_text = f"Kamera: az={self.camera.azimuth:.1f}° el={self.camera.elevation:.1f}°"
                self.ui_renderer.draw_text_2d(cam_text, (self.width // 2 - 180, 135),
                                              color=text_color, font_size=14)

                # Plane grid mód
                grid_modes = ["vypnutý", "grid", "grid jemný", "grid hrubý", "vyplnená + grid", "vyplnená"]
                grid_text = f"Plane grid (B): {grid_modes[self.plane_grid_mode]}"
                self.ui_renderer.draw_text_2d(grid_text, (self.width // 2 - 180, 155),
                                              color=text_color, font_size=14)

                if not self.camera.animating_to_plane:
                    self.ui_renderer.draw_text_2d("V=zobraz rovinu",
                                                  (self.width // 2 - 180, 175),
                                                  color=text_color, font_size=14)

                # V metóde render_ui() - nájdi túto časť a nahraď ju:

                if self.vector_manager.span_controller.active:
                    text_color = (1, 1, 1) if self.background_dark else (0, 0, 0)

                    step_text = f"Span - Krok {self.vector_manager.span_controller.current_step + 1}"
                    self.ui_renderer.draw_text_2d(step_text, (self.width // 2 - 150, 20),
                                                  color=text_color, font_size=24)

                    comb = self.vector_manager.span_controller.combinations[
                        self.vector_manager.span_controller.current_step]
                    desc = f"c1·v1 + c2·v2 kde c1={comb['c1']:.2f}, c2={comb['c2']:.2f}"
                    self.ui_renderer.draw_text_2d(desc, (self.width // 2 - 200, 50),
                                                  color=text_color, font_size=18)

                    # UPRAVENÝ TEXT
                    if self.vector_manager.span_controller.auto_play:
                        control_text = "P=pauza | BACKSPACE=späť | C=zrušiť"
                        play_status = "▶ AUTO-PLAY AKTÍVNY"
                    else:
                        control_text = "P=play | SPACE=ďalší | BACKSPACE=späť | C=zrušiť"
                        play_status = "⏸ Manuálny režim"

                    self.ui_renderer.draw_text_2d(control_text, (self.width // 2 - 200, 75),
                                                  color=text_color, font_size=16)
                    self.ui_renderer.draw_text_2d(play_status, (self.width // 2 - 120, 95),
                                                  color=text_color, font_size=14)

                    self.ui_renderer.draw_text_2d("(Nekonečné krokovanie - nové kombinácie sa generujú priebežne)",
                                                  (self.width // 2 - 240, 115),
                                                  color=text_color, font_size=14)

        glEnable(GL_DEPTH_TEST)

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
        """Vykreslí matematické zobrazenie operácie - S ANIMÁCIOU"""
        math_renderer = MathRenderer()

        start_x = 20
        start_y = 80
        current_x = start_x

        op_type = math_info['operation']
        step = math_info['step']
        operands = math_info['operands']
        result = math_info['result']
        constant = math_info['constant']

        # Získaj animation progress
        ctrl = self.vector_manager.animation_controller
        anim_progress = ctrl.animation_progress if ctrl.animating else 1.0

        is_matrix_op = self._is_matrix(operands[0]) if operands else False
        text_color = (1, 1, 1) if self.background_dark else (0, 0, 0)

        # Pomocná funkcia pre flatten matice na vektory pre zobrazenie
        def flatten_for_display(data):
            if isinstance(data[0], (list, tuple)):
                # Matica - zobraz ako stĺpce
                return [row[0] if len(row) > 0 else 0 for row in data]
            return data

        # SČÍTANIE
        if op_type == 'add':
            op1_display = flatten_for_display(operands[0])
            op2_display = flatten_for_display(operands[1])
            result_display = flatten_for_display(result)

            if step == 0:
                # v1 + v2 (statické)
                current_x = math_renderer.draw_vector_math(current_x, start_y, op1_display,
                                                           color=(0.8, 0.8, 0.8), font_size=22)
                current_x = math_renderer.draw_operator(current_x, start_y, "+",
                                                        color=text_color, font_size=28)
                current_x = math_renderer.draw_vector_math(current_x, start_y, op2_display,
                                                           color=(0.8, 0.8, 0.8), font_size=22)

            elif step == 1:
                # v1 (highlight) + v2 (animovaný posun)
                current_x = math_renderer.draw_vector_math(current_x, start_y, op1_display,
                                                           color=(1, 0.5, 0), font_size=22, highlight=True)
                current_x = math_renderer.draw_operator(current_x, start_y, "+",
                                                        color=text_color, font_size=28)
                current_x = math_renderer.draw_vector_math(current_x, start_y, op2_display,
                                                           color=(0, 0.5, 1), font_size=22, highlight=True)

                # Druhý riadok: animovaný rozklad operácie
                current_x = start_x
                start_y += len(op1_display) * 27 + 20

                # Animuj zobrazenie operácie (fade in)
                if anim_progress < 1.0:
                    # Postupne zobrazuj
                    alpha = anim_progress
                    adj_color = (0.8 * alpha, 0.8 * alpha, 0.8 * alpha)
                else:
                    adj_color = (0.8, 0.8, 0.8)

                current_x = math_renderer.draw_vector_component_operation(
                    current_x, start_y, op1_display, op2_display, "+",
                    color=adj_color, font_size=20)

            elif step == 2:
                # v1 + v2 = result (animovaný výsledok)
                current_x = math_renderer.draw_vector_math(current_x, start_y, op1_display,
                                                           color=(1, 0.5, 0), font_size=22, highlight=True)
                current_x = math_renderer.draw_operator(current_x, start_y, "+",
                                                        color=text_color, font_size=28)
                current_x = math_renderer.draw_vector_math(current_x, start_y, op2_display,
                                                           color=(0, 0.5, 1), font_size=22, highlight=True)

                # Animované "="
                current_x = math_renderer.draw_equals_animated(current_x, start_y,
                                                               anim_progress, color=text_color, font_size=28)

                # Animovaný výsledok - čísla rastú od 0 do výsledku
                if anim_progress < 1.0:
                    # Interpoluj výsledok
                    animated_result = [r * anim_progress for r in result_display]
                    current_x = math_renderer.draw_vector_math(current_x, start_y, animated_result,
                                                               color=self.RESULT_COLOR, font_size=22, highlight=True)
                else:
                    current_x = math_renderer.draw_vector_math(current_x, start_y, result_display,
                                                               color=self.RESULT_COLOR, font_size=22, highlight=True)

                # Druhý riadok: animovaná transformácia operácie na výsledok
                current_x = start_x
                start_y += len(op1_display) * 27 + 20

                # Animuj prechod z "3+2" na "5"
                math_renderer.draw_component_operation_animated(
                    current_x, start_y, op1_display, op2_display, "+",
                    anim_progress, color=(0.8, 0.8, 0.8), font_size=20)

            elif step == 3:
                # KOMUTATÍVNOSŤ
                self.ui_renderer.draw_text_2d("Pôvodné:", (start_x, start_y - 25),
                                              color=text_color, font_size=16)
                current_x = math_renderer.draw_vector_math(current_x, start_y, op1_display,
                                                           color=(1, 0.5, 0), font_size=20, highlight=True)
                current_x = math_renderer.draw_operator(current_x, start_y, "+",
                                                        color=text_color, font_size=24)
                current_x = math_renderer.draw_vector_math(current_x, start_y, op2_display,
                                                           color=(0, 0.5, 1), font_size=20, highlight=True)
                current_x = math_renderer.draw_equals(current_x, start_y,
                                                      color=text_color, font_size=24)
                current_x = math_renderer.draw_vector_math(current_x, start_y, result_display,
                                                           color=self.RESULT_COLOR, font_size=20, highlight=True)

                # Druhý riadok: v2 + v1 (animované fade in)
                current_x = start_x
                start_y += len(op1_display) * 25 + 25

                # Fade in pre "Nové" text
                if anim_progress < 1.0:
                    alpha = anim_progress
                    new_color = (text_color[0] * alpha, text_color[1] * alpha, text_color[2] * alpha)
                else:
                    new_color = text_color

                self.ui_renderer.draw_text_2d("Nové (komutatívnosť):", (start_x, start_y - 25),
                                              color=new_color, font_size=16)

                # Animované zobrazenie opačného poradia
                v2_color = (0.3 * anim_progress + 0.1, 0.8 * anim_progress + 0.1, 1.0 * anim_progress)
                v1_color = (1.0 * anim_progress, 0.8 * anim_progress + 0.1, 0.4 * anim_progress + 0.1)

                current_x = math_renderer.draw_vector_math(current_x, start_y, op2_display,
                                                           color=v2_color, font_size=20, highlight=True)
                current_x = math_renderer.draw_operator(current_x, start_y, "+",
                                                        color=new_color, font_size=24)
                current_x = math_renderer.draw_vector_math(current_x, start_y, op1_display,
                                                           color=v1_color, font_size=20, highlight=True)
                current_x = math_renderer.draw_equals_animated(current_x, start_y,
                                                               anim_progress, color=text_color, font_size=24)

                # Výsledok fade in
                result_color = (self.RESULT_COLOR[0] * anim_progress + 0.1,
                                self.RESULT_COLOR[1] * anim_progress + 0.1,
                                self.RESULT_COLOR[2] * anim_progress + 0.1)
                current_x = math_renderer.draw_vector_math(current_x, start_y, result_display,
                                                           color=result_color, font_size=20, highlight=True)

                # Tretí riadok
                if anim_progress >= 0.8:
                    current_x = start_x
                    start_y += len(op1_display) * 25 + 20
                    final_alpha = (anim_progress - 0.8) / 0.2
                    final_color = (text_color[0] * final_alpha, text_color[1] * final_alpha,
                                   text_color[2] * final_alpha)
                    self.ui_renderer.draw_text_2d("→ Oba spôsoby vedú k rovnakému výsledku",
                                                  (current_x, start_y), color=final_color, font_size=16)

        # ODČÍTANIE
        elif op_type == 'subtract':
            op1_display = flatten_for_display(operands[0])
            op2_display = flatten_for_display(operands[1])
            neg_op2_display = [-x for x in op2_display]
            result_display = flatten_for_display(result)

            if step == 0:
                current_x = math_renderer.draw_vector_math(current_x, start_y, op1_display,
                                                           color=(0.8, 0.8, 0.8), font_size=22)
                current_x = math_renderer.draw_operator(current_x, start_y, "-",
                                                        color=text_color, font_size=28)
                current_x = math_renderer.draw_vector_math(current_x, start_y, op2_display,
                                                           color=(0.8, 0.8, 0.8), font_size=22)

            elif step == 1:
                # Animácia negácie: v2 → -v2
                current_x = math_renderer.draw_vector_math(current_x, start_y, op1_display,
                                                           color=(1, 0.5, 0), font_size=22, highlight=True)
                current_x = math_renderer.draw_operator(current_x, start_y, "+",
                                                        color=text_color, font_size=28)

                # Animovaná negácia
                if anim_progress < 1.0:
                    animated_neg = [op2_display[i] + (neg_op2_display[i] - op2_display[i]) * anim_progress
                                    for i in range(len(op2_display))]
                    current_x = math_renderer.draw_vector_math(current_x, start_y, animated_neg,
                                                               color=(1, 0, 0.5), font_size=22, highlight=True)
                else:
                    current_x = math_renderer.draw_vector_math(current_x, start_y, neg_op2_display,
                                                               color=(1, 0, 0.5), font_size=22, highlight=True)

            elif step == 2:
                current_x = math_renderer.draw_vector_math(current_x, start_y, op1_display,
                                                           color=(1, 0.5, 0), font_size=22, highlight=True)
                current_x = math_renderer.draw_operator(current_x, start_y, "+",
                                                        color=text_color, font_size=28)
                current_x = math_renderer.draw_vector_math(current_x, start_y, neg_op2_display,
                                                           color=(1, 0, 0.5), font_size=22, highlight=True)

                current_x = start_x
                start_y += len(op1_display) * 27 + 20

                # Animovaný rozklad
                alpha = anim_progress
                adj_color = (0.8 * alpha, 0.8 * alpha, 0.8 * alpha)
                current_x = math_renderer.draw_vector_component_operation(
                    current_x, start_y, op1_display, neg_op2_display, "+",
                    color=adj_color, font_size=20)

            elif step == 3:
                current_x = math_renderer.draw_vector_math(current_x, start_y, op1_display,
                                                           color=(1, 0.5, 0), font_size=22, highlight=True)
                current_x = math_renderer.draw_operator(current_x, start_y, "+",
                                                        color=text_color, font_size=28)
                current_x = math_renderer.draw_vector_math(current_x, start_y, neg_op2_display,
                                                           color=(1, 0, 0.5), font_size=22, highlight=True)
                current_x = math_renderer.draw_equals_animated(current_x, start_y,
                                                               anim_progress, color=text_color, font_size=28)

                # Animovaný výsledok
                if anim_progress < 1.0:
                    animated_result = [r * anim_progress for r in result_display]
                    current_x = math_renderer.draw_vector_math(current_x, start_y, animated_result,
                                                               color=self.RESULT_COLOR, font_size=22, highlight=True)
                else:
                    current_x = math_renderer.draw_vector_math(current_x, start_y, result_display,
                                                               color=self.RESULT_COLOR, font_size=22, highlight=True)

        # NÁSOBENIE KONŠTANTOU
        elif op_type == 'scalar_mult':
            op1_display = flatten_for_display(operands[0])
            result_display = flatten_for_display(result)

            if step == 0:
                current_x = math_renderer.draw_scalar(current_x, start_y, constant,
                                                      color=text_color, font_size=26)
                current_x = math_renderer.draw_operator(current_x - 10, start_y, "·",
                                                        color=text_color, font_size=28)
                current_x = math_renderer.draw_vector_math(current_x, start_y, op1_display,
                                                           color=(1, 0.5, 0), font_size=22, highlight=True)

            elif step == 1:
                current_x = math_renderer.draw_scalar(current_x, start_y, constant,
                                                      color=text_color, font_size=26)
                current_x = math_renderer.draw_operator(current_x - 10, start_y, "·",
                                                        color=text_color, font_size=28)
                current_x = math_renderer.draw_vector_math(current_x, start_y, op1_display,
                                                           color=(1, 0.5, 0), font_size=22, highlight=True)
                current_x = math_renderer.draw_equals_animated(current_x, start_y,
                                                               anim_progress, color=text_color, font_size=28)

                # Animovaný výsledok - interpolácia od pôvodného k výsledku
                if anim_progress < 1.0:
                    animated_result = [op1_display[i] + (result_display[i] - op1_display[i]) * anim_progress
                                       for i in range(len(result_display))]
                    current_x = math_renderer.draw_vector_math(current_x, start_y, animated_result,
                                                               color=self.RESULT_COLOR, font_size=22, highlight=True)
                else:
                    current_x = math_renderer.draw_vector_math(current_x, start_y, result_display,
                                                               color=self.RESULT_COLOR, font_size=22, highlight=True)

        # LINEÁRNA KOMBINÁCIA
        elif op_type == 'linear_combination':
            c1, c2 = constant if constant else [1, 1]
            op1_display = flatten_for_display(operands[0])
            op2_display = flatten_for_display(operands[1])
            result_display = flatten_for_display(result)

            scaled_v1 = [c1 * x for x in op1_display]
            scaled_v2 = [c2 * x for x in op2_display]

            if step == 0:
                current_x = math_renderer.draw_scalar(current_x, start_y, c1,
                                                      color=text_color, font_size=24)
                current_x = math_renderer.draw_operator(current_x - 10, start_y, "·",
                                                        color=text_color, font_size=26)
                current_x = math_renderer.draw_vector_math(current_x, start_y, op1_display,
                                                           color=(0.8, 0.8, 0.8), font_size=20)
                current_x = math_renderer.draw_operator(current_x, start_y, "+",
                                                        color=text_color, font_size=26)
                current_x = math_renderer.draw_scalar(current_x, start_y, c2,
                                                      color=text_color, font_size=24)
                current_x = math_renderer.draw_operator(current_x - 10, start_y, "·",
                                                        color=text_color, font_size=26)
                current_x = math_renderer.draw_vector_math(current_x, start_y, op2_display,
                                                           color=(0.8, 0.8, 0.8), font_size=20)

            elif step == 1:
                # Animácia škálovania v1
                if anim_progress < 1.0:
                    animated_scaled = [op1_display[i] + (scaled_v1[i] - op1_display[i]) * anim_progress
                                       for i in range(len(op1_display))]
                else:
                    animated_scaled = scaled_v1

                current_x = math_renderer.draw_vector_math(current_x, start_y, animated_scaled,
                                                           color=(1, 0.5, 0), font_size=20, highlight=True)
                current_x = math_renderer.draw_operator(current_x, start_y, "+",
                                                        color=text_color, font_size=26)
                current_x = math_renderer.draw_scalar(current_x, start_y, c2,
                                                      color=text_color, font_size=24)
                current_x = math_renderer.draw_operator(current_x - 10, start_y, "·",
                                                        color=text_color, font_size=26)
                current_x = math_renderer.draw_vector_math(current_x, start_y, op2_display,
                                                           color=(0.8, 0.8, 0.8), font_size=20)

            elif step == 2:
                # Animácia škálovania v2
                current_x = math_renderer.draw_vector_math(current_x, start_y, scaled_v1,
                                                           color=(1, 0.5, 0), font_size=20, highlight=True)
                current_x = math_renderer.draw_operator(current_x, start_y, "+",
                                                        color=text_color, font_size=26)

                if anim_progress < 1.0:
                    animated_scaled = [op2_display[i] + (scaled_v2[i] - op2_display[i]) * anim_progress
                                       for i in range(len(op2_display))]
                else:
                    animated_scaled = scaled_v2

                current_x = math_renderer.draw_vector_math(current_x, start_y, animated_scaled,
                                                           color=(0, 0.5, 1), font_size=20, highlight=True)

            elif step == 3:
                current_x = math_renderer.draw_vector_math(current_x, start_y, scaled_v1,
                                                           color=(1, 0.5, 0), font_size=20, highlight=True)
                current_x = math_renderer.draw_operator(current_x, start_y, "+",
                                                        color=text_color, font_size=26)
                current_x = math_renderer.draw_vector_math(current_x, start_y, scaled_v2,
                                                           color=(0, 0.5, 1), font_size=20, highlight=True)

                current_x = start_x
                start_y += len(scaled_v1) * 27 + 20

                alpha = anim_progress
                adj_color = (0.8 * alpha, 0.8 * alpha, 0.8 * alpha)
                current_x = math_renderer.draw_vector_component_operation(
                    current_x, start_y, scaled_v1, scaled_v2, "+",
                    color=adj_color, font_size=20)

            elif step == 4:
                current_x = math_renderer.draw_vector_math(current_x, start_y, scaled_v1,
                                                           color=(1, 0.5, 0), font_size=20, highlight=True)
                current_x = math_renderer.draw_operator(current_x, start_y, "+",
                                                        color=text_color, font_size=26)
                current_x = math_renderer.draw_vector_math(current_x, start_y, scaled_v2,
                                                           color=(0, 0.5, 1), font_size=20, highlight=True)
                current_x = math_renderer.draw_equals_animated(current_x, start_y,
                                                               anim_progress, color=text_color, font_size=26)

                if anim_progress < 1.0:
                    animated_result = [r * anim_progress for r in result_display]
                    current_x = math_renderer.draw_vector_math(current_x, start_y, animated_result,
                                                               color=(0, 1, 0), font_size=20, highlight=True)
                else:
                    current_x = math_renderer.draw_vector_math(current_x, start_y, result_display,
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
        """Hlavná slučka - OPTIMALIZOVANÉ"""
        self.run_startup_screen()

        if not self.running:
            TextureCache.clear()  # Vyčisti textúry
            pygame.quit()
            return

        self.run_baza_input()

        if not self.running:
            TextureCache.clear()
            pygame.quit()
            return

        while self.running:
            dt = self.clock.tick(60) / 1000.0  # Cap na 60 FPS

            self.handle_events()
            self.update(dt)
            self.render()

        TextureCache.clear()  # Vyčisti textúry pred ukončením
        pygame.quit()


def main():
    """Hlavná funkcia"""
    app = Application()
    app.run()


if __name__ == "__main__":
    main()