from Configurations.Colors import Colors
from Rendering.MathRenderer import MathRenderer
from VectorStuff.VectorManager import VectorManager
from VectorStuff.VectorRenderer import VectorRenderer
from Handler.InputHandler import InputHandler
from Cache.TextureCache import TextureCache
from Rendering.UIRenderer import UIRenderer
from Rendering.EnhancedUIRenderer import EnhancedUIRenderer
from Configurations.Config import Config
from Camera.Camera import Camera
from Rendering.GridRenderer import GridRenderer
from Rendering.AxesRenderer import AxesRenderer
from Controllers.TransformationController import TransformationController
from Controllers.TransformationController import TransformationController



import math
import pygame
import numpy as np
from pygame.locals import *
from OpenGL.GL import *
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
        self.transformation_controller = TransformationController()

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
        self.show_helper_lines = False

        # Vectors
        self.saved_baza = []
        self.pending_vector = None

        # UI State
        self.matrix_inputs = [["" for _ in range(3)] for _ in range(3)]
        self.matrix_active_cell = (-1, -1)
        self.show_matrix_input = True

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

    def _move_camera_to_operation_plane(self):
        """Presunie kameru na rovinu aktuálnej operácie (len 3D)"""
        if self.view_2d_mode:
            return

        ctrl = self.vector_manager.animation_controller
        plane = ctrl.current_plane
        if not plane:
            return

        max_val = self.get_max_from_vectors()
        distance = max(25.0, max_val * 3.0 + 5)

        if plane == "3D":
            optimal_view = (ctrl.compute_optimal_view_for_current_step()
                            or ctrl.compute_optimal_view_for_3d())
            if optimal_view:
                self.camera.move_to_plane(plane, distance, custom_view=optimal_view)
            else:
                self.camera.move_to_plane(plane, distance)
        else:
            self.camera.move_to_plane(plane, distance)

    def _draw_single_vector_3d(self, vec, offset, v_color, v_alpha, cam_pos):
        """Vykreslí jeden 3D vektor so šípkou"""
        import numpy as np

        glPushMatrix()
        glTranslatef(offset[0], offset[1], offset[2])

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

        self.vector_renderer.draw_conehead_3d(
            (0, 0, 0), end_point,
            color=v_color, size=0.7,
            camera_pos=cam_pos, radius=0.25,
            alpha=v_alpha
        )
        glPopMatrix()

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
            transform_rect = pygame.Rect(basis_x, basis_y + 110, 190, 44)

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
            matrix_y = basis_y + 220
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

                    # === TRANSFORMÁCIA BÁZY TLAČIDLO ===
                    if transform_rect.collidepoint(mx, my) and self.view_2d_mode:
                        panel_width = 2 * (Config.MATRIX_CELL_W + Config.MATRIX_GAP)
                        panel_height = 2 * (Config.MATRIX_CELL_H + Config.MATRIX_GAP)

                        start_x = self.width // 2 - panel_width // 2
                        start_y = self.height // 2 - panel_height // 2

                        pending_input_panel = {
                            'type': 'transformation',
                            'operation': 'Transformácia bázy',
                            'values': [["" for _ in range(2)] for _ in range(2)],
                            'active_cell': (0, 0),
                            'x': start_x,
                            'y': start_y,
                            'panels': [{
                                'type': 'matrix',
                                'rows': 2, 'cols': 2,
                                'values': [["" for _ in range(2)] for _ in range(2)],
                                'active_cell': (0, 0),
                                'x': start_x,
                                'y': start_y
                            }],
                            'active_panel': 0,
                            'num_panels': 1,
                            'symbol': None,
                            'has_constant': False,
                            'is_transformation': True
                        }
                        span_input_panel = None
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
                                    # === TRANSFORMÁCIA BÁZY ===
                                    if pending_input_panel.get('is_transformation', False):
                                        try:
                                            panel = pending_input_panel["panels"][0]
                                            mat = [[float(v) for v in row] for row in panel["values"]]
                                            # Nastav transformáciu
                                            self.transformation_controller.setup_transformation(mat, is_2d=True)
                                            # Použi štandardnú bázu
                                            self.saved_baza.append(((1, 0), (0, 1)))
                                            pending_input_panel = None
                                            self.is_not_baza = False
                                            print(f"Transformácia bázy nastavená: {mat}")
                                        except Exception as e:
                                            print(f"Chyba pri parsovaní transformačnej matice: {e}")
                                        continue

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

            # === TRANSFORMÁCIA BÁZY TLAČIDLO ===
            if self.view_2d_mode:
                hover_transform = transform_rect.collidepoint(mx, my)
                self.ui_renderer.draw_button_2d(transform_rect.x, transform_rect.y,
                                                transform_rect.w, transform_rect.h,
                                                "Transformácia bázy",
                                                is_dark=self.background_dark,
                                                hover=hover_transform, primary=True)
                self.ui_renderer.draw_text_2d("Animovaná zmena bázy (3b1b)",
                                              (basis_x + 5, transform_rect.y + 48),
                                              sub_col, 13)

            # === VLASTNÁ BÁZA ===
            self.ui_renderer.draw_card(basis_x, basis_y + 175, 190, 180, self.background_dark, "Vlastná báza")
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
                if pending_input_panel.get("is_transformation"):
                    # Transformácia bázy panel
                    panel = pending_input_panel["panels"][0]
                    color_symbol = (1, 1, 1) if self.background_dark else (0, 0, 0)

                    # Card pozadie
                    panel_width = 2 * (Config.MATRIX_CELL_W + Config.MATRIX_GAP) + 40
                    panel_height = 2 * (Config.MATRIX_CELL_H + Config.MATRIX_GAP) + 80
                    bg_x = panel["x"] - 20
                    bg_y = panel["y"] - 40

                    self.ui_renderer.draw_card(bg_x, bg_y, panel_width, panel_height,
                                               self.background_dark, "Transformácia bázy")

                    # Stĺpcové labely
                    self.ui_renderer.draw_text_2d("i",
                                                  (panel["x"] + 10, panel["y"] - 25),
                                                  (0.2, 0.85, 0.2), 18)
                    self.ui_renderer.draw_text_2d("j",
                                                  (panel["x"] + Config.MATRIX_CELL_W + Config.MATRIX_GAP + 10,
                                                   panel["y"] - 25),
                                                  (0.85, 0.2, 0.2), 18)

                    for r in range(panel["rows"]):
                        for c in range(panel["cols"]):
                            x = panel["x"] + c * (Config.MATRIX_CELL_W + Config.MATRIX_GAP)
                            y = panel["y"] + r * (Config.MATRIX_CELL_H + Config.MATRIX_GAP)
                            active = (r, c) == panel["active_cell"]
                            col_color = [(0.2, 0.85, 0.2), (0.85, 0.2, 0.2)][c]
                            self.ui_renderer.draw_input_box_3d(x, y, Config.MATRIX_CELL_W,
                                                               Config.MATRIX_CELL_H,
                                                               panel["values"][r][c], active,
                                                               fill_color_outline=col_color,
                                                               is_dark=self.background_dark)

                    help_y = panel["y"] + 2 * (Config.MATRIX_CELL_H + Config.MATRIX_GAP) + 10
                    self.ui_renderer.draw_text_2d("Stĺpce = kam pôjde i a j",
                                                  (panel["x"] - 30, help_y),
                                                  sub_col, 14)
                    self.ui_renderer.draw_text_2d("Enter = potvrdiť | ESC = zrušiť",
                                                  (panel["x"] - 50, help_y + 20),
                                                  sub_col, 14)

                elif pending_input_panel.get("is_linear_combination"):
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
        self.transformation_controller.clear()

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

        # ===== TRANSFORMÁCIA BÁZY KROKOVANIE =====
        if self.transformation_controller.active:
            if event.key == pygame.K_SPACE:
                self.transformation_controller.next_step()
                return
            elif event.key == pygame.K_BACKSPACE:
                if not any([
                    self.input_handler.show_input_active,
                    self.input_handler.show_multiplication_active,
                    self.input_handler.show_matrix_size_active,
                    self.input_handler.show_random_range_active,
                    self.matrix_active_cell != (-1, -1)
                ]):
                    self.transformation_controller.prev_step()
                    return
            elif event.key == pygame.K_c:
                self.transformation_controller.clear()
                return

        # ===== KROKOVANIE ANIMÁCIE - PÔVODNÝ KÓD =====
        if event.key == pygame.K_SPACE:
            if self.vector_manager.animation_controller.current_operation:
                self.vector_manager.animation_controller.next_step()
                self._move_camera_to_operation_plane()
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
                    self._move_camera_to_operation_plane()
                    return

        if event.key == pygame.K_c:
            self.vector_manager.animation_controller.clear()
            return

        if event.key == pygame.K_v:
            if self.vector_manager.animation_controller.current_operation:
                self._move_camera_to_operation_plane()
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
            self.show_helper_lines = not self.show_helper_lines
        elif event.key == pygame.K_i:
            self.input_handler.show_input_active = True


    def handle_mouse_down(self, event, mx, my):
        """Spracuje stlačenie myši"""

        # === Tlačidlo späť do menu ===
        back_button_rect = pygame.Rect(20, 10, 120, 40)
        if back_button_rect.collidepoint(mx, my):
            self.go_back_to_menu()
            self.run_startup_screen()
            if self.running:
                self.run_baza_input()
            return

        # === Toggle background ===
        toggle_bg_rect = pygame.Rect(20, self.height - 50, 120, 35)
        if toggle_bg_rect.collidepoint(mx, my):
            self.background_dark = not self.background_dark
            if self.background_dark:
                glClearColor(0.0, 0.0, 0.0, 1.0)
            else:
                glClearColor(1.0, 1.0, 1.0, 1.0)
            return

        # === Draw Vector / As Points toggle ===
        draw_button_rect = pygame.Rect(self.width - 160, 120, 140, 35)
        if draw_button_rect.collidepoint(mx, my):
            self.vectors_as_points = not self.vectors_as_points
            if not self.vectors_as_points:
                self.vector_manager.reset_animations()
            return

        # === Camera controls ===
        if event.button == 1:
            if not self.vector_manager.span_controller.show_all_mode:
                self.camera.rotating = not self.view_2d_mode
                self.camera.panning = self.view_2d_mode
            self.camera.last_mouse = event.pos
        elif event.button == 3:
            if not self.vector_manager.span_controller.show_all_mode:
                self.camera.panning = True
            self.camera.last_mouse = event.pos
        elif event.button == 4:  # scroll up
            span_ctrl = self.vector_manager.span_controller
            if span_ctrl.show_all_mode and span_ctrl.locked_ortho_scale:
                new_scale = self.camera.ortho_scale / 1.1
                min_scale = span_ctrl.locked_ortho_scale * 0.5
                if new_scale >= min_scale:
                    self.camera.ortho_scale = new_scale
            else:
                self.camera.zoom_in(self.view_2d_mode)
        elif event.button == 5:  # scroll down
            span_ctrl = self.vector_manager.span_controller
            if span_ctrl.show_all_mode and span_ctrl.locked_ortho_scale:
                new_scale = self.camera.ortho_scale * 1.1
                max_scale = span_ctrl.locked_ortho_scale * 1.5
                if new_scale <= max_scale:
                    self.camera.ortho_scale = new_scale
            else:
                self.camera.zoom_out(self.view_2d_mode)

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

        # Update transformácie
        if self.transformation_controller.active:
            self.transformation_controller.update(dt)

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

        # Draw grid (pôvodná šedá mriežka - vždy ako pozadie)
        if self.grid_mode % 2 == 0:
            self.grid_renderer.draw_grid_2d(self.camera.ortho_scale,
                                            self.camera.pan_offset_x,
                                            self.camera.pan_offset_y,
                                            self.width, self.height)

        # === TRANSFORMÁCIA BÁZY - kresli transformovanú mriežku ===
        if self.transformation_controller.active:
            self.draw_transformation_grid_2d()
            self.draw_transformation_basis_vectors_2d()

        # Draw axes — stlmiť ak prebieha transformácia
        if self.show_axes:
            tc = self.transformation_controller
            if tc.active and tc.t > 0.01:
                # Stlmené pôvodné osi — čím viac transformácia, tým menej viditeľné
                dim_alpha = max(0.15, 1.0 - tc.t * 0.8)
                # self._draw_axes_2d_dimmed(dim_alpha)
            else:
                self.axes_renderer.draw_axes_2d(self.camera.ortho_scale,
                                                self.camera.pan_offset_x,
                                                self.camera.pan_offset_y,
                                                self.width, self.height,
                                                self.ui_renderer,self.background_dark)

        # NOVÉ: Vykresli body kruhu ak je span aktívny
        # debug purposes
        if self.vector_manager.span_controller.active:
            self.draw_span_circle_points()

        # Draw vectors
        color = (0, 0, 0) if not self.background_dark else (1, 1, 1)
        self.draw_vectors_2d(color)

        # Draw helper lines (component decomposition) if enabled
        if self.show_helper_lines:
            self.draw_helper_lines_2d()

    def _draw_axes_2d_dimmed(self, alpha):
        """Nakreslí pôvodné 2D osi so stlmenou priehľadnosťou"""
        width, height = pygame.display.get_window_size()
        aspect = width / max(height, 1)
        ortho = self.camera.ortho_scale
        pan_x = self.camera.pan_offset_x
        pan_y = self.camera.pan_offset_y

        if aspect >= 1.0:
            left = -ortho * aspect + pan_x
            right = ortho * aspect + pan_x
            bottom = -ortho + pan_y
            top = ortho + pan_y
        else:
            left = -ortho + pan_x
            right = ortho + pan_x
            bottom = -ortho / aspect + pan_y
            top = ortho / aspect + pan_y

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glLineWidth(1.5)
        glBegin(GL_LINES)
        glColor4f(1, 0, 0, alpha)  # X axis — stlmená červená
        glVertex2f(left, 0)
        glVertex2f(right, 0)
        glColor4f(0, 1, 0, alpha)  # Y axis — stlmená zelená
        glVertex2f(0, bottom)
        glVertex2f(0, top)
        glEnd()
        glDisable(GL_BLEND)

        # Stlmené labely
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, width, height, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        dim_red = (1.0 * alpha, 0, 0)
        dim_green = (0, 1.0 * alpha, 0)

        def world_to_screen(wx, wy, color=(1, 1, 1), label=""):
            sx = (wx - left) / (right - left) * width
            sy = height - (wy - bottom) / (top - bottom) * height
            if label in ["X1+", "X1-"]:
                sx += -30 if label == "X1+" else 10
                sy += 5
            if label in ["X2+", "X2-"]:
                sy += -30 if label == "X2-" else 10
                sx += 5
            self.ui_renderer.draw_text_2d(label, (sx, sy), color=color, font_size=20)

        world_to_screen(right, 0, color=dim_red, label="X1+")
        world_to_screen(left, 0, color=dim_red, label="X1-")
        world_to_screen(0, top, color=dim_green, label="X2+")
        world_to_screen(0, bottom, color=dim_green, label="X2-")

        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)

    def draw_transformation_grid_2d(self):
        """Vykreslí transformovanú mriežku — rovnaký princíp ako klasická draw_grid_2d"""
        tc = self.transformation_controller
        if not tc.active or tc.t < 0.01:
            return

        M = tc.get_interpolated_matrix()
        m00, m10 = float(M[0, 0]), float(M[1, 0])
        m01, m11 = float(M[0, 1]), float(M[1, 1])

        # === 1) Viditeľná oblasť (rovnako ako draw_grid_2d) ===
        width, height = pygame.display.get_window_size()
        aspect = width / max(height, 1)
        ortho = self.camera.ortho_scale
        pan_x = self.camera.pan_offset_x
        pan_y = self.camera.pan_offset_y

        if aspect >= 1.0:
            left = -ortho * aspect + pan_x
            right = ortho * aspect + pan_x
            bottom = -ortho + pan_y
            top = ortho + pan_y
        else:
            left = -ortho + pan_x
            right = ortho + pan_x
            bottom = -ortho / aspect + pan_y
            top = ortho / aspect + pan_y

        # === 2) Inverzná matica — mapuj rohy obrazovky do pôvodných súradníc ===
        det = m00 * m11 - m01 * m10
        if abs(det) < 1e-10:
            return  # Singulárna matica, nedá sa invertovať

        inv_det = 1.0 / det
        # Inverzia 2x2: [[m11, -m01], [-m10, m00]] / det
        i00 = m11 * inv_det
        i01 = -m01 * inv_det
        i10 = -m10 * inv_det
        i11 = m00 * inv_det

        # Transformuj 4 rohy obrazovky do pôvodných súradníc
        corners_screen = [(left, bottom), (right, bottom), (right, top), (left, top)]
        orig_xs = []
        orig_ys = []
        for sx, sy in corners_screen:
            ox = i00 * sx + i01 * sy
            oy = i10 * sx + i11 * sy
            orig_xs.append(ox)
            orig_ys.append(oy)

        # Rozsah v pôvodných súradniciach (s rezervou)
        orig_left = min(orig_xs)
        orig_right = max(orig_xs)
        orig_bottom = min(orig_ys)
        orig_top = max(orig_ys)

        orig_visible = max(orig_right - orig_left, orig_top - orig_bottom)

        # === 3) Dynamický step (rovnako ako draw_grid_2d) ===
        step = tc.grid_step
        display_step = step
        max_lines = 100
        while orig_visible / display_step > max_lines:
            display_step *= 2

        # Zarovnaj rozsahy na display_step
        start_u = math.floor(orig_left / display_step) * display_step
        end_u = math.ceil(orig_right / display_step) * display_step
        start_v = math.floor(orig_bottom / display_step) * display_step
        end_v = math.ceil(orig_top / display_step) * display_step

        # === 4) Kresli priamo (rovnako ako draw_grid_2d) ===
        r_col = 0.4 * tc.t
        g_col = 0.6 * tc.t
        b_col = 0.2 * tc.t
        alpha = 0.4 + 0.4 * tc.t

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glLineWidth(1.5)
        glColor4f(r_col, g_col, b_col, alpha)

        glBegin(GL_LINES)

        # Vertikálne čiary (konštantné u, v sa mení)
        u = start_u
        while u <= end_u:
            x1 = m00 * u + m01 * start_v
            y1 = m10 * u + m11 * start_v
            x2 = m00 * u + m01 * end_v
            y2 = m10 * u + m11 * end_v
            glVertex3f(x1, y1, 0.05)
            glVertex3f(x2, y2, 0.05)
            u += display_step

        # Horizontálne čiary (konštantné v, u sa mení)
        v = start_v
        while v <= end_v:
            x1 = m00 * start_u + m01 * v
            y1 = m10 * start_u + m11 * v
            x2 = m00 * end_u + m01 * v
            y2 = m10 * end_u + m11 * v
            glVertex3f(x1, y1, 0.05)
            glVertex3f(x2, y2, 0.05)
            v += display_step

        glEnd()
        glDisable(GL_BLEND)

        # === 5) Transformované osi — zvýraznené čiary pre u=0 a v=0 ===
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        axis_alpha = 0.6 + 0.4 * tc.t

        # Extent pre osi — ťahajú sa cez celú viditeľnú oblasť
        axis_extent_u = max(abs(start_u), abs(end_u))
        axis_extent_v = max(abs(start_v), abs(end_v))

        glLineWidth(3.0)

        # Nová X os (u-os): v=0, u sa mení — zelená (smer î)
        #glColor4f(0.2, 0.85, 0.2, axis_alpha)
        if self.background_dark:
            glColor3f(1, 1, 1)
        else:
            glColor3f(0, 0, 0)
        glBegin(GL_LINES)
        x1 = m00 * (-axis_extent_u)
        y1 = m10 * (-axis_extent_u)
        x2 = m00 * axis_extent_u
        y2 = m10 * axis_extent_u
        glVertex3f(x1, y1, 0.06)
        glVertex3f(x2, y2, 0.06)
        glEnd()

        # Nová Y os (v-os): u=0, v sa mení — červená (smer ĵ)
        #glColor4f(0.85, 0.2, 0.2, axis_alpha)
        if self.background_dark:
            glColor3f(1, 1, 1)
        else:
            glColor3f(0, 0, 0)
        glBegin(GL_LINES)
        x1 = m01 * (-axis_extent_v)
        y1 = m11 * (-axis_extent_v)
        x2 = m01 * axis_extent_v
        y2 = m11 * axis_extent_v
        glVertex3f(x1, y1, 0.06)
        glVertex3f(x2, y2, 0.06)
        glEnd()

        glDisable(GL_BLEND)

    def draw_transformation_basis_vectors_2d(self):
        """Vykreslí animované bázové vektory pri transformácii"""
        tc = self.transformation_controller
        if not tc.active:
            return

        i_hat, j_hat = tc.get_basis_vectors_2d()

        arrow_size = max(0.15, min(0.12 * self.camera.ortho_scale, 2.0))

        # î vektor (zelený)
        ix, iy = i_hat[0], i_hat[1]
        vec_length = math.sqrt(ix * ix + iy * iy)
        if vec_length > 0.05:
            angle = math.atan2(iy, ix)
            line_end_x = ix - arrow_size * 0.35 * math.cos(angle)
            line_end_y = iy - arrow_size * 0.35 * math.sin(angle)

            glLineWidth(5)
            glColor3f(*tc.color_i)
            glBegin(GL_LINES)
            glVertex3f(0, 0, 0.4)
            glVertex3f(line_end_x, line_end_y, 0.4)
            glEnd()

            self.vector_renderer.draw_triangle_arrowhead_2d(
                ix, iy, angle, arrow_size, tc.color_i, 1.0, z=0.4
            )

        # ĵ vektor (červený)
        jx, jy = j_hat[0], j_hat[1]
        vec_length = math.sqrt(jx * jx + jy * jy)
        if vec_length > 0.05:
            angle = math.atan2(jy, jx)
            line_end_x = jx - arrow_size * 0.35 * math.cos(angle)
            line_end_y = jy - arrow_size * 0.35 * math.sin(angle)

            glLineWidth(5)
            glColor3f(*tc.color_j)
            glBegin(GL_LINES)
            glVertex3f(0, 0, 0.4)
            glVertex3f(line_end_x, line_end_y, 0.4)
            glEnd()

            self.vector_renderer.draw_triangle_arrowhead_2d(
                jx, jy, angle, arrow_size, tc.color_j, 1.0, z=0.4
            )

        # Labely pre bázové vektory (v screen coordinates)
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

        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, width, height, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        def world_to_screen(wx, wy):
            sx = (wx - left) / (right - left) * width
            sy = height - (wy - bottom) / (top - bottom) * height
            return sx, sy

        # î label
        sx, sy = world_to_screen(ix, iy)
        i_label = f"i ({ix:.1f}, {iy:.1f})"
        self.ui_renderer.draw_text_2d(i_label, (sx + 10, sy - 20),
                                       color=tc.color_i, font_size=16)

        # ĵ label
        sx, sy = world_to_screen(jx, jy)
        j_label = f"j ({jx:.1f}, {jy:.1f})"
        self.ui_renderer.draw_text_2d(j_label, (sx + 10, sy - 20),
                                       color=tc.color_j, font_size=16)

        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)

    def draw_helper_lines_2d(self):
        """Vykreslí pomocné čiary (rozklad na komponenty) pre všetky viditeľné vektory v 2D"""
        # Zozbieraj vektory na vykreslenie
        vectors_to_draw = []

        if self.vector_manager.span_controller.active:
            all_vecs = self.vector_manager.span_controller.get_current_vectors()
            vectors_to_draw = [v for v in all_vecs if not v.get('is_persistent', False)]
        elif self.vector_manager.animation_controller.current_operation:
            vectors_to_draw = self.vector_manager.animation_controller.get_vectors_to_draw()
        else:
            vectors_to_draw = self.vector_manager.animated_vectors

        if not vectors_to_draw:
            return

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

        # Kreslenie pomocných čiar v world coordinates (sme stále v 2D projekcii)
        for v in vectors_to_draw:
            vec = v.get('vec', None)
            if vec is None:
                continue

            offset = v.get('offset', [0, 0])
            v_alpha = v.get('alpha', 1.0)

            # Preskočíme veľmi priesvitné vektory
            if v_alpha < 0.3:
                continue

            # Spracuj jednoduché vektory (nie matice)
            if isinstance(vec[0], (int, float)):
                ox = offset[0]
                oy = offset[1] if len(offset) > 1 else 0
                vx = vec[0]
                vy = vec[1] if len(vec) > 1 else 0

                # Preskočíme nulové vektory
                if abs(vx) < 0.01 and abs(vy) < 0.01:
                    continue

                end_x = ox + vx
                end_y = oy + vy

                line_alpha = min(v_alpha, 0.7)

                glEnable(GL_BLEND)
                glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
                glLineWidth(8.0)

                # Štýl čiar - prerušované čiary simulované krátkymi segmentmi
                dash_length = 0.15 * self.camera.ortho_scale / 6.0

                # Horizontálna čiara (x-komponent) - zelená
                # Od offsetu po (end_x, oy) - ukazuje x-zložku
                if abs(vx) > 0.01:

                    glColor4f(0.85, 0.2, 0.2, line_alpha)
                    self._draw_dashed_line_2d(ox, oy, end_x, oy, dash_length, z=0.15)

                # Vertikálna čiara (y-komponent) - červená
                # Od (end_x, oy) po (end_x, end_y) - ukazuje y-zložku
                if abs(vy) > 0.01:
                    glColor4f(0.2, 0.85, 0.2, line_alpha)
                    self._draw_dashed_line_2d(end_x, oy, end_x, end_y, dash_length, z=0.15)

                glDisable(GL_BLEND)

                # Labely pre komponenty - v screen coordinates
                glMatrixMode(GL_PROJECTION)
                glPushMatrix()
                glLoadIdentity()
                glOrtho(0, width, height, 0, -1, 1)
                glMatrixMode(GL_MODELVIEW)
                glPushMatrix()
                glLoadIdentity()

                def world_to_screen(wx, wy):
                    sx = (wx - left) / (right - left) * width
                    sy = height - (wy - bottom) / (top - bottom) * height
                    return sx, sy

                # X-komponent label (na strede horizontálnej čiary)
                if abs(vx) > 0.01:
                    mid_x_screen, mid_y_screen = world_to_screen(ox + vx / 2, oy)
                    label_x = f"{vx:.1f}" if isinstance(vx, float) else str(int(vx))
                    # Posun label pod/nad čiaru
                    y_offset = 15 if vy >= 0 else -20
                    self.ui_renderer.draw_text_2d(label_x, (mid_x_screen - 10, mid_y_screen + y_offset),

                                                  color=(0.85, 0.2, 0.2), font_size=18)

                # Y-komponent label (na strede vertikálnej čiary)
                if abs(vy) > 0.01:
                    mid_x_screen, mid_y_screen = world_to_screen(end_x, oy + vy / 2)
                    label_y = f"{vy:.1f}" if isinstance(vy, float) else str(int(vy))
                    # Posun label vľavo/vpravo od čiary
                    x_offset = 10 if vx >= 0 else -35
                    self.ui_renderer.draw_text_2d(label_y, (mid_x_screen + x_offset, mid_y_screen - 8),
                                                  color=(0.2, 0.85, 0.2), font_size=18)

                # Vektor label v zátvorkách pri konci vektora
                vec_label_x, vec_label_y = world_to_screen(end_x, end_y)
                vx_str = f"{vx:.1f}" if isinstance(vx, float) else str(int(vx))
                vy_str = f"{vy:.1f}" if isinstance(vy, float) else str(int(vy))
                bracket_text_1 = f"[{vx_str}]"
                bracket_text_2 = f"[{vy_str}]"

                text_color = (1, 1, 1) if self.background_dark else (0, 0, 0)
                self.ui_renderer.draw_text_2d(bracket_text_1,
                                               (vec_label_x + 15, vec_label_y - 25),
                                               color=text_color, font_size=16)
                self.ui_renderer.draw_text_2d(bracket_text_2,
                                               (vec_label_x + 15, vec_label_y - 5),
                                               color=text_color, font_size=16)

                glPopMatrix()
                glMatrixMode(GL_PROJECTION)
                glPopMatrix()
                glMatrixMode(GL_MODELVIEW)

            else:
                # Matica - spracuj každý riadok/stĺpec ako vektor
                row_offsets = v.get('row_offsets', None)
                for i, row in enumerate(vec):
                    if row_offsets and i < len(row_offsets):
                        ox = row_offsets[i][0]
                        oy = row_offsets[i][1] if len(row_offsets[i]) > 1 else 0
                    else:
                        ox = offset[0]
                        oy = offset[1] if len(offset) > 1 else 0

                    vx = row[0]
                    vy = row[1] if len(row) > 1 else 0

                    if abs(vx) < 0.01 and abs(vy) < 0.01:
                        continue

                    end_x = ox + vx
                    end_y = oy + vy

                    line_alpha = min(v_alpha, 0.7)

                    glEnable(GL_BLEND)
                    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
                    glLineWidth(8.0)

                    dash_length = 0.15 * self.camera.ortho_scale / 6.0

                    if abs(vx) > 0.01:

                        glColor4f(0.85, 0.2, 0.2, line_alpha)
                        self._draw_dashed_line_2d(ox, oy, end_x, oy, dash_length, z=0.15)

                    if abs(vy) > 0.01:
                        glColor4f(0.2, 0.85, 0.2, line_alpha)
                        self._draw_dashed_line_2d(end_x, oy, end_x, end_y, dash_length, z=0.15)

                    glDisable(GL_BLEND)

    def _draw_dashed_line_2d(self, x1, y1, x2, y2, dash_length=0.2, z=0.15):
        """Vykreslí prerušovanú čiaru v 2D world coordinates"""
        dx = x2 - x1
        dy = y2 - y1
        length = math.sqrt(dx * dx + dy * dy)

        if length < 0.001:
            return

        # Normalizuj smer
        ndx = dx / length
        ndy = dy / length

        # Kresli segmenty
        num_dashes = max(1, int(length / (dash_length * 2)))
        actual_dash = length / (num_dashes * 2)

        glBegin(GL_LINES)
        for i in range(num_dashes):
            start_t = i * 2 * actual_dash
            end_t = start_t + actual_dash

            sx = x1 + ndx * start_t
            sy = y1 + ndy * start_t
            ex = x1 + ndx * end_t
            ey = y1 + ndy * end_t

            glVertex3f(sx, sy, z)
            glVertex3f(ex, ey, z)
        glEnd()

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
        arrow_size = max(0.15, min(0.12 * self.camera.ortho_scale, 2.0))
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

        text_color = (1, 1, 1) if self.background_dark else (0, 0, 0)

        # Toggle background button
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
        self.ui_renderer.draw_text_2d("ESC = späť do menu", (150, 20),
                                      color=text_color, font_size=14)

        # Helper lines indicator
        if self.view_2d_mode:
            helper_status = "M: Pomocné čiary [ZAP]" if self.show_helper_lines else "M: Pomocné čiary [VYP]"
            helper_color = (0.3, 0.9, 0.3) if self.show_helper_lines else text_color
            self.ui_renderer.draw_text_2d(helper_status, (150, 40),
                                          color=helper_color, font_size=14)

        # Draw vector toggle button
        draw_button_rect = pygame.Rect(self.width - 160, 120, 140, 35)
        button_label = "As Points" if self.vectors_as_points else "As Vectors"
        self.ui_renderer.draw_button_2d(
            draw_button_rect.x, draw_button_rect.y,
            draw_button_rect.width, draw_button_rect.height,
            button_label, active=True)

        # Basis matrix v 3D
        if self.saved_baza and not self.view_2d_mode:
            self.enhanced_ui.draw_basis_matrix_top_right(
                self.saved_baza[0],
                self.width, self.height,
                font_size=20,
                color=(0, 0, 0) if not self.background_dark else (1, 1, 1)
            )

        # === INDIKÁTOR KROKOVANIA OPERÁCIE ===
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

            # Informácie o rovine (len 3D)
            plane = ctrl.current_plane
            normal = ctrl.operation_plane_normal

            if plane and not self.view_2d_mode:
                self.ui_renderer.draw_text_2d(f"Rovina: {plane}",
                                              (self.width // 2 - 180, 95),
                                              color=text_color, font_size=16)
                if normal:
                    self.ui_renderer.draw_text_2d(
                        f"Normála: ({normal[0]:.2f}, {normal[1]:.2f}, {normal[2]:.2f})",
                        (self.width // 2 - 180, 115), color=text_color, font_size=14)

                self.ui_renderer.draw_text_2d(
                    f"Kamera: az={self.camera.azimuth:.1f}° el={self.camera.elevation:.1f}°",
                    (self.width // 2 - 180, 135), color=text_color, font_size=14)

                grid_modes = ["vypnutý", "grid", "grid jemný", "grid hrubý", "vyplnená + grid", "vyplnená"]
                self.ui_renderer.draw_text_2d(
                    f"Plane grid (B): {grid_modes[self.plane_grid_mode]}",
                    (self.width // 2 - 180, 155), color=text_color, font_size=14)

                if not self.camera.animating_to_plane:
                    self.ui_renderer.draw_text_2d("V=zobraz rovinu",
                                                  (self.width // 2 - 180, 175),
                                                  color=text_color, font_size=14)

        # === TRANSFORMÁCIA BÁZY INDIKÁTOR ===
        if self.transformation_controller.active:
            tc = self.transformation_controller
            desc = tc.get_step_description()

            self.ui_renderer.draw_text_2d(f"Transformácia bázy - Krok {tc.current_step}/{tc.max_steps}",
                                          (self.width // 2 - 180, 20),
                                          color=text_color, font_size=24)
            self.ui_renderer.draw_text_2d(desc, (self.width // 2 - 200, 50),
                                          color=text_color, font_size=18)
            self.ui_renderer.draw_text_2d("SPACE=ďalší | BACKSPACE=späť | C=zrušiť",
                                          (self.width // 2 - 180, 75),
                                          color=text_color, font_size=16)

            # Zobraz maticu
            mat = tc.get_matrix_display()
            if mat:
                mat_text = f"[{mat[0][0]:.1f}  {mat[0][1]:.1f}]"
                mat_text2 = f"[{mat[1][0]:.1f}  {mat[1][1]:.1f}]"
                self.ui_renderer.draw_text_2d("Matica:", (20, self.height - 120),
                                              color=text_color, font_size=16)
                self.ui_renderer.draw_text_2d(mat_text, (20, self.height - 100),
                                              color=text_color, font_size=18)
                self.ui_renderer.draw_text_2d(mat_text2, (20, self.height - 78),
                                              color=text_color, font_size=18)
                self.ui_renderer.draw_text_2d(f"det = {tc.determinant:.2f}",
                                              (20, self.height - 55),
                                              color=text_color, font_size=14)

        # === SPAN INDIKÁTOR ===
        if self.vector_manager.span_controller.active:
            span_ctrl = self.vector_manager.span_controller

            step_text = f"Span - Krok {span_ctrl.current_step + 1}"
            self.ui_renderer.draw_text_2d(step_text, (self.width // 2 - 150, 20),
                                          color=text_color, font_size=24)

            comb = span_ctrl.combinations[span_ctrl.current_step]
            desc = f"c1·v1 + c2·v2 kde c1={comb['c1']:.2f}, c2={comb['c2']:.2f}"
            self.ui_renderer.draw_text_2d(desc, (self.width // 2 - 200, 50),
                                          color=text_color, font_size=18)

            if span_ctrl.auto_play:
                control_text = "P=pauza | BACKSPACE=späť | C=zrušiť"
                play_status = "▶ AUTO-PLAY AKTÍVNY"
            else:
                control_text = "P=play | SPACE=ďalší | BACKSPACE=späť | C=zrušiť"
                play_status = "⏸ Manuálny režim"

            self.ui_renderer.draw_text_2d(control_text, (self.width // 2 - 200, 75),
                                          color=text_color, font_size=16)
            self.ui_renderer.draw_text_2d(play_status, (self.width // 2 - 120, 95),
                                          color=text_color, font_size=14)
            self.ui_renderer.draw_text_2d(
                "(Nekonečné krokovanie - nové kombinácie sa generujú priebežne)",
                (self.width // 2 - 240, 115), color=text_color, font_size=14)

        glEnable(GL_DEPTH_TEST)
        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)

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

        is_matrix_op = self.vector_manager.animation_controller._is_matrix(operands[0]) if operands else False
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