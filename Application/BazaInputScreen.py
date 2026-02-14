"""Obrazovka pre zadanie bázy a operácií — extrahované z Application.py"""

import numpy as np
import pygame
from pygame.locals import *
from OpenGL.GL import *

from Configurations.Colors import Colors
from Configurations.Config import Config


class BazaInputMixin:
    """Mixin trieda pre baza input screen logiku.
    Predpokladá že self má: camera, ui_renderer, vector_manager, input_handler,
    transformation_controller, matrix_inputs, matrix_active_cell, saved_baza,
    view_2d_mode, background_dark, width, height, running, is_not_baza, startup_screen
    """

    def run_baza_input(self):
        """Spustí obrazovku pre zadanie bázy - KOMPLETNÁ OPRAVENÁ VERZIA"""
        pending_input_panel = None
        span_input_panel = None

        while self.is_not_baza and self.running:
            dt = self.clock.tick(Config.FPS) / 1000.0

            rows = 2 if self.view_2d_mode else 3
            cols = 2 if self.view_2d_mode else 3

            needs_reinit = len(self.matrix_inputs) != rows
            if not needs_reinit and len(self.matrix_inputs) > 0:
                needs_reinit = len(self.matrix_inputs[0]) != cols
            if needs_reinit:
                self.matrix_inputs = [["" for _ in range(cols)] for _ in range(rows)]

            mx, my = pygame.mouse.get_pos()

            # === DEFINUJ RECT-y ===
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

            # === EVENT HANDLING ===
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
                    result = self._handle_baza_mouse_click(
                        mx, my, back_rect, std_basis_rect, span_rect, transform_rect,
                        toggle_rect, vec_op_rects, mat_op_rects, matrix_cell_rects,
                        vec_op_names, pending_input_panel, span_input_panel, rows, cols
                    )
                    if result:
                        action = result.get('action')
                        if action == 'back':
                            restart_loop = True
                            break
                        elif action == 'set_pending':
                            pending_input_panel = result.get('panel')
                            span_input_panel = None
                        elif action == 'set_span':
                            span_input_panel = result.get('panel')
                            pending_input_panel = None
                        elif action == 'done':
                            return
                        elif action == 'toggle_theme':
                            pass

                elif event.type == pygame.KEYDOWN:
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

                    result = self._handle_baza_keypress(
                        event, span_input_panel, pending_input_panel, rows, cols
                    )
                    if result:
                        action = result.get('action')
                        if action == 'clear_span':
                            span_input_panel = None
                        elif action == 'clear_pending':
                            pending_input_panel = None
                        elif action == 'done':
                            self.is_not_baza = False

            if restart_loop:
                continue

            # === RENDER ===
            self._render_baza_screen(
                mx, my, back_rect, toggle_rect, std_basis_rect, span_rect,
                transform_rect, vec_op_rects, mat_op_rects, matrix_cell_rects,
                pending_input_panel, span_input_panel, rows, cols,
                card_x, card_y, card_w, card_h, card2_x, basis_x, basis_y
            )
            pygame.display.flip()

    def _handle_baza_mouse_click(self, mx, my, back_rect, std_basis_rect, span_rect,
                                  transform_rect, toggle_rect, vec_op_rects, mat_op_rects,
                                  matrix_cell_rects, vec_op_names, pending_input_panel,
                                  span_input_panel, rows, cols):
        """Spracuje kliknutie myši na baza input obrazovke. Vráti dict s akciou."""
        # Späť
        if back_rect.collidepoint(mx, my):
            self.startup_screen = True
            self.is_not_baza = True
            self.matrix_active_cell = (-1, -1)
            self.run_startup_screen()
            if not self.running:
                return {'action': 'done'}
            new_rows = 2 if self.view_2d_mode else 3
            new_cols = 2 if self.view_2d_mode else 3
            self.matrix_inputs = [["" for _ in range(new_cols)] for _ in range(new_rows)]
            return {'action': 'back'}

        # Štandardná báza
        if std_basis_rect.collidepoint(mx, my):
            if self.view_2d_mode:
                self.saved_baza.append(((1, 0), (0, 1)))
            else:
                self.saved_baza.append(((1, 0, 0), (0, 1, 0), (0, 0, 1)))
            self.is_not_baza = False
            return {'action': 'done'}

        # Span
        if span_rect.collidepoint(mx, my) and self.view_2d_mode:
            panel = self._create_span_panel()
            return {'action': 'set_span', 'panel': panel}

        # Transformácia bázy
        if transform_rect.collidepoint(mx, my) and self.view_2d_mode:
            panel = self._create_transformation_panel()
            return {'action': 'set_pending', 'panel': panel}

        # Theme toggle
        if toggle_rect.collidepoint(mx, my):
            self.background_dark = not self.background_dark
            return {'action': 'toggle_theme'}

        # Vektorové operácie
        for i, rect in enumerate(vec_op_rects):
            if rect.collidepoint(mx, my):
                panel = self._create_operation_panel(vec_op_names[i], 'vector')
                if panel:
                    return {'action': 'set_pending', 'panel': panel}
                break

        # Maticové operácie
        for i, rect in enumerate(mat_op_rects):
            if rect.collidepoint(mx, my):
                panel = self._create_operation_panel(vec_op_names[i], 'matrix')
                if panel:
                    return {'action': 'set_pending', 'panel': panel}
                break

        # Panel kliknutia
        if pending_input_panel:
            self._handle_panel_click(mx, my, pending_input_panel)

        if span_input_panel:
            self._handle_span_click(mx, my, span_input_panel)

        # Matrix cells pre vlastnú bázu
        if not pending_input_panel and not span_input_panel:
            for r in range(rows):
                for c in range(cols):
                    if matrix_cell_rects[r][c].collidepoint(mx, my):
                        self.matrix_active_cell = (r, c)
                        break

        return None

    def _create_span_panel(self):
        """Vytvorí panel pre span input"""
        panel_width = 2 * (Config.MATRIX_CELL_W + Config.MATRIX_GAP)
        panel_height = 2 * (Config.MATRIX_CELL_H + Config.MATRIX_GAP)
        start_x = self.width // 2 - panel_width // 2
        start_y = self.height // 2 - panel_height // 2
        return {
            'values': [["" for _ in range(2)] for _ in range(2)],
            'active_cell': (0, 0),
            'x': start_x,
            'y': start_y
        }

    def _create_transformation_panel(self):
        """Vytvorí panel pre transformáciu bázy"""
        panel_width = 2 * (Config.MATRIX_CELL_W + Config.MATRIX_GAP)
        panel_height = 2 * (Config.MATRIX_CELL_H + Config.MATRIX_GAP)
        start_x = self.width // 2 - panel_width // 2
        start_y = self.height // 2 - panel_height // 2
        return {
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

    def _create_operation_panel(self, name, op_type):
        """Vytvorí input panel pre danú operáciu"""
        rows_panel = 2 if self.view_2d_mode else 3
        cols_panel = rows_panel if op_type == 'matrix' else 1

        if name == "Lineárna kombinácia":
            return self._create_linear_combination_panel(op_type, rows_panel, cols_panel)

        if name in ["Sčítania", "Odčitanie"]:
            symbol = "+" if name == "Sčítania" else "-"
            num_panels = 2
            has_constant = False
        elif name == "Násobenie Konštantou":
            symbol = "×"
            num_panels = 1
            has_constant = True
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
                "type": "constant", "rows": 1, "cols": 1,
                "values": [[""]], "active_cell": (0, 0),
                "x": start_x,
                "y": self.height // 2 - Config.MATRIX_CELL_H // 2
            })
            panels.append({
                "type": op_type, "rows": rows_panel, "cols": cols_panel,
                "values": [["" for _ in range(cols_panel)] for _ in range(rows_panel)],
                "active_cell": (-1, -1),
                "x": start_x + constant_width + symbol_width,
                "y": self.height // 2 - (rows_panel * (Config.MATRIX_CELL_H + Config.MATRIX_GAP)) // 2
            })
        else:
            for j in range(num_panels):
                panels.append({
                    "type": op_type, "rows": rows_panel, "cols": cols_panel,
                    "values": [["" for _ in range(cols_panel)] for _ in range(rows_panel)],
                    "active_cell": (0, 0) if j == 0 else (-1, -1),
                    "x": start_x + j * (panel_width + symbol_width),
                    "y": self.height // 2 - (rows_panel * (Config.MATRIX_CELL_H + Config.MATRIX_GAP)) // 2
                })

        return {
            "type": op_type, "operation": name, "symbol": symbol,
            "num_panels": len(panels), "has_constant": has_constant,
            "panels": panels, "active_panel": 0
        }

    def _create_linear_combination_panel(self, op_type, rows_panel, cols_panel):
        """Vytvorí panel pre lineárnu kombináciu"""
        constant_width = 60
        data_panel_width = cols_panel * (Config.MATRIX_CELL_W + Config.MATRIX_GAP)
        symbol_width = 30
        total_width = (constant_width + data_panel_width + symbol_width) * 2
        start_x = self.width // 2 - total_width // 2

        panels = []
        current_x = start_x

        for idx, (c_label, v_label) in enumerate([("c1", "v1" if op_type == "vector" else "mat1"),
                                                    ("c2", "v2" if op_type == "vector" else "mat2")]):
            panels.append({
                "type": "constant", "rows": 1, "cols": 1,
                "values": [[""]], "active_cell": (0, 0) if idx == 0 else (-1, -1),
                "x": current_x,
                "y": self.height // 2 - Config.MATRIX_CELL_H // 2,
                "label": c_label
            })
            current_x += constant_width + 10

            panels.append({
                "type": op_type, "rows": rows_panel, "cols": cols_panel,
                "values": [["" for _ in range(cols_panel)] for _ in range(rows_panel)],
                "active_cell": (-1, -1),
                "x": current_x,
                "y": self.height // 2 - (rows_panel * (Config.MATRIX_CELL_H + Config.MATRIX_GAP)) // 2,
                "label": v_label
            })
            current_x += data_panel_width + 30

        return {
            "type": op_type, "operation": "Lineárna kombinácia", "symbol": None,
            "num_panels": len(panels), "has_constant": False,
            "panels": panels, "active_panel": 0,
            "is_linear_combination": True
        }

    def _handle_panel_click(self, mx, my, pending_input_panel):
        """Spracuje kliknutie na input panel bunky"""
        for panel_idx, panel in enumerate(pending_input_panel["panels"]):
            if panel.get("type") == "constant":
                rect = pygame.Rect(panel["x"], panel["y"], 50, Config.MATRIX_CELL_H)
                if rect.collidepoint(mx, my):
                    for p in pending_input_panel["panels"]:
                        p["active_cell"] = (-1, -1)
                    pending_input_panel["active_panel"] = panel_idx
                    panel["active_cell"] = (0, 0)
                    self.matrix_active_cell = (-1, -1)
                    return
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
                            return

    def _handle_span_click(self, mx, my, span_input_panel):
        """Spracuje kliknutie na span panel bunky"""
        for r in range(2):
            for c in range(2):
                x = span_input_panel['x'] + c * (Config.MATRIX_CELL_W + Config.MATRIX_GAP)
                y = span_input_panel['y'] + r * (Config.MATRIX_CELL_H + Config.MATRIX_GAP)
                rect = pygame.Rect(x, y, Config.MATRIX_CELL_W, Config.MATRIX_CELL_H)
                if rect.collidepoint(mx, my):
                    span_input_panel['active_cell'] = (r, c)
                    return

    def _handle_baza_keypress(self, event, span_input_panel, pending_input_panel, rows, cols):
        """Spracuje stlačenie klávesy na baza input obrazovke"""
        span_handled = False
        pending_handled = False

        # === SPAN PANEL INPUT ===
        if span_input_panel:
            r, c = span_input_panel['active_cell']
            span_handled = True

            if event.key == pygame.K_RETURN:
                all_filled = all(val.strip() for row in span_input_panel['values'] for val in row)
                if all_filled:
                    try:
                        v1 = [float(span_input_panel['values'][0][0]),
                              float(span_input_panel['values'][1][0])]
                        v2 = [float(span_input_panel['values'][0][1]),
                              float(span_input_panel['values'][1][1])]
                        self.vector_manager.span_controller.setup_span(v1, v2)
                        self.saved_baza.append(((1, 0), (0, 1)))
                        return {'action': 'done'}
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
                    return self._process_pending_submit(pending_input_panel)
                elif event.key == pygame.K_BACKSPACE:
                    panel["values"][r][c] = panel["values"][r][c][:-1]
                elif event.unicode.isdigit() or event.unicode in ".-":
                    panel["values"][r][c] += event.unicode

        # === Originálna báza input ===
        if not span_handled and not pending_handled and self.matrix_active_cell != (-1, -1):
            r, c = self.matrix_active_cell
            if event.key == pygame.K_RETURN:
                return self._process_custom_basis_submit(rows, cols)
            elif event.key == pygame.K_BACKSPACE:
                self.matrix_inputs[r][c] = self.matrix_inputs[r][c][:-1]
            elif event.unicode.isdigit() or event.unicode in ".-":
                self.matrix_inputs[r][c] += event.unicode

        return None

    def _process_pending_submit(self, pending_input_panel):
        """Spracuje potvrdenie pending input panelu"""
        all_filled = all(
            val.strip()
            for p in pending_input_panel["panels"]
            for row in p["values"]
            for val in row
        )
        if not all_filled:
            return None

        # Transformácia bázy
        if pending_input_panel.get('is_transformation', False):
            try:
                panel = pending_input_panel["panels"][0]
                mat = [[float(v) for v in row] for row in panel["values"]]
                self.transformation_controller.setup_transformation(mat, is_2d=True)
                self.saved_baza.append(((1, 0), (0, 1)))
                return {'action': 'done'}
            except Exception as e:
                print(f"Chyba pri parsovaní transformačnej matice: {e}")
                return None

        # Bežné operácie
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
        op_type = pending_input_panel["type"]

        try:
            if op_type == "vector":
                self._process_vector_operation(operation, data_panels, constant)
            else:
                self._process_matrix_operation(operation, data_panels, constant)
        except Exception as e:
            print(f"Chyba pri spracovaní operácie: {e}")
            return None

        if not self.view_2d_mode:
            self._move_camera_to_operation_plane()

        return {'action': 'clear_pending'}

    def _process_vector_operation(self, operation, data_panels, constant):
        """Spracuje vektorovú operáciu"""
        ctrl = self.vector_manager.animation_controller

        if operation == "Sčítania":
            vec1 = [float(v[0]) for v in data_panels[0]["values"]]
            vec2 = [float(v[0]) for v in data_panels[1]["values"]]
            result = [a + b for a, b in zip(vec1, vec2)]
            ctrl.setup_operation('add', [vec1, vec2], result)
        elif operation == "Odčitanie":
            vec1 = [float(v[0]) for v in data_panels[0]["values"]]
            vec2 = [float(v[0]) for v in data_panels[1]["values"]]
            result = [a - b for a, b in zip(vec1, vec2)]
            ctrl.setup_operation('subtract', [vec1, vec2], result)
        elif operation == "Násobenie Konštantou":
            vec = [float(v[0]) for v in data_panels[0]["values"]]
            result = [constant[0] * v for v in vec]
            ctrl.setup_operation('scalar_mult', [vec], result, constant[0])
        elif operation == "Lineárna kombinácia":
            vec1 = [float(v[0]) for v in data_panels[0]["values"]]
            vec2 = [float(v[0]) for v in data_panels[1]["values"]]
            c1, c2 = constant[0], constant[1]
            result = [c1 * v1 + c2 * v2 for v1, v2 in zip(vec1, vec2)]
            ctrl.setup_operation('linear_combination', [vec1, vec2], result, [c1, c2])

    def _process_matrix_operation(self, operation, data_panels, constant):
        """Spracuje maticovú operáciu"""
        ctrl = self.vector_manager.animation_controller

        if operation == "Sčítania":
            mat1 = [[float(v) for v in row] for row in data_panels[0]["values"]]
            mat2 = [[float(v) for v in row] for row in data_panels[1]["values"]]
            result = [[a + b for a, b in zip(r1, r2)] for r1, r2 in zip(mat1, mat2)]
            ctrl.setup_operation('add', [mat1, mat2], result)
        elif operation == "Odčitanie":
            mat1 = [[float(v) for v in row] for row in data_panels[0]["values"]]
            mat2 = [[float(v) for v in row] for row in data_panels[1]["values"]]
            result = [[a - b for a, b in zip(r1, r2)] for r1, r2 in zip(mat1, mat2)]
            ctrl.setup_operation('subtract', [mat1, mat2], result)
        elif operation == "Násobenie Konštantou":
            mat = [[float(v) for v in row] for row in data_panels[0]["values"]]
            result = [[constant[0] * v for v in row] for row in mat]
            ctrl.setup_operation('scalar_mult', [mat], result, constant[0])
        elif operation == "Lineárna kombinácia":
            mat1 = [[float(v) for v in row] for row in data_panels[0]["values"]]
            mat2 = [[float(v) for v in row] for row in data_panels[1]["values"]]
            c1, c2 = constant[0], constant[1]
            result = [[c1 * v1 + c2 * v2 for v1, v2 in zip(r1, r2)] for r1, r2 in zip(mat1, mat2)]
            ctrl.setup_operation('linear_combination', [mat1, mat2], result, [c1, c2])

    def _process_custom_basis_submit(self, rows, cols):
        """Spracuje potvrdenie vlastnej bázy"""
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
                        return None
                else:
                    if mat_np.shape != (3, 3) or np.linalg.matrix_rank(mat_np) < 3:
                        print("Zadané vektory NEtvoria bázu 3D!")
                        return None

                self.saved_baza.append(tuple(matrix_values))
                self.matrix_inputs = [["" for _ in range(cols)] for _ in range(rows)]
                self.matrix_active_cell = (-1, -1)
                return {'action': 'done'}
        except Exception as e:
            print("Invalid matrix input:", e)
        return None

    def _render_baza_screen(self, mx, my, back_rect, toggle_rect, std_basis_rect,
                             span_rect, transform_rect, vec_op_rects, mat_op_rects,
                             matrix_cell_rects, pending_input_panel, span_input_panel,
                             rows, cols, card_x, card_y, card_w, card_h, card2_x,
                             basis_x, basis_y):
        """Vykreslí baza input obrazovku"""
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

        # Späť
        hover_back = back_rect.collidepoint(mx, my)
        self.ui_renderer.draw_button_2d(back_rect.x, back_rect.y, back_rect.w, back_rect.h,
                                        "← Späť", is_dark=self.background_dark, hover=hover_back)
        self.ui_renderer.draw_text_2d("ESC", (120, 25), sub_col, 12)

        # Titulok
        mode_str = "2D" if self.view_2d_mode else "3D"
        self.ui_renderer.draw_text_2d(f"Operácie ({mode_str})", (self.width // 2 - 60, 20), text_col, 22)

        # Vektorové operácie
        self.ui_renderer.draw_card(card_x, card_y, card_w, card_h, self.background_dark, "Vektorové operácie")
        vec_labels = ["Sčítanie", "Odčítanie", "Násobenie k.", "Lin. kombinácia"]
        for i, rect in enumerate(vec_op_rects):
            hover = rect.collidepoint(mx, my)
            self.ui_renderer.draw_button_2d(rect.x, rect.y, rect.w, rect.h,
                                            vec_labels[i], is_dark=self.background_dark, hover=hover)

        # Maticové operácie
        self.ui_renderer.draw_card(card2_x, card_y, card_w, card_h, self.background_dark, "Maticové operácie")
        for i, rect in enumerate(mat_op_rects):
            hover = rect.collidepoint(mx, my)
            self.ui_renderer.draw_button_2d(rect.x, rect.y, rect.w, rect.h,
                                            vec_labels[i], is_dark=self.background_dark, hover=hover)

        # Štandardná báza
        hover_std = std_basis_rect.collidepoint(mx, my)
        self.ui_renderer.draw_button_2d(std_basis_rect.x, std_basis_rect.y,
                                        std_basis_rect.w, std_basis_rect.h,
                                        "Štandardná báza", is_dark=self.background_dark,
                                        hover=hover_std, primary=True)
        basis_desc = "Identitná matica I₂" if self.view_2d_mode else "Identitná matica I₃"
        self.ui_renderer.draw_text_2d(basis_desc, (basis_x + 30, basis_y + 52), sub_col, 13)

        # Span
        if self.view_2d_mode:
            hover_span = span_rect.collidepoint(mx, my)
            self.ui_renderer.draw_button_2d(span_rect.x, span_rect.y, span_rect.w, span_rect.h,
                                            "Span (Lineárny obal)", is_dark=self.background_dark,
                                            hover=hover_span, primary=True)
            self.ui_renderer.draw_text_2d("Zadaj 2 vektory (stĺpce)",
                                          (basis_x + 20, span_rect.y + 48), sub_col, 13)

        # Transformácia bázy
        if self.view_2d_mode:
            hover_transform = transform_rect.collidepoint(mx, my)
            self.ui_renderer.draw_button_2d(transform_rect.x, transform_rect.y,
                                            transform_rect.w, transform_rect.h,
                                            "Transformácia bázy", is_dark=self.background_dark,
                                            hover=hover_transform, primary=True)
            self.ui_renderer.draw_text_2d("Animovaná zmena bázy (3b1b)",
                                          (basis_x + 5, transform_rect.y + 48), sub_col, 13)

        # Vlastná báza
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

        # Theme toggle
        hover_toggle = toggle_rect.collidepoint(mx, my)
        theme_label = "Tmavý režim" if self.background_dark else "Svetlý režim"
        self.ui_renderer.draw_button_2d(toggle_rect.x, toggle_rect.y, toggle_rect.w, toggle_rect.h,
                                        theme_label, is_dark=self.background_dark, hover=hover_toggle)

        # Input panely
        if pending_input_panel:
            self._render_pending_panel(pending_input_panel, sub_col)
        if span_input_panel:
            self._render_span_panel(span_input_panel, text_col, sub_col)

    def _render_pending_panel(self, pending_input_panel, sub_col):
        """Vykreslí pending input panel"""
        color_symbol = (1, 1, 1) if self.background_dark else (0, 0, 0)

        if pending_input_panel.get("is_transformation"):
            panel = pending_input_panel["panels"][0]
            panel_width = 2 * (Config.MATRIX_CELL_W + Config.MATRIX_GAP) + 40
            panel_height = 2 * (Config.MATRIX_CELL_H + Config.MATRIX_GAP) + 80
            bg_x = panel["x"] - 20
            bg_y = panel["y"] - 40

            self.ui_renderer.draw_card(bg_x, bg_y, panel_width, panel_height,
                                       self.background_dark, "Transformácia bázy")

            self.ui_renderer.draw_text_2d("i", (panel["x"] + 10, panel["y"] - 25),
                                          (0.85, 0.2, 0.2), 18)
            self.ui_renderer.draw_text_2d("j",
                                          (panel["x"] + Config.MATRIX_CELL_W + Config.MATRIX_GAP + 10,
                                           panel["y"] - 25),
                                          (0.2, 0.85, 0.2), 18)

            for r in range(panel["rows"]):
                for c in range(panel["cols"]):
                    x = panel["x"] + c * (Config.MATRIX_CELL_W + Config.MATRIX_GAP)
                    y = panel["y"] + r * (Config.MATRIX_CELL_H + Config.MATRIX_GAP)
                    active = (r, c) == panel["active_cell"]
                    col_color = [(0.85, 0.2, 0.2), (0.2, 0.85, 0.2)][c]
                    self.ui_renderer.draw_input_box_3d(x, y, Config.MATRIX_CELL_W,
                                                       Config.MATRIX_CELL_H,
                                                       panel["values"][r][c], active,
                                                       fill_color_outline=col_color,
                                                       is_dark=self.background_dark)

            help_y = panel["y"] + 2 * (Config.MATRIX_CELL_H + Config.MATRIX_GAP) + 10
            self.ui_renderer.draw_text_2d("Stĺpce = kam pôjde i a j",
                                          (panel["x"] - 30, help_y), sub_col, 14)
            self.ui_renderer.draw_text_2d("Enter = potvrdiť | ESC = zrušiť",
                                          (panel["x"] - 50, help_y + 20), sub_col, 14)

        elif pending_input_panel.get("is_linear_combination"):
            self._render_linear_combination_panel(pending_input_panel, color_symbol)
        else:
            self._render_standard_panel(pending_input_panel, color_symbol)

    def _render_linear_combination_panel(self, panel_data, color_symbol):
        """Vykreslí panel lineárnej kombinácie"""
        for panel_idx, panel in enumerate(panel_data["panels"]):
            is_active_panel = (panel_idx == panel_data["active_panel"])
            if panel.get("type") == "constant":
                x, y = panel["x"], panel["y"]
                active = is_active_panel and panel["active_cell"] == (0, 0)
                self.ui_renderer.draw_input_box_3d(x, y, 50, Config.MATRIX_CELL_H,
                                                   panel["values"][0][0], active)
                label = panel.get("label", "")
                if label:
                    self.ui_renderer.draw_text_2d(label, (x + 15, y - 25),
                                                  color=color_symbol, font_size=18)
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

            # Symboly medzi panelmi
            if panel_idx == 0:
                sx = panel["x"] + 50 + 5
                sy = panel["y"] + Config.MATRIX_CELL_H // 2 - 10
                self.ui_renderer.draw_text_2d("×", (sx, sy), color=color_symbol, font_size=30)
            elif panel_idx == 1:
                sx = panel["x"] + panel["cols"] * (Config.MATRIX_CELL_W + Config.MATRIX_GAP) + 10
                ph = panel["rows"] * (Config.MATRIX_CELL_H + Config.MATRIX_GAP) - Config.MATRIX_GAP
                sy = panel["y"] + (ph - 35) // 2
                self.ui_renderer.draw_text_2d("+", (sx, sy), color=color_symbol, font_size=35)
            elif panel_idx == 2:
                sx = panel["x"] + 50 + 5
                sy = panel["y"] + Config.MATRIX_CELL_H // 2 - 10
                self.ui_renderer.draw_text_2d("×", (sx, sy), color=color_symbol, font_size=30)

    def _render_standard_panel(self, panel_data, color_symbol):
        """Vykreslí štandardný operačný panel"""
        for panel_idx, panel in enumerate(panel_data["panels"]):
            is_active_panel = (panel_idx == panel_data["active_panel"])
            if panel.get("type") == "constant":
                x, y = panel["x"], panel["y"]
                active = is_active_panel and panel["active_cell"] == (0, 0)
                self.ui_renderer.draw_input_box_3d(x, y, 50, Config.MATRIX_CELL_H,
                                                   panel["values"][0][0], active)
            else:
                for r in range(panel["rows"]):
                    for c in range(panel["cols"]):
                        x = panel["x"] + c * (Config.MATRIX_CELL_W + Config.MATRIX_GAP)
                        y = panel["y"] + r * (Config.MATRIX_CELL_H + Config.MATRIX_GAP)
                        active = is_active_panel and (r, c) == panel["active_cell"]
                        self.ui_renderer.draw_input_box_3d(x, y, Config.MATRIX_CELL_W,
                                                           Config.MATRIX_CELL_H,
                                                           panel["values"][r][c], active)

            if panel_data["symbol"]:
                if panel_data.get("has_constant") and panel_idx == 0:
                    sx = panel["x"] + 50 + 5
                    sy = panel["y"] + Config.MATRIX_CELL_H // 2 - 13
                    self.ui_renderer.draw_text_2d(panel_data["symbol"], (sx, sy),
                                                  color=color_symbol, font_size=40)
                elif not panel_data.get("has_constant") and panel_idx < len(panel_data["panels"]) - 1:
                    sx = panel["x"] + panel["cols"] * (Config.MATRIX_CELL_W + Config.MATRIX_GAP) + 10
                    ph = panel["rows"] * (Config.MATRIX_CELL_H + Config.MATRIX_GAP) - Config.MATRIX_GAP
                    sy = panel["y"] + (ph - 40) // 2
                    self.ui_renderer.draw_text_2d(panel_data["symbol"], (sx, sy),
                                                  color=color_symbol, font_size=40)

    def _render_span_panel(self, span_input_panel, text_col, sub_col):
        """Vykreslí span input panel"""
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

        self.ui_renderer.draw_text_2d("v1", (span_input_panel['x'] + 5, span_input_panel['y'] - 25),
                                      text_col, 16)
        self.ui_renderer.draw_text_2d("v2",
                                      (span_input_panel['x'] + Config.MATRIX_CELL_W + Config.MATRIX_GAP + 5,
                                       span_input_panel['y'] - 25), text_col, 16)

        help_y = span_input_panel['y'] + 2 * (Config.MATRIX_CELL_H + Config.MATRIX_GAP) + 10
        self.ui_renderer.draw_text_2d("Enter = potvrdiť | ESC = zrušiť",
                                      (span_input_panel['x'] - 50, help_y), sub_col, 14)
