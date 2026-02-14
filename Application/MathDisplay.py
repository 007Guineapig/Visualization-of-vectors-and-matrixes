"""Matematické zobrazenie operácií — extrahované z Application.py"""

from Rendering.MathRenderer import MathRenderer


class MathDisplayMixin:
    """Mixin trieda pre render_math_display.
    Predpokladá self má: vector_manager, background_dark, ui_renderer, RESULT_COLOR
    """

    @staticmethod
    def _flatten_for_display(data):
        """Flatten matice na vektor pre zobrazenie"""
        if isinstance(data[0], (list, tuple)):
            return [row[0] if len(row) > 0 else 0 for row in data]
        return data

    def render_math_display(self, math_info):
        """Vykreslí matematické zobrazenie operácie - S ANIMÁCIOU"""
        math_renderer = MathRenderer()

        start_x = 20
        start_y = 80

        op_type = math_info['operation']
        step = math_info['step']
        operands = math_info['operands']
        result = math_info['result']
        constant = math_info['constant']

        ctrl = self.vector_manager.animation_controller
        anim_progress = ctrl.animation_progress if ctrl.animating else 1.0
        text_color = (1, 1, 1) if self.background_dark else (0, 0, 0)

        if op_type == 'add':
            self._render_math_add(math_renderer, start_x, start_y, operands, result,
                                  step, anim_progress, text_color)
        elif op_type == 'subtract':
            self._render_math_subtract(math_renderer, start_x, start_y, operands, result,
                                       step, anim_progress, text_color)
        elif op_type == 'scalar_mult':
            self._render_math_scalar_mult(math_renderer, start_x, start_y, operands, result,
                                          constant, step, anim_progress, text_color)
        elif op_type == 'linear_combination':
            self._render_math_linear_combination(math_renderer, start_x, start_y, operands,
                                                  result, constant, step, anim_progress, text_color)

    def _render_math_add(self, mr, sx, sy, operands, result, step, anim, tc):
        """Render sčítania"""
        f = self._flatten_for_display
        op1, op2, res = f(operands[0]), f(operands[1]), f(result)
        cx = sx

        if step == 0:
            cx = mr.draw_vector_math(cx, sy, op1, color=(0.8, 0.8, 0.8), font_size=22)
            cx = mr.draw_operator(cx, sy, "+", color=tc, font_size=28)
            cx = mr.draw_vector_math(cx, sy, op2, color=(0.8, 0.8, 0.8), font_size=22)

        elif step == 1:
            cx = mr.draw_vector_math(cx, sy, op1, color=(1, 0.5, 0), font_size=22, highlight=True)
            cx = mr.draw_operator(cx, sy, "+", color=tc, font_size=28)
            cx = mr.draw_vector_math(cx, sy, op2, color=(0, 0.5, 1), font_size=22, highlight=True)

            cx = sx
            sy += len(op1) * 27 + 20
            adj = tuple(0.8 * anim for _ in range(3))
            mr.draw_vector_component_operation(cx, sy, op1, op2, "+", color=adj, font_size=20)

        elif step == 2:
            cx = mr.draw_vector_math(cx, sy, op1, color=(1, 0.5, 0), font_size=22, highlight=True)
            cx = mr.draw_operator(cx, sy, "+", color=tc, font_size=28)
            cx = mr.draw_vector_math(cx, sy, op2, color=(0, 0.5, 1), font_size=22, highlight=True)
            cx = mr.draw_equals_animated(cx, sy, anim, color=tc, font_size=28)

            anim_res = [r * anim for r in res] if anim < 1.0 else res
            cx = mr.draw_vector_math(cx, sy, anim_res, color=self.RESULT_COLOR, font_size=22, highlight=True)

            cx = sx
            sy += len(op1) * 27 + 20
            mr.draw_component_operation_animated(cx, sy, op1, op2, "+", anim, color=(0.8, 0.8, 0.8), font_size=20)

        elif step == 3:
            # Komutatívnosť
            self.ui_renderer.draw_text_2d("Pôvodné:", (sx, sy - 25), color=tc, font_size=16)
            cx = mr.draw_vector_math(cx, sy, op1, color=(1, 0.5, 0), font_size=20, highlight=True)
            cx = mr.draw_operator(cx, sy, "+", color=tc, font_size=24)
            cx = mr.draw_vector_math(cx, sy, op2, color=(0, 0.5, 1), font_size=20, highlight=True)
            cx = mr.draw_equals(cx, sy, color=tc, font_size=24)
            cx = mr.draw_vector_math(cx, sy, res, color=self.RESULT_COLOR, font_size=20, highlight=True)

            cx = sx
            sy += len(op1) * 25 + 25
            new_c = tuple(tc[i] * anim for i in range(3))
            self.ui_renderer.draw_text_2d("Nové (komutatívnosť):", (sx, sy - 25), color=new_c, font_size=16)

            v2c = (0.3 * anim + 0.1, 0.8 * anim + 0.1, 1.0 * anim)
            v1c = (1.0 * anim, 0.8 * anim + 0.1, 0.4 * anim + 0.1)
            cx = mr.draw_vector_math(cx, sy, op2, color=v2c, font_size=20, highlight=True)
            cx = mr.draw_operator(cx, sy, "+", color=new_c, font_size=24)
            cx = mr.draw_vector_math(cx, sy, op1, color=v1c, font_size=20, highlight=True)
            cx = mr.draw_equals_animated(cx, sy, anim, color=tc, font_size=24)

            rc = tuple(self.RESULT_COLOR[i] * anim + 0.1 for i in range(3))
            cx = mr.draw_vector_math(cx, sy, res, color=rc, font_size=20, highlight=True)

            if anim >= 0.8:
                cx = sx
                sy += len(op1) * 25 + 20
                fa = (anim - 0.8) / 0.2
                fc = tuple(tc[i] * fa for i in range(3))
                self.ui_renderer.draw_text_2d("→ Oba spôsoby vedú k rovnakému výsledku",
                                              (cx, sy), color=fc, font_size=16)

    def _render_math_subtract(self, mr, sx, sy, operands, result, step, anim, tc):
        """Render odčítania"""
        f = self._flatten_for_display
        op1, op2, res = f(operands[0]), f(operands[1]), f(result)
        neg_op2 = [-x for x in op2]
        cx = sx

        if step == 0:
            cx = mr.draw_vector_math(cx, sy, op1, color=(0.8, 0.8, 0.8), font_size=22)
            cx = mr.draw_operator(cx, sy, "-", color=tc, font_size=28)
            cx = mr.draw_vector_math(cx, sy, op2, color=(0.8, 0.8, 0.8), font_size=22)

        elif step == 1:
            cx = mr.draw_vector_math(cx, sy, op1, color=(1, 0.5, 0), font_size=22, highlight=True)
            cx = mr.draw_operator(cx, sy, "+", color=tc, font_size=28)
            anim_neg = [op2[i] + (neg_op2[i] - op2[i]) * anim for i in range(len(op2))] if anim < 1.0 else neg_op2
            cx = mr.draw_vector_math(cx, sy, anim_neg, color=(1, 0, 0.5), font_size=22, highlight=True)

        elif step == 2:
            cx = mr.draw_vector_math(cx, sy, op1, color=(1, 0.5, 0), font_size=22, highlight=True)
            cx = mr.draw_operator(cx, sy, "+", color=tc, font_size=28)
            cx = mr.draw_vector_math(cx, sy, neg_op2, color=(1, 0, 0.5), font_size=22, highlight=True)

            cx = sx
            sy += len(op1) * 27 + 20
            adj = tuple(0.8 * anim for _ in range(3))
            mr.draw_vector_component_operation(cx, sy, op1, neg_op2, "+", color=adj, font_size=20)

        elif step == 3:
            cx = mr.draw_vector_math(cx, sy, op1, color=(1, 0.5, 0), font_size=22, highlight=True)
            cx = mr.draw_operator(cx, sy, "+", color=tc, font_size=28)
            cx = mr.draw_vector_math(cx, sy, neg_op2, color=(1, 0, 0.5), font_size=22, highlight=True)
            cx = mr.draw_equals_animated(cx, sy, anim, color=tc, font_size=28)
            anim_res = [r * anim for r in res] if anim < 1.0 else res
            cx = mr.draw_vector_math(cx, sy, anim_res, color=self.RESULT_COLOR, font_size=22, highlight=True)

    def _render_math_scalar_mult(self, mr, sx, sy, operands, result, constant, step, anim, tc):
        """Render násobenia konštantou"""
        f = self._flatten_for_display
        op1, res = f(operands[0]), f(result)
        cx = sx

        if step == 0:
            cx = mr.draw_scalar(cx, sy, constant, color=tc, font_size=26)
            cx = mr.draw_operator(cx - 10, sy, "·", color=tc, font_size=28)
            cx = mr.draw_vector_math(cx, sy, op1, color=(1, 0.5, 0), font_size=22, highlight=True)

        elif step == 1:
            cx = mr.draw_scalar(cx, sy, constant, color=tc, font_size=26)
            cx = mr.draw_operator(cx - 10, sy, "·", color=tc, font_size=28)
            cx = mr.draw_vector_math(cx, sy, op1, color=(1, 0.5, 0), font_size=22, highlight=True)
            cx = mr.draw_equals_animated(cx, sy, anim, color=tc, font_size=28)
            anim_res = [op1[i] + (res[i] - op1[i]) * anim for i in range(len(res))] if anim < 1.0 else res
            cx = mr.draw_vector_math(cx, sy, anim_res, color=self.RESULT_COLOR, font_size=22, highlight=True)

    def _render_math_linear_combination(self, mr, sx, sy, operands, result, constant, step, anim, tc):
        """Render lineárnej kombinácie"""
        f = self._flatten_for_display
        c1, c2 = constant if constant else [1, 1]
        op1, op2, res = f(operands[0]), f(operands[1]), f(result)
        sv1 = [c1 * x for x in op1]
        sv2 = [c2 * x for x in op2]
        cx = sx

        if step == 0:
            cx = mr.draw_scalar(cx, sy, c1, color=tc, font_size=24)
            cx = mr.draw_operator(cx - 10, sy, "·", color=tc, font_size=26)
            cx = mr.draw_vector_math(cx, sy, op1, color=(0.8, 0.8, 0.8), font_size=20)
            cx = mr.draw_operator(cx, sy, "+", color=tc, font_size=26)
            cx = mr.draw_scalar(cx, sy, c2, color=tc, font_size=24)
            cx = mr.draw_operator(cx - 10, sy, "·", color=tc, font_size=26)
            cx = mr.draw_vector_math(cx, sy, op2, color=(0.8, 0.8, 0.8), font_size=20)

        elif step == 1:
            anim_s = [op1[i] + (sv1[i] - op1[i]) * anim for i in range(len(op1))] if anim < 1.0 else sv1
            cx = mr.draw_vector_math(cx, sy, anim_s, color=(1, 0.5, 0), font_size=20, highlight=True)
            cx = mr.draw_operator(cx, sy, "+", color=tc, font_size=26)
            cx = mr.draw_scalar(cx, sy, c2, color=tc, font_size=24)
            cx = mr.draw_operator(cx - 10, sy, "·", color=tc, font_size=26)
            cx = mr.draw_vector_math(cx, sy, op2, color=(0.8, 0.8, 0.8), font_size=20)

        elif step == 2:
            cx = mr.draw_vector_math(cx, sy, sv1, color=(1, 0.5, 0), font_size=20, highlight=True)
            cx = mr.draw_operator(cx, sy, "+", color=tc, font_size=26)
            anim_s = [op2[i] + (sv2[i] - op2[i]) * anim for i in range(len(op2))] if anim < 1.0 else sv2
            cx = mr.draw_vector_math(cx, sy, anim_s, color=(0, 0.5, 1), font_size=20, highlight=True)

        elif step == 3:
            cx = mr.draw_vector_math(cx, sy, sv1, color=(1, 0.5, 0), font_size=20, highlight=True)
            cx = mr.draw_operator(cx, sy, "+", color=tc, font_size=26)
            cx = mr.draw_vector_math(cx, sy, sv2, color=(0, 0.5, 1), font_size=20, highlight=True)

            cx = sx
            sy += len(sv1) * 27 + 20
            adj = tuple(0.8 * anim for _ in range(3))
            mr.draw_vector_component_operation(cx, sy, sv1, sv2, "+", color=adj, font_size=20)

        elif step == 4:
            cx = mr.draw_vector_math(cx, sy, sv1, color=(1, 0.5, 0), font_size=20, highlight=True)
            cx = mr.draw_operator(cx, sy, "+", color=tc, font_size=26)
            cx = mr.draw_vector_math(cx, sy, sv2, color=(0, 0.5, 1), font_size=20, highlight=True)
            cx = mr.draw_equals_animated(cx, sy, anim, color=tc, font_size=26)
            anim_res = [r * anim for r in res] if anim < 1.0 else res
            cx = mr.draw_vector_math(cx, sy, anim_res, color=(0, 1, 0), font_size=20, highlight=True)
