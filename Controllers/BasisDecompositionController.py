"""Rozklad vektora do bázy — vizualizácia v štýle 3Blue1Brown.

Postup animácie:
  0 — Štandardná báza e1,e2 + cieľový vektor v
  1 — ANIMOVANÁ transformácia mriežky z identity na novú bázu [b1|b2]
      (rovnaký štýl ako TransformationController — mriežka sa plynule deformuje)
  2 — Kolmice/komponenty pozdĺž nových osí (c1 na b1 osi, c2 na b2 osi)
  3 — Bázové vektory c1*b1 a c2*b2 sa posúvajú aby vytvorili v
  4 — Výsledok: v = c1*b1 + c2*b2, súradnice [c1, c2]_B
"""

import math
import numpy as np


class BasisDecompositionController:
    """Ovládač animácie rozkladu vektora do bázy"""

    def __init__(self):
        self.active = False

        # Vstupné dáta
        self.target_vector = None   # Cieľový vektor [x, y]
        self.basis_v1 = None        # Prvý bázový vektor (b1)
        self.basis_v2 = None        # Druhý bázový vektor (b2)
        self.c1 = 0.0               # Koeficient pre b1
        self.c2 = 0.0               # Koeficient pre b2

        # Bázová matica M = [b1 | b2]
        self.matrix = None           # np.array 2x2 — cieľová matica

        # Animácia
        self.current_step = 0
        self.max_steps = 4
        self.animating = False
        self.animation_progress = 0.0

        # t parameter pre mriežku — ROVNAKÝ ako TransformationController.t
        # t=0 → identita (štandardná mriežka), t=1 → nová báza
        self.t = 0.0

        # Rýchlosť animácie per-step
        self.SPEED_GRID_TRANSFORM = 0.3   # Rovnaká ako TransformationController
        self.SPEED_NORMAL = 1.2           # Pre ostatné kroky

        # Mriežka
        self.grid_step = 1.0

        # Farby — ROVNAKÉ ako TransformationController
        self.color_i = (0.85, 0.2, 0.2)   # b1 — červená (ako î)
        self.color_j = (0.2, 0.85, 0.2)   # b2 — zelená (ako ĵ)

        # Farby pre škálované a výsledky
        self.color_target = (0.9, 0.9, 0.2)       # Žltá — cieľový vektor
        self.color_scaled_b1 = (1.0, 0.5, 0.15)   # Oranžová — c1*b1
        self.color_scaled_b2 = (0.3, 0.75, 1.0)   # Svetlomodrá — c2*b2
        self.color_parallelogram = (0.5, 0.3, 0.8) # Fialová — oblasť

        # Determinant
        self.determinant = 1.0

    # ------------------------------------------------------------------
    # SETUP
    # ------------------------------------------------------------------

    def setup(self, target_vector, basis_v1, basis_v2):
        """Nastaví rozklad pre daný vektor a bázu."""
        v1 = np.array(basis_v1, dtype=float)
        v2 = np.array(basis_v2, dtype=float)
        target = np.array(target_vector, dtype=float)

        # Skontroluj lineárnu nezávislosť
        det = v1[0] * v2[1] - v1[1] * v2[0]
        if abs(det) < 1e-8:
            print("Bazove vektory su linearne zavisle!")
            return False

        # Koeficienty: target = c1*v1 + c2*v2
        B = np.column_stack([v1, v2])
        coeffs = np.linalg.solve(B, target)
        self.c1 = float(coeffs[0])
        self.c2 = float(coeffs[1])

        self.target_vector = target.tolist()
        self.basis_v1 = v1.tolist()
        self.basis_v2 = v2.tolist()
        self.matrix = B.copy()
        self.determinant = float(det)

        self.active = True
        self.current_step = 0
        self.t = 0.0
        self.animating = False
        self.animation_progress = 1.0

        print(f"Rozklad: v = {self.c1:.2f}*b1 + {self.c2:.2f}*b2")
        print(f"   b1 = {self.basis_v1}, b2 = {self.basis_v2}")
        print(f"   v  = {self.target_vector}")
        print(f"   det = {self.determinant:.2f}")
        return True

    # ------------------------------------------------------------------
    # KROKOVANIE
    # ------------------------------------------------------------------

    def next_step(self):
        if self.current_step < self.max_steps:
            self.current_step += 1
            self.animating = True
            self.animation_progress = 0.0
            return True
        return False

    def prev_step(self):
        if self.current_step > 0:
            self.current_step -= 1
            self.animating = True
            self.animation_progress = 0.0
            return True
        return False

    def _get_current_speed(self):
        """Vráti rýchlosť animácie pre aktuálny krok"""
        if self.current_step == 1:
            return self.SPEED_GRID_TRANSFORM  # Pomalá transformácia mriežky
        return self.SPEED_NORMAL

    def update(self, dt):
        if not self.animating:
            return

        speed = self._get_current_speed()
        self.animation_progress += dt * speed

        if self.animation_progress >= 1.0:
            self.animation_progress = 1.0
            self.animating = False

        # Aktualizuj t parameter pre mriežku
        # ROVNAKÁ logika ako TransformationController.update
        smooth_t = self._ease(self.animation_progress)

        if self.current_step == 0:
            # Krok 0: štandardná báza → t sa vracia k 0
            self.t = 1.0 - smooth_t
        elif self.current_step == 1:
            # Krok 1: transformácia mriežky → t: 0 -> 1
            self.t = smooth_t
        else:
            # Kroky 2-4: mriežka je plne transformovaná
            self.t = 1.0

    def _ease(self, t):
        """Smooth ease-in-out — rovnaký ako TransformationController"""
        return t * t * (3.0 - 2.0 * t)

    # ------------------------------------------------------------------
    # MRIEŽKA — ROVNAKÉ ROZHRANIE AKO TransformationController
    # ------------------------------------------------------------------

    def get_interpolated_matrix(self):
        """Vráti interpolovanú maticu medzi identitou a cieľovou bázou.

        M(t) = (1-t)*I + t*[b1|b2]

        ROVNAKÁ metóda ako TransformationController.get_interpolated_matrix()
        """
        if self.matrix is None:
            return np.eye(2)
        identity = np.eye(2)
        return (1.0 - self.t) * identity + self.t * self.matrix

    def get_basis_vectors_2d(self):
        """Vráti aktuálne interpolované bázové vektory.

        ROVNAKÁ metóda ako TransformationController.get_basis_vectors_2d()
        """
        M = self.get_interpolated_matrix()
        i_hat = M[:, 0].tolist()
        j_hat = M[:, 1].tolist()
        return i_hat, j_hat

    # ------------------------------------------------------------------
    # VEKTORY NA VYKRESLENIE
    # ------------------------------------------------------------------

    def get_vectors_to_draw(self):
        """Vráti zoznam vektorov na vykreslenie podľa aktuálneho kroku."""
        if not self.active:
            return []

        t = self._ease(self.animation_progress) if self.animating else 1.0
        v1 = self.basis_v1
        v2 = self.basis_v2
        tv = self.target_vector

        scaled_v1 = [self.c1 * v1[0], self.c1 * v1[1]]
        scaled_v2 = [self.c2 * v2[0], self.c2 * v2[1]]

        vectors = []

        # ── KROK 0: Štandardná báza e1,e2 + cieľový vektor ──
        if self.current_step == 0:
            vectors.append(self._vec(tv, [0, 0], self.color_target, 1.0,
                                     f"v = [{tv[0]:.1f}, {tv[1]:.1f}]", True))
            # Štandardné bázové vektory e1, e2
            vectors.append(self._vec([1, 0], [0, 0], self.color_i, 0.7,
                                     "e1", True))
            vectors.append(self._vec([0, 1], [0, 0], self.color_j, 0.7,
                                     "e2", True))

        # ── KROK 1: Mriežka sa transformuje (bázové vektory sú v overlay) ──
        elif self.current_step == 1:
            # Len cieľový vektor — bázové vektory kreslí overlay cez
            # get_basis_vectors_2d() + _draw_basis_vector_2d()
            vectors.append(self._vec(tv, [0, 0], self.color_target, 0.6,
                                     "v", True))

        # ── KROK 2: Kolmice — komponenty pozdĺž nových osí ──
        elif self.current_step == 2:
            vectors.append(self._vec(tv, [0, 0], self.color_target, 1.0,
                                     "v", True))
            # c1*b1 pozdĺž b1 osi (od originu)
            anim_sv1 = [scaled_v1[0] * t, scaled_v1[1] * t]
            vectors.append(self._vec(anim_sv1, [0, 0], self.color_scaled_b1, 0.9,
                                     f"{self.c1:.2f}*b1", True))
            # c2*b2 pozdĺž b2 osi (od originu)
            anim_sv2 = [scaled_v2[0] * t, scaled_v2[1] * t]
            vectors.append(self._vec(anim_sv2, [0, 0], self.color_scaled_b2, 0.9,
                                     f"{self.c2:.2f}*b2", True))

        # ── KROK 3: Posúvanie — c2*b2 sa posúva na koniec c1*b1 ──
        elif self.current_step == 3:
            vectors.append(self._vec(tv, [0, 0], self.color_target, 0.5,
                                     "v", False))
            # c1*b1 zostáva
            vectors.append(self._vec(scaled_v1, [0, 0], self.color_scaled_b1, 1.0,
                                     f"{self.c1:.2f}*b1", True))
            # c2*b2 sa posúva z originu na koniec c1*b1
            anim_offset = [scaled_v1[0] * t, scaled_v1[1] * t]
            vectors.append(self._vec(scaled_v2, anim_offset, self.color_scaled_b2, 1.0,
                                     f"{self.c2:.2f}*b2", True))

        # ── KROK 4: Výsledok ──
        elif self.current_step == 4:
            vectors.append(self._vec(tv, [0, 0], self.color_target, 1.0,
                                     f"v = {self.c1:.2f}*b1 + {self.c2:.2f}*b2", True))
            vectors.append(self._vec(scaled_v1, [0, 0], self.color_scaled_b1, 0.7,
                                     f"{self.c1:.2f}*b1", True))
            vectors.append(self._vec(scaled_v2, scaled_v1, self.color_scaled_b2, 0.7,
                                     f"{self.c2:.2f}*b2", True))
            # Opačná cesta — stlmená
            vectors.append(self._vec(scaled_v2, [0, 0], self.color_scaled_b2, 0.25,
                                     "", False))
            vectors.append(self._vec(scaled_v1, scaled_v2, self.color_scaled_b1, 0.25,
                                     "", False))

        return vectors

    def get_parallelogram_vertices(self):
        """Vráti vrcholy paralelogramu (krok 4)."""
        if not self.active or self.current_step < 4:
            return None

        sv1 = [self.c1 * self.basis_v1[0], self.c1 * self.basis_v1[1]]
        sv2 = [self.c2 * self.basis_v2[0], self.c2 * self.basis_v2[1]]

        return [
            (0, 0),
            (sv1[0], sv1[1]),
            (sv1[0] + sv2[0], sv1[1] + sv2[1]),
            (sv2[0], sv2[1])
        ]

    def get_component_lines(self):
        """Vráti čiary komponentov (kolmice na nové osi) pre krok 2+.

        Čiarkované čiary od konca c1*b1 a c2*b2 k cieľovému vektoru,
        tvoriace paralelogram.
        """
        if not self.active or self.current_step < 2:
            return []

        sv1 = [self.c1 * self.basis_v1[0], self.c1 * self.basis_v1[1]]
        sv2 = [self.c2 * self.basis_v2[0], self.c2 * self.basis_v2[1]]
        tv = self.target_vector

        alpha = 0.4
        lines = []

        # Od konca c1*b1 smerom k v (= smer c2*b2)
        lines.append(((sv1[0], sv1[1]),
                       (tv[0], tv[1]),
                       self.color_scaled_b2, alpha))
        # Od konca c2*b2 smerom k v (= smer c1*b1)
        lines.append(((sv2[0], sv2[1]),
                       (tv[0], tv[1]),
                       self.color_scaled_b1, alpha))

        return lines

    # ------------------------------------------------------------------
    # POPIS KROKOV
    # ------------------------------------------------------------------

    def get_step_description(self):
        if self.current_step == 0:
            return "Standardna baza e1, e2 a cielovy vektor v"
        elif self.current_step == 1:
            if self.animating:
                return f"Transformacia bazy... ({self.animation_progress*100:.0f}%)"
            return f"Nova baza b1, b2 (det = {self.determinant:.2f})"
        elif self.current_step == 2:
            return f"Komponenty: c1={self.c1:.2f} podla b1, c2={self.c2:.2f} podla b2"
        elif self.current_step == 3:
            if self.animating:
                return "Skladanie: c1*b1 + c2*b2 ..."
            return "c2*b2 posunuty na koniec c1*b1 = v"
        elif self.current_step == 4:
            return f"v = {self.c1:.2f}*b1 + {self.c2:.2f}*b2  ->  [{self.c1:.2f}, {self.c2:.2f}]_B"
        return ""

    def get_matrix_display(self):
        """Vráti maticu pre UI zobrazenie (rovnaká ako TransformationController)"""
        if self.matrix is None:
            return None
        return self.matrix.tolist()

    # ------------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------------

    @staticmethod
    def _vec(vec, offset, color, alpha, label, show_label):
        return {
            'vec': list(vec),
            'offset': list(offset) + [0] * (3 - len(offset)),
            'color': color,
            'alpha': alpha,
            'label': label,
            'show_label': show_label,
            'is_persistent': False
        }

    def clear(self):
        self.active = False
        self.target_vector = None
        self.basis_v1 = None
        self.basis_v2 = None
        self.matrix = None
        self.c1 = 0.0
        self.c2 = 0.0
        self.current_step = 0
        self.animating = False
        self.animation_progress = 0.0
        self.t = 0.0
        self.determinant = 1.0