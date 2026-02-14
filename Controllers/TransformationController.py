import math
import numpy as np


class TransformationController:
    """Ovládač animácie lineárnej transformácie bázy (štýl 3Blue1Brown)

    Vizualizuje ako sa mriežka a bázové vektory plynule transformujú
    keď aplikujeme maticu (novú bázu). Čiary zostávajú rovné a rovnobežné.
    """

    def __init__(self):
        self.active = False

        # Transformačná matica (2x2 alebo 3x3)
        self.matrix = None  # Cieľová matica
        self.inverse_matrix = None  # Pre spätný krok

        # Animačný stav
        self.animating = False
        self.animation_progress = 0.0
        self.animation_speed = 0.3  # Rýchlosť animácie

        # Krokovanie
        self.current_step = 0
        self.max_steps = 2  # 0=pôvodná báza, 1=transformácia, 2=hotovo

        # Interpolovaná matica (medzi identitou a cieľom)
        # t=0 -> identita, t=1 -> cieľová matica
        self.t = 0.0

        # Nastavenia mriežky
        self.grid_range = 10  # Rozsah mriežky
        self.grid_step = 1.0  # Krok mriežky

        # Determinant pre vizuálnu indikáciu
        self.determinant = 1.0

        # Pre 2D/3D
        self.is_2d = True

        # Farby bázových vektorov
        self.color_j = (0.2, 0.85, 0.2)   # j - zelená
        self.color_i = (0.85, 0.2, 0.2)   # i - červená
        self.color_k = (0.2, 0.2, 0.85)   # k̂ - modrá (3D)

    def setup_transformation(self, matrix, is_2d=True):
        """Nastaví transformáciu pre danú maticu

        Args:
            matrix: 2x2 alebo 3x3 matica (zoznam zoznamov alebo tuple)
            is_2d: True pre 2D, False pre 3D
        """
        self.is_2d = is_2d
        self.matrix = np.array(matrix, dtype=float)

        if is_2d:
            if self.matrix.shape != (2, 2):
                print("Chyba: Matica musí byť 2x2 pre 2D transformáciu")
                return False
            self.determinant = np.linalg.det(self.matrix)
        else:
            if self.matrix.shape != (3, 3):
                print("Chyba: Matica musí byť 3x3 pre 3D transformáciu")
                return False
            self.determinant = np.linalg.det(self.matrix)

        self.active = True
        self.current_step = 0
        self.t = 0.0
        self.animating = False
        self.animation_progress = 1.0

        print(f"Transformácia nastavená: det = {self.determinant:.2f}")
        print(f"Matica:\n{self.matrix}")
        return True

    def next_step(self):
        """Prejdi na ďalší krok animácie"""
        if self.current_step < self.max_steps:
            self.current_step += 1
            self.animating = True
            self.animation_progress = 0.0
            return True
        return False

    def prev_step(self):
        """Vráť sa na predošlý krok"""
        if self.current_step > 0:
            self.current_step -= 1
            self.animating = True
            self.animation_progress = 0.0
            return True
        return False

    def update(self, dt):
        """Aktualizuj animáciu"""
        if not self.animating:
            return

        self.animation_progress += dt * self.animation_speed
        if self.animation_progress >= 1.0:
            self.animation_progress = 1.0
            self.animating = False

        # Smooth easing
        smooth_t = self._ease_in_out(self.animation_progress)

        # Aktualizuj interpolačný parameter t podľa kroku
        if self.current_step == 0:
            # Krok 0: pôvodná báza (t -> 0)
            self.t = 1.0 - smooth_t  # Ak sme sa vrátili z kroku 1
        elif self.current_step == 1:
            # Krok 1: animácia transformácie (t: 0 -> 1)
            self.t = smooth_t
        elif self.current_step == 2:
            # Krok 2: hotovo (t = 1)
            self.t = 1.0

    def _ease_in_out(self, t):
        """Smooth ease-in-out interpolácia"""
        return t * t * (3.0 - 2.0 * t)

    def get_interpolated_matrix(self):
        """Vráti interpolovanú maticu medzi identitou a cieľom

        Returns:
            Numpy matica interpolovaná podľa self.t
        """
        if self.matrix is None:
            if self.is_2d:
                return np.eye(2)
            else:
                return np.eye(3)

        size = 2 if self.is_2d else 3
        identity = np.eye(size)

        # Lineárna interpolácia: M(t) = (1-t)*I + t*Target
        return (1.0 - self.t) * identity + self.t * self.matrix

    def get_current_t(self):
        """Vráti aktuálny interpolačný parameter"""
        if not self.animating:
            if self.current_step == 0:
                return 0.0
            elif self.current_step >= 1:
                return 1.0
        return self.t

    def transform_point_2d(self, x, y):
        """Transformuj bod podľa aktuálnej interpolovanej matice"""
        M = self.get_interpolated_matrix()
        point = np.array([x, y])
        result = M @ point
        return result[0], result[1]

    def transform_point_3d(self, x, y, z):
        """Transformuj 3D bod"""
        M = self.get_interpolated_matrix()
        point = np.array([x, y, z])
        result = M @ point
        return result[0], result[1], result[2]

    def get_basis_vectors_2d(self):
        """Vráti aktuálne interpolované bázové vektory pre 2D

        Returns:
            (i_hat, j_hat) - tuple dvoch vektorov [x, y]
        """
        M = self.get_interpolated_matrix()
        i_hat = M[:, 0].tolist()  # Prvý stĺpec
        j_hat = M[:, 1].tolist()  # Druhý stĺpec
        return i_hat, j_hat

    def get_basis_vectors_3d(self):
        """Vráti aktuálne bázové vektory pre 3D"""
        M = self.get_interpolated_matrix()
        i_hat = M[:, 0].tolist()
        j_hat = M[:, 1].tolist()
        k_hat = M[:, 2].tolist()
        return i_hat, j_hat, k_hat

    def get_grid_lines_2d(self, ortho_scale=None, pan_x=0, pan_y=0, aspect=1.0):
        """Vráti transformované čiary mriežky pre 2D vykresľovanie

        Dynamicky počíta rozsah podľa viditeľnej oblasti (ortho_scale),
        takže grid je vždy "nekonečný" — pokrýva celú obrazovku.

        Returns:
            list of ((x1,y1), (x2,y2)) - páry bodov pre GL_LINES
        """
        M = self.get_interpolated_matrix()

        # Dynamický rozsah podľa zoomu
        if ortho_scale is not None:
            # Koľko priestoru je viditeľného
            visible = ortho_scale * max(aspect, 1.0) + max(abs(pan_x), abs(pan_y))

            # Ak je matica deformovaná, transformácia môže zmenšiť vektory
            # Potrebujeme väčší rozsah aby sme pokryli celú obrazovku
            try:
                inv_M = np.linalg.inv(M)
                # Maximálne zväčšenie inverznou maticou
                scale_factor = max(np.linalg.norm(inv_M[:, 0]), np.linalg.norm(inv_M[:, 1]))
            except np.linalg.LinAlgError:
                scale_factor = 10.0

            # Rozsah = viditeľná oblasť * scale_factor + bezpečnostná rezerva
            r = int(math.ceil(visible * scale_factor)) + 5
            r = max(r, 15)  # Minimálne 15
        else:
            r = self.grid_range

        lines = []
        step = self.grid_step

        # Dynamický step - ak je príliš veľa čiar, zväčši krok
        max_lines = 200
        display_step = step
        while (2 * r / display_step) > max_lines:
            display_step *= 2

        # Vertikálne čiary (konštantné x)
        i = -r
        while i <= r:
            x = float(i) * display_step
            p1 = M @ np.array([x, -r * display_step])
            p2 = M @ np.array([x, r * display_step])
            lines.append(((p1[0], p1[1]), (p2[0], p2[1])))
            i += 1

        # Horizontálne čiary (konštantné y)
        i = -r
        while i <= r:
            y = float(i) * display_step
            p1 = M @ np.array([-r * display_step, y])
            p2 = M @ np.array([r * display_step, y])
            lines.append(((p1[0], p1[1]), (p2[0], p2[1])))
            i += 1

        return lines

    def get_step_description(self):
        """Vráti popis aktuálneho kroku"""
        if self.current_step == 0:
            return "Pôvodná báza (identita) - î a ĵ"
        elif self.current_step == 1:
            if self.animating:
                return f"Transformácia bázy... ({self.animation_progress*100:.0f}%)"
            else:
                return f"Transformovaná báza (det = {self.determinant:.2f})"
        elif self.current_step == 2:
            return f"Transformácia dokončená (det = {self.determinant:.2f})"
        return ""

    def get_matrix_display(self):
        """Vráti maticu pre zobrazenie v UI"""
        if self.matrix is None:
            return None
        return self.matrix.tolist()

    def clear(self):
        """Vyčisti transformáciu"""
        self.active = False
        self.matrix = None
        self.inverse_matrix = None
        self.animating = False
        self.animation_progress = 0.0
        self.current_step = 0
        self.max_steps = 2
        self.t = 0.0
        self.determinant = 1.0