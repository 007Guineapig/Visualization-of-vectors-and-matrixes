import math
import random

import numpy as np


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