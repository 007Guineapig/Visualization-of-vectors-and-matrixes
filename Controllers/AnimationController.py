import numpy as np


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