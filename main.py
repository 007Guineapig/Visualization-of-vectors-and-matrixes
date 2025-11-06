
import math
import pygame
import numpy as np
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from numpy import array, cross, linalg
from math import sin, cos, pi
import ast
import random
from utils import *
from vectorAnimations import *
# --- Settings ---
WIDTH, HEIGHT = 900, 700
FPS = 60
LENGTH_XYZ = 10.

# --- Input / Vectors ---
input_text = ""
show_input_active = False
multiplication_input = ""
show_multiplication_active = False
vectors_to_draw = []

dropdown_open = False
dropdown_rect = pygame.Rect(10, 10, 150, 30)
dropdown_item_height = 25
selected_vector_index = -1  # -1 = none selected

def format_number(x):
    """Format number nicely — remove .0 when not needed"""
    if float(x).is_integer():
        return str(int(x))
    else:
        return f"{x:.2f}".rstrip('0').rstrip('.')

def format_vector(vec):
    """Format vectors and matrices cleanly for display"""
    if isinstance(vec[0], (int, float)):
        # Simple vector like (1,2,3)
        return "(" + ", ".join(format_number(x) for x in vec) + ")"
    else:
        # Matrix like ((1,2,3),(4,5,6))
        return "(" + ", ".join("(" + ", ".join(format_number(x) for x in row) + ")" for row in vec) + ")"


def compute_radius(start, end, base_max_radius, min_radius, camera_pos, reference_distance):
    # 1) ŽIADNY tapering – všetky segmenty (X, Y, Z) začínajú rovnako hrubé
    radius = float(base_max_radius)

    # 2) Zachovaj tvoje overené zoom-škálovanie: cam_dist/reference_distance
    if camera_pos and reference_distance and reference_distance > 0:
        mid = ((start[0] + end[0]) * 0.5,
               (start[1] + end[1]) * 0.5,
               (start[2] + end[2]) * 0.5)
        cam_dist = math.sqrt((camera_pos[0] - mid[0])**2 +
                             (camera_pos[1] - mid[1])**2 +
                             (camera_pos[2] - mid[2])**2)
        cam_dist = max(cam_dist, 1e-6)

        # DÔLEŽITÉ: rovnaký smer ako v tvojej verzii (blízko -> tenšie)
        scale = cam_dist / reference_distance

        # Jemný clamp, aby extrémy zoomu nevystrelili/nezmizli (prispôsob si)
        scale = max(0.5, min(2.0, scale))

        radius *= scale

    # 3) Dolný limit
    return max(radius, float(min_radius))
# Helper function for radius
####DONT DELETE THIS BELOW###
def compute_radius1(start, end, base_max_radius, min_radius, camera_pos, reference_distance):
    dist = sum((c)**2 for c in end)**0.5
    full_length = sum((ee - ss)**2 for ss, ee in zip(start, end))**0.5
    if full_length == 0:
        radius = base_max_radius
    else:
        radius = base_max_radius*(1 - dist/full_length) + min_radius
    if camera_pos:
        mid = [(s+e)/2 for s,e in zip(start,end)]
        cam_dist = sum((camera_pos[i]-mid[i])**2 for i in range(3))**0.5
        radius *= cam_dist/reference_distance
    return max(radius,min_radius)




def reset_vector_animation(vectors):
    for v in vectors:
        # single vector
        v['progress'] = 0.0
        if 'segments' in v:
            v['segments'].clear()

        # matrix of vectors
        if 'row_progress' in v:
            # nastav nuly pre každú riadkovú animáciu
            v['row_progress'] = [0.0] * len(v['row_progress'])
        if 'row_segments' in v:
            # vyčisti každý zoznam segmentov v riadkoch
            for lst in v['row_segments']:
                lst.clear()


def reset_vector_animation1(vectors):
    for v in vectors:
        v['progress'] = 0.0
        if 'row_progress' in v:
            v['row_progress'] = [0.0] * len(v['row_progress'])

def draw_vectors_2d(vectors, line_width=6, arrow_fraction=0.15):
    glPushAttrib(GL_LINE_BIT | GL_CURRENT_BIT)
    glLineWidth(line_width)
    for v in vectors:
        if len(v) >= 2:
            x, y = v[0], v[1]
            length = math.hypot(x, y)
            arrow_size = max(length * arrow_fraction, 0.1)  # minimum size

            # Draw the line
            glColor3f(1, 0, 1)
            glBegin(GL_LINES)
            glVertex2f(0, 0)
            glVertex2f(x, y)
            glEnd()

            # Draw the arrowhead
            angle = math.atan2(y, x)
            glBegin(GL_TRIANGLES)
            glVertex2f(x, y)
            glVertex2f(x - arrow_size * math.cos(angle - 0.3),
                       y - arrow_size * math.sin(angle - 0.3))
            glVertex2f(x - arrow_size * math.cos(angle + 0.3),
                       y - arrow_size * math.sin(angle + 0.3))
            glEnd()
    glPopAttrib()

def max_from_vectors(vectors):
    """
    Vráti najväčšiu hodnotu zo všetkých vektorov/matrixov.
    Vectors: list, kde každý prvok môže byť:
        - vektor: [x, y] alebo [x, y, z]
        - matica: [[x1, y1], [x2, y2], ...]
    """
    max_val = None

    for v in vectors:
        vec = v['vec'] if isinstance(v, dict) and 'vec' in v else v

        if isinstance(vec[0], (int, float)):
            # Single vector
            current_max = max(vec)
        else:
            # Matrix
            current_max = max(max(row) for row in vec)

        if max_val is None or current_max > max_val:
            max_val = current_max

    return max_val
# Add this at the top of your imports

def draw_parenthesis(x, y, height, left=True, color=(1, 1, 1), thickness=3):
    """Draws a smooth, LaTeX-style parenthesis using a cubic Bézier curve."""

    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    glOrtho(0, WIDTH, HEIGHT, 0, -1, 1)
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()

    glDisable(GL_DEPTH_TEST)
    glLineWidth(thickness)
    glColor3f(*color)

    # Control points
    radius = height * 0.15  # horizontal "roundness"
    top_y = y
    bottom_y = y + height
    mid_y = (top_y + bottom_y) / 2

    # Bézier control points (approximation of ⎛⎜⎝)
    if left:
        p0 = (x, bottom_y)  # bottom start
        p1 = (x - radius, bottom_y - height * 0.15)  # bottom curve control
        p2 = (x - radius, top_y + height * 0.15)  # top curve control
        p3 = (x, top_y)  # top end
    else:
        p0 = (x, bottom_y)
        p1 = (x + radius, bottom_y - height * 0.15)
        p2 = (x + radius, top_y + height * 0.15)
        p3 = (x, top_y)

    # Draw the cubic Bézier curve
    glBegin(GL_LINE_STRIP)
    segments = 50
    for i in range(segments + 1):
        t = i / segments
        # Cubic Bézier interpolation
        vx = (
            (1 - t) ** 3 * p0[0]
            + 3 * (1 - t) ** 2 * t * p1[0]
            + 3 * (1 - t) * t ** 2 * p2[0]
            + t ** 3 * p3[0]
        )
        vy = (
            (1 - t) ** 3 * p0[1]
            + 3 * (1 - t) ** 2 * t * p1[1]
            + 3 * (1 - t) * t ** 2 * p2[1]
            + t ** 3 * p3[1]
        )
        glVertex2f(vx, vy)
    glEnd()

    glEnable(GL_DEPTH_TEST)
    glPopMatrix()
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)


def main():


    matrix_size_input = ""
    random_range_input = ""
    show_matrix_size_active = False
    show_random_range_active = False

    matrix_size_rect = pygame.Rect(WIDTH - 210, 240, 60, 30)   # left of matrix input
    random_range_rect = pygame.Rect(WIDTH - 140, 240, 60, 30)  # next to size input
    random_button_rect = pygame.Rect(WIDTH - 70, 240, 60, 30) # next to random range inpu



    global input_text, show_input_active
    multiplication_input = ""  # stores the text typed in multiplication input
    show_multiplication_active = False  # tracks if multiplication input is active
    background_dark = True
    multiplication_rect = pygame.Rect(WIDTH - 160, 120, 140, 30)  # position of new input box

    global input_text, show_input_active, selected_vector_index  # <--- add this
    pending_vector = None
    vectors_as_points = True


    matrix_inputs = [["" for _ in range(3)] for _ in range(3)]
    matrix_active_cell = (-1, -1)  # (row, col)
    matrix_cell_w, matrix_cell_h = 40, 30
    matrix_gap = 5
    matrix_start_x, matrix_start_y = WIDTH - 155, 130  # adjust position
    matrix_start_xx, matrix_start_yy = WIDTH - 155, 300  # adjust position
    show_matrix_input = True  # toggle with a key (M)

    def get_matrix_values():
        try:
            return [[float(cell) if cell else 0.0 for cell in row] for row in matrix_inputs]
        except ValueError:
            print("Invalid matrix entry")
            return None

    # --- Dropdown variables ---
    #dropdown_rect = pygame.Rect(WIDTH - 160, 160, 140, 30)
    dropdown_open = False
    if 'selected_vector_index' not in globals():
        selected_vector_index = None

    pygame.init()

    # --- Enable hardware multisampling before creating the window ---
    pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLEBUFFERS, 1)
    pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLESAMPLES, 4)

    # --- Create the OpenGL window ---
    screen = pygame.display.set_mode((WIDTH, HEIGHT), DOUBLEBUF | OPENGL)
    pygame.display.set_caption("XYZ axes — mouse control (Antialiasing ON)")

    # --- Enable depth and smoothing ---
    glEnable(GL_DEPTH_TEST)
    # --- Antialiasing setup ---
    glEnable(GL_MULTISAMPLE)  # enable MSAA (multisample anti-aliasing)
    glEnable(GL_BLEND)  # enable blending for smooth edges
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glEnable(GL_LINE_SMOOTH)  # smooth lines
    glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
    glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST)

    glClearColor(0.0, 0.0, 0.0, 1.0)
    #glClearColor(1.0, 1.0, 1.0, 1.0)

    distance, azimuth, elevation = 7.0, 45.0, 25.0
    target = [0.0, 0.0, 0.0]
    pan_offset_x, pan_offset_y = 0.0, 0.0
    ortho_scale = 6.0

    rotating = panning = False
    last_mouse = (0, 0)
    yaw_speed, pitch_speed = 0.3, 0.3
    pan_speed, zoom_speed = 0.01, 1.5

    clock = pygame.time.Clock()
    view_2d_mode = True
    button_rect = (WIDTH - 160, 30, 140, 35)
    draw_button_rect = (WIDTH - 160, 75, 140, 35)

    grid_mode = 0  # 0 = none, 1 = planes, 2 = grid
    show_axes = True
    running = True
    animated_vectors = []

    while running:

        dt = clock.tick(FPS) / 1000.0

        def snap_number(x, precision=6):
            """Round to nearest number; convert to int if whole, else keep float."""
            x = round(x, precision)
            if x.is_integer():
                return int(x)
            return x

        for v in animated_vectors:
            if 'frames_left' in v and v['frames_left'] > 0:
                if isinstance(v['vec'][0], (list, tuple)):  # matrix of vectors
                    v['vec'] = [
                        [c + d for c, d in zip(list(curr), delt)]
                        for curr, delt in zip(v['vec'], v['delta'])
                    ]
                else:  # single vector
                    v['vec'] = [c + d for c, d in zip(list(v['vec']), v['delta'])]

                v['frames_left'] -= 1

                # Snap to clean numbers when animation finishes
                if v['frames_left'] == 0:
                    if isinstance(v['vec'][0], (list, tuple)):  # matrix
                        v['vec'] = [[snap_number(c) for c in row] for row in v['vec']]
                    else:  # single vector
                        v['vec'] = [snap_number(c) for c in v['vec']]

        if len(animated_vectors) == 0:
            LENGTH_XYZ = 10
        else:
            temp_hihest = max_from_vectors(animated_vectors)
            LENGTH_XYZ = max(10,temp_hihest)

        mx, my = pygame.mouse.get_pos()


        # --- EVENTS ---
        for event in pygame.event.get():
            if show_matrix_input:
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if matrix_size_rect.collidepoint(event.pos):
                        show_matrix_size_active = True
                        show_random_range_active = False
                    elif random_range_rect.collidepoint(event.pos):
                        show_random_range_active = True
                        show_matrix_size_active = False
                    elif random_button_rect.collidepoint(event.pos):
                        # Generate random matrix
                        try:
                            rows, cols = map(int, matrix_size_input.split(','))
                            if not (1 <= rows <= 3 and 1 <= cols <= 3):
                                print("Only up to 3x3 supported")
                            else:
                                low, high = (0, 9)
                                if random_range_input:
                                    parts = random_range_input.split(',')
                                    if len(parts) == 2:
                                        low, high = map(float, parts)

                                matrix_inputs = [[str(random.randint(int(low), int(high))) for _ in range(cols)]
                                                 for _ in range(rows)]
                                # fill empty cells if smaller than 3×3
                                for r in range(3 - len(matrix_inputs)):
                                    matrix_inputs.append([""] * 3)
                                for r in range(len(matrix_inputs)):
                                    while len(matrix_inputs[r]) < 3:
                                        matrix_inputs[r].append("")
                                print(f"Generated random {rows}x{cols} matrix.")
                        except Exception as e:
                            print("Invalid size or range:", e)
                    else:
                        show_matrix_size_active = show_random_range_active = False

                elif event.type == pygame.KEYDOWN:

                    if matrix_active_cell != (-1, -1) and not (show_matrix_size_active or show_random_range_active):
                        r, c = matrix_active_cell
                        if event.key == pygame.K_RETURN:
                            matrix_active_cell = (-1, -1)  # finish editing
                        elif event.key == pygame.K_BACKSPACE:
                            matrix_inputs[r][c] = matrix_inputs[r][c][:-1]
                        elif event.unicode.isdigit() or event.unicode in ".-":
                            matrix_inputs[r][c] += event.unicode


                    if show_matrix_size_active:
                        if event.key == pygame.K_RETURN:
                            show_matrix_size_active = False
                        elif event.key == pygame.K_BACKSPACE:
                            matrix_size_input = matrix_size_input[:-1]
                        elif event.unicode.isdigit() or event.unicode == ',':
                            matrix_size_input += event.unicode
                    elif show_random_range_active:
                        if event.key == pygame.K_RETURN:
                            show_random_range_active = False
                        elif event.key == pygame.K_BACKSPACE:
                            random_range_input = random_range_input[:-1]
                        elif event.unicode.isdigit() or event.unicode in ",.-":
                            random_range_input += event.unicode
                    elif event.key == pygame.K_RETURN and show_matrix_input:
                        try:
                            # Convert filled cells to floats
                            #matrix_values = []
                            #row_lengths = set()
                            #for row in matrix_inputs:
                                # Count only filled cells
                                #filled_cells = [cell for cell in row if cell.strip()]
                                #if not filled_cells:
                            #    continue  # skip entirely empty rows
                                #float_row = [float(cell) for cell in filled_cells]
                                #matrix_values.append(tuple(float_row))
                            # row_lengths.add(len(float_row))

                            cols = len(matrix_inputs[0])
                            rows = len(matrix_inputs)
                            matrix_values = []
                            col_lengths = set()
                            for c in range(cols):
                                column_cells = [matrix_inputs[r][c] for r in range(rows) if matrix_inputs[r][c].strip()]
                                if not column_cells:
                                    continue  # skip empty column
                                float_col = [float(cell) for cell in column_cells]
                                matrix_values.append(tuple(float_col))
                                col_lengths.add(len(float_col))

                            if not matrix_values:
                                raise ValueError("Matrix cannot be empty")

                            # All rows must have the same number of elements
                            #if len(row_lengths) != 1:
                            #    raise ValueError("All rows must have the same number of elements")
                            if len(col_lengths) != 1:
                                raise ValueError("All columns must have the same number of filled cells")

                            # Store as tuple-of-tuples
                            matrix_values = tuple(matrix_values)
                            animated_vectors.append({'vec': matrix_values, 'progress': 0.0})

                            # Reset input
                            matrix_inputs = [["" for _ in range(3)] for _ in range(3)]
                            matrix_active_cell = (-1, -1)
                            show_matrix_input = True

                        except Exception as e:
                            print("Invalid matrix input:", e)

            # Toggle matrix input visibility
            if event.type == pygame.KEYDOWN and event.key == pygame.K_m:
                show_matrix_input = not show_matrix_input
            if event.type == QUIT:
                running = False
            elif event.type == KEYDOWN:

                if event.key == K_ESCAPE:
                    running = False
                elif show_multiplication_active:
                    if event.key == K_RETURN and selected_vector_index is not None:
                        try:
                            val = ast.literal_eval(multiplication_input)
                            selected = animated_vectors[selected_vector_index]

                            selected_np = np.array(selected['vec'])
                            val_np = np.array(val)

                            # Determine multiplication type
                            if selected_np.ndim == 1 and val_np.ndim == 1:
                                result = np.dot(selected_np, val_np)
                            else:
                                result = selected_np @ val_np

                            # Animate existing vector instead of creating new
                            if isinstance(result, np.ndarray):
                                frames = 30  # number of frames for smooth movement

                                # Single vector
                                if result.ndim == 1:
                                    target = result.tolist()
                                    current = selected['vec']
                                    selected['delta'] = [(t - c) / frames for c, t in zip(current, target)]
                                    selected['frames_left'] = frames

                                # Matrix of vectors
                                else:
                                    target_vectors = result.tolist()
                                    current_vectors = selected['vec']
                                    selected['delta'] = [
                                        [(t - c) / frames for c, t in zip(curr_vec, targ_vec)]
                                        for curr_vec, targ_vec in zip(current_vectors, target_vectors)
                                    ]
                                    selected['frames_left'] = frames
                        except Exception as e:
                            print("Multiplication failed:", e)
                            multiplication_input = ""
                            show_multiplication_active = False
                    elif event.key == K_BACKSPACE:
                        multiplication_input = multiplication_input[:-1]
                    else:
                        multiplication_input += event.unicode
                elif event.key == K_g:
                    grid_mode = (grid_mode + 1) % 4
                elif event.key == K_a:
                    show_axes = not show_axes
                elif event.key == K_r:
                    distance, azimuth, elevation = 7.0, 45.0, 25.0
                    target = [0.0, 0.0, 0.0]
                    pan_offset_x, pan_offset_y, ortho_scale = 0.0, 0.0, 6.0
                elif show_input_active:
                    if event.key == K_RETURN:
                        try:
                            val = ast.literal_eval(input_text)
                            # Instead of splitting, store the whole matrix if it's 2D/3D
                            if isinstance(val, (list, tuple)):
                                # Check if it's a matrix (tuple/list of tuples/lists)
                                if all(isinstance(row, (list, tuple)) and len(row) in (2, 3) for row in val):
                                    # Store entire matrix as one entry
                                    animated_vectors.append({'vec': val, 'progress': 0.0})
                                # Single vector case
                                elif len(val) in (2, 3) and all(isinstance(c, (int, float)) for c in val):
                                    animated_vectors.append({'vec': val, 'progress': 0.0})
                            input_text = ""
                            show_input_active = False
                        except Exception as e:
                            print("Invalid input:", e)
                            input_text = ""
                            show_input_active = False
                    elif event.key == K_BACKSPACE:
                        input_text = input_text[:-1]
                    else:
                        input_text += event.unicode
                elif event.key == K_i:
                    show_input_active = True

            # --- MOUSE ---
            elif event.type == MOUSEBUTTONDOWN:
                if show_matrix_input:
                    for r in range(3):
                        for c in range(3):
                            x = matrix_start_x + c * (matrix_cell_w + matrix_gap)
                            y = matrix_start_y + r * (matrix_cell_h + matrix_gap)
                            rect = pygame.Rect(x, y, matrix_cell_w, matrix_cell_h)
                            if rect.collidepoint(mx, my):
                                matrix_active_cell = (r, c)
                                break
                if toggle_bg_rect.collidepoint(mx, my):
                    background_dark = not background_dark
                    if background_dark:
                        glClearColor(0.0, 0.0, 0.0, 1.0)
                    else:
                        glClearColor(1.0, 1.0, 1.0, 1.0)
                if multiplication_rect.collidepoint(mx, my):
                    show_multiplication_active = True
                    show_input_active = False  # optionally deactivate the other input
                else:
                    show_multiplication_active = False
                if dropdown_rect.collidepoint(mx, my):
                    dropdown_open = not dropdown_open
                elif dropdown_open:
                    # Check each item
                    for i in range(len(animated_vectors)):
                        item_rect = pygame.Rect(dropdown_rect.x,
                                                dropdown_rect.y + dropdown_rect.height + i * dropdown_item_height,
                                                dropdown_rect.width,
                                                dropdown_item_height)
                        if item_rect.collidepoint(mx, my):
                            selected_vector_index = i
                            dropdown_open = False
                            break
                    else:
                        dropdown_open = False
                bx, by, bw, bh = button_rect
                input_rect = pygame.Rect(WIDTH - 160, 75, 140, 30)
                if bx <= mx <= bx + bw and by <= my <= by + bh:
                    view_2d_mode = not view_2d_mode
                elif draw_button_rect[0] <= mx <= draw_button_rect[0] + draw_button_rect[2] and \
                        draw_button_rect[1] <= my <= draw_button_rect[1] + draw_button_rect[3]:
                    if pending_vector:
                        animated_vectors.append({'vec': pending_vector, 'progress': 0.0})
                        pending_vector = None
                    else:
                        vectors_as_points = not vectors_as_points
                        if not vectors_as_points:
                            reset_vector_animation(animated_vectors)
                elif input_rect.collidepoint(mx, my):
                    show_input_active = True
                else:
                    if event.button == 1:
                        rotating = not view_2d_mode
                        panning = view_2d_mode
                        last_mouse = event.pos
                    elif event.button == 3:
                        panning = True
                        last_mouse = event.pos
                    elif event.button == 4:  # scroll up
                        if view_2d_mode:
                            ortho_scale /= 1.1
                        else:
                            distance = max(1.0, distance - zoom_speed)
                    elif event.button == 5:  # scroll down
                        if view_2d_mode:
                            ortho_scale *= 1.1
                        else:
                            distance += zoom_speed

            elif event.type == MOUSEBUTTONUP:
                if event.button in (1, 3):
                    rotating = panning = False
            elif event.type == MOUSEMOTION:
                dx, dy = event.pos[0] - last_mouse[0], event.pos[1] - last_mouse[1]
                last_mouse = event.pos
                if view_2d_mode and panning:
                    pan_offset_x -= dx * 0.002 * ortho_scale
                    pan_offset_y += dy * 0.002 * ortho_scale
                elif not view_2d_mode:
                    if rotating:
                        azimuth -= dx * yaw_speed  # ← invert horizontal rotation
                        elevation = max(-89.9, min(89.9, elevation + dy * pitch_speed))
                    if panning:
                        rad_az, rad_el = math.radians(azimuth), math.radians(elevation)
                        fx = math.cos(rad_el) * math.sin(rad_az)
                        fy = math.sin(rad_el)
                        fz = math.cos(rad_el) * math.cos(rad_az)
                        rx, ry, rz = fz, 0.0, -fx
                        ux = ry * fz - rz * fy
                        uy = rz * fx - rx * fz
                        uz = rx * fy - ry * fx
                        target[0] -= (rx * dx + ux * dy) * pan_speed
                        target[1] -= (ry * dx + uy * dy) * pan_speed
                        target[2] -= (rz * dx + uz * dy) * pan_speed

        # --- DRAW ---
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        all_vectors = animated_vectors.copy()
        if pending_vector:
            all_vectors.append({'vec': pending_vector, 'progress': 0.0})

        if view_2d_mode:
            glMatrixMode(GL_PROJECTION)
            glLoadIdentity()
            glOrtho(-ortho_scale + pan_offset_x,
                    ortho_scale + pan_offset_x,
                    -ortho_scale + pan_offset_y,
                    ortho_scale + pan_offset_y,
                    -1, 1)
            glMatrixMode(GL_MODELVIEW)
            glLoadIdentity()
            draw_grid_2d.ortho = ortho_scale
            draw_grid_2d.pan = (pan_offset_x, pan_offset_y)
            if grid_mode % 2 == 0:
                draw_grid_2d(step=1.0, z=-0.1)
            if show_axes:
                draw_axes_2d()

            if not background_dark:  # Light mode → always black
                sphere_color = (0, 0, 0)
            else:  # Dark mode
                sphere_color = (1, 1, 1)

            if vectors_as_points:

                for v in all_vectors:
                    vec = v['vec']
                    base_radius = 0.07
                    radius = base_radius * ortho_scale / 6.5
                    # Single vector
                    if isinstance(vec[0], (int, float)):
                        draw_circle_2d([vec[0], vec[1], 0.2], radius=radius, color=sphere_color)

                    else:
                        for row in vec:
                            draw_circle_2d([row[0], row[1], 0.2], radius=radius, color=sphere_color)
            else:
                draw_vectors_2d_animated(all_vectors, dt, ortho_scale=ortho_scale, color = sphere_color)

        else:
            rad_az, rad_el = math.radians(azimuth), math.radians(elevation)
            cam_x = target[0] + distance * math.cos(rad_el) * math.sin(rad_az)
            cam_y = target[1] + distance * math.sin(rad_el)
            cam_z = target[2] + distance * math.cos(rad_el) * math.cos(rad_az)

            glMatrixMode(GL_PROJECTION)
            glLoadIdentity()
            gluPerspective(45, (WIDTH / HEIGHT), 0.1, 500.0)

            glMatrixMode(GL_MODELVIEW)
            glLoadIdentity()
            gluLookAt(cam_x, cam_y, cam_z, target[0], target[1], target[2], 0, 1, 0)

            if grid_mode == 1:
                draw_planes_3D(LENGTH_XYZ, camera_pos = camera_pos)
            elif grid_mode == 2:
                draw_planes_3D(LENGTH_XYZ,colored = True, camera_pos = camera_pos)
            elif grid_mode == 3:
                draw_grid_3D(LENGTH_XYZ)

            if show_axes:
                camera_pos = (cam_x, cam_y, cam_z)
                draw_axes_3D(LENGTH_XYZ,camera_pos)

            if not background_dark:  # Light mode → always black
                sphere_color = (0, 0, 0)
            else:  # Dark mode
                if grid_mode in (1, 2):
                    sphere_color = (0, 0, 0)
                else:
                    sphere_color = (1, 1, 1)

            if vectors_as_points:

                for v in all_vectors:
                    vec = v['vec']
                    base_radius = 0.07
                    radius = base_radius * math.sqrt(distance / 7.0)
                    radius = max(radius, 0.03)  # minimum size

                    # Single vector
                    if isinstance(vec[0], (int, float)):
                        vec3d = list(vec) + [0] * (3 - len(vec))
                        draw_sphere(vec3d, radius=radius, color=sphere_color)
                    # Matrix
                    else:
                        for row in vec:
                            vec3d = list(row) + [0] * (3 - len(row))
                            draw_sphere(vec3d, radius=radius, color=sphere_color)

            else:
                camera_pos = (cam_x, cam_y, cam_z)
                draw_vectors_3d_animated(all_vectors, dt, camera_pos=camera_pos,color = sphere_color)



        #if show_matrix_input:



        total_height = 100
        total_width = 130

        if  background_dark:
            color = (1, 1, 1)
        else:
            color = (0, 0, 0)
        draw_parenthesis(matrix_start_x - 10, matrix_start_y - 5, total_height + 10, left=True, thickness=4,color = color)
        draw_parenthesis(matrix_start_x + total_width + 10, matrix_start_y - 5, total_height + 10, left=False,color = color,
                         thickness=4)


        for r in range(3):
                for c in range(3):
                    x = matrix_start_x + c * (matrix_cell_w + matrix_gap)
                    y = matrix_start_y + r * (matrix_cell_h + matrix_gap)
                    active = (r, c) == matrix_active_cell
                    # assign color per column

                    if c == 0:
                        col_color_outline = (1.0, 0.0, 1.0)
                    elif c == 1:
                        col_color_outline = (0.0, 1.0, 1.0)
                    elif c == 2:
                        col_color_outline = (1.0, 1.0, 0.0)

                    if r == 0:
                        col_color = (0.65, 0.25, 0.25)  # jemne svetlejšia červenohnedá
                    elif r == 1:
                        col_color = (0.2, 0.55, 0.2)  # tlmená, ale sviežejšia olivovozelená
                    elif r == 2:
                        col_color = (0.2, 0.4, 0.6)  # svetlejšia modrosivá

                    # if r == 0:
                    #     col_color_outline = (1,0,0)#(0.65, 0.25, 0.25)
                    # elif r == 1:

                    #     col_color_outline = (0,1,0)#(0.2, 0.55, 0.2)
                    #  elif r == 2:

                    #      col_color_outline = (0,0,1)#(0.2, 0.4, 0.6)

                    # if c == 0:
                    #     col_color = (1, 0.0, 1.0)  # jemne svetlejšia červenohnedá
                    # if c == 1:
                    #     col_color = (0.0, 1.0, 1.0)  # tlmená, ale sviežejšia olivovozelená
                    # if c == 2:
                    #     col_color =(1.0, 1.0, 0.0)  # svetlejšia modrosivá
                    draw_input_box_3d(x, y, matrix_cell_w, matrix_cell_h, matrix_inputs[r][c], active,fill_color = col_color,fill_color_outline = col_color_outline)

        draw_dropdown(animated_vectors, selected_vector_index, dropdown_rect, dropdown_open)
        draw_button_2d(*button_rect, "Switch 2D" if not view_2d_mode else "Switch 3D", active=view_2d_mode)
        # draw_button_rect = (WIDTH - 160, 120, 140, 35)
        draw_button_2d(*draw_button_rect, "Draw Vector", active=False)
        #input_rect = pygame.Rect(WIDTH - 160, 75, 140, 30)
        #draw_input_box_3d(*input_rect, input_text, active=show_input_active)
        toggle_bg_rect = pygame.Rect(20, HEIGHT - 50, 120, 35)
        label = "Dark Mode" if background_dark else "Light Mode"
        draw_button_2d(toggle_bg_rect.x, toggle_bg_rect.y, toggle_bg_rect.width, toggle_bg_rect.height, label,
                       active=True)

        draw_input_box_3d(matrix_size_rect.x, matrix_size_rect.y, matrix_size_rect.w, matrix_size_rect.h,
                          matrix_size_input, active=show_matrix_size_active)
        draw_input_box_3d(random_range_rect.x, random_range_rect.y, random_range_rect.w, random_range_rect.h,
                          random_range_input, active=show_random_range_active)
        draw_button_2d(random_button_rect.x, random_button_rect.y, random_button_rect.w, random_button_rect.h,
                       "Rand", active=True)

        multiplication_rect = pygame.Rect(WIDTH - 160, 280, 140, 30)  # Adjust position
        draw_input_box_3d(*multiplication_rect, multiplication_input, active=show_multiplication_active)

        pygame.display.flip()

if __name__ == "__main__":
    main()
