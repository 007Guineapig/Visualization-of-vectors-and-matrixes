import sys
import math
import pygame
import numpy as np
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

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

def draw_dropdown(vectors, selected_index, rect, open=False):
    """Draw a top-left dropdown for vectors."""
    # Draw main box
    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    glOrtho(0, WIDTH, HEIGHT, 0, -1, 1)
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

    glColor3f(0,0,0)
    glBegin(GL_LINE_LOOP)
    glVertex2f(rect.x, rect.y)
    glVertex2f(rect.x + rect.width, rect.y)
    glVertex2f(rect.x + rect.width, rect.y + rect.height)
    glVertex2f(rect.x, rect.y + rect.height)
    glEnd()

    # Draw selected vector text
    label = "Select Vector" if selected_index == -1 else str(vectors[selected_index]['vec'])
    draw_text_2d(label, (rect.x + 5, rect.y + 5), color=(0,0,0), font_size=18)

    # Draw dropdown items if open
    if open:
        for i, v in enumerate(vectors):
            y = rect.y + rect.height + i * dropdown_item_height
            glColor3f(0.9, 0.9, 0.9) if i != selected_index else glColor3f(0.7,0.7,1)
            glBegin(GL_QUADS)
            glVertex2f(rect.x, y)
            glVertex2f(rect.x + rect.width, y)
            glVertex2f(rect.x + rect.width, y + dropdown_item_height)
            glVertex2f(rect.x, y + dropdown_item_height)
            glEnd()
            # Draw border
            glColor3f(0,0,0)
            glBegin(GL_LINE_LOOP)
            glVertex2f(rect.x, y)
            glVertex2f(rect.x + rect.width, y)
            glVertex2f(rect.x + rect.width, y + dropdown_item_height)
            glVertex2f(rect.x, y + dropdown_item_height)
            glEnd()
            # Draw text
            draw_text_2d(str(v['vec']), (rect.x + 5, y + 5), color=(0,0,0), font_size=16)

    glPopMatrix()
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)


def draw_arrowhead_3d(start, end, color=(1, 0, 1), size=0.2, camera_pos=None,radius = 0.05):
    """
    Draws a 3D arrowhead with oval/tube-like lines.
    """

    sx, sy, sz = start
    ex, ey, ez = end
    dir_vec = np.array([ex - sx, ey - sy, ez - sz], dtype=float)
    length = np.linalg.norm(dir_vec)
    if length == 0:
        return
    dir_vec /= length

    # Default "up" vector
    up = np.array([0, 1, 0], dtype=float)

    # Adjust toward camera for billboarding effect
    if camera_pos is not None:
        view_dir = np.array(camera_pos) - np.array(end)
        if np.linalg.norm(view_dir) > 0:
            view_dir /= np.linalg.norm(view_dir)
            side = np.cross(dir_vec, view_dir)
            if np.linalg.norm(side) > 1e-6:
                side /= np.linalg.norm(side)
                up = np.cross(side, dir_vec)
                up /= np.linalg.norm(up)

    spread_angle = math.radians(25)
    side1 = np.cross(dir_vec, up)
    side1 /= np.linalg.norm(side1)

    left_dir = np.cos(spread_angle) * -dir_vec + np.sin(spread_angle) * side1
    right_dir = np.cos(spread_angle) * -dir_vec - np.sin(spread_angle) * side1

    left_end = end + left_dir * size
    right_end = end + right_dir * size

    # Draw two oval/tube lines
    draw_oval_line(end, left_end, radius=radius, color=color)
    draw_oval_line(end, right_end, radius=radius, color=color)


def draw_sphere(position, radius=0.1, slices=12, stacks=12, color=(1,0,1)):
    glPushMatrix()
    glTranslatef(*position)
    glColor3fv(color)
    quad = gluNewQuadric()
    gluSphere(quad, radius, slices, stacks)
    gluDeleteQuadric(quad)
    glPopMatrix()

def draw_input_box_3d(x, y, w, h, text, active=False):
    # Switch to 2D
    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    glOrtho(0, WIDTH, HEIGHT, 0, -1, 1)
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()

    # Disable depth test so text is on top
    glDisable(GL_DEPTH_TEST)

    # Draw rectangle
    glColor3f(1, 1, 1) if not active else glColor3f(0.9, 0.9, 1)
    glBegin(GL_QUADS)
    glVertex2f(x, y)
    glVertex2f(x + w, y)
    glVertex2f(x + w, y + h)
    glVertex2f(x, y + h)
    glEnd()

    glColor3f(0, 0, 0)
    glBegin(GL_LINE_LOOP)
    glVertex2f(x, y)
    glVertex2f(x + w, y)
    glVertex2f(x + w, y + h)
    glVertex2f(x, y + h)
    glEnd()

    # Render text
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

    # Draw text on top of rectangle
    glBindTexture(GL_TEXTURE_2D, tex_id)
    glBegin(GL_QUADS)
    glTexCoord2f(0, 0);
    glVertex2f(x + 5, y + 5 + th)
    glTexCoord2f(1, 0);
    glVertex2f(x + 5 + tw, y + 5 + th)
    glTexCoord2f(1, 1);
    glVertex2f(x + 5 + tw, y + 5)
    glTexCoord2f(0, 1);
    glVertex2f(x + 5, y + 5)
    glEnd()

    glDisable(GL_TEXTURE_2D)
    glDisable(GL_BLEND)
    glDeleteTextures([tex_id])

    # Restore depth test
    glEnable(GL_DEPTH_TEST)

    glPopMatrix()
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)



def draw_arrowhead(dir_vec, color=(1,0,0), arrow_size=0.15):
    dir_vec = np.array(dir_vec, dtype=float)
    length = np.linalg.norm(dir_vec)
    if length == 0:
        return

    # Normalize
    dir_norm = dir_vec / length

    # Find a perpendicular vector for the base of the arrow
    if abs(dir_norm[0]) < 0.001 and abs(dir_norm[1]) < 0.001:
        up = np.array([0,1,0])
    else:
        up = np.array([0,0,1])
    side = np.cross(dir_norm, up)
    side /= np.linalg.norm(side)
    up_vec = np.cross(side, dir_norm)

    # Arrow tip at end
    tip = dir_vec
    # Base center
    base = tip - dir_norm * arrow_size
    glColor3fv(color)
    glBegin(GL_TRIANGLES)
    for angle in np.linspace(0, 2*np.pi, 6, endpoint=False):
        offset = side * np.cos(angle) * arrow_size/2 + up_vec * np.sin(angle) * arrow_size/2
        glVertex3fv(tip)
        glVertex3fv(base + offset)
        glVertex3fv(base - offset)
    glEnd()

def draw_axes(length=2.0, camera_pos = None):
    # Draw axis lines
    glLineWidth(5.0)
    glBegin(GL_LINES)
    # X axis
    glColor3f(1, 0, 0)
    glVertex3f(-length, 0, 0)
    glVertex3f(length, 0, 0)
    # Y axis
    glColor3f(0, 1, 0)
    glVertex3f(0, -length, 0)
    glVertex3f(0, length, 0)
    # Z axis
    glColor3f(0, 0, 1)
    glVertex3f(0, 0, -length)
    glVertex3f(0, 0, length)
    glEnd()
    glLineWidth(2.0)
    # Draw arrowheads
    arrow_size = 1

    draw_arrowhead_3d((0, 0, 0), (length, 0, 0), color=(1, 0, 0),size=arrow_size,camera_pos=camera_pos)
    draw_arrowhead_3d((0, 0, 0), (0, length, 0), color=(0, 1, 0),size=arrow_size,camera_pos=camera_pos)
    draw_arrowhead_3d((0, 0, 0), (0, 0, length), color=(0, 0, 1),size=arrow_size,camera_pos=camera_pos)
    draw_arrowhead_3d((0, 0, 0), (-length, 0, 0), color=(1, 0, 0),size=arrow_size,camera_pos=camera_pos)
    draw_arrowhead_3d((0, 0, 0), (0, -length, 0), color=(0, 1, 0),size=arrow_size,camera_pos=camera_pos)
    draw_arrowhead_3d((0, 0, 0), (0, 0, -length), color=(0, 0, 1),size=arrow_size,camera_pos=camera_pos)


    # Draw axis labels
    label_size = 100  # scale for text

    draw_text_3d("+X", (length + 0.2, 0, 0), color=(1,0,0), font_size=label_size)
    draw_text_3d("-X", (-length - 0.6, 0, 0), color=(1,0,0), font_size=label_size)
    draw_text_3d("+Y", (0, length + 0.2, 0), color=(0,1,0), font_size=label_size)
    draw_text_3d("-Y", (0, -length - 0.6, 0), color=(0,1,0), font_size=label_size)
    draw_text_3d("+Z", (0, 0, length + 0.2), color=(0,0,1), font_size=label_size)
    draw_text_3d("-Z", (0, 0, -length - 0.6), color=(0,0,1), font_size=label_size)

def draw_text_2d(text, position, color=(1,1,1), font_size=20):
    font = pygame.font.SysFont("Arial", font_size, True)
    text_surface = font.render(text, True, (int(color[0]*255), int(color[1]*255), int(color[2]*255)))
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
    glDisable(GL_DEPTH_TEST)  # important

    x, y = position
    glColor3f(1, 1, 1)  # make sure the texture color is not tinted
    glBindTexture(GL_TEXTURE_2D, tex_id)
    glBegin(GL_QUADS)
    glTexCoord2f(0, 0); glVertex2f(x, y + h)
    glTexCoord2f(1, 0); glVertex2f(x + w, y + h)
    glTexCoord2f(1, 1); glVertex2f(x + w, y)
    glTexCoord2f(0, 1); glVertex2f(x, y)
    glEnd()

    glEnable(GL_DEPTH_TEST)
    glDisable(GL_TEXTURE_2D)
    glDisable(GL_BLEND)
    glDeleteTextures([tex_id])

def draw_grid(size=10.0, step=1.0):
    """
    Draw a 3D grid along XZ, XY, and YZ planes with colors for each axis:
    X = red, Y = green, Z = blue.
    """
    glLineWidth(1.0)
    glBegin(GL_LINES)

    for i in range(-int(size), int(size) + 1, int(step)):

        glColor3f(0.0, 0.0, 1)
        glVertex3f(-size, 0, i)
        glVertex3f(size, 0, i)
        glVertex3f(i, 0, -size)
        glVertex3f(i, 0, size)

        glColor3f(1, 0.0, 0.0)
        glVertex3f(-size, i, 0)
        glVertex3f(size, i, 0)
        glVertex3f(i, -size, 0)
        glVertex3f(i, size, 0)

        glColor3f(0.0, 1, 0.0)
        glVertex3f(0, -size, i)
        glVertex3f(0, size, i)
        glVertex3f(0, i, -size)
        glVertex3f(0, i, size)

    glEnd()


def draw_grid_2d(step=1.0, z=-0.1, max_lines=200):
    """Draw a pseudo-infinite 2D grid that dynamically expands with zoom/pan."""
    ortho_scale = getattr(draw_grid_2d, 'ortho', 10.0)
    pan_x, pan_y = getattr(draw_grid_2d, 'pan', (0, 0))

    left, right = -ortho_scale + pan_x, ortho_scale + pan_x
    bottom, top = -ortho_scale + pan_y, ortho_scale + pan_y

    # Adaptive step to avoid thousands of lines when zoomed out
    visible_range = max(right - left, top - bottom)
    display_step = step
    if visible_range > 50:
        display_step *= int(visible_range / 50)

    # Limit number of lines for performance
    start_x = max(math.floor(left / display_step) * display_step, pan_x - max_lines * display_step)
    end_x = min(math.ceil(right / display_step) * display_step, pan_x + max_lines * display_step)
    start_y = max(math.floor(bottom / display_step) * display_step, pan_y - max_lines * display_step)
    end_y = min(math.ceil(top / display_step) * display_step, pan_y + max_lines * display_step)

    glColor3f(0.7, 0.7, 0.7)
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

def draw_oval_line(start, end, radius=0.005, segments=12, color=(1,1,0)):



    from numpy import array, cross, linalg
    from math import sin, cos, pi

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

    glColor3fv(color)  # <--- Set the color here

    glBegin(GL_TRIANGLE_STRIP)
    for i in range(segments+1):
        theta = 2 * pi * i / segments
        offset = radius * cos(theta) * side + radius * sin(theta) * up_vec
        glVertex3fv(start + offset)
        glVertex3fv(end + offset)
    glEnd()


def draw_planes(length=2.0):
    # XY plane
    glColor3f(1, 0, 0)

    glBegin(GL_QUADS)
    glVertex3f(-length, -length, 0)
    glVertex3f(length, -length, 0)
    glVertex3f(length, length, 0)
    glVertex3f(-length, length, 0)
    glEnd()

    # YZ plane
    glColor3f(0, 1, 0)
    glBegin(GL_QUADS)
    glVertex3f(0, -length, -length)
    glVertex3f(0, length, -length)
    glVertex3f(0, length, length)
    glVertex3f(0, -length, length)
    glEnd()

    # XZ plane
    glColor3f(0, 0, 1)
    glBegin(GL_QUADS)
    glVertex3f(-length, 0, -length)
    glVertex3f(length, 0, -length)
    glVertex3f(length, 0, length)
    glVertex3f(-length, 0, length)
    glEnd()

def draw_circle_2d(position, radius=0.1, segments=24, color=(1,1,0)):
    glColor3fv(color)
    glBegin(GL_TRIANGLE_FAN)
    glVertex3f(position[0], position[1], position[2])  # center
    for i in range(segments+1):
        angle = 2 * math.pi * i / segments
        x = position[0] + math.cos(angle) * radius
        y = position[1] + math.sin(angle) * radius
        glVertex3f(x, y, position[2])
    glEnd()

def draw_text_3d(text, position, color=(1, 1, 1), font_size=24, scale=0.01):
    """Draw 3D text at a world position, facing the camera."""
    font = pygame.font.SysFont("Arial", font_size, True)
    text_surface = font.render(text, True, (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)))
    text_data = pygame.image.tostring(text_surface, "RGBA", True)
    w, h = text_surface.get_size()

    tex_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex_id)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, text_data)

    glPushAttrib(GL_ENABLE_BIT | GL_COLOR_BUFFER_BIT)
    glEnable(GL_TEXTURE_2D)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    glPushMatrix()
    glTranslatef(*position)

    # Make text face camera
    modelview = glGetDoublev(GL_MODELVIEW_MATRIX).flatten()
    for i in range(3):
        for j in range(3):
            modelview[i * 4 + j] = (1.0 if i == j else 0.0)
    glLoadMatrixd(modelview)

    glBindTexture(GL_TEXTURE_2D, tex_id)
    glColor3f(1.0, 1.0, 1.0)  # Reset color to white to use texture color

    sx = w * scale
    sy = h * scale

    glBegin(GL_QUADS)
    glTexCoord2f(0, 0);
    glVertex3f(0, 0, 0)
    glTexCoord2f(1, 0);
    glVertex3f(sx, 0, 0)
    glTexCoord2f(1, 1);
    glVertex3f(sx, sy, 0)
    glTexCoord2f(0, 1);
    glVertex3f(0, sy, 0)
    glEnd()

    glPopMatrix()
    glDisable(GL_TEXTURE_2D)
    glDisable(GL_BLEND)
    glPopAttrib()
    glDeleteTextures([tex_id])


def draw_button_2d(x, y, w, h, label, active=False):
    # --- Set up orthographic 2D projection ---
    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    glOrtho(0, WIDTH, HEIGHT, 0, -1, 1)  # Top-left origin
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()

    # --- Draw button rectangle ---
    glColor3f(0.5, 0.9, 0.5) if active else glColor3f(0.8, 0.8, 0.8)
    glBegin(GL_QUADS)
    glVertex2f(x, y)
    glVertex2f(x + w, y)
    glVertex2f(x + w, y + h)
    glVertex2f(x, y + h)
    glEnd()

    # --- Draw button border ---
    glColor3f(0, 0, 0)
    glBegin(GL_LINE_LOOP)
    glVertex2f(x, y)
    glVertex2f(x + w, y)
    glVertex2f(x + w, y + h)
    glVertex2f(x, y + h)
    glEnd()

    # --- Render text ---
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
    glDisable(GL_DEPTH_TEST)  # make sure text is on top

    # Center text inside button
    tx = x + (w - tw) / 2
    ty = y + (h - th) / 2

    glBindTexture(GL_TEXTURE_2D, tex_id)
    glBegin(GL_QUADS)
    glTexCoord2f(0, 0); glVertex2f(tx, ty)
    glTexCoord2f(1, 0); glVertex2f(tx + tw, ty)
    glTexCoord2f(1, 1); glVertex2f(tx + tw, ty + th)
    glTexCoord2f(0, 1); glVertex2f(tx, ty + th)
    glEnd()

    # --- Cleanup ---
    glDisable(GL_TEXTURE_2D)
    glDisable(GL_BLEND)
    glEnable(GL_DEPTH_TEST)
    glDeleteTextures([tex_id])

    # --- Restore matrices ---
    glPopMatrix()
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)

def draw_input_box(screen, text, rect):
    pygame.draw.rect(screen, (255, 255, 255), rect)
    pygame.draw.rect(screen, (0, 0, 0), rect, 2)
    font = pygame.font.SysFont("Arial", 20)
    text_surf = font.render(text, True, (0, 0, 0))
    screen.blit(text_surf, (rect.x + 5, rect.y + 5))

def draw_axes_2d():
    """Draw axes extending to current viewport edges with all end labels."""
    ortho = getattr(draw_grid_2d, 'ortho', 15.0)
    pan_x, pan_y = getattr(draw_grid_2d, 'pan', (0,0))
    left, right = -ortho + pan_x, ortho + pan_x
    bottom, top = -ortho + pan_y, ortho + pan_y

    # Draw axes lines
    glLineWidth(3.0)
    glBegin(GL_LINES)
    glColor3f(1, 1, 1)  # X axis line
    glVertex2f(left, 0)
    glVertex2f(right, 0)
    glColor3f(1, 1, 1)  # Y axis line
    glVertex2f(0, bottom)
    glVertex2f(0, top)
    glEnd()

    # Draw axis labels in screen space
    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    glOrtho(0, WIDTH, HEIGHT, 0, -1, 1)
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()

    # Updated world_to_screen function with color
    def world_to_screen(wx, wy, color=(1,1,1), label=""):

        sx = (wx - left) / (right - left) * WIDTH
        sy = HEIGHT - (wy - bottom) / (top - bottom) * HEIGHT
        if label == "X+":
            sx -= 25
        if label == "Y-":
            sy -=25


        draw_text_2d(label, (sx, sy), color=color, font_size=20)
        return sx, sy

    # Draw each axis label with specific color
    world_to_screen(right, 0, color=(1,0,0), label="X+")   # Red X+
    world_to_screen(left, 0, color=(1,0,0), label="X-")    # Red X-
    world_to_screen(0, top, color=(0,1,0), label="Y+")     # Green Y+
    world_to_screen(0, bottom, color=(0,1,0), label="Y-")  # Green Y-

    glPopMatrix()
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)


def frange(start, stop, step):
    while start <= stop:
        yield start
        start += step

def draw_vectors_3d(vectors, line_width=6, base_arrow_size=0.15):
    glPushAttrib(GL_LINE_BIT | GL_CURRENT_BIT)
    glLineWidth(line_width)
    for v in vectors:
        vec = list(v) + [0] * (3 - len(v))
        # Draw the vector line
        draw_oval_line((0, 0, 0), vec, radius=0.005)
        # Draw the arrowhead at the tip
        arrow_size = 0.2
        draw_arrowhead_3d((0, 0, 0), vec, color=(1, 0, 1), size=arrow_size)
    glPopAttrib()

def draw_vectors_2d_animated(vectors, dt, line_width=6, arrow_fraction=0.15, speed=2.0, ortho_scale=6.0):
    glPushAttrib(GL_LINE_BIT | GL_CURRENT_BIT)
    glLineWidth(line_width)

    zoom_scale = ortho_scale / 6.0

    for v in vectors:
        vec = v['vec']

        # Single vector case
        if isinstance(vec[0], (int, float)):
            v['progress'] = min(v.get('progress', 0.0) + dt * speed, 1.0)
            x, y = vec[0], vec[1]
            px, py = x * v['progress'], y * v['progress']

            # Draw line
            glColor3f(1, 1, 0)
            glBegin(GL_LINES)
            glVertex2f(0, 0)
            glVertex2f(px, py)
            glEnd()

            # Draw arrowhead
            angle = math.atan2(y, x)
            vec_len = math.hypot(x, y) * v['progress']
            base_arrow_size = line_width * arrow_fraction * zoom_scale
            arrow_size = min(base_arrow_size, vec_len / 5.0)
            arrow_angle = 0.3
            glBegin(GL_LINES)
            glColor3f(1, 0, 1)
            # Left side
            glVertex2f(px, py)
            glVertex2f(px - arrow_size * math.cos(angle - arrow_angle),
                       py - arrow_size * math.sin(angle - arrow_angle))
            # Right side
            glVertex2f(px, py)
            glVertex2f(px - arrow_size * math.cos(angle + arrow_angle),
                       py - arrow_size * math.sin(angle + arrow_angle))
            glEnd()

        # Matrix of vectors case
        else:
            # Initialize per-row progress if not exists
            if 'row_progress' not in v:
                v['row_progress'] = [0.0] * len(vec)

            for i, row in enumerate(vec):
                x, y = row[0], row[1]
                v['row_progress'][i] = min(v['row_progress'][i] + dt * speed, 1.0)
                px, py = x * v['row_progress'][i], y * v['row_progress'][i]

                glColor3f(1, 1, 0)
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
                glColor3f(1, 0, 1)
                glVertex2f(px, py)
                glVertex2f(px - arrow_size * math.cos(angle - arrow_angle),
                           py - arrow_size * math.sin(angle - arrow_angle))
                glVertex2f(px, py)
                glVertex2f(px - arrow_size * math.cos(angle + arrow_angle),
                           py - arrow_size * math.sin(angle + arrow_angle))
                glEnd()
    glPopAttrib()

def draw_vectors_3d_animated(vectors, dt, line_width=6, base_arrow_size=0.15, speed=2.0, camera_pos=None):
    glPushAttrib(GL_LINE_BIT | GL_CURRENT_BIT)
    glLineWidth(line_width)

    reference_distance = 2.0  # distance at which the radius is "normal"
    base_max_radius = 0.02    # max radius at origin
    min_radius = 0.005        # minimum radius after zoom

    for v in vectors:
        # Single vector case
        if isinstance(v['vec'][0], (int, float)):
            if 'start_vec' not in v:
                v['start_vec'] = [0.0, 0.0, 0.0]

            v['progress'] = min(v.get('progress', 0.0) + dt * speed, 1.0)

            start_vec = list(v['start_vec']) + [0] * (3 - len(v['start_vec']))
            end_vec = list(v['vec']) + [0] * (3 - len(v['vec']))
            px, py, pz = [s + (e - s) * v['progress'] for s, e in zip(start_vec, end_vec)]

            # Compute distance along vector for tapering
            distance_along_vector = (px ** 2 + py ** 2 + pz ** 2) ** 0.5
            full_length = sum((e - s) ** 2 for s, e in zip(start_vec, end_vec)) ** 0.5

            # Base taper: thicker at origin, thinner at tip
            radius = base_max_radius * (1 - distance_along_vector / full_length) + min_radius

            # Adjust radius for camera distance
            if camera_pos is not None:
                mid_x, mid_y, mid_z = px / 2, py / 2, pz / 2
                cam_dist = ((camera_pos[0] - mid_x) ** 2 +
                            (camera_pos[1] - mid_y) ** 2 +
                            (camera_pos[2] - mid_z) ** 2) ** 0.5
                radius *= cam_dist / reference_distance

            radius = max(radius, min_radius)

            # Draw line and arrowhead
            draw_oval_line(start_vec, (px, py, pz), radius=radius)
            if v['progress'] >= 1.0:
                draw_arrowhead_3d(
                    start_vec, (px, py, pz),
                    color=v.get('color', (1, 0, 1)),
                    size=max(0.5, radius * 8),
                    camera_pos=camera_pos,
                    radius=radius
                )

        # Multiple vectors in a matrix
        else:
            if 'start_vecs' not in v:
                v['start_vecs'] = [[0.0, 0.0, 0.0] for _ in v['vec']]
            if 'row_progress' not in v:
                v['row_progress'] = [0.0] * len(v['vec'])

            for i, row in enumerate(v['vec']):
                v['row_progress'][i] = min(v['row_progress'][i] + dt * speed, 1.0)
                start_vec = list(v['start_vecs'][i]) + [0] * (3 - len(v['start_vecs'][i]))
                end_vec = list(row) + [0] * (3 - len(row))
                px, py, pz = [s + (e - s) * v['row_progress'][i] for s, e in zip(start_vec, end_vec)]

                distance_along_vector = (px ** 2 + py ** 2 + pz ** 2) ** 0.5
                full_length = sum((e - s) ** 2 for s, e in zip(start_vec, end_vec)) ** 0.5

                radius = base_max_radius * (1 - distance_along_vector / full_length) + min_radius

                if camera_pos is not None:
                    mid_x, mid_y, mid_z = px / 2, py / 2, pz / 2
                    cam_dist = ((camera_pos[0] - mid_x) ** 2 +
                                (camera_pos[1] - mid_y) ** 2 +
                                (camera_pos[2] - mid_z) ** 2) ** 0.5
                    radius *= cam_dist / reference_distance

                radius = max(radius, min_radius)
                draw_oval_line(start_vec, (px, py, pz), radius=radius)

                if v['row_progress'][i] >= 1.0:
                    draw_arrowhead_3d(
                        start_vec, (px, py, pz),
                        color=v.get('color', (1, 0, 1)),
                        size=max(0.5, radius * 8),
                        camera_pos=camera_pos,
                        radius=radius
                    )

    glPopAttrib()





def reset_vector_animation(vectors):
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

import ast
# Add this at the top of your imports
def main():
    global input_text, show_input_active
    pending_vector = None
    vectors_as_points = True
    draw_button_rect = (WIDTH - 160, 120, 140, 35)

    multiplication_input = ""  # stores the text typed in multiplication input
    show_multiplication_active = False  # tracks if multiplication input is active
    multiplication_rect = pygame.Rect(WIDTH - 160, 120, 140, 30)  # position of new input box

    global input_text, show_input_active, selected_vector_index  # <--- add this
    pending_vector = None
    vectors_as_points = True
    draw_button_rect = (WIDTH - 160, 120, 140, 35)

    # --- Dropdown variables ---
    #dropdown_rect = pygame.Rect(WIDTH - 160, 160, 140, 30)
    dropdown_item_height = 25
    dropdown_open = False
    if 'selected_vector_index' not in globals():
        selected_vector_index = None

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT), DOUBLEBUF | OPENGL)
    pygame.display.set_caption("XYZ axes — mouse control")

    glEnable(GL_DEPTH_TEST)
    glClearColor(0.0, 0.0, 0.0, 1.0)

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
                    grid_mode = (grid_mode + 1) % 3
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
            if grid_mode != 0:
                draw_grid_2d(step=1.0, z=-0.1)
            if show_axes:
                draw_axes_2d()

            if vectors_as_points:
                for v in all_vectors:
                    vec = v['vec']
                    base_radius = 0.07
                    radius = base_radius * ortho_scale / 6.5
                    # Single vector
                    if isinstance(vec[0], (int, float)):
                        draw_circle_2d([vec[0], vec[1], 0], radius=radius, color=(1, 1, 0))
                    else:
                        for row in vec:
                            draw_circle_2d([row[0], row[1], 0], radius=radius, color=(1, 1, 0))
            else:
                draw_vectors_2d_animated(all_vectors, dt, ortho_scale=ortho_scale)

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
                draw_planes(LENGTH_XYZ)
            elif grid_mode == 2:
                glColor3f(1, 1, 1)
                draw_grid(LENGTH_XYZ)

            if show_axes:
                camera_pos = (cam_x, cam_y, cam_z)
                draw_axes(LENGTH_XYZ,camera_pos)

            if vectors_as_points:
                for v in all_vectors:
                    vec = v['vec']
                    base_radius = 0.07
                    radius = base_radius * math.sqrt(distance / 7.0)
                    radius = max(radius, 0.03)  # minimum size

                    # Single vector
                    if isinstance(vec[0], (int, float)):
                        vec3d = list(vec) + [0] * (3 - len(vec))
                        draw_sphere(vec3d, radius=radius, color=(1, 1, 0))
                    # Matrix
                    else:
                        for row in vec:
                            vec3d = list(row) + [0] * (3 - len(row))
                            draw_sphere(vec3d, radius=radius, color=(1, 1, 0))

            else:
                camera_pos = (cam_x, cam_y, cam_z)
                draw_vectors_3d_animated(all_vectors, dt, camera_pos=camera_pos)

        draw_dropdown(animated_vectors, selected_vector_index, dropdown_rect, dropdown_open)
        draw_button_2d(*button_rect, "Switch 2D" if not view_2d_mode else "Switch 3D", active=view_2d_mode)
        draw_button_2d(*draw_button_rect, "Draw Vector", active=False)
        input_rect = pygame.Rect(WIDTH - 160, 75, 140, 30)
        draw_input_box_3d(*input_rect, input_text, active=show_input_active)

        multiplication_rect = pygame.Rect(WIDTH - 160, 170, 140, 30)  # Adjust position
        draw_input_box_3d(*multiplication_rect, multiplication_input, active=show_multiplication_active)

        pygame.display.flip()

if __name__ == "__main__":
    main()
