from main import *

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
    glTexCoord2f(0, 0)
    glVertex3f(0, 0, 0)
    glTexCoord2f(1, 0)
    glVertex3f(sx, 0, 0)
    glTexCoord2f(1, 1)
    glVertex3f(sx, sy, 0)
    glTexCoord2f(0, 1)
    glVertex3f(0, sy, 0)
    glEnd()

    glPopMatrix()
    glDisable(GL_TEXTURE_2D)
    glDisable(GL_BLEND)
    glPopAttrib()
    glDeleteTextures([tex_id])



def draw_dropdown(vectors, selected_index, rect, open=False):
    """Draw a top-left dropdown for vectors."""
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

    # Draw selected vector text (formatted)
    if selected_index == -1:
        label = "Select Vector"
    else:
        label = format_vector(vectors[selected_index]['vec'])
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
            # Border
            glColor3f(0,0,0)
            glBegin(GL_LINE_LOOP)
            glVertex2f(rect.x, y)
            glVertex2f(rect.x + rect.width, y)
            glVertex2f(rect.x + rect.width, y + dropdown_item_height)
            glVertex2f(rect.x, y + dropdown_item_height)
            glEnd()
            # Formatted text
            draw_text_2d(format_vector(v['vec']), (rect.x + 5, y + 5), color=(0,0,0), font_size=16)

    glPopMatrix()
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)

def draw_conehead_3d(start, end, color=(1.0, 0.0, 1.0), size=None, camera_pos = None, radius=0.3):
    start = np.array(start, dtype=float)
    end = np.array(end, dtype=float)

    # Ak size je None alebo skalár, spravíme skalárny vektor pozdĺž šípky
    if size is None:
        dir_vec = end - start
        length = np.linalg.norm(dir_vec)
        if length == 0:
            return
        size_vec = 0.2 * dir_vec  # 20% vektora
    elif np.isscalar(size):
        dir_vec = end - start
        length = np.linalg.norm(dir_vec)
        if length == 0:
            return
        size_vec = dir_vec / length * size
    else:
        # size je už vektor
        size_vec = np.array(size, dtype=float)

    cone_base_center = end - size_vec
    dir_vec = size_vec / np.linalg.norm(size_vec)  # jednotkový smer

    glColor3f(*color)
    glPushMatrix()
    glTranslatef(*cone_base_center)

    # Rotácia z osi Z na dir_vec
    up = np.array([0, 0, 1])
    axis = np.cross(up, dir_vec)
    axis_len = np.linalg.norm(axis)
    if axis_len > 1e-6:
        axis /= axis_len
        angle = math.degrees(np.arccos(np.clip(np.dot(up, dir_vec), -1.0, 1.0)))
        glRotatef(angle, *axis)
    elif dir_vec[2] < 0:
        glRotatef(180, 1, 0, 0)

    height = np.linalg.norm(size_vec)

    quad = gluNewQuadric()
    gluQuadricNormals(quad, GLU_SMOOTH)
    gluCylinder(quad, radius, 0.0, height, 20, 5)
    gluDeleteQuadric(quad)
    glPopMatrix()



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

def draw_input_box_3d(x, y, w, h, text, active=False,fill_color = (0.7,0.7,0.7),fill_color_outline = (0,0,0)):
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
    r, g, b = fill_color
    if active:
        # Slightly brighten the active cell
        r, g, b = min(r + 0.2, 1.0), min(g + 0.2, 1.0), min(b + 0.2, 1.0)
    glColor3f(r, g, b)
    glBegin(GL_QUADS)
    glVertex2f(x, y)
    glVertex2f(x + w, y)
    glVertex2f(x + w, y + h)
    glVertex2f(x, y + h)
    glEnd()
    r,g,b = fill_color_outline
    glLineWidth(4.0)
    glColor3f(r, g,b)
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
    glTexCoord2f(0, 0)
    glVertex2f(x + 5, y + 5 + th)
    glTexCoord2f(1, 0)
    glVertex2f(x + 5 + tw, y + 5 + th)
    glTexCoord2f(1, 1)
    glVertex2f(x + 5 + tw, y + 5)
    glTexCoord2f(0, 1)
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

def draw_axes_3D(length=2.0, camera_pos = None):
    # Draw axis lines
    length = length - 0.5
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
    length = length + 0.5
    draw_conehead_3d((0, 0, 0), (length, 0, 0), color=(1, 0, 0), size=arrow_size, camera_pos=camera_pos)
    draw_conehead_3d((0, 0, 0), (0, length, 0), color=(0, 1, 0), size=arrow_size, camera_pos=camera_pos)
    draw_conehead_3d((0, 0, 0), (0, 0, length), color=(0, 0, 1), size=arrow_size, camera_pos=camera_pos)
    draw_conehead_3d((0, 0, 0), (-length, 0, 0), color=(1, 0, 0), size=arrow_size, camera_pos=camera_pos)
    draw_conehead_3d((0, 0, 0), (0, -length, 0), color=(0, 1, 0), size=arrow_size, camera_pos=camera_pos)
    draw_conehead_3d((0, 0, 0), (0, 0, -length), color=(0, 0, 1), size=arrow_size, camera_pos=camera_pos)
    # Draw axis labels
    label_size = 100  # scale for text

    draw_text_3d("+X1", (length + 0.2, 0, 0), color=(1,0,0), font_size=label_size)
    draw_text_3d("-X1", (-length - 0.6, 0, 0), color=(1,0,0), font_size=label_size)
    draw_text_3d("+X2", (0, length + 0.2, 0), color=(0,1,0), font_size=label_size)
    draw_text_3d("-X2", (0, -length - 0.6, 0), color=(0,1,0), font_size=label_size)
    draw_text_3d("+X3", (0, 0, length + 0.2), color=(0,0,1), font_size=label_size)
    draw_text_3d("-X3", (0, 0, -length - 0.6), color=(0,0,1), font_size=label_size)


def draw_cylinder_axis(start, end, radius=0.1, slices=16, color=(1.0, 0.0, 0.0)):
    # Compute vector from start to end
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    dz = end[2] - start[2]
    length = (dx ** 2 + dy ** 2 + dz ** 2) ** 0.5

    # Save current matrix
    glPushMatrix()

    # Move to start position
    glTranslatef(*start)

    # Compute rotation
    if length > 0.0001:
        # axis of rotation is cross product of (0,0,1) and direction vector
        ax = [-dy, dx, 0.0]
        angle = math.degrees(math.acos(dz / length))
        glRotatef(angle, *ax)

    glColor3f(*color)
    quad = gluNewQuadric()
    gluCylinder(quad, radius, radius, length, slices, 1)
    gluDeleteQuadric(quad)

    glPopMatrix()

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

def draw_grid_3D(size=10.0, step=1.0):
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

def draw_oval_line(start, end, radius=0.005, segments=12, color=(1,1,0), alpha=1.0):
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

    # Enable blending if not already enabled
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    # Use RGBA color
    glColor4f(color[0], color[1], color[2], alpha)

    glBegin(GL_TRIANGLE_STRIP)
    for i in range(segments + 1):
        theta = 2 * pi * i / segments
        offset = radius * cos(theta) * side + radius * sin(theta) * up_vec
        glVertex3fv(start + offset)
        glVertex3fv(end + offset)
    glEnd()

def draw_planes_3D(size=2.0, step=1.0, colored=False, camera_pos = (0,0,0)):
    # ---- Draw plain white planes ----
    if colored:  # White color
        glColor3f(0.8, 0.8, 0.8)
    else:
        glColor3f(1.0, 0, 0)
    # XZ plane
    glBegin(GL_QUADS)
    glVertex3f(-size, 0, -size)
    glVertex3f(size, 0, -size)
    glVertex3f(size, 0, size)
    glVertex3f(-size, 0, size)
    glEnd()

    if colored:  # White color
        glColor3f(0.8, 0.8, 0.8)
    else:
        glColor3f(0, 1, 0)

    # XY plane
    glBegin(GL_QUADS)
    glVertex3f(-size, -size, 0)
    glVertex3f(size, -size, 0)
    glVertex3f(size, size, 0)
    glVertex3f(-size, size, 0)
    glEnd()

    if colored:  # White color
        glColor3f(0.8, 0.8, 0.8)
    else:
        glColor3f(0, 0, 1)

    # YZ plane
    glBegin(GL_QUADS)
    glVertex3f(0, -size, -size)
    glVertex3f(0, size, -size)
    glVertex3f(0, size, size)
    glVertex3f(0, -size, size)
    glEnd()

    # ---- Draw grid lines ----
    glColor3f(0, 0, 0)  # Gray lines
    # XZ plane grid
    glBegin(GL_LINES)

    for x in range(-int(size), int(size) + 1, int(step)):
        glVertex3f(x, 0, -size)
        glVertex3f(x, 0, size)
    for z in range(-int(size), int(size) + 1, int(step)):
        glVertex3f(-size, 0, z)
        glVertex3f(size, 0, z)
    glEnd()

    # XY plane grid
    glBegin(GL_LINES)
    for x in range(-int(size), int(size) + 1, int(step)):
        glVertex3f(x, -size, 0)
        glVertex3f(x, size, 0)
    for y in range(-int(size), int(size) + 1, int(step)):
        glVertex3f(-size, y, 0)
        glVertex3f(size, y, 0)
    glEnd()

    # YZ plane grid
    glBegin(GL_LINES)
    for y in range(-int(size), int(size) + 1, int(step)):
        glVertex3f(0, y, -size)
        glVertex3f(0, y, size)
    for z in range(-int(size), int(size) + 1, int(step)):
        glVertex3f(0, -size, z)
        glVertex3f(0, size, z)
    glEnd()

    # ---- Draw axes ----
    draw_cylinder_axis(start=(-size, 0, 0), end=(size, 0, 0), radius=0.05, color=(0.0, 0.0, 0.0))
    draw_cylinder_axis(start=(0, -size, 0), end=(0, size, 0), radius=0.05, color=(0.0, 0.0, 0.0))
    draw_cylinder_axis(start=(0, 0, -size), end=(0, 0, size), radius=0.05, color=(0.0, 0.0, 0.0))


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
    glColor3f(1, 0, 0)  # X axis line
    glVertex2f(left, 0)
    glVertex2f(right, 0)
    glColor3f(0, 1, 0)  # Y axis line
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
    def world_to_screen(wx, wy, color=(1, 1, 1), label=""):
        sx = (wx - left) / (right - left) * WIDTH
        sy = HEIGHT - (wy - bottom) / (top - bottom) * HEIGHT

        # Horizontal offsets for left/right labels
        if label in ["X1+", "X1-"]:
            sx += -30 if label == "X1+" else 10
            sy += 5

        # Vertical and small horizontal offsets for top/bottom labels
        if label in ["X2+", "X2-"]:
            sy += -30 if label == "X2-" else 10
            sx += 5  # shift a little right to avoid overlapping Y-axis

        draw_text_2d(label, (sx, sy), color=color, font_size=20)
        return sx, sy

    # Draw each axis label with specific color
    world_to_screen(right, 0, color=(1,0,0), label="X1+")   # Red X+
    world_to_screen(left, 0, color=(1,0,0), label="X1-")    # Red X-
    world_to_screen(0, top, color=(0,1,0), label="X2+")     # Green Y+
    world_to_screen(0, bottom, color=(0,1,0), label="X2-")  # Green Y-

    glPopMatrix()
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)