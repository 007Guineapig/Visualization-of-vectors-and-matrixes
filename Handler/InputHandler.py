import ast
import pygame
from pygame import K_BACKSPACE


class InputHandler:
    """Handles user input for vector/matrix entry"""

    def __init__(self):
        self.input_text = ""
        self.show_input_active = False
        self.multiplication_input = ""
        self.show_multiplication_active = False
        self.matrix_size_input = ""
        self.random_range_input = ""
        self.show_matrix_size_active = False
        self.show_random_range_active = False

    def handle_text_input(self, event, active_field='input'):
        """Handle keyboard input for text fields"""
        if active_field == 'input' and self.show_input_active:
            if event.key == K_BACKSPACE:
                self.input_text = self.input_text[:-1]
            else:
                self.input_text += event.unicode

        elif active_field == 'multiplication' and self.show_multiplication_active:
            if event.key == K_BACKSPACE:
                self.multiplication_input = self.multiplication_input[:-1]
            else:
                self.multiplication_input += event.unicode

        elif active_field == 'matrix_size' and self.show_matrix_size_active:
            if event.key == K_BACKSPACE:
                self.matrix_size_input = self.matrix_size_input[:-1]
            elif event.unicode.isdigit() or event.unicode == ',':
                self.matrix_size_input += event.unicode

        elif active_field == 'random_range' and self.show_random_range_active:
            if event.key == K_BACKSPACE:
                self.random_range_input = self.random_range_input[:-1]
            elif event.unicode.isdigit() or event.unicode in ",.-":
                self.random_range_input += event.unicode

    def parse_input(self):
        """Parse and return vector from input_text"""
        try:
            val = ast.literal_eval(self.input_text)
            if isinstance(val, (list, tuple)):
                if all(isinstance(row, (list, tuple)) and len(row) in (2, 3) for row in val):
                    return val  # Matrix
                elif len(val) in (2, 3) and all(isinstance(c, (int, float)) for c in val):
                    return val  # Single vector
            return None
        except:
            return None

    def parse_multiplication(self):
        """Parse multiplication input"""
        try:
            return ast.literal_eval(self.multiplication_input)
        except:
            return None