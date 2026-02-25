import numpy as np
from Controllers.AnimationController import AnimationController
from Controllers.SpanAnimationController import SpanAnimationController
from Controllers.BasisDecompositionController import BasisDecompositionController
from VectorStuff.VectorUtils import VectorUtils


class VectorManager:
    """Manages vector operations and animations"""

    def __init__(self):
        self.animated_vectors = []
        self.pending_vector = None
        self.selected_vector_index = -1
        self.animation_controller = AnimationController()
        self.span_controller = SpanAnimationController()
        self.decomposition_controller = BasisDecompositionController()  # NOVÉ

    def add_vector(self, vec, progress=0.0):
        """Add a new vector"""
        self.animated_vectors.append({'vec': vec, 'progress': progress})

    def reset_animations(self):
        """Reset all vector animations"""
        for v in self.animated_vectors:
            v['progress'] = 0.0
            if 'segments' in v:
                v['segments'].clear()
            if 'row_progress' in v:
                v['row_progress'] = [0.0] * len(v['row_progress'])
            if 'row_segments' in v:
                for lst in v['row_segments']:
                    lst.clear()

    def multiply_selected_vector(self, multiplier):
        """Multiply selected vector by a value (scalar or matrix)"""
        if self.selected_vector_index == -1:
            return False

        selected = self.animated_vectors[self.selected_vector_index]
        selected_np = np.array(selected['vec'])
        val_np = np.array(multiplier)

        if selected_np.ndim == 1 and val_np.ndim == 1:
            result = np.dot(selected_np, val_np)
        else:
            result = selected_np @ val_np

        frames = 30
        if isinstance(result, np.ndarray):
            if result.ndim == 1:
                target = result.tolist()
                current = selected['vec']
                selected['delta'] = [(t - c) / frames for c, t in zip(current, target)]
                selected['frames_left'] = frames
            else:
                target_vectors = result.tolist()
                current_vectors = selected['vec']
                selected['delta'] = [
                    [(t - c) / frames for c, t in zip(curr_vec, targ_vec)]
                    for curr_vec, targ_vec in zip(current_vectors, target_vectors)
                ]
                selected['frames_left'] = frames
        return True

    def update_animations(self, dt):
        """Update frame-based animations"""
        self.animation_controller.update(dt)
        self.span_controller.update(dt)
        self.decomposition_controller.update(dt)  # NOVÉ

        for v in self.animated_vectors:
            if 'frames_left' in v and v['frames_left'] > 0:
                if isinstance(v['vec'][0], (list, tuple)):
                    v['vec'] = [
                        [c + d for c, d in zip(list(curr), delt)]
                        for curr, delt in zip(v['vec'], v['delta'])
                    ]
                else:
                    v['vec'] = [c + d for c, d in zip(list(v['vec']), v['delta'])]

                v['frames_left'] -= 1

                if v['frames_left'] == 0:
                    if isinstance(v['vec'][0], (list, tuple)):
                        v['vec'] = [[VectorUtils.snap_number(c) for c in row] for row in v['vec']]
                    else:
                        v['vec'] = [VectorUtils.snap_number(c) for c in v['vec']]