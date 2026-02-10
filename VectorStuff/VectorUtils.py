
class VectorUtils:
    """Utility functions for vector operations"""

    @staticmethod
    def format_number(x):
        """Format number nicely — remove .0 when not needed"""
        if float(x).is_integer():
            return str(int(x))
        else:
            return f"{x:.2f}".rstrip('0').rstrip('.')

    @staticmethod
    def format_vector(vec):
        """Format vectors and matrices cleanly for display"""
        if isinstance(vec[0], (int, float)):
            return "(" + ", ".join(VectorUtils.format_number(x) for x in vec) + ")"
        else:
            return "(" + ", ".join(
                "(" + ", ".join(VectorUtils.format_number(x) for x in row) + ")" for row in vec) + ")"

    @staticmethod
    def snap_number(x, precision=6):
        """Round to nearest number; convert to int if whole, else keep float."""
        x = round(x, precision)
        if x.is_integer():
            return int(x)
        return x

    @staticmethod
    def max_from_vectors(vectors):
        """Return max value from all vectors/matrices"""
        max_val = None
        for v in vectors:
            vec = v['vec'] if isinstance(v, dict) and 'vec' in v else v
            if isinstance(vec[0], (int, float)):
                current_max = max(vec)
            else:
                current_max = max(max(row) for row in vec)
            if max_val is None or current_max > max_val:
                max_val = current_max
        return max_val