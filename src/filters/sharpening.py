import numpy as np
from src.filters.base import BaseKernelFilter

class SharpenFilter(BaseKernelFilter):
    """
    Standard sharpening filter. Enhances differences between adjacent pixels,
    making edges appear more distinct.
    """
    def __init__(self) -> None:
        kernel = np.array([
            [ 0, -1,  0],
            [-1,  5, -1],
            [ 0, -1,  0]
        ], dtype=np.float32)
        super().__init__(kernel=kernel, name="Sharpen Filter")

class UnsharpMaskFilter(BaseKernelFilter):
    """
    Unsharp Masking approximation using a single high-contrast kernel.
    Amplifies high-frequency details (edges) more aggressively than standard sharpen.
    """
    def __init__(self) -> None:
        kernel = np.array([
            [-1, -2, -1],
            [-2, 13, -2],
            [-1, -2, -1]
        ], dtype=np.float32)
        super().__init__(kernel=kernel, name="Unsharp Masking")