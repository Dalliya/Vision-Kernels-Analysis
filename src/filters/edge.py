import numpy as np
from src.filters.base import BaseKernelFilter

class SobelFilterX(BaseKernelFilter):
    def __init__(self) -> None:
        kernel = np.array([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=np.float32)
        super().__init__(kernel=kernel, name="Sobel X-Axis")

class SobelFilterY(BaseKernelFilter):
    def __init__(self) -> None:
        kernel = np.array([
            [-1, -2, -1],
            [ 0,  0,  0],
            [ 1,  2,  1]
        ], dtype=np.float32)
        super().__init__(kernel=kernel, name="Sobel Y-Axis")

class LaplacianFilter(BaseKernelFilter):
    def __init__(self) -> None:
        kernel = np.array([
            [ 0,  1,  0],
            [ 1, -4,  1],
            [ 0,  1,  0]
        ], dtype=np.float32)
        super().__init__(kernel=kernel, name="Laplacian (2nd Derivative)")

class PrewittFilterX(BaseKernelFilter):
    def __init__(self) -> None:
        kernel = np.array([
            [-1, 0, 1],
            [-1, 0, 1],
            [-1, 0, 1]
        ], dtype=np.float32)
        super().__init__(kernel=kernel, name="Prewitt X-Axis")

class PrewittFilterY(BaseKernelFilter):
    def __init__(self) -> None:
        kernel = np.array([
            [-1, -1, -1],
            [ 0,  0,  0],
            [ 1,  1,  1]
        ], dtype=np.float32)
        super().__init__(kernel=kernel, name="Prewitt Y-Axis")

class RobertsFilter(BaseKernelFilter):
    def __init__(self) -> None:
        kernel = np.array([
            [ 1,  0],
            [ 0, -1]
        ], dtype=np.float32)
        super().__init__(kernel=kernel, name="Roberts Cross (Diagonal)")