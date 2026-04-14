import numpy as np
import cv2
from src.filters.base import BaseKernelFilter


class MeanFilter(BaseKernelFilter):
    """
    Applies a normalized box blur (mean) filter.
    Replaces each pixel with the average of its neighborhood.
    """
    
    def __init__(self, kernel_size: int = 3) -> None:
        """
        Initializes the Mean Filter.
        
        Args:
            kernel_size (int): Dimension of the square kernel (e.g., 3, 5, 7).
                               Must be an odd number.
        """
        if kernel_size % 2 == 0:
            raise ValueError("Kernel size must be an odd integer.")
            
        # Create a matrix of 1s and normalize by the total number of elements
        # Example for 3x3: all elements become 1/9
        kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size ** 2)
        
        super().__init__(kernel=kernel, name=f"Mean Filter ({kernel_size}x{kernel_size})")


class GaussianFilter(BaseKernelFilter):
    """
    Applies a Gaussian blur filter.
    Uses a weighted average where central pixels have higher importance,
    preserving edges better than a standard Mean Filter.
    """
    
    def __init__(self) -> None:
        """
        Initializes a standard 3x3 Gaussian approximation kernel.
        """
        # Standard 3x3 Gaussian kernel approximation
        # Normalized by dividing by the sum of all weights (16)
        kernel = np.array([
            [1, 2, 1],
            [2, 4, 2],
            [1, 2, 1]
        ], dtype=np.float32) / 16.0
        
        super().__init__(kernel=kernel, name="Gaussian Filter (3x3)")


class MedianFilter:
    """
    Applies a non-linear Median blur filter.
    Replaces each pixel with the median value of its local neighborhood.
    Highly effective at removing impulse noise (salt-and-pepper, raindrops) 
    while strictly preserving geometric edges.
    """
    
    def __init__(self, kernel_size: int = 3) -> None:
        """
        Initializes the Median Filter.
        
        Args:
            kernel_size (int): Dimension of the square window (e.g., 3, 5, 7).
                               Must be a positive odd integer.
        """
        if kernel_size % 2 == 0 or kernel_size < 3:
            raise ValueError("Kernel size must be an odd integer >= 3.")
            
        self.kernel_size = kernel_size
        self.name = f"Median Filter ({kernel_size}x{kernel_size})"

    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        Applies the median sorting algorithm to the image using cv2.medianBlur.
        
        Args:
            image (np.ndarray): The input image (single or multi-channel).
            
        Returns:
            np.ndarray: The noise-filtered image.
        """
        return cv2.medianBlur(image, self.kernel_size)

    def update_kernel(self, new_size: int) -> None:
        """
        Updates the window size dynamically for parameter tuning (Task #7).
        
        Args:
            new_size (int): The new dimension for the median window.
        """
        if new_size % 2 == 0 or new_size < 3:
            print(f"[WARNING] Invalid size {new_size}. Must be odd and >= 3.")
            return
            
        self.kernel_size = new_size
        self.name = f"Median Filter ({new_size}x{new_size})"