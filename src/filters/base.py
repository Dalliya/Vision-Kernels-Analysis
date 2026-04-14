import cv2
import numpy as np


class BaseKernelFilter:
    """
    Base class for applying matrix filters (convolution kernels) to images.
    
    All specific filters (e.g., Sobel, Gaussian, Median) must inherit 
    from this class or utilize its core convolution logic.
    """
    
    def __init__(self, kernel: np.ndarray, name: str) -> None:
        """
        Initializes the filter with a specific kernel.
        
        Args:
            kernel (np.ndarray): The weight matrix (convolution kernel).
            name (str): The designated name of the filter for logging 
                        and visualization purposes.
        """
        self.kernel = kernel
        self.name = name

    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        Applies the convolution kernel to the input image using 
        the optimized cv2.filter2D function.
        
        For derivative filters (like Sobel/Laplacian), it uses a 64-bit float 
        depth to capture negative gradients, then calculates the absolute 
        values to form a valid visual tensor without overflowing Matplotlib memory.
        
        Args:
            image (np.ndarray): The input image (single-channel or multi-channel).
            
        Returns:
            np.ndarray: The filtered (convolved) image in standard uint8 format.
        """
        # Step 1: Convolve using 64-bit float to prevent data truncation
        filtered_image = cv2.filter2D(src=image, ddepth=cv2.CV_64F, kernel=self.kernel)
        
        # Step 2: Take absolute values and convert back to uint8 (0-255)
        # This ensures Matplotlib can render it without memory/deepcopy crashes
        normalized_image = cv2.convertScaleAbs(filtered_image)
        
        return normalized_image

    def update_kernel(self, new_kernel: np.ndarray) -> None:
        """
        Updates the filter kernel dynamically.
        
        Used for parameter tuning and comparative analysis 
        (e.g., incrementing/decrementing kernel dimensions or weights).
        
        Args:
            new_kernel (np.ndarray): The new weight matrix to be applied.
        """
        self.kernel = new_kernel