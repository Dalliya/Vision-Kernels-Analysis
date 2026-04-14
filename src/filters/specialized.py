import numpy as np
import cv2
from src.filters.base import BaseKernelFilter

class GaborFilter(BaseKernelFilter):
    """
    Gabor filter for texture analysis.
    Extracts features at specific orientations and frequencies.
    """
    def __init__(self, ksize: int = 31, sigma: float = 4.0, theta: float = np.pi/4, 
                 lambd: float = 10.0, gamma: float = 0.5, psi: float = 0.0) -> None:
        
        kernel = cv2.getGaborKernel(
            (ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_32F
        )
        
        theta_deg = int(np.degrees(theta))
        super().__init__(
            kernel=kernel, 
            name=f"Gabor (Theta={theta_deg}°, Lambda={lambd})"
        )

    # === КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ ЗДЕСЬ ===
    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        Overrides the base apply method to FORCE float32 calculation.
        This prevents OpenCV from clipping values to 255 before we can normalize them.
        """
        return cv2.filter2D(image, cv2.CV_32F, self.kernel)


class GaborFilterBank:
    """
    Applies a bank of Gabor filters at different orientations (0, 45, 90, 135 degrees)
    and combines the results. This is a crucial step for generalized texture detection.
    """
    def __init__(self) -> None:
        self.filters = [
            GaborFilter(theta=0),             # Horizontal
            GaborFilter(theta=np.pi/4),       # 45 degrees diagonal
            GaborFilter(theta=np.pi/2),       # 90 degrees vertical
            GaborFilter(theta=3*np.pi/4)      # 135 degrees diagonal
        ]
        self.name = "Gabor Filter Bank (Combined)"

    def apply(self, image: np.ndarray) -> np.ndarray:
        print(f"      [Applying {self.name} - This might take a second...]")
        
        # 1. Apply each filter and take the ABSOLUTE value
        responses = [np.abs(f.apply(image)) for f in self.filters]
        
        # 2. Combine by taking the maximum response at each pixel
        combined = np.max(np.array(responses), axis=0)
        
        # 3. NORMALIZATION: Squeeze the float32 numbers safely back into 0-255 range
        combined_min = np.min(combined)
        combined_max = np.max(combined)
        
        if combined_max > combined_min:
            normalized = (combined - combined_min) / (combined_max - combined_min)
            normalized_img = (normalized * 255).astype(np.uint8)
        else:
            normalized_img = combined.astype(np.uint8)
            
        return normalized_img