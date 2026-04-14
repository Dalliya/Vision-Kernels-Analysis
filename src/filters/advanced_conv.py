import numpy as np
import cv2

class DilatedConvolution:
    """
    Dilated (Atrous) Convolution.
    Expands the receptive field without increasing the number of parameters or computation.
    Crucial for semantic segmentation networks like DeepLab.
    """
    def __init__(self, kernel: np.ndarray, dilation_rate: int = 2):
        self.original_kernel = kernel
        self.dilation_rate = dilation_rate
        self.name = f"Dilated Conv (Rate {dilation_rate})"
        
        # We physically construct the dilated kernel by injecting zeros (holes)
        kh, kw = kernel.shape
        d_kh = (kh - 1) * dilation_rate + 1
        d_kw = (kw - 1) * dilation_rate + 1
        
        self.dilated_kernel = np.zeros((d_kh, d_kw), dtype=np.float32)
        for i in range(kh):
            for j in range(kw):
                self.dilated_kernel[i * dilation_rate, j * dilation_rate] = kernel[i, j]

    def apply(self, image: np.ndarray) -> np.ndarray:
        # Calculate convolution using CV_32F to avoid clipping negative/large values
        result = cv2.filter2D(image, cv2.CV_32F, self.dilated_kernel)
        
        # Normalize back to uint8 safely
        result_abs = np.abs(result)
        r_min, r_max = np.min(result_abs), np.max(result_abs)
        if r_max > r_min:
            normalized = (result_abs - r_min) / (r_max - r_min)
            return (normalized * 255).astype(np.uint8)
        return result_abs.astype(np.uint8)


class Conv1x1:
    """
    1x1 Convolution (Pointwise Convolution).
    Used in ResNet and Inception architectures as a 'bottleneck' to reduce dimensionality 
    across the channel (depth) axis, acting as a smart channel mixer.
    """
    def __init__(self, weights: list):
        # Weights represent how much of each channel to mix into the final output
        self.weights = np.array(weights, dtype=np.float32)
        self.name = "1x1 Conv (Channel Bottleneck)"

    def apply(self, image_stack: np.ndarray) -> np.ndarray:
        """
        Expects an image_stack of shape (Height, Width, Channels).
        """
        # Multiplies each channel by its specific 1x1 weight and sums them up (reducing depth to 1)
        # This is exactly what a 1x1 convolution does under the hood.
        bottleneck_output = np.sum(image_stack * self.weights, axis=2)
        
        # Normalize
        b_min, b_max = np.min(bottleneck_output), np.max(bottleneck_output)
        if b_max > b_min:
            normalized = (bottleneck_output - b_min) / (b_max - b_min)
            return (normalized * 255).astype(np.uint8)
        return bottleneck_output.astype(np.uint8)