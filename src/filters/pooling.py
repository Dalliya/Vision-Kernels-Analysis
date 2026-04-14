import numpy as np

class MaxPooling:
    """
    Implementation of the Max Pooling layer found in Convolutional Neural Networks.
    Reduces spatial dimensions while retaining the most prominent features (edges).
    """
    def __init__(self, pool_size: int = 2, stride: int = 2) -> None:
        self.pool_size = pool_size
        self.stride = stride
        self.name = f"Max Pooling ({pool_size}x{pool_size}, Stride {stride})"

    def apply(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape
        # Calculate new dimensions
        out_h = (h - self.pool_size) // self.stride + 1
        out_w = (w - self.pool_size) // self.stride + 1

        pooled_image = np.zeros((out_h, out_w), dtype=image.dtype)

        for y in range(out_h):
            for x in range(out_w):
                y_start = y * self.stride
                y_end = y_start + self.pool_size
                x_start = x * self.stride
                x_end = x_start + self.pool_size

                # Extract the local window and find the maximum value
                window = image[y_start:y_end, x_start:x_end]
                pooled_image[y, x] = np.max(window)

        return pooled_image


class AveragePooling:
    """
    Implementation of the Average Pooling layer.
    Smooths the image and reduces dimensions by taking the mean of the window.
    """
    def __init__(self, pool_size: int = 2, stride: int = 2) -> None:
        self.pool_size = pool_size
        self.stride = stride
        self.name = f"Average Pooling ({pool_size}x{pool_size}, Stride {stride})"

    def apply(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape
        out_h = (h - self.pool_size) // self.stride + 1
        out_w = (w - self.pool_size) // self.stride + 1

        pooled_image = np.zeros((out_h, out_w), dtype=image.dtype)

        for y in range(out_h):
            for x in range(out_w):
                y_start = y * self.stride
                y_end = y_start + self.pool_size
                x_start = x * self.stride
                x_end = x_start + self.pool_size

                window = image[y_start:y_end, x_start:x_end]
                pooled_image[y, x] = np.mean(window)

        return pooled_image