import cv2
import numpy as np
import os


def load_and_convert_image(filepath: str, max_dim: int = 1024) -> np.ndarray:
    """
    Loads an image, logs metadata, resizes if too large to prevent OOM errors, 
    and converts it to a single channel (Grayscale).
    """
    image: np.ndarray = cv2.imread(filepath)
    
    if image is None:
        raise FileNotFoundError(f"Failed to load image: {filepath}")

    height, width = image.shape[:2]
    channels = image.shape[2] if len(image.shape) == 3 else 1
    
    print(f"[INFO] Loaded: {filepath}")
    print(f"[INFO] Original Dimensions: {width}x{height} px")

    # Embedded size optimization block
    if width > max_dim or height > max_dim:
        scale = max_dim / max(width, height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        print(f"[INFO] Resized to: {new_width}x{new_height} px")

    if channels > 1:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        print("[INFO] Converted to single channel (Grayscale).\n")
    else:
        gray_image = image
        print("[INFO] Image is already single channel.\n")
        
    return gray_image


def save_image(filepath: str, image: np.ndarray) -> None:
    """
    Saves an image tensor to the disk. 
    Automatically creates directories if they do not exist.
    
    Args:
        filepath (str): The full path where the image should be saved.
        image (np.ndarray): The image tensor to save.
    """
    # Extract the directory path from the full filepath
    directory = os.path.dirname(filepath)
    
    # Create the directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    # Save the image using OpenCV
    cv2.imwrite(filepath, image)
    print(f"[INFO] Successfully saved: {filepath}")