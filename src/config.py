import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    """
    Global configuration for the project.
    Using frozen=True ensures immutability of settings during runtime.
    """
    
    # Project root directory derivation
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Input data paths
    IMG_HIGH_CONTRAST: str = os.path.join(BASE_DIR, "data", "raw", "clear_highway.jpg")
    IMG_LOW_CONTRAST: str = os.path.join(BASE_DIR, "data", "raw", "rainy_city.jpg")