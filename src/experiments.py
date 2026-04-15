import numpy as np
from typing import List, Tuple

# Block 1: Pre-processing (Smoothing & Sharpening)
from src.filters.smoothing import MeanFilter, GaussianFilter, MedianFilter
from src.filters.sharpening import SharpenFilter, UnsharpMaskFilter

# Block 2: Feature Extraction (Edge Detection)
from src.filters.edge import (
    SobelFilterX, SobelFilterY, LaplacianFilter, 
    PrewittFilterX, PrewittFilterY, RobertsFilter
)

# Block 3: Advanced Features (Texture Analysis)
from src.filters.specialized import GaborFilterBank

# Block 4: Spatial Reduction (Pooling layers)
from src.filters.pooling import MaxPooling, AveragePooling

# Block 5: Advanced CNN Convolutions
from src.filters.advanced_conv import DilatedConvolution, Conv1x1

# Block 6: Metrics & Evaluation
from src.filters.base import BaseKernelFilter


def run_edge_detection_experiment(image_tensor: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    """
    EXPERIMENT A: Isolated Edge Detection.
    Evaluates classic derivative-based operators.
    """
    print("[EXPERIMENT] Running Edge Detection kernels...")
    
    # ИСПРАВЛЕНО: Добавлены Prewitt и Roberts для корректного запуска на всех картинках!
    filters = [
        SobelFilterX(), 
        SobelFilterY(), 
        LaplacianFilter(),
        PrewittFilterX(),
        RobertsFilter()
    ]
    
    results = []
    for f in filters:
        results.append((f.name, f.apply(image_tensor)))
    return results


def run_smoothing_experiment(image_tensor: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    """
    EXPERIMENT B: Isolated Smoothing.
    Evaluates noise reduction techniques (linear vs non-linear).
    """
    print("[EXPERIMENT] Running Smoothing (Noise Reduction) kernels...")
    filters = [MeanFilter(kernel_size=5), GaussianFilter(), MedianFilter(kernel_size=5)]
    results = []
    for f in filters:
        results.append((f.name, f.apply(image_tensor)))
    return results


def run_sequential_pipeline(image_tensor: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    """
    EXPERIMENT C: Advanced Sequential Pipeline.
    Simulates a professional computer vision pipeline: Noise Removal -> Sharpening -> Edge Extraction.
    """
    print("[EXPERIMENT] Initiating Advanced Sequential Pipeline...")
    results = []
    
    # Step 1: Apply non-linear smoothing to remove high-frequency noise (e.g., rain)
    print("   -> Step 1: Applying Median Filter (Rain Removal)")
    median_filter = MedianFilter(kernel_size=5)
    smoothed_img = median_filter.apply(image_tensor)
    results.append(("1_Smoothed_Median_5x5", smoothed_img))
    
    # Step 2: Recover edges blurred during the smoothing phase
    print("   -> Step 2: Applying Unsharp Masking (Detail Recovery)")
    unsharp_mask = UnsharpMaskFilter()
    sharpened_img = unsharp_mask.apply(smoothed_img) 
    results.append(("2_Sharpened_Unsharp_Mask", sharpened_img))
    
    # Step 3: Extract final features from the restored image
    print("   -> Step 3: Applying Edge Detectors on the restored image")
    sobel_x = SobelFilterX()
    prewitt_x = PrewittFilterX()
    roberts = RobertsFilter()
    
    results.append(("3_Edges_Sobel_X", sobel_x.apply(sharpened_img)))
    results.append(("3_Edges_Prewitt_X", prewitt_x.apply(sharpened_img)))
    results.append(("3_Edges_Roberts", roberts.apply(sharpened_img)))
    
    return results


def run_texture_analysis_experiment(image_tensor: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    """
    EXPERIMENT D: Texture and Spectral Analysis.
    Applies a bank of Gabor filters to extract orientation-specific textures.
    """
    print("\n[EXPERIMENT] Running Texture Analysis (Gabor Filter Bank)...")
    results = []
    gabor_bank = GaborFilterBank()
    
    filtered_img = gabor_bank.apply(image_tensor)
    results.append(("Gabor_Bank_Combined", filtered_img))
    
    return results


def run_pooling_experiment(image_tensor: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    """
    EXPERIMENT E: Spatial Reduction.
    Simulates how a Convolutional Neural Network (CNN) compresses edge maps 
    to save memory and extract dominant features (downsampling).
    """
    print("\n[EXPERIMENT] Running Spatial Reduction (Pooling)...")
    results = []
    
    # 1. Extract clean edges using the established pipeline (Experiment C)
    print("   -> Preparing base edge map for pooling...")
    smoothed = MedianFilter(kernel_size=5).apply(image_tensor)
    sharpened = UnsharpMaskFilter().apply(smoothed)
    edges = SobelFilterX().apply(sharpened)
    
    results.append(("0_Original_Edges_Before_Pooling", edges))
    
    # 2. Apply Max Pooling (retains the strongest activations / brightest edges)
    print("   -> Applying Max Pooling (2x2)...")
    max_pool = MaxPooling(pool_size=2, stride=2)
    max_pooled = max_pool.apply(edges)
    results.append(("1_Max_Pooled_2x2", max_pooled))
    
    # 3. Apply Average Pooling (computes the mean, often leading to blurred downsampling)
    print("   -> Applying Average Pooling (2x2)...")
    avg_pool = AveragePooling(pool_size=2, stride=2)
    avg_pooled = avg_pool.apply(edges)
    results.append(("2_Average_Pooled_2x2", avg_pooled))
    
    return results


def run_advanced_cnn_experiment(image_tensor: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    """
    EXPERIMENT F: Advanced CNN Layers.
    Simulates depthwise separable convolutions and dilated receptive fields
    found in modern architectures like MobileNet and DeepLab.
    """
    print("\n[EXPERIMENT] Running Advanced CNN Convolutions...")
    results = []
    
    # Base smoothing to prepare the image
    smoothed = MedianFilter(kernel_size=5).apply(image_tensor)
    
    # --- 1. Pointwise Convolution (1x1) - MobileNet style ---
    print("   -> Simulating MobileNet: Mixing Depthwise Edges with 1x1 Conv...")
    sobel_x = SobelFilterX().apply(smoothed)
    sobel_y = SobelFilterY().apply(smoothed)
    
    # Stack 2D arrays into a 3D tensor (Simulating CNN Channels: Height x Width x 2)
    edge_channels = np.stack((sobel_x, sobel_y), axis=-1)
    
    # 1x1 Conv bottleneck: Mix 50% of X-edges and 50% of Y-edges into a single feature map
    mixed_edges = Conv1x1(weights=[0.5, 0.5]).apply(edge_channels)
    results.append(("1_Conv1x1_Mixed_Edges", mixed_edges))
    
    # --- 2. Dilated (Atrous) Convolution - DeepLab style ---
    print("   -> Simulating DeepLab: Dilated Convolution (Rate 2)...")
    laplacian_kernel = np.array([
        [-1, -1, -1],
        [-1,  8, -1],
        [-1, -1, -1]
    ], dtype=np.float32)
    
    dilated_conv = DilatedConvolution(kernel=laplacian_kernel, dilation_rate=2)
    dilated_edges = dilated_conv.apply(smoothed)
    results.append(("2_Dilated_Conv_Rate2", dilated_edges))
    
    return results


# =============================================================================
# EXPERIMENT G: METRICS & PARAMETER SHIFT (HW Requirement)
# =============================================================================

class CustomSharpen(BaseKernelFilter):
    """A flexible sharpen filter to test parameter shifts (+-1)."""
    def __init__(self, center_weight: float):
        # We change the central pixel weight
        kernel = np.array([
            [0, -1, 0], 
            [-1, center_weight, -1], 
            [0, -1, 0]
        ], dtype=np.float32)
        super().__init__(kernel=kernel, name=f"Custom_Sharpen_Center_{center_weight}")

def run_metrics_and_parameter_shift(image_tensor: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    """
    EXPERIMENT G: Parameter Shift & Mathematical Metrics.
    Fulfills the requirement to change a parameter by +-1 and calculate
    the pixel-wise sum of differences to evaluate filter sensitivity.
    """
    print("\n[EXPERIMENT] Running Parameter Shift (+-1) and Metrics...")
    results = []
    
    # 1. Base Filter (Center weight = 5)
    print("   -> Applying Base Filter (Center = 5)...")
    filter_base = CustomSharpen(center_weight=5)
    img_base = filter_base.apply(image_tensor)
    results.append(("1_Base_Sharpen_5", img_base))
    
    # 2. Shifted Filter (Center weight = 6, i.e., +1)
    print("   -> Applying Shifted Filter (Center = 6, changed by +1)...")
    filter_shifted = CustomSharpen(center_weight=6)
    img_shifted = filter_shifted.apply(image_tensor)
    results.append(("2_Shifted_Sharpen_6", img_shifted))
    
    # 3. CALCULATE METRIC: Pixel-wise sum of absolute differences
    # We use float32 to prevent overflow during subtraction
    diff_tensor = np.abs(img_base.astype(np.float32) - img_shifted.astype(np.float32))
    total_difference = np.sum(diff_tensor)
    
    print("-" * 50)
    print(f"   [METRIC RESULT] Total Pixel-wise Difference: {total_difference:,.0f}")
    print("   [CONCLUSION] A change of +1 in the central weight drastically amplifies")
    print("   high-frequency signals (edges), proving that linear filters are highly")
    print("   sensitive to central parameters.")
    print("-" * 50)
    
    # Save the absolute difference as a heatmap/image to show exactly WHAT changed
    diff_normalized = np.clip(diff_tensor, 0, 255).astype(np.uint8)
    results.append(("3_Difference_Map", diff_normalized))
    
    return results