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
    """A flexible sharpen filter used to evaluate parameter sensitivity (±1 shift)."""
    def __init__(self, center_weight: float):
        # Dynamically define the kernel by adjusting the central pixel weight
        kernel = np.array([
            [0, -1, 0], 
            [-1, center_weight, -1], 
            [0, -1, 0]
        ], dtype=np.float32)
        super().__init__(kernel=kernel, name=f"Custom_Sharpen_Center_{center_weight}")

def run_metrics_and_parameter_shift(image_tensor: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    """
    EXPERIMENT G: Parameter Shift & Mathematical Metrics.
    Evaluates filter sensitivity by shifting a central parameter by ±1 
    and calculating the pixel-wise Sum of Absolute Differences (SAD).
    """
    print("\n[EXPERIMENT] Running Parameter Shift (±1) and Sensitivity Metrics...")
    results = []
    
    # 1. Shifted Filter -1 (Center weight = 4)
    print("   -> Applying Shifted Filter (Center = 4, decreased by 1)...")
    filter_minus = CustomSharpen(center_weight=4)
    img_minus = filter_minus.apply(image_tensor)
    results.append(("shifted_sharpen_4", img_minus))
    
    # 2. Base Filter (Center weight = 5)
    print("   -> Applying Base Filter (Baseline, Center = 5)...")
    filter_base = CustomSharpen(center_weight=5)
    img_base = filter_base.apply(image_tensor)
    results.append(("1_base_sharpen_5", img_base))
    
    # 3. Shifted Filter +1 (Center weight = 6)
    print("   -> Applying Shifted Filter (Center = 6, increased by 1)...")
    filter_plus = CustomSharpen(center_weight=6)
    img_plus = filter_plus.apply(image_tensor)
    results.append(("2_shifted_sharpen_6", img_plus))
    
    # 4. METRIC CALCULATION: Sum of Absolute Differences (SAD)
    # Compare both -1 and +1 shifts against the baseline image
    diff_minus = np.abs(img_base.astype(np.float32) - img_minus.astype(np.float32))
    diff_plus = np.abs(img_base.astype(np.float32) - img_plus.astype(np.float32))
    
    total_diff_minus = np.sum(diff_minus)
    total_diff_plus = np.sum(diff_plus)
    
    print("-" * 50)
    print(f"   [METRIC -1] Total Pixel-wise Difference (Base vs 4): {total_diff_minus:,.0f}")
    print(f"   [METRIC +1] Total Pixel-wise Difference (Base vs 6): {total_diff_plus:,.0f}")
    print("   [CONCLUSION] Massive SAD values prove the hypersensitivity of linear filters.")
    print("-" * 50)
    
    # 5. Generate and save the Difference Map (for README visualization)
    diff_normalized = np.clip(diff_plus, 0, 255).astype(np.uint8)
    results.append(("3_difference_map", diff_normalized))
    
    # CRITICAL: Return the collected results to the main pipeline runner
    return results