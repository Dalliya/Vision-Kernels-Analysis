import sys
import os
import cv2
import numpy as np

# Append project root to sys.path for absolute imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.utils.image_io import load_and_convert_image
from src.utils.visuals import annotate_image
from src.experiments import (
    run_edge_detection_experiment, 
    run_smoothing_experiment,
    run_sequential_pipeline,
    run_texture_analysis_experiment,
    run_pooling_experiment,
    run_advanced_cnn_experiment,
    run_metrics_and_parameter_shift
)

def run_pipeline(cfg: Config) -> None:
    print("[SYSTEM] Pipeline execution started.\n")
    
    # 1. Create directory for annotated results
    output_dir = os.path.join(os.path.dirname(cfg.IMG_HIGH_CONTRAST), "..", "processed_annotated")
    os.makedirs(output_dir, exist_ok=True) 
    
    # 2. Load input tensors
    tensor_high_contrast = load_and_convert_image(cfg.IMG_HIGH_CONTRAST)
    tensor_low_contrast = load_and_convert_image(cfg.IMG_LOW_CONTRAST)
    
    # =========================================================================
    # EXPERIMENT A: Edge Detection
    # =========================================================================
    print("[SYSTEM] Processing Experiment A...")
    results_a = run_edge_detection_experiment(tensor_high_contrast)
    for filter_name, img_tensor in results_a:
        annotated = annotate_image(img_tensor, filter_name)
        safe_name = filter_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        cv2.imwrite(os.path.join(output_dir, f"clear_highway_{safe_name}.jpg"), annotated)
    
    # =========================================================================
    # EXPERIMENT B: Smoothing
    # =========================================================================
    print("[SYSTEM] Processing Experiment B...")
    
    # 1. Processing Low Contrast (Rainy City)
    results_b_low = run_smoothing_experiment(tensor_low_contrast)
    for filter_name, img_tensor in results_b_low:
        annotated = annotate_image(img_tensor, filter_name)
        safe_name = filter_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        cv2.imwrite(os.path.join(output_dir, f"rainy_city_{safe_name}.jpg"), annotated)
        
    # 2. Processing High Contrast (Clear Highway)
    results_b_high = run_smoothing_experiment(tensor_high_contrast)
    for filter_name, img_tensor in results_b_high:
        annotated = annotate_image(img_tensor, filter_name)
        safe_name = filter_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        cv2.imwrite(os.path.join(output_dir, f"clear_highway_{safe_name}.jpg"), annotated)

    # =========================================================================
    # EXPERIMENT C: Sequential Pipeline
    # =========================================================================
    print("[SYSTEM] Processing Experiment C...")
    results_c = run_sequential_pipeline(tensor_low_contrast)
    for filter_name, img_tensor in results_c:
        annotated = annotate_image(img_tensor, filter_name)
        safe_name = filter_name.lower().replace(" ", "_")
        cv2.imwrite(os.path.join(output_dir, f"pipeline_{safe_name}.jpg"), annotated)

    # =========================================================================
    # EXPERIMENT D: Texture Analysis
    # =========================================================================
    print("[SYSTEM] Running Texture Analysis...")
    res_d_high = run_texture_analysis_experiment(tensor_high_contrast)
    for filter_name, img_tensor in res_d_high:
        annotated = annotate_image(img_tensor, filter_name)
        cv2.imwrite(os.path.join(output_dir, f"texture_highway_{filter_name.lower()}.jpg"), annotated)

    res_d_low = run_texture_analysis_experiment(tensor_low_contrast)
    for filter_name, img_tensor in res_d_low:
        annotated = annotate_image(img_tensor, filter_name)
        cv2.imwrite(os.path.join(output_dir, f"texture_rainy_{filter_name.lower()}.jpg"), annotated)

    # =========================================================================
    # EXPERIMENT E: Pooling
    # =========================================================================
    print("[SYSTEM] Running Spatial Reduction (Pooling)...")
    
    # 1. Processing High Contrast (Clear Highway)
    results_e_high = run_pooling_experiment(tensor_high_contrast)
    for filter_name, img_tensor in results_e_high:
        annotated = annotate_image(img_tensor, filter_name)
        cv2.imwrite(os.path.join(output_dir, f"pooling_highway_{filter_name.lower()}.jpg"), annotated)
        
    # 2. Processing Low Contrast (Rainy City)
    results_e_low = run_pooling_experiment(tensor_low_contrast)
    for filter_name, img_tensor in results_e_low:
        annotated = annotate_image(img_tensor, filter_name)
        cv2.imwrite(os.path.join(output_dir, f"pooling_rainy_{filter_name.lower()}.jpg"), annotated)

    # =========================================================================
    # EXPERIMENT F: Advanced CNN Layers
    # =========================================================================
    print("[SYSTEM] Running Advanced CNN Architectural Layers...")
    
    # 1. Processing High Contrast (Clear Highway)
    results_f_high = run_advanced_cnn_experiment(tensor_high_contrast)
    for filter_name, img_tensor in results_f_high:
        annotated = annotate_image(img_tensor, filter_name)
        cv2.imwrite(os.path.join(output_dir, f"cnn_highway_{filter_name.lower()}.jpg"), annotated)

    # 2. Processing Low Contrast (Rainy City)
    results_f_low = run_advanced_cnn_experiment(tensor_low_contrast)
    for filter_name, img_tensor in results_f_low:
        annotated = annotate_image(img_tensor, filter_name)
        cv2.imwrite(os.path.join(output_dir, f"cnn_rainy_{filter_name.lower()}.jpg"), annotated)

    # =========================================================================
    # EXPERIMENT G: Parameter Sensitivity & SAD Metrics
    # =========================================================================
    print("[SYSTEM] Running Sensitivity Analysis (Metrics)...")
    
    # 1. Processing High Contrast (Clear Highway)
    results_g_high = run_metrics_and_parameter_shift(tensor_high_contrast)
    for filter_name, img_tensor in results_g_high:
        annotated = annotate_image(img_tensor, filter_name)
        cv2.imwrite(os.path.join(output_dir, f"metrics_highway_{filter_name.lower()}.jpg"), annotated)

    # 2. Processing Low Contrast (Rainy City)
    results_g_low = run_metrics_and_parameter_shift(tensor_low_contrast)
    for filter_name, img_tensor in results_g_low:
        annotated = annotate_image(img_tensor, filter_name)
        cv2.imwrite(os.path.join(output_dir, f"metrics_rainy_{filter_name.lower()}.jpg"), annotated)

    # =========================================================================
    # PIPELINE COMPLETE
    # =========================================================================
    print("\n" + "="*80)
    print("[SUCCESS] All 7 Experiments completed successfully!")
    print(f"[SUCCESS] Results saved in: {os.path.abspath(output_dir)}")
    print("="*80 + "\n")

if __name__ == "__main__":
    project_config = Config()
    run_pipeline(project_config)