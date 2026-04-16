<div align="center">

# 👁️ Convolutional Kernels & Neural Topology for ADAS
### *A Mathematical Deconstruction of Computer Vision for Autonomous Driving*

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Data_Science-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![Machine Learning](https://img.shields.io/badge/Domain-Autonomous_Driving_&_CV-FF6F00?style=for-the-badge)](#)

*An in-depth analysis of spatial filters, transitioning from classical feature extraction to advanced architectural neural layers used in self-driving perception systems.*

---
</div>

## 📌 Executive Summary & Project Architecture
This research bridges the gap between classical Computer Vision algorithms and modern Deep Learning architectures in the context of **Autonomous Driving Systems (ADAS)**. By systematically deconstructing convolutional filters, this project explores the mathematical foundations of obstacle detection, weather-resilient noise suppression, and spatial dimensionality reduction.

The pipeline is designed as an end-to-end analytical framework: transitioning from isolated matrix convolutions (e.g., Sobel, Laplacian) to sequential image restoration pipelines, and ultimately evaluating advanced neural layers (Atrous convolutions, 1x1 Pointwise bottlenecks, and Pooling operations) used in onboard vehicle computers. Furthermore, the study introduces a rigorous mathematical sensitivity analysis, proving the necessity of automated gradient descent by quantifying filter instability through the Sum of Absolute Differences (SAD) metric.

### 📊 Experimental Setup: Dataset Modality & Tensors
To rigorously stress-test the kernels against real-world driving conditions, two diametrically opposed visual environments were selected. Multi-channel RGB arrays were compressed into single-channel (grayscale) 2D tensors to isolate structural luminance and eliminate chromatic interference.

| Image A: High-Contrast (Clear Highway) | Image B: Low-Contrast & Noisy (Rainy City) |
| :---: | :---: |
| <img src="data/raw/clear_highway.jpg" width="400"> | <img src="data/raw/rainy_city.jpg" width="400"> |
| **Topology:** Features strong geometric primitives (lane markers, horizon) and a uniform background.<br>🎯 *Acts as a baseline for ideal autopilot perception (Lane Keeping Assist).* | **Topology:** Complex details severely obscured by high-frequency stochastic noise (rain streaks).<br>🌪️ *Acts as a stress-test for noise-resilient obstacle avoidance pipelines.* |

---

## 🔬 Block 1: Feature Extraction (Edge Detection)
*Evaluating discrete derivative convolutions. To demonstrate environmental robustness, raw Edge Detection on the Clear Highway is compared against the Rainy City (after noise-removal pipeline).*

| Filter Type | High Contrast (Clear Highway) | Low Contrast (Restored Rainy City) |
| :--- | :---: | :---: |
| **Sobel X**<br>*(Vertical Edges)* | <img src="data/processed_annotated/clear_highway_sobel_x-axis.jpg" width="400"> | <img src="data/processed_annotated/pipeline_3_edges_sobel_x.jpg" width="400"> |
| **Prewitt X**<br>*(Harsher Edge Map)* | <img src="data/processed_annotated/clear_highway_prewitt_x-axis.jpg" width="400"> | <img src="data/processed_annotated/pipeline_3_edges_prewitt_x.jpg" width="400"> |
| **Roberts**<br>*(Diagonal Cross)* | <img src="data/processed_annotated/clear_highway_roberts_cross_diagonal.jpg" width="400"> | <img src="data/processed_annotated/pipeline_3_edges_roberts.jpg" width="400"> |

> **💡 Engineering Conclusion:** > * **Clear Highway:** Sobel X efficiently isolates vertical obstacles (vehicles, poles) for Time-to-Collision calculations, while Sobel Y (not shown here) excels at horizontal lane marking detection. Roberts acts as a digital scalpel, generating highly precise anchor points for OCR on traffic signs.
> * **Rainy City:** Raw derivatives fail completely on noisy data (causing "structural blindness"). However, by applying them *after* a median smoothing pipeline, we successfully extract the underlying geometry of the vehicles to prevent forward collisions.

---

## 🔬 Block 2: Smoothing & Blurring (Noise Reduction)
*Analyzing linear vs. non-linear spatial filters to process the highly noisy Rainy City tensor before passing it to the autonomous navigation stack.*

| Kernel Operation | Result on Noisy Image (Rainy City) | Analytical Commentary |
| :--- | :---: | :--- |
| **Mean Filter**<br>*(Linear Average)* | <img src="data/processed_annotated/rainy_city_mean_filter_5x5.jpg" width="250"> | Uniform averaging successfully dilutes rain noise, but aggressively blurs critical structural boundaries (car silhouettes), threatening object detection. |
| **Gaussian Blur**<br>*(Linear Weighted)* | <img src="data/processed_annotated/rainy_city_gaussian_filter_3x3.jpg" width="250"> | Provides a more natural blur by weighting central pixels higher. However, it still fails to isolate the "sharp" stochastic noise of rain droplets. |
| **Median Filter**<br>*(Non-Linear)* | <img src="data/processed_annotated/rainy_city_median_filter_5x5.jpg" width="250"> | **🏆 Optimal Approach.** Rank-order statistics effectively eradicate impulsive rain noise while perfectly maintaining hard geometric boundaries. The gold standard for ADAS pre-processing. |

---

## 🔬 Block 3: Sharpening (Detail Recovery)
*Applying Unsharp Masking to recover frequencies lost during the smoothing phase.*

<div align="center">
  <img src="data/processed_annotated/pipeline_2_sharpened_unsharp_mask.jpg" width="600">
</div>

> **💡 Engineering Conclusion:** By subtracting a blurred version of the image from the original (Unsharp Masking), we successfully enhance the high-frequency components of the cars. This non-linear restoration is critical for preparing data for deeper network layers.

---

## 🔬 Block 4: Specialized Filters & CNN Topologies
*Evaluating domain-specific kernels and modern architectural convolution techniques used in networks like DeepLab and MobileNet.*

### 1. Gabor Filter Bank (Drivable Area Segmentation)
| High Contrast (Highway) | Low Contrast (Rainy City) |
| :---: | :---: |
| <img src="data/processed_annotated/texture_highway_gabor_bank_combined.jpg" width="400"> | <img src="data/processed_annotated/texture_rainy_gabor_bank_combined.jpg" width="400"> |

> **Conclusion:** The Gabor bank successfully isolates orientation-specific textures (e.g., asphalt grain vs. sky). In autonomous driving, this spectral analysis is crucial for Free Space Segmentation, allowing the vehicle to differentiate smooth roads from hazardous terrain.

### 2. Architectural Kernels (Dilated & 1x1 Pointwise)
<div align="center">
  <img src="data/processed_annotated/cnn_highway_2_dilated_conv_rate2.jpg" width="400">
  <img src="data/processed_annotated/cnn_highway_1_conv1x1_mixed_edges.jpg" width="400">
</div>

> **Conclusion:** **Dilated (Atrous) filters** artificially expand the receptive field without increasing computational load, allowing the AI to understand broader contexts (e.g., a massive truck vs. a wall). **1x1 Convolutions** successfully mix depthwise feature maps (Sobel X and Y) into a compressed tensor, mimicking the extreme computational efficiency required for onboard vehicle processors.

---

## 🔬 Block 5: Spatial Reduction (Pooling)
*Simulating CNN dimensionality reduction to achieve translational invariance.*

| Original Edges | Max Pooling (2x2) | Average Pooling (2x2) |
| :---: | :---: | :---: |
| <img src="data/processed_annotated/pooling_highway_0_original_edges_before_pooling.jpg" width="250"> | <img src="data/processed_annotated/pooling_highway_1_max_pooled_2x2.jpg" width="250"> | <img src="data/processed_annotated/pooling_highway_2_average_pooled_2x2.jpg" width="250"> |

> **💡 Engineering Conclusion:** **Max Pooling** (non-linear) acts as a strict significance filter. It discards background noise and passes only the strongest activations (brightest contours) to the next neural layers. This ensures translational invariance—the self-driving network recognizes the vehicle regardless of its exact pixel coordinates. **Average Pooling** dilutes the signal, leading to dangerous anti-aliasing of lane markings.

---

## 🔬 Block 6: Metrics & Sensitivity Analysis (Safety Critical Test)
*Comprehensive stability analysis within the ±1 parameter range. This experiment quantifies how minor variations in hard-coded kernel weights affect the safety of the output signal.*

### 🖼️ Parameter Shift Comparison (Rainy City)
| Shift -1 (Weight = 4) | **Base (Weight = 5)** | Shift +1 (Weight = 6) |
| :---: | :---: | :---: |
| <img src="data/processed_annotated/metrics_rainy_shifted_sharpen_4.jpg" width="260"> | <img src="data/processed_annotated/metrics_rainy_1_base_sharpen_5.jpg" width="260"> | <img src="data/processed_annotated/metrics_rainy_2_shifted_sharpen_6.jpg" width="260"> |
| *Signal degradation & "Structural Blindness"* | *Optimal feature balance* | *Phantom Braking risk (Noise explosion)* |

<br>

<div align="center">
  <b>Visualizing the Difference (SAD Map: +1 Shift vs Base)</b><br>
  <img src="data/processed_annotated/metrics_rainy_3_difference_map.jpg" width="600">
</div>

### 📊 Metric Evaluation: Sum of Absolute Differences (SAD)
To quantify the impact of these bidirectional shifts, we calculate the pixel-wise divergence from the base state:

$$SAD = \sum_{i,j} |I_{Base}(i,j) - I_{Shifted}(i,j)|$$

> **💡 Final Conclusion on Parameter Efficiency:**
> The experiment demonstrates that convolutional kernels are hypersensitive to manual weight adjustments:
> 1. **Shift -1:** Leads to a significant drop in edge activation. In real-world conditions, the autopilot simply "won't see" thin objects like pedestrians or distant traffic signs.
> 2. **Shift +1:** Results in explosive high-frequency noise amplification (visible in the SAD Difference Map). Radar and cameras would transmit thousands of false micro-obstacles, triggering a dangerous Phantom Braking scenario.
>
> **Project Thesis:** The massive SAD values recorded for both $+1$ and $-1$ shifts mathematically prove why manual crafting of convolutional kernels (Classical CV) is volatile and completely non-scalable for Safety-Critical Systems. It validates the industry's transition to **Deep Learning (CNNs)**, where neural weights are not hard-coded, but dynamically optimized via Gradient Descent to find the most stable local minima for any weather condition.

---

---
<div align="center">

### 👩‍💻 Dariia Zhdanova
**ML Developer | Architect of Neural Topology**

*I specialize in deconstructing complex Deep Learning concepts down to their mathematical foundations. True engineering isn't about calling `model.fit()`, but about understanding the exact geometry of the hyperplanes we build.*

</div>

**Project Thesis:** "In this study, I transitioned from manual mathematical foundations to automated linear stress-testing. This research proves that the massive performance gap between structured silhouettes and chaotic textures is the exact point where pure logic demands deeper neural connections."

<div align="center">
<br>

📫 **Connect:** &nbsp; [LinkedIn](https://www.linkedin.com/in/dariia-z-b7146223a) &nbsp;|&nbsp; [GitHub (@Dalliya)](https://github.com/Dalliya)

</div>
