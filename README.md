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

### 🧠 Deep Mathematical & ADAS Analysis

**1. Sobel Operator (Spatial Semantic Decomposition)**
In autonomous driving, splitting gradients into X and Y axes is not just a mathematical formality; it is a fundamental spatial decomposition for two distinct safety systems:
* **Sobel X (Collision Avoidance):** Scans for horizontal changes in luminance to isolate vertical boundaries (e.g., vehicle sides, poles). On the *Clear Highway*, it flawlessly extracted the leading car's geometry, providing critical data for **Time-to-Collision (TTC)** calculations. However, in the *Rainy City*, the filter suffers from "structural blindness," generating thousands of false micro-gradients from rain streaks.
* **Sobel Y (Lane Keeping Assist):** Scans vertical changes to analyze the drivable surface. In ideal conditions, it perfectly extracts lane markings and the horizon line, which are essential for centering the vehicle's motion vector.

**2. Prewitt Filter (Hard-Edge Structural Mapping)**
Unlike Sobel, the Prewitt operator utilizes a uniform weight matrix (lacking central pixel amplification). This makes its edge response much "harsher" but strips away any natural noise resistance.
* **Clear Highway:** Acts as a high-precision structural scanner. It provides aggressive, high-contrast boundaries ideal for generating millimeter-accurate **Bounding Boxes** around obstacles.
* **Rainy City:** The lack of built-in weight smoothing plays a fatal role. Prewitt hyperbolizes residual high-frequency rain noise, causing the vehicle's silhouette to dissolve into a chaotic texture. In engineering terms, this causes **Algorithm Noise Overload**, leading to a complete failure of the detection system.

**3. Roberts Cross (High-Frequency Localization)**
Employing a microscopic 2x2 matrix, this filter calculates diagonal spatial derivatives. It offers extreme spatial localization but lacks mathematical noise-smoothing mechanisms entirely.
* **Clear Highway:** Acts as a "digital scalpel." Instead of thick lines, it pinpoints sharp corners and geometric intersections. These isolated activations serve as **Anchor Points** for 3D model alignment and Optical Character Recognition (OCR) on traffic signs.
* **Rainy City:** The linear logic of a 2x2 matrix collapses under stochastic noise. The filter reacts to every micro-fluctuation (droplets, lens glare, sensor noise), scattering the image into a "starry sky" of isolated pixels. For an autopilot, this is the most dangerous scenario, directly triggering **Phantom Braking** as the system falsely detects thousands of micro-obstacles.

> **💡 Engineering Conclusion for Feature Extraction:**
> Classical derivative kernels perfectly demonstrate the engineering trade-off between edge sharpness and signal robustness. While matrices like Prewitt and Roberts are excellent for laboratory conditions and OCR, their hard-coded linear nature makes them mathematically volatile in stochastic environments. Safe real-world navigation requires dynamic scaling of the receptive field—proving why modern ADAS relies on Convolutional Neural Networks (CNNs) to dynamically adapt weights rather than relying on static differential math.

---

---

## 🔬 Block 2: Smoothing & Blurring (Noise Reduction)
*Analyzing linear vs. non-linear spatial filters to process sensor data in diametrically opposed driving conditions (Ideal Baseline vs. Stochastic Stress-Test).*

| Kernel Operation & Deep ADAS Analysis | High Contrast (Clear Highway) | Low Contrast (Rainy City) |
| :--- | :---: | :---: |
| **Mean Filter**<br>*(Linear Average)*<br><br>**Rejected for ADAS.** Linear averaging acts as a low-pass frequency eraser. While it "dissolves" rain streaks, it simultaneously erodes the high-frequency structural data of objects. Losing edge sharpness is equivalent to losing the object itself in the navigation stack. | <img src="data/processed_annotated/clear_highway_mean_filter.jpg" width="500"> | <img src="data/processed_annotated/rainy_city_mean_filter_5x5.jpg" width="500"> |
| **Gaussian Blur**<br>*(Linear Weighted)*<br><br>**Weighted Smoothing.** Provides a natural distribution, concentrating on the central pixel. Excellent for reducing sensor white noise on a highway, but mathematically insufficient to isolate sharp impulsive noise like rain streaks or lens glare. | <img src="data/processed_annotated/clear_highway_gaussian_filter.jpg" width="500"> | <img src="data/processed_annotated/rainy_city_gaussian_filter_3x3.jpg" width="500"> |
| **Median Filter**<br>*(Non-Linear)*<br><br>**🏆 Optimal ADAS Solution.** Unlike averaging, this kernel uses rank statistics. By selecting the median value, it treats rain droplets as outliers and effectively eradicates them while **perfectly maintaining mathematical edge sharpness**. This "Edge-Preserving" property is the industry standard for low-visibility pre-processing. | <img src="data/processed_annotated/clear_highway_median_filter.jpg" width="500"> | <img src="data/processed_annotated/rainy_city_median_filter_5x5.jpg" width="500"> |
### 🧠 Deep Comparative Analysis: Linear Pitfalls vs. Non-Linear Resilience

In a professional computer vision pipeline for autonomous vehicles, noise suppression is the critical stage before feature extraction. My experiment reveals a fundamental divergence between mathematical approaches:

**1. The "Structural Erosion" Problem (Clear Highway)**
On the *Clear Highway*, noise is minimal, but structural precision is vital for **Lane Keeping Assist (LKA)**. 
* **Linear Failure:** Mean and Gaussian filters blur the boundaries of lane markings and the horizon. In a high-speed scenario, this loss of precision can lead to lateral drift errors.
* **Non-Linear Precision:** The Median filter preserves the exact pixel-perfect edges of the highway markers, proving that even in "clean" conditions, non-linear processing provides superior data for geometry-based navigation.

**2. Stochastic Noise Overload (Rainy City)**
In the *Rainy City* scenario, the sensor is bombarded with high-frequency impulsive noise (rain streaks).
* **The "Ghosting" Effect:** Linear filters attempt to "average out" the rain, but they end up creating a "ghostly" blur where the car silhouettes dissolve into the background. For an autopilot, this creates a **False Negative** risk — a car is physically there, but its mathematical signature is too weak to trigger a braking response.
* **The Outlier Elimination:** The Median filter treats every rain streak as a mathematical outlier. Since rain occupies a small portion of the 5x5 local window compared to the solid object (the car), the rank-order statistic effectively "deletes" the rain from the frame, restoring a clear view of the urban obstacles.

> **💡 Engineering Conclusion:** > The **Median Filter** is the definitive winner for ADAS applications. Its ability to maintain structural integrity while eradicating stochastic noise makes it the primary tool for "weather-proofing" an autopilot's vision. My research proves that **Edge-Preserving Smoothing** is not just an aesthetic choice, but a safety-critical requirement for autonomous navigation.

---
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
