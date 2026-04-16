<div align="center">

# 👁️ Convolutional Kernels & Neural Topology
### *A Mathematical and Empirical Deconstruction of Computer Vision*

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Data_Science-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![Machine Learning](https://img.shields.io/badge/Domain-Computer_Vision-FF6F00?style=for-the-badge)](#)

*An in-depth analysis of linear and non-linear spatial filters, transitioning from classical mathematical feature extraction to advanced architectural neural layers.*

---
</div>

## 📌 Executive Summary & Project Architecture
This research bridges the gap between classical Computer Vision algorithms and modern Deep Learning architectures. By systematically deconstructing convolutional filters using **OpenCV**, this project explores the mathematical foundations of feature extraction, non-linear noise suppression, and spatial dimensionality reduction. 

The pipeline is designed as an end-to-end analytical framework: transitioning from isolated matrix convolutions (e.g., Sobel, Laplacian) to sequential image restoration pipelines, and ultimately evaluating advanced neural layers (Atrous convolutions, 1x1 Pointwise bottlenecks, and Pooling operations). Furthermore, the study introduces a rigorous mathematical sensitivity analysis, proving the necessity of automated gradient descent by quantifying filter instability through the Sum of Absolute Differences (SAD) metric.

### 📊 Experimental Setup: Dataset Modality & Tensors
To rigorously stress-test the kernels against real-world variance, two diametrically opposed visual environments were selected. Upon loading, dimensional configurations were logged, and multi-channel RGB arrays were compressed into single-channel (grayscale) 2D tensors. This preprocessing is crucial to isolate structural luminance and eliminate chromatic interference during matrix multiplication.

| Image A: High-Contrast (Clear Highway) | Image B: Low-Contrast & Noisy (Rainy City) |
| :---: | :---: |
| <img src="data/raw/clear_highway.jpg" width="400"> | <img src="data/raw/rainy_city.jpg" width="400"> |
| **Topology:** Features strong, predictable geometric primitives (lane markers, strict horizon) and a uniform background.<br>🎯 *Acts as a baseline for ideal kernel activation.* | **Topology:** Features complex, clustered details severely obscured by extreme high-frequency stochastic noise (rain streaks).<br>🌪️ *Acts as a stress-test for noise-resilient pipelines.* |

---

## 🔬 Block 1: Feature Extraction (Edge Detection)
*Evaluating discrete derivative convolutions. To demonstrate environmental robustness, raw Edge Detection on the Clear Highway is compared against Edge Detection applied to the Rainy City (after noise-removal pipeline).*

| Filter Type | High Contrast (Clear Highway) | Low Contrast (Restored Rainy City) |
| :--- | :---: | :---: |
| **Sobel X**<br>*(Horizontal Gradient)* | <img src="data/processed_annotated/clear_highway_sobel_x-axis.jpg" width="400"> | <img src="data/processed_annotated/pipeline_3_edges_sobel_x.jpg" width="400"> |
| **Prewitt X**<br>*(Harsher Edge Map)* | <img src="data/processed_annotated/clear_highway_prewitt_x-axis.jpg" width="400"> | <img src="data/processed_annotated/pipeline_3_edges_prewitt_x.jpg" width="400"> |
| **Roberts**<br>*(Diagonal Cross)* | <img src="data/processed_annotated/clear_highway_roberts_cross_diagonal.jpg" width="400"> | <img src="data/processed_annotated/pipeline_3_edges_roberts.jpg" width="400"> |

> **💡 Engineering Conclusion:** > * **Clear Highway:** Linear derivative filters (Sobel, Prewitt) efficiently extract structural boundaries. Roberts excels at pinpointing high-contrast intersections.
> * **Rainy City:** Raw derivatives fail completely on noisy data (amplifying rain). However, by applying them *after* a smoothing pipeline, we successfully extract the underlying geometry of the vehicles.

---

## 🔬 Block 2: Smoothing & Blurring (Noise Reduction)
*Analyzing linear vs. non-linear spatial filters to process the highly noisy Rainy City tensor.*

| Kernel Operation | Result on Noisy Image (Rainy City) | Analytical Commentary |
| :--- | :---: | :--- |
| **Mean Filter**<br>*(Linear Average)* | <img src="data/processed_annotated/rainy_city_mean_filter_5x5.jpg" width="250"> | Uniform averaging successfully dilutes rain noise, but aggressively blurs critical structural boundaries (car silhouettes). |
| **Gaussian Blur**<br>*(Linear Weighted)* | <img src="data/processed_annotated/rainy_city_gaussian_filter_3x3.jpg" width="250"> | Provides a more natural blur by weighting central pixels higher. However, it still fails to preserve sharp edges due to its linear interpolation. |
| **Median Filter**<br>*(Non-Linear)* | <img src="data/processed_annotated/rainy_city_median_filter_5x5.jpg" width="250"> | **🏆 Optimal Approach.** Rank-order statistics effectively eradicate "salt-and-pepper" high-frequency rain noise while perfectly maintaining hard geometric boundaries. |

---

## 🔬 Block 3: Sharpening (Detail Recovery)
*Applying Unsharp Masking to recover frequencies lost during the smoothing phase.*

<div align="center">
  <img src="data/processed_annotated/pipeline_2_sharpened_unsharp_mask.jpg" width="600">
</div>

> **💡 Engineering Conclusion:** By subtracting a blurred version of the image from the original (Unsharp Masking), we successfully enhance the high-frequency components of the cars. This non-linear restoration is critical for preparing data for deeper network layers.

---

## 🔬 Block 4: Specialized Filters (Texture & Advanced CNNs)
*Evaluating domain-specific kernels and modern architectural convolution techniques.*

### 1. Gabor Filter Bank (Texture & Frequency Analysis)
| High Contrast (Highway) | Low Contrast (Rainy City) |
| :---: | :---: |
| <img src="data/processed_annotated/texture_highway_gabor_bank_combined.jpg" width="400"> | <img src="data/processed_annotated/texture_rainy_gabor_bank_combined.jpg" width="400"> |

> **Conclusion:** The Gabor bank successfully isolates orientation-specific textures (e.g., asphalt grain), functioning similarly to the human visual cortex.

### 2. Architectural Kernels (Dilated & 1x1 Pointwise)
<div align="center">
  <img src="data/processed_annotated/cnn_2_dilated_conv_rate2.jpg" width="400">
  <img src="data/processed_annotated/cnn_1_conv1x1_mixed_edges.jpg" width="400">
</div>

> **Conclusion:** **Dilated (Atrous) filters** expand the receptive field without increasing computational parameters, capturing broader context. **1x1 Convolutions** successfully mix depthwise feature maps (Sobel X and Y) into a compressed tensor, mimicking MobileNet efficiency.

---

## 🔬 Block 5: Spatial Reduction (Pooling)
*Simulating CNN dimensionality reduction on extracted feature maps.*

| Original Edges | Max Pooling (2x2) | Average Pooling (2x2) |
| :---: | :---: | :---: |
| <img src="data/processed_annotated/pooling_0_original_edges_before_pooling.jpg" width="250"> | <img src="data/processed_annotated/pooling_1_max_pooled_2x2.jpg" width="250"> | <img src="data/processed_annotated/pooling_2_average_pooled_2x2.jpg" width="250"> |

> **💡 Engineering Conclusion:** **Max Pooling** (non-linear) is vastly superior for feature retention, preserving peak edge activations while compressing spatial dimensions. **Average Pooling** (linear) dilutes the signal, resulting in a washed-out feature map.

---

## 🔬 Block 6: Metrics & Sensitivity Analysis
*Comprehensive stability analysis within the $\pm 1$ parameter range. This experiment quantifies how minor variations in kernel weights affect the stochastic properties of the output signal.*

### 🖼️ Parameter Shift Comparison
| Shift -1 (Weight = 4) | **Base (Weight = 5)** | Shift +1 (Weight = 6) |
| :---: | :---: | :---: |
| <img src="data/processed_annotated/metrics_shifted_sharpen_4.jpg" width="260"> | <img src="data/processed_annotated/metrics_1_base_sharpen_5.jpg" width="260"> | <img src="data/processed_annotated/metrics_2_shifted_sharpen_6.jpg" width="260"> |
| *Signal degradation & Detail loss* | *Optimal feature balance* | *High-frequency noise amplification* |

<br>

<div align="center">
  <b>Visualizing the Difference (SAD Map: +1 Shift vs Base)</b><br>
  <img src="data/processed_annotated/metrics_3_difference_map.jpg" width="600">
</div>

### 📊 Metric Evaluation: Sum of Absolute Differences (SAD)
To quantify the impact of these bidirectional shifts, we calculate the pixel-wise divergence from the base state:

$$SAD = \sum_{i,j} |I_{Base}(i,j) - I_{Shifted}(i,j)|$$

> **💡 Final Conclusion on Parameter Efficiency:**
> The experiment demonstrates that convolutional kernels are hypersensitive to manual weight adjustments:
> 1. **Shift -1:** Leads to a significant drop in edge activation, effectively "softening" the image beyond the intended sharpening effect and losing critical high-frequency data.
> 2. **Shift +1:** Results in explosive high-frequency noise amplification (visible in the SAD Difference Map), making the image unstable for subsequent analysis.
>
> **Project Thesis:** The massive SAD values recorded for both $+1$ and $-1$ shifts prove why manual crafting of convolutional kernels (Classical CV) is mathematically volatile and non-scalable for complex environments. It validates the necessity of Automated Gradient Descent (Deep Learning) as the only reliable method to discover stable, generalized architectural weights.


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
