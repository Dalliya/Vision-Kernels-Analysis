# 👁️ Vision Kernels & Neural Topology: A 21-Step Research Report

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?style=for-the-badge&logo=opencv&logoColor=white)

## 📌 Abstract
This project is a systematic deconstruction of convolutional feature extraction. We analyze 21 distinct states of image transformation, proving the transition from classical mathematical kernels to neural architectural layers.

---

## 🔬 Phase I: Edge Detection (Experiment A)

### 1. Highway: Sobel-X
<img src="data/processed_annotated/clear_highway_sobel_x.jpg" width="800">
**Conclusion:** Successfully captures vertical lane markers. Proves that first-order derivatives are ideal for extracting structured geometric primitives.

### 2. Highway: Sobel-Y
<img src="data/processed_annotated/clear_highway_sobel_y.jpg" width="800">
**Conclusion:** Isolates horizontal boundaries and horizon lines. Essential for establishing the full gradient vector field of the scene.

### 3. Highway: Laplacian
<img src="data/processed_annotated/clear_highway_laplacian.jpg" width="800">
**Conclusion:** A second-order operator acting as a skeletal map. It highlights isotropic intensity changes but shows high sensitivity to asphalt texture.

### 4. Highway: Prewitt
<img src="data/processed_annotated/clear_highway_prewitt.jpg" width="800">
**Conclusion:** Similar to Sobel but with a simpler mask. Provides a harsher, high-contrast edge response suitable for hard-contour detection.

### 5. Highway: Roberts
<img src="data/processed_annotated/clear_highway_roberts_cross.jpg" width="800">
**Conclusion:** Uses a 2x2 diagonal mask. Offers superior precision for sharp intersections and high-contrast diagonal features.

### 6. Rainy City: Sobel-X
<img src="data/processed_annotated/rainy_city_sobel_x.jpg" width="800">
**Conclusion:** Failed State. The gradient extraction is heavily corrupted by raindrops, which the kernel interprets as valid vertical features.

### 7. Rainy City: Sobel-Y
<img src="data/processed_annotated/rainy_city_sobel_y.jpg" width="800">
**Conclusion:** Failed State. Horizontal rain streaks create massive signal noise, completely obscuring vehicle lane markers.

### 8. Rainy City: Laplacian
<img src="data/processed_annotated/rainy_city_laplacian.jpg" width="800">
**Conclusion:** Maximum Entropy. The second-order derivative amplifies every single raindrop, proving that isotropic detectors are mathematically dangerous in noisy environments.

### 9. Rainy City: Prewitt
<img src="data/processed_annotated/rainy_city_prewitt.jpg" width="800">
**Conclusion:** High signal-to-noise ratio issues. Without preprocessing, the Prewitt operator creates a chaotic feature map.

### 10. Rainy City: Roberts
<img src="data/processed_annotated/rainy_city_roberts_cross.jpg" width="800">
**Conclusion:** Sharp but noisy. Captures the high-frequency "salt" of the rain, highlighting the absolute need for low-pass filtering.

---

## 🔬 Phase II: Smoothing & Restoration (Experiment B & C)

### 11. Rainy: Gaussian Blur
<img src="data/processed_annotated/rainy_city_gaussian.jpg" width="800">
**Conclusion:** Linear Smoothing. Successfully attenuates noise but "bleeds" the edges of the vehicles, causing a loss of structural resolution.

### 12. Rainy: Mean Filter
<img src="data/processed_annotated/rainy_city_mean.jpg" width="800">
**Conclusion:** Box Smoothing. Leads to significant blurring of car silhouettes, proving that uniform averaging is too aggressive for feature preservation.

### 13. Rainy: Median Filter
<img src="data/processed_annotated/rainy_city_median.jpg" width="800">
**Conclusion:** Mathematical Winner. Rank-order statistics effectively eradicate raindrops while maintaining perfectly sharp boundaries of the car geometry.

### 14. Sequential Pipeline: Restored Edges
<img src="data/processed_annotated/pipeline_sequential_pipeline.jpg" width="800">
**Conclusion:** Proves that structural recovery via Unsharp Masking after denoising allows for high-confidence feature extraction in chaotic conditions.

---

## 🔬 Phase III: Texture & CNN Dynamics (Experiment D, E & F)

### 15. Texture: Highway Gabor (0°)
<img src="data/processed_annotated/texture_highway_gabor_0.jpg" width="800">
**Conclusion:** Isolates horizontal asphalt textures. Mimics the human visual system's orientation-specific neuron responses.

### 16. Texture: Highway Gabor (Combined)
<img src="data/processed_annotated/texture_highway_combined.jpg" width="800">
**Conclusion:** Established a multi-directional texture map. Essential for surface classification and road condition analysis.

### 17. Max Pooling (2x2)
<img src="data/processed_annotated/pooling_max_pooling_2x2.jpg" width="800">
**Conclusion:** Structural compression. Successfully retains peak edge activations while reducing data dimensionality by 75%.

### 18. Average Pooling (2x2)
<img src="data/processed_annotated/pooling_average_pooling_2x2.jpg" width="800">
**Conclusion:** Signal dilution. The averaging process washes out critical edge gradients, making it inferior to Max Pooling for feature retention.

### 19. CNN: 1x1 Pointwise Convolution
<img src="data/processed_annotated/cnn_conv1x1.jpg" width="800">
**Conclusion:** Feature mixing. Successfully compressed Sobel X and Y channels into a single optimized tensor, simulating MobileNet's efficient architecture.

### 20. CNN: Dilated (Atrous) Convolution
<img src="data/processed_annotated/cnn_dilated_convolution.jpg" width="800">
**Conclusion:** Receptive field expansion. Captured larger structural context in the rainy scene without increasing the kernel's parameter count.

---

## 🔬 Phase IV: Sensitivity Metrics (Experiment G)

### 21. SAD Difference Map (The ±1 Shift)
<img src="data/processed_annotated/metrics_difference_map.jpg" width="800">
**Conclusion:** Explosive variance. A minor shift from 5 to 6 causes massive signal amplification. This validates why automated gradient descent is the only way to find stable weights.

---

## 👩‍💻 About the Author
**Dariia Zhdanova** *ML Explorer | Architect of Neural Topology*

"I specialize in deconstructing complex Deep Learning concepts down to their mathematical foundations. I believe that true engineering isn't about calling `model.fit()`, but about understanding the exact geometry of the hyperplanes we build."

"In this study, I transitioned from manual mathematical foundations to automated linear stress-testing, proving that the massive performance gap between structured silhouettes and chaotic satellite textures is the exact point where pure logic demands deeper neural connections."

📫 **Connect with me:**
* **GitHub:** [@Dalliya](https://github.com/Dalliya)
* **LinkedIn:** [Dariia Zhdanova](https://www.linkedin.com/in/dariia-z-b7146223a)