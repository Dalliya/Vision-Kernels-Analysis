👁️ Vision Kernels & Neural Topology: A Comparative Study of Convolutional Feature Extraction
📌 Project Overview
This repository implements an advanced research laboratory for deconstructing Digital Image Processing and Convolutional Neural Network (CNN) layers. The core objective is to analyze how specific mathematical kernels react to different data topologies: the structured geometry of a Clear Highway versus the stochastic, high-frequency noise of a Rainy City.

The project transitions from classical discrete differentiation to modern architectural layers, concluding with a rigorous Parameter Sensitivity Analysis evaluating linear system volatility.

🔬 Experimental Breakdown & Technical Analysis
🔹 Block I: Gradient Feature Extraction (Edge Detection)

Kernels: Sobel (X/Y), Prewitt, Roberts, and Laplacian operators.

Analysis: These filters approximate the partial derivatives of the image intensity. Sobel and Prewitt are highly effective for directional features, while Laplacian (2nd order) is an isotropic blob detector.

Conclusion: Structured environments allow for high-confidence gradient extraction. However, raw derivative filters are catastrophic in noisy environments, where stochastic point-anomalies (rain) are misinterpreted as valid edges.

🔹 Block II: Stochastic Noise Suppression (Smoothing)

Kernels: Gaussian, Mean (Box), and Median filters.

Analysis: Linear kernels (Gaussian) attenuate noise but blur structural boundaries. The Median Filter uses non-linear rank-order statistics to eradicate "salt-and-pepper" noise while preserving pristine edges.

Conclusion: For autonomous systems in adverse weather, non-linear filtering is mathematically superior to linear blurring for edge preservation.

🔹 Block III: Hierarchical Restoration Pipeline

Logic: A sequential chain: Median Denoising -> Unsharp Masking -> Edge Extraction.

Conclusion: This demonstrates Hierarchical Feature Recovery. We prove that high-frequency data lost during denoising can be recovered via Unsharp Masking before final feature extraction.

🔹 Block IV: Texture Analysis (Gabor Filter Banks)

Logic: Directional wavelets designed to mimic the human primary visual cortex (V1).

Conclusion: Gabor filters elegantly segment road textures but trigger chaotic responses on rain droplets, proving that point-anomalies contain all spatial frequencies simultaneously.

🔹 Block V: Spatial Reduction (CNN Pooling)

Logic: Comparing Max Pooling (peak activation) vs. Average Pooling (local mean).

Conclusion: Max Pooling is the industry standard because it retains the strongest feature signals (edges) during a 75% dimensionality reduction, whereas Average Pooling dilutes critical feature responses.

🔹 Block VI: Advanced CNN Topology

Logic: Pointwise (1x1) convolutions and Dilated (Atrous) kernels.

Conclusion: 1x1 convolutions act as optimized channel mixers, while Dilated kernels successfully expand the receptive field to capture larger structures without resolution loss.

🔹 Block VII: Parameter Sensitivity (The ±1 Stress Test)

The Experiment: We shifted the central weight of a Sharpen kernel from 5 to 6 (+1 shift).

Metrics: Sum of Absolute Differences (SAD).

Conclusion: A minor weight shift causes explosive signal variance. This mathematically validates why automated gradient descent is mandatory: manual weight tuning in high-dimensional space is fundamentally unstable.

📸 Research Results: Visual Gallery
1. Edge Extraction Stability (Highway vs. Rain)

Dataset	Filter: Sobel-X	Filter: Laplacian
Clear Highway		
Rainy City		
2. Denoising & Noise Restoration

Gaussian Blur (Linear)	Median Filter (Non-Linear)	Pipeline: Restored Edges
3. CNN Architecture & Downsampling

Max Pooling (2x2)	Average Pooling (2x2)	Dilated (Atrous) Conv
4. Parameter Volatility (SAD Metric Map)

Base Sharpen (W=5)	Shifted Sharpen (W=6)	SAD Difference Map
🚀 Replicating the Study
1. Clone the environment:

Bash
git clone https://github.com/Dalliya/Vision-Kernels-Analysis.git
cd Vision-Kernels-Analysis
pip install numpy opencv-python pillow
2. Execute the Research Engine:

Bash
python src/main.py
Annotations are rendered using an anti-aliased PIL engine for maximum legibility.

📂 Project Structure
Plaintext
├── data/
│   ├── raw/                  # Source Input Tensors
│   └── processed_annotated/  # Final Annotated Reports
├── src/
│   ├── filters/              # OOP Kernel Classes
│   ├── utils/
│   │   ├── visuals.py        # Professional UI Annotation Engine
│   │   └── image_io.py       # Load/Save Wrappers
│   ├── main.py               # Orchestrator
│   └── experiments.py        # Research Logics (A-G)
└── README.md

👩‍💻 About the Author
Dariia Zhdanova (@Dalliya)
Architect of Neural Topology | ML Researcher

I specialize in deconstructing complex Deep Learning concepts down to their mathematical foundations. I believe that true engineering isn't about calling model.fit(), but about understanding the exact geometry of the feature spaces we build.

📫 Connect: GitHub | LinkedIn