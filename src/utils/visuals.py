import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import textwrap

def get_professional_description(filter_name: str) -> str:
    """
    Academic-level technical descriptions of each convolution kernel.
    Focuses on the transformation of the pixel intensity distribution.
    """
    n = filter_name.lower()
    
    # Feature Extraction (Edges)
    if "sobel" in n and "x" in n: 
        return "FILTER: SOBEL-X. Approximates the vertical derivative of the image intensity. It effectively isolates vertical edges by computing horizontal intensity gradients."
    if "sobel" in n and "y" in n: 
        return "FILTER: SOBEL-Y. Approximates the horizontal derivative of the image intensity. It specializes in extracting horizontal edges and lane markings."
    if "laplacian" in n: 
        return "FILTER: LAPLACIAN. A second-order isotropic derivative operator. It identifies regions of rapid intensity change (blobs/edges) but is sensitive to noise."
    if "prewitt" in n: 
        return "FILTER: PREWITT. Utilizes a discrete differentiation mask to calculate gradient magnitude. Provides sharp, high-contrast edge responses for structural mapping."
    if "roberts" in n: 
        return "FILTER: ROBERTS CROSS. Employs 2x2 differential operators to calculate diagonal gradients. Highly effective for pinpointing sharp, high-contrast intersections."

    # Frequency Domain (Smoothing)
    if "mean" in n: 
        return "FILTER: MEAN (BOX). Performs local spatial averaging. While reducing high-frequency noise, it attenuates signal sharpness, leading to structural degradation."
    if "gaussian" in n: 
        return "FILTER: GAUSSIAN. Convolves the input with a normal distribution kernel. It suppresses noise while maintaining superior structural integrity compared to box filters."
    if "median" in n: 
        return "FILTER: MEDIAN. A non-linear rank-order filter. It is the gold standard for removing impulsive noise while maintaining perfectly sharp edge gradients."
    if "unsharp" in n: 
        return "FILTER: UNSHARP MASK. Enhances edge contrast by amplifying high-frequency components. It compensates for blurring introduced during initial denoising stages."

    # CNN Structural Elements
    if "max_pool" in n: 
        return "OPERATOR: MAX POOLING (2x2). Performs non-linear spatial downsampling by selecting peak local activations. Ensures translational invariance and reduces data density."
    if "average_pool" in n: 
        return "OPERATOR: AVERAGE POOLING (2x2). Reduces spatial dimensions by calculating local mean. Provides a smoother but less localized feature representation."
    if "original_edges" in n: 
        return "INPUT: HIGH-RES FEATURE MAP. The original gradient baseline serving as the reference for subsequent spatial reduction and pooling experiments."

    # Advanced Architectures
    if "conv1x1" in n: 
        return "LAYER: POINTWISE CONVOLUTION (1x1). Mixes information across feature channels (Sobel X/Y). Learns linear combinations of kernel responses to optimize feature space."
    if "dilated" in n: 
        return "LAYER: DILATED (ATROUS) CONVOLUTION. Increases receptive field without increasing parameters by injecting zeros into the kernel (dilation rate > 1)."

    # Metrics
    if "shifted_sharpen_4" in n:
        return "KERNEL: SHIFTED SHARPEN (W=4). An intentional -1 central weight shift resulting in signal degradation and loss of high-frequency edge data."
    if "base_sharpen" in n: 
        return "KERNEL: BASELINE SHARPEN (W=5). A linear sharpening operator serving as the control group for the mathematical sensitivity evaluation."
    if "shifted_sharpen_6" in n: 
        return "KERNEL: SHIFTED SHARPEN (W=6). An intentional +1 central weight shift causing explosive high-frequency noise amplification."
    if "difference_map" in n: 
        return "METRIC: SAD DIFFERENCE MAP. Visualizes the Sum of Absolute Differences between kernels. Proves extreme volatility when weights are adjusted by a single unit."
    
    return "KERNEL: CONVOLUTION OPERATOR. Executing spatial feature extraction on target image topology."

def annotate_image(img_tensor: np.ndarray, title: str) -> np.ndarray:
    """
    Renders a high-fidelity technical UI panel.
    Calculates dynamic height to prevent text clipping on narrow images.
    The green border is applied ONLY to the text panel at the bottom.
    """
    # 1. Normalize image to 8-bit
    img_8bit = np.clip(img_tensor, 0, 255).astype(np.uint8)
    
    # 2. Convert to RGB for drawing colored UI
    if len(img_8bit.shape) == 2:
        img_color = cv2.cvtColor(img_8bit, cv2.COLOR_GRAY2RGB)
    else:
        img_color = cv2.cvtColor(img_8bit, cv2.COLOR_BGR2RGB)
        
    h, w, _ = img_color.shape
    description = get_professional_description(title)
    
    # 3. Load High-Quality Fonts
    try:
        title_font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 20)
        desc_font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 14)
    except:
        title_font = desc_font = ImageFont.load_default()

    # 4. DYNAMIC TEXT WRAPPING & HEIGHT CALCULATION (The Fix)
    char_width_px = 8.5 
    # Защита от очень узких картинок
    wrap_w = max(15, int((w - 40) / char_width_px))
    lines = textwrap.wrap(description, width=wrap_w)
    
    # Высота плашки = Отступ заголовка (45) + (Кол-во строк * 20) + Отступ снизу (15)
    bar_height = 45 + (len(lines) * 20) + 15
    total_height = h + bar_height
    
    # 5. UI Aesthetics
    neon_green = (0, 255, 0)   # Pure #00FF00 Matrix Green
    border_width = 2
    
    canvas = Image.new('RGB', (w, total_height), color=(0, 0, 0))
    img_pil = Image.fromarray(img_color)
    canvas.paste(img_pil, (0, 0))
    
    draw = ImageDraw.Draw(canvas)
    
    # 6. Draw the Hacker/Technical UI Frame (ONLY around the bottom text panel)
    draw.rectangle([0, h, w - 1, total_height - 1], outline=neon_green, width=border_width)
    
    # Tiny UI accents on the top-left and top-right of the text panel
    draw.line([(0, h), (15, h)], fill=neon_green, width=5)
    draw.line([(w - 15, h), (w, h)], fill=neon_green, width=5)
    
    # 7. Text Rendering
    display_title = title.replace("_", " ").upper()
    draw.text((20, h + 15), f"[{display_title}]", font=title_font, fill=neon_green)
    
    y_text = h + 45
    for line in lines:
        draw.text((20, y_text), line, font=desc_font, fill=neon_green)
        y_text += 20  # Line spacing
    
    return cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR)