import cv2
import numpy as np
from typing import List, Tuple


def plot_filter_results(
    original: np.ndarray, 
    filtered_images: List[Tuple[str, np.ndarray]], 
    main_title: str
) -> None:
    """
    Renders images using OpenCV's native C++ GUI engine with robust window management.
    """
    print(f"\n[VISUALIZER] Initializing native OpenCV graphics for: {main_title}")
    print("[VISUALIZER] >>> PRESS 'Q' OR 'ESC' in any active image window to close and continue <<<")
    print("[VISUALIZER] Note: On macOS, the windows might open behind your editor. Look for the Python icon in your dock.")
    
    # 1. Display the original base image
    cv2.imshow("Original Image (Grayscale)", original)
    
    # 2. Iterate through the results and create a new window for each filter
    for title, img in filtered_images:
        cv2.imshow(title, img)
        
    # 3. Robust event loop to prevent accidental window closures.
    while True:
        key = cv2.waitKey(0) & 0xFF 
        if key == 27 or key == ord('q') or key == ord('Q'):
            break
            
    # 4. Safely clean up RAM and destroy native OS windows
    cv2.destroyAllWindows()
    
    # Flush the macOS GUI event queue
    cv2.waitKey(1) 
    
    print("[VISUALIZER] Native OS windows closed successfully.\n")