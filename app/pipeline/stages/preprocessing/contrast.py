"""
AI Document Classification System
Developed by Rohan Shekokar
"""

import cv2
import numpy as np
from app.pipeline.stages.preprocessing.filters import ImageFilter

class ContrastFilter(ImageFilter):
    """
    Improves contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization).
    Works best on the L channel of LAB color space to avoid hue shifts.
    """
    
    def __init__(self, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def apply(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            # Convert to LAB space
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L-channel
            cl = self.clahe.apply(l)
            
            # Merge back
            limg = cv2.merge((cl, a, b))
            return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        else:
            # Already grayscale
            return self.clahe.apply(image)
