import cv2
import numpy as np
from app.pipeline.stages.preprocessing.filters import ImageFilter

class ResizeFilter(ImageFilter):
    """
    Intelligently resizes images. If the image is too large, it downscales to a max_edge.
    Maintains aspect ratio.
    """
    
    def __init__(self, max_edge: int = 1800):
        self.max_edge = max_edge

    def apply(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        longest_edge = max(h, w)
        
        if longest_edge > self.max_edge:
            scale = self.max_edge / float(longest_edge)
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            # Using INTER_AREA for downscaling
            return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        return image
