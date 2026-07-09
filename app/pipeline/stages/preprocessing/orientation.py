"""
AI Document Classification System
Developed by Rohan Shekokar
"""

import cv2
import numpy as np
from app.pipeline.stages.preprocessing.filters import ImageFilter

class OrientationFilter(ImageFilter):
    """
    Attempts to correct orientation based on projection profiles.
    Often, document text lines have higher variance when projected horizontally
    than vertically (assuming text runs left to right).
    Note: For 180-degree rotation, we typically rely on OCR engine capabilities.
    This filter only handles 90/270 degree rotation heuristically.
    """
    
    def apply(self, image: np.ndarray) -> np.ndarray:
        # We need a binary image to calculate projection profiles
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        gray = cv2.bitwise_not(gray)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        
        # Calculate horizontal and vertical projection profiles
        h_proj = np.sum(thresh, axis=1)
        v_proj = np.sum(thresh, axis=0)
        
        # Calculate variance of the profiles
        h_var = np.var(h_proj)
        v_var = np.var(v_proj)
        
        # If vertical variance is significantly higher, image is likely rotated 90 or 270 degrees
        if v_var > h_var * 1.5:
            # We assume a 90 degree rotation clockwise. 
            # Differentiating 90 vs 270 without text recognition is unreliable,
            # but we rotate 90 degrees to make it horizontal at least.
            rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            return rotated
            
        return image
