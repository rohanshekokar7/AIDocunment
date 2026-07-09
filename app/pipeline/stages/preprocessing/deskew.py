"""
AI Document Classification System
Developed by Rohan Shekokar
"""

import cv2
import numpy as np
from app.pipeline.stages.preprocessing.filters import ImageFilter

class DeskewFilter(ImageFilter):
    """
    Detects the skew angle of the text and rotates the image to correct it.
    """
    
    def apply(self, image: np.ndarray) -> np.ndarray:
        # Convert to grayscale if not already
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # Invert colors (text becomes white, background becomes black)
        gray = cv2.bitwise_not(gray)
        
        # Threshold to get purely binary image
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        
        # Find coordinates of all non-zero pixels
        coords = np.column_stack(np.where(thresh > 0))
        
        if len(coords) == 0:
            return image # Nothing to deskew
            
        # Calculate minimum bounding rectangle
        angle = cv2.minAreaRect(coords)[-1]
        
        # minAreaRect returns angles in range [-90, 0)
        # Depending on OpenCV version, the angle might be different.
        # General adjustment:
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
            
        # Limit deskewing to small angles (e.g., +/- 10 degrees). 
        # Large angles might indicate the document is intentionally landscape or it's a false positive.
        if abs(angle) > 10:
            return image
            
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        
        # Get rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Rotate the original image
        # Determine background color to fill borders (assume white for documents)
        if len(image.shape) == 3:
            border_value = (255, 255, 255)
        else:
            border_value = 255
            
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=border_value)
        return rotated
