import cv2
import numpy as np
from app.pipeline.stages.preprocessing.filters import ImageFilter

class ThresholdFilter(ImageFilter):
    """
    Applies Gaussian Adaptive Thresholding to convert the image to binary.
    Excellent for removing shadows and uneven illumination in documents.
    """
    
    def __init__(self, block_size: int = 21, c_value: int = 10):
        # block_size must be odd
        self.block_size = block_size if block_size % 2 == 1 else block_size + 1
        self.c_value = c_value

    def apply(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        thresh = cv2.adaptiveThreshold(
            gray, 
            255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 
            self.block_size, 
            self.c_value
        )
        
        # Convert back to 3 channel so subsequent filters or models don't crash
        # if they expect 3 channels
        return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
