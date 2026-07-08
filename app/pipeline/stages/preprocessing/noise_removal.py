import cv2
import numpy as np
from app.pipeline.stages.preprocessing.filters import ImageFilter

class NoiseRemovalFilter(ImageFilter):
    """
    Removes grain and background artifacts using Non-Local Means Denoising.
    """
    
    def __init__(self, h: float = 10.0, template_window_size: int = 7, search_window_size: int = 21):
        self.h = h
        self.template_window_size = template_window_size
        self.search_window_size = search_window_size

    def apply(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            # Color image
            return cv2.fastNlMeansDenoisingColored(
                image, 
                None, 
                self.h, 
                self.h, 
                self.template_window_size, 
                self.search_window_size
            )
        else:
            # Grayscale image
            return cv2.fastNlMeansDenoising(
                image, 
                None, 
                self.h, 
                self.template_window_size, 
                self.search_window_size
            )
