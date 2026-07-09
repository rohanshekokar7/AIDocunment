"""
AI Document Classification System
Developed by Rohan Shekokar
"""

from typing import List, Optional
import numpy as np
from PIL import Image

from app.pipeline.interfaces.preprocessor import ImagePreprocessor
from app.pipeline.stages.preprocessing.filters import ImageFilter
from app.utils.image_utils import convert_pdf_to_images, load_image

class AdvancedPreprocessor(ImagePreprocessor):
    """
    Production-ready preprocessor orchestrating a chain of ImageFilters.
    Follows SOLID principles:
    - OCP/SRP: Behavior is extended by passing different filters.
    - DIP: Depends on the ImageFilter interface, not concrete implementations.
    """
    
    def __init__(self, filters: Optional[List[ImageFilter]] = None):
        if filters is None:
            self.filters = []
        else:
            self.filters = filters

    def process(self, file_path: str, all_pages: bool) -> List[Image.Image]:
        ext = file_path.split(".")[-1].lower()
        
        # Load raw PIL images using existing utilities
        if ext == "pdf":
            raw_images = convert_pdf_to_images(file_path, all_pages=all_pages)
        elif ext in ["jpg", "jpeg", "png"]:
            # load_image already applies EXIF rotation safely
            raw_images = [load_image(file_path)]
        else:
            raise ValueError(f"Unsupported extension for processing: {ext}")
            
        processed_images = []
        
        for pil_img in raw_images:
            # Convert PIL to OpenCV format (numpy array)
            # PIL is RGB, OpenCV is BGR
            cv_img = np.array(pil_img)
            if len(cv_img.shape) == 3:
                cv_img = cv_img[:, :, ::-1].copy() # RGB to BGR
                
            # Apply filter chain
            for img_filter in self.filters:
                cv_img = img_filter.apply(cv_img)
                
            # Convert back to PIL
            if len(cv_img.shape) == 3:
                cv_img = cv_img[:, :, ::-1] # BGR to RGB
                final_pil = Image.fromarray(cv_img)
            else:
                final_pil = Image.fromarray(cv_img, mode='L')
                
            processed_images.append(final_pil)
            
        return processed_images
