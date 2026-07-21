"""
AI Document Classification System
Developed by Rohan Shekokar
"""

from typing import List, Tuple
from PIL import Image
from app.pipeline.interfaces.preprocessor import ImagePreprocessor
from app.utils.image_utils import process_document

class StandardPreprocessor(ImagePreprocessor):
    def process(self, file_path: str, all_pages: bool) -> Tuple[List[Image.Image], List[Image.Image]]:
        # Utilizing existing image conversion utility
        images = process_document(file_path, process_all_pages=all_pages)
        return images, images
