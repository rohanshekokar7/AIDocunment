"""
AI Document Classification System
Developed by Rohan Shekokar
"""

from abc import ABC, abstractmethod
from typing import List
from PIL import Image

class ImagePreprocessor(ABC):
    @abstractmethod
    def process(self, file_path: str, all_pages: bool) -> List[Image.Image]:
        pass
