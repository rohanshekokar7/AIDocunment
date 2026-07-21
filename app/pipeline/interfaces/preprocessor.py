"""
AI Document Classification System
Developed by Rohan Shekokar
"""

from abc import ABC, abstractmethod
from typing import List, Tuple
from PIL import Image

class ImagePreprocessor(ABC):
    @abstractmethod
    def process(self, file_path: str, all_pages: bool) -> Tuple[List[Image.Image], List[Image.Image]]:
        pass
