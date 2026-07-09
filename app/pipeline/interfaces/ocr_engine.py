"""
AI Document Classification System
Developed by Rohan Shekokar
"""

from abc import ABC, abstractmethod
from app.pipeline.models import PageData

class OCREngine(ABC):
    @abstractmethod
    def extract_text(self, page_data: PageData) -> PageData:
        pass
