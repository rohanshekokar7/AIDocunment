"""
AI Document Classification System
Developed by Rohan Shekokar
"""

from abc import ABC, abstractmethod
from app.pipeline.models import PageData

class LayoutEngine(ABC):
    @abstractmethod
    def detect_layout(self, page_data: PageData) -> PageData:
        pass
