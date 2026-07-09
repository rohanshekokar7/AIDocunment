"""
AI Document Classification System
Developed by Rohan Shekokar
"""

from abc import ABC, abstractmethod

class ConfidenceEstimator(ABC):
    @abstractmethod
    def estimate(self, slm_output: dict, ocr_confidence: float) -> float:
        pass
