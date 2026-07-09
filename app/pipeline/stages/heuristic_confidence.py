"""
AI Document Classification System
Developed by Rohan Shekokar
"""

from app.pipeline.interfaces.confidence_estimator import ConfidenceEstimator

class HeuristicConfidenceEstimator(ConfidenceEstimator):
    def estimate(self, slm_output: dict, ocr_confidence: float) -> float:
        # Get SLM stated confidence
        slm_conf = slm_output.get("confidence", 0.5)
        try:
            slm_conf = float(slm_conf)
        except ValueError:
            slm_conf = 0.5
            
        # If we have an OCR confidence (e.g., average), we could blend it.
        # For simplicity, if OCR confidence is high, we trust the SLM. If low, we penalize it.
        if ocr_confidence > 0.0:
            final_conf = (slm_conf * 0.7) + (ocr_confidence * 0.3)
        else:
            final_conf = slm_conf
            
        return round(min(max(final_conf, 0.0), 1.0), 2)
