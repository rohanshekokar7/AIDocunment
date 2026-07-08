import numpy as np
from paddleocr import PPStructureV3
from app.pipeline.interfaces.layout_engine import LayoutEngine
from app.pipeline.models import PageData, LayoutRegion
from app.core.config import settings

class PaddleLayoutEngine(LayoutEngine):
    def __init__(self):
        # We initialize PPStructure for layout detection. 
        self.engine = PPStructureV3(
            lang=settings.OCR_LANG,
            use_doc_orientation_classify=False,
            use_textline_orientation=False,
            use_seal_recognition=False,
            use_formula_recognition=False,
            use_chart_recognition=False,
            use_table_recognition=False,
            use_doc_unwarping=False
        )

    def detect_layout(self, page_data: PageData) -> PageData:
        # BYPASS: Layout detection on MacOS (MPS) via background threads causes intermittent deadlocks 
        # and hangs for complex documents like Resumes. 
        # For document classification, OCR text is sufficient for the SLM.
        return page_data
