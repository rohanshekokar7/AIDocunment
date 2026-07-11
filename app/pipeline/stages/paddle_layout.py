"""
AI Document Classification System
Developed by Rohan Shekokar
"""

import numpy as np
try:
    from paddleocr import PPStructure
    LAYOUT_CLASS = PPStructure
    KWARGS = {"show_log": False, "use_mkldnn": False, "recovery": False, "ocr_version": "PP-OCRv3"}
except ImportError:
    from paddleocr import PPStructureV3
    LAYOUT_CLASS = PPStructureV3
    KWARGS = {
        "use_doc_orientation_classify": False,
        "use_textline_orientation": False,
        "use_seal_recognition": False,
        "use_formula_recognition": False,
        "use_chart_recognition": False,
        "use_table_recognition": False,
        "use_doc_unwarping": False,
        "ocr_version": "PP-OCRv3"
    }

from app.pipeline.interfaces.layout_engine import LayoutEngine
from app.pipeline.models import PageData, LayoutRegion
from app.core.config import settings

class PaddleLayoutEngine(LayoutEngine):
    def __init__(self):
        # Dynamically initialize based on paddleocr version
        self.engine = LAYOUT_CLASS(
            lang=settings.OCR_LANG,
            **KWARGS
        )

    def detect_layout(self, page_data: PageData) -> PageData:
        # Convert PIL Image to BGR numpy array for PaddleOCR
        img_np = np.array(page_data.image.convert('RGB'))
        img_bgr = img_np[:, :, ::-1]
        
        try:
            # Run layout detection
            layout_results = self.engine(img_bgr)
            
            for region in layout_results:
                region_type = region.get('type', 'unknown')
                bbox = region.get('bbox', [])
                
                # Ensure bbox is a list of ints [x1, y1, x2, y2]
                flat_bbox = [int(coord) for coord in bbox] if bbox else []
                
                layout_region = LayoutRegion(
                    region_type=region_type,
                    bbox=flat_bbox
                )
                page_data.layout_regions.append(layout_region)
        except Exception as e:
            from app.core.logging import logger
            logger.warning(f"Layout detection failed for page {page_data.page_number}: {e}")
            
        return page_data
