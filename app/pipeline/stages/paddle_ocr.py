import numpy as np
from paddleocr import PaddleOCR
from app.pipeline.interfaces.ocr_engine import OCREngine
from app.pipeline.models import PageData, TextBlock
from app.core.config import settings

class PaddleOCREngine(OCREngine):
    def __init__(self):
        self.engine = PaddleOCR(
            use_angle_cls=False,
            lang=settings.OCR_LANG,
            use_mkldnn=False
        )

    def extract_text(self, page_data: PageData) -> PageData:
        img_np = np.array(page_data.image.convert('RGB'))
        img_bgr = img_np[:, :, ::-1]
        
        result = self.engine.ocr(img_bgr)
        
        if result and result[0]:
            res = result[0]
            if isinstance(res, dict) or hasattr(res, 'keys'):
                # New paddleocr >= 2.8 format (paddlex)
                polys = res.get('dt_polys', [])
                texts = res.get('rec_texts', [])
                scores = res.get('rec_scores', [])
                for bbox, text, conf in zip(polys, texts, scores):
                    flat_bbox = [int(pt) for sublist in bbox for pt in sublist]
                    page_data.raw_text_blocks.append(
                        TextBlock(text=text, bbox=flat_bbox, confidence=float(conf))
                    )
            else:
                for line in res:
                    bbox = line[0]  # [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                    text = line[1][0]
                    conf = line[1][1]
                    # Flatten bbox to list of ints for simplicity
                    flat_bbox = [int(pt) for sublist in bbox for pt in sublist]
                    page_data.raw_text_blocks.append(
                        TextBlock(text=text, bbox=flat_bbox, confidence=float(conf))
                    )
        return page_data
