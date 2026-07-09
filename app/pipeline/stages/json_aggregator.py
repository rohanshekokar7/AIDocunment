"""
AI Document Classification System
Developed by Rohan Shekokar
"""

import json
from app.pipeline.interfaces.aggregator import FeatureAggregator
from app.pipeline.models import DocumentContext

class JSONAggregator(FeatureAggregator):
    def aggregate(self, document_context: DocumentContext) -> DocumentContext:
        aggregated_data = {
            "file_name": document_context.file_name,
            "pages": []
        }
        
        for page in document_context.pages:
            page_info = {
                "page_number": page.page_number,
                "text_blocks": [{"text": tb.text, "bbox": tb.bbox, "confidence": tb.confidence} for tb in page.raw_text_blocks],
                "layout_regions": [{"type": lr.region_type, "bbox": lr.bbox, "html": lr.html_content} for lr in page.layout_regions]
            }
            aggregated_data["pages"].append(page_info)
            
        document_context.aggregated_text = json.dumps(aggregated_data, ensure_ascii=False)
        return document_context
