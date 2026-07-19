"""
AI Document Classification System
Developed by Rohan Shekokar
"""

from typing import List, Dict, Any
from app.pipeline.models import DocumentContext, PageData, ClassificationResult
from app.pipeline.interfaces.validator import DocumentValidator
from app.pipeline.interfaces.preprocessor import ImagePreprocessor
from app.pipeline.interfaces.ocr_engine import OCREngine
from app.pipeline.interfaces.layout_engine import LayoutEngine
from app.pipeline.interfaces.aggregator import FeatureAggregator
from app.pipeline.interfaces.slm_engine import SLMEngine
from app.pipeline.interfaces.vision_engine import VisionEngine
from app.pipeline.interfaces.confidence_estimator import ConfidenceEstimator

class PipelineOrchestrator:
    def __init__(
        self,
        validator: DocumentValidator,
        preprocessor: ImagePreprocessor,
        ocr_engine: OCREngine,
        layout_engine: LayoutEngine,
        aggregator: FeatureAggregator,
        slm_engine: SLMEngine,
        vision_engine: VisionEngine,
        rule_engine: SLMEngine,
        confidence_estimator: ConfidenceEstimator
    ):
        self.validator = validator
        self.preprocessor = preprocessor
        self.ocr_engine = ocr_engine
        self.layout_engine = layout_engine
        self.aggregator = aggregator
        self.slm_engine = slm_engine
        self.vision_engine = vision_engine
        self.rule_engine = rule_engine
        self.confidence_estimator = confidence_estimator

    def process(self, file_path: str, filename: str, all_pages: bool = False, approach: str = "llm") -> ClassificationResult:
        from app.core.logging import logger
        
        # 1. Validate
        logger.info("Pipeline Step 1: Validating document...")
        self.validator.validate(file_path)
        
        # 2. Preprocess
        logger.info("Pipeline Step 2: Preprocessing document (PDF to Images)...")
        images = self.preprocessor.process(file_path, all_pages)
        logger.info(f"Generated {len(images)} page images.")
        
        # Initialize context
        doc_context = DocumentContext(file_name=filename)
        
        for i, img in enumerate(images):
            logger.info(f"Processing Page {i+1}/{len(images)}...")
            page_data = PageData(page_number=i+1, image=img)
            
            # 3. OCR
            logger.info(f"Page {i+1}: Running OCR...")
            page_data = self.ocr_engine.extract_text(page_data)
            
            # 4. Layout
            logger.info(f"Page {i+1}: Running Layout Detection...")
            page_data = self.layout_engine.detect_layout(page_data)
            
            doc_context.pages.append(page_data)
            
        # 5. Aggregate
        logger.info("Pipeline Step 5: Aggregating features...")
        doc_context = self.aggregator.aggregate(doc_context)
        
        if approach == "rule":
            # Rule-Based Path
            logger.info("Pipeline Step 6: Running Rule-Based Inference...")
            classification_out = self.rule_engine.classify(doc_context)
        else:
            # LLM-Based Path
            # 6. SLM Inference
            logger.info("Pipeline Step 6: Running SLM Inference (This may take a while)...")
            classification_out = self.slm_engine.classify(doc_context)
            
        # 6.5 Vision Engine Inference for Writing Type (Run for both approaches)
        logger.info("Pipeline Step 6.5: Running Vision Engine for writing type detection...")
        writing_type_out = self.vision_engine.detect_writing_type(doc_context)
        
        # 7. Confidence Estimation
        logger.info("Pipeline Step 7: Estimating confidence...")
        # Calculate average OCR confidence across all pages and blocks
        total_conf = 0
        count = 0
        for page in doc_context.pages:
            for tb in page.raw_text_blocks:
                total_conf += tb.confidence
                count += 1
        avg_ocr_conf = (total_conf / count) if count > 0 else 0.0
        
        final_conf = self.confidence_estimator.estimate(classification_out, avg_ocr_conf)
        
        # 8. Output Result
        return ClassificationResult(
            document_type=classification_out.get("document_type", "Other / Unknown"),
            writing_type=writing_type_out,
            language=classification_out.get("language", "Unknown"),
            summary=classification_out.get("summary", "No details extracted."),
            confidence=final_conf
        )
