import os
from app.pipeline.interfaces.vision_engine import VisionEngine
from app.pipeline.models import DocumentContext
from app.core.logging import logger

class LocalVisionEngine(VisionEngine):
    def __init__(self):
        self.model_id = "microsoft/Florence-2-base"
        self.model = None
        self.processor = None
        self.load_failed = False
        
    def _load_model(self):
        if self.model is not None or self.load_failed:
            return
            
        try:
            logger.info(f"Loading local Vision Model {self.model_id} on CPU...")
            from transformers import AutoProcessor, AutoModelForCausalLM
            # Use trust_remote_code=True for Florence-2
            self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_id, trust_remote_code=True)
            self.model.to("cpu")
            self.model.eval()
            logger.info("Successfully loaded Florence-2 model.")
        except Exception as e:
            logger.error(f"Failed to load local Vision Model: {e}. Will fallback to OCR heuristics.")
            self.load_failed = True

    def _heuristic_fallback(self, document_context: DocumentContext) -> str:
        """
        Ultra-fast zero-model approach analyzing PaddleOCR confidence and bounding boxes.
        """
        logger.info("Running OCR Heuristics for writing type detection...")
        if not document_context.pages:
            return "Unknown"
            
        total_blocks = 0
        low_confidence_blocks = 0
        
        for page in document_context.pages:
            for tb in page.raw_text_blocks:
                total_blocks += 1
                # PaddleOCR confidence drops significantly for messy handwriting
                if tb.confidence < 0.88:
                    low_confidence_blocks += 1
                    
        if total_blocks == 0:
            return "Unknown"
            
        low_conf_ratio = low_confidence_blocks / total_blocks
        
        # If a huge portion is low confidence, it's highly likely messy handwriting
        if low_conf_ratio > 0.6:
            return "Handwritten"
        # If there's a significant mix of high and low confidence, likely mixed (form + handwriting)
        elif low_conf_ratio > 0.15:
            return "Mixed"
        else:
            return "Printed"

    def detect_writing_type(self, document_context: DocumentContext) -> str:
        if not document_context.pages:
            return "Unknown"
            
        # 1. Try to load and use Florence-2
        self._load_model()
        
        if self.model is not None and self.processor is not None:
            try:
                logger.info("Running Florence-2 for writing type detection...")
                image = document_context.pages[0].original_image or document_context.pages[0].image
                
                # Florence-2 works best with specific task prompts
                task_prompt = "<MORE_DETAILED_CAPTION>"
                prompt = task_prompt
                
                inputs = self.processor(text=prompt, images=image, return_tensors="pt")
                # Ensure tensors are on CPU
                inputs = {k: v.to("cpu") for k, v in inputs.items() if hasattr(v, "to")}
                
                generated_ids = self.model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=1024,
                    num_beams=3,
                    do_sample=False
                )
                generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
                parsed_answer = self.processor.post_process_generation(generated_text, task=task_prompt, image_size=(image.width, image.height))
                
                answer_text = parsed_answer[task_prompt].lower()
                logger.info(f"Florence-2 Output: {answer_text}")
                
                # Simple keyword matching from the detailed caption
                if "handwritten" in answer_text or "handwriting" in answer_text:
                    if "print" in answer_text or "typed" in answer_text:
                        return "Mixed"
                    return "Handwritten"
                elif "print" in answer_text or "typed" in answer_text:
                    return "Printed"
                    
                # If Florence-2 doesn't give a clear text characteristic, we fallback
                logger.warning("Florence-2 answer was ambiguous. Falling back to heuristics.")
            except Exception as e:
                logger.error(f"Florence-2 inference failed: {e}. Falling back to heuristics.")
                
        # 2. Fallback to OCR Heuristics
        return self._heuristic_fallback(document_context)
