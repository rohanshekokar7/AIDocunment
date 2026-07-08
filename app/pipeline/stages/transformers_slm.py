import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from app.pipeline.interfaces.slm_engine import SLMEngine
from app.pipeline.models import DocumentContext
from app.core.config import settings
from app.core.logging import logger
from app.prompts.classification_prompt import build_classification_prompt

class TransformersSLMEngine(SLMEngine):
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = None

    def load_model(self):
        if self.model is not None:
            return

        logger.info(f"Loading SLM: {settings.SLM_MODEL_NAME}")
        self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                settings.SLM_MODEL_NAME, 
                torch_dtype=torch.float16, 
                device_map="auto" if self.device == "cuda" else None
            )
            if self.device != "cuda":
                self.model.to(self.device)
                
            self.tokenizer = AutoTokenizer.from_pretrained(settings.SLM_MODEL_NAME)
            logger.info("SLM loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load SLM: {e}")
            raise RuntimeError(f"SLM init failed: {e}")

    def classify(self, document_context: DocumentContext) -> dict:
        if self.model is None:
            self.load_model()
            
        prompt = build_classification_prompt(document_context.aggregated_text)
        messages = [{"role": "user", "content": prompt}]
        
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=256, temperature=0.1)
            trimmed = outputs[0][inputs["input_ids"].shape[1]:]
            output_text = self.tokenizer.decode(trimmed, skip_special_tokens=True).strip()
            
        # Parse output_text
        cleaned = output_text.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        
        try:
            return json.loads(cleaned.strip())
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse SLM JSON: {e}\nRaw: {output_text}")
            return {"document_type": "Other / Unknown", "writing_type": "Unknown"}
