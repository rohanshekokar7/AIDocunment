import io
import base64
import json
import httpx
from PIL import Image
from app.pipeline.interfaces.vision_engine import VisionEngine
from app.pipeline.models import DocumentContext
from app.core.config import settings
from app.core.logging import logger

class APIVisionEngine(VisionEngine):
    def __init__(self):
        self.api_url = settings.EXTERNAL_LLM_API_URL
        self.api_key = settings.EXTERNAL_LLM_API_KEY
        self.model_name = "llava:7b"  # Changed from glm-ocr to avoid timeout hallucination
        
    def detect_writing_type(self, document_context: DocumentContext) -> str:
        if not document_context.pages:
            return "Unknown"
            
        if not self.api_url or not self.api_key:
            logger.error("API URL or API Key is missing for Vision Engine.")
            return "Unknown"
            
        # Get the first page image
        first_page_img = document_context.pages[0].image
        
        # Convert PIL Image to base64
        buffered = io.BytesIO()
        first_page_img.save(buffered, format="JPEG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        headers = {
            "api-key": self.api_key,
            "Content-Type": "application/json"
        }
        
        prompt = "Look at this document. Is the text primarily Printed, Handwritten, or Mixed? Answer with only one word: 'Printed', 'Handwritten', or 'Mixed'."
        
        payload = {
            "model": self.model_name,
            "query": prompt,
            "images": [img_b64]
        }
        
        try:
            logger.info(f"Sending request to external Vision API ({self.api_url}) for model: {self.model_name}")
            response = httpx.post(self.api_url, headers=headers, json=payload, timeout=60.0)
            response.raise_for_status()
            
            data = response.json()
            
            output_text = ""
            if isinstance(data, dict):
                if "response" in data:
                    output_text = data["response"]
                elif "choices" in data and len(data["choices"]) > 0:
                    choice = data["choices"][0]
                    if "message" in choice and "content" in choice["message"]:
                        output_text = choice["message"]["content"]
                    elif "text" in choice:
                        output_text = choice["text"]
                elif "text" in data:
                    output_text = data["text"]
                elif "content" in data:
                    output_text = data["content"]
            else:
                output_text = str(data)
                
            cleaned = output_text.strip().lower()
            if "mixed" in cleaned:
                return "Mixed"
            elif "handwritten" in cleaned:
                return "Handwritten"
            elif "print" in cleaned:
                return "Printed"
            
            return "Unknown"
            
        except Exception as e:
            logger.error(f"Error calling Vision API: {e}")
            return "Unknown"
