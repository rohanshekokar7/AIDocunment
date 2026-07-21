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
        self.model_name = "qwen2.5-vl-7b-instruct"
        
    def detect_writing_type(self, document_context: DocumentContext) -> str:
        if not document_context.pages:
            return "Unknown"
            
        if not self.api_url or not self.api_key:
            logger.error("API URL or API Key is missing for Vision Engine.")
            return "Unknown"
            
        # Get the original (non-thresholded) RGB image if available, else fallback
        first_page_img = document_context.pages[0].original_image or document_context.pages[0].image
        
        # Convert PIL Image to base64 with high quality to preserve details for VLM
        buffered = io.BytesIO()
        first_page_img.save(buffered, format="JPEG", quality=95)
        img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        headers = {
            "api-key": self.api_key,
            "Content-Type": "application/json"
        }
        
        prompt = """### SYSTEM INSTRUCTION ###
You are an expert Document Analysis AI. Your task is to analyze the provided document image and determine if the text is 'Printed', 'Handwritten', or 'Mixed'.

CRITICAL RULES:
1. "Printed": The document consists entirely of machine-printed text (e.g., standard digital invoices, typed letters).
2. "Handwritten": The document is entirely written by hand (e.g., a handwritten note or prescription).
3. "Mixed": The document contains BOTH printed text AND handwritten elements. 
   - *Exception Rule*: If a document is a printed form but contains a handwritten signature, handwritten date, or filled-in handwritten fields, it MUST be classified as "Mixed".

Analyze the visual layout, ink styles, and text uniformity. Provide your response as a valid JSON object. Do not include markdown formatting like ```json.

REQUIRED JSON FORMAT:
{
    "reasoning": "Step-by-step visual analysis of the text elements...",
    "writing_type": "<Printed | Handwritten | Mixed>",
    "confidence": <float between 0.0 and 1.0>
}
"""
        
        payload = {
            "model": self.model_name,
            "query": prompt,
            "images": [img_b64],
            "temperature": 0.1,
            "top_p": 0.9
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
                
            # Parse output_text to find JSON
            cleaned = output_text.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
                
            start_idx = cleaned.find("{")
            end_idx = cleaned.rfind("}")
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                cleaned = cleaned[start_idx:end_idx+1]
                
            try:
                parsed_json = json.loads(cleaned)
                writing_type = parsed_json.get("writing_type", "Unknown")
                logger.info(f"Vision Reasoning: {parsed_json.get('reasoning', 'None')}")
                
                # Normalize output
                w_lower = writing_type.lower()
                if "mixed" in w_lower: return "Mixed"
                elif "handwritten" in w_lower: return "Handwritten"
                elif "print" in w_lower: return "Printed"
                else: return "Unknown"
            except json.JSONDecodeError:
                logger.error(f"Failed to parse Vision JSON: {cleaned}")
                # Fallback heuristic
                cleaned_lower = cleaned.lower()
                if "mixed" in cleaned_lower: return "Mixed"
                elif "handwritten" in cleaned_lower: return "Handwritten"
                elif "print" in cleaned_lower: return "Printed"
            
            return "Unknown"
            
        except Exception as e:
            logger.error(f"Error calling Vision API: {e}")
            return "Unknown"
