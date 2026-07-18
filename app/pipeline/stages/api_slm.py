"""
AI Document Classification System
External API SLM Engine
"""

import json
import httpx
from app.pipeline.interfaces.slm_engine import SLMEngine
from app.pipeline.models import DocumentContext
from app.core.config import settings
from app.core.logging import logger
from app.prompts.classification_prompt import build_classification_prompt

class APISLMEngine(SLMEngine):
    def __init__(self):
        self.api_url = settings.EXTERNAL_LLM_API_URL
        self.api_key = settings.EXTERNAL_LLM_API_KEY
        self.model_name = settings.EXTERNAL_LLM_MODEL_NAME

    def classify(self, document_context: DocumentContext) -> dict:
        if not self.api_url or not self.api_key:
            logger.error("API URL or API Key is missing in settings.")
            return {"document_type": "Other / Unknown", "writing_type": "Unknown", "confidence": 0.0}

        prompt = build_classification_prompt(document_context.aggregated_text)
        
        headers = {
            "api-key": self.api_key,
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model_name,
            "query": prompt
        }
        
        try:
            logger.info(f"Sending request to external LLM API ({self.api_url}) for model: {self.model_name}")
            # Use timeout=120 as LLM generation can be slow
            response = httpx.post(self.api_url, headers=headers, json=payload, timeout=120.0)
            response.raise_for_status()
            
            data = response.json()
            
            # Extract the generated text. 
            # Trying to handle different potential response structures from "LLM-as-a-Service".
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
                    # If it's already a JSON structure that looks like our expected output
                    if "document_type" in data:
                        return data
                    output_text = str(data)
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
                
            # Try to find JSON block if there's conversational wrapper text
            start_idx = cleaned.find("{")
            end_idx = cleaned.rfind("}")
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                cleaned = cleaned[start_idx:end_idx+1]
                
            return json.loads(cleaned.strip())
            
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP Error from External API: {e.response.status_code} - {e.response.text}")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from External API response: {e}\nRaw Output: {output_text}")
        except Exception as e:
            logger.error(f"Unexpected error calling External API: {e}")
            
        return {"document_type": "Other / Unknown", "writing_type": "Unknown", "confidence": 0.0}
