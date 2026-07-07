import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from app.core.config import settings
from app.core.logging import logger
from PIL import Image
from typing import List

class ModelService:
    """
    Singleton service to handle Qwen2.5-VL model loading and inference.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelService, cls).__new__(cls)
            cls._instance.model = None
            cls._instance.processor = None
            cls._instance.device = None
        return cls._instance

    def load_model(self):
        """
        Lazily loads the model into memory (GPU if available).
        """
        if self.model is not None:
            return

        logger.info(f"Loading model: {settings.MODEL_NAME}")
        
        # Determine device
        if settings.USE_GPU_IF_AVAILABLE:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = "cpu"
            
        logger.info(f"Using device for model: {self.device}")

        try:
            # We use bfloat16 to save memory (6GB instead of 12GB) AND prevent 'inf'/'nan' math errors
            torch_dtype = torch.bfloat16
            
            # Load model
            # Note: For MPS on macs, device_map="auto" might be buggy, manual mapping is safer
            if self.device == "cuda":
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    settings.MODEL_NAME, 
                    torch_dtype=torch_dtype, 
                    device_map="auto"
                )
            else:
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    settings.MODEL_NAME, 
                    torch_dtype=torch_dtype
                )
                self.model.to(self.device)
                
            self.processor = AutoProcessor.from_pretrained(settings.MODEL_NAME)
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise RuntimeError(f"Model initialization failed: {str(e)}")

    def generate(self, images: List[Image.Image], prompt_text: str) -> str:
        """
        Runs inference on the provided images using the given prompt.
        """
        if self.model is None:
            self.load_model()
            
        # Resize images to avoid massive CPU computation times for high-res PDFs
        for img in images:
            img.thumbnail((768, 768))

        # Format messages for Qwen2.5-VL
        content = [{"type": "image", "image": img} for img in images]
        content.append({"type": "text", "text": prompt_text})
        
        messages = [
            {"role": "user", "content": content}
        ]
        
        # Preprocess using qwen_vl_utils
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        
        # Move inputs to target device
        inputs = inputs.to(self.device)
        
        # Generate Output
        try:
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_new_tokens=128)
                
                # Trim the input prompt tokens from the output
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                
                output_text = self.processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                
            return output_text[0].strip()
        except Exception as e:
            logger.error(f"Inference error: {str(e)}")
            if "out of memory" in str(e).lower():
                raise RuntimeError("GPU Out of Memory error during inference")
            raise RuntimeError(f"Inference failed: {str(e)}")

# Instantiate the singleton
model_service = ModelService()
