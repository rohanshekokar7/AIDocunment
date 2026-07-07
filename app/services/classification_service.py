import time
import json
import os
from fastapi import UploadFile, HTTPException
from typing import List

from app.core.logging import logger
from app.models.schemas import ClassificationResponse, BatchClassificationResponse
from app.utils.file_utils import save_upload_file_tmp
from app.utils.image_utils import process_document
from app.prompts.classification_prompt import CLASSIFICATION_PROMPT
from app.services.model_service import model_service

def parse_model_output(output_str: str) -> dict:
    """
    Parses the JSON string returned by the VLM.
    Strips out markdown code blocks if the model wrapped it.
    """
    cleaned = output_str.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    if cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
        
    cleaned = cleaned.strip()
    
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.error(f"JSON Parse Error: {e}\nRaw output:\n{output_str}")
        raise ValueError("Model did not return a valid JSON object.")

async def classify_document(upload_file: UploadFile, all_pages: bool = False) -> ClassificationResponse:
    """
    Orchestrates the classification of a single document.
    Saves to tmp -> Converts to images -> Infers -> Parses -> Cleans up.
    """
    start_time = time.time()
    tmp_path = None
    
    try:
        logger.info(f"Processing classification for file: {upload_file.filename}")
        tmp_path = save_upload_file_tmp(upload_file)
        
        # Pre-process: PDF to Images or Image loading
        images = process_document(tmp_path, process_all_pages=all_pages)
        
        # Inference
        output_str = model_service.generate(images, CLASSIFICATION_PROMPT)
        
        # Parse JSON
        parsed_data = parse_model_output(output_str)
        
        processing_time = round(time.time() - start_time, 2)
        
        response = ClassificationResponse(
            document_type=parsed_data.get("document_type", "Other / Unknown"),
            writing_type=parsed_data.get("writing_type", "Unknown"),
            confidence=float(parsed_data.get("confidence", 0.0)),
            processing_time=processing_time
        )
        logger.info(f"Classified {upload_file.filename} as '{response.document_type}' (Confidence: {response.confidence}) in {processing_time}s")
        return response
        
    except HTTPException:
        # Re-raise FastApi exceptions so they return correct status codes
        raise
    except ValueError as ve:
        logger.error(f"Value Error: {ve}")
        raise HTTPException(status_code=500, detail=f"Model output format error: {str(ve)}")
    except RuntimeError as re:
        logger.error(f"Runtime Error: {re}")
        raise HTTPException(status_code=500, detail=f"Model inference failed: {str(re)}")
    except Exception as e:
        logger.error(f"Unhandled error classifying document: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during classification")
    finally:
        # Cleanup
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception as cleanup_err:
                logger.warning(f"Failed to delete tmp file {tmp_path}: {cleanup_err}")

async def batch_classify_documents(files: List[UploadFile], all_pages: bool = False) -> BatchClassificationResponse:
    """
    Processes multiple documents sequentially to manage VRAM.
    If one document fails, it returns an error result for that document rather than failing the batch.
    """
    start_time = time.time()
    results = []
    
    for file in files:
        try:
            res = await classify_document(file, all_pages)
            results.append(res)
        except HTTPException as http_exc:
            logger.error(f"Batch processing HTTP error for {file.filename}: {http_exc.detail}")
            results.append(ClassificationResponse(
                document_type="Error",
                writing_type="Error",
                confidence=0.0,
                processing_time=0.0
            ))
        except Exception as e:
            logger.error(f"Error in batch for {file.filename}: {str(e)}")
            results.append(ClassificationResponse(
                document_type="Error",
                writing_type="Error",
                confidence=0.0,
                processing_time=0.0
            ))
            
    total_time = round(time.time() - start_time, 2)
    return BatchClassificationResponse(results=results, total_processing_time=total_time)
