"""
AI Document Classification System
Developed by Rohan Shekokar
"""

import time
import os
import traceback
from fastapi import UploadFile, HTTPException
from typing import List

from app.core.logging import logger
from app.models.schemas import ClassificationResponse, BatchClassificationResponse
from app.utils.file_utils import save_upload_file_tmp

# Pipeline imports
from app.pipeline.stages.basic_validator import BasicValidator
from app.pipeline.stages.preprocessing.advanced_preprocessor import AdvancedPreprocessor
from app.pipeline.stages.preprocessing.orientation import OrientationFilter
from app.pipeline.stages.preprocessing.deskew import DeskewFilter
from app.pipeline.stages.preprocessing.resize import ResizeFilter
from app.pipeline.stages.preprocessing.noise_removal import NoiseRemovalFilter
from app.pipeline.stages.preprocessing.contrast import ContrastFilter
from app.pipeline.stages.preprocessing.thresholding import ThresholdFilter
from app.pipeline.stages.paddle_ocr import PaddleOCREngine
from app.pipeline.stages.paddle_layout import PaddleLayoutEngine
from app.pipeline.stages.json_aggregator import JSONAggregator
from app.pipeline.stages.transformers_slm import TransformersSLMEngine
from app.pipeline.stages.heuristic_confidence import HeuristicConfidenceEstimator
from app.pipeline.orchestrator import PipelineOrchestrator

# Initialize the pipeline orchestrator
pipeline = PipelineOrchestrator(
    validator=BasicValidator(),
    preprocessor=AdvancedPreprocessor(filters=[
        OrientationFilter(),
        DeskewFilter(),
        ResizeFilter(max_edge=1800),
        NoiseRemovalFilter(),
        ContrastFilter(),
        ThresholdFilter()
    ]),
    ocr_engine=PaddleOCREngine(),
    layout_engine=PaddleLayoutEngine(),
    aggregator=JSONAggregator(),
    slm_engine=TransformersSLMEngine(),
    confidence_estimator=HeuristicConfidenceEstimator()
)

async def classify_document(upload_file: UploadFile, all_pages: bool = False) -> ClassificationResponse:
    """
    Orchestrates the classification of a single document using the modular pipeline.
    """
    start_time = time.time()
    tmp_path = None
    
    try:
        logger.info(f"Processing classification for file: {upload_file.filename}")
        tmp_path = save_upload_file_tmp(upload_file)
        
        import asyncio
        # Execute the modular pipeline in a background thread so we don't block FastAPI's event loop
        result = await asyncio.to_thread(pipeline.process, tmp_path, upload_file.filename, all_pages)
        
        processing_time = round(time.time() - start_time, 2)
        
        response = ClassificationResponse(
            document_type=result.document_type,
            writing_type=result.writing_type,
            confidence=result.confidence,
            processing_time=processing_time
        )
        logger.info(f"Classified {upload_file.filename} as '{response.document_type}' (Confidence: {response.confidence}) in {processing_time}s")
        return response
        
    except HTTPException:
        raise
    except ValueError as ve:
        logger.error(f"Value Error: {ve}")
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(ve)}")
    except RuntimeError as re:
        logger.error(f"Runtime Error: {re}")
        raise HTTPException(status_code=500, detail=f"Pipeline inference failed: {str(re)}")
    except Exception as e:
        logger.error(f"Unhandled error classifying document: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Internal server error during classification")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception as cleanup_err:
                logger.warning(f"Failed to delete tmp file {tmp_path}: {cleanup_err}")

async def batch_classify_documents(files: List[UploadFile], all_pages: bool = False) -> BatchClassificationResponse:
    """
    Processes multiple documents sequentially.
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
