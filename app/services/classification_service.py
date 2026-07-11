"""
AI Document Classification System
Developed by Rohan Shekokar
"""

import time
import os
import traceback
import uuid
from fastapi import UploadFile, HTTPException
from typing import List, Dict, Any

from app.core.logging import logger
from app.models.schemas import ClassificationResponse, BatchClassificationResponse, JobResponse, JobStatusResponse
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

# Global in-memory job store
jobs: Dict[str, Dict[str, Any]] = {}

def process_job_background(job_id: str, tmp_path: str, filename: str, all_pages: bool):
    """
    Executes the heavy pipeline process in a background thread.
    """
    start_time = time.time()
    try:
        logger.info(f"Background job {job_id} started for file: {filename}")
        
        # Pipeline process is synchronous, so it will block this background thread,
        # which is exactly what we want in FastAPI BackgroundTasks.
        result = pipeline.process(tmp_path, filename, all_pages)
        
        processing_time = round(time.time() - start_time, 2)
        response = ClassificationResponse(
            document_type=result.document_type,
            writing_type=result.writing_type,
            confidence=result.confidence,
            processing_time=processing_time
        )
        
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["result"] = response
        logger.info(f"Background job {job_id} completed. Classified as '{response.document_type}' in {processing_time}s")
        
    except Exception as e:
        logger.error(f"Background job {job_id} failed: {traceback.format_exc()}")
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = "Internal server error during classification"
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception as cleanup_err:
                logger.warning(f"Failed to delete tmp file {tmp_path}: {cleanup_err}")

def queue_classification_job(upload_file: UploadFile, all_pages: bool = False) -> tuple[str, str, str]:
    """
    Saves the file to disk and prepares a background job.
    Returns (job_id, tmp_path, filename).
    """
    try:
        tmp_path = save_upload_file_tmp(upload_file)
        job_id = str(uuid.uuid4())
        jobs[job_id] = {
            "status": "processing",
            "result": None,
            "error": None
        }
        return job_id, tmp_path, upload_file.filename
    except Exception as e:
        logger.error(f"Error queueing job: {e}")
        raise HTTPException(status_code=500, detail="Failed to queue document processing")

async def batch_classify_documents(files: List[UploadFile], all_pages: bool = False):
    """
    Batch processing is temporarily disabled for background job architecture.
    Could be implemented by queueing multiple jobs.
    """
    raise HTTPException(status_code=501, detail="Batch classification is currently unsupported with the background polling architecture.")
