"""
AI Document Classification System
Developed by Rohan Shekokar
"""

import os
import uuid
import json
import sys
import subprocess
from fastapi import UploadFile, HTTPException
from typing import List, Dict, Any

from app.core.logging import logger
from app.models.schemas import ClassificationResponse, BatchClassificationResponse, JobResponse, JobStatusResponse
from app.utils.file_utils import save_upload_file_tmp

def get_job_file_path(job_id: str) -> str:
    return f"/tmp/job_{job_id}.json"

def get_job_status(job_id: str) -> dict:
    """
    Reads the job status from the file system.
    Returns None if the job doesn't exist.
    """
    job_file = get_job_file_path(job_id)
    if os.path.exists(job_file):
        try:
            with open(job_file, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            # File might be partially written, assume processing
            return {"status": "processing", "result": None, "error": None}
    return None

def queue_classification_job(upload_file: UploadFile, all_pages: bool = False, approach: str = "llm") -> tuple[str, str, str]:
    """
    Saves the file to disk and spawns a background OS process to process it.
    Returns (job_id, tmp_path, filename).
    """
    try:
        tmp_path = save_upload_file_tmp(upload_file)
        job_id = str(uuid.uuid4())
        
        # Initialize job state
        job_file = get_job_file_path(job_id)
        with open(job_file, 'w') as f:
            json.dump({"status": "processing", "result": None, "error": None}, f)
            
        # Spawn the background worker using a separate process
        # sys.executable ensures we use the exact same python environment
        cmd = [
            sys.executable,
            "worker.py",
            job_id,
            tmp_path,
            upload_file.filename,
            str(all_pages),
            approach
        ]
        
        # We don't wait for it to finish, it runs detached in the background
        subprocess.Popen(cmd)
        
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
