"""
AI Document Classification System
Developed by Rohan Shekokar
"""

from fastapi import APIRouter, File, UploadFile, Query, HTTPException
from typing import List

from app.models.schemas import ClassificationResponse, BatchClassificationResponse, JobResponse, JobStatusResponse
from app.services.classification_service import batch_classify_documents, queue_classification_job, get_job_status

router = APIRouter()

@router.post("/classify", response_model=JobResponse)
async def classify_endpoint(
    file: UploadFile = File(..., description="The document image or PDF to classify"),
    all_pages: bool = Query(False, description="If true, processes all pages of a PDF"),
    approach: str = Query("llm", description="The classification approach: 'llm' or 'rule'")
):
    """
    Queues a single uploaded document for classification.
    Returns a job_id to poll for status.
    """
    job_id, tmp_path, filename = queue_classification_job(file, all_pages, approach)
    return JobResponse(job_id=job_id, status="processing")

@router.get("/status/{job_id}", response_model=JobStatusResponse)
async def status_endpoint(job_id: str):
    """
    Check the status of a classification job.
    """
    job_data = get_job_status(job_id)
    if job_data is None:
        raise HTTPException(status_code=404, detail="Job not found")
        
    return JobStatusResponse(
        job_id=job_id,
        status=job_data.get("status", "error"),
        result=job_data.get("result"),
        error=job_data.get("error")
    )

@router.post("/batch-classify", response_model=BatchClassificationResponse)
async def batch_classify_endpoint(
    files: List[UploadFile] = File(..., description="List of documents to classify"),
    all_pages: bool = Query(False, description="If true, processes all pages of PDFs")
):
    """
    Classifies a batch of uploaded documents.
    They are processed sequentially to optimize GPU memory footprint.
    """
    return await batch_classify_documents(files, all_pages)
