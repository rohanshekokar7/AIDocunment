"""
AI Document Classification System
Developed by Rohan Shekokar
"""

from pydantic import BaseModel, Field
from typing import List, Optional

class ClassificationResponse(BaseModel):
    """
    Standard output schema for document classification.
    Matches the user's requested JSON format.
    """
    document_type: str = Field(..., description="The classified type of the document")
    writing_type: str = Field(..., description="Printed, Handwritten, or Mixed")
    language: str = Field(..., description="The primary language of the document")
    summary: str = Field(..., description="Key details extracted from the document")
    confidence: float = Field(..., description="Confidence score from 0.0 to 1.0")
    processing_time: float = Field(..., description="Time taken to process the document in seconds")

class BatchClassificationResponse(BaseModel):
    """
    Schema for batch processing multiple documents.
    """
    results: List[ClassificationResponse] = Field(..., description="List of individual classification results")
    total_processing_time: float = Field(..., description="Total time taken to process the batch")

class JobResponse(BaseModel):
    """
    Immediate response when a job is queued.
    """
    job_id: str = Field(..., description="Unique ID for the background job")
    status: str = Field("processing", description="Current status of the job")

class JobStatusResponse(BaseModel):
    """
    Response when checking the status of a job.
    """
    job_id: str = Field(..., description="Unique ID for the background job")
    status: str = Field(..., description="Status: processing, completed, or error")
    result: Optional[ClassificationResponse] = Field(None, description="The result if the job is completed")
    error: Optional[str] = Field(None, description="Error message if the job failed")
