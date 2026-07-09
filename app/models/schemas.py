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
    confidence: float = Field(..., description="Confidence score from 0.0 to 1.0")
    processing_time: float = Field(..., description="Time taken to process the document in seconds")

class BatchClassificationResponse(BaseModel):
    """
    Schema for batch processing multiple documents.
    """
    results: List[ClassificationResponse] = Field(..., description="List of individual classification results")
    total_processing_time: float = Field(..., description="Total time taken to process the batch")
