from fastapi import APIRouter, File, UploadFile, Query
from typing import List

from app.models.schemas import ClassificationResponse, BatchClassificationResponse
from app.services.classification_service import classify_document, batch_classify_documents

router = APIRouter()

@router.post("/classify", response_model=ClassificationResponse)
async def classify_endpoint(
    file: UploadFile = File(..., description="The document image or PDF to classify"),
    all_pages: bool = Query(False, description="If true, processes all pages of a PDF")
):
    """
    Classifies a single uploaded document.
    Supported formats: PDF, JPG, JPEG, PNG.
    """
    return await classify_document(file, all_pages)

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
