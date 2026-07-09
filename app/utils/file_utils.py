"""
AI Document Classification System
Developed by Rohan Shekokar
"""

import os
import shutil
from tempfile import NamedTemporaryFile
from fastapi import UploadFile, HTTPException
from typing import IO

from app.core.config import settings

def validate_file_extension(filename: str) -> None:
    """
    Validates if the uploaded file has an allowed extension.
    Raises HTTPException if invalid.
    """
    if not filename:
        raise HTTPException(status_code=400, detail="Empty filename uploaded")
        
    ext = filename.split(".")[-1].lower()
    if ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file extension '{ext}'. Allowed: {', '.join(settings.ALLOWED_EXTENSIONS)}"
        )

def validate_file_size(file: IO) -> None:
    """
    Validates if the file size is within the allowed limit.
    Raises HTTPException if too large.
    """
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0) # Reset cursor
    
    max_size_bytes = settings.MAX_FILE_SIZE_MB * 1024 * 1024
    if file_size == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    if file_size > max_size_bytes:
        raise HTTPException(
            status_code=413, 
            detail=f"File too large. Maximum allowed size is {settings.MAX_FILE_SIZE_MB}MB"
        )

def save_upload_file_tmp(upload_file: UploadFile) -> str:
    """
    Saves an uploaded file to a temporary file on disk and returns the path.
    This is useful for pdf2image or other tools that require a file path.
    """
    validate_file_extension(upload_file.filename)
    validate_file_size(upload_file.file)
    
    try:
        suffix = os.path.splitext(upload_file.filename)[1]
        with NamedTemporaryFile(delete=False, suffix=suffix, dir=settings.UPLOAD_DIR) as tmp:
            shutil.copyfileobj(upload_file.file, tmp)
            tmp_path = tmp.name
        return tmp_path
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving file: {str(e)}")
    finally:
        upload_file.file.seek(0)
