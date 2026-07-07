from pdf2image import convert_from_path
from PIL import Image, ImageOps
import os
from typing import List
from fastapi import HTTPException
from app.core.logging import logger

def convert_pdf_to_images(pdf_path: str, all_pages: bool = False) -> List[Image.Image]:
    """
    Converts a PDF file to a list of PIL Images.
    If all_pages is False, only converts the first page.
    """
    try:
        # We only need the first page by default to save memory and processing time
        last_page = None if all_pages else 1
        images = convert_from_path(pdf_path, last_page=last_page)
        if not images:
            raise ValueError("No pages found in PDF")
        return images
    except Exception as e:
        logger.error(f"Failed to convert PDF to image: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error processing PDF: {str(e)}. (Ensure poppler-utils is installed on the host)")

def load_image(image_path: str) -> Image.Image:
    """
    Loads an image from disk and corrects orientation if EXIF data is present.
    """
    try:
        img = Image.open(image_path)
        img = ImageOps.exif_transpose(img) # Fix orientation issues from mobile cameras
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img
    except Exception as e:
        logger.error(f"Failed to load image {image_path}: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Corrupted or invalid image: {str(e)}")

def process_document(file_path: str, process_all_pages: bool = False) -> List[Image.Image]:
    """
    Processes a document path (PDF or Image) and returns a list of PIL Images.
    """
    ext = file_path.split(".")[-1].lower()
    
    if ext == "pdf":
        return convert_pdf_to_images(file_path, all_pages=process_all_pages)
    elif ext in ["jpg", "jpeg", "png"]:
        return [load_image(file_path)]
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported extension for processing: {ext}")
