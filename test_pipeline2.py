import asyncio
from app.services.classification_service import pipeline
from app.utils.file_utils import save_upload_file_tmp
from fastapi import UploadFile
import os

def test():
    print("Testing pipeline directly...")
    class DummyUploadFile(UploadFile):
        def __init__(self, filename):
            self.filename = filename
            self.file = open(filename, "rb")
            
    dummy = DummyUploadFile("dummy.pdf")
    tmp = save_upload_file_tmp(dummy)
    print("Validating...")
    pipeline.validator.validate(tmp)
    print("Preprocessing (pdf2image)...")
    images = pipeline.preprocessor.process(tmp, False)
    print("Got", len(images), "images")
    import sys
    sys.exit(0)

if __name__ == "__main__":
    test()
