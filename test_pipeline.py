import asyncio
from app.services.classification_service import classify_document
from fastapi import UploadFile
import os

async def main():
    print("Testing pipeline...")
    # Create a dummy file object from the existing dummy.pdf
    class DummyUploadFile(UploadFile):
        def __init__(self, filename):
            self.filename = filename
            self.file = open(filename, "rb")

    if not os.path.exists("dummy.pdf"):
        print("dummy.pdf not found!")
        return
        
    dummy = DummyUploadFile("dummy.pdf")
    try:
        res = await classify_document(dummy, all_pages=False)
        print("Result:", res)
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    asyncio.run(main())
