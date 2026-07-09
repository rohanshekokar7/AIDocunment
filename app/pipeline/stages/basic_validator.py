"""
AI Document Classification System
Developed by Rohan Shekokar
"""

import os
from app.pipeline.interfaces.validator import DocumentValidator

class BasicValidator(DocumentValidator):
    def validate(self, file_path: str) -> bool:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        if os.path.getsize(file_path) == 0:
            raise ValueError(f"File is empty: {file_path}")
        return True
