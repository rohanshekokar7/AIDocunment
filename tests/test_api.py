"""
AI Document Classification System
Developed by Rohan Shekokar
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from app.main import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "Welcome" in response.json()["message"]

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

@patch("app.services.classification_service.model_service.generate")
@patch("app.services.classification_service.process_document")
def test_classify_endpoint_mocked(mock_process, mock_generate):
    # Mock pre-processing and model generation to avoid loading heavy models in tests
    mock_process.return_value = [MagicMock()] # Mock PIL Image
    mock_generate.return_value = '{"document_type": "PAN Card", "writing_type": "Printed", "confidence": 0.98}'
    
    files = {"file": ("test.jpg", b"dummy image data", "image/jpeg")}
    response = client.post("/api/v1/classify", files=files)
    
    assert response.status_code == 200
    data = response.json()
    assert data["document_type"] == "PAN Card"
    assert data["writing_type"] == "Printed"
    assert data["confidence"] == 0.98
    assert "processing_time" in data

def test_classify_no_file():
    response = client.post("/api/v1/classify")
    assert response.status_code == 422 # Validation Error for missing file
