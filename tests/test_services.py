import pytest
from fastapi import HTTPException
from app.utils.file_utils import validate_file_extension
from app.services.classification_service import parse_model_output

def test_validate_file_extension_valid():
    # Should not raise any exception
    validate_file_extension("document.pdf")
    validate_file_extension("image.jpg")
    validate_file_extension("scan.PNG")

def test_validate_file_extension_invalid():
    with pytest.raises(HTTPException) as exc:
        validate_file_extension("document.txt")
    assert exc.value.status_code == 400
    assert "Unsupported file extension" in exc.value.detail

def test_parse_model_output_clean():
    raw_output = '{"document_type": "Invoice", "writing_type": "Printed", "confidence": 0.99}'
    parsed = parse_model_output(raw_output)
    assert parsed["document_type"] == "Invoice"
    assert parsed["writing_type"] == "Printed"
    assert parsed["confidence"] == 0.99

def test_parse_model_output_markdown_wrapped():
    raw_output = '```json\n{"document_type": "Passport", "writing_type": "Mixed", "confidence": 0.85}\n```'
    parsed = parse_model_output(raw_output)
    assert parsed["document_type"] == "Passport"

def test_parse_model_output_invalid():
    raw_output = "I am a helpful assistant. The document is an Invoice."
    with pytest.raises(ValueError):
        parse_model_output(raw_output)
