# AI Document Classification System

A production-ready AI-powered Document Classification System built with **FastAPI**, **PaddleOCR**, and **Qwen2.5-1.5B** (Small Language Model).

## Features

- **Document Classification**: Classifies 20+ document types (Aadhaar, PAN, Invoices, etc.).
- **Writing Type Detection**: Determines if the document is `Printed`, `Handwritten`, or `Mixed`.
- **Confidence Scoring**: Self-evaluated model confidence for predictions.
- **Modular Pipeline**: Analyzes PDFs and Images directly using an 8-stage pipeline combining OCR, Layout Detection, and a Small Language Model.
- **FastAPI**: Asynchronous, high-performance API with Swagger UI docs.
- **Batch Processing**: Supports classifying multiple documents in one request.

## Architecture

```text
project/
├── app/
│   ├── api/
│   │   └── routes.py         # API endpoints
│   ├── core/
│   │   ├── config.py         # Pydantic settings
│   │   └── logging.py        # Custom logger
│   ├── models/
│   │   └── schemas.py        # Request/Response schemas
│   ├── prompts/
│   │   ├── pipeline/         # Modular 8-stage OCR+SLM pipeline
│   │   └── services/         # Orchestration services
│   │   └── classification_service.py # Orchestrates preprocessing & parsing
│   ├── utils/
│   │   ├── file_utils.py     # File validation
│   │   └── image_utils.py    # PDF to Image conversion
│   ├── main.py               # FastAPI application
├── tests/                    # Pytest unit & API tests
├── Dockerfile                # Containerization
├── docker-compose.yml        # Docker composition
└── requirements.txt          # Python dependencies
```

## Setup & Installation

### Prerequisites

- **Python 3.10+**
- **Poppler**: Required for `pdf2image` to convert PDFs.
  - Mac: `brew install poppler`
  - Linux: `sudo apt-get install poppler-utils`
- **GPU (Optional but recommended)**: CUDA or MPS compatible hardware.

### Local Setup

1. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
2. Set up environment variables:
   ```bash
   cp .env.example .env
   ```
3. Run the application:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```

### Docker Setup

To run using Docker (which handles Poppler automatically):

```bash
docker-compose up --build
```
*(Note: To use GPU inside Docker, uncomment the `deploy` block in `docker-compose.yml` and ensure NVIDIA Container Toolkit is installed).*

## API Endpoints

Once running, navigate to [http://localhost:8000/docs](http://localhost:8000/docs) for the interactive Swagger UI.

### `POST /api/v1/classify`
Classifies a single document.

- **Parameters**: 
  - `file`: The document (PDF, PNG, JPG).
  - `all_pages` (query boolean): If true, processes every page of a PDF.
- **Response**:
```json
{
  "document_type": "PAN Card",
  "writing_type": "Printed",
  "confidence": 0.98,
  "processing_time": 2.14
}
```

### `POST /api/v1/batch-classify`
Classifies multiple documents.

- **Parameters**: 
  - `files`: A list of documents.
  - `all_pages` (query boolean): If true, processes every page.
- **Response**:
```json
{
  "results": [
    {
      "document_type": "Invoice",
      "writing_type": "Mixed",
      "confidence": 0.95,
      "processing_time": 1.50
    }
  ],
  "total_processing_time": 1.50
}
```

## Future Improvements

- **Asynchronous Task Queue**: Integrate Celery + Redis for processing extremely large batch uploads in the background.
- **Model Quantization**: Use AWQ or GPTQ versions of the SLM to drastically reduce VRAM usage.
- **Extended Formats**: Add support for DOCX/XLSX by converting them to PDF first.
