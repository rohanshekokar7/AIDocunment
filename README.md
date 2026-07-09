---
title: AI Document Classifier
emoji: 📄
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---
# AI Document Classification System

A production-ready AI-powered Document Classification System built with **FastAPI**, **PaddleOCR**, and **Qwen2.5-1.5B** (Small Language Model).

🔥 **Live Demo:** [Try the deployed app on Hugging Face Spaces!](https://huggingface.co/spaces/rohanshekokar/ai-document-classifier)

---

## Features

- **Document Classification**: Classifies 19+ specific document types including Aadhaar, PAN, Passports, Bank Statements, Invoices, utility bills, and medical reports.
- **Writing Type Detection**: Determines if the document is `Printed`, `Handwritten`, or `Mixed`.
- **Confidence Scoring**: Blends SLM confidence with OCR extraction quality for highly reliable heuristics.
- **Modular Pipeline**: Analyzes PDFs and Images directly using an 8-stage pipeline combining OCR, Layout Detection, and a Small Language Model.
- **FastAPI**: Asynchronous, high-performance API with a beautiful web frontend and Swagger UI docs.
- **Mobile-Friendly**: The web UI works perfectly on mobile browsers.

---

## How to Run This Project Locally

If you want to run this code on your own machine instead of using the live demo, follow these steps:

### Prerequisites

- **Python 3.10+**
- **Poppler**: Required for converting PDFs to images.
  - **Mac:** `brew install poppler`
  - **Linux:** `sudo apt-get install poppler-utils`
  - **Windows:** Download [Poppler for Windows](https://github.com/oschwartz10612/poppler-windows/releases/) and add it to your PATH.

### Step-by-Step Setup

1. **Clone the repository and enter the directory:**
   ```bash
   git clone https://github.com/rohanshekokar7/AIDocunment.git
   cd AIDocunment
   ```

2. **Create a virtual environment (Recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the FastAPI Server:**
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```

5. **Open the App:**
   - **Web UI:** Open your browser and go to [http://localhost:8000](http://localhost:8000)
   - **Interactive API Docs:** Go to [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Run with Docker (Easiest Method)

If you don't want to manually install Poppler and Python dependencies, you can run the entire system inside an isolated Docker container:

```bash
docker-compose up --build
```
Once it finishes building, the app will be available at [http://localhost:8000](http://localhost:8000).

---

## API Endpoints

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

---

## Architecture Overview

```text
project/
├── app/
│   ├── api/                  # API endpoints
│   ├── core/                 # Settings and logging
│   ├── models/               # Request/Response schemas
│   ├── pipeline/             # Modular 8-stage OCR+SLM pipeline
│   ├── prompts/              # SLM prompts and guidelines
│   ├── services/             # Orchestration services
│   ├── utils/                # File validation and image utils
│   └── main.py               # FastAPI application entrypoint
├── static/                   # Web UI HTML/CSS/JS
├── Dockerfile                # Containerization
├── docker-compose.yml        # Docker composition
└── requirements.txt          # Pinned Python dependencies
```
