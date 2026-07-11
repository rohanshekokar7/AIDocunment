"""
AI Document Classification System
Developed by Rohan Shekokar
"""

import os
# Workaround: Disable PIR and MKLDNN to prevent PaddlePaddle 2.6.2 hardware crashes on shared CPU instances
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_use_mkldnn"] = "0"

import asyncio
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

from app.api.routes import router
from app.core.config import settings
from app.core.logging import logger
from app.services.classification_service import pipeline

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="AI-powered Document Classification System using Modular OCR + SLM Pipeline",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

@app.on_event("startup")
async def startup_event():
    logger.info("Starting up FastAPI application...")
    # TODO: Pre-warm the SLM in a background thread here if cold-starts become too slow
    pass

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global Error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred. Please check the server logs."},
    )

app.include_router(router, prefix=settings.API_V1_STR)

@app.get("/", response_class=HTMLResponse)
def read_root():
    return FileResponse(str(BASE_DIR / "static" / "index.html"))

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "slm_loaded": pipeline.slm_engine.model is not None,
        "device": getattr(pipeline.slm_engine, "device", None)
    }
