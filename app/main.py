import asyncio
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

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
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.on_event("startup")
async def startup_event():
    logger.info("Starting up FastAPI application...")
    # Optional: Load model asynchronously so the first request doesn't block indefinitely
    # asyncio.create_task(asyncio.to_thread(model_service.load_model))
    # For now, it will load lazily on the first request.
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
    return FileResponse("static/index.html")

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "slm_loaded": pipeline.slm_engine.model is not None,
        "device": getattr(pipeline.slm_engine, "device", None)
    }
