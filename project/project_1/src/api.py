from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional
import time
import logging

from .model import ModelInference, load_model

from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="ML Model Serving API",
    description="REST API for serving image classification models",
    version="1.0.0"
)

request_count = Counter(
    'api_requests_total',
    'Total API requests',
    ['endpoint', 'method', 'status_code']
)

request_duration = Histogram(
    'api_request_duration_seconds',
    'Request duration in seconds',
    ['endpoint', 'method'],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
)

class PredictionRequest(BaseModel):
    image_url: Optional[str] = None
    top_k: int = Field(default=5, ge=1, le=10)
    threshold: float = Field(default=0.0, ge=0.0, le=1.0)

class Prediction(BaseModel):
    class_id: int
    class_name: str
    confidence: float

class PredictResponse(BaseModel):
    predictions: List[Prediction]
    inference_time_ms: float
    model_version: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str
    uptime_seconds: float

app.state.model = None
app.state.start_time = time.time()

@app.on_event("startup")
async def startup_event():
    logger.info("Starting ML Model Serving API...")
    try:
        logger.info("Loading ML model...")
        app.state.model = load_model()
        app.state.start_time = time.time()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down ML Model Serving API...")


@app.middleware("http")
async def track_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    logger.info(f"{request.method} {request.url.path} - {response.status_code} - {duration:.3f}s")

    request_count.labels(
        endpoint = request.url.path,
        method=request.method,
        status_code=response.status_code
    ).inc()

    request_duration.labels(
        endpoint = request.url.path,
        method=request.method
    ).observe(duration)
    return response
    

@app.get("/")
async def root():
    return {
        "name": "ML Model Serving API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    model_loaded = app.state.model is not None

    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return HealthResponse(
        status="healthy",
        model_loaded=True,
        version="1.0.0",
        uptime_seconds=time.time() - app.state.start_time
    )


