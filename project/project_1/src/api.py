#api.py

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional
import time
import logging

try:
    from .model import ModelInference, load_model
except ImportError:
    from model import ModelInference, load_model

from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

from PIL import Image
import io
from torchvision import transforms

import torch.nn.functional as F

from starlette.responses import Response

import torch

import uvicorn


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

prediction_count = Counter(
    "api_prediction_count_total",
    "Total prediction requests",
    ["status_code", "success"]
)

error_count = Counter(
    "api_errors_total",
    "Total errors by type",
    ["error_type"]
)


class PredictionRequest(BaseModel):
    image_url: Optional[str] = None
    top_k: int = Field(default=5, ge=1, le=10)
    threshold: float = Field(default=0.0, ge=0.0, le=1.0)

class Prediction(BaseModel):
    class_id: int
    class_name: str
    confidence: float

class PredictionResponse(BaseModel):
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


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...), top_k: int = 5, threshold: float = 0.0):
    try:
        # Validate input
        if file.content_type not in ["image/jpeg", "image/png"]:
            raise HTTPException(400, "Invalid image format")

        # Read and preprocess
        image_bytes = await file.read()
        image = preprocess_image(image_bytes)

        # Run inference
        start_time = time.time()
        predictions = app.state.model.predict(image, top_k=top_k)
        inference_time = (time.time() - start_time) * 1000  # ms

        # Update metrics
        prediction_count.labels(status_code=200, success="true").inc()

        # Format response
        return PredictionResponse(
            predictions=predictions,
            inference_time_ms=inference_time,
            model_version="1.0.0"
        )

    except ValueError as e:
        error_count.labels(error_type="validation").inc()
        raise HTTPException(400, str(e))
    except Exception as e:
        error_count.labels(error_type="inference").inc()
        logger.error(f"Prediction error: {e}")
        raise HTTPException(500, "Inference failed")

@app.get("/metrics")
async def metrics():
    metrics_data = generate_latest()

    return Response(
        content=metrics_data,
        media_type=CONTENT_TYPE_LATEST
    )







def preprocess_image(image_bytes: bytes):
    image = Image.open(io.BytesIO(image_bytes))

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)
    return input_batch


def format_predictions(outputs, top_k: int):
    
    probabilities = F.softmax(outputs, dim=1)
    top_probs, top_indices = torch.topk(probabilities, top_k)

    predictions = []
    for i in range(top_k):
        predictions.append(Prediction(
            class_id=int(top_indices[0][i]),
            class_name=get_class_name(int(top_indices[0][i])),
            confidence=float(top_probs[0][i])
        ))

    return predictions

def get_class_name(class_id: int) -> str:
    with open('imagenet_classes.txt', 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    return classes[class_id]


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )