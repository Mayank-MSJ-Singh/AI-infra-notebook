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




