#utils.py
"""
Utility functions for ML Model Serving API

This module provides helper functions for:
- Image processing and validation
- Model loading and caching
- Metrics and monitoring
- Logging utilities
- Error handling

Students should implement these utility functions following the TODO instructions.
"""

import io
import os
import json
import hashlib
import logging
import time
import traceback
import threading
from collections import OrderedDict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image
import torch
import torchvision.transforms as transforms
import numpy as np
from pathlib import Path


# ==============================================================================
# Setup Logging
# ==============================================================================

class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""

    def format(self, record):
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        if record.exc_info and record.exc_info[0] is not None:
            log_data["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_data)


def setup_logging(log_level: str = "INFO", json_format: bool = True) -> logging.Logger:
    """
    Setup logging configuration.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        json_format: Use JSON format if True, standard format if False

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger("ml_serving")
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Avoid adding duplicate handlers on repeated calls
    if not logger.handlers:
        handler = logging.StreamHandler()
        if json_format:
            handler.setFormatter(JSONFormatter())
        else:
            handler.setFormatter(
                logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            )
        logger.addHandler(handler)

    return logger


# ==============================================================================
# Image Processing Functions
# ==============================================================================

def validate_image_file(file_content: bytes, max_size_mb: int = 10) -> Tuple[bool, str]:
    """
    Validate image file for size, format, dimensions, and corruption.

    Args:
        file_content: Raw bytes of uploaded file
        max_size_mb: Maximum allowed file size in MB

    Returns:
        Tuple of (is_valid, error_message). If valid, error_message is empty.
    """
    # Check file size
    size_mb = len(file_content) / (1024 * 1024)
    if size_mb > max_size_mb:
        return False, f"File size {size_mb:.1f}MB exceeds maximum {max_size_mb}MB"

    # Try to open and verify the image
    try:
        image = Image.open(io.BytesIO(file_content))
    except Exception:
        return False, "File is not a valid image or is corrupted"

    # Validate format
    allowed_formats = {"JPEG", "PNG", "JPG"}
    if image.format and image.format.upper() not in allowed_formats:
        return False, f"Unsupported image format '{image.format}'. Allowed: {allowed_formats}"

    # Validate dimensions
    width, height = image.size
    max_dim = 4096
    if width > max_dim or height > max_dim:
        return False, f"Image dimensions {width}x{height} exceed maximum {max_dim}x{max_dim}"

    if width == 0 or height == 0:
        return False, "Image has zero width or height"

    # Verify image is not corrupted
    try:
        image.verify()
    except Exception:
        return False, "Image file is corrupted"

    return True, ""


def load_image_from_bytes(image_bytes: bytes) -> Image.Image:
    """
    Load PIL Image from bytes and convert to RGB.

    Args:
        image_bytes: Raw image bytes

    Returns:
        PIL Image in RGB format

    Raises:
        RuntimeError: If image cannot be loaded
    """
    try:
        image = Image.open(io.BytesIO(image_bytes))
        if image.mode != "RGB":
            image = image.convert("RGB")
        return image
    except Exception as e:
        raise RuntimeError(f"Failed to load image: {e}")


def resize_image(image: Image.Image, size: Tuple[int, int] = (224, 224)) -> Image.Image:
    """
    Resize image using center-crop strategy (resize shorter edge, then crop).

    Args:
        image: PIL Image to resize
        size: Target size as (width, height)

    Returns:
        Resized PIL Image
    """
    target_w, target_h = size
    w, h = image.size

    # Resize so the shorter edge matches the target dimension
    scale = max(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    image = image.resize((new_w, new_h), Image.BILINEAR)

    # Center crop to exact target dimensions
    left = (new_w - target_w) // 2
    top = (new_h - target_h) // 2
    image = image.crop((left, top, left + target_w, top + target_h))

    return image


def preprocess_image(
    image: Image.Image,
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225],
) -> torch.Tensor:
    """
    Preprocess image for model inference (ImageNet normalization).

    Args:
        image: PIL Image to preprocess
        mean: Mean values for normalization (ImageNet default)
        std: Std values for normalization (ImageNet default)

    Returns:
        Preprocessed tensor of shape (1, 3, 224, 224)
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    tensor = transform(image).unsqueeze(0)
    return tensor


# ==============================================================================
# Model Management Functions
# ==============================================================================

def get_model_path(model_name: str, models_dir: str = "models") -> Path:
    """
    Get path to model file.

    Args:
        model_name: Name of the model (e.g., "resnet18")
        models_dir: Directory containing model files

    Returns:
        Path to model file
    """
    model_dir = Path(models_dir)
    model_path = model_dir / f"{model_name}.pth"
    return model_path


def load_class_labels(labels_file: str = "imagenet_classes.txt") -> List[str]:
    """
    Load class labels from file.

    Args:
        labels_file: Path to class labels file

    Returns:
        List of class names
    """
    labels_path = Path(labels_file)
    if not labels_path.exists():
        # Try JSON format as fallback
        json_path = labels_path.with_suffix(".json")
        if json_path.exists():
            with open(json_path, "r") as f:
                labels_dict = json.load(f)
            # Sort by key (class index) and return values
            return [labels_dict[str(i)] for i in range(len(labels_dict))]

        # Return generic labels if file not found
        return [f"class_{i}" for i in range(1000)]

    with open(labels_path, "r") as f:
        return [line.strip() for line in f.readlines()]


def warm_up_model(model: torch.nn.Module, device: torch.device, num_iterations: int = 10):
    """
    Warm up model with dummy inputs to trigger JIT compilation and CUDA kernel init.

    Args:
        model: PyTorch model to warm up
        device: Device (cpu or cuda)
        num_iterations: Number of warm-up iterations
    """
    dummy_input = torch.randn(1, 3, 224, 224).to(device)

    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(dummy_input)


# ==============================================================================
# Prediction Post-processing Functions
# ==============================================================================

def get_top_predictions(
    probabilities: torch.Tensor,
    class_labels: List[str],
    top_k: int = 5,
    threshold: float = 0.0,
) -> List[Dict[str, Any]]:
    """
    Get top-k predictions from model output logits/probabilities.

    Args:
        probabilities: Model output logits or probabilities
        class_labels: List of class names
        top_k: Number of top predictions to return
        threshold: Minimum confidence threshold

    Returns:
        List of prediction dicts with class_name, class_id, confidence
    """
    # Apply softmax if values don't look like probabilities
    probs = torch.nn.functional.softmax(probabilities, dim=0)

    top_probs, top_indices = torch.topk(probs, top_k)

    results = []
    for prob, idx in zip(top_probs, top_indices):
        confidence = float(prob.item())
        if confidence >= threshold:
            class_id = int(idx.item())
            results.append({
                "class_name": class_labels[class_id] if class_id < len(class_labels) else f"class_{class_id}",
                "class_id": class_id,
                "confidence": confidence,
            })

    return results


# ==============================================================================
# Monitoring and Metrics Functions
# ==============================================================================

def calculate_latency_percentiles(latencies: List[float]) -> Dict[str, float]:
    """
    Calculate latency percentiles from a list of measurements.

    Args:
        latencies: List of latency measurements (in milliseconds)

    Returns:
        Dictionary with min, max, mean, p50, p95, p99
    """
    if not latencies:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0}

    arr = np.array(latencies)
    return {
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
    }


def log_prediction(
    logger: logging.Logger,
    image_name: str,
    prediction: Dict,
    latency_ms: float,
    status: str = "success",
):
    """
    Log prediction with structured data.

    Args:
        logger: Logger instance
        image_name: Name of the image file
        prediction: Prediction result dictionary
        latency_ms: Inference latency in milliseconds
        status: Status of prediction (success/error)
    """
    log_data = {
        "event": "prediction",
        "image": image_name,
        "prediction": prediction,
        "latency_ms": round(latency_ms, 2),
        "status": status,
    }

    if status == "success":
        logger.info(f"Prediction completed: {json.dumps(log_data)}")
    else:
        logger.error(f"Prediction failed: {json.dumps(log_data)}")


# ==============================================================================
# Error Handling Functions
# ==============================================================================

def handle_inference_error(error: Exception, logger: logging.Logger) -> Dict[str, str]:
    """
    Handle inference errors and return formatted error response.

    Args:
        error: Exception that occurred
        logger: Logger instance

    Returns:
        Error response dict with error type, message, and details.
    """
    logger.error(f"Inference error: {error}", exc_info=True)

    error_map = {
        ValueError: ("validation_error", "Invalid input provided"),
        RuntimeError: ("runtime_error", "Model execution failed"),
        MemoryError: ("memory_error", "Insufficient memory for inference"),
        TimeoutError: ("timeout_error", "Inference timed out"),
    }

    error_type, user_message = error_map.get(
        type(error), ("internal_error", "An unexpected error occurred")
    )

    return {
        "error": error_type,
        "message": user_message,
        "details": str(error),
    }


# ==============================================================================
# Caching Functions
# ==============================================================================

class PredictionCache:
    """
    Thread-safe in-memory LRU prediction cache with TTL expiry.

    Usage:
        cache = PredictionCache(max_size=1000, ttl_seconds=300)
        cached = cache.get(image_hash)
        if cached is None:
            result = model.predict(image)
            cache.set(image_hash, result)
    """

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 300):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict[str, Tuple[Dict, float]] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[Dict]:
        """Get value from cache. Returns None on miss or expiry."""
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None

            value, timestamp = self._cache[key]

            # Check TTL
            if time.time() - timestamp > self.ttl_seconds:
                del self._cache[key]
                self._misses += 1
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self._hits += 1
            return value

    def set(self, key: str, value: Dict) -> None:
        """Set value in cache with LRU eviction."""
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._cache[key] = (value, time.time())
            else:
                if len(self._cache) >= self.max_size:
                    self._cache.popitem(last=False)  # Evict LRU
                self._cache[key] = (value, time.time())

    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    def size(self) -> int:
        """Get current cache size."""
        with self._lock:
            return len(self._cache)

    def hit_rate(self) -> float:
        """Get cache hit rate (0.0 to 1.0)."""
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0


# ==============================================================================
# Health Check Functions
# ==============================================================================

def check_model_health(model: torch.nn.Module, device: torch.device) -> Dict[str, Any]:
    """
    Check if model is healthy by running a test inference.

    Args:
        model: PyTorch model to check
        device: Device (cpu or cuda)

    Returns:
        Health status dictionary
    """
    result: Dict[str, Any] = {
        "healthy": False,
        "model_loaded": model is not None,
        "device_available": True,
        "test_inference_ms": None,
        "error": None,
    }

    if model is None:
        result["error"] = "Model is not loaded"
        return result

    # Check device availability
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        result["device_available"] = False
        result["error"] = "CUDA device not available"
        return result

    # Run test inference
    try:
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        start = time.time()
        with torch.no_grad():
            _ = model(dummy_input)
        result["test_inference_ms"] = round((time.time() - start) * 1000, 2)
        result["healthy"] = True
    except Exception as e:
        result["error"] = f"Test inference failed: {e}"

    return result


def get_system_info() -> Dict[str, Any]:
    """
    Get system information (CPU, memory, GPU, disk).

    Returns:
        System information dictionary
    """
    info: Dict[str, Any] = {}

    try:
        import psutil
        info["cpu_count"] = psutil.cpu_count()
        info["cpu_percent"] = psutil.cpu_percent(interval=0.1)
        mem = psutil.virtual_memory()
        info["memory_total_gb"] = round(mem.total / (1024 ** 3), 1)
        info["memory_available_gb"] = round(mem.available / (1024 ** 3), 1)
        info["memory_percent"] = mem.percent
        disk = psutil.disk_usage("/")
        info["disk_usage_percent"] = disk.percent
    except ImportError:
        import multiprocessing
        info["cpu_count"] = multiprocessing.cpu_count()
        info["note"] = "Install psutil for full system metrics"

    # GPU info
    info["gpu_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        info["gpu_count"] = torch.cuda.device_count()
        info["gpu_name"] = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory
        info["gpu_memory_gb"] = round(gpu_mem / (1024 ** 3), 1)
    else:
        info["gpu_count"] = 0

    return info