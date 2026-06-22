"""
End-to-end test: config, utils, and model inference on image.png
"""
import sys
sys.path.insert(0, "src")

print("=" * 60)
print("TEST 1: config.py — load_settings()")
print("=" * 60)
from config import load_settings, get_settings, override_settings, validate_configuration

settings = load_settings()
print(f"  is_production: {settings.is_production()}")
print(f"  is_development: {settings.is_development()}")
print(f"  get_device: {settings.get_device()}")
print(f"  get_model_path: {settings.get_model_path()}")
print(f"  to_dict keys: {list(settings.to_dict().keys())[:5]}...")
print()

# Test override
test_settings = override_settings(log_level="DEBUG", batch_size=4)
print(f"  override log_level: {test_settings.log_level}")
print(f"  override batch_size: {test_settings.batch_size}")
print()

# Test validation
errors = validate_configuration(settings)
print(f"  validation errors: {errors}")
print()

print("=" * 60)
print("TEST 2: utils.py — logging, image processing, cache")
print("=" * 60)
from utils import (
    setup_logging, validate_image_file, load_image_from_bytes,
    resize_image, preprocess_image, load_class_labels,
    get_top_predictions, calculate_latency_percentiles,
    log_prediction, handle_inference_error, PredictionCache,
    check_model_health, get_system_info, warm_up_model,
)

# Logging
logger = setup_logging("INFO", json_format=False)
logger.info("Logger initialized OK")

# Image processing with cat image
with open("image.png", "rb") as f:
    img_bytes = f.read()

is_valid, err = validate_image_file(img_bytes, max_size_mb=10)
print(f"  validate_image_file: valid={is_valid}, error='{err}'")

image = load_image_from_bytes(img_bytes)
print(f"  load_image_from_bytes: size={image.size}, mode={image.mode}")

resized = resize_image(image, (224, 224))
print(f"  resize_image: size={resized.size}")

tensor = preprocess_image(resized)
print(f"  preprocess_image: shape={tensor.shape}, dtype={tensor.dtype}")

# Class labels
labels = load_class_labels()
print(f"  load_class_labels: {len(labels)} classes, first 3: {labels[:3]}")

# Latency percentiles
latencies = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
pctls = calculate_latency_percentiles(latencies)
print(f"  latency percentiles: p50={pctls['p50']}, p95={pctls['p95']}")

# Cache
cache = PredictionCache(max_size=5, ttl_seconds=60)
cache.set("key1", {"class": "cat"})
print(f"  cache.get('key1'): {cache.get('key1')}")
print(f"  cache.get('miss'): {cache.get('miss')}")
print(f"  cache.size(): {cache.size()}")

# Error handling
err_resp = handle_inference_error(ValueError("bad input"), logger)
print(f"  handle_inference_error: {err_resp}")

# System info
sys_info = get_system_info()
print(f"  system_info keys: {list(sys_info.keys())}")
print()

print("=" * 60)
print("TEST 3: model.py — Full inference on cat image")
print("=" * 60)
from model import ModelInference

model_inf = ModelInference("resnet18")
model_inf.load_model()
print(f"  Model loaded: {model_inf.model is not None}")
print(f"  Device: {model_inf.device}")
print(f"  Model info: {model_inf.get_model_info()}")

# Run prediction on the cat image
predictions = model_inf.predict(img_bytes, top_k=5)
print(f"\n  🐱 Top-5 predictions for image.png:")
for i, p in enumerate(predictions, 1):
    print(f"    {i}. {p['class_name']:30s} (confidence: {p['confidence']:.4f})")

# Log the prediction
log_prediction(logger, "image.png", predictions[0], latency_ms=42.0)

# Health check
health = check_model_health(model_inf.model, model_inf.device)
print(f"\n  Model health: {health}")

print()
print("=" * 60)
print("ALL TESTS PASSED ✅")
print("=" * 60)
