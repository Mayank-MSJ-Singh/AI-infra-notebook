# Lesson 04: ML Frameworks Fundamentals

**Duration:** 6 hours
**Objectives:** Understand PyTorch and TensorFlow from an infrastructure engineer's perspective

## Introduction

As an ML infrastructure engineer, you don't need to be an expert in training models, but you do need to understand ML frameworks well enough to:

- Deploy models effectively
- Troubleshoot framework-related issues
- Optimize inference performance
- Choose appropriate serving strategies
- Communicate with data scientists

This lesson covers PyTorch and TensorFlow from an **infrastructure perspective** - focusing on what you need to know to build and maintain ML infrastructure.

## PyTorch vs TensorFlow: Overview

### PyTorch
**Developer:** Meta (Facebook) AI Research
**Released:** 2016
**Philosophy:** Dynamic computation graphs, Pythonic, research-friendly

**Strengths:**
- Easy to learn and debug
- Dynamic graphs allow flexibility
- Popular in research community
- Strong ecosystem (Hugging Face, Lightning)
- Better for experimentation

**Use Cases:**
- Research projects
- NLP (transformers, LLMs)
- Computer vision research
- Rapid prototyping

### TensorFlow
**Developer:** Google Brain
**Released:** 2015
**Philosophy:** Static computation graphs (TF 1.x), production-optimized

**Strengths:**
- Mature production tools (TF Serving)
- Mobile/edge deployment (TF Lite)
- Large enterprise adoption
- Comprehensive ecosystem
- Better for production at scale

**Use Cases:**
- Production ML systems
- Mobile and edge AI
- Large-scale distributed training
- Enterprise deployments

### Market Share (2025)
- **Research papers:** ~70% PyTorch, ~30% TensorFlow
- **Production deployments:** ~50% PyTorch, ~50% TensorFlow
- **Trend:** PyTorch gaining in production, TF strong in mobile/edge

## PyTorch Fundamentals for Infrastructure Engineers

### 1. Understanding PyTorch Models

**Model Components:**
```python
import torch
import torch.nn as nn

# Basic model structure
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.layer1 = nn.Linear(10, 50)
        self.layer2 = nn.Linear(50, 1)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.layer2(x)
        return x

# Create model
model = SimpleModel()

# Model has parameters (weights)
print(f"Parameters: {sum(p.numel() for p in model.parameters())}")
```

**What Infrastructure Engineers Need to Know:**
- Models are Python objects (nn.Module)
- Models have state (parameters/weights)
- forward() defines inference logic
- Parameters need to be loaded before serving

### 2. Model Serialization (Saving/Loading)

**Two ways to save models:**

```python
# Method 1: Save entire model (easier but less flexible)
torch.save(model, 'model_complete.pth')
loaded_model = torch.load('model_complete.pth')

# Method 2: Save state dict (recommended for production)
torch.save(model.state_dict(), 'model_weights.pth')

# To load:
model = SimpleModel()  # Must recreate architecture
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()  # Set to evaluation mode!
```

**Infrastructure Implications:**
- State dict is ~50% smaller (no architecture code)
- State dict is more portable across PyTorch versions
- Must have model architecture code to load state dict
- Always call `.eval()` before serving!

### 3. Inference Modes

```python
# WRONG - Training mode (uses dropout, batch norm training behavior)
output = model(input_data)

# CORRECT - Evaluation mode
model.eval()
with torch.no_grad():  # Disable gradient computation
    output = model(input_data)
```

**Why This Matters:**
- `.eval()` changes behavior of layers (dropout, batch norm)
- `torch.no_grad()` saves memory and speeds up inference
- Forgetting these causes incorrect predictions!

### 4. Device Management (CPU vs GPU)

```python
# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Move model to device
model = model.to(device)

# Move input to same device
input_tensor = input_tensor.to(device)

# Inference
with torch.no_grad():
    output = model(input_tensor)

# Move output back to CPU for processing
output = output.cpu().numpy()
```

**Infrastructure Considerations:**
- Model and input must be on same device
- GPU inference is 10-100x faster for large models
- Moving data between CPU/GPU adds latency (~1-10ms)
- One model can use one GPU (without model parallelism)

### 5. Batch Inference

```python
# Single inference (inefficient)
for image in images:
    output = model(image.unsqueeze(0))  # Add batch dimension

# Batch inference (much better!)
batch = torch.stack(images)  # Stack into batch
outputs = model(batch)  # Process all at once
```

**Performance Impact:**
- Batch inference is 3-10x faster than single inference
- GPU utilization improves dramatically with batching
- Tradeoff: Batch inference adds latency (waiting for batch)

### 6. Model Optimization for Inference

```python
# 1. TorchScript - Compile model for faster inference
scripted_model = torch.jit.script(model)
scripted_model.save("model_scripted.pt")

# Load and use
scripted_model = torch.jit.load("model_scripted.pt")
output = scripted_model(input_tensor)

# 2. Quantization - Reduce model size and increase speed
import torch.quantization
quantized_model = torch.quantization.quantize_dynamic(
    model, {nn.Linear}, dtype=torch.qint8
)
```

**Optimization Benefits:**
- **TorchScript:** 10-30% faster inference
- **Quantization:** 4x smaller model, 2-4x faster
- Both can be combined

## TensorFlow Fundamentals for Infrastructure Engineers

### 1. Understanding TensorFlow Models

**TensorFlow 2.x uses Keras API:**

```python
import tensorflow as tf
from tensorflow import keras

# Create model
model = keras.Sequential([
    keras.layers.Dense(50, activation='relu', input_shape=(10,)),
    keras.layers.Dense(1)
])

# Model must be compiled
model.compile(optimizer='adam', loss='mse')

# Model summary
model.summary()
```

**Key Differences from PyTorch:**
- Models are higher-level abstractions
- Must be compiled before use
- Static computation graph (optimized at graph creation)

### 2. Model Serialization (Saving/Loading)

```python
# Method 1: SavedModel format (recommended for production)
model.save('my_model')  # Creates directory with model

# Load
loaded_model = keras.models.load_model('my_model')

# Method 2: HDF5 format (older, single file)
model.save('model.h5')
loaded_model = keras.models.load_model('model.h5')
```

**Infrastructure Implications:**
- SavedModel is the standard format for TF Serving
- Includes graph, weights, and training config
- Can be large (100s of MB to GBs)
- Version management is critical

### 3. Inference with TensorFlow

```python
# Inference
predictions = model.predict(input_data)

# For better performance, use call directly
predictions = model(input_data, training=False)
```

**Performance Tips:**
- `model()` is faster than `model.predict()` for small batches
- `training=False` disables training-specific behavior
- Batch inference is crucial for GPU utilization

### 4. TensorFlow Serving Format

```python
# Save for TensorFlow Serving
import tensorflow as tf

# Save with version number (important!)
export_path = './serving_model/1'  # Version 1
model.save(export_path)

# Directory structure:
# serving_model/
#   1/
#     saved_model.pb
#     variables/
#       variables.data-00000-of-00001
#       variables.index
```

**Serving Implications:**
- Version numbers enable safe updates
- TF Serving can load multiple versions
- Directory structure is standardized

### 5. GPU Configuration

```python
# List available GPUs
gpus = tf.config.list_physical_devices('GPU')
print(f"GPUs available: {len(gpus)}")

# Limit GPU memory growth (prevents OOM on shared GPUs)
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Use specific GPU
with tf.device('/GPU:0'):
    predictions = model(input_data)
```

## Model Formats and Conversion

### ONNX (Open Neural Network Exchange)

**Why ONNX Matters:**
- Universal model format
- Convert between frameworks
- Deploy with ONNX Runtime (fast!)

```python
# PyTorch to ONNX
import torch.onnx

dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    export_params=True,
    opset_version=12,
    input_names=['input'],
    output_names=['output']
)

# TensorFlow to ONNX (requires tf2onnx)
import tf2onnx

spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),)
model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec)
with open("model.onnx", "wb") as f:
    f.write(model_proto.SerializeToString())
```

**Infrastructure Benefits:**
- Framework-agnostic serving
- Often faster inference than native frameworks
- Smaller deployment footprint

### Model Size Comparison

| Model Type | PyTorch | TensorFlow | ONNX | TorchScript | Quantized |
|------------|---------|------------|------|-------------|-----------|
| ResNet-50  | 98 MB   | 99 MB      | 97 MB| 98 MB       | 25 MB     |
| BERT Base  | 440 MB  | 438 MB     | 438 MB| 440 MB     | 110 MB    |
| GPT-2 Small| 548 MB  | 550 MB     | 548 MB| 548 MB     | 137 MB    |

## Framework Ecosystem

### PyTorch Ecosystem

**Hugging Face Transformers** (NLP/LLMs)
```python
from transformers import AutoModel, AutoTokenizer

# Load pre-trained model
model = AutoModel.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
```

**PyTorch Lightning** (Training framework)
- Simplified training code
- Multi-GPU support built-in
- Useful for understanding training infrastructure

**TorchServe** (Model serving)
- Official PyTorch serving solution
- We'll use this in later lessons

### TensorFlow Ecosystem

**TensorFlow Hub** (Pre-trained models)
```python
import tensorflow_hub as hub

# Load model from TF Hub
model = hub.load("https://tfhub.dev/google/imagenet/resnet_v2_50/classification/5")
```

**TF Lite** (Mobile/Edge)
- Deploy to mobile devices
- Optimized for edge inference
- Critical for mobile ML applications

**TensorFlow Serving** (Production serving)
- Industry-standard serving solution
- High-performance gRPC and REST APIs

## Infrastructure Considerations by Framework

### Memory Management

**PyTorch:**
- Dynamic memory allocation
- Can lead to OOM if not careful
- Use `torch.cuda.empty_cache()` to free GPU memory

**TensorFlow:**
- Automatic memory management
- Can limit growth with `set_memory_growth()`
- Generally more stable for long-running services

### Multi-GPU Serving

**PyTorch:**
```python
# Multiple models on different GPUs
models = []
for i in range(num_gpus):
    model = create_model()
    model.to(f'cuda:{i}')
    models.append(model)

# Route requests to appropriate GPU
def predict(input_data, gpu_id):
    with torch.cuda.device(gpu_id):
        return models[gpu_id](input_data)
```

**TensorFlow:**
```python
# TensorFlow has built-in multi-GPU support
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = create_model()
```

### Warm-up Requirements

Both frameworks benefit from warm-up:

```python
# PyTorch warm-up
model.eval()
dummy_input = torch.randn(1, 3, 224, 224).to(device)
with torch.no_grad():
    for _ in range(10):  # Run 10 times to warm up
        _ = model(dummy_input)

# TensorFlow warm-up
dummy_input = tf.random.normal((1, 224, 224, 3))
for _ in range(10):
    _ = model(dummy_input, training=False)
```

**Why Warm-up:**
- First inference is 10-100x slower (JIT compilation)
- CUDA kernels need initialization
- Memory allocation happens on first run
- Essential for accurate latency testing

## Choosing Framework for Infrastructure

### Choose PyTorch When:
- ✅ Working with NLP/LLMs (Hugging Face ecosystem)
- ✅ Research-heavy environment
- ✅ Need flexibility and debuggability
- ✅ Python-first deployment is acceptable
- ✅ Working with academic research code

### Choose TensorFlow When:
- ✅ Need mobile/edge deployment (TF Lite)
- ✅ Large-scale production (TF Serving)
- ✅ Enterprise environment with stability focus
- ✅ Google Cloud Platform integration
- ✅ Need JavaScript deployment (TensorFlow.js)

### Framework-Agnostic Approach:
- ✅ Convert to ONNX
- ✅ Use ONNX Runtime for serving
- ✅ Best performance in many cases
- ✅ Framework independence

## Common Infrastructure Issues

### Issue 1: OOM (Out of Memory)

**Symptoms:**
- `CUDA out of memory` error
- Python process killed

**Solutions:**
```python
# Reduce batch size
batch_size = 16  # Try 8, 4, 2, 1

# Use gradient checkpointing (training)
# Use mixed precision (FP16 instead of FP32)
# Clear cache between batches
torch.cuda.empty_cache()

# Monitor memory
print(f"Allocated: {torch.cuda.memory_allocated() / 1e9} GB")
print(f"Cached: {torch.cuda.memory_reserved() / 1e9} GB")
```

### Issue 2: Slow First Inference

**Cause:** Model compilation, CUDA initialization

**Solution:** Always warm up!

### Issue 3: Model-Input Shape Mismatch

**Symptoms:**
- Runtime error about tensor shapes

**Solution:**
```python
# Always verify input shape
expected_shape = (1, 3, 224, 224)
assert input_tensor.shape == expected_shape, f"Expected {expected_shape}, got {input_tensor.shape}"
```

### Issue 4: CPU vs GPU Mismatch

**Symptoms:**
- "Expected tensor on cuda:0 but got tensor on cpu"

**Solution:**
```python
# Ensure consistent device
device = next(model.parameters()).device
input_tensor = input_tensor.to(device)
```

## Practical Exercise

Create a simple inference benchmark:

```python
import time
import torch
import torchvision.models as models

# Load model
model = models.resnet18(pretrained=True)
model.eval()

# Create dummy input
dummy_input = torch.randn(1, 3, 224, 224)

# Warm up
for _ in range(10):
    with torch.no_grad():
        _ = model(dummy_input)

# Benchmark
num_runs = 100
start = time.time()
with torch.no_grad():
    for _ in range(num_runs):
        _ = model(dummy_input)
end = time.time()

avg_time = (end - start) / num_runs
print(f"Average inference time: {avg_time*1000:.2f} ms")
print(f"Throughput: {1/avg_time:.2f} inferences/sec")
```

**Tasks:**
1. Run on CPU and GPU (if available)
2. Compare different batch sizes (1, 8, 16, 32)
3. Test with and without warm-up
4. Try different models (ResNet-18 vs ResNet-50)

## Key Takeaways

1. **Framework choice matters** - PyTorch for research/NLP, TensorFlow for production/mobile
2. **Always use eval mode** - `model.eval()` and `torch.no_grad()`
3. **Warm-up is essential** - First inference is much slower
4. **Batch inference is faster** - Especially on GPU
5. **Memory management matters** - OOM is common with GPUs
6. **Optimization is possible** - TorchScript, quantization, ONNX
7. **Device consistency** - Keep model and inputs on same device

## Self-Check Questions

1. What's the difference between PyTorch and TensorFlow's computation graphs?
2. Why must you call `model.eval()` before inference?
3. What are the two ways to save PyTorch models and when to use each?
4. What is TorchScript and how does it improve inference?
5. What is ONNX and why is it useful for infrastructure?
6. How do you warm up a model and why is it necessary?
7. What's the difference between batch inference and single inference?

## Additional Resources

- [PyTorch Documentation](https://pytorch.org/docs/)
- [TensorFlow Documentation](https://www.tensorflow.org/guide)
- [ONNX Runtime](https://onnxruntime.ai/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [TorchServe Documentation](https://pytorch.org/serve/)
- [TensorFlow Serving](https://www.tensorflow.org/tfx/guide/serving)

---

**Next Lesson:** [05-cloud-intro.md](./05-cloud-intro.md) - Cloud platforms for ML infrastructure