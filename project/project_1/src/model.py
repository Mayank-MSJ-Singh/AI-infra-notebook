import torch
import torchvision.models as models
import torchvision.transforms as transforms
from typing import List, Tuple, Dict, Any
from PIL import Image
import io
import logging
import os

import requests
import numpy as np

logger = logging.getLogger(__name__)

class ModelInference:

    def __init__(self, model_name: str = "resnet18"):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.preprocess = self._create_transforms()
        self.classes = self._load_class_labels()

        logger.info(f"Initialized ModelInference with {self.model_name} on {self.device}")

    def load_model(self) -> None:
        try:
            if self.model_name == "resnet18":
                self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            elif self.model_name == "resnet50":
                self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            else:
                raise ValueError(f"Unsupported model: {self.model_name}")
        
            self.model.eval()
            self.model.to(self.device)

            dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
            with torch.no_grad():
                _ = self.model(dummy_input)
            
            logger.info(f"Model {self.model_name} loaded successfully")

        except ValueError as e:
            logger.error(f"Invalid model name: {e}")
            raise

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Model loading failed: {e}")

    def _create_transforms(self) -> transforms.Compose:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])  
    
    def _load_class_labels(self) -> List[str]:
        try:
            class_file = "imagenet_classes.txt"
            if os.path.exists(class_file):
                with open(class_file, 'r') as f:
                    return [line.strip() for line in f.readlines()]

            else:
                import urllib.request
                url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
                with urllib.request.urlopen(url) as response:
                    classes = [line.decode('utf-8').strip() for line in response]
                return classes

        except Exception as e:
            logger.error(f"Failed to load class labels: {e}")
            return [f"class_{i}" for i in range(1000)]        
        
    def preprocess_image(self, image_bytes: bytes) -> torch.Tensor:
        try:
            image = Image.open(io.BytesIO(image_bytes))

            if image.mode != 'RGB':
                image = image.convert('RGB')

            input_tensor = self.preprocess(image)

            # Add batch dimension and move to device
            input_batch = input_tensor.unsqueeze(0).to(self.device)

            return input_batch
        except Exception as e:
            logger.error(f"Failed to preprocess image: {e}")
            raise RuntimeError(f"Image preprocessing failed: {e}")    

    def predict(
        self,
        image_bytes: bytes,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:

        logger.info(f"Starting prediction for {self.model_name}")
        try:
            if self.model is None:
                raise RuntimeError("Model not loaded. Call load_model() first.")

            if not 1 <= top_k <= 10:
                raise ValueError("top_k must be between 1 and 10")

                    # Preprocess
            input_batch = self.preprocess_image(image_bytes)

            # Inference
            with torch.no_grad():
                output = self.model(input_batch)

            # Get probabilities
            probabilities = torch.nn.functional.softmax(output[0], dim=0)

            # Get top-k
            top_probs, top_indices = torch.topk(probabilities, top_k)

            # Format results
            predictions = []
            for i in range(top_k):
                class_id = int(top_indices[i])
                predictions.append({
                    'class_id': class_id,
                    'class_name': self.classes[class_id],
                    'confidence': float(top_probs[i])
                })
            logger.debug(f"Running prediction with top_k={top_k}")

            return predictions

        except Exception as e:
            logger.error(f"Inference failed: {e}")
            raise RuntimeError(f"Inference failed: {e}")

    
    def predict_from_url(
        self,
        image_url: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        try:
            if not image_url:
                raise ValueError("image_url is required")
            
            if not 1 <= top_k <= 10:
                raise ValueError("top_k must be between 1 and 10")
            
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()

            content_type = response.headers.get("Content-Type", "")
            if not content_type.startswith("image/"):
                raise ValueError(f"URL does not point to an image: {content_type}")

            image_bytes = response.content

            return self.predict(image_bytes, top_k)
        
        except ValueError as e:
            logger.error(f"Invalid value: {e}")
            raise

        except requests.exceptions.HTTPError as e:
            logger.error(f"Invalid URL or content type: {e}")
            raise

        except Exception as e:
            logger.error(f"Failed to download image from URL: {e}")
            raise RuntimeError(f"Image download failed: {e}")

    def get_model_info(self) -> Dict[str, Any]:
        return {
            'model_name': self.model_name,
            'device': str(self.device),
            'loaded': self.model is not None,
            'num_classes': len(self.classes) if self.classes else 0,
            'input_size': (224, 224),
            'framework': 'PyTorch',
            'version': torch.__version__
        }

def load_model(
    model_name: str = "resnet18"
    ) -> ModelInference:

        inference = ModelInference(model_name)
        inference.load_model()
        return inference

def validate_image(image_bytes: bytes) -> bool:
        try:
            image = Image.open(io.BytesIO(image_bytes))
            image.verify()
            return True
        except Exception:
            return False

def get_supported_models() -> List[str]:
        return [
            'resnet18',
            'resnet34',
            'resnet50',
            'resnet101',
            'resnet152',
            'mobilenet_v2',
            'efficientnet_b0'
        ]

def test_model_loading():
        print("Testing model loading...")

        model = ModelInference("resnet18")
        model.load_model()

        assert model.model is not None, "Model not loaded"
        assert model.classes is not None, "Classes not loaded"

        dummy_image = Image.fromarray(
            np.random.randint(
                0, 255,
                (224, 224, 3),
                dtype=np.uint8
            )
        )

        buffer = io.BytesIO()
        dummy_image.save(buffer, format="JPEG")
        image_bytes = buffer.getvalue()

        predictions = model.predict(
            image_bytes,
            top_k=5
        )

        assert len(predictions) == 5
        assert all(
            "class_name" in p
            for p in predictions
        )
        assert all(
            "confidence" in p
            for p in predictions
        )

        print("All tests passed!")

if __name__ == "__main__":
    test_model_loading()