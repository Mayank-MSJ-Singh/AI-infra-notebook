import torch
import torchvision.models as models
import torchvision.transforms as transforms
from typing import List, Tuple, Dict, Any
from PIL import Image
import io
import logging
import os

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