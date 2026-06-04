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
