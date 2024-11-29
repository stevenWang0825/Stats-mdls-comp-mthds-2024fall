import torch.nn.functional as F
import torch.nn as nn
import random
import torch
import numpy as np
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Make a CNN for classifying MNIST
class SimpleMNISTNet(nn.Module):
    def __init__(self):
        super(SimpleMNISTNet, self).__init__()
            # Convolutional layers
        self.conv_layers = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),  # Output: [32, 28, 28]
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),                 # Output: [32, 14, 14]
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), # Output: [64, 14, 14]
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)                  # Output: [64, 7, 7]
        )
        # Fully connected layers
        self.fc_layers = nn.Sequential(
                nn.Flatten(),                                         # Flatten: [64 * 7 * 7]
                nn.Linear(64 * 7 * 7, 128),                           # Fully connected layer
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 10)                                    # Output: 10 classes
        )
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x