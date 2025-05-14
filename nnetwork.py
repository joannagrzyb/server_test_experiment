from torch import nn
import torch.nn.functional as F


class NeuralNetwork(nn.Module):
    def __init__(self, input_size, num_classes, device):
        super(NeuralNetwork, self).__init__()
        
        self.device = device  # Store the device
        self.fc1 = nn.Linear(input_size, 50, device=device)  # Fully connected layer 1
        self.fc2 = nn.Linear(50, num_classes, device=device)  # Fully connected layer 2
    
    def forward(self, x):
        x = F.relu(self.fc1(x))  # Apply ReLU activation
        x = self.fc2(x)  # Output layer
        return x
