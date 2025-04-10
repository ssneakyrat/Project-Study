import torch
import torch.nn as nn
from complextensor import ComplexTensor
from complex_ops import ComplexConv1d, complex_leaky_relu, ComplexLayerNorm

class ComplexEncoder(nn.Module):
    def __init__(self, input_channels, hidden_dims=[64, 128, 256]):
        super().__init__()
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(
            nn.Sequential(
                ComplexConv1d(input_channels, hidden_dims[0], kernel_size=7, stride=2, padding=3),
                ComplexLayerNorm([hidden_dims[0]]),
            )
        )
        
        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            self.layers.append(
                nn.Sequential(
                    ComplexConv1d(hidden_dims[i], hidden_dims[i+1], kernel_size=5, stride=2, padding=2),
                    ComplexLayerNorm([hidden_dims[i+1]]),
                )
            )
    
    def forward(self, x):
        """
        x: Tensor of shape [batch_size, input_channels, length]
        Returns:
            z: Encoded representation
            features: List of intermediate features for skip connections
        """
        features = []
        
        # Convert to ComplexTensor if not already
        if not isinstance(x, ComplexTensor):
            x = ComplexTensor(x, torch.zeros_like(x))
        
        # Apply layers and store features for skip connections
        for layer in self.layers:
            x = layer(x)
            x = complex_leaky_relu(x)
            features.append(x)
            
        return x, features