import torch
import torch.nn as nn

class Encoder(nn.Module):
    """Encoder network for wavelet coefficients
    
    Implements a progressive dimension reduction through hidden layers:
    Forward pass: h_i = σ(W_i·h_{i-1} + b_i) with h_0 = input
    
    Following information bottleneck principle:
    I(X;Z) - β·I(Z;Y) optimized during training
    """
    def __init__(self, input_dim, hidden_dims, latent_dim):
        super().__init__()
        
        # Create sequential layers
        layers = []
        prev_dim = input_dim
        
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            prev_dim = dim
        
        # Final layer to latent space
        layers.append(nn.Linear(prev_dim, latent_dim))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        """Forward pass through encoder"""
        return self.model(x)