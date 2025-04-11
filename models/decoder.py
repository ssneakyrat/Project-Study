import torch
import torch.nn as nn

class Decoder(nn.Module):
    """Decoder network for reconstructing wavelet coefficients
    
    Reconstruction: ĥ_i = σ(Ŵ_i·ĥ_{i-1} + b̂_i) with ĥ_0 = z
    
    Objective: min E[||W - Ŵ||²]
    
    Mirror of encoder with progressive expansion:
    [latent_dim → ... → hidden_2 → hidden_1 → total_coeffs]
    """
    def __init__(self, latent_dim, hidden_dims, output_dim):
        super().__init__()
        
        # Create sequential layers (in reverse order of encoder)
        layers = []
        prev_dim = latent_dim
        
        for dim in reversed(hidden_dims):
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            prev_dim = dim
        
        # Final layer to output space
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        """Forward pass through decoder"""
        return self.model(x)