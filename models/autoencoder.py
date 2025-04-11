import torch
import torch.nn as nn
from models.encoder import Encoder
from models.decoder import Decoder

class WaveletAutoencoder(nn.Module):
    """Combined encoder-decoder model for wavelet coefficients
    
    Follows rate-distortion theory principles:
    R(D) = min_{p(x̂|x): E[d(x,x̂)] ≤ D} I(X;X̂)
    
    Compression ratio: audio_length / latent_dim
    """
    def __init__(self, input_dim, hidden_dims, latent_dim):
        super().__init__()
        
        self.encoder = Encoder(input_dim, hidden_dims, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dims, input_dim)
    
    def forward(self, x):
        """Forward pass through autoencoder"""
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z