import torch
import torch.nn as nn
import torch.nn.functional as F

class LightweightWaveletEncoder(nn.Module):
    """Efficient encoder optimized for wavelet coefficients with focus on approximation components
    
    Mathematical basis:
    - Energy compaction: Approximation coefficients contain ~80-90% of signal energy
    - Dimensionality reduction: R(D) = min_{p(z|x): E[d(x,z)]≤D} I(X;Z)
    """
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Calculate level dimensions (estimated approximation vs detail split)
        approx_dim = min(input_dim // 4, 512)  # Limit size for very large inputs
        detail_dim = input_dim - approx_dim
        
        # Create separate pathways for approximation and detail coefficients
        self.approx_encoder = nn.Sequential(
            nn.Linear(approx_dim, min(256, approx_dim // 2)),
            nn.LeakyReLU(0.2),
            nn.Linear(min(256, approx_dim // 2), latent_dim // 2),
        )
        
        # Simplified encoder for detail coefficients (just a single projection)
        self.detail_encoder = nn.Linear(detail_dim, latent_dim // 2)
        
        # Optional normalization layer to stabilize training
        self.norm = nn.LayerNorm(latent_dim)
        
    def forward(self, x):
        """Encode wavelet coefficients with priority on approximation coefficients
        
        Args:
            x: Wavelet coefficients [batch_size, input_dim]
            
        Returns:
            Latent representation [batch_size, latent_dim]
        """
        # Split input into approximation and detail parts
        approx_dim = min(self.input_dim // 4, 512)
        
        # Handle case where input is smaller than expected
        if x.shape[1] < approx_dim:
            # Fall back to simple encoding
            return self.detail_encoder(x)
        
        # Split coefficients
        approx_coeff = x[:, :approx_dim]
        detail_coeff = x[:, approx_dim:]
        
        # Encode approximation coefficients with the dedicated path
        approx_z = self.approx_encoder(approx_coeff)
        
        # Encode detail coefficients with the simpler path
        detail_z = self.detail_encoder(detail_coeff)
        
        # Combine latent representations
        z = torch.cat([approx_z, detail_z], dim=1)
        
        # Apply normalization
        z = self.norm(z)
        
        return z


class LightweightWaveletDecoder(nn.Module):
    """Efficient decoder optimized for wavelet coefficients with focus on approximation components
    
    Mathematical basis:
    - Inverse energy compaction: Focus computational resources on perceptually important components
    - Rate-distortion theory: Allocate bits according to coefficient importance
    """
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        
        # Calculate level dimensions (estimated approximation vs detail split)
        approx_dim = min(output_dim // 4, 512)  # Limit size for very large outputs
        detail_dim = output_dim - approx_dim
        
        # Create separate pathways for approximation and detail coefficients
        self.approx_decoder = nn.Sequential(
            nn.Linear(latent_dim // 2, min(256, approx_dim // 2)),
            nn.LeakyReLU(0.2),
            nn.Linear(min(256, approx_dim // 2), approx_dim),
        )
        
        # Simplified decoder for detail coefficients (just a single projection)
        self.detail_decoder = nn.Linear(latent_dim // 2, detail_dim)
        
    def forward(self, z):
        """Decode latent representation with priority on approximation coefficients
        
        Args:
            z: Latent representation [batch_size, latent_dim]
            
        Returns:
            Reconstructed wavelet coefficients [batch_size, output_dim]
        """
        # Split latent representation in half for both pathways
        z_approx = z[:, :self.latent_dim // 2]
        z_detail = z[:, self.latent_dim // 2:]
        
        # Calculate output dimensions
        approx_dim = min(self.output_dim // 4, 512)
        detail_dim = self.output_dim - approx_dim
        
        # Decode approximation coefficients with more complex path
        approx_coeffs = self.approx_decoder(z_approx)
        
        # Decode detail coefficients with simpler path
        detail_coeffs = self.detail_decoder(z_detail)
        
        # Combine coefficients
        output = torch.cat([approx_coeffs, detail_coeffs], dim=1)
        
        return output


class ConvWaveletAutoencoder(nn.Module):
    """Lightweight wavelet autoencoder that prioritizes approximation coefficients
    
    Key efficiency improvements:
    1. Separate pathways for approximation and detail coefficients
    2. Heavier computation only for approximation coefficients
    3. Minimal processing for detail coefficients
    4. Reduced parameter count through simplified architecture
    """
    def __init__(self, input_dim, latent_dim, base_channels=16):
        super().__init__()
        
        # Create efficient encoder and decoder
        self.encoder = LightweightWaveletEncoder(input_dim, latent_dim)
        self.decoder = LightweightWaveletDecoder(latent_dim, input_dim)
    
    def forward(self, x):
        """Forward pass through autoencoder
        
        Args:
            x: Input wavelet coefficients [batch_size, input_dim]
            
        Returns:
            Tuple of (reconstructed_coefficients, latent_code)
        """
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z