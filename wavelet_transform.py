import torch
import torch.nn as nn
from kymatio.torch import Scattering1D

class WaveletScatteringTransform(nn.Module):
    def __init__(self, 
                 J=8,  # Number of scales
                 Q=8,  # Number of wavelets per octave
                 T=16000,  # Signal length
                 max_order=2,  # Maximum scattering order
                 out_type='array'):
        """Wavelet Scattering Transform module.
        
        Args:
            J: Number of scales
            Q: Number of wavelets per octave
            T: Signal length
            max_order: Maximum scattering order
            out_type: Output format
        """
        super().__init__()
        self.scattering = Scattering1D(J=J, shape=T, Q=Q, max_order=max_order, out_type=out_type)
        self.J = J
        self.Q = Q
        self.T = T
        self.max_order = max_order
        self.out_type = out_type
        
        # Precompute coefficients count for each order
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, T)
            Sx = self.scattering(dummy_input)
            if Sx.dim() == 4:
                self.total_coeffs = Sx.shape[1] * Sx.shape[2]
            else:
                self.total_coeffs = Sx.shape[1]
            
            # Calculate normalization statistics for more stable training
            random_input = torch.randn(16, 1, T)
            random_sx = self.scattering(random_input)
            
            # Compute mean magnitude of coefficients for better normalization
            self.scale_factor = torch.mean(torch.abs(random_sx)).item()
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, 1, T) or (batch_size, T)
        
        Returns:
            Scattering coefficients reshaped for complex CNN
        """
        # Ensure input has the right shape
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        # Pad or trim input if necessary to match expected length T
        _, _, L = x.shape
        if L != self.T:
            if L < self.T:
                # Pad
                pad_size = self.T - L
                x = torch.nn.functional.pad(x, (0, pad_size))
            else:
                # Trim
                x = x[:, :, :self.T]
        
        # Apply scattering transform
        Sx = self.scattering(x)
        
        # Reshape the output for the complex CNN
        # Kymatio outputs [batch, order, coef, time] but we need [batch, channels, time]
        if Sx.dim() == 4:
            # Combine order and coef dimensions
            B, O, C, T = Sx.shape
            Sx = Sx.reshape(B, O * C, T)
        
        # Apply proper normalization to preserve phase information
        # Avoid the previous log1p transform which distorts phase
        Sx = Sx / (self.scale_factor + 1e-8)
        
        return Sx