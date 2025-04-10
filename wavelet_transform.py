import torch
import torch.nn as nn
from kymatio.torch import Scattering1D

class WaveletScatteringTransform(nn.Module):
    def __init__(self, 
                 J=6,  # Reduced from 8
                 Q=6,  # Reduced from 8
                 T=16000,  # Signal length
                 max_order=1,  # Reduced from 2
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
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, 1, T)
        
        Returns:
            Scattering coefficients with reduced dimensionality as 4D tensor [B, 3, T, F]
        """
        # Ensure input has the right shape
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        # Apply scattering transform
        Sx = self.scattering(x)
        
        # Original shape from Kymatio: [batch_size, channels, time, frequency]
        B, C, T, F = Sx.shape
        
        # Split channels into 3 groups (low, mid, high frequencies)
        c1 = max(1, C//3)
        c2 = max(1, 2*C//3)
        
        # Average each group along channel dimension to reduce from C to 3 channels
        # Don't use keepdim to avoid extra dimension
        low_freq = Sx[:, :c1, :, :].mean(dim=1)  # [B, T, F]
        mid_freq = Sx[:, c1:c2, :, :].mean(dim=1)  # [B, T, F]
        high_freq = Sx[:, c2:, :, :].mean(dim=1)  # [B, T, F]
        
        # Combine the three frequency bands into channels
        # Result shape: [B, 3, T, F]
        Sx_reduced = torch.stack([low_freq, mid_freq, high_freq], dim=1)
        
        return Sx_reduced