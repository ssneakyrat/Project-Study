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
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, 1, T)
        
        Returns:
            Scattering coefficients with shape (batch_size, 1, time, frequency)
        """
        # Ensure input has the right shape
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        # Apply scattering transform
        Sx = self.scattering(x)
        
        # Keep the 4D shape but reshape to have channel dimension = 1 for Conv2d
        # Original shape from Kymatio: [batch_size, channels, time, frequency]
        B, C, T, F = Sx.shape
        
        # Reshape to [batch_size, 1, time, frequency*channels] to treat channels as frequencies
        # This preserves all dimensions for 2D convolution
        return Sx.reshape(B, 1, T, F*C)