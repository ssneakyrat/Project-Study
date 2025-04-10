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
            Scattering coefficients reshaped for complex CNN
        """
        # Ensure input has the right shape
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        # Apply scattering transform
        Sx = self.scattering(x)
        
        # Reshape the output for the complex CNN
        # Kymatio outputs [batch, order, coef, time] but we need [batch, channels, time]
        if Sx.dim() == 4:
            # Combine order and coef dimensions
            B, O, C, T = Sx.shape
            Sx = Sx.reshape(B, O * C, T)
        
        return Sx