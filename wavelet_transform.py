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
            Scattering coefficients
        """
        # Ensure input has the right shape
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        # Apply scattering transform
        Sx = self.scattering(x)
        
        # Reshape for the complex CNN
        # The output of the scattering transform is a 3D tensor with shape (batch, channels, time)
        # We need to reshape it to be compatible with our complex CNN
        B = Sx.shape[0]  # Batch size
        
        # Get the number of channels from the scattering transform output
        C = Sx.shape[1]
        
        # Time dimension (if out_type is 'array')
        T = Sx.shape[2]
        
        # Return as a tensor ready for the complex CNN
        return Sx