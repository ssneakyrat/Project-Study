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
        self.J = J
        self.Q = Q
        self.T = T
        self.max_order = max_order
        self.out_type = out_type
        self._create_scattering()
        
    def _create_scattering(self, T=None):
        """Create or recreate scattering object with possibly updated T"""
        T = T if T is not None else self.T
        self.scattering = Scattering1D(J=self.J, shape=T, Q=self.Q, 
                                       max_order=self.max_order, out_type=self.out_type)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, 1, T)
        
        Returns:
            Scattering coefficients as 4D tensor [B, 3, T, F]
        """
        # Ensure input has the right shape
        if x.dim() == 2:
            x = x.unsqueeze(1)
            
        # Check if we need to adapt to a different signal length
        current_T = x.shape[-1]
        expected_T = self.T
        
        # If signal is too short, pad it
        if current_T < expected_T:
            pad_size = expected_T - current_T
            x = torch.nn.functional.pad(x, (0, pad_size))
            
        # If signal length doesn't match scattering expectation, recreate scattering
        if current_T != expected_T and current_T > self.J * (2**self.J):
            # Only update if signal is long enough for the current J value
            self._create_scattering(current_T)
            self.T = current_T
        
        try:
            # Apply scattering transform
            Sx = self.scattering(x)
            
            # Print shape for debugging
            # Shape should be [batch_size, channels, time, frequency]
            # print(f"Scattering output shape: {Sx.shape}")
            
        except ValueError as e:
            if "Indefinite padding size" in str(e):
                # Handle case where signal is too short for current J
                # Reduce J and recreate scattering
                reduced_J = max(1, self.J - 1)
                temp_scattering = Scattering1D(J=reduced_J, shape=current_T, 
                                               Q=self.Q, max_order=self.max_order, 
                                               out_type=self.out_type)
                Sx = temp_scattering(x)
            else:
                raise e
        
        # Original shape from Kymatio: [batch_size, channels, time, frequency]
        B, C, T, F = Sx.shape
        
        # Split channels into 3 groups (low, mid, high frequencies)
        c1 = max(1, C//3)
        c2 = max(1, 2*C//3)
        
        # Average each group along channel dimension to reduce from C to 3 channels
        # Don't use keepdim to avoid extra dimension
        low_freq = Sx[:, :c1, :, :].mean(dim=1, keepdim=True)  # [B, 1, T, F]
        mid_freq = Sx[:, c1:c2, :, :].mean(dim=1, keepdim=True)  # [B, 1, T, F]
        high_freq = Sx[:, c2:, :, :].mean(dim=1, keepdim=True)  # [B, 1, T, F]
        
        # Combine the three frequency bands into channels
        # Result shape: [B, 3, T, F]
        Sx_reduced = torch.cat([low_freq, mid_freq, high_freq], dim=1)
        
        return Sx_reduced