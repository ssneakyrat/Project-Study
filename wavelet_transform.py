import torch
import torch.nn as nn
from kymatio.torch import Scattering1D

class WaveletScatteringTransform(nn.Module):
    def __init__(self, 
                 J=6,  # Reduced from 8 to 6 for better time resolution
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
        
        # Register normalization parameters as buffers
        self.register_buffer('mean', torch.zeros(1))
        self.register_buffer('std', torch.ones(1))
        self.register_buffer('initialized', torch.tensor(0))
        
    def _update_normalization_stats(self, Sx):
        """Update running statistics for better normalization"""
        if not self.training:
            return
            
        with torch.no_grad():
            # Update mean and std for more stable training
            batch_mean = torch.mean(Sx)
            batch_std = torch.std(Sx)
            
            if self.initialized == 0:
                self.mean = batch_mean
                self.std = batch_std
                self.initialized = torch.tensor(1)
            else:
                # Exponential moving average
                momentum = 0.1
                self.mean = (1 - momentum) * self.mean + momentum * batch_mean
                self.std = (1 - momentum) * self.std + momentum * batch_std
        
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
        
        # Apply Hann window to reduce edge effects
        window = torch.hann_window(self.T, device=x.device)
        x = x * window.view(1, 1, -1)
        
        # Pad or trim input if necessary to match expected length T
        _, _, L = x.shape
        if L != self.T:
            if L < self.T:
                # Pad with reflection padding to reduce boundary artifacts
                pad_size = self.T - L
                x = torch.nn.functional.pad(x, (0, pad_size), mode='reflect')
            else:
                # Trim
                x = x[:, :, :self.T]
        
        # Apply scattering transform
        Sx = self.scattering(x)
        
        # Reshape the output for the complex CNN
        if Sx.dim() == 4:
            # Combine order and coef dimensions
            B, O, C, T = Sx.shape
            Sx = Sx.reshape(B, O * C, T)
        
        # Update normalization stats
        self._update_normalization_stats(Sx)
        
        # Apply proper normalization
        Sx_normalized = (Sx - self.mean) / (self.std + 1e-8)
        
        return Sx_normalized