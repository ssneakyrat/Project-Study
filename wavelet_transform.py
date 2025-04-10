import torch
import torch.nn as nn
from kymatio.torch import Scattering1D

class WaveletScatteringTransform(nn.Module):
    def __init__(self, 
                 J=7,  # Increased from 6 to 7 for better frequency resolution
                 Q=12,  # Increased from 8 to 12 for finer frequency bands
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
        
        # Register normalization parameters as buffers with improved initialization
        self.register_buffer('mean', torch.zeros(1))
        self.register_buffer('std', torch.ones(1))
        self.register_buffer('initialized', torch.tensor(0))
        
        # Register per-channel statistics for better normalization
        dummy_input = torch.zeros(1, 1, T)
        with torch.no_grad():
            dummy_output = self.scattering(dummy_input)
            C = dummy_output.shape[1]
            self.register_buffer('channel_mean', torch.zeros(1, C, 1))
            self.register_buffer('channel_std', torch.ones(1, C, 1))
            self.register_buffer('channel_initialized', torch.tensor(0))
            
    def _update_normalization_stats(self, Sx):
        """Update running statistics for better normalization using per-channel stats"""
        if not self.training:
            return
            
        with torch.no_grad():
            # Update global stats
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
                
            # Update per-channel stats for more precise normalization
            batch_channel_mean = torch.mean(Sx, dim=(0, 2), keepdim=True)
            batch_channel_std = torch.std(Sx, dim=(0, 2), keepdim=True)
            
            if self.channel_initialized == 0:
                self.channel_mean = batch_channel_mean
                self.channel_std = batch_channel_std
                self.channel_initialized = torch.tensor(1)
            else:
                # Exponential moving average for channels
                self.channel_mean = (1 - momentum) * self.channel_mean + momentum * batch_channel_mean
                self.channel_std = (1 - momentum) * self.channel_std + momentum * batch_channel_std
        
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
        
        # Apply tapered Hann window to reduce edge effects
        # Using a tapered window (0.92 * Hann + 0.08) preserves more signal at boundaries
        window = 0.92 * torch.hann_window(self.T, device=x.device) + 0.08
        x = x * window.view(1, 1, -1)
        
        # Pad or trim input if necessary to match expected length T
        _, _, L = x.shape
        if L != self.T:
            if L < self.T:
                # Use reflection padding to reduce boundary artifacts
                pad_size = self.T - L
                left_pad = pad_size // 2
                right_pad = pad_size - left_pad
                x = torch.nn.functional.pad(x, (left_pad, right_pad), mode='reflect')
            else:
                # Center-crop to get most important part of signal
                start = (L - self.T) // 2
                x = x[:, :, start:start+self.T]
        
        # Apply scattering transform
        Sx = self.scattering(x)
        
        # Reshape the output for the complex CNN
        if Sx.dim() == 4:
            # Combine order and coef dimensions
            B, O, C, T = Sx.shape
            Sx = Sx.reshape(B, O * C, T)
        
        # Update normalization stats
        self._update_normalization_stats(Sx)
        
        # Apply per-channel normalization for more balanced learning across coefficients
        Sx_normalized = (Sx - self.channel_mean) / (self.channel_std + 1e-8)
        
        # Apply logarithmic scaling to WST coefficients for better dynamic range handling
        # This compresses large coefficient values while preserving small details
        # log(1+x) preserves small values while compressing large ones
        epsilon = 1e-5
        sign = torch.sign(Sx_normalized)
        log_transform = sign * torch.log1p(torch.abs(Sx_normalized) / epsilon) * epsilon
        
        # Weighted blend of original and log-transformed coefficients
        alpha = 0.7  # Balance between original and log-transformed
        Sx_processed = alpha * Sx_normalized + (1 - alpha) * log_transform
        
        return Sx_processed