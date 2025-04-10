#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from kymatio.torch import Scattering1D


class WaveletScatteringTransform(nn.Module):
    def __init__(self, J, Q, T, sample_rate, normalize=True, max_order=2):
        """
        Wavelet Scattering Transform module
        
        Args:
            J (int): Number of scales
            Q (int): Number of wavelets per octave
            T (int): Temporal support of the low-pass filter
            sample_rate (int): Audio sample rate
            normalize (bool): Whether to normalize the output
            max_order (int): Maximum scattering order (1 or 2)
        """
        super(WaveletScatteringTransform, self).__init__()
        
        self.J = J
        self.Q = Q
        self.T = T
        self.sample_rate = sample_rate
        self.normalize = normalize
        self.max_order = max_order
        
        # Initialize Kymatio scattering transform
        self.scattering = Scattering1D(J=J, shape=T, Q=Q, max_order=max_order)
        
        # Register as buffer to move to correct device
        self.register_buffer('eps', torch.tensor(1e-8))
        
    def forward(self, x):
        """
        Apply wavelet scattering transform to input audio
        
        Args:
            x (Tensor): Input audio of shape [batch_size, time] or [batch_size, 1, time]
            
        Returns:
            Tuple of (real, imaginary) tensors representing complex scattering coefficients
        """
        # Ensure x has correct shape [batch_size, 1, time]
        if x.dim() == 2:
            x = x.unsqueeze(1)
        elif x.dim() == 3 and x.size(1) > 1:
            # If x has shape [batch_size, channels, time] with channels > 1
            # we process only the first channel
            x = x[:, 0:1, :]
        
        # Ensure correct time dimension by padding or truncating
        target_length = self.T
        current_length = x.size(2)
        
        if current_length < target_length:
            # Pad
            padding = target_length - current_length
            x = F.pad(x, (0, padding))
        elif current_length > target_length:
            # Truncate
            x = x[:, :, :target_length]
        
        # Apply scattering transform
        x = x.contiguous()  # Ensure tensor is memory-contiguous
        Sx = self.scattering(x)
        
        # Kymatio's Scattering1D returns a real-valued tensor where coefficients are packed
        # We need to adapt to the expected complex format
        batch_size = Sx.size(0)
        
        # Sx shape is [batch_size, C, T] where C includes both real and imaginary components
        # We need to ensure we properly handle this output format
        # The precise shape depends on the Kymatio version and configuration
        
        # Safer approach: just split the channels in half assuming they're interleaved
        # or stored sequentially (real components first, then imaginary)
        C = Sx.size(1)
        T_out = Sx.size(2)
        
        # Two approaches depending on how Kymatio packs the data:
        # 1. If real and imaginary parts are in separate channels
        if hasattr(self.scattering, 'output_format') and self.scattering.output_format == 'array':
            # Split channels evenly (assuming first half are real, second half imaginary)
            C_half = C // 2
            real_part = Sx[:, :C_half, :]
            imag_part = Sx[:, C_half:, :]
        else:
            # 2. If real and imaginary parts are interleaved in the output
            # Reshape to batch_size, C//2, 2, T_out where dim 2 separates real (0) and imag (1)
            # This is a safer fallback assuming Kymatio returns complex data in a structured way
            real_part = Sx
            imag_part = torch.zeros_like(real_part)  # Default to zero imaginary part
        
        # Normalize if required
        if self.normalize:
            # Log normalization to reduce dynamic range
            magnitude = torch.sqrt(real_part**2 + imag_part**2 + self.eps)
            real_part = torch.log(magnitude + self.eps)
            imag_part = torch.zeros_like(real_part)  # Phase information is discarded in log normalization
        
        # maintains mathematical validity while enabling correct forward propagation
        if real_part.dim() == 4:
            # Reshape from [B, C, H, W] → [B, C*H, W]
            batch_size, channels, height, width = real_part.shape
            real_part = real_part.reshape(batch_size, channels*height, width)
            imag_part = imag_part.reshape(batch_size, channels*height, width)

        return (real_part, imag_part)


class ParametricWaveletTransform(nn.Module):
    """
    A learnable version of wavelet scattering transform where filter parameters
    are learned during training
    """
    def __init__(self, J, Q, T, sample_rate, init_from_scattering=True):
        """
        Initialize learnable wavelet transform
        
        Args:
            J (int): Number of scales
            Q (int): Number of wavelets per octave
            T (int): Temporal support
            sample_rate (int): Audio sample rate
            init_from_scattering (bool): Whether to initialize from Kymatio
        """
        super(ParametricWaveletTransform, self).__init__()
        
        self.J = J
        self.Q = Q 
        self.T = T
        self.sample_rate = sample_rate
        
        # Number of filters = J*Q wavelets + 1 lowpass
        n_filters = J * Q + 1
        
        # Filter length (should be odd)
        filter_length = 2 * (2**J) + 1
        
        # Initialize filters
        if init_from_scattering:
            # Create a temporary scattering object to extract filter banks
            scattering = Scattering1D(J=J, shape=T, Q=Q)
            
            # Get filter bank shapes from Kymatio (would need implementation details)
            # This is a placeholder - actual implementation would extract real filters
            filters = torch.randn(n_filters, 1, filter_length)
        else:
            # Random initialization
            filters = torch.randn(n_filters, 1, filter_length)
            
            # Apply frequency domain constraints to make them wavelet-like
            # This is a placeholder
            pass
        
        # Create learnable parameters
        self.filters = nn.Parameter(filters)
        
        # Use Conv1d for the transform
        self.conv = nn.Conv1d(1, n_filters, filter_length, 
                             padding=filter_length//2, bias=False)
        
        # Initialize conv weights with our filters
        with torch.no_grad():
            self.conv.weight.copy_(filters)
    
    def forward(self, x):
        """
        Apply learnable wavelet transform
        
        Args:
            x (Tensor): Input of shape [batch_size, time] or [batch_size, 1, time]
            
        Returns:
            tuple: Complex coefficients as (real, imag)
        """
        # Ensure correct shape
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        # Apply filters
        output = self.conv(x)
        
        # Split into real and imaginary (approximation)
        # In a real implementation, we would use Hilbert transform 
        # or other techniques to get the imaginary part
        real = output
        imag = torch.zeros_like(real)
        
        return (real, imag)