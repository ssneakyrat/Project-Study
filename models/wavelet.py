#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from kymatio.torch import Scattering1D


class WaveletScatteringTransform(nn.Module):
    def __init__(self, J, Q, T, sample_rate, normalize=True, max_order=1, ensure_output_dim=True):
        """
        Wavelet Scattering Transform module with improved dimension handling
        
        Args:
            J (int): Number of scales
            Q (int): Number of wavelets per octave
            T (int): Temporal support of the low-pass filter
            sample_rate (int): Audio sample rate
            normalize (bool): Whether to normalize the output
            max_order (int): Maximum scattering order (1 or 2)
            ensure_output_dim (bool): Force output dimension consistency
        """
        super(WaveletScatteringTransform, self).__init__()
        
        self.J = J
        self.Q = Q
        self.T = T
        self.sample_rate = sample_rate
        self.normalize = normalize
        self.max_order = max_order
        self.ensure_output_dim = ensure_output_dim
        
        # Initialize Kymatio scattering transform
        self.scattering = Scattering1D(J=J, shape=T, Q=Q, max_order=max_order)
        
        # Register as buffer to move to correct device
        self.register_buffer('eps', torch.tensor(1e-8))
        
        # Calculate expected output shape for dimension consistency
        self.expected_output_time_dim = T // (2**J)
        
        # Use optimal padding computation based on J
        self.optimal_pad = 2**J
        
    def forward(self, x):
        """
        Apply wavelet scattering transform to input audio with simplified dimension handling
        
        Args:
            x (Tensor): Input audio of shape [batch_size, time] or [batch_size, 1, time]
                
        Returns:
            Tuple of (real, imaginary) tensors representing complex scattering coefficients
        """
        # Store original batch size
        batch_size = x.size(0)
        
        # Ensure x has correct shape [batch_size, 1, time]
        if x.dim() == 2:
            x = x.unsqueeze(1)
        elif x.dim() == 3 and x.size(1) > 1:
            # If x has shape [batch_size, channels, time] with channels > 1
            # we process only the first channel
            x = x[:, 0:1, :]
        
        # Calculate optimal padding to ensure dimensions work with wavelet scales
        target_length = self.T
        current_length = x.size(2)
        
        # Compute padding to nearest multiple of 2^J
        pad_length = 0
        if current_length % self.optimal_pad != 0:
            pad_length = self.optimal_pad - (current_length % self.optimal_pad)
        
        # Add extra padding for consistent output dimensions
        original_signal_padded = False
        if current_length < target_length:
            # Pad to target length
            padding = target_length - current_length
            x = F.pad(x, (0, padding + pad_length))
            original_signal_padded = True
        elif current_length > target_length:
            # For longer signals, segment processing would be better
            # but for this fix we'll truncate and note the issue
            print(f"Warning: Input signal length {current_length} exceeds target {target_length}.")
            x = x[:, :, :target_length]
            # Still apply optimal padding
            x = F.pad(x, (0, pad_length))
        else:
            # Just apply optimal padding
            x = F.pad(x, (0, pad_length))
        
        # Store the padding info for potential reconstruction
        self.pad_info = {
            'original_length': current_length,
            'padded_length': x.size(2),
            'target_length': target_length,
            'was_padded': original_signal_padded
        }
        
        print(f"WST input after padding: {x.shape}")
        
        # Apply scattering transform
        x = x.contiguous()  # Ensure tensor is memory-contiguous
        Sx = self.scattering(x)
        
        print(f"Raw scattering output shape: {Sx.shape}")
        
        # SIMPLIFIED APPROACH: Handle different output formats more robustly
        # Reshape to [batch, channels, time] without complex reshaping logic
        
        if Sx.dim() == 4:
            # Format is [batch, order, coeff, time]
            
            # SIMPLE FIX: Just flatten the middle two dimensions
            # This avoids complex channel extraction logic that might break
            B, O, C, T = Sx.shape
            Sx = Sx.reshape(B, O*C, T)
            print(f"Reshaped scattering output: {Sx.shape}")
            
            # Real part is the reshaped Sx, imaginary is zeros with same shape
            real_part = Sx
            imag_part = torch.zeros_like(real_part)
        else:
            # Standard format
            real_part = Sx
            imag_part = torch.zeros_like(real_part)
        
        # Normalize if requested
        if self.normalize:
            # Compute per-channel normalization for better stability
            mean_value = torch.mean(torch.abs(real_part), dim=2, keepdim=True)
            std_value = torch.std(real_part, dim=2, keepdim=True) + 1e-6
            
            # Simple standardization for numerical stability
            real_part = (real_part - mean_value) / std_value * 0.1
            imag_part = imag_part / (std_value + self.eps) * 0.1
        
        # Print final dimensions for debugging
        print(f"Final WST output - Real: {real_part.shape}, Imag: {imag_part.shape}")
        
        return (real_part, imag_part)


class ParametricWaveletTransform(nn.Module):
    """
    A learnable version of wavelet scattering transform where filter parameters
    are learned during training
    """
    def __init__(self, J, Q, T, sample_rate, init_from_scattering=True, ensure_output_dim=True):
        """
        Initialize learnable wavelet transform
        
        Args:
            J (int): Number of scales
            Q (int): Number of wavelets per octave
            T (int): Temporal support
            sample_rate (int): Audio sample rate
            init_from_scattering (bool): Whether to initialize from Kymatio
            ensure_output_dim (bool): Force output dimension consistency
        """
        super(ParametricWaveletTransform, self).__init__()
        
        self.J = J
        self.Q = Q 
        self.T = T
        self.sample_rate = sample_rate
        self.ensure_output_dim = ensure_output_dim
        
        # Expected output time dimension
        self.expected_output_time_dim = T // (2**J)
        
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
        
        # Ensure correct time dimension
        if x.size(2) != self.T:
            if x.size(2) < self.T:
                # Pad
                x = F.pad(x, (0, self.T - x.size(2)))
            else:
                # Truncate
                x = x[:, :, :self.T]
        
        # Apply filters
        output = self.conv(x)
        
        # Ensure consistent output dimensions if flag is set
        if self.ensure_output_dim:
            T_out = output.size(2)
            if T_out != self.expected_output_time_dim:
                if T_out > self.expected_output_time_dim:
                    # Trim output
                    output = output[:, :, :self.expected_output_time_dim]
                else:
                    # Pad output
                    pad_size = self.expected_output_time_dim - T_out
                    output = F.pad(output, (0, pad_size))
        
        # Split into real and imaginary (approximation)
        # In a real implementation, we would use Hilbert transform 
        # or other techniques to get the imaginary part
        real = output
        imag = torch.zeros_like(real)
        
        return (real, imag)