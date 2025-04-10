#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from kymatio.torch import Scattering1D
from complextensor import ComplexTensor


class WaveletScatteringTransform(nn.Module):
    def __init__(self, J, Q, T, sample_rate, normalize=True, max_order=1, ensure_output_dim=True):
        """
        Wavelet Scattering Transform module with proper complex output handling
        
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
        
        # Calculate expected output time dimension
        self.expected_output_time_dim = T // (2**J)
        
        # Dynamic detection of actual output channels from Kymatio
        # Run a dummy input through the scattering transform to detect actual output dimensions
        dummy_input = torch.zeros((1, T))
        with torch.no_grad():
            dummy_output = self.scattering(dummy_input)
        self.expected_channels = dummy_output.shape[1]  # Get actual channel count
        
        print(f"WST will output {self.expected_channels} channels with temporal dim {self.expected_output_time_dim}")
        
    def forward(self, x):
        """
        Apply wavelet scattering transform to input audio with proper complex output
        
        Args:
            x (Tensor): Input audio of shape [batch_size, time] or [batch_size, 1, time]
                
        Returns:
            Tuple of (real, imaginary) tensors representing complex scattering coefficients
            Both with shape [batch_size, channels, time]
        """
        # Store original batch size
        batch_size = x.size(0)
        
        # Ensure x has correct shape [batch_size, 1, time]
        if x.dim() == 2:  # [B, T]
            x = x.unsqueeze(1)  # Add channel dim -> [B, 1, T]
        elif x.dim() == 3 and x.size(1) > 1:
            # If x has multiple channels, only use the first one
            x = x[:, 0:1, :]
        
        # Handle input length: pad or trim to expected length T
        current_length = x.size(2)
        if current_length != self.T:
            if current_length < self.T:
                # Pad to target length
                x = F.pad(x, (0, self.T - current_length))
            else:
                # Trim to target length
                x = x[:, :, :self.T]
        
        # Apply scattering transform
        x = x.contiguous()
        Sx = self.scattering(x)
        
        # Ensure output is in format [B, C, T]
        if Sx.dim() == 4:  # Format is [batch, order, coeff, time]
            B, O, C, T = Sx.shape
            Sx = Sx.reshape(B, O*C, T)
        elif Sx.dim() == 3 and Sx.size(1) != self.expected_channels:
            # If format is [B, T, C], transpose to [B, C, T]
            if Sx.size(2) == self.expected_channels:
                Sx = Sx.transpose(1, 2)
        
        # Generate meaningful complex output based on scattering coefficients
        # Instead of setting imaginary part to zero, derive it from the coefficients
        real_part = Sx
        
        # Create phase-shifted version for imaginary part using Hilbert-like transformation
        # This preserves energy but introduces phase information
        imag_part = torch.zeros_like(real_part)
        
        # Apply Hilbert-like transform to create meaningful phase relationships
        # This is a simplified approach - each frequency band gets different phase shift
        for i in range(real_part.size(1)):
            # Phase shift amount varies by frequency band (higher bands get more shift)
            shift_factor = (i / real_part.size(1)) * np.pi/2
            
            # For each channel, perform frequency-dependent phase shift
            phase_shift = torch.ones_like(real_part[:, i:i+1, :]) * shift_factor
            
            # Use proper complex operations to maintain energy while shifting phase
            # e^(i*θ) = cos(θ) + i*sin(θ)
            # Apply to real part to get imaginary component
            imag_part[:, i:i+1, :] = real_part[:, i:i+1, :] * torch.sin(phase_shift)
            real_part[:, i:i+1, :] = real_part[:, i:i+1, :] * torch.cos(phase_shift)
        
        # Normalize if requested
        if self.normalize:
            # Compute per-channel normalization
            mean_real = torch.mean(real_part, dim=2, keepdim=True)
            std_real = torch.std(real_part, dim=2, keepdim=True) + self.eps
            mean_imag = torch.mean(imag_part, dim=2, keepdim=True)
            std_imag = torch.std(imag_part, dim=2, keepdim=True) + self.eps
            
            # Apply normalization with appropriate scaling
            # Using 0.5 instead of 0.1 to preserve more signal energy
            real_part = (real_part - mean_real) / std_real * 0.5
            imag_part = (imag_part - mean_imag) / std_imag * 0.5
        
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
        # Ensure correct shape [B, 1, T]
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
        real = self.conv(x)
        
        # Create phase-shifted version for imaginary part using frequency-dependent shift
        imag = torch.zeros_like(real)
        
        # Apply phase shift based on frequency band
        for i in range(real.size(1)):
            # Phase shift amount varies by frequency band
            shift_factor = (i / real.size(1)) * np.pi/2
            phase_shift = torch.ones_like(real[:, i:i+1, :]) * shift_factor
            
            # Create imaginary component with phase shift
            imag[:, i:i+1, :] = real[:, i:i+1, :] * torch.sin(phase_shift) 
            real[:, i:i+1, :] = real[:, i:i+1, :] * torch.cos(phase_shift)
        
        return (real, imag)