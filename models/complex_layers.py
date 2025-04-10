#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from complextensor import ComplexTensor


class ComplexConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        """
        Complex-valued 1D convolution layer
        
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            kernel_size (int): Size of the convolving kernel
            stride (int): Stride of the convolution
            padding (int): Zero-padding added to both sides of the input
            dilation (int): Spacing between kernel elements
            groups (int): Number of blocked connections from input to output channels
            bias (bool): If True, adds a learnable bias to the output
        """
        super(ComplexConv1d, self).__init__()
        
        # Real and imaginary components of weights
        self.conv_re = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.conv_im = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        
        # Save configuration for reference
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
    def forward(self, x):
        """
        Forward pass with consistent dimension handling
        
        Args:
            x: Either a ComplexTensor, tuple of (real, imag) tensors, or real tensor
               Expected shape: [B, C, T] for each tensor component
               
        Returns:
            tuple: (real, imag) output tensors with shape [B, C_out, T_out]
        """
        if isinstance(x, ComplexTensor):
            x_real, x_imag = x.real, x.imag
        elif isinstance(x, tuple):
            x_real, x_imag = x
        else:  # Handle the case where input is real
            x_real, x_imag = x, torch.zeros_like(x)
        
        # First handle extra dimensions
        if x_real.dim() > 3:
            print(f"WARNING: Input has too many dimensions: {x_real.shape}, squeezing...")
            x_real = x_real.squeeze(2)
            x_imag = x_imag.squeeze(2)
        
        # CRITICAL DIMENSION HANDLING - Ensure [B, C, T] format
        # Check if channels and time are swapped and fix if needed
        if x_real.dim() == 3 and x_real.size(2) == self.in_channels and x_real.size(1) != self.in_channels:
            print(f"WARNING: Input has swapped dimensions [B, T, C]: {x_real.shape}, transposing...")
            x_real = x_real.transpose(1, 2)
            x_imag = x_imag.transpose(1, 2)
        
        # Check and fix channel dimensions if needed
        if x_real.size(1) != self.in_channels:
            print(f"Adjusting channel dimensions from {x_real.size(1)} to {self.in_channels}")
            # Create adapted tensors
            adapted_real = torch.zeros(x_real.size(0), self.in_channels, x_real.size(2), device=x_real.device)
            adapted_imag = torch.zeros(x_imag.size(0), self.in_channels, x_imag.size(2), device=x_imag.device)
            
            # Copy as many channels as we can
            copy_channels = min(x_real.size(1), self.in_channels)
            adapted_real[:, :copy_channels] = x_real[:, :copy_channels]
            adapted_imag[:, :copy_channels] = x_imag[:, :copy_channels]
            
            x_real = adapted_real
            x_imag = adapted_imag
        
        # Regular convolution operations
        real = self.conv_re(x_real) - self.conv_im(x_imag)
        imag = self.conv_re(x_imag) + self.conv_im(x_real)       

        return (real, imag)


class ComplexConvTranspose1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1):
        """
        Complex-valued 1D transposed convolution layer
        
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            kernel_size (int): Size of the convolving kernel
            stride (int): Stride of the convolution
            padding (int): Zero-padding added to both sides of the input
            output_padding (int): Additional size added to one side of the output
            groups (int): Number of blocked connections from input to output channels
            bias (bool): If True, adds a learnable bias to the output
            dilation (int): Spacing between kernel elements
        """
        super(ComplexConvTranspose1d, self).__init__()
        
        self.deconv_re = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding, output_padding, groups, bias, dilation)
        self.deconv_im = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding, output_padding, groups, bias, dilation)
        
        # Save configuration for reference
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation

    def forward(self, x):
        """
        Forward pass with consistent dimension handling
        
        Args:
            x: Either a ComplexTensor, tuple of (real, imag) tensors, or real tensor
               Expected shape: [B, C, T] for each tensor component
               
        Returns:
            tuple: (real, imag) output tensors with shape [B, C_out, T_out]
        """
        if isinstance(x, ComplexTensor):
            x_real, x_imag = x.real, x.imag
        elif isinstance(x, tuple):
            x_real, x_imag = x
        else:  # Handle the case where input is real
            x_real, x_imag = x, torch.zeros_like(x)
        
        # CRITICAL DIMENSION HANDLING - Ensure [B, C, T] format
        # Check if channels and time are swapped and fix if needed
        if x_real.dim() == 3 and x_real.size(2) == self.in_channels and x_real.size(1) != self.in_channels:
            print(f"WARNING: Input has swapped dimensions [B, T, C]: {x_real.shape}, transposing...")
            x_real = x_real.transpose(1, 2)
            x_imag = x_imag.transpose(1, 2)
            
        # Check and fix channel dimensions if needed
        if x_real.size(1) != self.in_channels:
            print(f"Adjusting channel dimensions from {x_real.size(1)} to {self.in_channels}")
            # Create adapted tensors
            adapted_real = torch.zeros(x_real.size(0), self.in_channels, x_real.size(2), device=x_real.device)
            adapted_imag = torch.zeros(x_imag.size(0), self.in_channels, x_imag.size(2), device=x_imag.device)
            
            # Copy as many channels as we can
            copy_channels = min(x_real.size(1), self.in_channels)
            adapted_real[:, :copy_channels] = x_real[:, :copy_channels]
            adapted_imag[:, :copy_channels] = x_imag[:, :copy_channels]
            
            x_real = adapted_real
            x_imag = adapted_imag
        
        # Real component: (W_re * x_re - W_im * x_im)
        real = self.deconv_re(x_real) - self.deconv_im(x_imag)
        
        # Imaginary component: (W_re * x_im + W_im * x_re)
        imag = self.deconv_re(x_imag) + self.deconv_im(x_real)
        
        return (real, imag)


class ComplexBatchNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        """
        Complex-valued 1D batch normalization
        
        Args:
            num_features (int): Number of features
            eps (float): Small constant for numerical stability
            momentum (float): Momentum for running statistics
            affine (bool): If True, has learnable affine parameters
            track_running_stats (bool): If True, tracks running mean and variance
        """
        super(ComplexBatchNorm1d, self).__init__()
        
        self.bn_re = nn.BatchNorm1d(num_features, eps, momentum, affine, track_running_stats)
        self.bn_im = nn.BatchNorm1d(num_features, eps, momentum, affine, track_running_stats)
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x (tuple): Tuple of (real, imag) tensors
            
        Returns:
            tuple: Tuple of (real, imag) tensors
        """
        if isinstance(x, ComplexTensor):
            x_real, x_imag = x.real, x.imag
        elif isinstance(x, tuple):
            x_real, x_imag = x
        else:
            x_real, x_imag = x, torch.zeros_like(x)
        
        # Apply batch normalization separately to real and imaginary parts
        real = self.bn_re(x_real)
        imag = self.bn_im(x_imag)
        
        return (real, imag)


class ComplexLeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.01):
        """
        Complex-valued LeakyReLU activation
        
        Args:
            negative_slope (float): Controls the angle of the negative slope
        """
        super(ComplexLeakyReLU, self).__init__()
        self.leaky_relu = nn.LeakyReLU(negative_slope=negative_slope)
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x (tuple): Tuple of (real, imag) tensors
            
        Returns:
            tuple: Tuple of (real, imag) tensors
        """
        if isinstance(x, ComplexTensor):
            x_real, x_imag = x.real, x.imag
        elif isinstance(x, tuple):
            x_real, x_imag = x
        else:
            x_real, x_imag = x, torch.zeros_like(x)
        
        # Apply LeakyReLU separately to real and imaginary parts
        real = self.leaky_relu(x_real)
        imag = self.leaky_relu(x_imag)
        
        return (real, imag)


class ComplexTanh(nn.Module):
    def __init__(self):
        """Complex-valued tanh activation"""
        super(ComplexTanh, self).__init__()
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x (tuple): Tuple of (real, imag) tensors
            
        Returns:
            tuple: Tuple of (real, imag) tensors
        """
        if isinstance(x, ComplexTensor):
            x_real, x_imag = x.real, x.imag
        elif isinstance(x, tuple):
            x_real, x_imag = x
        else:
            x_real, x_imag = x, torch.zeros_like(x)
        
        # Complex tanh implementation
        # Mathematically more correct approach:
        # tanh(z) = (e^z - e^-z) / (e^z + e^-z) for complex z
        
        # Simple approximation - applying tanh separately to real and imaginary parts
        real = torch.tanh(x_real)
        imag = torch.tanh(x_imag)
        
        return (real, imag)


class ComplexDropout(nn.Module):
    def __init__(self, p=0.5):
        """
        Complex-valued dropout
        
        Args:
            p (float): Dropout probability
        """
        super(ComplexDropout, self).__init__()
        self.dropout = nn.Dropout(p)
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x (tuple): Tuple of (real, imag) tensors
            
        Returns:
            tuple: Tuple of (real, imag) tensors
        """
        if isinstance(x, ComplexTensor):
            x_real, x_imag = x.real, x.imag
        elif isinstance(x, tuple):
            x_real, x_imag = x
        else:
            x_real, x_imag = x, torch.zeros_like(x)
        
        # Use the same dropout mask for both real and imaginary parts
        # to preserve complex number properties
        if self.training:
            mask = torch.ones_like(x_real).bernoulli_(1 - self.dropout.p) / (1 - self.dropout.p)
            return (x_real * mask, x_imag * mask)
        else:
            return (x_real, x_imag)


class ComplexToReal(nn.Module):
    def __init__(self, mode='mag_phase'):  # Default to mag_phase for better reconstruction
        """
        Convert complex-valued tensor to real-valued tensor with improved phase handling
        
        Args:
            mode (str): Conversion mode, one of ['magnitude', 'real', 'imag', 'phase', 'mag_phase']
        """
        super(ComplexToReal, self).__init__()
        assert mode in ['magnitude', 'real', 'imag', 'phase', 'mag_phase'], \
            "Mode must be one of ['magnitude', 'real', 'imag', 'phase', 'mag_phase']"
        self.mode = mode
        
    def forward(self, x):
        """
        Forward pass with improved phase preservation
        
        Args:
            x (tuple): Tuple of (real, imag) tensors
            
        Returns:
            Tensor: Real-valued tensor
        """
        if isinstance(x, ComplexTensor):
            x_real, x_imag = x.real, x.imag
        elif isinstance(x, tuple):
            x_real, x_imag = x
        else:
            return x  # Already real
        
        if self.mode == 'magnitude':
            out = torch.sqrt(x_real**2 + x_imag**2)
        elif self.mode == 'real':
            out = x_real
        elif self.mode == 'imag':
            out = x_imag
        elif self.mode == 'phase':
            out = torch.atan2(x_imag, x_real)
        elif self.mode == 'mag_phase':
            # Enhanced magnitude and phase combination for better reconstruction
            # This preserves both amplitude and phase information
            mag = torch.sqrt(x_real**2 + x_imag**2)
            phase = torch.atan2(x_imag, x_real)
            
            # Apply a smooth non-linearity to magnitude for better dynamic range
            mag = torch.tanh(mag) * (1 + mag)
            
            # Reconstruct signal using magnitude and phase
            # This maintains temporal coherence better than simple magnitude
            out = mag * torch.cos(phase)
        
        # Ensure non-zero amplitudes to prevent artifacts
        zero_mask = (torch.abs(out) < 1e-7)
        if zero_mask.any():
            # Add small amount of noise to prevent dead regions
            noise = torch.randn_like(out) * 1e-6
            out = torch.where(zero_mask, noise, out)
        
        # Check for NaN or inf values
        has_nan = torch.isnan(out).any().item()
        has_inf = torch.isinf(out).any().item()
        if has_nan or has_inf:
            # Replace NaN/Inf with zeros with small noise
            out = torch.nan_to_num(out, nan=1e-6, posinf=1.0, neginf=-1.0)
        
        return out