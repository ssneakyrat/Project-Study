#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.complex_layers import ComplexConv1d, ComplexBatchNorm1d, ComplexLeakyReLU, ComplexDropout


class ComplexEncoder(nn.Module):
    def __init__(self, input_channels, channels, kernel_sizes, strides, dropout=0.1, use_batch_norm=True):
        """
        Complex-valued encoder network
        
        Args:
            input_channels (int): Number of input channels
            channels (list): List of channel dimensions for each layer
            kernel_sizes (list): List of kernel sizes for each layer
            strides (list): List of strides for each layer
            dropout (float): Dropout probability
            use_batch_norm (bool): Whether to use batch normalization
        """
        super(ComplexEncoder, self).__init__()
        
        # Ensure all lists have the same length
        assert len(channels) == len(kernel_sizes) == len(strides), "channels, kernel_sizes, and strides must have the same length"
        
        self.num_layers = len(channels)
        self.layers = nn.ModuleList()
        
        # Input dimension
        in_channels = input_channels
        
        # Create encoder layers
        for i in range(self.num_layers):
            layer = []
            
            # Complex convolution
            layer.append(ComplexConv1d(
                in_channels=in_channels,
                out_channels=channels[i],
                kernel_size=kernel_sizes[i],
                stride=strides[i],
                padding=kernel_sizes[i] // 2
            ))
            
            # Batch normalization
            if use_batch_norm:
                layer.append(ComplexBatchNorm1d(channels[i]))
            
            # Activation
            layer.append(ComplexLeakyReLU(0.2))
            
            # Dropout
            if dropout > 0:
                layer.append(ComplexDropout(dropout))
            
            # Update input channels for next layer
            in_channels = channels[i]
            
            # Add layer to module list
            self.layers.append(nn.Sequential(*layer))
    
    def forward(self, x):
        """
        Forward pass through encoder
        
        Args:
            x (tuple): Tuple of (real, imaginary) tensors
            
        Returns:
            tuple: Tuple of (encoded features, intermediate outputs)
                  encoded features: (real, imaginary) tensors
                  intermediates: list of (real, imaginary) tensors
        """
        intermediates = []
        
        # Pass through layers and store intermediates for skip connections
        for layer in self.layers:
            x = layer(x)
            intermediates.append(x)
        
        return x, intermediates


class DualPathEncoder(nn.Module):
    """
    Encoder with separate paths for magnitude and phase processing
    """
    def __init__(self, input_channels, channels, kernel_sizes, strides, dropout=0.1, use_batch_norm=True):
        super(DualPathEncoder, self).__init__()
        
        # Magnitude path (real-valued)
        self.mag_encoder = nn.ModuleList()
        
        # Phase path (real-valued but processes phase information)
        self.phase_encoder = nn.ModuleList()
        
        # Input dimension
        in_channels = input_channels
        
        # Create encoder layers
        for i in range(len(channels)):
            # Magnitude path
            mag_layer = []
            mag_layer.append(nn.Conv1d(
                in_channels=in_channels,
                out_channels=channels[i],
                kernel_size=kernel_sizes[i],
                stride=strides[i],
                padding=kernel_sizes[i] // 2
            ))
            
            if use_batch_norm:
                mag_layer.append(nn.BatchNorm1d(channels[i]))
            
            mag_layer.append(nn.LeakyReLU(0.2))
            
            if dropout > 0:
                mag_layer.append(nn.Dropout(dropout))
            
            self.mag_encoder.append(nn.Sequential(*mag_layer))
            
            # Phase path
            phase_layer = []
            phase_layer.append(nn.Conv1d(
                in_channels=in_channels,
                out_channels=channels[i],
                kernel_size=kernel_sizes[i],
                stride=strides[i],
                padding=kernel_sizes[i] // 2
            ))
            
            if use_batch_norm:
                phase_layer.append(nn.BatchNorm1d(channels[i]))
            
            phase_layer.append(nn.LeakyReLU(0.2))
            
            if dropout > 0:
                phase_layer.append(nn.Dropout(dropout))
            
            self.phase_encoder.append(nn.Sequential(*phase_layer))
            
            # Update input channels for next layer
            in_channels = channels[i]
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x (tuple): Tuple of (magnitude, phase) tensors
            
        Returns:
            tuple: Tuple of (encoded features, intermediates)
        """
        mag, phase = x
        
        mag_intermediates = []
        phase_intermediates = []
        
        # Process magnitude path
        for layer in self.mag_encoder:
            mag = layer(mag)
            mag_intermediates.append(mag)
        
        # Process phase path
        for layer in self.phase_encoder:
            phase = layer(phase)
            phase_intermediates.append(phase)
        
        # Return encoded features and intermediates
        encoded = (mag, phase)
        intermediates = (mag_intermediates, phase_intermediates)
        
        return encoded, intermediates


class ComplexResidualBlock(nn.Module):
    """
    Complex residual block for deeper encoders
    """
    def __init__(self, channels, kernel_size=3, stride=1, dropout=0.1, use_batch_norm=True):
        super(ComplexResidualBlock, self).__init__()
        
        # First complex convolution
        self.conv1 = ComplexConv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            stride=1,  # No striding in the residual block
            padding=kernel_size // 2
        )
        
        # Batch norm and activation
        self.bn1 = ComplexBatchNorm1d(channels) if use_batch_norm else nn.Identity()
        self.act1 = ComplexLeakyReLU(0.2)
        
        # Second complex convolution
        self.conv2 = ComplexConv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            stride=1,  # No striding in the residual block
            padding=kernel_size // 2
        )
        
        # Batch norm
        self.bn2 = ComplexBatchNorm1d(channels) if use_batch_norm else nn.Identity()
        
        # Final activation
        self.act2 = ComplexLeakyReLU(0.2)
        
        # Dropout
        self.dropout = ComplexDropout(dropout) if dropout > 0 else nn.Identity()
        
        # Strided convolution if stride > 1
        if stride > 1:
            self.downsample = ComplexConv1d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2
            )
        else:
            self.downsample = None
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x (tuple): Tuple of (real, imaginary) tensors
            
        Returns:
            tuple: Tuple of (real, imaginary) tensors
        """
        # Store input for residual connection
        identity = x
        
        # First conv block
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)
        
        # Second conv block
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout(out)
        
        # Apply downsampling to both out and identity if needed
        if self.downsample is not None:
            out = self.downsample(out)
            identity = self.downsample(identity)
        
        # Add residual connection
        x_real, x_imag = out
        identity_real, identity_imag = identity
        
        out = (x_real + identity_real, x_imag + identity_imag)
        
        # Final activation
        out = self.act2(out)
        
        return out