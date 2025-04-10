#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.complex_layers import ComplexConvTranspose1d, ComplexBatchNorm1d, ComplexLeakyReLU, ComplexTanh, ComplexDropout


class ComplexDecoder(nn.Module):
    def __init__(self, latent_channels, channels, kernel_sizes, strides, output_channels=1, dropout=0.1, use_batch_norm=True):
        """
        Complex-valued decoder network with skip connections
        
        Args:
            latent_channels (int): Number of input channels from latent space
            channels (list): List of channel dimensions for each layer (in reverse order)
            kernel_sizes (list): List of kernel sizes for each layer (in reverse order)
            strides (list): List of strides for each layer (in reverse order)
            output_channels (int): Number of output channels
            dropout (float): Dropout probability
            use_batch_norm (bool): Whether to use batch normalization
        """
        super(ComplexDecoder, self).__init__()
        
        # Ensure all lists have the same length
        assert len(channels) == len(kernel_sizes) == len(strides), "channels, kernel_sizes, and strides must have the same length"
        
        # Reverse the lists for decoder (will be using in reverse order of encoder)
        channels = channels[::-1]
        kernel_sizes = kernel_sizes[::-1]
        strides = strides[::-1]
        
        self.num_layers = len(channels)
        self.layers = nn.ModuleList()
        
        # Input dimension
        in_channels = latent_channels
        
        # Create decoder layers
        for i in range(self.num_layers):
            layer = []
            
            # For the last layer, output to the number of output channels
            out_channels = channels[i] if i < self.num_layers - 1 else output_channels
            
            # Complex transposed convolution
            layer.append(ComplexConvTranspose1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_sizes[i],
                stride=strides[i],
                padding=kernel_sizes[i] // 2,
                output_padding=strides[i] - 1  # Needed to match output size
            ))
            
            # Batch normalization (except for the last layer)
            if use_batch_norm and i < self.num_layers - 1:
                layer.append(ComplexBatchNorm1d(out_channels))
            
            # Activation (tanh for the last layer, LeakyReLU for others)
            if i == self.num_layers - 1:
                layer.append(ComplexTanh())
            else:
                layer.append(ComplexLeakyReLU(0.2))
            
            # Dropout (except for the last layer)
            if dropout > 0 and i < self.num_layers - 1:
                layer.append(ComplexDropout(dropout))
            
            # Update input channels for next layer
            in_channels = out_channels
            
            # Add layer to module list
            self.layers.append(nn.Sequential(*layer))
    
    def forward(self, x, skip_connections=None):
        """
        Forward pass through decoder with skip connections
        
        Args:
            x (tuple): Tuple of (real, imaginary) tensors from encoder
            skip_connections (list): List of (real, imaginary) tensors from encoder
            
        Returns:
            tuple: Tuple of (real, imaginary) tensors for output
        """
        # Reverse skip connections for proper alignment
        if skip_connections is not None:
            skip_connections = skip_connections[::-1]
        
        # Pass through layers with skip connections
        for i, layer in enumerate(self.layers):
            # Apply layer
            x = layer(x)
            
            # Add skip connection if available
            if skip_connections is not None and i < len(skip_connections) - 1:  # Skip last connection to output
                skip = skip_connections[i]
                
                # Get real and imaginary parts
                x_real, x_imag = x
                skip_real, skip_imag = skip
                
                # Ensure skip connection channels match current layer
                if x_real.shape[1] != skip_real.shape[1]:
                    # Number of channels doesn't match - can't use this skip connection
                    continue
                    
                # Resize skip connection if time dimensions don't match
                if x_real.shape[2] != skip_real.shape[2]:
                    try:
                        # Try to interpolate to match dimensions
                        skip_real = F.interpolate(skip_real, size=x_real.shape[2], mode='linear', align_corners=False)
                        skip_imag = F.interpolate(skip_imag, size=x_imag.shape[2], mode='linear', align_corners=False)
                    except RuntimeError:
                        # If interpolation fails (e.g., for extreme size differences), skip this connection
                        continue
                
                # Apply skip connection with dimensionality check
                if x_real.shape == skip_real.shape and x_imag.shape == skip_imag.shape:
                    alpha = 0.5  # Weighting factor for skip connection
                    x = (x_real + alpha * skip_real, x_imag + alpha * skip_imag)
        
        return x


class ComplexUpsampleBlock(nn.Module):
    """
    Upsample block for complex-valued decoder
    """
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, dropout=0.1, use_batch_norm=True):
        super(ComplexUpsampleBlock, self).__init__()
        
        # Complex transposed convolution for upsampling
        self.upsample = ComplexConvTranspose1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2 - 1,  # Adjusted for proper output size
            output_padding=stride - 1
        )
        
        # Batch normalization
        self.bn = ComplexBatchNorm1d(out_channels) if use_batch_norm else nn.Identity()
        
        # Activation
        self.act = ComplexLeakyReLU(0.2)
        
        # Dropout
        self.dropout = ComplexDropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x, skip=None):
        """
        Forward pass with optional skip connection
        
        Args:
            x (tuple): Tuple of (real, imaginary) tensors
            skip (tuple, optional): Skip connection from encoder
            
        Returns:
            tuple: Tuple of (real, imaginary) tensors
        """
        # Upsample
        x = self.upsample(x)
        
        # Apply batch norm and activation
        x = self.bn(x)
        x = self.act(x)
        
        # Apply dropout
        x = self.dropout(x)
        
        # Add skip connection if provided
        if skip is not None:
            # Get real and imaginary parts
            x_real, x_imag = x
            skip_real, skip_imag = skip
            
            # Resize skip connection if needed
            if x_real.shape[2] != skip_real.shape[2]:
                skip_real = F.interpolate(skip_real, size=x_real.shape[2], mode='linear', align_corners=False)
                skip_imag = F.interpolate(skip_imag, size=x_imag.shape[2], mode='linear', align_corners=False)
            
            # Add skip connection
            x = (x_real + skip_real, x_imag + skip_imag)
        
        return x


class DualPathDecoder(nn.Module):
    """
    Decoder with separate paths for magnitude and phase processing
    to be used with DualPathEncoder
    """
    def __init__(self, latent_channels, channels, kernel_sizes, strides, output_channels=1, dropout=0.1, use_batch_norm=True):
        super(DualPathDecoder, self).__init__()
        
        # Reverse the lists for decoder
        channels = channels[::-1]
        kernel_sizes = kernel_sizes[::-1]
        strides = strides[::-1]
        
        # Magnitude path
        self.mag_decoder = nn.ModuleList()
        
        # Phase path
        self.phase_decoder = nn.ModuleList()
        
        # Input dimension
        in_channels = latent_channels
        
        # Create decoder layers
        for i in range(len(channels)):
            # Set output channels
            out_channels = channels[i] if i < len(channels) - 1 else output_channels
            
            # Magnitude path
            mag_layer = []
            mag_layer.append(nn.ConvTranspose1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_sizes[i],
                stride=strides[i],
                padding=kernel_sizes[i] // 2,
                output_padding=strides[i] - 1
            ))
            
            if use_batch_norm and i < len(channels) - 1:
                mag_layer.append(nn.BatchNorm1d(out_channels))
            
            if i == len(channels) - 1:
                mag_layer.append(nn.Tanh())  # For output
            else:
                mag_layer.append(nn.LeakyReLU(0.2))
            
            if dropout > 0 and i < len(channels) - 1:
                mag_layer.append(nn.Dropout(dropout))
            
            self.mag_decoder.append(nn.Sequential(*mag_layer))
            
            # Phase path
            phase_layer = []
            phase_layer.append(nn.ConvTranspose1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_sizes[i],
                stride=strides[i],
                padding=kernel_sizes[i] // 2,
                output_padding=strides[i] - 1
            ))
            
            if use_batch_norm and i < len(channels) - 1:
                phase_layer.append(nn.BatchNorm1d(out_channels))
            
            if i == len(channels) - 1:
                # Phase should be between -π and π
                phase_layer.append(nn.Tanh())  # Scale to [-1, 1], will multiply by π later
            else:
                phase_layer.append(nn.LeakyReLU(0.2))
            
            if dropout > 0 and i < len(channels) - 1:
                phase_layer.append(nn.Dropout(dropout))
            
            self.phase_decoder.append(nn.Sequential(*phase_layer))
            
            # Update input channels for next layer
            in_channels = out_channels
    
    def forward(self, x, skip_connections=None):
        """
        Forward pass
        
        Args:
            x (tuple): Tuple of (magnitude, phase) encodings
            skip_connections (tuple, optional): Tuple of (mag_skips, phase_skips) from encoder
            
        Returns:
            tuple: Tuple of (magnitude, phase) outputs
        """
        mag, phase = x
        
        # Handle skip connections
        mag_skips, phase_skips = None, None
        if skip_connections is not None:
            mag_skips, phase_skips = skip_connections
            mag_skips = mag_skips[::-1]
            phase_skips = phase_skips[::-1]
        
        # Process magnitude path
        for i, layer in enumerate(self.mag_decoder):
            mag = layer(mag)
            
            # Add skip connection if available
            if mag_skips is not None and i < len(mag_skips) - 1:
                skip = mag_skips[i]
                
                # Resize if needed
                if mag.shape[2] != skip.shape[2]:
                    skip = F.interpolate(skip, size=mag.shape[2], mode='linear', align_corners=False)
                
                # Add skip connection
                mag = mag + 0.5 * skip
        
        # Process phase path
        for i, layer in enumerate(self.phase_decoder):
            phase = layer(phase)
            
            # Add skip connection if available
            if phase_skips is not None and i < len(phase_skips) - 1:
                skip = phase_skips[i]
                
                # Resize if needed
                if phase.shape[2] != skip.shape[2]:
                    skip = F.interpolate(skip, size=phase.shape[2], mode='linear', align_corners=False)
                
                # Add skip connection with circular mean for phase
                phase = phase + 0.5 * skip
        
        # For the phase output, scale to [-π, π]
        phase = phase * torch.pi
        
        return mag, phase