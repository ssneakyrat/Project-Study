#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.complex_layers import ComplexConvTranspose1d, ComplexBatchNorm1d, ComplexLeakyReLU, ComplexTanh, ComplexDropout


class ComplexDecoder(nn.Module):
    def __init__(self, latent_channels, channels, kernel_sizes, strides, paddings=None, output_paddings=None, 
                output_channels=1, dropout=0.1, use_batch_norm=True):
        """
        Complex-valued decoder network with skip connections
        
        Args:
            latent_channels (int): Number of input channels from latent space
            channels (list): List of channel dimensions for each layer (in reverse order)
            kernel_sizes (list): List of kernel sizes for each layer (in reverse order)
            strides (list): List of strides for each layer (in reverse order)
            paddings (list, optional): List of padding values for each layer
            output_paddings (list, optional): List of output padding values for each layer
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
        
        # Default paddings and output_paddings if not provided
        if paddings is None:
            paddings = [k // 2 for k in kernel_sizes]
        else:
            paddings = paddings[::-1]  # Reverse to match other parameters
            assert len(paddings) == len(channels), "paddings must have the same length as channels"
        
        if output_paddings is None:
            output_paddings = [s - 1 for s in strides]  # Default: stride - 1
        else:
            output_paddings = output_paddings[::-1]  # Reverse to match other parameters
            assert len(output_paddings) == len(channels), "output_paddings must have the same length as channels"
        
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
                padding=paddings[i],
                output_padding=output_paddings[i]
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
        
        # Store configuration for shape tracking
        self.latent_channels = latent_channels
        self.output_channels = output_channels
        self.channels = channels
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.paddings = paddings
        self.output_paddings = output_paddings
    
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
            # Debug shape tracking (uncomment for debugging)
            # if isinstance(x, tuple):
            #     print(f"Decoder layer {i} input shape: {x[0].shape}")
            # else:
            #     print(f"Decoder layer {i} input shape: {x.shape}")
            
            # Apply layer
            x = layer(x)
            
            # Debug shape tracking (uncomment for debugging)
            # if isinstance(x, tuple):
            #     print(f"Decoder layer {i} output shape: {x[0].shape}")
            # else:
            #     print(f"Decoder layer {i} output shape: {x.shape}")
            
            # Add skip connection if available
            if skip_connections is not None and i < len(skip_connections):
                skip = skip_connections[i]
                
                # Get real and imaginary parts
                if isinstance(x, tuple) and isinstance(skip, tuple):
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
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, output_padding=1, dropout=0.1, use_batch_norm=True):
        super(ComplexUpsampleBlock, self).__init__()
        
        # Complex transposed convolution for upsampling
        self.upsample = ComplexConvTranspose1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding
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