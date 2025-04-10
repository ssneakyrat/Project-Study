#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.complex_layers import ComplexConv1d, ComplexBatchNorm1d, ComplexLeakyReLU, ComplexDropout


class ComplexEncoder(nn.Module):
    def __init__(self, input_channels, channels, kernel_sizes, strides, paddings=None, dropout=0.1, use_batch_norm=True):
        """
        Complex-valued encoder network
        
        Args:
            input_channels (int): Number of input channels
            channels (list): List of channel dimensions for each layer
            kernel_sizes (list): List of kernel sizes for each layer
            strides (list): List of strides for each layer
            paddings (list, optional): List of padding values for each layer
            dropout (float): Dropout probability
            use_batch_norm (bool): Whether to use batch normalization
        """
        super(ComplexEncoder, self).__init__()
        
        # Ensure all lists have the same length
        assert len(channels) == len(kernel_sizes) == len(strides), "channels, kernel_sizes, and strides must have the same length"
        
        # Default paddings if not provided
        if paddings is None:
            paddings = [k // 2 for k in kernel_sizes]
        else:
            assert len(paddings) == len(channels), "paddings must have the same length as channels"
        
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
                padding=paddings[i]
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
        
        # Store configuration for shape tracking
        self.input_channels = input_channels
        self.output_channels = channels[-1]
        self.channels = channels
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.paddings = paddings
        
        # Flag for gradient checkpointing
        self.use_gradient_checkpointing = False
    
    def forward(self, x):
        """
        Forward pass through encoder with optional gradient checkpointing
        
        Args:
            x (tuple): Tuple of (real, imaginary) tensors
            
        Returns:
            tuple: Tuple of (encoded features, intermediate outputs)
                  encoded features: (real, imaginary) tensors
                  intermediates: list of (real, imaginary) tensors
        """
        intermediates = []
        
        # Pass through layers and store intermediates for skip connections
        for i, layer in enumerate(self.layers):
            # Apply gradient checkpointing if enabled
            if self.use_gradient_checkpointing and i > 0 and i < len(self.layers) - 1:
                # Only checkpoint middle layers for efficiency
                def checkpoint_function(module, module_input):
                    return module(module_input)
                
                x = torch.utils.checkpoint.checkpoint(
                    checkpoint_function, 
                    layer, 
                    x,
                    use_reentrant=False
                )
            else:
                x = layer(x)
                
            intermediates.append(x)
        
        return x, intermediates
    
    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing to reduce memory usage"""
        self.use_gradient_checkpointing = True

class ComplexResidualBlock(nn.Module):
    """
    Complex residual block for deeper encoders
    """
    def __init__(self, channels, kernel_size=3, stride=1, padding=None, dropout=0.1, use_batch_norm=True):
        super(ComplexResidualBlock, self).__init__()
        
        # Default padding if not provided
        if padding is None:
            padding = kernel_size // 2
        
        # First complex convolution
        self.conv1 = ComplexConv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            stride=1,  # No striding in the residual block
            padding=padding
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
            padding=padding
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
                padding=padding
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