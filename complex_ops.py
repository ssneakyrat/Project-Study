import torch
import torch.nn as nn
import torch.nn.functional as F
from complextensor import ComplexTensor

class ComplexConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv_real = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv_imag = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        
    def forward(self, x):
        """
        x: ComplexTensor of shape [batch_size, channels, length]
        """
        if not isinstance(x, ComplexTensor):
            # If input is real, convert to ComplexTensor
            x = ComplexTensor(x, torch.zeros_like(x))
            
        real_part = self.conv_real(x.real) - self.conv_imag(x.imag)
        imag_part = self.conv_real(x.imag) + self.conv_imag(x.real)
        
        return ComplexTensor(real_part, imag_part)

class ComplexConvTranspose1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv_real = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv_imag = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding)
        
    def forward(self, x):
        """
        x: ComplexTensor of shape [batch_size, channels, length]
        """
        if not isinstance(x, ComplexTensor):
            # If input is real, convert to ComplexTensor
            x = ComplexTensor(x, torch.zeros_like(x))
            
        real_part = self.conv_real(x.real) - self.conv_imag(x.imag)
        imag_part = self.conv_real(x.imag) + self.conv_imag(x.real)
        
        return ComplexTensor(real_part, imag_part)

class ComplexLayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        self.norm_real = nn.LayerNorm(normalized_shape)
        self.norm_imag = nn.LayerNorm(normalized_shape)
        
    def forward(self, x):
        """Apply layer normalization to real and imaginary parts separately"""
        if not isinstance(x, ComplexTensor):
            return self.norm_real(x)
            
        real_part = self.norm_real(x.real)
        imag_part = self.norm_imag(x.imag)
        
        return ComplexTensor(real_part, imag_part)

def complex_leaky_relu(x, negative_slope=0.2):
    """
    Apply Leaky ReLU to real and imaginary parts separately
    """
    if not isinstance(x, ComplexTensor):
        # If input is real, convert to ComplexTensor
        x = ComplexTensor(x, torch.zeros_like(x))
        
    real_part = F.leaky_relu(x.real, negative_slope)
    imag_part = F.leaky_relu(x.imag, negative_slope)
    
    return ComplexTensor(real_part, imag_part)

def complex_to_real(x):
    """
    Convert complex tensor to real by taking magnitude
    """
    if not isinstance(x, ComplexTensor):
        return x
        
    return torch.sqrt(x.real**2 + x.imag**2)