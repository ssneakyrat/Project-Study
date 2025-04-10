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
        self.normalized_shape = normalized_shape
        
    def forward(self, x):
        """
        Apply layer normalization to real and imaginary parts separately,
        properly handling 1D convolutional inputs (batch, channels, length)
        """
        if not isinstance(x, ComplexTensor):
            # For 1D convolutional inputs, normalize along the channel dimension
            # x shape: [batch, channels, length]
            mean = x.mean(dim=1, keepdim=True)
            var = x.var(dim=1, keepdim=True, unbiased=False)
            normalized = (x - mean) / torch.sqrt(var + 1e-5)
            return normalized
            
        # For complex inputs, normalize real and imaginary parts separately
        # Each part shape: [batch, channels, length]
        real_mean = x.real.mean(dim=1, keepdim=True)
        real_var = x.real.var(dim=1, keepdim=True, unbiased=False)
        real_normalized = (x.real - real_mean) / torch.sqrt(real_var + 1e-5)
        
        imag_mean = x.imag.mean(dim=1, keepdim=True)
        imag_var = x.imag.var(dim=1, keepdim=True, unbiased=False)
        imag_normalized = (x.imag - imag_mean) / torch.sqrt(imag_var + 1e-5)
        
        return ComplexTensor(real_normalized, imag_normalized)

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
        
    # Ensure we return a standard torch tensor, not a ComplexTensor
    magnitude = torch.sqrt(x.real**2 + x.imag**2)
    assert not isinstance(magnitude, ComplexTensor), "Magnitude calculation still resulted in ComplexTensor!"
    
    return magnitude