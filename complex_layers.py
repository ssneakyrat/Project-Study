import torch
import torch.nn as nn
import torch.nn.functional as F

class ComplexConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv_real = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv_imag = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        
    def forward(self, x):
        """
        Args:
            x: Complex tensor of shape (batch_size, channels, time)
        
        Returns:
            Complex tensor after convolution
        """
        real, imag = x.real, x.imag
        
        real_real = self.conv_real(real)
        real_imag = self.conv_imag(real)
        imag_real = self.conv_real(imag)
        imag_imag = self.conv_imag(imag)
        
        real_out = real_real - imag_imag
        imag_out = real_imag + imag_real
        
        return torch.complex(real_out, imag_out)


class ComplexConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        # Handle different kernel sizes for different dimensions
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)
            
        self.conv_real = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv_imag = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        
    def forward(self, x):
        """
        Args:
            x: Complex tensor of shape (batch_size, channels, height, width)
        
        Returns:
            Complex tensor after convolution
        """
        real, imag = x.real, x.imag
        
        real_real = self.conv_real(real)
        real_imag = self.conv_imag(real)
        imag_real = self.conv_real(imag)
        imag_imag = self.conv_imag(imag)
        
        real_out = real_real - imag_imag
        imag_out = real_imag + imag_real
        
        return torch.complex(real_out, imag_out)


class ComplexConvTranspose1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0):
        super().__init__()
        self.conv_transpose_real = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.conv_transpose_imag = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, stride, padding, output_padding)
        
    def forward(self, x):
        """
        Args:
            x: Complex tensor of shape (batch_size, channels, time)
        
        Returns:
            Complex tensor after transposed convolution
        """
        real, imag = x.real, x.imag
        
        real_real = self.conv_transpose_real(real)
        real_imag = self.conv_transpose_imag(real)
        imag_real = self.conv_transpose_real(imag)
        imag_imag = self.conv_transpose_imag(imag)
        
        real_out = real_real - imag_imag
        imag_out = real_imag + imag_real
        
        return torch.complex(real_out, imag_out)


class ComplexConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0):
        super().__init__()
        # Handle different kernel sizes for different dimensions
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)
        if isinstance(output_padding, int):
            output_padding = (output_padding, output_padding)
            
        self.conv_transpose_real = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.conv_transpose_imag = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride, padding, output_padding)
        
    def forward(self, x):
        """
        Args:
            x: Complex tensor of shape (batch_size, channels, height, width)
        
        Returns:
            Complex tensor after transposed convolution
        """
        real, imag = x.real, x.imag
        
        real_real = self.conv_transpose_real(real)
        real_imag = self.conv_transpose_imag(real)
        imag_real = self.conv_transpose_real(imag)
        imag_imag = self.conv_transpose_imag(imag)
        
        real_out = real_real - imag_imag
        imag_out = real_imag + imag_real
        
        return torch.complex(real_out, imag_out)


class ComplexBatchNorm1d(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.bn_real = nn.BatchNorm1d(num_features)
        self.bn_imag = nn.BatchNorm1d(num_features)
        
    def forward(self, x):
        """
        Args:
            x: Complex tensor of shape (batch_size, channels, time)
        
        Returns:
            Complex tensor after batch normalization
        """
        real, imag = x.real, x.imag
        
        real_out = self.bn_real(real)
        imag_out = self.bn_imag(imag)
        
        return torch.complex(real_out, imag_out)


class ComplexBatchNorm2d(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.bn_real = nn.BatchNorm2d(num_features)
        self.bn_imag = nn.BatchNorm2d(num_features)
        
    def forward(self, x):
        """
        Args:
            x: Complex tensor of shape (batch_size, channels, height, width)
        
        Returns:
            Complex tensor after batch normalization
        """
        real, imag = x.real, x.imag
        
        real_out = self.bn_real(real)
        imag_out = self.bn_imag(imag)
        
        return torch.complex(real_out, imag_out)


class ComplexPReLU(nn.Module):
    def __init__(self, num_parameters=1, init=0.25):
        super().__init__()
        self.prelu_real = nn.PReLU(num_parameters, init)
        self.prelu_imag = nn.PReLU(num_parameters, init)
        
    def forward(self, x):
        """
        Args:
            x: Complex tensor
        
        Returns:
            Complex tensor after PReLU activation
        """
        real, imag = x.real, x.imag
        
        real_out = self.prelu_real(real)
        imag_out = self.prelu_imag(imag)
        
        return torch.complex(real_out, imag_out)


class ComplexToReal(nn.Module):
    def __init__(self, mode='magnitude'):
        """Convert complex tensor to real tensor.
        
        Args:
            mode: How to convert complex to real, options:
                - 'magnitude': sqrt(real^2 + imag^2)
                - 'real': real part only
                - 'imag': imaginary part only
        """
        super().__init__()
        self.mode = mode
        
    def forward(self, x):
        """
        Args:
            x: Complex tensor
        
        Returns:
            Real tensor
        """
        if self.mode == 'magnitude':
            return torch.abs(x)
        elif self.mode == 'real':
            return x.real
        elif self.mode == 'imag':
            return x.imag
        else:
            raise ValueError(f"Unknown mode: {self.mode}")


class RealToComplex(nn.Module):
    def __init__(self, mode='zero_imag'):
        """Convert real tensor to complex tensor.
        
        Args:
            mode: How to convert real to complex, options:
                - 'zero_imag': set imaginary part to zero
                - 'equal': set imaginary part equal to real part
                - 'random': set imaginary part to random values
        """
        super().__init__()
        self.mode = mode
        
    def forward(self, x):
        """
        Args:
            x: Real tensor
        
        Returns:
            Complex tensor
        """
        if self.mode == 'zero_imag':
            return torch.complex(x, torch.zeros_like(x))
        elif self.mode == 'equal':
            return torch.complex(x, x.clone())
        elif self.mode == 'random':
            return torch.complex(x, torch.randn_like(x) * 0.01)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")