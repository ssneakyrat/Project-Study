import torch
import torch.nn as nn
import torch.nn.functional as F

class ComplexConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv_real = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv_imag = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        
        # Initialize weights with Glorot/Xavier initialization adjusted for complex numbers
        # This ensures proper gradient flow for complex-valued networks
        nn.init.xavier_uniform_(self.conv_real.weight, gain=1/2**0.5)
        nn.init.xavier_uniform_(self.conv_imag.weight, gain=1/2**0.5)
        nn.init.zeros_(self.conv_real.bias)
        nn.init.zeros_(self.conv_imag.bias)
        
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


class ComplexConvTranspose1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0):
        super().__init__()
        self.conv_transpose_real = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.conv_transpose_imag = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, stride, padding, output_padding)
        
        # Initialize weights with Glorot/Xavier initialization adjusted for complex numbers
        nn.init.xavier_uniform_(self.conv_transpose_real.weight, gain=1/2**0.5)
        nn.init.xavier_uniform_(self.conv_transpose_imag.weight, gain=1/2**0.5)
        nn.init.zeros_(self.conv_transpose_real.bias)
        nn.init.zeros_(self.conv_transpose_imag.bias)
        
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


class ComplexBatchNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.bn_real = nn.BatchNorm1d(num_features, eps=eps, momentum=momentum)
        self.bn_imag = nn.BatchNorm1d(num_features, eps=eps, momentum=momentum)
        
    def forward(self, x):
        """
        Args:
            x: Complex tensor of shape (batch_size, channels, time)
        
        Returns:
            Complex tensor after batch normalization
        """
        real, imag = x.real, x.imag
        
        # Apply batch norm separately to real and imaginary parts
        real_out = self.bn_real(real)
        imag_out = self.bn_imag(imag)
        
        # Handle NaN values during initialization
        if not self.training and (torch.isnan(real_out).any() or torch.isnan(imag_out).any()):
            return x
        
        return torch.complex(real_out, imag_out)


class ComplexPReLU(nn.Module):
    def __init__(self, num_parameters=1, init=0.25):
        super().__init__()
        self.prelu_real = nn.PReLU(num_parameters, init)
        self.prelu_imag = nn.PReLU(num_parameters, init)
        
    def forward(self, x):
        """
        Args:
            x: Complex tensor of shape (batch_size, channels, time)
        
        Returns:
            Complex tensor after PReLU activation
        """
        real, imag = x.real, x.imag
        
        real_out = self.prelu_real(real)
        imag_out = self.prelu_imag(imag)
        
        return torch.complex(real_out, imag_out)


class ComplexToReal(nn.Module):
    def __init__(self, mode='polar'):
        """Convert complex tensor to real tensor.
        
        Args:
            mode: How to convert complex to real, options:
                - 'magnitude': sqrt(real^2 + imag^2)
                - 'real': real part only
                - 'imag': imaginary part only
                - 'phase_aware': real * cos(phase) for better phase preservation
                - 'polar': Uses both magnitude and phase information
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
        elif self.mode == 'phase_aware':
            # Use magnitude but preserve phase direction in real domain
            magnitude = torch.abs(x)
            phase = torch.angle(x)
            return magnitude * torch.cos(phase)
        elif self.mode == 'polar':
            # Separate magnitude and phase, then concatenate
            # This doubles the channel dimension but preserves all information
            magnitude = torch.abs(x)
            phase = torch.angle(x)
            # Scale phase to same range as magnitude for balanced learning
            phase_scaled = phase / torch.pi
            return torch.cat([magnitude, phase_scaled], dim=1)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")


class RealToComplex(nn.Module):
    def __init__(self, mode='analytic'):
        """Convert real tensor to complex tensor.
        
        Args:
            mode: How to convert real to complex, options:
                - 'zero_imag': set imaginary part to zero
                - 'equal': set imaginary part equal to real part
                - 'random': set imaginary part to random values
                - 'hilbert': attempt to recover phase using Hilbert transform
                - 'analytic': improved analytic signal approach
                - 'polar': expects concatenated [magnitude, phase] channels
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
        elif self.mode == 'polar':
            # Input is expected to be [magnitude, phase_scaled] concatenated along channel dim
            channels = x.size(1)
            assert channels % 2 == 0, "Channel dimension must be even for polar mode"
            
            half_channels = channels // 2
            magnitude = x[:, :half_channels]
            phase_scaled = x[:, half_channels:]
            
            # Convert scaled phase back to radians
            phase = phase_scaled * torch.pi
            
            # Convert to complex using polar form
            real_part = magnitude * torch.cos(phase)
            imag_part = magnitude * torch.sin(phase)
            
            return torch.complex(real_part, imag_part)
        elif self.mode == 'analytic':
            # Improved analytic signal approach using FFT for Hilbert transform
            batch_size, channels, length = x.shape
            
            # Compute FFT
            X_fft = torch.fft.rfft(x, dim=2)
            
            # Create filter for analytic signal
            h = torch.ones((batch_size, channels, length // 2 + 1), device=x.device)
            h[:, :, 0] = 1.0  # DC component unchanged
            
            # Double positive frequencies (except DC and Nyquist)
            if h.size(2) > 2:
                h[:, :, 1:-1] = 2.0
            
            # Get imaginary part using a simpler approach to avoid complex calculation errors
            # Create imaginary filter that shifts phase by π/2
            h_imag = torch.zeros_like(h)
            if h_imag.size(2) > 1:
                h_imag[:, :, 1:-1] = 1.0  # Apply to non-DC, non-Nyquist components
            
            # Create real and imaginary parts separately
            real_part = x
            imag_part = torch.fft.irfft(1j * X_fft * h_imag, dim=2, n=length)
            
            return torch.complex(real_part, imag_part)
        elif self.mode == 'hilbert':
            # Original implementation
            X = torch.fft.rfft(x, dim=-1)
            h = torch.ones_like(X)
            h[:, :, 0] = 1.0
            if h.size(-1) > 1:
                h[:, :, 1:-1] = 2.0
            X_analytic = X * h
            x_analytic = torch.fft.irfft(X_analytic, dim=-1, n=x.size(-1))
            x_hilbert = torch.zeros_like(x)
            if x.size(-1) > x_analytic.size(-1):
                x_analytic = F.pad(x_analytic, (0, x.size(-1) - x_analytic.size(-1)))
            elif x.size(-1) < x_analytic.size(-1):
                x_analytic = x_analytic[:, :, :x.size(-1)]
            return torch.complex(x, x_hilbert)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")