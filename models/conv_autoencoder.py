import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvEncoder(nn.Module):
    """Convolutional encoder for wavelet coefficients
    
    Uses 1D convolutions with appropriate reshaping for efficient parameter usage.
    Includes residual connections for improved gradient flow during training.
    """
    def __init__(self, input_dim, latent_dim, base_channels=32):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Use more aggressive dimensionality reduction for large inputs
        # Calculate intermediate dimension with logarithmic scaling
        if input_dim > 16384:
            intermediate_dim = 2048
        elif input_dim > 8192:
            intermediate_dim = 1024
        elif input_dim > 4096:
            intermediate_dim = 512
        else:
            intermediate_dim = 256
            
        # Fixed sequence length - reshape to this for convolution
        seq_len = 64
        channels = max(8, intermediate_dim // seq_len)
        
        # Much smaller projection dimension to reduce parameters
        self.projection_dim = channels * seq_len
        
        # Initial projection from input to shaped tensor with bottleneck
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, intermediate_dim),  # First reduction
            nn.LeakyReLU(0.2),
            nn.Linear(intermediate_dim, self.projection_dim),  # Second reduction
            nn.LeakyReLU(0.2)
        )
        
        # Convolutional encoder
        self.conv_encoder = nn.Sequential(
            # Reshape for convolution
            Lambda(lambda x: x.view(x.size(0), channels, seq_len)),
            
            # First convolutional block with residual connection
            ResidualConvBlock(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.AvgPool1d(2),  # seq_len → seq_len/2
            
            # Second convolutional block
            ConvBlock(channels, channels*2, kernel_size=3, stride=1, padding=1),
            nn.AvgPool1d(2),  # seq_len/2 → seq_len/4
            
            # Third convolutional block
            ConvBlock(channels*2, channels*4, kernel_size=3, stride=1, padding=1),
            nn.AvgPool1d(2),  # seq_len/4 → seq_len/8
            
            # Flatten for final projection
            nn.Flatten()
        )
        
        # Calculate size after convolutions
        self.conv_output_size = (channels * 4) * (seq_len // 8)
        
        # Projection to latent space
        self.latent_projection = nn.Sequential(
            nn.Linear(self.conv_output_size, latent_dim)
        )
    
    def forward(self, x):
        # Initial projection
        x = self.input_projection(x)
        
        # Convolutional encoding
        x = self.conv_encoder(x)
        
        # Project to latent space
        z = self.latent_projection(x)
        
        return z

class ConvDecoder(nn.Module):
    """Convolutional decoder for wavelet coefficients
    
    Uses transposed convolutions to efficiently decode latent representation.
    Mirror of encoder with residual connections for improved gradient flow.
    """
    def __init__(self, latent_dim, output_dim, base_channels=32):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        
        # Use more aggressive dimensionality reduction for large outputs
        # Calculate intermediate dimension with logarithmic scaling
        if output_dim > 16384:
            intermediate_dim = 2048
        elif output_dim > 8192:
            intermediate_dim = 1024
        elif output_dim > 4096:
            intermediate_dim = 512
        else:
            intermediate_dim = 256
        
        # Fixed sequence length
        seq_len = 64
        channels = max(8, intermediate_dim // seq_len)
        
        # Much smaller projection dimension
        self.projection_dim = channels * seq_len
        
        # Calculate dimensions after convolutions (before upsampling)
        self.conv_input_size = (channels * 4) * (seq_len // 8)
        
        # Projection from latent space to convolution input with smaller sizes
        self.latent_projection = nn.Sequential(
            nn.Linear(latent_dim, min(512, self.conv_input_size)),
            nn.LeakyReLU(0.2),
            nn.Linear(min(512, self.conv_input_size), self.conv_input_size),
            nn.LeakyReLU(0.2)
        )
        
        # Convolutional decoder with upsampling
        self.conv_decoder = nn.Sequential(
            # Reshape for convolution
            Lambda(lambda x: x.view(x.size(0), channels*4, seq_len//8)),
            
            # First upsampling block
            ConvTransposeBlock(channels*4, channels*2, kernel_size=4, stride=2, padding=1),
            
            # Second upsampling block
            ConvTransposeBlock(channels*2, channels, kernel_size=4, stride=2, padding=1),
            
            # Third upsampling block with residual connection
            ResidualConvTransposeBlock(channels, channels, kernel_size=4, stride=2, padding=1),
            
            # Flatten for final projection
            nn.Flatten()
        )
        
        # Final projection to output space with bottleneck
        self.output_projection = nn.Sequential(
            nn.Linear(self.projection_dim, intermediate_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(intermediate_dim, output_dim)
        )
    
    def forward(self, z):
        # Project from latent space
        x = self.latent_projection(z)
        
        # Convolutional decoding with upsampling
        x = self.conv_decoder(x)
        
        # Final projection to output space
        output = self.output_projection(x)
        
        return output

class ConvWaveletAutoencoder(nn.Module):
    """Combined convolutional encoder-decoder for wavelet coefficients
    
    Uses convolutional layers for parameter efficiency and improved
    frequency locality with residual connections for better training dynamics.
    """
    def __init__(self, input_dim, latent_dim, base_channels=64):
        super().__init__()
        
        self.encoder = ConvEncoder(input_dim, latent_dim, base_channels)
        self.decoder = ConvDecoder(latent_dim, input_dim, base_channels)
    
    def forward(self, x):
        """Forward pass through autoencoder"""
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z


# Helper modules for convolutional networks

class Lambda(nn.Module):
    """Lambda layer for custom operations"""
    def __init__(self, func):
        super().__init__()
        self.func = func
    
    def forward(self, x):
        return self.func(x)

class ConvBlock(nn.Module):
    """Basic convolutional block with normalization and activation"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        self.act = nn.LeakyReLU(0.2)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x

class ResidualConvBlock(nn.Module):
    """Convolutional block with residual connection"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, out_channels, kernel_size, stride, padding)
        self.conv2 = ConvBlock(out_channels, out_channels, kernel_size, stride, padding)
        
        # Residual connection
        self.residual = nn.Identity() if in_channels == out_channels else nn.Conv1d(in_channels, out_channels, 1)
    
    def forward(self, x):
        residual = self.residual(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return x + residual

class ConvTransposeBlock(nn.Module):
    """Transposed convolutional block for upsampling"""
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        super().__init__()
        self.conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        self.act = nn.LeakyReLU(0.2)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x

class ResidualConvTransposeBlock(nn.Module):
    """Transposed convolutional block with residual connection"""
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        super().__init__()
        self.conv1 = ConvTransposeBlock(in_channels, out_channels, kernel_size, stride, padding)
        self.conv2 = ConvBlock(out_channels, out_channels, 3, 1, 1)
        
        # Residual connection with upsampling
        self.residual = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv1d(in_channels, out_channels, 1)
        ) if stride > 1 else (
            nn.Identity() if in_channels == out_channels else nn.Conv1d(in_channels, out_channels, 1)
        )
    
    def forward(self, x):
        residual = self.residual(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return x + residual