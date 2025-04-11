import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl

class WaveletParameterNetwork(nn.Module):
    """
    Neural network that learns to modulate the wavelet function
    ψ_θ(t) = ψ(t) · f_θ(t) where f_θ is learnable
    
    Implements a 3-layer MLP with ~2K parameters as specified in the architecture
    """
    def __init__(self, config, channels=16, kernel_size=101):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        
        # Fixed 3-layer MLP architecture with approximately 2K parameters
        # Input dimension: 1 (time position)
        # Hidden dimensions: 16, 32
        # Output dimension: channels
        self.mlp = nn.Sequential(
            nn.Linear(1, 16),                # 1*16 + 16 = 32 parameters
            nn.LeakyReLU(),
            nn.Linear(16, 32),               # 16*32 + 32 = 544 parameters
            nn.LeakyReLU(),
            nn.Linear(32, channels),         # 32*channels + channels = 32*16 + 16 = 528 parameters
            nn.Sigmoid()                     # Total: ~1104 parameters for channels=16
        )
        
    def forward(self, t):
        """
        Generate modulation factors for wavelet functions
        t: Tensor of shape [kernel_size] with normalized positions in [-1, 1]
        Returns: Tensor of shape [channels, kernel_size]
        """
        # Reshape t for processing
        t = t.view(-1, 1)  # [kernel_size, 1]
        
        # Generate modulation factors for each position
        modulation = self.mlp(t)  # [kernel_size, channels]
        
        # Transpose to get [channels, kernel_size]
        return modulation.transpose(0, 1)

class AdaptiveWaveletTransform(nn.Module):
    def __init__(self, config, wavelet_type="morlet", channels=16, input_size=16000):
        super().__init__()
        self.wavelet_type = wavelet_type
        self.channels = channels
        self.input_size = input_size
        self.output_size = input_size // 2
        self.kernel_size = 101
        
        # Base wavelet filter bank
        self.filter_bank = nn.Conv1d(
            1, channels, kernel_size=self.kernel_size, padding=self.kernel_size//2, bias=False
        )
        
        # Initialize with selected wavelet type
        self._initialize_wavelets(wavelet_type)
        
        # Wavelet parameter network for adaptive modulation
        # Implements ψ_θ(t) = ψ(t) · f_θ(t)
        self.parameter_network = WaveletParameterNetwork(config, channels, self.kernel_size)
        
        # Register normalized position vector as a buffer (non-parameter tensor)
        t = torch.linspace(-1, 1, self.kernel_size)
        self.register_buffer('t', t)
        
        # Adaptive feature modulation
        self.feature_modulation = nn.Conv1d(channels, channels, kernel_size=1)
    
    def _initialize_wavelets(self, wavelet_type):
        """Initialize filter bank with specified wavelet type"""
        with torch.no_grad():
            scales = torch.logspace(1, 3, self.channels)
            t = torch.linspace(-50, 50, self.kernel_size)
            
            for i, scale in enumerate(scales):
                if wavelet_type == "morlet":
                    # Morlet wavelet: e^(-t²/2σ²) * cos(5t/σ)
                    wavelet = torch.exp(-t**2/(2*scale**2)) * torch.cos(5*t/scale)
                elif wavelet_type == "mexican_hat":
                    # Mexican hat (second derivative of Gaussian): (1 - t²/σ²) * e^(-t²/2σ²)
                    wavelet = (1 - t**2/scale**2) * torch.exp(-t**2/(2*scale**2))
                elif wavelet_type == "learnable":
                    # Initialize with Morlet but will be fully learnable
                    wavelet = torch.exp(-t**2/(2*scale**2)) * torch.cos(5*t/scale)
                    # Don't freeze this parameter
                    self.filter_bank.weight.requires_grad = True
                else:
                    raise ValueError(f"Unknown wavelet type: {wavelet_type}")
                
                # Normalize according to the wavelet transform formula: |a|^(-1/2)
                wavelet = wavelet / torch.sqrt(scale) / torch.norm(wavelet)
                self.filter_bank.weight[i, 0, :] = wavelet
                
            # Only freeze weights if not using learnable wavelets
            if wavelet_type != "learnable":
                self.filter_bank.weight.requires_grad = False
    
    def forward(self, x):
        """
        Adaptive wavelet transform implementing W_ψx = |a|^(-1/2) ∫x(t)ψ_θ((t-b)/a)dt
        x: Tensor of shape [B, 1, T]
        Returns: Tensor of shape [B, C, T/2]
        """
        batch_size = x.shape[0]
        
        # Get modulation factors from parameter network
        # This implements f_θ(t) in ψ_θ(t) = ψ(t) · f_θ(t)
        modulation = self.parameter_network(self.t)  # [C, K]
        
        # Apply modulation to filter bank weights to create adaptive wavelets
        # This implements ψ_θ(t) = ψ(t) · f_θ(t)
        adaptive_filters = self.filter_bank.weight * modulation.unsqueeze(1)  # [C, 1, K]
        
        # Apply adaptive wavelet transform using F.conv1d
        # This implements W_ψx = |a|^(-1/2) ∫x(t)ψ_θ((t-b)/a)dt
        wavelet_output = F.conv1d(
            x, adaptive_filters, padding=self.kernel_size//2
        )
        
        # Apply feature modulation for additional adaptivity
        wavelet_output = self.feature_modulation(wavelet_output)
        
        # Downsample to target length
        return wavelet_output[:, :, ::2]  # [B, C, T/2]

class WaveletEncoder(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        
        # Extract configuration
        input_size = config['model']['input_size']
        wavelet_channels = config['model']['wavelet_channels']
        hidden_channels = config['model']['hidden_channels']
        bottleneck_channels = config['model']['bottleneck_channels']
        wavelet_type = config['model']['wavelet_type']
        
        # Adaptive wavelet transform: [B,1,16000] → [B,16,8000]
        self.wavelet_transform = AdaptiveWaveletTransform(
            config=config,
            wavelet_type=wavelet_type,
            channels=wavelet_channels,
            input_size=input_size
        )
        
        # Feature extraction: 2×Conv1D(kernel=3,stride=2) → [B,32,2000]
        self.conv1 = nn.Conv1d(
            wavelet_channels,
            hidden_channels,
            kernel_size=3,
            stride=2,
            padding=1
        )
        
        self.conv2 = nn.Conv1d(
            hidden_channels,
            hidden_channels,
            kernel_size=3,
            stride=2,
            padding=1
        )
        
        # Bottleneck: Conv1D(kernel=5) → [B,256,2000] → Avg-pool → [B,256]
        self.bottleneck = nn.Conv1d(
            hidden_channels,
            bottleneck_channels,
            kernel_size=5,
            stride=1,
            padding=2
        )
        
        # Global average pooling
        self.global_avgpool = nn.AdaptiveAvgPool1d(1)
        
        # Additional outputs for VAE-style latent space, enables KL loss
        self.fc_mean = nn.Linear(bottleneck_channels, bottleneck_channels)
        self.fc_logvar = nn.Linear(bottleneck_channels, bottleneck_channels)
    
    def forward(self, x):
        """
        Forward pass through the encoder
        x: [B, 1, T] where T is input_size (16000)
        Returns: 
          z: Sampled latent vector [B, 256]
          z_mean: Mean vector [B, 256]
          z_logvar: Log variance vector [B, 256]
        """
        # Wavelet transform: [B, 1, 16000] → [B, 16, 8000]
        x = self.wavelet_transform(x)
        
        # Feature extraction: 2×Conv1D → [B, 32, 2000]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        
        # Bottleneck: Conv1D → [B, 256, 2000]
        x = F.relu(self.bottleneck(x))
        
        # Global avg pooling → [B, 256, 1]
        x = self.global_avgpool(x)
        
        # Flatten → [B, 256]
        x = x.flatten(1)
        
        # VAE reparameterization
        z_mean = self.fc_mean(x)
        z_logvar = self.fc_logvar(x)
        
        # Sample from the distribution
        std = torch.exp(0.5 * z_logvar)
        eps = torch.randn_like(std)
        z = z_mean + eps * std
        
        return z, z_mean, z_logvar
    
    def training_step(self, batch, batch_idx):
        x = batch
        
        # Encode
        z, z_mean, z_logvar = self(x)
        
        # For standalone encoder testing only
        # In real implementation, loss calculation is handled by the full model
        target = torch.randn_like(z)
        loss = F.mse_loss(z, target)
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch
        
        # Encode
        z, z_mean, z_logvar = self(x)
        
        # Simple validation metric for standalone testing
        target = torch.randn_like(z)
        loss = F.mse_loss(z, target)
        
        # Log metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
    
    def configure_optimizers(self):
        """Adam optimizer with learning rate from config"""
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.config['training']['learning_rate']
        )
        return optimizer