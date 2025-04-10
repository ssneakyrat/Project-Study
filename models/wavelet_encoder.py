import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl

class AdaptiveWaveletTransform(nn.Module):
    def __init__(self, wavelet_type="morlet", channels=16, input_size=16000):
        super().__init__()
        self.wavelet_type = wavelet_type
        self.channels = channels
        self.input_size = input_size
        self.output_size = input_size // 2
        
        # Create fixed filter banks instead of adaptive filters for initial implementation
        # We'll use Morlet wavelets with different scales
        self.filter_bank = nn.Conv1d(
            1, channels, kernel_size=101, padding=50, bias=False
        )
        
        # Initialize the filter bank with Morlet wavelets at different scales
        with torch.no_grad():
            scales = torch.logspace(1, 3, channels)
            for i, scale in enumerate(scales):
                t = torch.linspace(-50, 50, 101)
                # Morlet wavelet: e^(-t²/2) * cos(5t)
                morlet = torch.exp(-t**2/(2*scale**2)) * torch.cos(5*t/scale)
                # Normalize
                morlet = morlet / torch.sqrt(scale) / torch.norm(morlet)
                self.filter_bank.weight[i, 0, :] = morlet
        
        # Simple channel attention to allow adaptive weighting
        self.channel_attention = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Feature modulation - approximates the adaptive part
        self.feature_modulation = nn.Conv1d(channels, channels, kernel_size=1)
    
    def forward(self, x):
        """
        Simplified wavelet transform using fixed filter bank
        x: Tensor of shape [B, 1, T]
        Returns: Tensor of shape [B, C, T/2]
        """
        # Apply filter bank (equivalent to wavelet transform with fixed wavelets)
        # This implements W_ψx = |a|^(-1/2) ∫x(t)ψ((t-b)/a)dt with fixed ψ
        wavelet_output = self.filter_bank(x)  # [B, C, T]
        
        # Apply channel attention (adaptive weighting)
        attention = self.channel_attention(wavelet_output)
        wavelet_output = wavelet_output * attention
        
        # Apply feature modulation (approximates the adaptive part of ψ_θ(t) = ψ(t) · f_θ(t))
        wavelet_output = self.feature_modulation(wavelet_output)
        
        # Downsample to target length
        return wavelet_output[:, :, ::2]  # [B, C, T/2]

class WaveletEncoder(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        
        # Adaptive wavelet transform: [B,1,16000] → [B,16,8000]
        self.wavelet_transform = AdaptiveWaveletTransform(
            wavelet_type=config['model']['wavelet_type'],
            channels=config['model']['wavelet_channels'],
            input_size=config['model']['input_size']
        )
        
        # Feature extraction: 2×Conv1D(kernel=3,stride=2) → [B,32,2000]
        self.conv1 = nn.Conv1d(
            config['model']['wavelet_channels'],
            config['model']['hidden_channels'],
            kernel_size=3,
            stride=2,
            padding=1
        )
        
        self.conv2 = nn.Conv1d(
            config['model']['hidden_channels'],
            config['model']['hidden_channels'],
            kernel_size=3,
            stride=2,
            padding=1
        )
        
        # Bottleneck: Conv1D(kernel=5) → [B,256,2000] → Avg-pool → [B,256]
        self.bottleneck = nn.Conv1d(
            config['model']['hidden_channels'],
            config['model']['bottleneck_channels'],
            kernel_size=5,
            stride=1,
            padding=2
        )
        
        # Global average pooling
        self.global_avgpool = nn.AdaptiveAvgPool1d(1)
    
    def forward(self, x):
        """
        Forward pass through the encoder
        x: [B, 1, T] where T is input_size (16000)
        Returns: [B, 256] bottleneck representation
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
        
        return x
    
    def training_step(self, batch, batch_idx):
        x = batch
        
        # Encode
        z = self(x)
        
        # Use MSE loss between z and a random target for testing
        # In a real implementation, this would be replaced with the actual loss
        target = torch.randn_like(z)
        loss = F.mse_loss(z, target)
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch
        
        # Encode
        z = self(x)
        
        # Simple validation metric
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