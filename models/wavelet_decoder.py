import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl

class InverseWaveletTransform(nn.Module):
    def __init__(self, wavelet_type="morlet", channels=8, output_size=16000):
        super().__init__()
        self.wavelet_type = wavelet_type
        self.channels = channels
        self.output_size = output_size
        
        # Inverse wavelet transform implemented as transposed convolution with stride=2 for upsampling
        self.synthesis_filters = nn.ConvTranspose1d(
            channels, 1, kernel_size=101, stride=2, padding=50, bias=False
        )
        
        # Initialize the synthesis filters with approximate inverse of analysis filters
        with torch.no_grad():
            scales = torch.logspace(1, 3, channels)
            for i, scale in enumerate(scales):
                t = torch.linspace(-50, 50, 101)
                # Approximate inverse of Morlet wavelet
                morlet = torch.exp(-t**2/(2*scale**2)) * torch.cos(5*t/scale)
                # Normalize
                morlet = morlet / torch.sqrt(scale) / torch.norm(morlet)
                self.synthesis_filters.weight[i, 0, :] = morlet
        
        # Feature modulation for adaptive synthesis
        self.feature_modulation = nn.Conv1d(channels, channels, kernel_size=1)
    
    def forward(self, x):
        """
        Inverse wavelet transform using synthesis filter bank with upsampling
        x: [B, C, T]
        Returns: [B, 1, 2*T] (upsampled by factor of 2)
        """
        # Apply feature modulation (adaptive part)
        x = self.feature_modulation(x)
        
        # Apply synthesis filters (inverse wavelet transform) with implicit upsampling (stride=2)
        output = self.synthesis_filters(x)
        
        # Make sure the output size matches the expected size
        if output.size(2) != self.output_size:
            output = F.interpolate(output, size=self.output_size, mode='linear', align_corners=False)
            
        return output

class WaveletDecoder(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        
        bottleneck_channels = config['model']['bottleneck_channels']
        hidden_channels = config['model']['hidden_channels']
        
        # Upsampling: [B,256] → Linear → [B,256,125] → Transpose-Conv1D → [B,128,500]
        self.initial_linear = nn.Linear(bottleneck_channels, bottleneck_channels * 125)
        
        self.upconv1 = nn.ConvTranspose1d(
            bottleneck_channels, 
            128, 
            kernel_size=4, 
            stride=4, 
            padding=0
        )
        
        # Reconstruction: 3×Transpose-Conv1D(kernel=4,stride=2)
        # [B,128,500] → [B,64,1000] → [B,32,2000] → [B,16,4000] → [B,8,8000]
        self.upconv2 = nn.ConvTranspose1d(
            128, 64, kernel_size=4, stride=2, padding=1
        )
        
        self.upconv3 = nn.ConvTranspose1d(
            64, 32, kernel_size=4, stride=2, padding=1
        )
        
        self.upconv4 = nn.ConvTranspose1d(
            32, 16, kernel_size=4, stride=2, padding=1
        )
        
        self.upconv5 = nn.ConvTranspose1d(
            16, 8, kernel_size=4, stride=2, padding=1
        )
        
        # Inverse wavelet transform: [B,8,8000] → [B,1,16000]
        self.inverse_wavelet = InverseWaveletTransform(
            wavelet_type=config['model']['wavelet_type'],
            channels=8,
            output_size=config['model']['input_size']
        )
    
    def forward(self, z):
        """
        Forward pass through the decoder
        z: [B, 256] bottleneck representation
        Returns: [B, 1, 16000] reconstructed audio
        """
        # Upsampling: [B,256] → Linear → [B,256,125]
        x = self.initial_linear(z)
        x = x.view(x.shape[0], -1, 125)
        
        # [B,256,125] → Transpose-Conv1D → [B,128,500]
        x = F.relu(self.upconv1(x))
        
        # Reconstruction: 3×Transpose-Conv1D
        # [B,128,500] → [B,64,1000]
        x = F.relu(self.upconv2(x))
        
        # [B,64,1000] → [B,32,2000]
        x = F.relu(self.upconv3(x))
        
        # [B,32,2000] → [B,16,4000]
        x = F.relu(self.upconv4(x))
        
        # [B,16,4000] → [B,8,8000]
        x = F.relu(self.upconv5(x))
        
        # Inverse wavelet transform: [B,8,8000] → [B,1,16000]
        x = self.inverse_wavelet(x)
        
        return x
    
    def training_step(self, batch, batch_idx):
        x = batch
        
        # For standalone testing, we generate random latent vectors
        # In real training this would come from the encoder
        batch_size = x.shape[0]
        z = torch.randn(batch_size, self.config['model']['bottleneck_channels'], device=x.device)
        
        # Decode
        x_hat = self(z)
        
        # MSE reconstruction loss
        loss = F.mse_loss(x_hat, x)
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch
        
        # Generate random latent vectors for standalone testing
        batch_size = x.shape[0]
        z = torch.randn(batch_size, self.config['model']['bottleneck_channels'], device=x.device)
        
        # Decode
        x_hat = self(z)
        
        # MSE reconstruction loss
        loss = F.mse_loss(x_hat, x)
        
        # Log metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
    
    def configure_optimizers(self):
        """Adam optimizer with learning rate from config"""
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.config['training']['learning_rate']
        )
        return optimizer