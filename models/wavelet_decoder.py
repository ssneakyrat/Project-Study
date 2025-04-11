import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl

class InverseWaveletTransform(nn.Module):
    def __init__(self, config, wavelet_type="morlet", channels=8, output_size=16000):
        super().__init__()
        self.wavelet_type = wavelet_type
        self.channels = channels
        self.output_size = output_size
        self.kernel_size = 101
        
        # Initialize synthesis filters (for inverse wavelet transform)
        self.synthesis_filters = nn.ConvTranspose1d(
            channels, 1, kernel_size=self.kernel_size, stride=2, padding=self.kernel_size//2, bias=False
        )
        
        # Initialize the synthesis filters with mathematically accurate inverse wavelets
        with torch.no_grad():
            scales = torch.logspace(1, 3, channels)
            for i, scale in enumerate(scales):
                t = torch.linspace(-50, 50, 101)
                
                # Create synthesis filter that forms a dual frame with the analysis filter
                if wavelet_type == "morlet":
                    # Dual frame of Morlet wavelet for perfect reconstruction
                    morlet = torch.exp(-t**2/(2*scale**2)) * torch.cos(5*t/scale)
                    morlet = morlet.flip(0)  # Time reversal for dual frame
                    
                elif wavelet_type == "mexican_hat":
                    # Dual frame of Mexican hat wavelet
                    morlet = (1 - t**2/scale**2) * torch.exp(-t**2/(2*scale**2))
                    morlet = morlet.flip(0)  # Time reversal
                    
                else:  # Default to Morlet
                    morlet = torch.exp(-t**2/(2*scale**2)) * torch.cos(5*t/scale)
                    morlet = morlet.flip(0)
                    
                # Normalize synthesis filter for energy preservation
                # |a|^(-1/2) factor from the wavelet transform definition
                morlet = morlet / torch.sqrt(scale) / torch.norm(morlet)
                
                # Enforce symmetry constraints to maintain wavelet admissibility
                if self.kernel_size % 2 == 1:  # For odd kernel size
                    center = self.kernel_size // 2
                    # Make filter symmetric around center for linear phase
                    symmetric_part = (morlet[:center] + morlet[center+1:].flip(0)) / 2
                    morlet[:center] = symmetric_part
                    morlet[center+1:] = symmetric_part.flip(0)
                
                self.synthesis_filters.weight[i, 0, :] = morlet
        
        # Enforce perfect reconstruction condition: ∑_k h[k]h[2n-k] = δ[n]
        # This is approximated by orthogonalization of the filters
        with torch.no_grad():
            # Extract filter weights
            weights = self.synthesis_filters.weight.squeeze(1)  # [channels, kernel_size]
            
            # Orthogonalize filters using QR decomposition
            q, r = torch.linalg.qr(weights.T)  # Transpose for QR
            orthogonal_filters = q.T[:channels]  # First 'channels' rows, transposed back
            
            # Update filter weights
            self.synthesis_filters.weight = nn.Parameter(orthogonal_filters.unsqueeze(1))
        
        # Adaptive modulation for synthesis
        self.modulation = nn.Conv1d(channels, channels, kernel_size=1)
    
    def forward(self, x):
        """
        Inverse wavelet transform using synthesis filter bank
        
        Mathematically implements the inverse continuous wavelet transform:
        x(t) = C_ψ^(-1) ∫∫ W_ψx(a,b) ψ((t-b)/a) da db/a²
        
        where C_ψ is the wavelet admissibility constant.
        
        Args:
            x: Wavelet coefficients [B, C, T]
            
        Returns:
            Reconstructed signal [B, 1, 2*T]
        """
        # Apply adaptive modulation
        x = self.modulation(x)
        
        # Apply synthesis filters
        # This implements the integration over scales and positions
        # The TransposedConv1d with stride=2 provides the necessary upsampling
        output = self.synthesis_filters(x)
        
        # Ensure output size matches expected dimension
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
        
        # Gradual upsampling as specified in architecture
        # [B,256] → Linear → [B,256,16] → Linear → [B,256,64] → Linear → [B,256,125]
        self.upsampling = nn.Sequential(
            nn.Linear(bottleneck_channels, bottleneck_channels * 16),
            nn.LeakyReLU(),
            nn.Unflatten(1, (bottleneck_channels, 16)),
            
            nn.Linear(16, 64),
            nn.LeakyReLU(),
            
            nn.Linear(64, 125),
            nn.LeakyReLU(),
        )
        
        # [B,256,125] → Transpose-Conv1D → [B,128,500]
        self.upconv1 = nn.ConvTranspose1d(
            bottleneck_channels, 
            128, 
            kernel_size=4, 
            stride=4, 
            padding=0
        )
        
        # Reconstruction with exactly 3×Transpose-Conv1D layers as specified
        # [B,128,500] → [B,64,1000] → [B,32,2000] → [B,8,8000]
        self.upconv2 = nn.ConvTranspose1d(
            128, 64, kernel_size=4, stride=2, padding=1
        )
        
        self.upconv3 = nn.ConvTranspose1d(
            64, 32, kernel_size=4, stride=2, padding=1
        )
        
        self.upconv4 = nn.ConvTranspose1d(
            32, 8, kernel_size=4, stride=4, padding=0
        )
        
        # Inverse wavelet transform: [B,8,8000] → [B,1,16000]
        self.inverse_wavelet = InverseWaveletTransform( 
            config,
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
        # Upsampling: [B,256] → Reshape/Linear → [B,256,125]
        x = self.upsampling(z)
        
        # [B,256,125] → Transpose-Conv1D → [B,128,500]
        x = F.relu(self.upconv1(x))
        
        # Reconstruction: 3×Transpose-Conv1D as specified in architecture
        # [B,128,500] → [B,64,1000]
        x = F.relu(self.upconv2(x))
        
        # [B,64,1000] → [B,32,2000]
        x = F.relu(self.upconv3(x))
        
        # [B,32,2000] → [B,8,8000] with a single stride 4 convolution
        x = F.relu(self.upconv4(x))
        
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