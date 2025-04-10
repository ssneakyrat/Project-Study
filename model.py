import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from complex_layers import (
    ComplexConv1d, ComplexConvTranspose1d, ComplexBatchNorm1d, 
    ComplexPReLU, ComplexToReal, RealToComplex
)
from wavelet_transform import WaveletScatteringTransform


class WSTVocoder(pl.LightningModule):
    def __init__(self,
                 sample_rate=16000,
                 wst_J=8,
                 wst_Q=8,
                 channels=[64, 128, 256],
                 latent_dim=64,
                 kernel_sizes=[5, 5, 5],
                 strides=[2, 2, 2],
                 compression_factor=16,
                 learning_rate=1e-4):
        """WST-based vocoder model.
        
        Args:
            sample_rate: Audio sample rate
            wst_J: Number of scales for WST
            wst_Q: Number of wavelets per octave for WST
            channels: List of channel dimensions for encoder/decoder
            latent_dim: Dimension of latent space
            kernel_sizes: List of kernel sizes for encoder/decoder
            strides: List of strides for encoder/decoder
            compression_factor: Compression factor for latent dimension
            learning_rate: Learning rate for optimizer
        """
        super().__init__()
        self.save_hyperparameters()
        
        self.sample_rate = sample_rate
        self.learning_rate = learning_rate
        self.channels = channels
        self.latent_dim = latent_dim
        
        # WST layer
        self.wst = WaveletScatteringTransform(
            J=wst_J,
            Q=wst_Q,
            T=sample_rate * 2,  # 2 seconds of audio
            max_order=2,
            out_type='array'
        )
        
        # Pre-compute actual WST output shape with a dummy input
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, sample_rate * 2)
            dummy_output = self.wst(dummy_input)
            self.wst_channels = dummy_output.shape[1]
            self.wst_time_dim = dummy_output.shape[2]
        
        # Store original length for final upsampling
        self.original_length = sample_rate * 2
        
        # Convert real audio to complex
        self.real_to_complex = RealToComplex(mode='zero_imag')
        
        # Encoder
        self.encoder_layers = nn.ModuleList()
        
        for i, (out_channels, kernel_size, stride) in enumerate(zip(channels, kernel_sizes, strides)):
            self.encoder_layers.append(
                nn.Sequential(
                    ComplexConv1d(
                        self.wst_channels if i == 0 else channels[i-1],
                        out_channels,
                        kernel_size,
                        stride=stride,
                        padding=kernel_size // 2
                    ),
                    ComplexBatchNorm1d(out_channels),
                    ComplexPReLU()
                )
            )
        
        # Latent projection
        self.latent_projection = ComplexConv1d(
            channels[-1],
            latent_dim,
            kernel_size=1
        )
        
        # Decoder
        self.decoder_layers = nn.ModuleList()
        
        for i, (in_channels, out_channels, kernel_size, stride) in enumerate(
            zip(
                [latent_dim] + channels[:-1],
                channels,
                kernel_sizes,
                strides
            )
        ):
            self.decoder_layers.append(
                nn.Sequential(
                    ComplexConvTranspose1d(
                        in_channels,
                        out_channels,
                        kernel_size,
                        stride=stride,
                        padding=kernel_size // 2,
                        output_padding=stride - 1
                    ),
                    ComplexBatchNorm1d(out_channels),
                    ComplexPReLU()
                )
            )
        
        # Output layer
        self.output_layer = ComplexConvTranspose1d(
            channels[-1],
            1,  # Output channels
            kernel_size=kernel_sizes[0],
            stride=strides[0],
            padding=kernel_sizes[0] // 2,
            output_padding=strides[0] - 1
        )
        
        # Convert complex back to real
        self.complex_to_real = ComplexToReal(mode='real')
        
        # Skip projections from encoder outputs to decoder inputs
        encoder_channels = channels  # [64, 128, 256]
        decoder_input_channels = [latent_dim] + channels[:-1]  # [64, 64, 128]
        
        self.skip_projections = nn.ModuleList()
        for i in range(len(encoder_channels)):
            enc_ch = encoder_channels[i]
            dec_idx = len(decoder_input_channels) - 1 - i
            dec_ch = decoder_input_channels[dec_idx]
            
            self.skip_projections.append(
                ComplexConv1d(enc_ch, dec_ch, kernel_size=1)
            )
        
        # Final temporal upsampling to match original audio length
        self.final_upsampling = nn.Upsample(size=sample_rate * 2, mode='linear', align_corners=False)
        
    def compute_expected_output_size(self, input_size, layers):
        """Calculate the expected output size after a series of conv/deconv layers."""
        size = input_size
        
        for i, (kernel_size, stride) in enumerate(zip(self.hparams.kernel_sizes, self.hparams.strides)):
            # For encoder (conv): output_size = (input_size + 2*padding - kernel_size) / stride + 1
            # With padding = kernel_size // 2
            size = (size + 2*(kernel_size // 2) - kernel_size) // stride + 1
            
        return size
        
    def forward(self, x):
        """
        Args:
            x: Input audio tensor of shape (batch_size, time)
        
        Returns:
            Reconstructed audio tensor of shape (batch_size, time)
        """
        batch_size = x.size(0)
        input_length = x.size(1)
        
        # Apply WST - will return shape [B, C, T'] where C = combined scattering orders and coefficients
        # and T' is the time dimension after wavelet transform (usually smaller than input)
        x_wst = self.wst(x.unsqueeze(1))
        
        # Log shape information
        B, C, T = x_wst.shape
        self.log("wst_channels", C, on_step=False, on_epoch=True)
        self.log("wst_timesteps", T, on_step=False, on_epoch=True)
        
        # Convert to complex
        x_complex = self.real_to_complex(x_wst)
        
        # Store skip connections
        skip_connections = []
        
        # Encoder
        for layer in self.encoder_layers:
            x_complex = layer(x_complex)
            skip_connections.append(x_complex)
        
        # Latent space
        z = self.latent_projection(x_complex)
        
        # Decoder with skip connections
        x_complex = z
        
        for i, layer in enumerate(self.decoder_layers):
            # Add skip connection before decoder layer
            skip_idx = len(skip_connections) - 1 - i
            if skip_idx >= 0:
                skip = skip_connections[skip_idx]
                
                # Project channels to match decoder input
                skip = self.skip_projections[skip_idx](skip)
                
                # Resize time dimension if needed using interpolation
                if skip.shape[2] != x_complex.shape[2]:
                    skip_real = F.interpolate(skip.real, size=x_complex.shape[2], mode='linear', align_corners=False)
                    skip_imag = F.interpolate(skip.imag, size=x_complex.shape[2], mode='linear', align_corners=False)
                    skip = torch.complex(skip_real, skip_imag)
                
                # Add skip connection
                x_complex = x_complex + skip
            
            # Apply decoder layer
            x_complex = layer(x_complex)
        
        # Output layer
        x_complex = self.output_layer(x_complex)
        
        # Convert to real
        x_out = self.complex_to_real(x_complex).squeeze(1)
        
        # The final output may still not match the original audio length
        # Use interpolation to match the exact input size
        if x_out.shape[1] != input_length:
            x_out = self.final_upsampling(x_out.unsqueeze(1)).squeeze(1)
            
        return x_out
    
    def training_step(self, batch, batch_idx):
        x = batch["audio"]
        x_hat = self(x)
        
        # Ensure shapes match
        if x_hat.shape != x.shape:
            self.log('shape_mismatch', 1.0)
            # Resize output to match input if needed (should not happen with proper upsampling)
            x_hat = F.interpolate(x_hat.unsqueeze(1), size=x.shape[1], mode='linear', align_corners=False).squeeze(1)
        
        # L1 loss
        loss = F.l1_loss(x_hat, x)
        
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch["audio"]
        x_hat = self(x)
        
        # Ensure shapes match
        if x_hat.shape != x.shape:
            self.log('shape_mismatch', 1.0)
            # Resize output to match input if needed (should not happen with proper upsampling)
            x_hat = F.interpolate(x_hat.unsqueeze(1), size=x.shape[1], mode='linear', align_corners=False).squeeze(1)
        
        # L1 loss
        loss = F.l1_loss(x_hat, x)
        
        self.log('val_loss', loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch"
            }
        }


class WSTVocoderLoss(nn.Module):
    """Loss function for WST vocoder.
    
    Combines L1 loss, spectral loss, and potentially adversarial loss.
    """
    def __init__(self, alpha=1.0, beta=1.0, gamma=0.0):
        """
        Args:
            alpha: Weight for L1 loss
            beta: Weight for spectral loss
            gamma: Weight for adversarial loss (if used)
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
    def forward(self, x_hat, x, d_fake=None):
        """
        Args:
            x_hat: Reconstructed audio
            x: Original audio
            d_fake: Discriminator output for fake audio (if using adversarial loss)
        
        Returns:
            Weighted sum of losses
        """
        # L1 loss
        l1_loss = F.l1_loss(x_hat, x)
        
        # Spectral loss (simplified for PoC)
        # In a full implementation, we would use STFT loss
        spec_loss = torch.tensor(0.0, device=x.device)
        
        # Adversarial loss (if using)
        adv_loss = torch.tensor(0.0, device=x.device)
        if d_fake is not None and self.gamma > 0:
            adv_loss = F.mse_loss(d_fake, torch.ones_like(d_fake))
        
        # Total loss
        total_loss = self.alpha * l1_loss + self.beta * spec_loss + self.gamma * adv_loss
        
        return total_loss