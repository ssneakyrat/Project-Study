import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from complex_layers import (
    ComplexConv2d, ComplexConvTranspose2d, ComplexBatchNorm2d, 
    ComplexPReLU, ComplexToReal, RealToComplex
)
from wavelet_transform import WaveletScatteringTransform


class EncoderBlock(nn.Module):
    """Standardized encoder block to ensure consistent dimensions"""
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv = ComplexConv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = ComplexBatchNorm2d(out_channels)
        self.prelu = ComplexPReLU()
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.prelu(x)
        return x


class DecoderBlock(nn.Module):
    """Standardized decoder block to ensure consistent dimensions"""
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding):
        super().__init__()
        self.conv_transpose = ComplexConvTranspose2d(
            in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.bn = ComplexBatchNorm2d(out_channels)
        self.prelu = ComplexPReLU()
        
    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.bn(x)
        x = self.prelu(x)
        return x


class WSTVocoder(pl.LightningModule):
    def __init__(self,
                 sample_rate=16000,
                 wst_J=6,  # Reduced from 8
                 wst_Q=6,  # Reduced from 8
                 channels=[32, 64, 128],  # Reduced from [64, 128, 256]
                 latent_dim=32,  # Reduced from 64
                 kernel_size=3,
                 stride=2,
                 compression_factor=16,
                 learning_rate=1e-4):
        """WST-based vocoder model with standardized dimensions.
        
        Args:
            sample_rate: Audio sample rate
            wst_J: Number of scales for WST
            wst_Q: Number of wavelets per octave for WST
            channels: List of channel dimensions for encoder/decoder
            latent_dim: Dimension of latent space
            kernel_size: Kernel size for all layers (int or tuple)
            stride: Stride for all layers (int or tuple)
            compression_factor: Compression factor for latent dimension
            learning_rate: Learning rate for optimizer
        """
        super().__init__()
        self.save_hyperparameters()
        
        self.sample_rate = sample_rate
        self.learning_rate = learning_rate
        
        # Standardize kernel_size and stride to tuples
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
            
        # Calculate padding based on kernel size
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        output_padding = (stride[0] - 1, stride[1] - 1)
        
        # WST layer with reduced parameters to save memory
        self.wst = WaveletScatteringTransform(
            J=min(6, wst_J),  # Cap at 6 to limit memory
            Q=min(6, wst_Q),  # Cap at 6 to limit memory
            T=sample_rate,    # Reduced from 2 seconds to 1 second
            max_order=1,      # Reduced from 2 to 1
            out_type='array'
        )
        
        # Convert real audio to complex
        self.real_to_complex = RealToComplex(mode='zero_imag')
        
        # Initial projection layer handles the reduced WST dimension (now 3 channels)
        self.initial_projection = ComplexConv2d(3, channels[0], kernel_size=1)
        
        # Encoder blocks
        self.encoder_blocks = nn.ModuleList()
        for i in range(len(channels)-1):
            self.encoder_blocks.append(
                EncoderBlock(
                    channels[i],
                    channels[i+1],
                    kernel_size,
                    stride,
                    padding
                )
            )
        
        # Latent projection - derive bottleneck dimension using the formula
        # d = √(T×F)/c where T=time, F=frequency bins, c=compression factor
        T_reduced = sample_rate // (2 ** (len(channels)))  # Time after encoder
        F_reduced = 3  # Frequency bins after WST reduction
        computed_latent = int((T_reduced * F_reduced) ** 0.5 / compression_factor)
        self.latent_dim = min(latent_dim, computed_latent)  # Use smaller of two values
        
        self.latent_projection = ComplexConv2d(
            channels[-1],
            self.latent_dim,
            kernel_size=1
        )
        
        # Decoder blocks
        self.decoder_blocks = nn.ModuleList()
        reversed_channels = list(reversed(channels))
        
        # First decoder block from latent to first decoder channel
        self.decoder_blocks.append(
            DecoderBlock(
                self.latent_dim,
                reversed_channels[0],  # First decoder channel (= channels[-1])
                kernel_size,
                stride,
                padding,
                output_padding
            )
        )
        
        # Remaining decoder blocks
        for i in range(len(reversed_channels)-1):
            self.decoder_blocks.append(
                DecoderBlock(
                    reversed_channels[i],
                    reversed_channels[i+1],
                    kernel_size,
                    stride,
                    padding,
                    output_padding
                )
            )
        
        # Skip connections (1×1 projections) to handle potential channel mismatches
        self.skip_connections = nn.ModuleList()
        # Skip from encoder output to first decoder block
        self.skip_connections.append(
            ComplexConv2d(channels[-1], reversed_channels[0], kernel_size=1) 
            if channels[-1] != reversed_channels[0] else nn.Identity()
        )
        
        # Remaining skip connections
        for i in range(len(channels) - 1):
            enc_ch = channels[-(i+2)]  # Going backwards from the second-to-last encoder layer
            dec_ch = reversed_channels[i+1]  # Going forwards from the second decoder layer
            
            self.skip_connections.append(
                ComplexConv2d(enc_ch, dec_ch, kernel_size=1) 
                if enc_ch != dec_ch else nn.Identity()
            )
        
        # Output layer
        self.output_layer = ComplexConvTranspose2d(
            channels[0],  # Last decoder channel
            1,  # Output channels
            kernel_size,
            stride,
            padding,
            output_padding
        )
        
        # Convert complex back to real
        self.complex_to_real = ComplexToReal(mode='real')
        
    def _resize_feature_map(self, source, target):
        """Resize source feature map to match target spatial dimensions"""
        if source.shape[2:] == target.shape[2:]:
            return source
            
        if torch.is_complex(source):
            real_resized = F.interpolate(
                source.real, 
                size=target.shape[2:], 
                mode='bilinear', 
                align_corners=False
            )
            imag_resized = F.interpolate(
                source.imag, 
                size=target.shape[2:], 
                mode='bilinear', 
                align_corners=False
            )
            return torch.complex(real_resized, imag_resized)
        else:
            return F.interpolate(
                source, 
                size=target.shape[2:], 
                mode='bilinear', 
                align_corners=False
            )
        
    def forward(self, x):
        """
        Args:
            x: Input audio tensor of shape (batch_size, time)
        
        Returns:
            Reconstructed audio tensor of shape (batch_size, time)
        """
        # Apply WST - output is now [B, 3, T, F] with reduced dimensions
        x_wst = self.wst(x.unsqueeze(1))
        
        # Convert to complex
        x_complex = self.real_to_complex(x_wst)
        
        # Initial projection
        x = self.initial_projection(x_complex)
        
        # Store encoder features for skip connections
        encoder_features = []
        encoder_features.append(x)  # Store initial feature
        
        # Encoder
        for block in self.encoder_blocks:
            x = block(x)
            encoder_features.append(x)
        
        # Latent space
        z = self.latent_projection(x)
        
        # Decoder with skip connections
        x = z
        
        for i, block in enumerate(self.decoder_blocks):
            # Apply decoder block
            x = block(x)
            
            # Add skip connection if not the output block
            if i < len(self.skip_connections):
                # Get the corresponding encoder feature map
                skip_source = encoder_features[-(i+1)]
                
                # Apply channel projection if needed
                skip = self.skip_connections[i](skip_source)
                
                # Resize to match spatial dimensions if needed
                skip = self._resize_feature_map(skip, x)
                
                # Add skip connection
                x = x + skip
                
            # Clear encoder features after use to save memory
            if i > 0 and i < len(encoder_features):
                # Delete reference to free memory
                encoder_features[-(i)] = None
        
        # Output layer
        x = self.output_layer(x)
        
        # Convert to real
        x_out = self.complex_to_real(x)
        
        # Average over frequency dimension to get back to audio
        # x_out shape is [batch_size, 1, time, freq]
        x_out = x_out.mean(dim=3)
        
        return x_out.squeeze(1)
    
    def training_step(self, batch, batch_idx):
        # Free memory before processing batch
        torch.cuda.empty_cache()
        
        x = batch["audio"]
        x_hat = self(x)
        
        # Ensure same length (WST might change the length)
        min_len = min(x.shape[1], x_hat.shape[1])
        x = x[:, :min_len]
        x_hat = x_hat[:, :min_len]
        
        # L1 loss
        loss = F.l1_loss(x_hat, x)
        
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        # Free memory before processing batch
        torch.cuda.empty_cache()
        
        x = batch["audio"]
        x_hat = self(x)
        
        # Ensure same length (WST might change the length)
        min_len = min(x.shape[1], x_hat.shape[1])
        x = x[:, :min_len]
        x_hat = x_hat[:, :min_len]
        
        # L1 loss
        loss = F.l1_loss(x_hat, x)
        
        self.log('val_loss', loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer