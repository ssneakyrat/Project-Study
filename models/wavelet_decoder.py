import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl
from utils.loss import TriangularWindow

class InverseWaveletTransform(nn.Module):
    def __init__(self, config, wavelet_type="morlet", channels=16, output_size=16000):
        super().__init__()
        self.wavelet_type = wavelet_type
        self.channels = channels  # Store the expected number of channels
        self.output_size = output_size
        self.kernel_size = 101
        
        # Initialize synthesis filters (for inverse wavelet transform)
        self.synthesis_filters = nn.ConvTranspose1d(
            channels, 1, kernel_size=self.kernel_size, stride=2, padding=self.kernel_size//2, bias=False
        )
        
        # Initialize with basic filters - these will be properly set by the encoder
        self._initialize_wavelets(wavelet_type)
        
        # Channel-wise scaling - dynamically handle channel count in forward()
        self.register_parameter('scales', nn.Parameter(torch.ones(channels)))
        
        # Flag to indicate if encoder parameters have been set
        self.encoder_params_set = False
    
    def _initialize_wavelets(self, wavelet_type):
        """Initialize with basic wavelet filters that will be replaced"""
        with torch.no_grad():
            scales = torch.logspace(1, 3, self.channels)
            for i, scale in enumerate(scales):
                t = torch.linspace(-50, 50, self.kernel_size)
                
                if wavelet_type == "morlet":
                    wavelet = torch.exp(-t**2/(2*scale**2)) * torch.cos(5*t/scale)
                    wavelet = wavelet.flip(0)  # Time reversal for synthesis
                elif wavelet_type == "mexican_hat":
                    wavelet = (1 - t**2/scale**2) * torch.exp(-t**2/(2*scale**2))
                    wavelet = wavelet.flip(0)
                else:
                    wavelet = torch.exp(-t**2/(2*scale**2)) * torch.cos(5*t/scale)
                    wavelet = wavelet.flip(0)
                
                # Normalize
                wavelet = wavelet / torch.norm(wavelet)
                self.synthesis_filters.weight[i, 0, :] = wavelet
    
    def set_encoder_parameters(self, wavelet_params):
        """
        Set synthesis filters directly from encoder analysis filters with minimal processing
        
        Args:
            wavelet_params: Dictionary containing 'adaptive_filters'
        """
        if wavelet_params is None or 'adaptive_filters' not in wavelet_params:
            return
        
        # Get analysis filters from encoder
        analysis_filters = wavelet_params['adaptive_filters']
        
        # Check for channel mismatch and adapt accordingly
        encoder_channels = analysis_filters.size(0)
        if encoder_channels != self.channels:
            # Rebuild synthesis filters with correct channel count
            self.channels = encoder_channels
            self.synthesis_filters = nn.ConvTranspose1d(
                self.channels, 1, kernel_size=self.kernel_size, 
                stride=2, padding=self.kernel_size//2, bias=False
            ).to(analysis_filters.device)
            
            # Update scales parameter
            self.register_parameter('scales', nn.Parameter(torch.ones(self.channels, device=analysis_filters.device)))
        
        # Simple approach: just use time-reversed analysis filters
        with torch.no_grad():
            # Time-reverse and transpose for TransposedConv1d
            synthesis_filters = analysis_filters.flip(2)
            self.synthesis_filters.weight.copy_(synthesis_filters)
            
            # Reset scaling factors to identity
            self.scales.data.fill_(1.0)
        
        self.encoder_params_set = True
    
    def forward(self, x):
        """
        Simple inverse wavelet transform with dynamic channel handling
        
        Args:
            x: Wavelet coefficients [B, C, T]
            
        Returns:
            Reconstructed signal [B, 1, 2*T]
        """
        # Handle dynamic channel count
        input_channels = x.size(1)
        if input_channels != self.channels:
            # If this happens, it means the encoder hasn't properly set parameters
            # and we need to adapt to the incoming channel count
            device = x.device
            
            # Rebuild with correct channel count
            self.channels = input_channels
            self.synthesis_filters = nn.ConvTranspose1d(
                self.channels, 1, kernel_size=self.kernel_size, 
                stride=2, padding=self.kernel_size//2, bias=False
            ).to(device)
            
            # Re-initialize filters
            self._initialize_wavelets(self.wavelet_type)
            
            # Recreate scales parameter with correct size
            self.register_parameter('scales', nn.Parameter(torch.ones(self.channels, device=device)))
        
        # Apply channel-wise scaling - reshape scales to [C, 1, 1] for broadcasting
        x = x * self.scales.view(-1, 1, 1)
        
        # Apply synthesis filters
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
        wavelet_channels = config['model']['wavelet_channels']
        
        # Upsampling: [B,256] → [B,256,2000]
        self.upsampling = nn.Sequential(
            nn.Linear(bottleneck_channels, bottleneck_channels * 16),
            nn.LeakyReLU(),
            nn.Unflatten(1, (bottleneck_channels, 16)),
            nn.Upsample(scale_factor=125, mode='nearest')
        )
        
        # Channel reduction: [B,256,2000] → [B,32,2000]
        self.channel_reduction = nn.Conv1d(
            bottleneck_channels, 
            hidden_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )
        
        # [B,32,2000] → [B,32,4000]
        self.upconv1 = nn.ConvTranspose1d(
            hidden_channels,
            hidden_channels, 
            kernel_size=4, 
            stride=2, 
            padding=1
        )
        
        # [B,32,4000] → [B,wavelet_channels,8000]
        self.upconv2 = nn.ConvTranspose1d(
            hidden_channels,
            wavelet_channels, 
            kernel_size=4, 
            stride=2, 
            padding=1
        )
        
        # Inverse wavelet transform: [B,wavelet_channels,8000] → [B,1,16000]
        self.inverse_wavelet = InverseWaveletTransform( 
            config,
            wavelet_type=config['model']['wavelet_type'],
            channels=wavelet_channels,
            output_size=config['model']['input_size']
        )
        
        # Triangular window for overlap-add reconstruction
        self.window = TriangularWindow(config['model']['input_size'])
        
        # Track previous frame for overlapping
        self.prev_frame = None
        
        # Overlap factor for frame processing
        self.overlap_factor = 0.25
    
    def set_encoder_parameters(self, wavelet_params):
        """Set the encoder parameters for the inverse wavelet transform"""
        if wavelet_params is not None:
            self.inverse_wavelet.set_encoder_parameters(wavelet_params)
    
    def forward(self, z):
        """
        Forward pass through the decoder
        z: [B, 256] bottleneck representation
        Returns: [B, 1, 16000] reconstructed audio
        """
        # Upsampling: [B,256] → [B,256,2000]
        x = self.upsampling(z)
        
        # Channel reduction: [B,256,2000] → [B,32,2000]
        x = self.channel_reduction(x)
        
        # First upconv: [B,32,2000] → [B,32,4000]
        x = F.relu(self.upconv1(x))
        
        # Second upconv: [B,32,4000] → [B,wavelet_channels,8000]
        x = F.relu(self.upconv2(x))
        
        # Inverse wavelet transform: [B,wavelet_channels,8000] → [B,1,16000]
        x = self.inverse_wavelet(x)
        
        return x
    
    def process_overlapping_frames(self, frame):
        """
        Process overlapping frames with triangular windowing for smooth reconstruction
        
        Args:
            frame: Current reconstructed frame [B, 1, T]
            
        Returns:
            Processed frame with overlap-add if previous frame exists
        """
        # Apply triangular window to current frame
        windowed_frame = self.window(frame)
        
        # If no previous frame, store current and return as is
        if self.prev_frame is None:
            self.prev_frame = frame.detach()
            return windowed_frame
            
        # Calculate overlap size
        T = frame.size(2)
        overlap_size = int(T * self.overlap_factor)
        
        # Create output buffer
        output = torch.zeros_like(frame)
        
        # Copy previous frame's end (with overlap)
        output[:, :, :overlap_size] = self.prev_frame[:, :, -overlap_size:]
        
        # Add overlapping region with windowing for smooth transition
        output[:, :, :overlap_size] += windowed_frame[:, :, :overlap_size]
        
        # Copy remaining part of current frame
        output[:, :, overlap_size:] = windowed_frame[:, :, overlap_size:]
        
        # Store current frame for next iteration
        self.prev_frame = frame.detach()
        
        return output
    
    def encode_decode(self, z, wavelet_params=None, use_overlap=True):
        """
        Decode with optional overlap-add processing
        
        Args:
            z: Latent vector [B, 256]
            wavelet_params: Wavelet parameters from encoder
            use_overlap: Whether to use overlap-add processing
            
        Returns:
            Reconstructed signal and previous frame for loss calculation
        """
        # Set encoder parameters for the inverse wavelet transform
        if wavelet_params is not None:
            self.set_encoder_parameters(wavelet_params)
        
        # Decode the latent vector
        x_hat = self(z)
        
        # Store previous frame for loss calculation
        prev_frame = self.prev_frame
        
        # Apply overlap-add processing if requested
        if use_overlap:
            x_hat = self.process_overlapping_frames(x_hat)
        
        return x_hat, prev_frame
    
    def training_step(self, batch, batch_idx):
        x = batch
        
        # For standalone testing, we generate random latent vectors
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