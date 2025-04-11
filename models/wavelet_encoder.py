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
        
        # Extract wavelet MLP layers from config
        wavelet_mlp_layers = config['model'].get('wavelet_mlp_layers', [1, 32, 64, 32])
        
        # Ensure first layer is size 1 (time position)
        wavelet_mlp_layers[0] = 1
        
        # Ensure last hidden layer can output to channels
        if len(wavelet_mlp_layers) < 2:
            wavelet_mlp_layers.append(32)
        
        # Build MLP layers dynamically from config
        layers = []
        for i in range(len(wavelet_mlp_layers) - 1):
            in_dim = wavelet_mlp_layers[i]
            out_dim = wavelet_mlp_layers[i + 1]
            layers.append(nn.Linear(in_dim, out_dim))
            
            # Add activation except for last layer
            if i < len(wavelet_mlp_layers) - 2:
                layers.append(nn.LeakyReLU())
        
        # Add final output layer to match channels
        layers.append(nn.Linear(wavelet_mlp_layers[-1], channels))
        layers.append(nn.Sigmoid())
        
        self.mlp = nn.Sequential(*layers)
        
        # Calculate parameter count for verification
        param_count = 0
        for i in range(len(wavelet_mlp_layers) - 1):
            in_dim = wavelet_mlp_layers[i]
            out_dim = wavelet_mlp_layers[i + 1]
            # Weight + bias parameters
            param_count += (in_dim * out_dim + out_dim)
        
        # Final layer parameters
        param_count += (wavelet_mlp_layers[-1] * channels + channels)
        
        # Set as attribute for debugging
        self.param_count = param_count
        
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
        
        # Save the most recent adaptive filters for decoder use
        self.register_buffer('last_adaptive_filters', torch.zeros(channels, 1, self.kernel_size))
        self.register_buffer('last_feature_modulation_weights', torch.zeros(channels, channels, 1))
    
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
                
                # Apply parameter symmetry constraints for wavelet admissibility
                if self.kernel_size % 2 == 1:  # For odd kernel size
                    center = self.kernel_size // 2
                    # Ensure wavelet has zero mean (for admissibility)
                    wavelet = wavelet - torch.mean(wavelet)
                    
                    # Apply symmetry/anti-symmetry based on wavelet type
                    if wavelet_type == "morlet":
                        # Anti-symmetric part for Morlet (imaginary component)
                        anti_sym_part = (wavelet[:center] - wavelet[center+1:].flip(0)) / 2
                        wavelet[:center] = anti_sym_part
                        wavelet[center+1:] = -anti_sym_part.flip(0)
                    else:
                        # Ensure symmetry for Mexican hat and others
                        sym_part = (wavelet[:center] + wavelet[center+1:].flip(0)) / 2
                        wavelet[:center] = sym_part
                        wavelet[center+1:] = sym_part.flip(0)
                
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
        
        # Store current adaptive filters for use in the decoder
        self.last_adaptive_filters.copy_(adaptive_filters.detach())
        
        # Apply adaptive wavelet transform using F.conv1d
        # This implements W_ψx = |a|^(-1/2) ∫x(t)ψ_θ((t-b)/a)dt
        wavelet_output = F.conv1d(
            x, adaptive_filters, padding=self.kernel_size//2
        )
        
        # Apply feature modulation for additional adaptivity
        feature_weights = self.feature_modulation.weight.clone()
        self.last_feature_modulation_weights.copy_(feature_weights.detach())
        wavelet_output = self.feature_modulation(wavelet_output)
        
        # Downsample to target length
        return wavelet_output[:, :, ::2]  # [B, C, T/2]
    
    def get_wavelet_parameters(self):
        """Get the current wavelet parameters for use in the decoder"""
        return {
            'adaptive_filters': self.last_adaptive_filters.clone(),
            'feature_modulation_weights': self.last_feature_modulation_weights.clone()
        }

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
          wavelet_params: Dictionary of wavelet parameters for the decoder
        """
        # Wavelet transform: [B, 1, 16000] → [B, 16, 8000]
        x = self.wavelet_transform(x)
        
        # First feature extraction conv: [B, 16, 8000] → [B, 32, 4000]
        x = F.relu(self.conv1(x))
        
        # Second feature extraction conv: [B, 32, 4000] → [B, 32, 2000]
        x = F.relu(self.conv2(x))
        
        # Bottleneck: Conv1D → [B, 256, 2000]
        x_bottleneck = F.relu(self.bottleneck(x))
        
        # Store the shape for potential skip connections or dimensionality reference
        batch_size, channels, time_dim = x_bottleneck.shape
        
        # Global avg pooling → [B, 256, 1]
        x = self.global_avgpool(x_bottleneck)
        
        # Flatten → [B, 256]
        x = x.flatten(1)
        
        # VAE reparameterization
        z_mean = self.fc_mean(x)
        z_logvar = self.fc_logvar(x)
        
        # Sample from the distribution
        std = torch.exp(0.5 * z_logvar)
        eps = torch.randn_like(std)
        z = z_mean + eps * std
        
        # Get wavelet parameters for the decoder
        wavelet_params = self.wavelet_transform.get_wavelet_parameters()
        
        return z, z_mean, z_logvar, wavelet_params
    
    def training_step(self, batch, batch_idx):
        x = batch
        
        # Encode
        z, z_mean, z_logvar, _ = self(x)
        
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
        z, z_mean, z_logvar, _ = self(x)
        
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