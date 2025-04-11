import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import numpy as np
from models.wavelet import WaveletTransformOptimized
from models.level_processor import WaveletLevelProcessor
from models.conv_autoencoder import ConvWaveletAutoencoder
from models.autoencoder import WaveletAutoencoder
from utils.utils import calculate_snr, compute_spectrogram

class WaveletAudioAE(pl.LightningModule):
    """Optimized PyTorch Lightning module for wavelet-based audio autoencoder
    
    Focused on approximation coefficients to dramatically improve training efficiency
    while maintaining audio quality.
    """
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        
        # Learning rate with scheduling
        self.learning_rate = config.learning_rate
        
        # Create enhanced wavelet transform with more aggressive pruning for faster training
        self.wavelet_transform = WaveletTransformOptimized(
            wavelet=config.wavelet, 
            level=config.dwt_level,
            threshold_factor=3.0  # More aggressive pruning for training speedup
        )
        
        # Get input dimension from wavelet transform based on audio_length
        self.input_dim = self.wavelet_transform.get_output_dim(config.audio_length)
        
        print(f"Audio length: {config.audio_length} samples")
        print(f"Wavelet coefficients dimension: {self.input_dim}")
        
        # Get level dimensions for level-based processing
        level_dims = self.wavelet_transform.get_level_dims(config.audio_length)
        print(f"Level dimensions: {level_dims}")
        
        # Create simplified level processor (focuses on approximation coefficients only)
        self.level_processor = WaveletLevelProcessor(level_dims)
        
        # Determine autoencoder type (default to standard if not specified)
        autoencoder_type = getattr(config, 'autoencoder_type', 'standard')
        
        # Create the appropriate autoencoder model based on configuration
        if autoencoder_type == 'convolutional' or autoencoder_type == 'conv':
            print(f"Using ConvWaveletAutoencoder for model")
            base_channels = getattr(config, 'base_channels', 32)
            self.autoencoder = ConvWaveletAutoencoder(
                input_dim=self.input_dim,
                latent_dim=config.latent_dim,
                base_channels=base_channels
            )
        else:  # 'standard' or any other value defaults to standard autoencoder
            print(f"Using WaveletAutoencoder for model")
            # Get hidden dimensions from config or use default
            hidden_dims = getattr(config, 'hidden_dims', [128, 64, 32])
            self.autoencoder = WaveletAutoencoder(
                input_dim=self.input_dim,
                hidden_dims=hidden_dims,
                latent_dim=config.latent_dim
            )
        
        # Calculate compression ratio
        self.compression_ratio = config.audio_length / config.latent_dim
        print(f"Compression ratio: {self.compression_ratio:.2f}x")
        
        # Track best validation loss for LR scheduling
        self.best_val_loss = float('inf')
        
        # Initialize loss weight parameters
        self.recon_loss_weight = 1.0
        self.approx_loss_weight = 0.3
    
    def forward(self, x, training=True):
        # Apply wavelet transform with adaptive pruning
        wavelet_coeffs = self.wavelet_transform.forward(x, training=training)
        
        # Apply minimal level-based processing (focuses on approximation coefficients)
        processed_coeffs = self.level_processor(wavelet_coeffs)
        
        # Encode and decode with lightweight autoencoder
        wavelet_coeffs_recon, z = self.autoencoder(processed_coeffs)
        
        # Apply inverse transform
        x_recon = self.wavelet_transform.inverse(wavelet_coeffs_recon)
        
        # Ensure same length (shouldn't be needed with improved wavelet transform)
        if x.shape[1] != x_recon.shape[1]:
            min_len = min(x.shape[1], x_recon.shape[1])
            x = x[:, :min_len]
            x_recon = x_recon[:, :min_len]
        
        return x_recon, wavelet_coeffs, wavelet_coeffs_recon, z
    
    def training_step(self, batch, batch_idx):
        x = batch
        x_recon, wavelet_coeffs, wavelet_coeffs_recon, _ = self(x, training=True)
        
        # Simplified loss function that prioritizes approximation coefficients
        # Get level dimensions
        level_dims = self.wavelet_transform.get_level_dims()
        
        # Compute primary reconstruction loss (most important)
        recon_loss = F.mse_loss(x_recon, x)
        
        # Compute approximation coefficients loss (next most important)
        approx_start_idx = 0
        approx_end_idx = level_dims[0] if level_dims else 0
        
        # Extract approximation coefficients
        if approx_end_idx > 0:
            approx_coeffs = wavelet_coeffs[:, approx_start_idx:approx_end_idx]
            approx_coeffs_recon = wavelet_coeffs_recon[:, approx_start_idx:approx_end_idx]
            approx_loss = F.mse_loss(approx_coeffs_recon, approx_coeffs)
        else:
            approx_loss = torch.tensor(0.0, device=x.device)
        
        # Total loss with weighted components
        total_loss = self.recon_loss_weight * recon_loss + self.approx_loss_weight * approx_loss
        
        # Calculate SNR
        snr = calculate_snr(x, x_recon)
        avg_snr = torch.mean(snr)
        
        # Log metrics
        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_recon_loss', recon_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_approx_loss', approx_loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log('train_snr', avg_snr, on_step=True, on_epoch=True, prog_bar=True)
        
        # Reduce visualization frequency to speed up training
        if (self.global_step % 100 == 0 or batch_idx == 0) and self.global_step > 0:
            self._log_audio_and_visualizations(x, x_recon, wavelet_coeffs, wavelet_coeffs_recon)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        x = batch
        x_recon, wavelet_coeffs, wavelet_coeffs_recon, _ = self(x, training=False)
        
        # Simplified loss calculation (same as training)
        recon_loss = F.mse_loss(x_recon, x)
        
        # Compute approximation coefficients loss
        level_dims = self.wavelet_transform.get_level_dims()
        approx_start_idx = 0
        approx_end_idx = level_dims[0] if level_dims else 0
        
        if approx_end_idx > 0:
            approx_coeffs = wavelet_coeffs[:, approx_start_idx:approx_end_idx]
            approx_coeffs_recon = wavelet_coeffs_recon[:, approx_start_idx:approx_end_idx]
            approx_loss = F.mse_loss(approx_coeffs_recon, approx_coeffs)
        else:
            approx_loss = torch.tensor(0.0, device=x.device)
        
        # Total loss with weighted components - using same weights as training
        total_loss = self.recon_loss_weight * recon_loss + self.approx_loss_weight * approx_loss
        
        # Calculate SNR
        snr = calculate_snr(x, x_recon)
        avg_snr = torch.mean(snr)
        
        # Log metrics
        self.log('val_loss', total_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_recon_loss', recon_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_approx_loss', approx_loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log('val_snr', avg_snr, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        # Monitor validation loss plateau
        if not hasattr(self, 'val_loss_history'):
            self.val_loss_history = []
            
        # Store recent validation loss values (keep only last 5)
        self.val_loss_history.append(total_loss.item())
        if len(self.val_loss_history) > 5:
            self.val_loss_history.pop(0)
        
        return total_loss
        
    def test_step(self, batch, batch_idx):
        """Test step for model evaluation"""
        x = batch
        # Use non-training mode for testing (less pruning, better quality)
        x_recon, wavelet_coeffs, wavelet_coeffs_recon, _ = self(x, training=False)
        
        # Same loss calculation
        recon_loss = F.mse_loss(x_recon, x)
        
        # Combined loss
        total_loss = recon_loss
        
        # Calculate SNR
        snr = calculate_snr(x, x_recon)
        avg_snr = torch.mean(snr)
        
        # Log metrics
        self.log('test_loss', total_loss)
        self.log('test_recon_loss', recon_loss)
        self.log('test_snr', avg_snr)
        
        # Only visualize first batch
        if batch_idx == 0:
            self._log_audio_and_visualizations(x, x_recon, wavelet_coeffs, wavelet_coeffs_recon)
        
        return total_loss
    
    def _log_audio_and_visualizations(self, x, x_recon, wavelet_coeffs, wavelet_coeffs_recon, window_boundaries=None):
        """Simplified visualization logging to reduce training overhead"""
        # Get first sample from batch
        sample_idx = 0
        
        # Log audio samples
        self.logger.experiment.add_audio(
            'original_audio',
            x[sample_idx].unsqueeze(0),
            self.global_step,
            sample_rate=self.config.sample_rate
        )
        
        self.logger.experiment.add_audio(
            'reconstructed_audio',
            x_recon[sample_idx].unsqueeze(0),
            self.global_step,
            sample_rate=self.config.sample_rate
        )
        
        # Convert tensors to numpy for visualization
        x_np = x[sample_idx].cpu().numpy()
        x_recon_np = x_recon[sample_idx].cpu().numpy()
        
        # Simplified waveform visualization
        fig, ax = plt.subplots(figsize=(10, 4))
        
        # Create time axis in seconds
        t = np.arange(len(x_np)) / self.config.sample_rate
        
        # Plot original and reconstructed waveforms
        ax.plot(t, x_np, alpha=0.7, label='Original', color='blue')
        ax.plot(t, x_recon_np, alpha=0.7, label='Reconstructed', color='red')
        
        # Calculate and display SNR
        snr_value = calculate_snr(
            torch.tensor(x_np).unsqueeze(0), 
            torch.tensor(x_recon_np).unsqueeze(0)
        )[0].item()
        ax.text(0.02, 0.92, f'SNR: {snr_value:.2f} dB', transform=ax.transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))
        
        ax.set_title('Waveform Comparison')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add figure to tensorboard
        self.logger.experiment.add_figure(
            'waveform_comparison', 
            fig, 
            self.global_step
        )
        plt.close(fig)
    
    def configure_optimizers(self):
        """Configure optimizers with learning rate scheduling"""
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.learning_rate,
            weight_decay=1e-4
        )
        
        # Use reduced LR on plateau scheduler
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=10,  # Reduced from 20 to speed up adaptation
                min_lr=1e-6,
                verbose=True
            ),
            'monitor': 'val_loss',
            'interval': 'epoch',
            'frequency': 1
        }
        
        return [optimizer], [scheduler]