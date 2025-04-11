import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import numpy as np
from models.wavelet import WaveletTransformOptimized
from models.level_processor import WaveletLevelProcessor
#from models.conv_autoencoder import ConvWaveletAutoencoder
from models.autoencoder import WaveletAutoencoder
from utils.utils import calculate_snr, compute_spectrogram

# Try to import torchaudio for mel-spectrogram
try:
    import torchaudio
    TORCHAUDIO_AVAILABLE = True
except ImportError:
    TORCHAUDIO_AVAILABLE = False
    print("torchaudio not available, perceptual loss will be disabled")

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
        
        # Create lightweight autoencoder (prioritizes approximation coefficients)
        self.autoencoder = WaveletAutoencoder(
            input_dim=self.input_dim,
            latent_dim=config.latent_dim,
            base_channels=32  # Reduced from 64 to save memory
        )
        
        # Calculate compression ratio
        self.compression_ratio = config.audio_length / config.latent_dim
        print(f"Compression ratio: {self.compression_ratio:.2f}x")
        
        # Track best validation loss for LR scheduling
        self.best_val_loss = float('inf')
        
        # Initialize mel-spectrogram transform for perceptual loss
        if TORCHAUDIO_AVAILABLE:
            n_fft = min(1024, config.audio_length // 4)  # Reasonable FFT size
            hop_length = n_fft // 4  # 75% overlap
            n_mels = 80  # Standard number of mel bands
            
            self.mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=config.sample_rate,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels,
                power=2.0  # Power spectrogram (magnitude squared)
            )
            
            # Log-scaling for better alignment with human perception
            self.log_offset = 1e-6  # Small offset to avoid log(0)
            
            # Initialize loss weight parameters
            # Start with higher weight for mel loss to focus on perceptual quality
            # These will be adapted during training
            self.mel_loss_weight = 0.2
            self.recon_loss_weight = 1.0
            self.approx_loss_weight = 0.3
            
            print(f"Perceptual mel-spectrogram loss enabled with weight {self.mel_loss_weight}")
        else:
            self.mel_transform = None
            print("Perceptual mel-spectrogram loss disabled (torchaudio not available)")
    
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
    
    def compute_mel_loss(self, x, x_recon):
        """Compute enhanced perceptual loss based on mel-spectrogram differences
        
        This loss helps overcome plateaus by providing gradient information
        aligned with human auditory perception. The mel scale emphasizes
        frequencies humans are more sensitive to:
        
        mel(f) = 2595 * log10(1 + f/700)
        
        With time and frequency weighting to ensure proper optimization across
        the entire signal.
        
        Args:
            x: Original audio [batch_size, audio_length]
            x_recon: Reconstructed audio [batch_size, audio_length]
            
        Returns:
            Perceptual loss value (or zero if mel transform not available)
        """
        if not hasattr(self, 'mel_transform') or self.mel_transform is None:
            return torch.tensor(0.0, device=x.device)
            
        # Move mel transform to the same device as the input
        if hasattr(self.mel_transform, 'device') and self.mel_transform.device != x.device:
            self.mel_transform = self.mel_transform.to(x.device)
        
        # Compute mel spectrograms (add channel dimension for torchaudio)
        mel_orig = self.mel_transform(x.unsqueeze(1)).squeeze(1)
        mel_recon = self.mel_transform(x_recon.unsqueeze(1)).squeeze(1)
        
        # Apply log10 scaling for better perceptual alignment (log10 matches human perception better)
        # log(mel + offset) better correlates with human perception
        mel_orig_log = torch.log10(mel_orig + self.log_offset)
        mel_recon_log = torch.log10(mel_recon + self.log_offset)
        
        # Get dimensions for processing
        batch_size, n_mels, n_frames = mel_orig_log.shape
        
        # Define frequency band weights (prioritize perceptually important bands)
        # Focus more on middle frequencies (where human hearing is most sensitive)
        if not hasattr(self, 'band_weights') or self.band_weights.shape[1] != n_mels:
            # Create band weights centered around 1.0 for middle frequencies
            # Tapered at high and low extremes based on auditory perception
            center = n_mels // 2
            band_weights = torch.ones(n_mels, device=x.device)
            # Gradually reduce weight for extreme frequencies
            for i in range(n_mels):
                # Calculate distance from center (normalized to [0,1])
                dist = abs(i - center) / (n_mels / 2)
                # Apply curve: 0.5 + 0.5*cos(π*dist), giving 1.0 at center, 0.5 at edges
                band_weights[i] = 0.5 + 0.5 * np.cos(np.pi * min(1, dist))
            self.band_weights = band_weights.view(1, -1, 1)  # Shape for broadcasting
        
        # Define time frame weights to ensure even optimization
        # We want higher weight for later frames since they tend to be neglected
        if not hasattr(self, 'time_weights') or self.time_weights.shape[-1] != n_frames:
            # Create time weights that gradually increase for later frames
            # This counteracts the tendency to optimize early frames more
            time_weights = torch.linspace(0.8, 2.0, n_frames, device=x.device)
            self.time_weights = time_weights.view(1, 1, -1)  # Shape for broadcasting
        
        # Apply weights to spectrogram differences
        # Scale differences by band and time importance
        diff = torch.abs(mel_orig_log - mel_recon_log)
        weighted_diff = diff * self.band_weights * self.time_weights
        
        # Compute overall loss (L1 norm)
        mel_loss = weighted_diff.mean()
        
        # Calculate frame-wise loss for monitoring
        frame_loss = diff.mean(dim=1)  # Mean across frequency
        
        # Store for visualization (keep last batch only to save memory)
        if hasattr(self, 'global_step'):
            if not hasattr(self, 'frame_losses'):
                self.frame_losses = {}
            # Store latest frame losses (for most recent batch only)
            self.frame_losses[self.global_step] = frame_loss[0].detach().cpu()
        
        return mel_loss
    
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
        
        # Compute perceptual loss based on mel-spectrograms
        mel_loss = self.compute_mel_loss(x, x_recon)
        
        # Total loss with weighted components
        total_loss = self.recon_loss_weight * recon_loss + self.approx_loss_weight * approx_loss
        
        # Only add mel loss if it's available
        if hasattr(self, 'mel_transform') and self.mel_transform is not None:
            total_loss += self.mel_loss_weight * mel_loss
            
            # Dynamic weight adjustment based on training progress
            # As training progresses, gradually increase mel loss weight
            # This helps overcome plateaus by changing the optimization landscape
            if self.global_step > 0 and self.global_step % 100 == 0:
                # Increase mel weight up to a maximum of 0.5
                current_mel_weight = self.mel_loss_weight
                # Sigmoid-based increase that starts slower, then accelerates, then slows again
                step_factor = min(self.global_step / 5000, 5.0)  # Limit maximum increase
                self.mel_loss_weight = min(0.5, current_mel_weight * (1.0 + 0.05 * step_factor))
                
                # Log weight adjustments
                self.log('mel_loss_weight', self.mel_loss_weight, on_step=True, prog_bar=False)
        
        # Calculate SNR
        snr = calculate_snr(x, x_recon)
        avg_snr = torch.mean(snr)
        
        # Log metrics
        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_recon_loss', recon_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_approx_loss', approx_loss, on_step=True, on_epoch=True, prog_bar=False)
        
        # Log mel loss if available
        if hasattr(self, 'mel_transform') and self.mel_transform is not None:
            self.log('train_mel_loss', mel_loss, on_step=True, on_epoch=True, prog_bar=True)
            
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
        
        # Compute perceptual loss based on mel-spectrograms
        mel_loss = self.compute_mel_loss(x, x_recon)
        
        # Total loss with weighted components - using same weights as training
        total_loss = self.recon_loss_weight * recon_loss + self.approx_loss_weight * approx_loss
        
        # Only add mel loss if it's available
        if hasattr(self, 'mel_transform') and self.mel_transform is not None:
            total_loss += self.mel_loss_weight * mel_loss
        
        # Calculate SNR
        snr = calculate_snr(x, x_recon)
        avg_snr = torch.mean(snr)
        
        # Log metrics
        self.log('val_loss', total_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_recon_loss', recon_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_approx_loss', approx_loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        
        # Log mel loss if available
        if hasattr(self, 'mel_transform') and self.mel_transform is not None:
            self.log('val_mel_loss', mel_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            
        self.log('val_snr', avg_snr, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        # Monitor validation loss plateau and potentially trigger loss weight adjustments
        if not hasattr(self, 'val_loss_history'):
            self.val_loss_history = []
            
        # Store recent validation loss values (keep only last 5)
        self.val_loss_history.append(total_loss.item())
        if len(self.val_loss_history) > 5:
            self.val_loss_history.pop(0)
            
        # Check for plateau - if last 3 validation losses are within 1% of each other
        if len(self.val_loss_history) >= 3:
            recent_losses = self.val_loss_history[-3:]
            mean_loss = sum(recent_losses) / 3
            max_diff = max([abs(loss - mean_loss) / mean_loss for loss in recent_losses])
            
            # If we're in a plateau (very small changes)
            if max_diff < 0.01 and hasattr(self, 'mel_loss_weight'):
                # Increase mel loss weight more aggressively to break plateau
                self.mel_loss_weight = min(0.8, self.mel_loss_weight * 1.2)
                self.log('mel_loss_weight', self.mel_loss_weight, on_step=False, on_epoch=True)
        
        return total_loss
        
    def test_step(self, batch, batch_idx):
        """Test step for model evaluation"""
        x = batch
        # Use non-training mode for testing (less pruning, better quality)
        x_recon, wavelet_coeffs, wavelet_coeffs_recon, _ = self(x, training=False)
        
        # Same loss calculation
        recon_loss = F.mse_loss(x_recon, x)
        
        # Compute perceptual loss
        mel_loss = self.compute_mel_loss(x, x_recon)
        
        # Combined loss
        total_loss = recon_loss
        
        # Only add mel loss if it's available
        if hasattr(self, 'mel_transform') and self.mel_transform is not None:
            total_loss += 0.05 * mel_loss
        
        # Calculate SNR
        snr = calculate_snr(x, x_recon)
        avg_snr = torch.mean(snr)
        
        # Log metrics
        self.log('test_loss', total_loss)
        self.log('test_recon_loss', recon_loss)
        
        # Log mel loss if available
        if hasattr(self, 'mel_transform') and self.mel_transform is not None:
            self.log('test_mel_loss', mel_loss)
            
        self.log('test_snr', avg_snr)
        
        # Only visualize first batch
        if batch_idx == 0:
            self._log_audio_and_visualizations(x, x_recon, wavelet_coeffs, wavelet_coeffs_recon)
        
        return total_loss
    
    def _visualize_frame_loss(self, frame_loss):
        """Create visualization of loss across time frames to help diagnose issues"""
        if not hasattr(self, 'logger') or self.logger is None:
            return
            
        fig, ax = plt.subplots(figsize=(10, 4))
        
        # Plot frame-wise loss
        time_frames = np.arange(len(frame_loss))
        ax.plot(time_frames, frame_loss.numpy())
        
        # Add time weights if available for comparison
        if hasattr(self, 'time_weights'):
            # Normalize time weights to be visible on same scale
            time_weights = self.time_weights.squeeze().cpu().numpy()
            max_loss = frame_loss.max().item()
            normalized_weights = time_weights * (max_loss / time_weights.max())
            
            # Plot on secondary y-axis
            ax2 = ax.twinx()
            ax2.plot(time_frames, normalized_weights, 'r--', alpha=0.7, label='Time Weights')
            ax2.set_ylabel('Weight Factor (Normalized)', color='r')
            ax2.tick_params(axis='y', labelcolor='r')
            
        ax.set_title('Perceptual Loss by Time Frame')
        ax.set_xlabel('Time Frame')
        ax.set_ylabel('Loss Magnitude')
        ax.grid(True, alpha=0.3)
        
        # Add to tensorboard
        self.logger.experiment.add_figure(
            'frame_loss', 
            fig, 
            self.global_step
        )
        plt.close(fig)
        
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
        
        # Add visualization for frame-wise loss
        if hasattr(self, 'frame_losses') and self.global_step in self.frame_losses:
            self._visualize_frame_loss(self.frame_losses[self.global_step])
        
        # Add mel-spectrogram visualization
        if hasattr(self, 'mel_transform') and self.mel_transform is not None:
            try:
                # Compute mel spectrograms for visualization
                with torch.no_grad():
                    # Make sure the input is on the correct device
                    x_device = x[sample_idx].to(self.device)
                    x_recon_device = x_recon[sample_idx].to(self.device)
                    
                    # Ensure mel transform is on the same device
                    if hasattr(self.mel_transform, 'device') and self.mel_transform.device != self.device:
                        self.mel_transform = self.mel_transform.to(self.device)
                    
                    # Compute spectrograms
                    mel_orig = self.mel_transform(x_device.unsqueeze(0).unsqueeze(0))
                    mel_recon = self.mel_transform(x_recon_device.unsqueeze(0).unsqueeze(0))
                
                # Convert to dB scale for visualization
                mel_orig_db = 20 * torch.log10(mel_orig + self.log_offset)
                mel_recon_db = 20 * torch.log10(mel_recon + self.log_offset)
                
                # Compute absolute difference for visualization
                mel_diff = torch.abs(mel_orig_db - mel_recon_db)[0, 0].cpu().numpy()
                
                # Visualize spectrograms and difference
                fig, axs = plt.subplots(3, 1, figsize=(10, 12))
                
                # Original spectrogram
                im0 = axs[0].imshow(
                    mel_orig_db[0, 0].cpu().numpy(),
                    aspect='auto',
                    origin='lower',
                    cmap='viridis'
                )
                axs[0].set_title('Original Mel-Spectrogram')
                axs[0].set_ylabel('Mel Bin')
                
                # Reconstructed spectrogram
                im1 = axs[1].imshow(
                    mel_recon_db[0, 0].cpu().numpy(),
                    aspect='auto',
                    origin='lower',
                    cmap='viridis'
                )
                axs[1].set_title('Reconstructed Mel-Spectrogram')
                axs[1].set_ylabel('Mel Bin')
                
                # Difference spectrogram (helps identify problem areas)
                im2 = axs[2].imshow(
                    mel_diff,
                    aspect='auto',
                    origin='lower',
                    cmap='inferno'  # Different colormap for difference
                )
                axs[2].set_title('Absolute Difference (dB)')
                axs[2].set_ylabel('Mel Bin')
                axs[2].set_xlabel('Time Frame')
                
                # Add colorbars
                fig.colorbar(im0, ax=axs[0], label='dB')
                fig.colorbar(im1, ax=axs[1], label='dB')
                fig.colorbar(im2, ax=axs[2], label='|Difference| (dB)')
                
                fig.tight_layout()
                
                # Add figure to tensorboard
                self.logger.experiment.add_figure(
                    'mel_spectrogram_comparison', 
                    fig, 
                    self.global_step
                )
                plt.close(fig)
                
                # Add band weights visualization if available
                if hasattr(self, 'band_weights'):
                    fig, ax = plt.subplots(figsize=(8, 4))
                    band_weights = self.band_weights.squeeze().cpu().numpy()
                    ax.plot(np.arange(len(band_weights)), band_weights)
                    ax.set_title('Frequency Band Weights')
                    ax.set_xlabel('Mel Bin')
                    ax.set_ylabel('Weight Factor')
                    ax.grid(True, alpha=0.3)
                    
                    # Add to tensorboard (do this once per few steps to save space)
                    if self.global_step % 100 == 0:
                        self.logger.experiment.add_figure(
                            'frequency_band_weights', 
                            fig, 
                            self.global_step
                        )
                    plt.close(fig)
                
            except Exception as e:
                print(f"Error in spectrogram visualization: {e}")
    
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