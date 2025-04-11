import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import numpy as np
from models.wavelet import WaveletTransformOptimized
from models.level_processor import WaveletLevelProcessor
from models.conv_autoencoder import ConvWaveletAutoencoder
from utils.utils import calculate_snr, compute_spectrogram

class WaveletAudioAE(pl.LightningModule):
    """Optimized PyTorch Lightning module for wavelet-based audio autoencoder
    
    Combines three efficiency improvements:
    1. Coefficient pruning via adaptive thresholding
    2. Level-based wavelet coefficient processing
    3. Convolutional encoding/decoding architecture
    
    This reduces model size while maintaining or improving quality.
    """
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        
        # Learning rate with scheduling
        self.learning_rate = config.learning_rate
        
        # Create enhanced wavelet transform with pruning
        self.wavelet_transform = WaveletTransformOptimized(
            wavelet=config.wavelet, 
            level=config.dwt_level,
            threshold_factor=2.0  # Control pruning strength
        )
        
        # Get input dimension from wavelet transform based on audio_length
        self.input_dim = self.wavelet_transform.get_output_dim(config.audio_length)
        
        print(f"Audio length: {config.audio_length} samples")
        print(f"Wavelet coefficients dimension: {self.input_dim}")
        
        # Get level dimensions for level-based processing
        level_dims = self.wavelet_transform.get_level_dims(config.audio_length)
        print(f"Level dimensions: {level_dims}")
        
        # Create level processor
        self.level_processor = WaveletLevelProcessor(level_dims)
        
        # Create convolutional autoencoder
        # Compression is now distributed across model stages
        self.autoencoder = ConvWaveletAutoencoder(
            input_dim=self.input_dim,
            latent_dim=config.latent_dim,
            base_channels=64  # Adjust based on model size needs
        )
        
        # Calculate compression ratio
        self.compression_ratio = config.audio_length / config.latent_dim
        print(f"Compression ratio: {self.compression_ratio:.2f}x")
        
        # Track best validation loss for LR scheduling
        self.best_val_loss = float('inf')
    
    def forward(self, x, training=True):
        # Apply wavelet transform with adaptive pruning
        wavelet_coeffs = self.wavelet_transform.forward(x, training=training)
        
        # Apply level-based processing
        processed_coeffs = self.level_processor(wavelet_coeffs)
        
        # Encode and decode
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
        
        # Enhanced loss function with frequency-weighted MSE
        # Low-frequency components more important for perception
        
        # Compute losses
        wavelet_loss = F.mse_loss(wavelet_coeffs_recon, wavelet_coeffs)
        recon_loss = F.mse_loss(x_recon, x)
        
        # Perceptual scaling of wavelet loss
        # Give more weight to lower frequency components (approximation coefficients)
        level_dims = self.wavelet_transform.get_level_dims()
        start_idx = 0
        total_loss = recon_loss  # Start with waveform reconstruction
        
        # Compute level-specific losses with decreasing weights
        for i, dim in enumerate(level_dims):
            end_idx = start_idx + dim
            level_coeffs = wavelet_coeffs[:, start_idx:end_idx]
            level_coeffs_recon = wavelet_coeffs_recon[:, start_idx:end_idx]
            
            # Decreasing weights for higher-frequency details
            if i == 0:
                # Approximation coefficients - highest weight
                weight = 0.2
            else:
                # Detail coefficients - decreasing weights
                weight = 0.1 / (2 ** (i-1))
            
            level_loss = F.mse_loss(level_coeffs_recon, level_coeffs)
            total_loss = total_loss + weight * level_loss
            
            # Track level-specific metrics
            self.log(f'train_level_{i}_loss', level_loss, on_step=True, on_epoch=True, prog_bar=False)
            
            start_idx = end_idx
        
        # Calculate SNR
        snr = calculate_snr(x, x_recon)
        avg_snr = torch.mean(snr)
        
        # Log metrics
        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_recon_loss', recon_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_wavelet_loss', wavelet_loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log('train_snr', avg_snr, on_step=True, on_epoch=True, prog_bar=True)
        
        # Periodically log audio and visualizations (less frequently to save resources)
        if (self.global_step % 50 == 0 or batch_idx == 0) and self.global_step > 0:
            self._log_audio_and_visualizations(x, x_recon, wavelet_coeffs, wavelet_coeffs_recon)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        x = batch
        x_recon, wavelet_coeffs, wavelet_coeffs_recon, _ = self(x, training=False)
        
        # Compute losses (same as training step but without gradient)
        wavelet_loss = F.mse_loss(wavelet_coeffs_recon, wavelet_coeffs)
        recon_loss = F.mse_loss(x_recon, x)
        
        # Use same level-based weighting as in training
        level_dims = self.wavelet_transform.get_level_dims()
        start_idx = 0
        total_loss = recon_loss
        
        for i, dim in enumerate(level_dims):
            end_idx = start_idx + dim
            level_coeffs = wavelet_coeffs[:, start_idx:end_idx]
            level_coeffs_recon = wavelet_coeffs_recon[:, start_idx:end_idx]
            
            if i == 0:
                weight = 0.2
            else:
                weight = 0.1 / (2 ** (i-1))
            
            level_loss = F.mse_loss(level_coeffs_recon, level_coeffs)
            total_loss = total_loss + weight * level_loss
            
            # Log metrics - ensure they're logged at epoch level for the scheduler
            self.log(f'val_level_{i}_loss', level_loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
            
            start_idx = end_idx
        
        # Calculate SNR
        snr = calculate_snr(x, x_recon)
        avg_snr = torch.mean(snr)
        
        # Log metrics - ensure they're logged at epoch level for the scheduler
        self.log('val_loss', total_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_recon_loss', recon_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_wavelet_loss', wavelet_loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log('val_snr', avg_snr, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        # Update best validation loss for LR scheduling
        if total_loss < self.best_val_loss:
            self.best_val_loss = total_loss
        
        return total_loss
        
    def test_step(self, batch, batch_idx):
        """Test step for model evaluation"""
        x = batch
        # Use non-training mode for testing (less pruning, better quality)
        x_recon, wavelet_coeffs, wavelet_coeffs_recon, _ = self(x, training=False)
        
        # Compute losses
        wavelet_loss = F.mse_loss(wavelet_coeffs_recon, wavelet_coeffs)
        recon_loss = F.mse_loss(x_recon, x)
        
        # Use same level-based weighting as in training/validation
        level_dims = self.wavelet_transform.get_level_dims()
        start_idx = 0
        total_loss = recon_loss
        
        for i, dim in enumerate(level_dims):
            end_idx = start_idx + dim
            level_coeffs = wavelet_coeffs[:, start_idx:end_idx]
            level_coeffs_recon = wavelet_coeffs_recon[:, start_idx:end_idx]
            
            if i == 0:
                weight = 0.2
            else:
                weight = 0.1 / (2 ** (i-1))
            
            level_loss = F.mse_loss(level_coeffs_recon, level_coeffs)
            total_loss = total_loss + weight * level_loss
            
            self.log(f'test_level_{i}_loss', level_loss)
            
            start_idx = end_idx
        
        # Calculate SNR
        snr = calculate_snr(x, x_recon)
        avg_snr = torch.mean(snr)
        
        # Log metrics
        self.log('test_loss', total_loss)
        self.log('test_recon_loss', recon_loss)
        self.log('test_wavelet_loss', wavelet_loss)
        self.log('test_snr', avg_snr)
        
        # If it's the first batch, log some audio samples for qualitative evaluation
        if batch_idx == 0:
            self._log_audio_and_visualizations(x, x_recon, wavelet_coeffs, wavelet_coeffs_recon)
        
        return total_loss
    
    def _log_audio_and_visualizations(self, x, x_recon, wavelet_coeffs, wavelet_coeffs_recon, window_boundaries=None):
        """Log audio samples and enhanced visualizations to TensorBoard
        
        Includes:
        - Original and reconstructed audio
        - Overlapping waveform comparison
        - Adaptive spectrogram scaling
        - Wavelet coefficient visualization
        - Window boundary markers (if available)
        """
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
        
        # 1. Enhanced Waveform Comparison
        fig, ax = plt.subplots(figsize=(12, 4))
        
        # Create time axis in seconds
        t = np.arange(len(x_np)) / self.config.sample_rate
        
        # Plot original waveform
        ax.plot(t, x_np, alpha=0.7, label='Original', color='blue')
        
        # Plot reconstructed waveform
        ax.plot(t, x_recon_np, alpha=0.7, label='Reconstructed', color='red')
        
        # Plot error region
        error = x_np - x_recon_np
        ax.fill_between(t, error, -error, color='grey', alpha=0.3, label='Error')
        
        # Plot window boundaries if available
        if window_boundaries is not None:
            for boundary in window_boundaries:
                boundary_time = boundary / self.config.sample_rate
                ax.axvline(x=boundary_time, color='green', linestyle='--', alpha=0.5)
        
        ax.set_title('Waveform Comparison')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Set reasonable y-limits
        max_amp = max(np.max(np.abs(x_np)), np.max(np.abs(x_recon_np)))
        ax.set_ylim(-max_amp * 1.1, max_amp * 1.1)
        
        # Add SNR value as text
        snr_value = calculate_snr(
            torch.tensor(x_np).unsqueeze(0), 
            torch.tensor(x_recon_np).unsqueeze(0)
        )[0].item()
        ax.text(0.02, 0.92, f'SNR: {snr_value:.2f} dB', transform=ax.transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))
        
        # Add zoom-in panel for detail
        zoom_start = len(x_np) // 3
        zoom_width = min(1000, len(x_np) // 10)
        zoom_end = zoom_start + zoom_width
        
        # Create inset axis for zoom
        axins = ax.inset_axes([0.05, 0.05, 0.3, 0.3])
        axins.plot(t[zoom_start:zoom_end], x_np[zoom_start:zoom_end], 'b-')
        axins.plot(t[zoom_start:zoom_end], x_recon_np[zoom_start:zoom_end], 'r-')
        axins.set_title('Zoom', fontsize=8)
        axins.grid(True, alpha=0.3)
        
        # Mark zoom region on main plot
        ax.indicate_inset_zoom(axins, edgecolor="black")
        
        # Add figure to tensorboard
        self.logger.experiment.add_figure(
            'enhanced_waveform_comparison', 
            fig, 
            self.global_step
        )
        plt.close(fig)
        
        # 2. Wavelet Coefficient Visualization with Level Boundaries
        coeffs = wavelet_coeffs[sample_idx].cpu().numpy()
        recon_coeffs = wavelet_coeffs_recon[sample_idx].cpu().detach().numpy()
        
        # Get level dimensions for visualization
        level_dims = self.wavelet_transform.get_level_dims()
        
        # Create comparison figure
        fig, axs = plt.subplots(3, 1, figsize=(12, 8))
        
        # Determine reasonable viewing window for coefficients
        # Show more coefficients for higher global steps as model improves
        view_size = min(2000, len(coeffs))
        
        # Original coefficients
        axs[0].plot(coeffs[:view_size], 'b-', label='Original')
        axs[0].set_title('Original Wavelet Coefficients')
        axs[0].grid(True, alpha=0.3)
        
        # Add level boundary markers
        start_idx = 0
        for i, dim in enumerate(level_dims):
            if start_idx < view_size:
                end_idx = min(start_idx + dim, view_size)
                if i == 0:
                    # Approximation coefficients
                    axs[0].axvspan(start_idx, end_idx, color='green', alpha=0.1, label='Approx' if i == 0 else None)
                else:
                    # Detail coefficients - alternate colors
                    color = 'blue' if i % 2 == 0 else 'red'
                    alpha = 0.1 - 0.01 * i  # Decrease opacity for higher levels
                    axs[0].axvspan(start_idx, end_idx, color=color, alpha=max(0.05, alpha), 
                                 label=f'Detail L{i}' if start_idx < view_size else None)
                
                # Add level name
                if end_idx - start_idx > 50:  # Only add text if enough space
                    text_pos = start_idx + (end_idx - start_idx) // 2
                    if text_pos < view_size:
                        y_pos = axs[0].get_ylim()[1] * 0.9
                        axs[0].text(text_pos, y_pos, f'L{i}', fontsize=8, 
                                  ha='center', va='top', alpha=0.7)
                
                start_idx = end_idx
        
        # Reconstructed coefficients
        axs[1].plot(recon_coeffs[:view_size], 'r-', label='Reconstructed')
        axs[1].set_title('Reconstructed Wavelet Coefficients')
        axs[1].grid(True, alpha=0.3)
        
        # Add the same level boundary markers
        start_idx = 0
        for i, dim in enumerate(level_dims):
            if start_idx < view_size:
                end_idx = min(start_idx + dim, view_size)
                if i == 0:
                    # Approximation coefficients
                    axs[1].axvspan(start_idx, end_idx, color='green', alpha=0.1)
                else:
                    # Detail coefficients - alternate colors
                    color = 'blue' if i % 2 == 0 else 'red'
                    alpha = 0.1 - 0.01 * i  # Decrease opacity for higher levels
                    axs[1].axvspan(start_idx, end_idx, color=color, alpha=max(0.05, alpha))
                
                # Add level name
                if end_idx - start_idx > 50:  # Only add text if enough space
                    text_pos = start_idx + (end_idx - start_idx) // 2
                    if text_pos < view_size:
                        y_pos = axs[1].get_ylim()[1] * 0.9
                        axs[1].text(text_pos, y_pos, f'L{i}', fontsize=8, 
                                  ha='center', va='top', alpha=0.7)
                
                start_idx = end_idx
        
        # Coefficient difference
        diff = coeffs[:view_size] - recon_coeffs[:view_size]
        axs[2].plot(diff, 'g-', label='Difference')
        axs[2].set_title('Coefficient Difference')
        axs[2].set_xlabel('Coefficient Index')
        axs[2].grid(True, alpha=0.3)
        
        # Add level boundary markers for difference plot too
        start_idx = 0
        for i, dim in enumerate(level_dims):
            if start_idx < view_size:
                end_idx = min(start_idx + dim, view_size)
                if i == 0:
                    # Approximation coefficients
                    axs[2].axvspan(start_idx, end_idx, color='green', alpha=0.1)
                else:
                    # Detail coefficients - alternate colors
                    color = 'blue' if i % 2 == 0 else 'red'
                    alpha = 0.1 - 0.01 * i  # Decrease opacity for higher levels
                    axs[2].axvspan(start_idx, end_idx, color=color, alpha=max(0.05, alpha))
                
                start_idx = end_idx
        
        # Set consistent y-limits for comparison
        coeff_max = max(np.max(np.abs(coeffs[:view_size])), np.max(np.abs(recon_coeffs[:view_size])))
        axs[0].set_ylim(-coeff_max * 1.1, coeff_max * 1.1)
        axs[1].set_ylim(-coeff_max * 1.1, coeff_max * 1.1)
        
        # Set y-limit for difference plot
        diff_max = np.max(np.abs(diff))
        axs[2].set_ylim(-diff_max * 1.1, diff_max * 1.1)
        
        # Add coefficient statistics
        mse = np.mean((diff) ** 2)
        axs[2].text(0.02, 0.92, f'MSE: {mse:.6f}', transform=axs[2].transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))
        
        # Add figure to tensorboard
        self.logger.experiment.add_figure(
            'enhanced_wavelet_comparison', 
            fig, 
            self.global_step
        )
        plt.close(fig)
        
        # 3. Enhanced Spectrogram Comparison with Adaptive Scaling
        # Compute spectrograms using FFT
        n_fft = min(1024, self.config.audio_length // 4)  # Adjust based on audio length
        hop_length = n_fft // 4
        
        # Compute spectrograms
        spec_orig = compute_spectrogram(x_np, n_fft, hop_length)
        spec_recon = compute_spectrogram(x_recon_np, n_fft, hop_length)
        
        # Convert to dB scale with proper normalization
        eps = 1e-10
        spec_orig_db = 20 * np.log10(spec_orig + eps)
        spec_recon_db = 20 * np.log10(spec_recon + eps)
        
        # Determine adaptive scaling based on data
        # Use percentiles to avoid outliers affecting scale
        all_values = np.concatenate([spec_orig_db.flatten(), spec_recon_db.flatten()])
        vmin = np.percentile(all_values, 1)  # 1st percentile for minimum
        vmax = np.percentile(all_values, 99)  # 99th percentile for maximum
        
        # Ensure a minimum dynamic range
        if vmax - vmin < 60:
            vmean = (vmax + vmin) / 2
            vmin = vmean - 30
            vmax = vmean + 30
        
        # Create frequency axis (Hz)
        freqs = np.linspace(0, self.config.sample_rate/2, spec_orig.shape[0])
        
        # Create time axis (seconds)
        spec_times = np.linspace(0, len(x_np)/self.config.sample_rate, spec_orig.shape[1])
        
        # Create spectrogram figure
        fig, axs = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot original spectrogram
        im0 = axs[0].imshow(spec_orig_db, aspect='auto', origin='lower', 
                        cmap='viridis', vmin=vmin, vmax=vmax)
        axs[0].set_title('Original Spectrogram')
        axs[0].set_ylabel('Frequency (Hz)')
        
        # Add frequency axis labels (show subset for clarity)
        n_freq_labels = 6
        freq_indices = np.linspace(0, len(freqs)-1, n_freq_labels, dtype=int)
        axs[0].set_yticks(freq_indices)
        axs[0].set_yticklabels([f'{freqs[i]:.0f}' for i in freq_indices])
        
        # Plot reconstructed spectrogram
        im1 = axs[1].imshow(spec_recon_db, aspect='auto', origin='lower', 
                        cmap='viridis', vmin=vmin, vmax=vmax)
        axs[1].set_title('Reconstructed Spectrogram')
        axs[1].set_ylabel('Frequency (Hz)')
        axs[1].set_xlabel('Time (s)')
        
        # Add frequency axis labels
        axs[1].set_yticks(freq_indices)
        axs[1].set_yticklabels([f'{freqs[i]:.0f}' for i in freq_indices])
        
        # Add time axis labels (to bottom plot only)
        n_time_labels = 6
        time_indices = np.linspace(0, len(spec_times)-1, n_time_labels, dtype=int)
        axs[1].set_xticks(time_indices)
        axs[1].set_xticklabels([f'{spec_times[i]:.2f}' for i in time_indices])
        
        # Add colorbar
        fig.colorbar(im0, ax=axs, orientation='vertical', label='Power (dB)')
        
        # Add window boundary markers if available
        if window_boundaries is not None:
            for boundary in window_boundaries:
                # Convert sample positions to spectrogram frame indices
                frame_idx = int(boundary / hop_length)
                if frame_idx < spec_orig.shape[1]:
                    axs[0].axvline(x=frame_idx, color='white', linestyle='--', alpha=0.5)
                    axs[1].axvline(x=frame_idx, color='white', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        
        # Add figure to tensorboard
        self.logger.experiment.add_figure(
            'enhanced_spectrogram_comparison', 
            fig, 
            self.global_step
        )
        plt.close(fig)
    
    def configure_optimizers(self):
        """Configure optimizers with learning rate scheduling"""
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.learning_rate,
            weight_decay=1e-4  # L2 regularization for better generalization
        )
        
        # Use reduced LR on plateau scheduler
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,  # Reduce LR by half when plateau
                patience=20,
                min_lr=1e-6,
                verbose=True
            ),
            'monitor': 'val_loss',  # Monitor validation loss
            'interval': 'epoch',
            'frequency': 1
        }
        
        return [optimizer], [scheduler]