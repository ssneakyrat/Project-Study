import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import numpy as np
from models.wavelet import WaveletTransform
from models.autoencoder import WaveletAutoencoder
from utils.utils import calculate_snr, compute_spectrogram

class WaveletAudioAE(pl.LightningModule):
    """PyTorch Lightning module for wavelet-based audio autoencoder
    
    Key metrics tracked:
    - MSE: (1/N)·Σ(x_i - x̂_i)² - direct signal comparison
    - SNR: 10·log₁₀(Σx_i²/Σ(x_i - x̂_i)²) dB - quality measure
    - Rate-distortion bound: R(D) ≥ (1/2)·log₂(σ²/D) - theoretical limit
    """
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        
        # Create wavelet transform
        self.wavelet_transform = WaveletTransform(
            wavelet=config.wavelet, 
            level=config.dwt_level
        )
        
        # Get input dimension from a dummy forward pass
        dummy_audio = torch.zeros(1, config.audio_length)
        wavelet_coeffs = self.wavelet_transform.forward(dummy_audio)
        self.input_dim = wavelet_coeffs.shape[1]
        
        print(f"Wavelet coefficients dimension: {self.input_dim}")
        
        # Create autoencoder
        self.autoencoder = WaveletAutoencoder(
            input_dim=self.input_dim,
            hidden_dims=config.hidden_dims,
            latent_dim=config.latent_dim
        )
        
        # Calculate compression ratio
        self.compression_ratio = config.audio_length / config.latent_dim
        print(f"Compression ratio: {self.compression_ratio:.2f}x")
    
    def forward(self, x):
        # Apply wavelet transform
        wavelet_coeffs = self.wavelet_transform.forward(x)
        
        # Encode and decode
        wavelet_coeffs_recon, z = self.autoencoder(wavelet_coeffs)
        
        # Apply inverse transform
        x_recon = self.wavelet_transform.inverse(wavelet_coeffs_recon)
        
        # Ensure same length
        min_len = min(x.shape[1], x_recon.shape[1])
        x = x[:, :min_len]
        x_recon = x_recon[:, :min_len]
        
        return x_recon, wavelet_coeffs, wavelet_coeffs_recon, z
    
    def training_step(self, batch, batch_idx):
        x = batch
        x_recon, wavelet_coeffs, wavelet_coeffs_recon, _ = self(x)
        
        # Compute losses
        wavelet_loss = F.mse_loss(wavelet_coeffs_recon, wavelet_coeffs)
        recon_loss = F.mse_loss(x_recon, x)
        
        # Total loss (with more weight on waveform reconstruction)
        loss = recon_loss + 0.1 * wavelet_loss
        
        # Calculate SNR
        snr = calculate_snr(x, x_recon)
        avg_snr = torch.mean(snr)
        
        # Log metrics
        self.log('train_loss', loss)
        self.log('train_recon_loss', recon_loss)
        self.log('train_wavelet_loss', wavelet_loss)
        self.log('train_snr', avg_snr)
        
        # Periodically log audio and visualizations
        if batch_idx % 10 == 0 or batch_idx == 0:
            self._log_audio_and_visualizations(x, x_recon, wavelet_coeffs, wavelet_coeffs_recon)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch
        x_recon, wavelet_coeffs, wavelet_coeffs_recon, _ = self(x)
        
        # Compute losses
        wavelet_loss = F.mse_loss(wavelet_coeffs_recon, wavelet_coeffs)
        recon_loss = F.mse_loss(x_recon, x)
        
        # Total loss
        loss = recon_loss + 0.1 * wavelet_loss
        
        # Calculate SNR
        snr = calculate_snr(x, x_recon)
        avg_snr = torch.mean(snr)
        
        # Log metrics
        self.log('val_loss', loss)
        self.log('val_recon_loss', recon_loss)
        self.log('val_wavelet_loss', wavelet_loss)
        self.log('val_snr', avg_snr)
        
        return loss
        
    def test_step(self, batch, batch_idx):
        """Test step for model evaluation
        
        Using same metrics as validation but with 'test_' prefix
        """
        x = batch
        x_recon, wavelet_coeffs, wavelet_coeffs_recon, _ = self(x)
        
        # Compute losses
        wavelet_loss = F.mse_loss(wavelet_coeffs_recon, wavelet_coeffs)
        recon_loss = F.mse_loss(x_recon, x)
        
        # Total loss
        loss = recon_loss + 0.1 * wavelet_loss
        
        # Calculate SNR
        snr = calculate_snr(x, x_recon)
        avg_snr = torch.mean(snr)
        
        # Log metrics
        self.log('test_loss', loss)
        self.log('test_recon_loss', recon_loss)
        self.log('test_wavelet_loss', wavelet_loss)
        self.log('test_snr', avg_snr)
        
        # If it's the first batch, log some audio samples for qualitative evaluation
        if batch_idx == 0:
            self._log_audio_and_visualizations(x, x_recon, wavelet_coeffs, wavelet_coeffs_recon)
        
        return loss
    
    def _log_audio_and_visualizations(self, x, x_recon, wavelet_coeffs, wavelet_coeffs_recon):
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
        
        # Plot waveforms and wavelet coefficients
        fig, axs = plt.subplots(2, 2, figsize=(10, 8))
        
        # Original waveform
        axs[0, 0].plot(x[sample_idx].cpu().numpy())
        axs[0, 0].set_title('Original Waveform')
        
        # Reconstructed waveform
        axs[0, 1].plot(x_recon[sample_idx].cpu().numpy())
        axs[0, 1].set_title('Reconstructed Waveform')
        
        # Original wavelet coefficients
        axs[1, 0].plot(wavelet_coeffs[sample_idx].cpu().numpy())
        axs[1, 0].set_title('Original Wavelet Coefficients')
        
        # Reconstructed wavelet coefficients
        axs[1, 1].plot(wavelet_coeffs_recon[sample_idx].cpu().detach().numpy())
        axs[1, 1].set_title('Reconstructed Wavelet Coefficients')
        
        plt.tight_layout()
        
        # Add figure to tensorboard
        self.logger.experiment.add_figure(
            'waveform_comparison', 
            fig, 
            self.global_step
        )
        plt.close(fig)
        
        # Plot spectrogram comparison if step is a multiple of 20
        if self.global_step % 20 == 0:
            # Compute spectrograms using FFT
            n_fft = 1024
            hop_length = 256
            
            x_np = x[sample_idx].cpu().numpy()
            x_recon_np = x_recon[sample_idx].cpu().numpy()
            
            # Compute spectrograms
            spec_orig = compute_spectrogram(x_np, n_fft, hop_length)
            spec_recon = compute_spectrogram(x_recon_np, n_fft, hop_length)
            
            # Convert to dB scale
            eps = 1e-10
            spec_orig_db = 20 * np.log10(spec_orig + eps)
            spec_recon_db = 20 * np.log10(spec_recon + eps)
            
            # Clip values for better visualization
            vmin = -80
            vmax = 0
            spec_orig_db = np.clip(spec_orig_db, vmin, vmax)
            spec_recon_db = np.clip(spec_recon_db, vmin, vmax)
            
            # Plot spectrograms
            fig, axs = plt.subplots(1, 2, figsize=(12, 5))
            
            axs[0].imshow(spec_orig_db, aspect='auto', origin='lower', cmap='viridis')
            axs[0].set_title('Original Spectrogram')
            axs[0].set_xlabel('Time Frames')
            axs[0].set_ylabel('Frequency Bins')
            
            axs[1].imshow(spec_recon_db, aspect='auto', origin='lower', cmap='viridis')
            axs[1].set_title('Reconstructed Spectrogram')
            axs[1].set_xlabel('Time Frames')
            
            plt.tight_layout()
            
            # Add figure to tensorboard
            self.logger.experiment.add_figure(
                'spectrogram_comparison', 
                fig, 
                self.global_step
            )
            plt.close(fig)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.config.learning_rate)