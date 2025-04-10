#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import numpy as np
import io
from torchvision.transforms import ToTensor
from PIL import Image

from models.encoder import ComplexEncoder
from models.decoder import ComplexDecoder
from models.wavelet import WaveletScatteringTransform
from models.complex_layers import ComplexToReal
from utils.losses import ComplexSTFTLoss, TimeFrequencyLoss, WaveletLoss
from utils.metrics import compute_metrics_dict


class ComplexAudioEncoderDecoder(pl.LightningModule):
    def __init__(self, config):
        """
        PyTorch Lightning module for complex-valued audio encoder-decoder
        
        Args:
            config (dict): Configuration dictionary
        """
        super(ComplexAudioEncoderDecoder, self).__init__()
        self.save_hyperparameters(config)
        self.config = config
        
        # Initialize wavelet scattering transform
        self.wst = WaveletScatteringTransform(
            J=config['wavelet']['J'],
            Q=config['wavelet']['Q'],
            T=config['wavelet']['T'],
            sample_rate=config['data']['sample_rate'],
            normalize=config['wavelet'].get('normalize', True),
            max_order=config['wavelet'].get('max_order', 1),
            ensure_output_dim=config['wavelet'].get('ensure_output_dim', True)
        )
        
        # Calculate input channels based on wavelet parameters
        J, Q = config['wavelet']['J'], config['wavelet']['Q']
        if config['wavelet'].get('max_order', 1) == 1:
            # For first-order: lowpass + bandpass filters
            input_channels = 1 + J * Q
        else:
            # For second-order: approximate formula
            input_channels = 1 + J * Q + (J * Q)**2 // 4
        
        print(f"Initializing encoder with {input_channels} input channels")
        
        # Extract model parameters from config
        channels = config['model']['channels']
        kernel_sizes = config['model']['kernel_sizes']
        strides = config['model']['strides']
        paddings = config['model'].get('paddings', None)
        output_paddings = config['model'].get('output_paddings', None)
        
        # Initialize encoder
        self.encoder = ComplexEncoder(
            input_channels=input_channels,
            channels=channels,
            kernel_sizes=kernel_sizes,
            strides=strides,
            paddings=paddings,
            dropout=config['model']['dropout'],
            use_batch_norm=config['model']['use_batch_norm']
        )
        
        # Initialize decoder
        self.decoder = ComplexDecoder(
            latent_channels=channels[-1],
            channels=channels,
            kernel_sizes=kernel_sizes,
            strides=strides,
            paddings=paddings,
            output_paddings=output_paddings,
            output_channels=1,  # Single channel audio output
            dropout=config['model']['dropout'],
            use_batch_norm=config['model']['use_batch_norm']
        )
        
        # Enable gradient checkpointing if configured
        if config['model'].get('use_gradient_checkpointing', False):
            self.encoder.enable_gradient_checkpointing()
            self.decoder.enable_gradient_checkpointing()
        
        # Convert complex output to real
        complex_repr = config['model'].get('complex_repr', 'mag_phase')
        self.to_real = ComplexToReal(mode=complex_repr)
        
        # Initialize loss functions
        self.stft_loss = ComplexSTFTLoss()
        self.time_freq_loss = TimeFrequencyLoss()
        self.wavelet_loss = WaveletLoss(self.wst)
        
        # Loss weights from config
        self.l1_weight = config['loss']['l1_weight']
        self.stft_weight = config['loss']['stft_weight']
        self.complex_loss_weight = config['loss']['complex_loss_weight']
        self.spectral_convergence_weight = config['loss']['spectral_convergence_weight']
        self.phase_consistency_weight = config['loss'].get('phase_consistency_weight', 0.3)
        
        # Metrics
        self.train_step_outputs = []
        self.val_step_outputs = []
        self.test_step_outputs = []
    
    def forward(self, x):
        """
        Forward pass through the model
        
        Args:
            x (Tensor): Input audio [B, T]
                
        Returns:
            Tensor: Reconstructed audio [B, T]
        """
        # Store original batch size and length
        batch_size = x.size(0)
        original_length = x.size(-1)
        
        # Apply wavelet scattering transform - returns [B, C, T] format
        wst_output = self.wst(x)
        
        # Pass through encoder - expects [B, C, T] format
        encoded, intermediates = self.encoder(wst_output)
        
        # Pass through decoder with skip connections
        decoded = self.decoder(encoded, intermediates)
        
        # Convert complex output to real
        output = self.to_real(decoded)
        
        # Reshape output to match original audio length if needed
        if output.size(-1) != original_length:
            # Use interpolation to match original length
            if output.dim() == 2:  # If [B, T]
                output = output.unsqueeze(1)  # Add channel dim -> [B, 1, T]
                
            output = F.interpolate(
                output, 
                size=original_length,
                mode='linear', 
                align_corners=False
            )
            
            if output.dim() == 3 and output.size(1) == 1:
                output = output.squeeze(1)  # Remove channel dim -> [B, T]
        
        # Check for NaN or Inf values
        if torch.isnan(output).any() or torch.isinf(output).any():
            print("WARNING: Output contains NaN or Inf values, replacing with zeros")
            output = torch.nan_to_num(output, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return output
    
    def configure_optimizers(self):
        """Configure optimizers for training"""
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.config['training']['lr'],
            betas=(self.config['training']['beta1'], self.config['training']['beta2'])
        )
        
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=self.config['training']['scheduler_gamma']
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1
            }
        }
    
    def training_step(self, batch, batch_idx):
        """
        Training step
        
        Args:
            batch (Tensor): Batch of audio samples [B, T]
            batch_idx (int): Batch index
            
        Returns:
            dict: Loss dictionary
        """
        # Forward pass
        x = batch
        x_hat = self(x)
        
        # Compute losses
        l1_loss = F.l1_loss(x_hat, x)
        complex_loss, mag_loss, phase_loss = self.stft_loss(x_hat, x)
        tf_loss, time_loss, freq_loss = self.time_freq_loss(x_hat, x)
        
        # Wavelet loss
        wst_loss = self.wavelet_loss(x_hat, x)
        
        # Calculate multi-resolution phase consistency loss
        phase_consistency_loss = 0.0
        for n_fft, hop_length in zip([512, 1024, 2048], [128, 256, 512]):
            window = torch.hann_window(n_fft).to(x.device)
            
            # Compute complex STFT
            x_stft = torch.stft(x, n_fft, hop_length, window=window, return_complex=True)
            x_hat_stft = torch.stft(x_hat, n_fft, hop_length, window=window, return_complex=True)
            
            # Extract phase and magnitude information
            x_phase = torch.angle(x_stft)
            x_hat_phase = torch.angle(x_hat_stft)
            x_mag = torch.abs(x_stft)
            
            # Weight phase difference by magnitude to focus on high-energy regions
            phase_diff = torch.abs(torch.remainder(x_phase - x_hat_phase + np.pi, 2 * np.pi) - np.pi)
            mag_weight = (x_mag / (torch.mean(x_mag) + 1e-8)) ** 2
            
            # Weighted phase consistency (1 - cos(Δφ))
            weighted_phase_loss = torch.mean(mag_weight * (1 - torch.cos(phase_diff)))
            phase_consistency_loss += weighted_phase_loss
        
        # Average across all resolutions
        phase_consistency_loss /= 3.0
        
        # Combined loss
        loss = (
            self.l1_weight * l1_loss +
            self.stft_weight * tf_loss +
            self.complex_loss_weight * complex_loss +
            self.phase_consistency_weight * phase_consistency_loss
        )
        
        # Log losses
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_l1_loss', l1_loss)
        self.log('train_stft_loss', tf_loss)
        self.log('train_complex_loss', complex_loss)
        self.log('train_phase_loss', phase_consistency_loss)
        
        # Store step outputs
        self.train_step_outputs.append({
            'loss': loss.detach(),
            'l1_loss': l1_loss.detach(),
            'stft_loss': tf_loss.detach(),
            'complex_loss': complex_loss.detach(),
            'phase_loss': phase_consistency_loss.detach()
        })
        
        return {'loss': loss}
    
    def validation_step(self, batch, batch_idx):
        """
        Validation step
        
        Args:
            batch (Tensor): Batch of audio samples [B, T]
            batch_idx (int): Batch index
            
        Returns:
            dict: Loss and metrics dictionary
        """
        # Forward pass
        x = batch
        x_hat = self(x)
        
        # Ensure shapes match for metrics calculation
        if x_hat.shape != x.shape:
            if x_hat.size(-1) > x.size(-1):
                x_hat = x_hat[..., :x.size(-1)]
            else:
                x_hat = F.pad(x_hat, (0, x.size(-1) - x_hat.size(-1)))
        
        # Compute losses (same as training)
        l1_loss = F.l1_loss(x_hat, x)
        complex_loss, mag_loss, phase_loss = self.stft_loss(x_hat, x)
        tf_loss, time_loss, freq_loss = self.time_freq_loss(x_hat, x)
        
        # Calculate phase consistency loss (same as training)
        phase_consistency_loss = 0.0
        for n_fft, hop_length in zip([512, 1024, 2048], [128, 256, 512]):
            window = torch.hann_window(n_fft).to(x.device)
            
            # Compute complex STFT
            x_stft = torch.stft(x, n_fft, hop_length, window=window, return_complex=True)
            x_hat_stft = torch.stft(x_hat, n_fft, hop_length, window=window, return_complex=True)
            
            # Extract phase and magnitude information
            x_phase = torch.angle(x_stft)
            x_hat_phase = torch.angle(x_hat_stft)
            x_mag = torch.abs(x_stft)
            
            # Weight phase difference by magnitude
            phase_diff = torch.abs(torch.remainder(x_phase - x_hat_phase + np.pi, 2 * np.pi) - np.pi)
            mag_weight = (x_mag / (torch.mean(x_mag) + 1e-8)) ** 2
            
            # Weighted phase consistency
            weighted_phase_loss = torch.mean(mag_weight * (1 - torch.cos(phase_diff)))
            phase_consistency_loss += weighted_phase_loss
        
        # Average across all resolutions
        phase_consistency_loss /= 3.0
        
        # Combined loss
        loss = (
            self.l1_weight * l1_loss +
            self.stft_weight * tf_loss +
            self.complex_loss_weight * complex_loss +
            self.phase_consistency_weight * phase_consistency_loss
        )
        
        # Compute metrics
        metrics = compute_metrics_dict(x_hat, x, sample_rate=self.config['data']['sample_rate'])
        
        # Log losses and metrics
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_l1_loss', l1_loss)
        self.log('val_stft_loss', tf_loss)
        self.log('val_complex_loss', complex_loss)
        self.log('val_phase_consistency_loss', phase_consistency_loss)
        
        for metric_name, metric_value in metrics.items():
            self.log(f'val_{metric_name}', metric_value)
        
        # Store step outputs
        output = {
            'loss': loss.detach(),
            'l1_loss': l1_loss.detach(),
            'stft_loss': tf_loss.detach(),
            'complex_loss': complex_loss.detach(),
            'phase_consistency_loss': phase_consistency_loss.detach(),
            'snr': metrics['snr_db'],
            'spectral_distortion': metrics['spectral_distortion'],
            'audio_input': x[0].detach().cpu(),
            'audio_output': x_hat[0].detach().cpu()
        }
        
        self.val_step_outputs.append(output)
        
        # Log audio and spectrograms for first batch only
        if batch_idx == 0:
            self._log_audio_spectrograms(x, x_hat)
        
        return output
    
    def test_step(self, batch, batch_idx):
        """
        Test step
        
        Args:
            batch (Tensor): Batch of audio samples [B, T]
            batch_idx (int): Batch index
            
        Returns:
            dict: Metrics dictionary
        """
        # Forward pass
        x = batch
        x_hat = self(x)
        
        # Ensure shapes match for metrics calculation
        if x_hat.shape != x.shape:
            if x_hat.size(-1) > x.size(-1):
                x_hat = x_hat[..., :x.size(-1)]
            else:
                x_hat = F.pad(x_hat, (0, x.size(-1) - x_hat.size(-1)))
        
        # Compute metrics
        metrics = compute_metrics_dict(x_hat, x, sample_rate=self.config['data']['sample_rate'])
        
        # Log metrics
        for metric_name, metric_value in metrics.items():
            self.log(f'test_{metric_name}', metric_value)
        
        # Store audio for later analysis
        self.test_step_outputs.append({
            'snr': metrics['snr_db'],
            'spectral_distortion': metrics['spectral_distortion'],
            'audio_input': x[0].detach().cpu(),
            'audio_output': x_hat[0].detach().cpu()
        })
        
        return metrics
    
    def on_train_epoch_end(self):
        """Called at the end of the training epoch"""
        # Calculate average losses
        avg_loss = torch.stack([x['loss'] for x in self.train_step_outputs]).mean()
        avg_l1_loss = torch.stack([x['l1_loss'] for x in self.train_step_outputs]).mean()
        avg_stft_loss = torch.stack([x['stft_loss'] for x in self.train_step_outputs]).mean()
        avg_complex_loss = torch.stack([x['complex_loss'] for x in self.train_step_outputs]).mean()
        
        # Log average losses
        self.log('train_avg_loss', avg_loss, prog_bar=True)
        self.log('train_avg_l1_loss', avg_l1_loss)
        self.log('train_avg_stft_loss', avg_stft_loss)
        self.log('train_avg_complex_loss', avg_complex_loss)
        
        # Clear step outputs
        self.train_step_outputs.clear()
    
    def on_validation_epoch_end(self):
        """Called at the end of the validation epoch"""
        # Calculate average losses and metrics
        avg_loss = torch.stack([x['loss'] for x in self.val_step_outputs]).mean()
        avg_l1_loss = torch.stack([x['l1_loss'] for x in self.val_step_outputs]).mean()
        avg_stft_loss = torch.stack([x['stft_loss'] for x in self.val_step_outputs]).mean()
        avg_complex_loss = torch.stack([x['complex_loss'] for x in self.val_step_outputs]).mean()
        avg_snr = np.mean([x['snr'] for x in self.val_step_outputs])
        avg_spectral_distortion = np.mean([x['spectral_distortion'] for x in self.val_step_outputs])
        
        # Log average losses and metrics
        self.log('val_avg_loss', avg_loss, prog_bar=True)
        self.log('val_avg_l1_loss', avg_l1_loss)
        self.log('val_avg_stft_loss', avg_stft_loss)
        self.log('val_avg_complex_loss', avg_complex_loss)
        self.log('val_avg_snr', avg_snr)
        self.log('val_avg_spectral_distortion', avg_spectral_distortion)
        
        # Clear step outputs
        self.val_step_outputs.clear()
    
    def on_test_epoch_end(self):
        """Called at the end of the test epoch"""
        # Calculate average metrics
        avg_snr = np.mean([x['snr'] for x in self.test_step_outputs])
        avg_spectral_distortion = np.mean([x['spectral_distortion'] for x in self.test_step_outputs])
        
        # Log average metrics
        self.log('test_avg_snr', avg_snr)
        self.log('test_avg_spectral_distortion', avg_spectral_distortion)
        
        # Clear step outputs
        self.test_step_outputs.clear()
    
    def _compute_stft(self, audio, n_fft=1024, hop_length=256):
        """
        Compute STFT for visualization
        
        Args:
            audio (Tensor): Audio signal [T]
            n_fft (int): FFT size
            hop_length (int): Hop length
            
        Returns:
            Tensor: Magnitude spectrogram
        """
        # Ensure audio is on CPU for visualization
        if not isinstance(audio, torch.Tensor):
            audio = torch.tensor(audio)
        
        # Move to CPU if on GPU
        if audio.is_cuda:
            audio = audio.cpu()
        
        # Compute STFT
        try:
            # Create window on the same device as audio
            window = torch.hann_window(n_fft, device=audio.device)
            
            stft = torch.stft(
                audio,
                n_fft=n_fft,
                hop_length=hop_length,
                window=window,
                return_complex=True
            )
            
            # Convert to magnitude in dB
            mag = torch.log10(torch.abs(stft) + 1e-8)
            
            return mag
        except Exception as e:
            print(f"Error computing STFT: {e}")
            # Return fallback empty spectrogram
            return torch.zeros((n_fft // 2 + 1, audio.shape[-1] // hop_length + 1))
    
    def _log_audio_spectrograms(self, input_audio, output_audio):
        """
        Log audio and spectrograms to TensorBoard
        
        Args:
            input_audio (Tensor): Input audio [B, T]
            output_audio (Tensor): Reconstructed audio [B, T]
        """
        # Take first example from batch
        input_sample = input_audio[0].detach().cpu()
        output_sample = output_audio[0].detach().cpu()
        
        # Log audio samples
        sample_rate = self.config['data']['sample_rate']
        self.logger.experiment.add_audio('input_audio', input_sample.unsqueeze(0), 
                                         global_step=self.global_step, sample_rate=sample_rate)
        self.logger.experiment.add_audio('output_audio', output_sample.unsqueeze(0), 
                                         global_step=self.global_step, sample_rate=sample_rate)
        
        # Compute STFT
        input_stft = self._compute_stft(input_sample)
        output_stft = self._compute_stft(output_sample)
        
        # Create figure with spectrograms
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        
        # Safely convert tensors to numpy arrays for plotting
        input_stft_np = input_stft.numpy() if input_stft.numel() > 0 else np.zeros((1, 1))
        output_stft_np = output_stft.numpy() if output_stft.numel() > 0 else np.zeros((1, 1))
        
        # Plot input spectrogram
        im0 = axes[0].imshow(input_stft_np, aspect='auto', origin='lower', cmap='viridis')
        axes[0].set_title('Input Spectrogram')
        axes[0].set_ylabel('Frequency bin')
        plt.colorbar(im0, ax=axes[0])
        
        # Plot output spectrogram
        im1 = axes[1].imshow(output_stft_np, aspect='auto', origin='lower', cmap='viridis')
        axes[1].set_title('Reconstructed Spectrogram')
        axes[1].set_xlabel('Time frame')
        axes[1].set_ylabel('Frequency bin')
        plt.colorbar(im1, ax=axes[1])
        
        plt.tight_layout()
        
        # Save figure to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        
        # Create image from buffer
        image = Image.open(buf)
        image = ToTensor()(image)
        
        # Log to TensorBoard
        self.logger.experiment.add_image('spectrograms', image, global_step=self.global_step)
        
        # Close plot
        plt.close(fig)
        
        # Also plot waveforms
        fig, axes = plt.subplots(2, 1, figsize=(10, 4))
        
        # Plot input waveform
        axes[0].plot(input_sample.numpy())
        axes[0].set_title('Input Waveform')
        axes[0].set_xlim(0, len(input_sample))
        
        # Plot output waveform
        axes[1].plot(output_sample.numpy())
        axes[1].set_title('Reconstructed Waveform')
        axes[1].set_xlabel('Sample')
        axes[1].set_xlim(0, len(output_sample))
        
        plt.tight_layout()
        
        # Save figure to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        
        # Create image from buffer
        image = Image.open(buf)
        image = ToTensor()(image)
        
        # Log to TensorBoard
        self.logger.experiment.add_image('waveforms', image, global_step=self.global_step)
        
        # Close plot
        plt.close(fig)