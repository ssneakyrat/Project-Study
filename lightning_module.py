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
        
        # Initialize wavelet scattering transform with reduced J value
        self.wst = WaveletScatteringTransform(
            J=config['wavelet']['J'],  # Now using J=2 instead of J=4
            Q=config['wavelet']['Q'],
            T=config['wavelet']['T'],
            sample_rate=config['data']['sample_rate'],
            normalize=True,
            ensure_output_dim=config['wavelet'].get('ensure_output_dim', True)
        )
        
        # Get the number of input channels from WST
        # Use a dummy input to get the output shape
        try:
            dummy_input = torch.zeros(1, config['wavelet']['T'])
            with torch.no_grad():
                wst_output = self.wst(dummy_input)
                # Handle both tuple return (real, imag) and single tensor return
                if isinstance(wst_output, tuple):
                    input_channels = wst_output[0].shape[1]  # Real part channels
                else:
                    input_channels = wst_output.shape[1]  # All channels
            print(f"Initializing encoder with {input_channels} input channels")
        except Exception as e:
            print(f"Warning: Error determining WST output shape: {e}")
            # Fallback to an estimate based on configuration
            J, Q = config['wavelet']['J'], config['wavelet']['Q']
            input_channels = J * Q  # Conservative estimate
        
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
        
        # Convert complex output to real
        # Changed from 'magnitude' to 'real' to preserve phase information
        self.to_real = ComplexToReal(mode='mag_phase')  
        
        # Initialize loss functions
        self.stft_loss = ComplexSTFTLoss()
        self.time_freq_loss = TimeFrequencyLoss()
        self.wavelet_loss = WaveletLoss(self.wst)
        
        # Loss weights from config
        self.l1_weight = config['loss']['l1_weight']
        self.stft_weight = config['loss']['stft_weight']
        self.complex_loss_weight = config['loss']['complex_loss_weight']  # Now 0.5 instead of 0.25
        self.spectral_convergence_weight = config['loss']['spectral_convergence_weight']
        
        # Metrics
        self.train_step_outputs = []
        self.val_step_outputs = []
        self.test_step_outputs = []
    
    def forward(self, x):
        """
        Forward pass through the model with improved dimension handling
        
        Args:
            x (Tensor): Input audio [B, T]
                
        Returns:
            Tensor: Reconstructed audio [B, T]
        """
        # Ensure input has correct shape
        if x.dim() == 1:  # Single audio sample without batch dimension
            x = x.unsqueeze(0)  # Add batch dimension
        
        # Store original shape for output reshaping
        batch_size = x.size(0)
        original_length = x.size(-1)
        original_dim = x.dim()  # Store original dimensionality
        
        # Ensure the input has the right length for wavelet transform
        target_length = self.config['wavelet']['T']
        if x.size(-1) != target_length:
            # Pad or trim input to match the expected wavelet transform length
            if x.size(-1) < target_length:
                # Pad
                padding = target_length - x.size(-1)
                x = F.pad(x, (0, padding))
            else:
                # Trim
                x = x[..., :target_length]
        
        # Apply wavelet scattering transform
        print(f"Input to WST: {x.shape}")
        wst_output = self.wst(x)
        
        # Verify tensor shapes after WST
        if isinstance(wst_output, tuple):
            real, imag = wst_output
            print(f"WST output - Real: {real.shape}, Imag: {imag.shape}")
        else:
            print(f"WST output: {wst_output.shape}")
        
        # Double-check for dimensional rotation
        if isinstance(wst_output, tuple):
            real, imag = wst_output
            # If channels < time dimension, transpose to fix rotation
            if real.size(1) < real.size(2):
                print(f"Lightning module fixing dimensions: {real.shape} -> ", end="")
                real = real.transpose(1, 2)
                imag = imag.transpose(1, 2)
                print(f"{real.shape}")
                wst_output = (real, imag)
        else:
            # If channels < time dimension, transpose to fix rotation
            if wst_output.size(1) < wst_output.size(2):
                print(f"Lightning module fixing dimensions: {wst_output.shape} -> ", end="")
                wst_output = wst_output.transpose(1, 2)
                print(f"{wst_output.shape}")
        
        # Pass through encoder
        print(f"Input to encoder: {wst_output[0].shape if isinstance(wst_output, tuple) else wst_output.shape}")
        encoded, intermediates = self.encoder(wst_output)
        
        # Print encoder output shape
        if isinstance(encoded, tuple):
            print(f"Encoder output - Real: {encoded[0].shape}, Imag: {encoded[1].shape}")
        else:
            print(f"Encoder output: {encoded.shape}")
        
        # Pass through decoder with skip connections
        decoded = self.decoder(encoded, intermediates)
        
        # Print decoder output shape
        if isinstance(decoded, tuple):
            print(f"Decoder output - Real: {decoded[0].shape}, Imag: {decoded[1].shape}")
        else:
            print(f"Decoder output: {decoded.shape}")
        
        # Convert complex output to real
        output = self.to_real(decoded)
        print(f"After ComplexToReal: {output.shape}")
        
        # Check for very small values in output
        num_zeros = torch.sum((torch.abs(output) < 1e-6).float()).item()
        total_elements = output.numel()
        print(f"Final output - Percentage of near-zero values: {100 * num_zeros / total_elements:.2f}%")
        
        # Ensure output has the same length as the original input
        if output.size(-1) != original_length:
            print(f"Resizing output from {output.size(-1)} to {original_length}")
            if output.size(-1) > original_length:
                # Trim
                output = output[..., :original_length]
            else:
                # Pad
                padding = original_length - output.size(-1)
                output = F.pad(output, (0, padding))
        
        # Match original dimensionality
        if output.dim() != original_dim:
            if original_dim == 2 and output.dim() == 3:
                output = output.squeeze(1)
        
        # Check for NaN or Inf values in final output
        has_nan = torch.isnan(output).any().item()
        has_inf = torch.isinf(output).any().item()
        if has_nan or has_inf:
            print("WARNING: Final output contains NaN or Inf values!")
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
        Training step with added phase consistency loss
        
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
        
        # Add explicit phase consistency loss
        # Compute STFT for phase comparison
        n_fft = 1024
        hop_length = 256
        window = torch.hann_window(n_fft).to(x.device)
        
        # Compute complex STFT
        x_stft = torch.stft(x, n_fft, hop_length, window=window, return_complex=True)
        x_hat_stft = torch.stft(x_hat, n_fft, hop_length, window=window, return_complex=True)
        
        # Extract phase information
        x_phase = torch.angle(x_stft)
        x_hat_phase = torch.angle(x_hat_stft)
        
        # Compute phase consistency loss (1 - cos(Δφ))
        # This penalizes phase differences proportionally to their angular distance
        phase_diff = x_phase - x_hat_phase
        phase_consistency_loss = torch.mean(1 - torch.cos(phase_diff))
        
        # Apply weight to phase consistency loss
        phase_consistency_weight = 0.3
        
        # Combined loss with new phase component
        loss = (
            self.l1_weight * l1_loss +
            self.stft_weight * tf_loss +
            self.complex_loss_weight * complex_loss +
            phase_consistency_weight * phase_consistency_loss
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
        Validation step with added phase consistency loss
        
        Args:
            batch (Tensor): Batch of audio samples [B, T]
            batch_idx (int): Batch index
            
        Returns:
            dict: Loss and metrics dictionary
        """
        # Forward pass
        x = batch
        x_hat = self(x)
        
        # Ensure shape matching for metrics calculation
        if x_hat.dim() != x.dim():
            if x_hat.dim() == 3 and x.dim() == 2:
                x_hat = x_hat.squeeze(1)
            elif x_hat.dim() == 2 and x.dim() == 3:
                x_hat = x_hat.unsqueeze(1)
        
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
        
        # Add phase consistency loss (same as in training)
        n_fft = 1024
        hop_length = 256
        window = torch.hann_window(n_fft).to(x.device)
        
        # Compute complex STFT
        x_stft = torch.stft(x, n_fft, hop_length, window=window, return_complex=True)
        x_hat_stft = torch.stft(x_hat, n_fft, hop_length, window=window, return_complex=True)
        
        # Extract phase information
        x_phase = torch.angle(x_stft)
        x_hat_phase = torch.angle(x_hat_stft)
        
        # Compute phase consistency loss
        phase_diff = x_phase - x_hat_phase
        phase_consistency_loss = torch.mean(1 - torch.cos(phase_diff))
        
        # Weight the phase consistency loss
        phase_consistency_weight = 0.3
        
        # Combined loss
        loss = (
            self.l1_weight * l1_loss +
            self.stft_weight * tf_loss +
            self.complex_loss_weight * complex_loss +
            phase_consistency_weight * phase_consistency_loss
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
        
        # Ensure shape matching for metrics calculation
        if x_hat.dim() != x.dim():
            if x_hat.dim() == 3 and x.dim() == 2:
                x_hat = x_hat.squeeze(1)
            elif x_hat.dim() == 2 and x.dim() == 3:
                x_hat = x_hat.unsqueeze(1)
        
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
            
            # Ensure result is on CPU as numpy
            if mag.is_cuda:
                mag = mag.cpu()
            
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