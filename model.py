import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import numpy as np
import math
from complex_layers import (
    ComplexConv1d, ComplexConvTranspose1d, ComplexBatchNorm1d, 
    ComplexPReLU, ComplexToReal, RealToComplex
)
from wavelet_transform import WaveletScatteringTransform


class WSTVocoder(pl.LightningModule):
    def __init__(self,
                 sample_rate=16000,
                 wst_J=7,  # Increased from 6 to 7 for better frequency resolution
                 wst_Q=12,  # Increased from 8 to 12 for finer frequency bands
                 channels=[64, 128, 256],
                 latent_dim=None,
                 kernel_sizes=[5, 5, 5],
                 strides=[2, 2, 2],
                 compression_factor=4,  # Reduced for better reconstruction
                 learning_rate=1e-4):
        """WST-based vocoder model."""
        super().__init__()
        self.save_hyperparameters()
        
        self.sample_rate = sample_rate
        self.learning_rate = learning_rate
        self.channels = channels
        self.compression_factor = compression_factor
        
        # WST layer with improved frequency-time resolution tradeoff
        self.wst = WaveletScatteringTransform(
            J=wst_J,
            Q=wst_Q,
            T=sample_rate * 2,  # 2 seconds of audio
            max_order=2,
            out_type='array'
        )
        
        # Pre-compute WST output shape with a dummy input
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, sample_rate * 2)
            dummy_output = self.wst(dummy_input)
            self.wst_channels = dummy_output.shape[1]
            self.wst_time_dim = dummy_output.shape[2]
            
            # Calculate bottleneck dimensions with reduced compression
            T = self.wst_time_dim
            F = self.wst_channels
            self.latent_dim = int(math.sqrt(T * F) / compression_factor * 2)
            self.latent_dim = max(32, self.latent_dim)  # Increased minimum size
            
            if latent_dim is not None:
                self.latent_dim = latent_dim
        
        # Store original length for final upsampling
        self.original_length = sample_rate * 2
        
        # Improved phase handling with modified analytic signal approach
        self.real_to_complex = RealToComplex(mode='analytic')
        
        # Encoder layers
        self.encoder_layers = nn.ModuleList()
        
        current_time_dim = self.wst_time_dim
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
            current_time_dim = (current_time_dim + 2*(kernel_size // 2) - kernel_size) // stride + 1
        
        # Latent projection with spectral normalization for stability
        self.latent_projection = nn.Sequential(
            ComplexConv1d(channels[-1], self.latent_dim, kernel_size=3, padding=1),
            ComplexBatchNorm1d(self.latent_dim),
            ComplexPReLU()
        )
        
        # Decoder layers
        self.decoder_layers = nn.ModuleList()
        
        for i, (in_channels, out_channels, kernel_size, stride) in enumerate(
            zip(
                [self.latent_dim] + channels[:-1],
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
        
        # Output layer with larger kernel for better time coherence
        self.output_layer = ComplexConvTranspose1d(
            channels[-1],
            1,  # Output channels
            kernel_size=7,  # Increased from 5 to 7
            stride=strides[0],
            padding=3,
            output_padding=strides[0] - 1
        )
        
        # Use refined polar mode for better phase preservation
        self.complex_to_real = ComplexToReal(mode='polar')
        
        # Skip projections with gradient scaling for stable training
        encoder_channels = channels
        decoder_input_channels = [self.latent_dim] + channels[:-1]
        
        self.skip_projections = nn.ModuleList()
        for i in range(len(encoder_channels)):
            enc_ch = encoder_channels[i]
            dec_idx = len(decoder_input_channels) - 1 - i
            dec_ch = decoder_input_channels[dec_idx]
            
            self.skip_projections.append(
                ComplexConv1d(enc_ch, dec_ch, kernel_size=1)
            )
        
        # Improved final upsampling with antialiasing
        self.final_upsampling = nn.Sequential(
            nn.Upsample(size=sample_rate * 2, mode='linear', align_corners=False),
            nn.Conv1d(2, 16, kernel_size=7, padding=3, padding_mode='reflect'),
            nn.LeakyReLU(0.2),
            nn.Conv1d(16, 16, kernel_size=7, padding=3, padding_mode='reflect'),
            nn.LeakyReLU(0.2),
            nn.Conv1d(16, 2, kernel_size=7, padding=3, padding_mode='reflect'),
        )
        
        # Energy scaling factor for output normalization
        self.output_scale = nn.Parameter(torch.ones(1))
        
        # Enhanced phase correction network with frequency-dependent processing
        self.phase_correction = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=15, padding=7, padding_mode='reflect'),
            nn.LeakyReLU(0.2),
            nn.Conv1d(32, 32, kernel_size=15, padding=7, padding_mode='reflect'),
            nn.LeakyReLU(0.2),
            nn.Conv1d(32, 32, kernel_size=15, padding=7, padding_mode='reflect'),
            nn.LeakyReLU(0.2),
            nn.Conv1d(32, 2, kernel_size=15, padding=7, padding_mode='reflect'),
            nn.Tanh()  # Output correction factor between -1 and 1
        )
        
        # Refined complex synthesis layer
        self.complex_synthesis = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=7, padding=3, padding_mode='reflect'),
            nn.LeakyReLU(0.2),
            nn.Conv1d(32, 32, kernel_size=7, padding=3, padding_mode='reflect'),
            nn.LeakyReLU(0.2),
            nn.Conv1d(32, 1, kernel_size=7, padding=3, padding_mode='reflect'),
        )
        
        # Initialize output layer weights carefully
        for m in self.complex_synthesis.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, a=0.2, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
    def compute_expected_output_size(self, input_size, layers):
        """Calculate the expected output size after a series of conv/deconv layers."""
        size = input_size
        
        for i, (kernel_size, stride) in enumerate(zip(self.hparams.kernel_sizes, self.hparams.strides)):
            # For encoder (conv): output_size = (input_size + 2*padding - kernel_size) / stride + 1
            # With padding = kernel_size // 2
            size = (size + 2*(kernel_size // 2) - kernel_size) // stride + 1
            
        return size
        
    def forward(self, x):
        """Forward pass through the model."""
        batch_size = x.size(0)
        input_length = x.size(1)
        
        # Track input energy for later normalization
        input_energy = torch.mean(x**2)
        
        # Apply WST (returns real-valued tensor)
        x_wst = self.wst(x.unsqueeze(1))
        
        # Convert real-valued WST coefficients to complex
        # This step is crucial for phase preservation
        x_complex = self.real_to_complex(x_wst)
        
        # Log shape information
        B, C, T = x_wst.shape
        self.log("wst_channels", C, on_step=False, on_epoch=True)
        self.log("wst_timesteps", T, on_step=False, on_epoch=True)
        self.log("latent_dim", self.latent_dim, on_step=False, on_epoch=True)
        
        # Store skip connections
        skip_connections = []
        
        # Encoder
        for layer in self.encoder_layers:
            x_complex = layer(x_complex)
            skip_connections.append(x_complex)
        
        # Latent space
        z = self.latent_projection(x_complex)
        
        # Calculate and log latent space statistics
        z_mag = torch.abs(z).mean().item()
        self.log("latent_magnitude", z_mag, on_step=False, on_epoch=True)
        
        # Decoder with skip connections
        x_complex = z
        
        for i, layer in enumerate(self.decoder_layers):
            # Add skip connection before decoder layer
            skip_idx = len(skip_connections) - 1 - i
            if skip_idx >= 0:
                skip = skip_connections[skip_idx]
                
                # Project skip connection
                skip = self.skip_projections[skip_idx](skip)
                
                # Match dimensions if needed
                if skip.shape[2] != x_complex.shape[2]:
                    if skip.shape[2] > x_complex.shape[2]:
                        skip = skip[:, :, :x_complex.shape[2]]
                    else:
                        pad_size = x_complex.shape[2] - skip.shape[2]
                        skip_real = F.pad(skip.real, (0, pad_size), mode='reflect')
                        skip_imag = F.pad(skip.imag, (0, pad_size), mode='reflect')
                        skip = torch.complex(skip_real, skip_imag)
                
                # Stronger skip connections (0.8 → 1.0)
                x_complex = x_complex + 1.0 * skip
            
            # Apply decoder layer
            x_complex = layer(x_complex)
        
        # Output layer
        x_complex = self.output_layer(x_complex)
        
        # Convert to polar representation (magnitude/phase)
        x_polar = self.complex_to_real(x_complex)  # [B, 2*C, T] - magnitude and phase channels
        
        # Apply enhanced phase correction with stronger effect
        phase_corr = self.phase_correction(x_polar)
        x_polar_corrected = x_polar + 0.3 * phase_corr  # Increased from 0.2 to 0.3
        
        # Apply tapered window to reduce boundary artifacts
        batch_size, channels, time_steps = x_polar_corrected.shape
        window = torch.hann_window(time_steps, device=x_polar_corrected.device)
        window = window.view(1, 1, -1).expand(batch_size, channels, -1)
        # Apply window with a blend factor to prevent complete silence at boundaries
        x_polar_corrected = x_polar_corrected * (0.95 * window + 0.05)
        
        # Ensure output has correct length
        if x_polar_corrected.shape[2] != input_length:
            x_polar_corrected = self.final_upsampling(x_polar_corrected)
            # Additional interpolation if needed
            if x_polar_corrected.shape[2] != input_length:
                x_polar_corrected = F.interpolate(
                    x_polar_corrected, 
                    size=input_length, 
                    mode='linear', 
                    align_corners=False
                )
        
        # Convert polar representation back to waveform
        # Split magnitude and phase
        magnitude = x_polar_corrected[:, 0:1]  # First half of channels
        phase_scaled = x_polar_corrected[:, 1:2]  # Second half of channels
        
        # Convert scaled phase back to radians
        phase = phase_scaled * torch.pi
        
        # Convert to complex time domain signal using a more stable approach
        real_part = magnitude * torch.cos(phase)
        imag_part = magnitude * torch.sin(phase)
        
        # Enhanced waveform synthesis
        x_out = self.complex_synthesis(torch.cat([real_part, imag_part], dim=1))
        
        # Apply output energy normalization with stability constraint
        output_energy = torch.mean(x_out**2) + 1e-8
        energy_ratio = torch.sqrt(input_energy / output_energy).clamp(0.1, 10.0)  # Clamp to prevent extreme scaling
        x_out = x_out * self.output_scale * energy_ratio
        
        # Final squeeze
        x_out = x_out.squeeze(1)
            
        return x_out
    
    def training_step(self, batch, batch_idx):
        x = batch["audio"]
        x_hat = self(x)
        
        # Ensure shapes match
        if x_hat.shape != x.shape:
            x_hat = F.interpolate(x_hat.unsqueeze(1), size=x.shape[1], mode='linear', align_corners=False).squeeze(1)
        
        # L1 loss
        l1_loss = F.l1_loss(x_hat, x)
        
        # Multi-resolution STFT loss with enhanced phase handling
        stft_losses = []
        for n_fft in [256, 512, 1024, 2048, 4096]:  # Added 4096 for better low-frequency resolution
            hop_length = n_fft // 4
            win_length = n_fft
            window = torch.hann_window(win_length).to(x.device)
            
            # Compute STFTs
            X = torch.stft(
                x, 
                n_fft=n_fft, 
                hop_length=hop_length, 
                win_length=win_length, 
                window=window, 
                return_complex=True
            )
            X_hat = torch.stft(
                x_hat, 
                n_fft=n_fft, 
                hop_length=hop_length, 
                win_length=win_length, 
                window=window, 
                return_complex=True
            )
            
            # Magnitude spectral loss (log domain)
            mag_loss = F.mse_loss(
                torch.log1p(torch.abs(X) + 1e-8),
                torch.log1p(torch.abs(X_hat) + 1e-8)
            )
            
            # Enhanced phase loss with complex exponential representation
            phase_x = torch.angle(X)
            phase_x_hat = torch.angle(X_hat)
            
            # Calculate phase difference using complex exponential
            complex_exp_orig = torch.exp(1j * phase_x)
            complex_exp_recon = torch.exp(1j * phase_x_hat)
            
            # Complex MSE calculation
            complex_diff = complex_exp_orig - complex_exp_recon
            phase_loss = torch.mean(torch.abs(complex_diff)**2)
            
            # Weight by magnitude to focus on perceptually important frequencies
            magnitude_weight = torch.abs(X) / (torch.abs(X).sum(dim=(1, 2), keepdim=True) + 1e-8)
            weighted_phase_loss = torch.mean(magnitude_weight * torch.abs(complex_diff))
            
            # New: Add weighted phase consistency loss for harmonic frequencies
            freq_bins = torch.arange(n_fft//2 + 1, device=x.device).float()
            harmonic_weight = torch.exp(-0.5 * ((freq_bins - 100*n_fft/1024)/100)**2)  # Gaussian centered on expected harmonics
            harmonic_weight = harmonic_weight.view(1, -1, 1).expand_as(magnitude_weight)
            
            harmonic_phase_loss = torch.mean(harmonic_weight * magnitude_weight * torch.abs(complex_diff)**2)
            
            stft_losses.append(mag_loss + 0.5 * weighted_phase_loss + 0.3 * harmonic_phase_loss)
        
        # Average the multi-resolution losses
        spec_loss = sum(stft_losses) / len(stft_losses)
        
        # Additional envelope matching loss
        envelope_orig = torch.abs(torch.fft.rfft(x))
        envelope_recon = torch.abs(torch.fft.rfft(x_hat))
        envelope_loss = F.mse_loss(
            torch.log1p(envelope_orig + 1e-8),
            torch.log1p(envelope_recon + 1e-8)
        )
        
        # New: Harmonic structure loss via auto-correlation
        def autocorr(x):
            # Compute autocorrelation via FFT for efficiency
            x_fft = torch.fft.rfft(x)
            power = torch.abs(x_fft)**2
            corr = torch.fft.irfft(power)
            return corr
            
        autocorr_orig = autocorr(x)
        autocorr_recon = autocorr(x_hat)
        # Focus on the first 1000 lags which contain pitch information
        max_lag = min(1000, autocorr_orig.shape[-1]//4)
        harmonic_loss = F.mse_loss(
            autocorr_orig[..., :max_lag],
            autocorr_recon[..., :max_lag]
        )
        
        # Refined loss balance with additional components
        loss = 1.0 * l1_loss + 0.5 * spec_loss + 0.3 * envelope_loss + 0.2 * harmonic_loss
        
        self.log('train_loss', loss)
        self.log('train_l1_loss', l1_loss)
        self.log('train_spec_loss', spec_loss)
        self.log('train_envelope_loss', envelope_loss)
        self.log('train_harmonic_loss', harmonic_loss)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch["audio"]
        x_hat = self(x)
        
        # Ensure shapes match
        if x_hat.shape != x.shape:
            x_hat = F.interpolate(x_hat.unsqueeze(1), size=x.shape[1], mode='linear', align_corners=False).squeeze(1)
        
        # L1 loss
        l1_loss = F.l1_loss(x_hat, x)
        
        # Multi-resolution STFT loss with enhanced phase handling
        stft_losses = []
        for n_fft in [256, 512, 1024, 2048, 4096]:  # Added 4096 for better low-frequency resolution
            hop_length = n_fft // 4
            win_length = n_fft
            window = torch.hann_window(win_length).to(x.device)
            
            # Compute STFTs
            X = torch.stft(
                x, 
                n_fft=n_fft, 
                hop_length=hop_length, 
                win_length=win_length, 
                window=window, 
                return_complex=True
            )
            X_hat = torch.stft(
                x_hat, 
                n_fft=n_fft, 
                hop_length=hop_length, 
                win_length=win_length, 
                window=window, 
                return_complex=True
            )
            
            # Magnitude spectral loss (log domain)
            mag_loss = F.mse_loss(
                torch.log1p(torch.abs(X) + 1e-8),
                torch.log1p(torch.abs(X_hat) + 1e-8)
            )
            
            # Improved phase loss calculation
            phase_x = torch.angle(X)
            phase_x_hat = torch.angle(X_hat)
            
            # Calculate complex exponentials
            complex_exp_orig = torch.exp(1j * phase_x)
            complex_exp_recon = torch.exp(1j * phase_x_hat)
            
            # Manual MSE calculation for complex values
            complex_diff = complex_exp_orig - complex_exp_recon
            phase_loss = torch.mean(torch.abs(complex_diff)**2)
            
            # Weight by magnitude
            magnitude_weight = torch.abs(X) / (torch.abs(X).sum(dim=(1, 2), keepdim=True) + 1e-8)
            weighted_phase_loss = torch.mean(magnitude_weight * torch.abs(complex_diff))
            
            # Add harmonic frequency weighting
            freq_bins = torch.arange(n_fft//2 + 1, device=x.device).float()
            harmonic_weight = torch.exp(-0.5 * ((freq_bins - 100*n_fft/1024)/100)**2)
            harmonic_weight = harmonic_weight.view(1, -1, 1).expand_as(magnitude_weight)
            
            harmonic_phase_loss = torch.mean(harmonic_weight * magnitude_weight * torch.abs(complex_diff)**2)
            
            stft_losses.append(mag_loss + 0.5 * weighted_phase_loss + 0.3 * harmonic_phase_loss)
        
        # Average the multi-resolution losses
        spec_loss = sum(stft_losses) / len(stft_losses)
        
        # Additional envelope matching loss
        envelope_orig = torch.abs(torch.fft.rfft(x))
        envelope_recon = torch.abs(torch.fft.rfft(x_hat))
        envelope_loss = F.mse_loss(
            torch.log1p(envelope_orig + 1e-8),
            torch.log1p(envelope_recon + 1e-8)
        )
        
        # Harmonic structure loss via auto-correlation
        def autocorr(x):
            x_fft = torch.fft.rfft(x)
            power = torch.abs(x_fft)**2
            corr = torch.fft.irfft(power)
            return corr
            
        autocorr_orig = autocorr(x)
        autocorr_recon = autocorr(x_hat)
        max_lag = min(1000, autocorr_orig.shape[-1]//4)
        harmonic_loss = F.mse_loss(
            autocorr_orig[..., :max_lag],
            autocorr_recon[..., :max_lag]
        )
        
        # Balanced loss with additional components
        loss = 1.0 * l1_loss + 0.5 * spec_loss + 0.3 * envelope_loss + 0.2 * harmonic_loss
        
        self.log('val_loss', loss)
        self.log('val_l1_loss', l1_loss)
        self.log('val_spec_loss', spec_loss)
        self.log('val_envelope_loss', envelope_loss)
        self.log('val_harmonic_loss', harmonic_loss)
        
        # Calculate signal-to-noise ratio (SNR)
        with torch.no_grad():
            signal_power = torch.mean(x**2, dim=1)
            noise_power = torch.mean((x - x_hat)**2, dim=1)
            snr = 10 * torch.log10(signal_power / (noise_power + 1e-8))
            mean_snr = torch.mean(snr)
            self.log('val_snr_db', mean_snr)
            
            # Calculate perceptual metrics via spectral contrast
            def spectral_contrast(audio, n_fft=2048):
                S = torch.stft(
                    audio, 
                    n_fft=n_fft, 
                    hop_length=n_fft//4, 
                    win_length=n_fft, 
                    window=torch.hann_window(n_fft).to(audio.device), 
                    return_complex=True
                )
                mag = torch.abs(S)
                # Calculate contrast as ratio between adjacent frequency bins
                contrast = torch.log1p(mag[:, 1:, :]) - torch.log1p(mag[:, :-1, :])
                return contrast
                
            orig_contrast = spectral_contrast(x)
            recon_contrast = spectral_contrast(x_hat)
            contrast_sim = F.cosine_similarity(
                orig_contrast.reshape(orig_contrast.shape[0], -1),
                recon_contrast.reshape(recon_contrast.shape[0], -1),
                dim=1
            )
            self.log('val_spectral_contrast_sim', torch.mean(contrast_sim))
            
            # Add pitched harmonic detection metric
            def harmonic_ratio(audio, n_fft=2048):
                S = torch.stft(
                    audio, 
                    n_fft=n_fft, 
                    hop_length=n_fft//4, 
                    win_length=n_fft, 
                    window=torch.hann_window(n_fft).to(audio.device), 
                    return_complex=True
                )
                mag = torch.abs(S)
                
                # Calculate harmonic peaks vs noise floor ratio
                # Simple approach: use variance of frequency bins as proxy
                freq_variance = torch.var(mag, dim=1)
                mean_mag = torch.mean(mag, dim=1)
                
                # Higher ratio indicates more distinct harmonic structure
                harmonic_noise_ratio = freq_variance / (mean_mag + 1e-8)
                return harmonic_noise_ratio
                
            orig_harmonic = harmonic_ratio(x)
            recon_harmonic = harmonic_ratio(x_hat)
            
            # Calculate relative error in harmonic structure
            harmonic_error = torch.abs(orig_harmonic - recon_harmonic) / (orig_harmonic + 1e-8)
            self.log('val_harmonic_error', torch.mean(harmonic_error))
        
        # Log audio and visualizations for first batch only
        if batch_idx == 0:
            # Log a few samples
            num_samples = min(4, x.size(0))
            
            # Create waveform comparison figure
            fig, axes = plt.subplots(num_samples, 2, figsize=(12, 3 * num_samples))
            if num_samples == 1:
                axes = np.array([axes])
            
            for i in range(num_samples):
                # Original waveform
                axes[i, 0].plot(x[i].cpu().numpy())
                axes[i, 0].set_title(f"Original Audio {i+1}")
                axes[i, 0].set_ylim(-1.1, 1.1)
                
                # Reconstructed waveform
                axes[i, 1].plot(x_hat[i].cpu().numpy())
                axes[i, 1].set_title(f"Reconstructed Audio {i+1}")
                axes[i, 1].set_ylim(-1.1, 1.1)
            
            plt.tight_layout()
            self.logger.experiment.add_figure("Waveform Comparison", fig, self.global_step)
            plt.close(fig)
            
            # Create spectrograms
            fig, axes = plt.subplots(num_samples, 2, figsize=(12, 4 * num_samples))
            if num_samples == 1:
                axes = np.array([axes])
            
            for i in range(num_samples):
                # Original spectrogram
                window = torch.hann_window(1024).to(x.device)  # Increased from 512 to 1024
                X = torch.stft(
                    x[i], 
                    n_fft=1024, 
                    hop_length=256, 
                    win_length=1024, 
                    window=window, 
                    return_complex=True
                )
                X_db = 20 * torch.log10(torch.abs(X) + 1e-10)
                axes[i, 0].imshow(X_db.cpu().numpy(), aspect='auto', origin='lower')
                axes[i, 0].set_title(f"Original Spectrogram {i+1}")
                
                # Reconstructed spectrogram
                X_hat = torch.stft(
                    x_hat[i], 
                    n_fft=1024, 
                    hop_length=256, 
                    win_length=1024, 
                    window=window, 
                    return_complex=True
                )
                X_hat_db = 20 * torch.log10(torch.abs(X_hat) + 1e-10)
                axes[i, 1].imshow(X_hat_db.cpu().numpy(), aspect='auto', origin='lower')
                axes[i, 1].set_title(f"Reconstructed Spectrogram {i+1}")
            
            plt.tight_layout()
            self.logger.experiment.add_figure("Spectrogram Comparison", fig, self.global_step)
            plt.close(fig)
            
            # Log WST coefficients visualization
            with torch.no_grad():
                # Original audio WST
                x_wst = self.wst(x[:num_samples].unsqueeze(1))
                
                # Reconstructed audio WST
                x_hat_wst = self.wst(x_hat[:num_samples].unsqueeze(1))
                
                # Create figure for WST coefficient comparison
                fig, axes = plt.subplots(num_samples, 2, figsize=(12, 4 * num_samples))
                if num_samples == 1:
                    axes = np.array([axes])
                
                for i in range(num_samples):
                    # Original WST coefficients (mean across channels for visualization)
                    # No need for .abs() since x_wst is real-valued
                    wst_orig = x_wst[i].mean(dim=0).cpu().numpy()
                    im = axes[i, 0].imshow(wst_orig.reshape(-1, 1), aspect='auto', origin='lower')
                    axes[i, 0].set_title(f"Orig WST Mean {i+1}")
                    
                    # Reconstructed WST coefficients
                    wst_recon = x_hat_wst[i].mean(dim=0).cpu().numpy()
                    im = axes[i, 1].imshow(wst_recon.reshape(-1, 1), aspect='auto', origin='lower')
                    axes[i, 1].set_title(f"Recon WST Mean {i+1}")
                
                plt.tight_layout()
                self.logger.experiment.add_figure("WST Coefficient Comparison", fig, self.global_step)
                plt.close(fig)
            
            # Log raw audio
            for i in range(num_samples):
                self.logger.experiment.add_audio(
                    f"Original Audio {i+1}",
                    x[i].unsqueeze(0).cpu(),
                    self.global_step,
                    sample_rate=self.sample_rate
                )
                self.logger.experiment.add_audio(
                    f"Reconstructed Audio {i+1}",
                    x_hat[i].unsqueeze(0).cpu(),
                    self.global_step,
                    sample_rate=self.sample_rate
                )
        
        return loss
    
    def configure_optimizers(self):
        # AdamW with improved learning rate schedule and weight decay
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.learning_rate, 
            weight_decay=1e-5,
            betas=(0.9, 0.99)  # Higher beta2 for more stable updates
        )
        
        # One-cycle learning rate schedule for faster convergence
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.learning_rate * 1.5,
            total_steps=50000,  # Adjust based on your expected training steps
            pct_start=0.3,  # Spend 30% of time in warm-up
            div_factor=25,  # Initial LR = max_lr/div_factor
            final_div_factor=1000,  # Final LR = max_lr/final_div_factor
            anneal_strategy='cos'
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"
            }
        }