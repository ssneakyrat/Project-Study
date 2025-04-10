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
                 wst_J=8,  # Increased from 7 to 8 for better low-frequency resolution
                 wst_Q=16,  # Increased from 12 to 16 for finer frequency bands
                 channels=[128, 256, 512],  # Increased channel capacity
                 latent_dim=None,
                 kernel_sizes=[7, 7, 7],  # Increased from 5 to 7 for larger receptive field
                 strides=[2, 2, 2],
                 compression_factor=2,  # Reduced for better reconstruction quality
                 learning_rate=5e-4):  # Slightly increased for faster convergence
        """WST-based vocoder model with enhanced phase preservation."""
        super().__init__()
        self.save_hyperparameters()
        
        self.sample_rate = sample_rate
        self.learning_rate = learning_rate
        self.channels = channels
        self.compression_factor = compression_factor
        
        # WST layer with improved time-frequency resolution
        self.wst = WaveletScatteringTransform(
            J=wst_J,
            Q=wst_Q,
            T=sample_rate * 2,  # 2 seconds of audio
            max_order=1,  # Reduced to focus on primary time-frequency components
            out_type='array',
            oversampling=4  # Increased for better time resolution
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
            # Use square root scaling for more balanced compression
            self.latent_dim = int(math.sqrt(T * F) / compression_factor)
            self.latent_dim = max(64, self.latent_dim)  # Increased minimum size
            
            if latent_dim is not None:
                self.latent_dim = latent_dim
        
        # Store original length for final upsampling
        self.original_length = sample_rate * 2
        
        # Enhanced phase-aware real-to-complex conversion
        # Changed to 'analytic' which handles arbitrary channel counts
        self.real_to_complex = RealToComplex(mode='analytic')
        
        # Encoder layers
        self.encoder_layers = nn.ModuleList()
        
        current_time_dim = self.wst_time_dim
        for i, (out_channels, kernel_size, stride) in enumerate(zip(channels, kernel_sizes, strides)):
            padding = kernel_size // 2
            self.encoder_layers.append(
                nn.Sequential(
                    ComplexConv1d(
                        self.wst_channels if i == 0 else channels[i-1],
                        out_channels,
                        kernel_size,
                        stride=stride,
                        padding=padding
                    ),
                    ComplexBatchNorm1d(out_channels),
                    ComplexPReLU()
                )
            )
            current_time_dim = (current_time_dim + 2*padding - kernel_size) // stride + 1
        
        # Latent projection with larger kernel for better frequency integration
        self.latent_projection = nn.Sequential(
            ComplexConv1d(channels[-1], self.latent_dim, kernel_size=5, padding=2),
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
            padding = kernel_size // 2
            output_padding = stride - 1
            self.decoder_layers.append(
                nn.Sequential(
                    ComplexConvTranspose1d(
                        in_channels,
                        out_channels,
                        kernel_size,
                        stride=stride,
                        padding=padding,
                        output_padding=output_padding
                    ),
                    ComplexBatchNorm1d(out_channels),
                    ComplexPReLU()
                )
            )
        
        # Output layer with larger kernel for better time coherence
        self.output_layer = ComplexConvTranspose1d(
            channels[-1],
            1,  # Output channels
            kernel_size=9,  # Increased for larger receptive field
            stride=strides[0],
            padding=4,
            output_padding=strides[0] - 1
        )
        
        # Use enhanced polar representation for better phase preservation
        self.complex_to_real = ComplexToReal(mode='coherent_polar')
        
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
        
        # Enhanced final upsampling with anti-aliasing
        self.final_upsampling = nn.Sequential(
            nn.Upsample(size=sample_rate * 2, mode='linear', align_corners=False),
            nn.Conv1d(2, 32, kernel_size=9, padding=4, padding_mode='reflect'),
            nn.LeakyReLU(0.1),
            nn.Conv1d(32, 32, kernel_size=9, padding=4, padding_mode='reflect'),
            nn.LeakyReLU(0.1),
            nn.Conv1d(32, 2, kernel_size=9, padding=4, padding_mode='reflect'),
        )
        
        # Energy scaling factor for output normalization
        self.output_scale = nn.Parameter(torch.ones(1))
        
        # Enhanced phase correction network with multi-scale processing
        self.phase_correction = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=25, padding=12, padding_mode='reflect'),
            nn.LeakyReLU(0.1),
            nn.Conv1d(64, 64, kernel_size=25, padding=12, padding_mode='reflect'),
            nn.LeakyReLU(0.1),
            nn.Conv1d(64, 64, kernel_size=25, padding=12, padding_mode='reflect'),
            nn.LeakyReLU(0.1),
            nn.Conv1d(64, 2, kernel_size=25, padding=12, padding_mode='reflect'),
            nn.Tanh()  # Output correction factor between -1 and 1
        )
        
        # Multi-scale harmonic synthesis layer for better frequency coherence
        self.harmonic_synthesis = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(2, 64, kernel_size=k, padding=k//2, padding_mode='reflect'),
                nn.LeakyReLU(0.1),
                nn.Conv1d(64, 64, kernel_size=k, padding=k//2, padding_mode='reflect'),
                nn.LeakyReLU(0.1),
                nn.Conv1d(64, 2, kernel_size=k, padding=k//2, padding_mode='reflect'),
            ) for k in [3, 9, 27]  # Multi-scale kernels for different frequency ranges
        ])
        
        # Harmonic blending layer
        self.harmonic_blend = nn.Sequential(
            nn.Conv1d(2*3, 64, kernel_size=9, padding=4, padding_mode='reflect'),
            nn.LeakyReLU(0.1),
            nn.Conv1d(64, 32, kernel_size=9, padding=4, padding_mode='reflect'),
            nn.LeakyReLU(0.1),
            nn.Conv1d(32, 1, kernel_size=9, padding=4, padding_mode='reflect'),
        )
        
        # Initialize output layer weights carefully
        for m in self.harmonic_blend.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, a=0.1, nonlinearity='leaky_relu')
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
        
        # Convert real-valued WST coefficients to complex with enhanced phase handling
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
        
        # Decoder with enhanced skip connections
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
                
                # Progressive skip connection strength
                # Stronger for later layers to emphasize high-level structure
                skip_strength = 0.8 + 0.2 * i / len(self.decoder_layers)
                x_complex = x_complex + skip_strength * skip
            
            # Apply decoder layer
            x_complex = layer(x_complex)
        
        # Output layer
        x_complex = self.output_layer(x_complex)
        
        # Convert to enhanced polar representation (magnitude/phase)
        x_polar = self.complex_to_real(x_complex)  # [B, 2*C, T] - magnitude and phase channels
        
        # Apply enhanced phase correction with stronger effect
        phase_corr = self.phase_correction(x_polar)
        x_polar_corrected = x_polar + 0.5 * phase_corr  # Increased to 0.5
        
        # Apply multi-scale harmonic enhancement for better frequency coherence
        harmonic_outputs = []
        for harmonic_layer in self.harmonic_synthesis:
            harmonic_outputs.append(harmonic_layer(x_polar_corrected))
            
        # Concatenate multi-scale outputs
        multi_scale_output = torch.cat(harmonic_outputs, dim=1)
        
        # Blend harmonics for final output
        harmonic_blended = self.harmonic_blend(multi_scale_output)
        
        # Apply tapered window to reduce boundary artifacts
        batch_size, _, time_steps = harmonic_blended.shape
        
        # Use less aggressive window (cosine) to preserve more signal at boundaries
        window = torch.cos(torch.linspace(-torch.pi/2, torch.pi/2, time_steps, device=harmonic_blended.device)) * 0.5 + 0.5
        window = window.view(1, 1, -1).expand(batch_size, 1, -1)
        
        # Apply window with a blend factor to prevent complete silence at boundaries
        harmonic_blended = harmonic_blended * (0.98 * window + 0.02)
        
        # Ensure output has correct length and handle potential errors
        if harmonic_blended.shape[2] != input_length:
            try:
                # Use improved upsampling for better temporal quality
                harmonic_blended = F.interpolate(
                    harmonic_blended, 
                    size=input_length, 
                    mode='linear', 
                    align_corners=False
                )
            except RuntimeError:
                # Fallback to simpler approach if there's an issue
                harmonic_blended = F.interpolate(
                    harmonic_blended, 
                    size=input_length, 
                    mode='nearest'
                )
        
        # Apply output energy normalization with improved stability
        output_energy = torch.mean(harmonic_blended**2) + 1e-8
        energy_ratio = torch.sqrt(input_energy / output_energy).clamp(0.5, 2.0)  # More conservative clamping
        x_out = harmonic_blended * self.output_scale * energy_ratio
        
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
        for n_fft in [128, 256, 512, 1024, 2048, 4096]:  # Added 128 for better high-frequency resolution
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
            
            # Magnitude spectral loss (log domain with adaptive scaling)
            mag_orig = torch.abs(X)
            mag_recon = torch.abs(X_hat)
            
            # Log-magnitude loss with epsilon to prevent log(0)
            log_mag_loss = F.mse_loss(
                torch.log1p(mag_orig + 1e-7),
                torch.log1p(mag_recon + 1e-7)
            )
            
            # Linear magnitude loss (more effective for high energy regions)
            lin_mag_loss = F.mse_loss(mag_orig, mag_recon)
            
            # Combine both for balanced learning across energy levels
            mag_loss = 0.5 * log_mag_loss + 0.5 * lin_mag_loss
            
            # Enhanced phase loss with complex expectation approach
            # 1. Extract phase angles
            phase_orig = torch.angle(X)
            phase_recon = torch.angle(X_hat)
            
            # 2. Compute complex exponential differences (more robust than direct angle difference)
            complex_exp_orig = torch.exp(1j * phase_orig)
            complex_exp_recon = torch.exp(1j * phase_recon)
            
            # 3. Compute phase coherence loss
            # This yields 0 when phases match perfectly and 2 when they're opposite
            phase_coherence_loss = torch.abs(complex_exp_orig - complex_exp_recon)**2
            
            # 4. Weight by magnitude to focus on perceptually important frequencies
            mag_weight = mag_orig / (torch.sum(mag_orig, dim=(1, 2), keepdim=True) + 1e-8)
            weighted_phase_loss = torch.mean(mag_weight * phase_coherence_loss)
            
            # 5. Add harmonic structure constraint using frequency relationships
            freq_bins = torch.arange(n_fft//2 + 1, device=x.device).float()
            
            # Focus more on fundamental frequencies (80-800 Hz region)
            # Convert Hz to FFT bin indices
            f_min, f_max = 80, 800
            bin_min = int(f_min * n_fft / self.sample_rate)
            bin_max = int(f_max * n_fft / self.sample_rate)
            
            # Create harmonic weighting
            harmonic_weight = torch.zeros_like(freq_bins)
            harmonic_weight[bin_min:bin_max+1] = 1.0
            
            # Also weight the harmonics
            for harmonic in range(2, 6):  # 2nd to 5th harmonics
                h_bin_min = bin_min * harmonic
                h_bin_max = min(bin_max * harmonic, n_fft//2)
                if h_bin_max > h_bin_min:
                    # Decreasing weight for higher harmonics
                    harmonic_weight[h_bin_min:h_bin_max+1] = 1.0 / harmonic
            
            # Reshape and normalize
            harmonic_weight = harmonic_weight.view(1, -1, 1).expand_as(mag_weight)
            
            # Compute harmonic phase consistency loss
            harmonic_phase_loss = torch.mean(harmonic_weight * mag_weight * phase_coherence_loss)
            
            # Combine losses with balanced weighting
            stft_loss = mag_loss + 0.5 * weighted_phase_loss + 0.5 * harmonic_phase_loss
            stft_losses.append(stft_loss)
        
        # Average the multi-resolution losses
        spec_loss = sum(stft_losses) / len(stft_losses)
        
        # Add temporal envelope matching loss
        # Use Hilbert envelope for better amplitude envelope matching
        def hilbert_envelope(x):
            # Compute analytic signal via FFT
            X = torch.fft.rfft(x)
            h = torch.ones_like(X)
            if h.size(-1) > 1:
                h[:, 1:] = 2.0
            X_analytic = X * h
            x_analytic = torch.fft.irfft(X_analytic, n=x.size(-1))
            
            # Compute envelope
            x_hilbert = torch.zeros_like(x)
            x_analytical = torch.complex(x, x_hilbert)
            return torch.abs(x_analytical)
        
        # Compute temporal envelopes
        env_orig = hilbert_envelope(x)
        env_recon = hilbert_envelope(x_hat)
        
        # Envelope matching loss in both domains
        envelope_loss = 0.5 * F.mse_loss(env_orig, env_recon) + \
                        0.5 * F.mse_loss(torch.log1p(env_orig + 1e-7), torch.log1p(env_recon + 1e-7))
        
        # Enhanced harmonic structure loss via auto-correlation with longer lags
        def autocorr(x):
            # Compute autocorrelation via FFT for efficiency
            x_fft = torch.fft.rfft(x)
            power = torch.abs(x_fft)**2
            corr = torch.fft.irfft(power)
            return corr
            
        autocorr_orig = autocorr(x)
        autocorr_recon = autocorr(x_hat)
        
        # Focus on the lags containing pitch information (up to ~500Hz fundamental)
        # For 16kHz audio, 16000/500 = 32 samples per cycle
        max_lag = min(2000, autocorr_orig.shape[-1]//2)
        min_lag = 16  # Skip very short lags (high-frequency noise)
        
        # Weight lags by importance (more weight to lower frequencies/longer lags)
        lag_weights = torch.linspace(0.5, 1.0, max_lag - min_lag, device=x.device)
        
        # Apply weights to autocorrelation difference
        autocorr_diff = (autocorr_orig[:, min_lag:max_lag] - autocorr_recon[:, min_lag:max_lag]) ** 2
        weighted_autocorr_diff = autocorr_diff * lag_weights.view(1, -1)
        
        harmonic_loss = torch.mean(weighted_autocorr_diff)
        
        # Add periodicity consistency loss
        # This loss encourages preserving periodic patterns at fundamental frequencies
        def get_period_consistency(x, min_period=32, max_period=320):
            """Compute consistency of signal periodicity"""
            consistencies = []
            
            # For each potential period, compute consistency
            for period in range(min_period, max_period, 8):  # Step by 8 for efficiency
                # Shift signal by period
                x_shifted = torch.roll(x, shifts=period, dims=-1)
                
                # Compute consistency (lower value = more periodic)
                period_diff = torch.mean((x - x_shifted)**2, dim=-1)
                consistencies.append(period_diff)
                
            # Stack and get the period with minimum difference (most consistent)
            return torch.stack(consistencies, dim=1)
        
        # Get periodicity consistency for both signals
        period_orig = get_period_consistency(x)
        period_recon = get_period_consistency(x_hat)
        
        # Period consistency loss - encourage similar periodicity behavior
        period_loss = F.mse_loss(period_orig, period_recon)
        
        # Refined loss balance with additional components
        loss = 1.0 * l1_loss + 0.8 * spec_loss + 0.5 * envelope_loss + 0.5 * harmonic_loss + 0.5 * period_loss
        
        self.log('train_loss', loss)
        self.log('train_l1_loss', l1_loss)
        self.log('train_spec_loss', spec_loss)
        self.log('train_envelope_loss', envelope_loss)
        self.log('train_harmonic_loss', harmonic_loss)
        self.log('train_period_loss', period_loss)
        
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
        for n_fft in [128, 256, 512, 1024, 2048, 4096]:  # Added 128 for better high-frequency resolution
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
            
            # Combined magnitude loss (log and linear domains)
            mag_orig = torch.abs(X)
            mag_recon = torch.abs(X_hat)
            
            log_mag_loss = F.mse_loss(
                torch.log1p(mag_orig + 1e-7),
                torch.log1p(mag_recon + 1e-7)
            )
            lin_mag_loss = F.mse_loss(mag_orig, mag_recon)
            mag_loss = 0.5 * log_mag_loss + 0.5 * lin_mag_loss
            
            # Enhanced phase loss with complex exponential representation
            phase_orig = torch.angle(X)
            phase_recon = torch.angle(X_hat)
            
            complex_exp_orig = torch.exp(1j * phase_orig)
            complex_exp_recon = torch.exp(1j * phase_recon)
            
            phase_coherence_loss = torch.abs(complex_exp_orig - complex_exp_recon)**2
            
            # Weight by magnitude
            mag_weight = mag_orig / (torch.sum(mag_orig, dim=(1, 2), keepdim=True) + 1e-8)
            weighted_phase_loss = torch.mean(mag_weight * phase_coherence_loss)
            
            # Add harmonic frequency weighting
            freq_bins = torch.arange(n_fft//2 + 1, device=x.device).float()
            
            # Focus on fundamentals and harmonics
            f_min, f_max = 80, 800
            bin_min = int(f_min * n_fft / self.sample_rate)
            bin_max = int(f_max * n_fft / self.sample_rate)
            
            harmonic_weight = torch.zeros_like(freq_bins)
            harmonic_weight[bin_min:bin_max+1] = 1.0
            
            for harmonic in range(2, 6):
                h_bin_min = bin_min * harmonic
                h_bin_max = min(bin_max * harmonic, n_fft//2)
                if h_bin_max > h_bin_min:
                    harmonic_weight[h_bin_min:h_bin_max+1] = 1.0 / harmonic
            
            harmonic_weight = harmonic_weight.view(1, -1, 1).expand_as(mag_weight)
            
            harmonic_phase_loss = torch.mean(harmonic_weight * mag_weight * phase_coherence_loss)
            
            stft_losses.append(mag_loss + 0.5 * weighted_phase_loss + 0.5 * harmonic_phase_loss)
        
        # Average the multi-resolution losses
        spec_loss = sum(stft_losses) / len(stft_losses)
        
        # Add envelope matching loss using safer Hilbert envelope
        def hilbert_envelope(x):
            try:
                # Try the proper Hilbert transform first
                X = torch.fft.rfft(x)
                h = torch.ones_like(X)
                if h.size(-1) > 1:
                    h[:, 1:] = 2.0
                X_analytic = X * h
                x_analytic = torch.fft.irfft(X_analytic, n=x.size(-1))
                
                # Create the analytic signal
                x_hilbert = torch.zeros_like(x)
                x_analytical = torch.complex(x, x_hilbert)
                return torch.abs(x_analytical)
            except Exception:
                # Fallback to simple envelope extraction
                # Use a sliding window max approach
                x_abs = torch.abs(x)
                kernel_size = 101
                padding = kernel_size // 2
                
                # Use max pooling as envelope follower
                x_env = F.max_pool1d(
                    x_abs.view(-1, 1, x.size(-1)), 
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding
                )
                return x_env.view(x.shape)
        
        env_orig = hilbert_envelope(x)
        env_recon = hilbert_envelope(x_hat)
        
        envelope_loss = 0.5 * F.mse_loss(env_orig, env_recon) + \
                        0.5 * F.mse_loss(torch.log1p(env_orig + 1e-7), torch.log1p(env_recon + 1e-7))
        
        # Enhanced harmonic structure loss via auto-correlation
        def autocorr(x):
            x_fft = torch.fft.rfft(x)
            power = torch.abs(x_fft)**2
            corr = torch.fft.irfft(power)
            return corr
            
        autocorr_orig = autocorr(x)
        autocorr_recon = autocorr(x_hat)
        
        max_lag = min(2000, autocorr_orig.shape[-1]//2)
        min_lag = 16
        
        lag_weights = torch.linspace(0.5, 1.0, max_lag - min_lag, device=x.device)
        autocorr_diff = (autocorr_orig[:, min_lag:max_lag] - autocorr_recon[:, min_lag:max_lag]) ** 2
        weighted_autocorr_diff = autocorr_diff * lag_weights.view(1, -1)
        
        harmonic_loss = torch.mean(weighted_autocorr_diff)
        
        # Add periodicity consistency loss
        def get_period_consistency(x, min_period=32, max_period=320):
            consistencies = []
            for period in range(min_period, max_period, 8):
                x_shifted = torch.roll(x, shifts=period, dims=-1)
                period_diff = torch.mean((x - x_shifted)**2, dim=-1)
                consistencies.append(period_diff)
            return torch.stack(consistencies, dim=1)
        
        period_orig = get_period_consistency(x)
        period_recon = get_period_consistency(x_hat)
        period_loss = F.mse_loss(period_orig, period_recon)
        
        # Balanced loss with additional components
        loss = 1.0 * l1_loss + 0.8 * spec_loss + 0.5 * envelope_loss + 0.5 * harmonic_loss + 0.5 * period_loss
        
        self.log('val_loss', loss)
        self.log('val_l1_loss', l1_loss)
        self.log('val_spec_loss', spec_loss)
        self.log('val_envelope_loss', envelope_loss)
        self.log('val_harmonic_loss', harmonic_loss)
        self.log('val_period_loss', period_loss)
        
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
            
            # Enhanced pitched harmonic detection metric
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
                
                # Calculate harmonic structure metrics
                
                # 1. Spectral flatness - ratio of geometric mean to arithmetic mean
                # Lower value indicates more distinct harmonics
                epsilon = 1e-8
                log_mag = torch.log(mag + epsilon)
                spectral_flatness = torch.exp(torch.mean(log_mag, dim=1)) / (torch.mean(mag, dim=1) + epsilon)
                
                # 2. Spectral crest - ratio of max to mean
                # Higher value indicates more distinct peaks
                spectral_crest = torch.max(mag, dim=1)[0] / (torch.mean(mag, dim=1) + epsilon)
                
                # Combine metrics (higher value = more harmonic structure)
                harmonic_metric = spectral_crest * (1 - spectral_flatness)
                
                return torch.mean(harmonic_metric, dim=-1)
                
            orig_harmonic = harmonic_ratio(x)
            recon_harmonic = harmonic_ratio(x_hat)
            
            # Calculate relative harmonic structure error
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
                window = torch.hann_window(1024).to(x.device)
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
        # AdamW with improved learning rate schedule
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.learning_rate, 
            weight_decay=1e-6,  # Increased for better regularization
            betas=(0.9, 0.999)  # Standard betas for stability
        )
        
        # One-cycle learning rate schedule with cosine annealing
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.learning_rate * 2.0,  # Higher peak learning rate
            total_steps=50000,
            pct_start=0.2,  # Faster warm-up (20% of training)
            div_factor=25,
            final_div_factor=1000,
            anneal_strategy='cos'
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"
            }
        }