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
                 wst_J=6,  # Reduced J for better time resolution
                 wst_Q=8,
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
        
        # WST layer with reduced J parameter for better time resolution
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
        
        # Improved phase handling with analytic signal approach
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
        
        # Latent projection
        self.latent_projection = ComplexConv1d(
            channels[-1],
            self.latent_dim,
            kernel_size=1
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
        
        # Output layer
        self.output_layer = ComplexConvTranspose1d(
            channels[-1],
            1,  # Output channels
            kernel_size=kernel_sizes[0],
            stride=strides[0],
            padding=kernel_sizes[0] // 2,
            output_padding=strides[0] - 1
        )
        
        # *** CRITICAL FIX: Use polar mode to preserve both magnitude and phase ***
        # This provides explicit phase preservation rather than just using magnitude
        self.complex_to_real = ComplexToReal(mode='polar')
        
        # Skip projections from encoder outputs to decoder inputs
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
        # *** CRITICAL FIX: Separate upsamplers for magnitude and phase ***
        self.final_upsampling = nn.Sequential(
            nn.Upsample(size=sample_rate * 2, mode='linear', align_corners=False),
            nn.Conv1d(2, 2, kernel_size=7, padding=3, padding_mode='reflect'),
            nn.LeakyReLU(0.2),
            nn.Conv1d(2, 2, kernel_size=7, padding=3, padding_mode='reflect'),
        )
        
        # Energy scaling factor for output normalization
        self.output_scale = nn.Parameter(torch.ones(1))
        
        # Improved phase correction network with frequency-dependent processing
        self.phase_correction = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=15, padding=7, padding_mode='reflect'),
            nn.LeakyReLU(0.2),
            nn.Conv1d(16, 16, kernel_size=15, padding=7, padding_mode='reflect'),
            nn.LeakyReLU(0.2),
            nn.Conv1d(16, 2, kernel_size=15, padding=7, padding_mode='reflect'),
            nn.Tanh()  # Output correction factor between -1 and 1
        )
        
        # *** CRITICAL FIX: Complex synthesis layer to convert back from magnitude/phase to audio ***
        self.complex_synthesis = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=7, padding=3),
            nn.LeakyReLU(0.2),
            nn.Conv1d(32, 32, kernel_size=7, padding=3),
            nn.LeakyReLU(0.2),
            nn.Conv1d(32, 1, kernel_size=7, padding=3),
        )
        
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
                
                # *** CRITICAL FIX: Stronger skip connections (0.8 → 1.0) ***
                x_complex = x_complex + 1.0 * skip
            
            # Apply decoder layer
            x_complex = layer(x_complex)
        
        # Output layer
        x_complex = self.output_layer(x_complex)
        
        # *** CRITICAL FIX: Convert to polar representation (magnitude/phase) ***
        # This gives us explicit control over both magnitude and phase
        x_polar = self.complex_to_real(x_complex)  # [B, 2*C, T] - magnitude and phase channels
        
        # Apply phase correction network
        phase_corr = self.phase_correction(x_polar)
        x_polar_corrected = x_polar + 0.2 * phase_corr  # Increased weight for phase correction
        
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
        
        # *** CRITICAL FIX: Convert polar representation back to waveform ***
        # Split magnitude and phase
        magnitude = x_polar_corrected[:, 0:1]  # First half of channels
        phase_scaled = x_polar_corrected[:, 1:2]  # Second half of channels
        
        # Convert scaled phase back to radians
        phase = phase_scaled * torch.pi
        
        # Convert to complex time domain signal
        real_part = magnitude * torch.cos(phase)
        imag_part = magnitude * torch.sin(phase)
        
        # Enhanced waveform synthesis
        x_out = self.complex_synthesis(torch.cat([real_part, imag_part], dim=1))
        
        # Apply output energy normalization
        output_energy = torch.mean(x_out**2) + 1e-8
        energy_ratio = torch.sqrt(input_energy / output_energy)
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
        
        # Multi-resolution STFT loss
        stft_losses = []
        for n_fft in [256, 512, 1024, 2048]:  # Added 2048 for better low-frequency resolution
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
            
            # *** CRITICAL FIX: Improved phase loss ***
            phase_x = torch.angle(X)
            phase_x_hat = torch.angle(X_hat)
            
            # Calculate phase difference with complex exponential (using manual calculation)
            # e^(iθ1) - e^(iθ2) directly captures phase difference
            complex_exp_orig = torch.exp(1j * phase_x)
            complex_exp_recon = torch.exp(1j * phase_x_hat)
            
            # Manual MSE calculation for complex values: mean(|z1 - z2|²)
            complex_diff = complex_exp_orig - complex_exp_recon
            phase_loss = torch.mean(torch.abs(complex_diff)**2)
            
            # Weight by magnitude to focus on perceptually important frequencies
            magnitude_weight = torch.abs(X) / (torch.abs(X).sum(dim=(1, 2), keepdim=True) + 1e-8)
            weighted_phase_loss = torch.mean(magnitude_weight * torch.abs(complex_diff))
            
            stft_losses.append(mag_loss + 0.5 * weighted_phase_loss)  # Increased phase weight
        
        # Average the multi-resolution losses
        spec_loss = sum(stft_losses) / len(stft_losses)
        
        # Additional envelope matching loss
        envelope_orig = torch.abs(torch.fft.rfft(x))
        envelope_recon = torch.abs(torch.fft.rfft(x_hat))
        envelope_loss = F.mse_loss(
            torch.log1p(envelope_orig + 1e-8),
            torch.log1p(envelope_recon + 1e-8)
        )
        
        # *** CRITICAL FIX: Balanced loss with higher weights for phase and envelope ***
        loss = l1_loss + 0.5 * spec_loss + 0.3 * envelope_loss
        
        self.log('train_loss', loss)
        self.log('train_l1_loss', l1_loss)
        self.log('train_spec_loss', spec_loss)
        self.log('train_envelope_loss', envelope_loss)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch["audio"]
        x_hat = self(x)
        
        # Ensure shapes match
        if x_hat.shape != x.shape:
            x_hat = F.interpolate(x_hat.unsqueeze(1), size=x.shape[1], mode='linear', align_corners=False).squeeze(1)
        
        # L1 loss
        l1_loss = F.l1_loss(x_hat, x)
        
        # Multi-resolution STFT loss
        stft_losses = []
        for n_fft in [256, 512, 1024, 2048]:  # Added 2048 for better low-frequency resolution
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
            
            # Improved phase loss calculation using manual MSE for complex values
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
            
            stft_losses.append(mag_loss + 0.5 * weighted_phase_loss)
        
        # Average the multi-resolution losses
        spec_loss = sum(stft_losses) / len(stft_losses)
        
        # Additional envelope matching loss
        envelope_orig = torch.abs(torch.fft.rfft(x))
        envelope_recon = torch.abs(torch.fft.rfft(x_hat))
        envelope_loss = F.mse_loss(
            torch.log1p(envelope_orig + 1e-8),
            torch.log1p(envelope_recon + 1e-8)
        )
        
        # Balanced loss with higher weights for phase and envelope
        loss = l1_loss + 0.5 * spec_loss + 0.3 * envelope_loss
        
        self.log('val_loss', loss)
        self.log('val_l1_loss', l1_loss)
        self.log('val_spec_loss', spec_loss)
        self.log('val_envelope_loss', envelope_loss)
        
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
                window = torch.hann_window(512).to(x.device)
                X = torch.stft(
                    x[i], 
                    n_fft=512, 
                    hop_length=128, 
                    win_length=512, 
                    window=window, 
                    return_complex=True
                )
                X_db = 20 * torch.log10(torch.abs(X) + 1e-10)
                axes[i, 0].imshow(X_db.cpu().numpy(), aspect='auto', origin='lower')
                axes[i, 0].set_title(f"Original Spectrogram {i+1}")
                
                # Reconstructed spectrogram
                X_hat = torch.stft(
                    x_hat[i], 
                    n_fft=512, 
                    hop_length=128, 
                    win_length=512, 
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
        # AdamW with improved learning rate schedule
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch"
            }
        }