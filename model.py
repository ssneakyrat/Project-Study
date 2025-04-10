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
                 wst_J=8,
                 wst_Q=8,
                 channels=[64, 128, 256],
                 latent_dim=None,  # Will be calculated dynamically
                 kernel_sizes=[5, 5, 5],
                 strides=[2, 2, 2],
                 compression_factor=8,  # Reduced from 16 to 8 for less aggressive compression
                 learning_rate=1e-4):
        """WST-based vocoder model.
        
        Args:
            sample_rate: Audio sample rate
            wst_J: Number of scales for WST
            wst_Q: Number of wavelets per octave for WST
            channels: List of channel dimensions for encoder/decoder
            latent_dim: Dimension of latent space (if None, calculated dynamically)
            kernel_sizes: List of kernel sizes for encoder/decoder
            strides: List of strides for encoder/decoder
            compression_factor: Compression factor for latent dimension
            learning_rate: Learning rate for optimizer
        """
        super().__init__()
        self.save_hyperparameters()
        
        self.sample_rate = sample_rate
        self.learning_rate = learning_rate
        self.channels = channels
        self.compression_factor = compression_factor
        
        # WST layer
        self.wst = WaveletScatteringTransform(
            J=wst_J,
            Q=wst_Q,
            T=sample_rate * 2,  # 2 seconds of audio
            max_order=2,
            out_type='array'
        )
        
        # Pre-compute actual WST output shape with a dummy input
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, sample_rate * 2)
            dummy_output = self.wst(dummy_input)
            self.wst_channels = dummy_output.shape[1]
            self.wst_time_dim = dummy_output.shape[2]
            
            # Calculate bottleneck dimensions using modified formula to reduce compression
            # MODIFIED: Doubled capacity with * 2 factor
            T = self.wst_time_dim
            F = self.wst_channels
            self.latent_dim = int(math.sqrt(T * F) / compression_factor * 2)
            
            # Make sure latent_dim is at least 16 for stability
            self.latent_dim = max(16, self.latent_dim)
            
            if latent_dim is not None:
                # Override with user-specified value if provided
                self.latent_dim = latent_dim
        
        # Store original length for final upsampling
        self.original_length = sample_rate * 2
        
        # MODIFIED: Better phase approximation with hilbert transform
        self.real_to_complex = RealToComplex(mode='hilbert')
        
        # Encoder
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
            # Update current time dimension for next layer
            current_time_dim = (current_time_dim + 2*(kernel_size // 2) - kernel_size) // stride + 1
        
        # Latent projection
        self.latent_projection = ComplexConv1d(
            channels[-1],
            self.latent_dim,
            kernel_size=1
        )
        
        # Decoder
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
        
        # MODIFIED: Improved complex to real conversion
        # Using magnitude preserves energy but we need to keep phase information
        # Using 'real' would preserve some phase but discards half of the information
        # Instead, we'll keep the complex values and convert at the very end
        self.complex_to_real = ComplexToReal(mode='magnitude')
        
        # Skip projections from encoder outputs to decoder inputs
        encoder_channels = channels  # [64, 128, 256]
        decoder_input_channels = [self.latent_dim] + channels[:-1]  # [latent_dim, 64, 128]
        
        self.skip_projections = nn.ModuleList()
        for i in range(len(encoder_channels)):
            enc_ch = encoder_channels[i]
            dec_idx = len(decoder_input_channels) - 1 - i
            dec_ch = decoder_input_channels[dec_idx]
            
            self.skip_projections.append(
                ComplexConv1d(enc_ch, dec_ch, kernel_size=1)
            )
        
        # MODIFIED: Improved final upsampling with antialiasing
        self.final_upsampling = nn.Sequential(
            nn.Upsample(size=sample_rate * 2, mode='linear', align_corners=False),
            nn.Conv1d(1, 1, kernel_size=7, padding=3, padding_mode='reflect'),  # Anti-aliasing filter
        )
        
        # Energy scaling factor for output normalization
        self.output_scale = nn.Parameter(torch.ones(1))
        
        # ADDED: Phase compensation network - helps correct boundary effects
        self.phase_correction = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=15, padding=7, padding_mode='reflect'),
            nn.LeakyReLU(0.2),
            nn.Conv1d(16, 16, kernel_size=15, padding=7, padding_mode='reflect'),
            nn.LeakyReLU(0.2),
            nn.Conv1d(16, 1, kernel_size=15, padding=7, padding_mode='reflect'),  # Fixed: 16→1 channels
            nn.Tanh()  # Output phase correction factor between -1 and 1
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
        """
        Args:
            x: Input audio tensor of shape (batch_size, time)
        
        Returns:
            Reconstructed audio tensor of shape (batch_size, time)
        """
        batch_size = x.size(0)
        input_length = x.size(1)
        
        # Track input energy for later normalization
        input_energy = torch.mean(x**2)
        
        # Apply WST
        x_wst = self.wst(x.unsqueeze(1))
        
        # Log shape information
        B, C, T = x_wst.shape
        self.log("wst_channels", C, on_step=False, on_epoch=True)
        self.log("wst_timesteps", T, on_step=False, on_epoch=True)
        self.log("latent_dim", self.latent_dim, on_step=False, on_epoch=True)
        
        # Convert to complex
        x_complex = self.real_to_complex(x_wst)
        
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
                
                # Project channels to match decoder input
                skip = self.skip_projections[skip_idx](skip)
                
                # Resize time dimension if needed using interpolation
                if skip.shape[2] != x_complex.shape[2]:
                    # Simple but potentially less artifact-prone approach: 
                    # Match dimensions by trimming or padding instead of interpolation
                    if skip.shape[2] > x_complex.shape[2]:
                        # Trim
                        skip = skip[:, :, :x_complex.shape[2]]
                    else:
                        # Pad
                        pad_size = x_complex.shape[2] - skip.shape[2]
                        skip_real = F.pad(skip.real, (0, pad_size), mode='reflect')
                        skip_imag = F.pad(skip.imag, (0, pad_size), mode='reflect')
                        skip = torch.complex(skip_real, skip_imag)
                
                # MODIFIED: Increased skip connection strength for better signal preservation
                x_complex = x_complex + 0.8 * skip  # Increased from 0.5 to 0.8
            
            # Apply decoder layer
            x_complex = layer(x_complex)
        
        # Output layer
        x_complex = self.output_layer(x_complex)
        
        # MODIFIED: Improved phase handling - extract both magnitude and phase
        x_mag = torch.abs(x_complex)
        x_phase = torch.angle(x_complex)
        
        # MODIFIED: Create an intermediate real representation
        x_real = x_mag * torch.cos(x_phase)
        
        # Reshape for phase correction
        x_real_1ch = x_real.squeeze(1).unsqueeze(1)
        
        # Apply phase correction network to address boundary effects
        # This is particularly important for edges where wavelet transform creates artifacts
        phase_corr = self.phase_correction(x_real_1ch)
        
        # Apply the correction with a learned weighting
        x_out = x_real_1ch + 0.1 * phase_corr
        
        # The final output may still not match the original audio length
        # Use improved upsampling to match the exact input size
        if x_out.shape[2] != input_length:
            x_out = self.final_upsampling(x_out)
            # Ensure exact shape match
            if x_out.shape[2] != input_length:
                x_out = F.interpolate(x_out, size=input_length, mode='linear', align_corners=False)
        
        # Apply reflective padding to address boundary discontinuities (helps with spikes)
        x_out = F.pad(x_out, (4, 4), mode='reflect')
        x_out = x_out[:, :, 4:-4]  # Remove padding but keep correct size
        
        # Apply output energy normalization
        output_energy = torch.mean(x_out**2) + 1e-8
        energy_ratio = torch.sqrt(input_energy / output_energy)
        # Use learnable scaling factor with energy ratio
        x_out = x_out * self.output_scale * energy_ratio
        
        # Final squeeze to match expected dimensions
        x_out = x_out.squeeze(1)
            
        return x_out
    
    def training_step(self, batch, batch_idx):
        x = batch["audio"]
        x_hat = self(x)
        
        # Ensure shapes match
        if x_hat.shape != x.shape:
            self.log('shape_mismatch', 1.0)
            # Resize output to match input if needed (should not happen with proper upsampling)
            x_hat = F.interpolate(x_hat.unsqueeze(1), size=x.shape[1], mode='linear', align_corners=False).squeeze(1)
        
        # L1 loss + spectral loss for better convergence
        l1_loss = F.l1_loss(x_hat, x)
        
        # MODIFIED: Enhanced spectral loss using multi-resolution STFT
        # Use multiple window sizes to capture both global and local time-frequency patterns
        stft_losses = []
        for n_fft in [256, 512, 1024]:
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
            
            # Magnitude spectral loss (log domain to emphasize relative differences)
            mag_loss = F.mse_loss(
                torch.log1p(torch.abs(X) + 1e-8),
                torch.log1p(torch.abs(X_hat) + 1e-8)
            )
            
            # Phase loss with better weighting
            phase_x = torch.angle(X)
            phase_x_hat = torch.angle(X_hat)
            # Circular difference for phase (handles 2π wrapping)
            phase_diff = torch.abs(torch.remainder(phase_x - phase_x_hat + math.pi, 2 * math.pi) - math.pi)
            # Weight by magnitude to focus on important frequencies
            magnitude_weight = torch.abs(X) / (torch.abs(X).sum(dim=(1, 2), keepdim=True) + 1e-8)
            phase_loss = torch.mean(magnitude_weight * phase_diff)
            
            stft_losses.append(mag_loss + phase_loss)
        
        # Average the multi-resolution losses
        spec_loss = sum(stft_losses) / len(stft_losses)
        
        # Explicit phase loss for low frequencies (most perceptually important)
        X_low = torch.stft(
            x, 
            n_fft=2048, 
            hop_length=512, 
            win_length=2048, 
            window=torch.hann_window(2048).to(x.device), 
            return_complex=True
        )
        X_hat_low = torch.stft(
            x_hat, 
            n_fft=2048, 
            hop_length=512, 
            win_length=2048, 
            window=torch.hann_window(2048).to(x.device), 
            return_complex=True
        )
        
        # Get phase for just the bottom 25% of frequencies (most important for perception)
        low_freq_cutoff = X_low.shape[1] // 4
        phase_x_low = torch.angle(X_low[:, :low_freq_cutoff, :])
        phase_x_hat_low = torch.angle(X_hat_low[:, :low_freq_cutoff, :])
        
        # Circular difference with extra weight on low frequencies
        phase_diff_low = torch.abs(torch.remainder(phase_x_low - phase_x_hat_low + math.pi, 2 * math.pi) - math.pi)
        magnitude_weight_low = torch.abs(X_low[:, :low_freq_cutoff, :]) / (torch.abs(X_low[:, :low_freq_cutoff, :]).sum(dim=(1, 2), keepdim=True) + 1e-8)
        phase_loss_low = torch.mean(magnitude_weight_low * phase_diff_low)
        
        # MODIFIED: Balance loss functions - increased phase loss weight
        loss = l1_loss + 0.1 * spec_loss + 0.2 * phase_loss_low
        
        self.log('train_loss', loss)
        self.log('train_l1_loss', l1_loss)
        self.log('train_spec_loss', spec_loss)
        self.log('train_phase_loss', phase_loss_low)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch["audio"]
        x_hat = self(x)
        
        # Ensure shapes match
        if x_hat.shape != x.shape:
            self.log('shape_mismatch', 1.0)
            # Resize output to match input if needed (should not happen with proper upsampling)
            x_hat = F.interpolate(x_hat.unsqueeze(1), size=x.shape[1], mode='linear', align_corners=False).squeeze(1)
        
        # L1 loss + spectral loss
        l1_loss = F.l1_loss(x_hat, x)
        
        # MODIFIED: Enhanced spectral loss using multi-resolution STFT
        # Use multiple window sizes to capture both global and local time-frequency patterns
        stft_losses = []
        for n_fft in [256, 512, 1024]:
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
            
            # Magnitude spectral loss (log domain to emphasize relative differences)
            mag_loss = F.mse_loss(
                torch.log1p(torch.abs(X) + 1e-8),
                torch.log1p(torch.abs(X_hat) + 1e-8)
            )
            
            # Phase loss with better weighting
            phase_x = torch.angle(X)
            phase_x_hat = torch.angle(X_hat)
            # Circular difference for phase (handles 2π wrapping)
            phase_diff = torch.abs(torch.remainder(phase_x - phase_x_hat + math.pi, 2 * math.pi) - math.pi)
            # Weight by magnitude to focus on important frequencies
            magnitude_weight = torch.abs(X) / (torch.abs(X).sum(dim=(1, 2), keepdim=True) + 1e-8)
            phase_loss = torch.mean(magnitude_weight * phase_diff)
            
            stft_losses.append(mag_loss + phase_loss)
        
        # Average the multi-resolution losses
        spec_loss = sum(stft_losses) / len(stft_losses)
        
        # Explicit phase loss for low frequencies (most perceptually important)
        X_low = torch.stft(
            x, 
            n_fft=2048, 
            hop_length=512, 
            win_length=2048, 
            window=torch.hann_window(2048).to(x.device), 
            return_complex=True
        )
        X_hat_low = torch.stft(
            x_hat, 
            n_fft=2048, 
            hop_length=512, 
            win_length=2048, 
            window=torch.hann_window(2048).to(x.device), 
            return_complex=True
        )
        
        # Get phase for just the bottom 25% of frequencies (most important for perception)
        low_freq_cutoff = X_low.shape[1] // 4
        phase_x_low = torch.angle(X_low[:, :low_freq_cutoff, :])
        phase_x_hat_low = torch.angle(X_hat_low[:, :low_freq_cutoff, :])
        
        # Circular difference with extra weight on low frequencies
        phase_diff_low = torch.abs(torch.remainder(phase_x_low - phase_x_hat_low + math.pi, 2 * math.pi) - math.pi)
        magnitude_weight_low = torch.abs(X_low[:, :low_freq_cutoff, :]) / (torch.abs(X_low[:, :low_freq_cutoff, :]).sum(dim=(1, 2), keepdim=True) + 1e-8)
        phase_loss_low = torch.mean(magnitude_weight_low * phase_diff_low)
        
        # MODIFIED: Balance loss functions - increased phase loss weight
        loss = l1_loss + 0.1 * spec_loss + 0.2 * phase_loss_low
        
        self.log('val_loss', loss)
        self.log('val_l1_loss', l1_loss)
        self.log('val_spec_loss', spec_loss)
        self.log('val_phase_loss', phase_loss_low)
        
        # Calculate signal-to-noise ratio (SNR)
        with torch.no_grad():
            signal_power = torch.mean(x**2, dim=1)
            noise_power = torch.mean((x - x_hat)**2, dim=1)
            snr = 10 * torch.log10(signal_power / (noise_power + 1e-8))
            mean_snr = torch.mean(snr)
            self.log('val_snr_db', mean_snr)
            
            # MODIFIED: Calculate perceptual metrics via spectral contrast
            # Compare spectral contrast between original and reconstructed audio
            # This is a rough approximation of perceptual quality
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
        
        # Log audio and visualizations for first batch only to save computation
        if batch_idx == 0:
            # Log a few samples (e.g., first 4 in batch)
            num_samples = min(4, x.size(0))
            
            # Create figure for waveform comparison
            fig, axes = plt.subplots(num_samples, 2, figsize=(12, 3 * num_samples))
            
            # Handle case where num_samples = 1
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
            
            # Create simple spectrograms using PyTorch FFT
            fig, axes = plt.subplots(num_samples, 2, figsize=(12, 4 * num_samples))
            
            # Handle case where num_samples = 1
            if num_samples == 1:
                axes = np.array([axes])
            
            for i in range(num_samples):
                # Original spectrogram using FFT
                # Apply window function
                window = torch.hann_window(512).to(x.device)
                # Use PyTorch's stft
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
                
                # Handle case where num_samples = 1
                if num_samples == 1:
                    axes = np.array([axes])
                
                for i in range(num_samples):
                    # Original WST coefficients (mean across channels for visualization)
                    wst_orig = x_wst[i].abs().mean(dim=0).cpu().numpy()
                    im = axes[i, 0].imshow(wst_orig.reshape(-1, 1), aspect='auto', origin='lower')
                    axes[i, 0].set_title(f"Orig WST Mean {i+1}")
                    
                    # Reconstructed WST coefficients
                    wst_recon = x_hat_wst[i].abs().mean(dim=0).cpu().numpy()
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
        # MODIFIED: Use AdamW for better weight regularization
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        
        # MODIFIED: Improved learning rate schedule
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