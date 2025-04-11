import os
import time
import yaml
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import pywt

# Config
class Config:
    def __init__(self):
        self.sample_rate = 16000
        self.audio_length = 16000  # 1 second
        self.n_training_samples = 100
        self.n_test_samples = 20
        self.batch_size = 16
        self.epochs = 200
        self.wavelet = 'db4'
        self.dwt_level = 3
        self.hidden_dims = [128, 64, 32]
        self.latent_dim = 16
        self.learning_rate = 1e-3
        
config = Config()

# Utility function to generate dummy audio data
def generate_dummy_audio_data(n_samples, sample_rate, duration):
    data = []
    for i in range(n_samples):
        # Generate a sinusoidal signal with random frequency
        freq = np.random.uniform(200, 2000)
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        # Add some harmonics for complexity
        signal = np.sin(2 * np.pi * freq * t)
        signal += 0.5 * np.sin(2 * np.pi * freq * 2 * t)
        signal += 0.25 * np.sin(2 * np.pi * freq * 3 * t)
        
        # Add some noise
        noise = np.random.normal(0, 0.05, len(signal))
        signal += noise
        
        # Normalize
        signal = signal / np.max(np.abs(signal))
        
        data.append(signal)
    return np.array(data)

# Dataset class
class DummyAudioDataset(Dataset):
    def __init__(self, data):
        self.data = torch.FloatTensor(data)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

# Simplified wavelet transform class
class WaveletTransform:
    def __init__(self, wavelet='db4', level=3):
        self.wavelet = wavelet
        self.level = level
    
    def forward(self, x):
        """Apply DWT to batch of signals and return flattened coefficients"""
        batch_size = x.shape[0]
        device = x.device
        coeffs_list = []
        
        for i in range(batch_size):
            signal = x[i].cpu().numpy()
            
            # Apply DWT
            coeffs = pywt.wavedec(signal, self.wavelet, level=self.level)
            
            # Store coefficient lengths for reconstruction
            if i == 0:
                self.coeffs_lengths = [len(c) for c in coeffs]
                self.coeffs_shapes = [c.shape for c in coeffs]
            
            # Flatten coefficients
            flat_coeffs = np.concatenate([c.flatten() for c in coeffs])
            coeffs_list.append(flat_coeffs)
        
        # Stack and convert to tensor
        coeffs_tensor = torch.tensor(np.stack(coeffs_list), dtype=torch.float32, device=device)
        return coeffs_tensor
    
    def inverse(self, coeffs_tensor):
        """Reconstruct signals from flattened wavelet coefficients"""
        batch_size = coeffs_tensor.shape[0]
        device = coeffs_tensor.device
        reconstructed = []
        
        for i in range(batch_size):
            flat_coeffs = coeffs_tensor[i].cpu().detach().numpy()
            
            # Split coefficients based on stored lengths
            coeffs = []
            start_idx = 0
            
            for j, shape in enumerate(self.coeffs_shapes):
                size = np.prod(shape)
                coeff = flat_coeffs[start_idx:start_idx+size].reshape(shape)
                coeffs.append(coeff)
                start_idx += size
            
            # Reconstruct signal
            rec_signal = pywt.waverec(coeffs, self.wavelet)
            reconstructed.append(rec_signal)
        
        # Match length of original signals by trimming or padding
        max_len = max(len(r) for r in reconstructed)
        for i in range(len(reconstructed)):
            if len(reconstructed[i]) < max_len:
                pad_len = max_len - len(reconstructed[i])
                reconstructed[i] = np.pad(reconstructed[i], (0, pad_len), 'constant')
        
        return torch.tensor(np.stack(reconstructed), dtype=torch.float32, device=device)

# Simple autoencoder model
class WaveletAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim):
        super().__init__()
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        
        for dim in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, dim))
            encoder_layers.append(nn.ReLU())
            prev_dim = dim
        
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        
        for dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(prev_dim, dim))
            decoder_layers.append(nn.ReLU())
            prev_dim = dim
        
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z

# Lightning module
class WaveletAudioAE(pl.LightningModule):
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
        noise = x - x_recon
        signal_power = torch.mean(x**2, dim=1)
        noise_power = torch.mean(noise**2, dim=1)
        snr = 10 * torch.log10(signal_power / (noise_power + 1e-10))
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
        noise = x - x_recon
        signal_power = torch.mean(x**2, dim=1)
        noise_power = torch.mean(noise**2, dim=1)
        snr = 10 * torch.log10(signal_power / (noise_power + 1e-10))
        avg_snr = torch.mean(snr)
        
        # Log metrics
        self.log('val_loss', loss)
        self.log('val_recon_loss', recon_loss)
        self.log('val_wavelet_loss', wavelet_loss)
        self.log('val_snr', avg_snr)
        
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
            
            # Compute FFT spectrogram manually without librosa
            def compute_spectrogram(signal, n_fft, hop_length):
                # Pad signal
                pad_signal = np.pad(signal, (0, n_fft), mode='constant')
                
                # Number of frames
                num_frames = 1 + (len(pad_signal) - n_fft) // hop_length
                
                # Initialize spectrogram
                spec = np.zeros((n_fft // 2 + 1, num_frames))
                
                # Compute spectrogram
                for i in range(num_frames):
                    start = i * hop_length
                    end = start + n_fft
                    frame = pad_signal[start:end] * np.hanning(n_fft)
                    spectrum = np.abs(np.fft.rfft(frame))
                    spec[:, i] = spectrum
                
                return spec
            
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

# Main function
def main():
    # Set random seed
    pl.seed_everything(42)
    
    # Create output directories
    os.makedirs('outputs', exist_ok=True)
    
    # Generate dummy audio data
    print("Generating dummy audio data...")
    train_data = generate_dummy_audio_data(
        config.n_training_samples, 
        config.sample_rate, 
        config.audio_length / config.sample_rate
    )
    
    test_data = generate_dummy_audio_data(
        config.n_test_samples, 
        config.sample_rate, 
        config.audio_length / config.sample_rate
    )
    
    # Create datasets and dataloaders
    train_dataset = DummyAudioDataset(train_data)
    test_dataset = DummyAudioDataset(test_data)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True,
        num_workers=4,
        persistent_workers=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.batch_size, 
        shuffle=False,
        num_workers=4,
        persistent_workers=True
    )
    
    # Initialize model
    model = WaveletAudioAE(config)
    
    # Initialize logger
    logger = TensorBoardLogger('logs', name='wavelet_ae')
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=config.epochs,
        logger=logger,
        log_every_n_steps=1,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        check_val_every_n_epoch=10
    )
    
    # Track GPU memory usage
    if torch.cuda.is_available():
        print("GPU Memory Before Training:")
        allocated_gb = torch.cuda.memory_allocated() / 1e9
        reserved_gb = torch.cuda.memory_reserved() / 1e9
        print(f"Allocated: {allocated_gb:.2f} GB")
        print(f"Reserved: {reserved_gb:.2f} GB")
    
    # Train model
    print("Starting training...")
    start_time = time.time()
    trainer.fit(model, train_loader)
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Track GPU memory after training
    if torch.cuda.is_available():
        print("GPU Memory After Training:")
        allocated_gb = torch.cuda.memory_allocated() / 1e9
        reserved_gb = torch.cuda.memory_reserved() / 1e9
        print(f"Allocated: {allocated_gb:.2f} GB")
        print(f"Reserved: {reserved_gb:.2f} GB")
    
    # Evaluate model on test data
    print("\nEvaluating model on test data...")
    model.eval()
    
    test_mse = 0
    test_snr = 0
    n_samples = 0
    
    with torch.no_grad():
        for batch in test_loader:
            x = batch
            x_recon, _, _, _ = model(x)
            
            # Calculate MSE
            mse = F.mse_loss(x_recon, x).item()
            test_mse += mse * x.shape[0]
            
            # Calculate SNR
            noise = x - x_recon
            signal_power = torch.mean(x**2, dim=1)
            noise_power = torch.mean(noise**2, dim=1)
            snr = 10 * torch.log10(signal_power / (noise_power + 1e-10))
            test_snr += torch.sum(snr).item()
            
            n_samples += x.shape[0]
    
    avg_mse = test_mse / n_samples
    avg_snr = test_snr / n_samples
    
    print("\nFinal Test Results:")
    print(f"Mean Squared Error (MSE): {avg_mse:.6f}")
    print(f"Signal-to-Noise Ratio (SNR): {avg_snr:.2f} dB")
    print(f"Compression Ratio: {model.compression_ratio:.2f}x")
    
    # Save model and configuration
    model_path = os.path.join('outputs', 'wavelet_ae_model.pt')
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    config_dict = {k: v for k, v in config.__dict__.items() if not k.startswith('__')}
    config_path = os.path.join('outputs', 'wavelet_ae_config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config_dict, f)
    print(f"Configuration saved to {config_path}")
    
    print("\nTest script completed successfully!")

if __name__ == "__main__":
    main()