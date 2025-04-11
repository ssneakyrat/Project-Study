import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter
import soundfile as sf
import librosa
import librosa.display

class TensorboardLogger:
    def __init__(self, log_dir, sample_rate=16000):
        """
        TensorBoard logger for audio model
        Args:
            log_dir: Directory for TensorBoard logs
            sample_rate: Sample rate of audio
        """
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)
        self.sample_rate = sample_rate
        
    def log_audio(self, tag, audio, global_step, max_examples=3):
        """
        Log audio to TensorBoard
        Args:
            tag: Tag for audio
            audio: Audio tensor of shape [B, T]
            global_step: Global step
            max_examples: Maximum number of examples to log
        """
        # Ensure audio is on CPU and convert to numpy
        if torch.is_tensor(audio):
            audio = audio.detach().cpu().numpy()
        
        # Log only a subset of examples
        num_examples = min(audio.shape[0], max_examples)
        
        for i in range(num_examples):
            # Normalize to [-1, 1]
            audio_i = audio[i]
            if np.max(np.abs(audio_i)) > 0:
                audio_i = audio_i / np.max(np.abs(audio_i))
            
            self.writer.add_audio(
                f'{tag}/{i}',
                audio_i,
                global_step,
                sample_rate=self.sample_rate
            )
    
    def log_spectrogram(self, tag, audio, global_step, max_examples=3):
        """
        Log spectrogram to TensorBoard
        Args:
            tag: Tag for spectrogram
            audio: Audio tensor of shape [B, T]
            global_step: Global step
            max_examples: Maximum number of examples to log
        """
        # Ensure audio is on CPU and convert to numpy
        if torch.is_tensor(audio):
            audio = audio.detach().cpu().numpy()
        
        # Log only a subset of examples
        num_examples = min(audio.shape[0], max_examples)
        
        for i in range(num_examples):
            plt.figure(figsize=(10, 4))
            
            # Compute spectrogram
            D = librosa.amplitude_to_db(
                np.abs(librosa.stft(audio[i], hop_length=256, n_fft=2048)),
                ref=np.max
            )
            
            librosa.display.specshow(
                D,
                y_axis='linear',
                x_axis='time',
                sr=self.sample_rate,
                hop_length=256
            )
            plt.colorbar(format='%+2.0f dB')
            plt.title(f'Spectrogram {i}')
            
            self.writer.add_figure(f'{tag}/{i}', plt.gcf(), global_step)
            plt.close()
    
    def log_model_weights(self, model, global_step):
        """
        Log model weight histograms to TensorBoard
        Args:
            model: PyTorch model
            global_step: Global step
        """
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.writer.add_histogram(f'weights/{name}', param.data, global_step)
                self.writer.add_histogram(f'grads/{name}', param.grad, global_step)
    
    def log_latent_space(self, latent, global_step):
        """
        Log latent space visualization to TensorBoard
        Args:
            latent: Latent vectors of shape [B, D]
            global_step: Global step
        """
        # Ensure latent is on CPU and convert to numpy
        if torch.is_tensor(latent):
            latent = latent.detach().cpu().numpy()
        
        # Log latent space statistics
        self.writer.add_histogram('latent/values', latent, global_step)
        
        # Log latent space norms
        norms = np.linalg.norm(latent, axis=1)
        self.writer.add_histogram('latent/norms', norms, global_step)
        
        # If latent space is 2D or 3D, visualize it directly
        if latent.shape[1] == 2:
            fig = plt.figure(figsize=(8, 8))
            plt.scatter(latent[:, 0], latent[:, 1], alpha=0.5)
            plt.title('Latent Space (2D)')
            plt.xlabel('Dimension 1')
            plt.ylabel('Dimension 2')
            plt.grid(True)
            self.writer.add_figure('latent/2d', fig, global_step)
            plt.close()
        elif latent.shape[1] == 3:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(latent[:, 0], latent[:, 1], latent[:, 2], alpha=0.5)
            ax.set_title('Latent Space (3D)')
            ax.set_xlabel('Dimension 1')
            ax.set_ylabel('Dimension 2')
            ax.set_zlabel('Dimension 3')
            self.writer.add_figure('latent/3d', fig, global_step)
            plt.close()
    
    def log_wavelet_coeffs(self, coeffs, global_step, example_idx=0):
        """
        Log wavelet coefficient visualization to TensorBoard
        Args:
            coeffs: Dictionary of wavelet coefficients
            global_step: Global step
            example_idx: Index of example to visualize
        """
        # Ensure coefficients are on CPU
        a = coeffs['a'][example_idx].detach().cpu().numpy()
        ds = [d[example_idx].detach().cpu().numpy() for d in coeffs['d']]
        
        # Create visualization of coefficients
        num_levels = len(ds)
        fig, axes = plt.subplots(num_levels + 1, 1, figsize=(10, 2 * (num_levels + 1)))
        
        # Plot approximation coefficients
        axes[0].plot(a)
        axes[0].set_title('Approximation Coefficients')
        axes[0].grid(True)
        
        # Plot detail coefficients for each level
        for j, d in enumerate(ds):
            axes[j+1].plot(d)
            axes[j+1].set_title(f'Detail Coefficients (Level {j+1})')
            axes[j+1].grid(True)
        
        plt.tight_layout()
        self.writer.add_figure('wavelet_coeffs', fig, global_step)
        plt.close()
    
    def close(self):
        """Close TensorBoard writer"""
        self.writer.close()