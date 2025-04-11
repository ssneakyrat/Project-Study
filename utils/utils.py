import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import yaml
from pathlib import Path

def set_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def count_parameters(model):
    """Count number of trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def save_config(config, config_path):
    """Save configuration to YAML file"""
    with open(config_path, 'w') as f:
        yaml.dump(config, f)

def create_directories(dirs):
    """Create directories if they don't exist"""
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

def plot_waveforms(original, reconstructed, save_path=None):
    """
    Plot original and reconstructed waveforms
    Args:
        original: Original audio waveform
        reconstructed: Reconstructed audio waveform
        save_path: Path to save plot
    """
    plt.figure(figsize=(10, 6))
    
    plt.subplot(2, 1, 1)
    plt.plot(original)
    plt.title('Original')
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(reconstructed)
    plt.title('Reconstructed')
    plt.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    
    plt.close()

def plot_spectrogram(audio, sample_rate=16000, save_path=None):
    """
    Plot spectrogram of audio
    Args:
        audio: Audio waveform
        sample_rate: Sample rate of audio
        save_path: Path to save plot
    """
    plt.figure(figsize=(10, 6))
    
    # Compute spectrogram
    D = librosa.amplitude_to_db(
        np.abs(librosa.stft(audio, hop_length=256, n_fft=2048)),
        ref=np.max
    )
    
    plt.imshow(D, origin='lower', aspect='auto', extent=[0, len(audio)/sample_rate, 0, sample_rate/2])
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    
    plt.close()

def plot_wavelet_coeffs(coeffs, level=None, save_path=None):
    """
    Plot wavelet coefficients
    Args:
        coeffs: Dictionary of wavelet coefficients
        level: Specific level to plot (None for all)
        save_path: Path to save plot
    """
    if level is None:
        # Plot all levels
        num_levels = len(coeffs['d'])
        plt.figure(figsize=(12, 2 * (num_levels + 1)))
        
        # Plot approximation coefficients
        plt.subplot(num_levels + 1, 1, 1)
        plt.plot(coeffs['a'][0].cpu().numpy())
        plt.title('Approximation Coefficients')
        plt.grid(True)
        
        # Plot detail coefficients for each level
        for j, d in enumerate(coeffs['d']):
            plt.subplot(num_levels + 1, 1, j + 2)
            plt.plot(d[0].cpu().numpy())
            plt.title(f'Detail Coefficients (Level {j+1})')
            plt.grid(True)
    else:
        # Plot specific level
        plt.figure(figsize=(10, 5))
        
        if level == 0:
            # Plot approximation coefficients
            plt.plot(coeffs['a'][0].cpu().numpy())
            plt.title('Approximation Coefficients')
        else:
            # Plot detail coefficients
            plt.plot(coeffs['d'][level-1][0].cpu().numpy())
            plt.title(f'Detail Coefficients (Level {level})')
        
        plt.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    
    plt.close()