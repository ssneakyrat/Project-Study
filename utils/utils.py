import os
import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt
from types import SimpleNamespace
import time

def load_config(config_path):
    """Load YAML config file and convert to Namespace object"""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    return SimpleNamespace(**config_dict)

def save_config(config, path):
    """Save config to YAML file"""
    config_dict = {k: v for k, v in config.__dict__.items() if not k.startswith('__')}
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        yaml.dump(config_dict, f)

def calculate_snr(original, reconstructed):
    """Calculate Signal-to-Noise Ratio in dB
    
    SNR = 10log₁₀(P_signal/P_noise) where:
    P_signal = Σx²/N and P_noise = Σ(x-x̂)²/N
    """
    noise = original - reconstructed
    signal_power = torch.mean(original**2, dim=1)
    noise_power = torch.mean(noise**2, dim=1) 
    snr = 10 * torch.log10(signal_power / (noise_power + 1e-10))
    return snr

def compute_spectrogram(signal, n_fft, hop_length):
    """Compute spectrogram using FFT
    
    Spectrogram is |STFT|² where STFT is the Short-Time Fourier Transform:
    STFT(x[n])[m,k] = Σₙ x[n]·w[n-m]·e^(-j2πkn/N)
    """
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

def print_gpu_stats():
    """Print GPU memory usage stats"""
    if torch.cuda.is_available():
        allocated_gb = torch.cuda.memory_allocated() / 1e9
        reserved_gb = torch.cuda.memory_reserved() / 1e9
        print(f"GPU Memory - Allocated: {allocated_gb:.2f} GB, Reserved: {reserved_gb:.2f} GB")