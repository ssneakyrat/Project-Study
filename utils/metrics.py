#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchaudio


class SNR(nn.Module):
    """
    Signal-to-Noise Ratio (SNR) metric
    """
    def __init__(self, eps=1e-8):
        super(SNR, self).__init__()
        self.eps = eps
    
    def forward(self, pred, target):
        """
        Args:
            pred (Tensor): Predicted audio [B, T]
            target (Tensor): Target audio [B, T]
            
        Returns:
            Tensor: SNR value in dB
        """
        # Ensure input shapes match
        assert pred.shape == target.shape, "Input shapes must match"
        
        # Calculate noise
        noise = target - pred
        
        # Calculate signal and noise power
        signal_power = torch.sum(target ** 2, dim=-1)
        noise_power = torch.sum(noise ** 2, dim=-1) + self.eps
        
        # Calculate SNR
        snr = 10 * torch.log10(signal_power / noise_power)
        
        return snr


class PESQ(nn.Module):
    """
    Perceptual Evaluation of Speech Quality (PESQ) metric
    Note: This is a wrapper around torchaudio's PESQ implementation
    """
    def __init__(self, sample_rate=16000, mode='nb'):
        super(PESQ, self).__init__()
        self.sample_rate = sample_rate
        self.mode = mode  # 'nb' for narrowband or 'wb' for wideband
        
        # PESQ requires to detach tensors and move to CPU as numpy arrays
        
    def forward(self, pred, target):
        """
        Args:
            pred (Tensor): Predicted audio [B, T]
            target (Tensor): Target audio [B, T]
            
        Returns:
            Tensor: PESQ score
        """
        # PESQ can only be computed on CPU with numpy arrays
        # In an actual implementation, this would use a PESQ estimator that works with PyTorch
        # For now, we'll use a placeholder implementation
        
        # Convert to numpy arrays
        pred_np = pred.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()
        
        batch_size = pred.shape[0]
        scores = []
        
        for i in range(batch_size):
            # This is a placeholder - in a real implementation, we would call:
            # score = pesq(self.sample_rate, target_np[i], pred_np[i], self.mode)
            
            # For now, return a simulated score based on MSE as a proxy
            mse = np.mean((pred_np[i] - target_np[i]) ** 2)
            simulated_score = 4.5 - 2.5 * np.sqrt(mse)  # Map MSE to a 1-5 range
            simulated_score = max(1.0, min(4.5, simulated_score))  # Clamp to typical PESQ range
            scores.append(simulated_score)
        
        # Return as tensor
        return torch.tensor(scores, device=pred.device)


class STOI(nn.Module):
    """
    Short-Time Objective Intelligibility (STOI) metric
    Note: This is a placeholder implementation that approximates STOI
    """
    def __init__(self, sample_rate=16000):
        super(STOI, self).__init__()
        self.sample_rate = sample_rate
        
    def forward(self, pred, target):
        """
        Args:
            pred (Tensor): Predicted audio [B, T]
            target (Tensor): Target audio [B, T]
            
        Returns:
            Tensor: STOI score
        """
        # STOI requires CPU computation with numpy arrays
        # In an actual implementation, we would use torchaudio's STOI
        # For now, we'll use a correlation-based approximation
        
        batch_size = pred.shape[0]
        scores = []
        
        for i in range(batch_size):
            # Normalize signals
            pred_norm = pred[i] / (torch.std(pred[i]) + 1e-8)
            target_norm = target[i] / (torch.std(target[i]) + 1e-8)
            
            # Calculate correlation coefficient
            corr = torch.sum(pred_norm * target_norm) / (torch.sqrt(torch.sum(pred_norm**2) * torch.sum(target_norm**2)) + 1e-8)
            
            # Map correlation to STOI range (typically 0-1)
            stoi_approx = (corr + 1) / 2
            scores.append(stoi_approx)
        
        return torch.stack(scores)


class SpectralDistortion(nn.Module):
    """
    Computes spectral distortion between predicted and target audio
    """
    def __init__(self, n_fft=1024, hop_length=256):
        super(SpectralDistortion, self).__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        
    def forward(self, pred, target):
        """
        Args:
            pred (Tensor): Predicted audio [B, T]
            target (Tensor): Target audio [B, T]
            
        Returns:
            Tensor: Spectral distortion measure
        """
        # Compute STFTs
        pred_stft = torch.stft(pred, n_fft=self.n_fft, hop_length=self.hop_length, 
                              window=torch.hann_window(self.n_fft).to(pred.device),
                              return_complex=True)
        target_stft = torch.stft(target, n_fft=self.n_fft, hop_length=self.hop_length, 
                                window=torch.hann_window(self.n_fft).to(target.device),
                                return_complex=True)
        
        # Compute log magnitude spectrograms
        pred_mag = torch.log(torch.abs(pred_stft) + 1e-7)
        target_mag = torch.log(torch.abs(target_stft) + 1e-7)
        
        # Compute mean square error between spectrograms
        spectral_distortion = F.mse_loss(pred_mag, target_mag, reduction='none').mean(dim=[1, 2])
        
        return spectral_distortion


class WaveletDistortion(nn.Module):
    """
    Computes distortion in wavelet domain
    """
    def __init__(self, wavelet_transform):
        super(WaveletDistortion, self).__init__()
        self.wavelet_transform = wavelet_transform
        
    def forward(self, pred, target):
        """
        Args:
            pred (Tensor): Predicted audio [B, T]
            target (Tensor): Target audio [B, T]
            
        Returns:
            Tensor: Wavelet domain distortion measure
        """
        # Apply wavelet transform
        pred_wavelet = self.wavelet_transform(pred)
        target_wavelet = self.wavelet_transform(target)
        
        # Unpack real and imaginary parts
        pred_real, pred_imag = pred_wavelet
        target_real, target_imag = target_wavelet
        
        # Compute distortion
        real_distortion = F.mse_loss(pred_real, target_real, reduction='none').mean(dim=[1, 2])
        imag_distortion = F.mse_loss(pred_imag, target_imag, reduction='none').mean(dim=[1, 2])
        
        # Combine real and imaginary distortions
        total_distortion = real_distortion + imag_distortion
        
        return total_distortion


def compute_metrics_dict(pred, target, sample_rate=16000):
    """
    Compute multiple audio quality metrics and return as dictionary
    
    Args:
        pred (Tensor): Predicted audio [B, T]
        target (Tensor): Target audio [B, T]
        sample_rate (int): Audio sample rate
        
    Returns:
        dict: Dictionary of metrics
    """
    # Initialize metrics
    snr_metric = SNR()
    spectral_dist = SpectralDistortion()
    
    # Compute metrics
    snr = snr_metric(pred, target).mean().item()
    spec_dist = spectral_dist(pred, target).mean().item()
    
    # Create metrics dictionary
    metrics = {
        'snr_db': snr,
        'spectral_distortion': spec_dist,
    }
    
    return metrics