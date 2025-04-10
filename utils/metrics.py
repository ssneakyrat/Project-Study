#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


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
        # Ensure inputs are [B, T]
        if pred.dim() == 3:
            pred = pred.squeeze(1)
        if target.dim() == 3:
            target = target.squeeze(1)
        
        # Calculate noise
        noise = target - pred
        
        # Calculate signal and noise power
        signal_power = torch.sum(target ** 2, dim=-1)
        noise_power = torch.sum(noise ** 2, dim=-1) + self.eps
        
        # Calculate SNR
        snr = 10 * torch.log10(signal_power / noise_power)
        
        return snr


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
        # Ensure inputs are [B, T]
        if pred.dim() == 3:
            pred = pred.squeeze(1)
        if target.dim() == 3:
            target = target.squeeze(1)
        
        # Create window on the same device as inputs
        window = torch.hann_window(self.n_fft).to(pred.device)
        
        # Compute STFTs
        pred_stft = torch.stft(pred, n_fft=self.n_fft, hop_length=self.hop_length, 
                             window=window, return_complex=True)
        target_stft = torch.stft(target, n_fft=self.n_fft, hop_length=self.hop_length, 
                               window=window, return_complex=True)
        
        # Compute log magnitude spectrograms
        pred_mag = torch.log(torch.abs(pred_stft) + 1e-7)
        target_mag = torch.log(torch.abs(target_stft) + 1e-7)
        
        # Compute mean square error between spectrograms
        spectral_distortion = F.mse_loss(pred_mag, target_mag, reduction='none').mean(dim=[1, 2])
        
        return spectral_distortion


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
    # Ensure both tensors are [B, T]
    if pred.dim() == 3:
        pred = pred.squeeze(1)
    if target.dim() == 3:
        target = target.squeeze(1)
    
    # Ensure lengths match
    if pred.size(-1) != target.size(-1):
        if pred.size(-1) > target.size(-1):
            pred = pred[..., :target.size(-1)]
        else:
            target = target[..., :pred.size(-1)]
    
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