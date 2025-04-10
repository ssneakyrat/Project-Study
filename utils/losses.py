#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ComplexL1Loss(nn.Module):
    """
    Complex L1 loss - computes L1 loss considering both real and imaginary parts
    """
    def __init__(self, reduction='mean'):
        super(ComplexL1Loss, self).__init__()
        self.reduction = reduction
    
    def forward(self, pred, target):
        """
        Args:
            pred (tuple): Tuple of (real, imag) tensors
            target (tuple): Tuple of (real, imag) tensors
            
        Returns:
            Tensor: L1 loss
        """
        pred_real, pred_imag = pred
        target_real, target_imag = target
        
        # Compute L1 loss for real and imaginary parts
        loss_real = torch.abs(pred_real - target_real)
        loss_imag = torch.abs(pred_imag - target_imag)
        
        # Combined loss
        loss = loss_real + loss_imag
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class SpectralConvergenceLoss(nn.Module):
    """
    Spectral Convergence Loss for comparing two spectrograms
    """
    def __init__(self):
        super(SpectralConvergenceLoss, self).__init__()
    
    def forward(self, pred_mag, target_mag):
        """
        Args:
            pred_mag (Tensor): Predicted magnitude spectrogram
            target_mag (Tensor): Target magnitude spectrogram
            
        Returns:
            Tensor: Spectral convergence loss
        """
        # Compute Frobenius norm
        return torch.norm(target_mag - pred_mag, p='fro') / torch.norm(target_mag, p='fro')


class LogSTFTMagnitudeLoss(nn.Module):
    """
    Log STFT magnitude loss
    """
    def __init__(self):
        super(LogSTFTMagnitudeLoss, self).__init__()
    
    def forward(self, pred_mag, target_mag):
        """
        Args:
            pred_mag (Tensor): Predicted magnitude spectrogram
            target_mag (Tensor): Target magnitude spectrogram
            
        Returns:
            Tensor: Log STFT magnitude loss
        """
        return F.l1_loss(torch.log(pred_mag + 1e-7), torch.log(target_mag + 1e-7))


class STFTLoss(nn.Module):
    """
    STFT loss module combining spectral convergence and log magnitude loss
    """
    def __init__(self, fft_sizes=[1024, 2048, 512], hop_sizes=[256, 512, 128], win_lengths=[1024, 2048, 512]):
        super(STFTLoss, self).__init__()
        
        self.sc_loss = SpectralConvergenceLoss()
        self.mag_loss = LogSTFTMagnitudeLoss()
        
        # Multiple FFT settings for multi-resolution STFT loss
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths
    
    def stft(self, x, fft_size, hop_size, win_length):
        """
        Compute STFT
        
        Args:
            x (Tensor): Input audio
            fft_size (int): FFT size
            hop_size (int): Hop size
            win_length (int): Window length
            
        Returns:
            Tensor: STFT magnitude
        """
        x_stft = torch.stft(x, fft_size, hop_size, win_length, window=torch.hann_window(win_length).to(x.device),
                           return_complex=True)
        return torch.abs(x_stft)
    
    def forward(self, pred, target):
        """
        Args:
            pred (Tensor): Predicted audio [B, T]
            target (Tensor): Target audio [B, T]
            
        Returns:
            tuple: (total loss, sc loss, mag loss)
        """
        # Reshape if needed
        if pred.dim() == 3:  # [B, C, T]
            pred = pred.squeeze(1)
        if target.dim() == 3:  # [B, C, T]
            target = target.squeeze(1)
        
        # Multi-resolution STFT loss
        sc_loss = 0.0
        mag_loss = 0.0
        
        for fft_size, hop_size, win_length in zip(self.fft_sizes, self.hop_sizes, self.win_lengths):
            pred_mag = self.stft(pred, fft_size, hop_size, win_length)
            target_mag = self.stft(target, fft_size, hop_size, win_length)
            
            sc_loss += self.sc_loss(pred_mag, target_mag)
            mag_loss += self.mag_loss(pred_mag, target_mag)
        
        # Average across all FFT settings
        sc_loss /= len(self.fft_sizes)
        mag_loss /= len(self.fft_sizes)
        
        total_loss = sc_loss + mag_loss
        
        return total_loss, sc_loss, mag_loss


class ComplexSTFTLoss(nn.Module):
    """
    Complex STFT loss incorporating both magnitude and phase information
    """
    def __init__(self, fft_sizes=[1024, 2048, 512], hop_sizes=[256, 512, 128], win_lengths=[1024, 2048, 512]):
        super(ComplexSTFTLoss, self).__init__()
        
        # Multi-resolution settings
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths
    
    def stft(self, x, fft_size, hop_size, win_length):
        """
        Compute complex STFT
        
        Args:
            x (Tensor): Input audio
            fft_size (int): FFT size
            hop_size (int): Hop size
            win_length (int): Window length
            
        Returns:
            Tensor: Complex STFT tensor
        """
        return torch.stft(x, fft_size, hop_size, win_length, window=torch.hann_window(win_length).to(x.device),
                         return_complex=True)
    
    def forward(self, pred, target):
        """
        Args:
            pred (Tensor): Predicted audio [B, T]
            target (Tensor): Target audio [B, T]
            
        Returns:
            tuple: (total loss, mag loss, phase loss)
        """
        # Reshape if needed
        if pred.dim() == 3:  # [B, C, T]
            pred = pred.squeeze(1)
        if target.dim() == 3:  # [B, C, T]
            target = target.squeeze(1)
        
        # Multi-resolution STFT loss
        mag_loss = 0.0
        phase_loss = 0.0
        complex_loss = 0.0
        
        for fft_size, hop_size, win_length in zip(self.fft_sizes, self.hop_sizes, self.win_lengths):
            pred_stft = self.stft(pred, fft_size, hop_size, win_length)
            target_stft = self.stft(target, fft_size, hop_size, win_length)
            
            # Magnitude loss
            pred_mag = torch.abs(pred_stft)
            target_mag = torch.abs(target_stft)
            
            # Log magnitude loss with better handling of small values
            eps = 1e-7  # Epsilon to prevent log(0)
            log_pred_mag = torch.log(pred_mag + eps)
            log_target_mag = torch.log(target_mag + eps)
            
            # Magnitude loss with spectral weighting
            # Weight higher frequencies more to improve high-frequency reconstruction
            freq_weight = torch.linspace(1.0, 2.0, pred_mag.size(1), device=pred_mag.device).view(1, -1, 1)
            weighted_mag_loss = F.l1_loss(log_pred_mag * freq_weight, log_target_mag * freq_weight)
            mag_loss += weighted_mag_loss
            
            # Improved phase loss with magnitude weighting
            pred_phase = torch.angle(pred_stft)
            target_phase = torch.angle(target_stft)
            
            # Circular phase distance with magnitude weighting
            phase_diff = torch.abs(torch.remainder(pred_phase - target_phase + torch.pi, 2 * torch.pi) - torch.pi)
            
            # Magnitude-weighted phase loss
            # Focus more on high-energy regions where phase is perceptually important
            mag_weight = torch.tanh(target_mag * 5)  # Saturate for stability
            norm_weight = mag_weight / (torch.mean(mag_weight) + eps)
            phase_loss += torch.mean(norm_weight * phase_diff)
            
            # Complex loss considering both real and imaginary together
            # This captures correlation between magnitude and phase
            complex_loss += F.l1_loss(pred_stft.real, target_stft.real) + F.l1_loss(pred_stft.imag, target_stft.imag)
        
        # Average across all FFT settings
        mag_loss /= len(self.fft_sizes)
        phase_loss /= len(self.fft_sizes)
        complex_loss /= len(self.fft_sizes)
        
        return complex_loss, mag_loss, phase_loss


class WaveletLoss(nn.Module):
    """
    Loss function based on wavelet coefficients
    """
    def __init__(self, wavelet_transform):
        super(WaveletLoss, self).__init__()
        self.wavelet_transform = wavelet_transform
    
    def forward(self, pred, target):
        """
        Args:
            pred (Tensor): Predicted audio
            target (Tensor): Target audio
            
        Returns:
            Tensor: Wavelet coefficient loss
        """
        # Apply wavelet transform
        pred_wavelet = self.wavelet_transform(pred)
        target_wavelet = self.wavelet_transform(target)
        
        # Unpack real and imaginary parts
        pred_real, pred_imag = pred_wavelet
        target_real, target_imag = target_wavelet
        
        # Compute loss on real and imaginary parts
        loss_real = F.mse_loss(pred_real, target_real)
        loss_imag = F.mse_loss(pred_imag, target_imag)
        
        return loss_real + loss_imag


class TimeFrequencyLoss(nn.Module):
    """
    Combined time and frequency domain loss
    """
    def __init__(self, fft_sizes=[1024, 2048, 512], hop_sizes=[256, 512, 128], win_lengths=[1024, 2048, 512], 
                 time_weight=1.0, freq_weight=1.0):
        super(TimeFrequencyLoss, self).__init__()
        
        self.time_weight = time_weight
        self.freq_weight = freq_weight
        self.stft_loss = STFTLoss(fft_sizes, hop_sizes, win_lengths)
    
    def forward(self, pred, target):
        """
        Args:
            pred (Tensor): Predicted audio
            target (Tensor): Target audio
            
        Returns:
            Tensor: Combined time and frequency loss
        """
        # Time domain loss (L1)
        time_loss = F.l1_loss(pred, target)
        
        # Frequency domain loss (STFT)
        freq_loss, sc_loss, mag_loss = self.stft_loss(pred, target)
        
        # Combine losses
        total_loss = self.time_weight * time_loss + self.freq_weight * freq_loss
        
        return total_loss, time_loss, freq_loss
    
class EnhancedPhaseConsistencyLoss(nn.Module):
    """
    Enhanced phase consistency loss that focuses on perceptually relevant phase relationships
    """
    def __init__(self, fft_sizes=[512, 1024, 2048], hop_sizes=[128, 256, 512]):
        super(EnhancedPhaseConsistencyLoss, self).__init__()
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
    
    def forward(self, pred, target):
        """
        Args:
            pred (Tensor): Predicted audio [B, T]
            target (Tensor): Target audio [B, T]
            
        Returns:
            Tensor: Phase consistency loss
        """
        # Handle dimension adjustment
        if pred.dim() == 3:
            pred = pred.squeeze(1)
        if target.dim() == 3:
            target = target.squeeze(1)
        
        # Multi-resolution phase consistency
        phase_loss = 0.0
        mag_weighted_phase_loss = 0.0
        
        for fft_size, hop_size in zip(self.fft_sizes, self.hop_sizes):
            # Create Hann window on the same device as input
            window = torch.hann_window(fft_size).to(pred.device)
            
            # Compute STFTs
            pred_stft = torch.stft(
                pred, 
                n_fft=fft_size, 
                hop_length=hop_size, 
                window=window,
                return_complex=True
            )
            
            target_stft = torch.stft(
                target, 
                n_fft=fft_size, 
                hop_length=hop_size, 
                window=window,
                return_complex=True
            )
            
            # Extract phase and magnitude
            pred_phase = torch.angle(pred_stft)
            target_phase = torch.angle(target_stft)
            target_mag = torch.abs(target_stft)
            
            # Compute phase difference using circular distance
            # This handles wrapping around π/-π boundary properly
            phase_diff = torch.abs(torch.remainder(pred_phase - target_phase + torch.pi, 2 * torch.pi) - torch.pi)
            
            # Basic phase loss (normalized between 0-1)
            basic_phase_loss = torch.mean(phase_diff / torch.pi)
            
            # Advanced: Weight phase importance by magnitude
            # Higher magnitude = more perceptually important phase
            norm_target_mag = target_mag / (torch.mean(target_mag) + 1e-8)
            
            # Use 1-cos for smoother gradients around small phase differences
            weighted_phase_loss = torch.mean(norm_target_mag * (1.0 - torch.cos(phase_diff)))
            
            # Additionally focus on temporal phase coherence
            # Compute temporal derivative of phase
            target_phase_deriv = target_phase[:, :, 1:] - target_phase[:, :, :-1]
            pred_phase_deriv = pred_phase[:, :, 1:] - pred_phase[:, :, :-1]
            
            # Normalize derivatives to account for wrapping
            target_phase_deriv = torch.remainder(target_phase_deriv + torch.pi, 2 * torch.pi) - torch.pi
            pred_phase_deriv = torch.remainder(pred_phase_deriv + torch.pi, 2 * torch.pi) - torch.pi
            
            # Phase derivative consistency (temporal phase coherence)
            deriv_diff = torch.abs(target_phase_deriv - pred_phase_deriv)
            phase_coherence_loss = torch.mean(deriv_diff)
            
            # Combine losses
            phase_loss += (basic_phase_loss + phase_coherence_loss) * 0.5
            mag_weighted_phase_loss += weighted_phase_loss
        
        # Average across resolutions and combine approaches
        phase_loss /= len(self.fft_sizes)
        mag_weighted_phase_loss /= len(self.fft_sizes)
        
        # Final loss as weighted combination
        final_loss = 0.4 * phase_loss + 0.6 * mag_weighted_phase_loss
        
        return final_loss