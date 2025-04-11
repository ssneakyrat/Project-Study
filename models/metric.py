import torch
import torch.nn as nn
import torch.nn.functional as F

class SignalToNoiseRatio(nn.Module):
    def __init__(self, reduction='mean'):
        super(SignalToNoiseRatio, self).__init__()
        self.reduction = reduction
        
    def forward(self, pred, target):
        """
        Calculate Signal-to-Noise Ratio (SNR) in dB
        Args:
            pred: Predicted audio signal of shape [B, T]
            target: Target audio signal of shape [B, T]
        Returns:
            snr: SNR in dB
        """
        # Calculate signal power
        signal_power = torch.sum(target ** 2, dim=1)
        
        # Calculate noise power
        noise = target - pred
        noise_power = torch.sum(noise ** 2, dim=1)
        
        # Add small epsilon to avoid division by zero
        eps = 1e-10
        
        # Calculate SNR
        snr = 10 * torch.log10(signal_power / (noise_power + eps))
        
        if self.reduction == 'mean':
            return torch.mean(snr)
        elif self.reduction == 'sum':
            return torch.sum(snr)
        else:
            return snr

class SpectralConvergence(nn.Module):
    def __init__(self, reduction='mean'):
        super(SpectralConvergence, self).__init__()
        self.reduction = reduction
        
    def forward(self, pred, target):
        """
        Calculate Spectral Convergence
        Args:
            pred: Predicted audio signal of shape [B, T]
            target: Target audio signal of shape [B, T]
        Returns:
            sc: Spectral Convergence (lower is better)
        """
        # Calculate FFT
        pred_fft = torch.fft.rfft(pred, dim=1)
        target_fft = torch.fft.rfft(target, dim=1)
        
        # Calculate spectral convergence
        num = torch.norm(torch.abs(target_fft) - torch.abs(pred_fft), p=2, dim=1)
        denom = torch.norm(torch.abs(target_fft), p=2, dim=1)
        
        # Add small epsilon to avoid division by zero
        eps = 1e-10
        sc = num / (denom + eps)
        
        if self.reduction == 'mean':
            return torch.mean(sc)
        elif self.reduction == 'sum':
            return torch.sum(sc)
        else:
            return sc

class LogSpectralDistance(nn.Module):
    def __init__(self, reduction='mean'):
        super(LogSpectralDistance, self).__init__()
        self.reduction = reduction
        
    def forward(self, pred, target):
        """
        Calculate Log Spectral Distance
        Args:
            pred: Predicted audio signal of shape [B, T]
            target: Target audio signal of shape [B, T]
        Returns:
            lsd: Log Spectral Distance (lower is better)
        """
        # Calculate FFT
        pred_fft = torch.fft.rfft(pred, dim=1)
        target_fft = torch.fft.rfft(target, dim=1)
        
        # Calculate magnitude spectra
        pred_mag = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)
        
        # Add small epsilon to avoid log(0)
        eps = 1e-10
        
        # Calculate log spectral distance
        lsd = torch.sqrt(torch.mean((10 * torch.log10(pred_mag + eps) - 10 * torch.log10(target_mag + eps)) ** 2, dim=1))
        
        if self.reduction == 'mean':
            return torch.mean(lsd)
        elif self.reduction == 'sum':
            return torch.sum(lsd)
        else:
            return lsd

class MultiMetric:
    def __init__(self):
        self.snr = SignalToNoiseRatio()
        self.sc = SpectralConvergence()
        self.lsd = LogSpectralDistance()
        
    def __call__(self, pred, target):
        """
        Calculate multiple metrics
        Args:
            pred: Predicted audio signal
            target: Target audio signal
        Returns:
            metrics: Dictionary of metrics
        """
        return {
            'snr': self.snr(pred, target).item(),
            'sc': self.sc(pred, target).item(),
            'lsd': self.lsd(pred, target).item()
        }