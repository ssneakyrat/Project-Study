import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

class AudioReconstructionLoss(nn.Module):
    def __init__(self, sample_rate=16000, alpha=1.0, beta=1.0, gamma=0.1):
        super().__init__()
        self.sample_rate = sample_rate
        
        # Loss weights
        self.alpha = alpha  # L1 loss weight
        self.beta = beta    # STFT loss weight
        self.gamma = gamma  # Adversarial loss weight (not used in initial test)
        
        # STFT loss components
        self.stft_loss = MultiResolutionSTFTLoss()
        
    def forward(self, y_pred, y_true):
        """
        Calculate weighted combination of losses
        
        Args:
            y_pred: Predicted audio [batch_size, 1, length]
            y_true: Target audio [batch_size, 1, length]
            
        Returns:
            total_loss: Combined loss value
            loss_dict: Dictionary of individual loss components
        """
        # Ensure same length
        min_len = min(y_pred.shape[-1], y_true.shape[-1])
        y_pred = y_pred[..., :min_len]
        y_true = y_true[..., :min_len]
        
        # L1 loss
        l1_loss = F.l1_loss(y_pred, y_true)
        
        # STFT loss
        stft_loss = self.stft_loss(y_pred, y_true)
        
        # Total loss
        total_loss = self.alpha * l1_loss + self.beta * stft_loss
        
        # Return total loss and components
        return total_loss, {
            'l1_loss': l1_loss.item(),
            'stft_loss': stft_loss.item()
        }

class MultiResolutionSTFTLoss(nn.Module):
    """Multi-resolution STFT loss."""
    
    def __init__(self, fft_sizes=[512, 1024, 2048], hop_sizes=None, win_lengths=None):
        super().__init__()
        if hop_sizes is None:
            hop_sizes = [fft_size // 4 for fft_size in fft_sizes]
        if win_lengths is None:
            win_lengths = fft_sizes
            
        self.stft_losses = nn.ModuleList([
            STFTLoss(fft_size, hop_size, win_length)
            for fft_size, hop_size, win_length in zip(fft_sizes, hop_sizes, win_lengths)
        ])
    
    def forward(self, y_pred, y_true):
        """Calculate multi-resolution STFT loss."""
        loss = 0.0
        for stft_loss in self.stft_losses:
            loss += stft_loss(y_pred, y_true)
        return loss / len(self.stft_losses)

class STFTLoss(nn.Module):
    """STFT loss module."""
    
    def __init__(self, fft_size=1024, hop_size=256, win_length=1024):
        super().__init__()
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_length = win_length
        self.register_buffer('window', torch.hann_window(win_length))
    
    def forward(self, y_pred, y_true):
        """Calculate STFT loss."""
        # Compute STFT for predicted audio
        stft_pred = torch.stft(
            y_pred.squeeze(1),
            n_fft=self.fft_size,
            hop_length=self.hop_size,
            win_length=self.win_length,
            window=self.window,
            return_complex=True
        )
        
        # Compute STFT for ground truth audio
        stft_true = torch.stft(
            y_true.squeeze(1),
            n_fft=self.fft_size,
            hop_length=self.hop_size,
            win_length=self.win_length,
            window=self.window,
            return_complex=True
        )
        
        # Magnitude loss
        mag_pred = torch.abs(stft_pred)
        mag_true = torch.abs(stft_true)
        mag_loss = F.l1_loss(mag_pred, mag_true)
        
        # Log magnitude loss
        log_mag_pred = torch.log(torch.clamp(mag_pred, min=1e-7))
        log_mag_true = torch.log(torch.clamp(mag_true, min=1e-7))
        log_mag_loss = F.l1_loss(log_mag_pred, log_mag_true)
        
        return mag_loss + log_mag_loss