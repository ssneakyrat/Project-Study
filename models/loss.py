import torch
import torch.nn as nn

class WaveletMSELoss(nn.Module):
    def __init__(self, levels=6, alpha=0.5):
        super(WaveletMSELoss, self).__init__()
        self.levels = levels
        self.alpha = alpha
        self.weights = self._calculate_weights()
        self.mse = nn.MSELoss(reduction='none')
        
    def _calculate_weights(self):
        """Calculate exponential weights for each wavelet level"""
        weights = []
        for j in range(self.levels + 1):  # +1 for approximation coefficients
            weights.append(2 ** (self.alpha * j))
        return torch.tensor(weights)
    
    def forward(self, pred_coeffs, target_coeffs):
        """
        Calculate weighted MSE loss across wavelet levels
        Args:
            pred_coeffs: Dictionary of predicted wavelet coefficients
            target_coeffs: Dictionary of target wavelet coefficients
        Returns:
            loss: Weighted MSE loss
        """
        device = pred_coeffs['a'].device
        self.weights = self.weights.to(device)
        
        # MSE for approximation coefficients
        mse_a = torch.mean(self.mse(pred_coeffs['a'], target_coeffs['a']))
        
        # MSE for detail coefficients at each level
        mse_d = []
        for j in range(self.levels):
            mse_d.append(torch.mean(self.mse(pred_coeffs['d'][j], target_coeffs['d'][j])))
        
        # Combine losses with weights
        mse_all = [mse_a] + mse_d
        weighted_loss = 0
        for j, mse in enumerate(mse_all):
            weighted_loss += self.weights[j] * mse
            
        return weighted_loss

class TimeDomainMSELoss(nn.Module):
    def __init__(self):
        super(TimeDomainMSELoss, self).__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, pred, target):
        """
        Calculate MSE in time domain
        Args:
            pred: Predicted audio signal
            target: Target audio signal
        Returns:
            loss: MSE loss
        """
        return self.mse(pred, target)

class CombinedLoss(nn.Module):
    def __init__(self, wavelet_weight=0.7, time_weight=0.3, levels=6, alpha=0.5):
        super(CombinedLoss, self).__init__()
        self.wavelet_weight = wavelet_weight
        self.time_weight = time_weight
        self.wavelet_loss = WaveletMSELoss(levels=levels, alpha=alpha)
        self.time_loss = TimeDomainMSELoss()
        
    def forward(self, pred, target, pred_coeffs, target_coeffs):
        """
        Calculate combined loss in wavelet and time domains
        Args:
            pred: Predicted audio signal
            target: Target audio signal
            pred_coeffs: Dictionary of predicted wavelet coefficients
            target_coeffs: Dictionary of target wavelet coefficients
        Returns:
            loss: Combined loss
        """
        w_loss = self.wavelet_loss(pred_coeffs, target_coeffs)
        t_loss = self.time_loss(pred, target)
        
        return self.wavelet_weight * w_loss + self.time_weight * t_loss