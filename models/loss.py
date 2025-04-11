import torch
import torch.nn as nn

class PositionAwareWaveletLoss(nn.Module):
    """
    Position-sensitive wavelet loss function that addresses
    the observed spatial imbalance in wavelet coefficient reconstruction
    """
    def __init__(self, levels=4, alpha=0.6):
        super(PositionAwareWaveletLoss, self).__init__()
        self.levels = levels
        self.alpha = alpha
        self.weights = self._calculate_weights()
        self.mse = nn.MSELoss(reduction='none')
        
    def _calculate_weights(self):
        """
        Calculate exponential weights for each wavelet level with boost for high-frequency details
        """
        weights = []
        for j in range(self.levels + 1):  # +1 for approximation coefficients
            if j == 0:  # Approximation coefficients
                weights.append(2 ** (self.alpha * j))
            elif j < 3:  # First 3 detail levels (high frequencies)
                # Boost high-frequency importance by 2.5x
                weights.append(2 ** (self.alpha * j) * 2.5)
            else:
                weights.append(2 ** (self.alpha * j))
        return torch.tensor(weights)
        
    def forward(self, pred_coeffs, target_coeffs):
        """
        Calculate position-aware weighted MSE loss across wavelet levels
        Args:
            pred_coeffs: Dictionary of predicted wavelet coefficients
            target_coeffs: Dictionary of target wavelet coefficients
        Returns:
            loss: Weighted position-aware MSE loss
        """
        device = pred_coeffs['a'].device
        self.weights = self.weights.to(device)
        
        # MSE for approximation coefficients
        mse_a = torch.mean(self.mse(pred_coeffs['a'], target_coeffs['a']))
        
        # Position-dependent MSE for detail coefficients at each level
        mse_d = []
        for j in range(self.levels):
            # Base MSE computation
            level_mse_values = self.mse(pred_coeffs['d'][j], target_coeffs['d'][j])
            
            # Create position-dependent weighting mask based on the observed reconstruction issues
            seq_len = level_mse_values.shape[1]
            position_weights = torch.ones((1, seq_len), device=device)
            
            if j == 0:  # Level 1 (highest frequency) - boost right side
                # Gradually increase weight from left (1.0) to right (2.0)
                position_weights = torch.linspace(1.0, 2.0, seq_len, device=device).view(1, -1)
            elif j == 1:  # Level 2 - boost right side more
                # Create weights emphasizing the right half
                mid_point = seq_len // 2
                left_weights = torch.ones(mid_point, device=device)
                right_weights = torch.linspace(1.0, 2.5, seq_len - mid_point, device=device)
                position_weights = torch.cat([left_weights, right_weights], dim=0).view(1, -1)
            elif j == 2:  # Level 3 - strong right boost
                # Create weights with stronger emphasis on the right side
                right_start = int(0.6 * seq_len)
                position_weights[:, right_start:] = 3.0
                
            # Apply position weights to MSE values
            weighted_mse = level_mse_values * position_weights.unsqueeze(0)
            level_mse = torch.mean(weighted_mse)
            mse_d.append(level_mse)
        
        # Combine losses with weights
        mse_all = [mse_a] + mse_d
        weighted_loss = 0
        for j, mse in enumerate(mse_all):
            weighted_loss += self.weights[j] * mse
            
        return weighted_loss

class WaveletMSELoss(nn.Module):
    def __init__(self, levels=6, alpha=0.5):
        super(WaveletMSELoss, self).__init__()
        self.levels = levels
        self.alpha = alpha
        self.weights = self._calculate_weights()
        self.mse = nn.MSELoss(reduction='none')
        
    def _calculate_weights(self):
        """
        Calculate exponential weights for each wavelet level with boost for high-frequency details
        """
        weights = []
        for j in range(self.levels + 1):  # +1 for approximation coefficients
            if j == 0:  # Approximation coefficients
                weights.append(2 ** (self.alpha * j))
            elif j < 3:  # First 3 detail levels (high frequencies)
                # Boost high-frequency importance by 2.5x (increased from 2.0x)
                weights.append(2 ** (self.alpha * j) * 2.5)
            else:
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
            level_mse = torch.mean(self.mse(pred_coeffs['d'][j], target_coeffs['d'][j]))
            mse_d.append(level_mse)
        
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

class SpectralGradientLoss(nn.Module):
    """
    New loss component that focuses on preserving spectral gradients
    which are critical for accurate high-frequency reproduction
    """
    def __init__(self):
        super(SpectralGradientLoss, self).__init__()
        
    def forward(self, pred_coeffs, target_coeffs, levels=3):
        """
        Calculate spectral gradient loss focusing on high frequencies
        Args:
            pred_coeffs: Predicted wavelet coefficients
            target_coeffs: Target wavelet coefficients
            levels: Number of high-frequency levels to consider
        Returns:
            loss: Spectral gradient loss
        """
        loss = 0.0
        
        # Process only high-frequency detail coefficients
        for j in range(min(levels, len(pred_coeffs['d']))):
            # Higher weight for higher frequencies (smaller j)
            level_weight = 1.0 / (j + 1)
            
            # Get detail coefficients
            pred_d = pred_coeffs['d'][j]
            target_d = target_coeffs['d'][j]
            
            # Calculate gradients (transitions between adjacent coefficients)
            # This captures the rate of change in the frequency domain
            pred_grad = torch.abs(torch.diff(pred_d, dim=1))
            target_grad = torch.abs(torch.diff(target_d, dim=1))
            
            # Create position-dependent weighting mask
            seq_len = pred_grad.shape[1]
            position_weights = torch.ones((1, seq_len), device=pred_d.device)
            
            if j == 0:  # Level 1 (highest frequency) - emphasize right side
                position_weights = torch.linspace(0.8, 2.0, seq_len, device=pred_d.device).view(1, -1)
            elif j == 1:  # Level 2 - emphasize right side
                mid_point = seq_len // 2
                position_weights[:, mid_point:] = 2.0
            elif j == 2:  # Level 3 - strong right emphasis
                right_start = int(0.6 * seq_len)
                position_weights[:, right_start:] = 2.5
            
            # Apply position-dependent weighting to gradients
            weighted_pred_grad = pred_grad * position_weights.unsqueeze(0)
            weighted_target_grad = target_grad * position_weights.unsqueeze(0)
            
            # Mean squared error on weighted gradients
            grad_mse = torch.mean((weighted_pred_grad - weighted_target_grad) ** 2)
            
            # Add weighted loss
            loss += level_weight * grad_mse
            
        return loss

class CombinedLoss(nn.Module):
    def __init__(self, wavelet_weight=0.7, time_weight=0.3, levels=6, alpha=0.5):
        super(CombinedLoss, self).__init__()
        self.wavelet_weight = wavelet_weight
        self.time_weight = time_weight
        self.wavelet_loss = PositionAwareWaveletLoss(levels=levels, alpha=alpha)
        self.time_loss = TimeDomainMSELoss()
        self.spectral_gradient_loss = SpectralGradientLoss()
        
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
        
        # Add specialized high-frequency loss component
        high_freq_loss = 0
        for j in range(min(3, len(pred_coeffs['d']))):  # Focus on first 3 levels
            # Inverse frequency weighting - higher frequencies (j=0) get higher weight
            level_weight = 1.5 / (j + 1)
            
            # Position-dependent weighting based on observed reconstruction issues
            seq_len = pred_coeffs['d'][j].shape[1]
            position_weights = torch.ones((1, seq_len), device=pred_coeffs['d'][j].device)
            
            if j == 0:  # Level 1 - emphasize right side (where it's weak)
                position_weights = torch.linspace(1.0, 2.0, seq_len, device=pred_coeffs['d'][j].device).view(1, -1)
            elif j == 1:  # Level 2 - emphasize right side more
                right_half = seq_len // 2
                position_weights[:, right_half:] = 2.0
            elif j == 2:  # Level 3 - strong right emphasis
                right_start = int(0.6 * seq_len)
                position_weights[:, right_start:] = 2.5
                
            # Apply position weights to MSE calculation
            coeff_diff = (pred_coeffs['d'][j] - target_coeffs['d'][j]) ** 2
            weighted_diff = coeff_diff * position_weights.unsqueeze(0)
            coeff_mse = torch.mean(weighted_diff)
            
            high_freq_loss += level_weight * coeff_mse
        
        # Add spectral gradient loss with position awareness
        gradient_loss = self.spectral_gradient_loss(pred_coeffs, target_coeffs)
        
        # Combine all losses with adjusted weights
        # Increased weight for high-frequency components (from 0.1 to 0.15)
        # Added new gradient loss component (0.1 weight)
        combined_loss = (self.wavelet_weight * w_loss + 
                        self.time_weight * t_loss + 
                        0.15 * high_freq_loss +
                        0.1 * gradient_loss)
        
        return combined_loss