import torch
import torch.nn as nn

class PositionAwareWaveletLoss(nn.Module):
    """
    Enhanced position-sensitive wavelet loss function that directly addresses
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
                # Boost high-frequency importance by 3x (increased from 2.5x)
                weights.append(2 ** (self.alpha * j) * 3.0)
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
        
        # Enhanced position-dependent MSE for detail coefficients at each level
        mse_d = []
        for j in range(self.levels):
            # Base MSE computation
            level_mse_values = self.mse(pred_coeffs['d'][j], target_coeffs['d'][j])
            
            # Create position-dependent weighting mask with stronger correction
            seq_len = level_mse_values.shape[1]
            position_weights = torch.ones((1, seq_len), device=device)
            
            if j == 0:  # Level 1 (highest frequency) - boost right side (weakness area)
                # Apply exponential weighting to emphasize right side errors (80% good on left)
                # Use power function for stronger emphasis
                position = torch.linspace(0, 1, seq_len, device=device).view(1, -1)
                position_weights = 1.0 + 4.0 * torch.pow(position, 2.0)  # 1.0 at left, up to 5.0 at right
                
            elif j == 1:  # Level 2 - boost left side (weakness area)
                # Create weights emphasizing the left side (50% good on right)
                position = torch.linspace(0, 1, seq_len, device=device).view(1, -1)
                position_weights = 5.0 - 4.0 * position  # 5.0 at left, 1.0 at right
                
            elif j == 2:  # Level 3 - strong left boost (weakness area)
                # Create weights with stronger emphasis on the left side (30% good on right)
                position = torch.linspace(0, 1, seq_len, device=device).view(1, -1)
                position_weights = 7.0 - 6.0 * torch.pow(position, 0.5)  # 7.0 at left, decreasing to 1.0
                
            # Apply position weights to MSE values - ensure batch dimension is correct
            batch_size = level_mse_values.shape[0]
            weighted_mse = level_mse_values * position_weights.expand(batch_size, -1)
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
    Enhanced loss component that focuses on preserving spectral gradients
    with targeted position-aware correction
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
            level_weight = 2.0 / (j + 1)  # Increased from 1.0 to 2.0
            
            # Get detail coefficients
            pred_d = pred_coeffs['d'][j]
            target_d = target_coeffs['d'][j]
            
            # Calculate gradients (transitions between adjacent coefficients)
            # This captures the rate of change in the frequency domain
            pred_grad = torch.abs(torch.diff(pred_d, dim=1))
            target_grad = torch.abs(torch.diff(target_d, dim=1))
            
            # Create enhanced position-dependent weighting mask
            seq_len = pred_grad.shape[1]
            position = torch.linspace(0, 1, seq_len, device=pred_d.device).view(1, -1)
            
            if j == 0:  # Level 1 (highest frequency) - target right side weakness
                # Strong emphasis on right side gradients
                position_weights = 1.0 + 4.0 * torch.pow(position, 2.0)  # 1.0 → 5.0
            elif j == 1:  # Level 2 - target left side weakness
                # Strong emphasis on left side gradients
                position_weights = 5.0 - 4.0 * position  # 5.0 → 1.0
            elif j == 2:  # Level 3 - stronger left emphasis for weakness
                # Very strong emphasis on left side
                position_weights = 6.0 - 5.0 * torch.pow(position, 0.5)  # 6.0 → 1.0
            else:
                position_weights = torch.ones_like(position)
                
            # Apply position-dependent weighting to gradients
            batch_size = pred_grad.shape[0]
            weighted_pred_grad = pred_grad * position_weights.expand(batch_size, -1)
            weighted_target_grad = target_grad * position_weights.expand(batch_size, -1)
            
            # Mean squared error on weighted gradients
            grad_mse = torch.mean((weighted_pred_grad - weighted_target_grad) ** 2)
            
            # Add weighted loss
            loss += level_weight * grad_mse
            
        return loss

class DetailPreservationLoss(nn.Module):
    """
    New specialized loss component focused on detail preservation in specific regions
    where reconstruction is weakest
    """
    def __init__(self):
        super(DetailPreservationLoss, self).__init__()
        
    def forward(self, pred_coeffs, target_coeffs):
        """
        Calculate detail preservation loss targeting specific regions in each level
        """
        loss = 0.0
        
        # Apply to first 3 levels only (where issues are observed)
        for j in range(min(3, len(pred_coeffs['d']))):
            # Get coefficients
            pred_d = pred_coeffs['d'][j]
            target_d = target_coeffs['d'][j]
            
            # Get shape and device
            batch_size, seq_len = pred_d.shape
            device = pred_d.device
            
            # Create position-based mask targeting the problem areas
            position = torch.linspace(0, 1, seq_len, device=device).view(1, -1)
            
            if j == 0:  # Level 1 - target right-side detail preservation (80% left good)
                # Identify right portion (where it's weakest)
                mask = (position > 0.7).float()
                # Extract coefficient differences in targeted area
                target_vals = target_d * mask
                pred_vals = pred_d * mask
                # Calculate loss on energy differences to ensure details are preserved
                energy_target = torch.sum(target_vals**2, dim=1)
                energy_pred = torch.sum(pred_vals**2, dim=1)
                # Detail energy preservation loss
                level_loss = torch.mean(torch.abs(energy_target - energy_pred)) / seq_len
                
            elif j == 1:  # Level 2 - target left-side detail preservation (50% right good)
                # Identify left portion (where it's weakest)
                mask = (position < 0.5).float()
                # Extract coefficient differences in targeted area
                target_vals = target_d * mask
                pred_vals = pred_d * mask
                # Calculate loss on energy differences to ensure details are preserved
                energy_target = torch.sum(target_vals**2, dim=1)
                energy_pred = torch.sum(pred_vals**2, dim=1)
                # Detail energy preservation loss
                level_loss = torch.mean(torch.abs(energy_target - energy_pred)) / seq_len
                
            elif j == 2:  # Level 3 - target left-side detail preservation (30% right good)
                # Identify left portion (where it's weakest)
                mask = (position < 0.7).float()
                # Extract coefficient differences in targeted area
                target_vals = target_d * mask
                pred_vals = pred_d * mask
                # Calculate loss on energy differences to ensure details are preserved
                energy_target = torch.sum(target_vals**2, dim=1)
                energy_pred = torch.sum(pred_vals**2, dim=1)
                # Detail energy preservation loss
                level_loss = torch.mean(torch.abs(energy_target - energy_pred)) / seq_len
                
            else:
                level_loss = 0.0
            
            # Weight by level importance (higher frequencies more important)
            level_weight = 3.0 / (j + 1)
            loss += level_weight * level_loss
            
        return loss

class CombinedLoss(nn.Module):
    def __init__(self, wavelet_weight=0.7, time_weight=0.3, levels=6, alpha=0.5):
        super(CombinedLoss, self).__init__()
        self.wavelet_weight = wavelet_weight
        self.time_weight = time_weight
        self.wavelet_loss = PositionAwareWaveletLoss(levels=levels, alpha=alpha)
        self.time_loss = TimeDomainMSELoss()
        self.spectral_gradient_loss = SpectralGradientLoss()
        self.detail_preservation_loss = DetailPreservationLoss()
        
    def forward(self, pred, target, pred_coeffs, target_coeffs):
        """
        Calculate combined loss with enhanced position-aware components
        Args:
            pred: Predicted audio signal
            target: Target audio signal
            pred_coeffs: Dictionary of predicted wavelet coefficients
            target_coeffs: Dictionary of target wavelet coefficients
        Returns:
            loss: Combined loss
        """
        # Base wavelet and time domain losses
        w_loss = self.wavelet_loss(pred_coeffs, target_coeffs)
        t_loss = self.time_loss(pred, target)
        
        # Add specialized losses to address the observed position-dependent weaknesses
        gradient_loss = self.spectral_gradient_loss(pred_coeffs, target_coeffs)
        detail_loss = self.detail_preservation_loss(pred_coeffs, target_coeffs)
        
        # Combine all losses with adjusted weights
        # Increased weights for specialized components that target the observed issues
        combined_loss = (self.wavelet_weight * w_loss + 
                         self.time_weight * t_loss + 
                         0.2 * gradient_loss +    # Increased from 0.1 to 0.2
                         0.15 * detail_loss)      # New component
        
        return combined_loss