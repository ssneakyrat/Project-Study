import torch
import numpy as np
import pywt

class DWT(torch.nn.Module):
    def __init__(self, wave='db4', level=6, mode='zero'):
        super(DWT, self).__init__()
        self.wave = wave
        self.level = level
        self.mode = mode
        
    def forward(self, x):
        """
        Apply DWT to input tensor
        Args:
            x: Input tensor of shape [B, T]
        Returns:
            coeffs: Dictionary of wavelet coefficients
        """
        # Convert to numpy for PyWavelets
        device = x.device
        x_np = x.detach().cpu().numpy()
        batch_size = x.shape[0]
        
        # Initialize coefficient lists
        approx = []
        details = [[] for _ in range(self.level)]
        
        # Apply DWT to each sample in batch
        for i in range(batch_size):
            coeffs = pywt.wavedec(x_np[i], self.wave, mode=self.mode, level=self.level)
            approx.append(coeffs[0])
            for j in range(self.level):
                details[j].append(coeffs[j+1])
        
        # Convert back to torch tensors
        approx = torch.tensor(np.array(approx), device=device)
        details = [torch.tensor(np.array(d), device=device) for d in details]
        
        # Create coefficient dictionary
        coeffs_dict = {
            'a': approx,
            'd': details
        }
        
        return coeffs_dict
    
    def normalize_coeffs(self, coeffs, means=None, stds=None):
        """
        Normalize wavelet coefficients
        Args:
            coeffs: Dictionary of wavelet coefficients
            means, stds: Optional precomputed statistics
        Returns:
            normalized_coeffs: Normalized coefficients
            stats: Means and stds for each level
        """
        normalized_coeffs = {'a': None, 'd': []}
        stats = {'means': [], 'stds': []}
        
        # Normalize approximation coefficients
        a = coeffs['a']
        if means is None or stds is None:
            mean_a = torch.mean(a, dim=1, keepdim=True)
            std_a = torch.std(a, dim=1, keepdim=True) + 1e-8
        else:
            mean_a = means[0]
            std_a = stds[0]
            
        normalized_coeffs['a'] = (a - mean_a) / std_a
        stats['means'].append(mean_a)
        stats['stds'].append(std_a)
        
        # Normalize detail coefficients for each level
        for j, d in enumerate(coeffs['d']):
            if means is None or stds is None:
                mean_d = torch.mean(d, dim=1, keepdim=True)
                std_d = torch.std(d, dim=1, keepdim=True) + 1e-8
            else:
                mean_d = means[j+1]
                std_d = stds[j+1]
                
            normalized_d = (d - mean_d) / std_d
            normalized_coeffs['d'].append(normalized_d)
            stats['means'].append(mean_d)
            stats['stds'].append(std_d)
            
        return normalized_coeffs, stats
    
    def _position_adaptive_scale(self, level, rel_position):
        """
        Generate position-dependent scaling factor for thresholding
        Args:
            level: Wavelet decomposition level
            rel_position: Relative position (0.0 to 1.0)
        Returns:
            scale: Position-dependent scaling factor
        """
        # Different scaling patterns for different levels to address the observed issues
        if level == 0:  # Level 1 (highest frequency) - 80% left reconstruction
            # Gradually increase threshold from left to right (lower threshold = more detail preserved)
            return 0.6 + 0.8 * rel_position  # 0.6 at left, 1.4 at right
        elif level == 1:  # Level 2 - 50% right reconstruction
            # Gradually decrease threshold from left to right
            return 1.4 - 0.8 * rel_position  # 1.4 at left, 0.6 at right
        elif level == 2:  # Level 3 - 30% right reconstruction
            # Similar pattern to level 2 but more pronounced
            return 1.6 - 1.0 * rel_position  # 1.6 at left, 0.6 at right
        else:
            # Neutral scaling for other levels
            return 1.0
    
    def threshold_coeffs(self, coeffs, lambda_values=None):
        """
        Apply position-aware adaptive thresholding to coefficients
        Args:
            coeffs: Dictionary of wavelet coefficients
            lambda_values: Optional threshold parameters
        Returns:
            thresholded_coeffs: Thresholded coefficients
        """
        thresholded_coeffs = {'a': coeffs['a'], 'd': []}
        
        # Default lambda values if not provided
        if lambda_values is None:
            lambda_values = [0.5] * self.level
        
        # Apply thresholding to detail coefficients
        for j, d in enumerate(coeffs['d']):
            batch_size = d.shape[0]
            N = d.shape[1]
            std_d = torch.std(d, dim=1, keepdim=True)
            
            # Frequency-aware adaptive scaling factor (gentler on high frequencies)
            scale_factor = max(0.2, 1.0 - 0.2 * (self.level - j - 1))
            
            # Create position-dependent threshold mask
            position_mask = torch.ones((batch_size, N), device=d.device)
            for pos in range(N):
                # Get position-adaptive scale for this level and relative position
                pos_scale = self._position_adaptive_scale(j, pos/N)
                position_mask[:, pos] = pos_scale
            
            # Universal threshold with position-aware scaling
            T_base = scale_factor * lambda_values[j] * std_d * torch.sqrt(torch.tensor(2.0 * np.log(N), device=d.device))
            T = T_base * position_mask.unsqueeze(1) if len(T_base.shape) > 2 else T_base * position_mask
            
            # Soft thresholding with proper broadcasting: sign(x) * max(|x| - T, 0)
            d_abs = torch.abs(d)
            thresholded_d = torch.sign(d) * torch.clamp(d_abs - T, min=0)
            
            thresholded_coeffs['d'].append(thresholded_d)
            
        return thresholded_coeffs

class IDWT(torch.nn.Module):
    def __init__(self, wave='db4', mode='zero'):
        super(IDWT, self).__init__()
        self.wave = wave
        self.mode = mode
        
    def forward(self, coeffs):
        """
        Apply IDWT to wavelet coefficients
        Args:
            coeffs: Dictionary of wavelet coefficients
        Returns:
            x: Reconstructed signal
        """
        # Convert to numpy for PyWavelets
        device = coeffs['a'].device
        a_np = coeffs['a'].detach().cpu().numpy()
        d_np = [d.detach().cpu().numpy() for d in coeffs['d']]
        
        batch_size = a_np.shape[0]
        reconstructed = []
        
        # Apply IDWT to each sample in batch
        for i in range(batch_size):
            # Prepare coefficients
            coeffs_list = [a_np[i]] + [d[i] for d in d_np]
            
            # Reconstruct signal
            rec = pywt.waverec(coeffs_list, self.wave, mode=self.mode)
            reconstructed.append(rec)
        
        # Convert back to torch tensor
        reconstructed = torch.tensor(np.array(reconstructed), device=device)
        
        return reconstructed