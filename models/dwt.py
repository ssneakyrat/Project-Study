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
    
    def threshold_coeffs(self, coeffs, lambda_values=None):
        """
        Apply adaptive thresholding to coefficients
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
            N = d.shape[1]
            std_d = torch.std(d, dim=1, keepdim=True)
            T = lambda_values[j] * std_d * torch.sqrt(torch.tensor(2.0 * np.log(N), device=d.device))
            
            # Soft thresholding
            d_abs = torch.abs(d)
            mask = d_abs > T
            thresholded_d = torch.zeros_like(d)
            thresholded_d[mask] = torch.sign(d[mask]) * (d_abs[mask] - T[mask[0]])
            
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