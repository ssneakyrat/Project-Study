import torch
import numpy as np
import pywt

class WaveletTransform:
    """Wavelet transform module for audio signals
    
    Implements Discrete Wavelet Transform (DWT) and its inverse (IDWT):
    DWT: W_ψ[j,k] = ∫ x(t)·ψ_{j,k}(t) dt where ψ_{j,k}(t) = 2^(-j/2)·ψ(2^(-j)t-k)
    
    Multi-resolution approximation:
    x(t) ≈ Σ_k c_{J,k}·φ_{J,k}(t) + Σ_{j=1}^J Σ_k d_{j,k}·ψ_{j,k}(t)
    
    Where c_{J,k} are approximation coefficients and d_{j,k} are detail coefficients
    """
    def __init__(self, wavelet='db4', level=3):
        self.wavelet = wavelet
        self.level = level
        self.coeffs_lengths = None
        self.coeffs_shapes = None
    
    def forward(self, x):
        """Apply DWT to batch of signals and return flattened coefficients"""
        batch_size = x.shape[0]
        device = x.device
        coeffs_list = []
        
        for i in range(batch_size):
            signal = x[i].cpu().numpy()
            
            # Apply DWT
            coeffs = pywt.wavedec(signal, self.wavelet, level=self.level)
            
            # Store coefficient lengths for reconstruction
            if i == 0:
                self.coeffs_lengths = [len(c) for c in coeffs]
                self.coeffs_shapes = [c.shape for c in coeffs]
            
            # Flatten coefficients
            flat_coeffs = np.concatenate([c.flatten() for c in coeffs])
            coeffs_list.append(flat_coeffs)
        
        # Stack and convert to tensor
        coeffs_tensor = torch.tensor(np.stack(coeffs_list), dtype=torch.float32, device=device)
        return coeffs_tensor
    
    def inverse(self, coeffs_tensor):
        """Reconstruct signals from flattened wavelet coefficients
        
        IDWT: x(t) = Σ_{j,k} W_ψ[j,k]·ψ_{j,k}(t)
        """
        batch_size = coeffs_tensor.shape[0]
        device = coeffs_tensor.device
        reconstructed = []
        
        for i in range(batch_size):
            flat_coeffs = coeffs_tensor[i].cpu().detach().numpy()
            
            # Split coefficients based on stored lengths
            coeffs = []
            start_idx = 0
            
            for j, shape in enumerate(self.coeffs_shapes):
                size = np.prod(shape)
                coeff = flat_coeffs[start_idx:start_idx+size].reshape(shape)
                coeffs.append(coeff)
                start_idx += size
            
            # Reconstruct signal
            rec_signal = pywt.waverec(coeffs, self.wavelet)
            reconstructed.append(rec_signal)
        
        # Match length of original signals by trimming or padding
        max_len = max(len(r) for r in reconstructed)
        for i in range(len(reconstructed)):
            if len(reconstructed[i]) < max_len:
                pad_len = max_len - len(reconstructed[i])
                reconstructed[i] = np.pad(reconstructed[i], (0, pad_len), 'constant')
        
        return torch.tensor(np.stack(reconstructed), dtype=torch.float32, device=device)
    
    def get_output_dim(self, input_length):
        """Calculate the dimension of wavelet coefficients for a given input length"""
        # Create a dummy signal
        dummy = np.zeros(input_length)
        
        # Apply DWT
        coeffs = pywt.wavedec(dummy, self.wavelet, level=self.level)
        
        # Calculate total size
        total_size = sum(len(c) for c in coeffs)
        
        return total_size