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
        """Initialize wavelet transform with specified wavelet and decomposition level
        
        Args:
            wavelet (str): Wavelet family to use (e.g., 'db4', 'haar')
            level (int): Decomposition level
        """
        self.wavelet = wavelet
        self.level = level
        
        # Will be dynamically calculated based on input audio length
        self.total_coeffs = None
        self.slice_indices = None
        self.expected_length = None
        
        # Initialize wavelet to validate it exists
        pywt.Wavelet(wavelet)
    
    def _calculate_expected_shapes(self, length):
        """Calculate expected coefficient shapes for a given input length
        
        For a signal of length N with decomposition level L:
        - Approx coefficients at level L: N_L ≈ N/2^L
        - Detail coefficients at level j: N_j ≈ N/2^j
        
        Returns a tuple of: (total_coeffs, slice_indices)
        """
        # Create a dummy signal to calculate expected shapes
        dummy = np.zeros(length)
        
        # Apply DWT to get coefficient sizes
        coeffs = pywt.wavedec(dummy, self.wavelet, level=self.level)
        
        # Calculate total size and slice indices for reconstruction
        total_coeffs = sum(len(c) for c in coeffs)
        
        # Calculate slice indices for reconstruction
        slice_indices = []
        start_idx = 0
        for c in coeffs:
            end_idx = start_idx + len(c)
            slice_indices.append((start_idx, end_idx, c.shape))
            start_idx = end_idx
            
        return total_coeffs, slice_indices
    
    def forward(self, x):
        """Apply DWT to batch of signals and return flattened coefficients
        
        Args:
            x: Input audio tensor [batch_size, audio_length]
            
        Returns:
            Flattened wavelet coefficients tensor [batch_size, total_coeffs]
        """
        batch_size = x.shape[0]
        device = x.device
        
        # If first time or different audio length, calculate expected shapes
        if self.total_coeffs is None or x.shape[1] != self.expected_length:
            self.expected_length = x.shape[1]
            self.total_coeffs, self.slice_indices = self._calculate_expected_shapes(x.shape[1])
            
        # Pre-allocate output tensor for efficiency
        coeffs_tensor = torch.zeros((batch_size, self.total_coeffs), dtype=torch.float32, device='cpu')
        
        # Process each sample in the batch
        for i in range(batch_size):
            # Move to CPU for PyWavelets (which doesn't support GPU)
            signal = x[i].cpu().numpy()
            
            # Apply DWT
            coeffs = pywt.wavedec(signal, self.wavelet, level=self.level)
            
            # Concatenate coefficients directly into pre-allocated tensor
            start_idx = 0
            for c in coeffs:
                end_idx = start_idx + len(c)
                coeffs_tensor[i, start_idx:end_idx] = torch.from_numpy(c.astype(np.float32))
                start_idx = end_idx
        
        # Move back to original device
        return coeffs_tensor.to(device)
    
    def inverse(self, coeffs_tensor):
        """Reconstruct signals from flattened wavelet coefficients
        
        IDWT: x(t) = Σ_{j,k} W_ψ[j,k]·ψ_{j,k}(t)
        
        Args:
            coeffs_tensor: Flattened coefficients [batch_size, total_coeffs]
            
        Returns:
            Reconstructed audio [batch_size, audio_length]
        """
        batch_size = coeffs_tensor.shape[0]
        device = coeffs_tensor.device
        
        # Move to CPU for processing with PyWavelets
        coeffs_tensor_cpu = coeffs_tensor.cpu()
        
        # Use a more efficient approach - pre-allocate output numpy array
        # Estimate the maximum possible length based on expected_length
        max_possible_length = int(self.expected_length * 1.1)  # Add 10% to account for boundary effects
        reconstructed = np.zeros((batch_size, max_possible_length), dtype=np.float32)
        actual_lengths = []
        
        for i in range(batch_size):
            # Extract coefficients for this sample
            coeffs = []
            for start_idx, end_idx, shape in self.slice_indices:
                coeff = coeffs_tensor_cpu[i, start_idx:end_idx].detach().numpy().reshape(shape)
                coeffs.append(coeff)
            
            # Reconstruct signal
            rec_signal = pywt.waverec(coeffs, self.wavelet)
            actual_lengths.append(len(rec_signal))
            
            # Store in pre-allocated array, handling potential length differences
            reconstructed[i, :len(rec_signal)] = rec_signal
        
        # Find common output length (use expected_length as target)
        output_length = self.expected_length
        
        # Truncate to output_length
        reconstructed = reconstructed[:, :output_length]
        
        return torch.tensor(reconstructed, dtype=torch.float32, device=device)
    
    def get_output_dim(self, input_length):
        """Calculate the dimension of wavelet coefficients for a given input length
        
        This helps with model initialization by computing the expected output size.
        
        Args:
            input_length: Length of input audio signal
            
        Returns:
            Total number of wavelet coefficients
        """
        total_coeffs, _ = self._calculate_expected_shapes(input_length)
        return total_coeffs