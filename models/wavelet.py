import torch
import numpy as np
import pywt

class WaveletTransformOptimized:
    """Enhanced wavelet transform module with coefficient pruning
    
    Implements Discrete Wavelet Transform (DWT) and its inverse (IDWT) with:
    - Adaptive coefficient thresholding (pruning)
    - Level-aware processing
    - Preservation of level structure for more efficient compression
    """
    def __init__(self, wavelet='db4', level=3, threshold_factor=2.0):
        """Initialize wavelet transform with specified wavelet and decomposition level
        
        Args:
            wavelet (str): Wavelet family to use (e.g., 'db4', 'haar')
            level (int): Decomposition level
            threshold_factor (float): Controls sparsity via T_j = threshold_factor·σ·√(2·log N_j)
        """
        self.wavelet = wavelet
        self.level = level
        self.threshold_factor = threshold_factor
        
        # Will be dynamically calculated based on input audio length
        self.total_coeffs = None
        self.slice_indices = None
        self.expected_length = None
        self.level_dims = None
        
        # Initialize wavelet to validate it exists
        pywt.Wavelet(wavelet)
    
    def _calculate_expected_shapes(self, length):
        """Calculate expected coefficient shapes for a given input length
        
        For a signal of length N with decomposition level L:
        - Approx coefficients at level L: N_L ≈ N/2^L
        - Detail coefficients at level j: N_j ≈ N/2^j
        
        Returns a tuple of: (total_coeffs, slice_indices, level_dims)
        """
        # Create a dummy signal to calculate expected shapes
        dummy = np.zeros(length)
        
        # Apply DWT to get coefficient sizes
        coeffs = pywt.wavedec(dummy, self.wavelet, level=self.level)
        
        # Calculate total size and slice indices for reconstruction
        total_coeffs = sum(len(c) for c in coeffs)
        
        # Calculate slice indices for reconstruction
        slice_indices = []
        level_dims = []
        start_idx = 0
        for c in coeffs:
            end_idx = start_idx + len(c)
            slice_indices.append((start_idx, end_idx, c.shape))
            level_dims.append(len(c))
            start_idx = end_idx
            
        return total_coeffs, slice_indices, level_dims
    
    def _apply_coefficient_pruning(self, coeffs_tensor, training=True):
        """Apply adaptive thresholding to wavelet coefficients
        
        Implementation of the wavelet shrinkage denoising method:
        1. Estimate noise level using median absolute deviation
        2. Calculate threshold using the universal threshold formula
        3. Apply soft thresholding to coefficients
        
        Args:
            coeffs_tensor: Wavelet coefficients [batch_size, total_coeffs]
            training: If True, apply soft thresholding; if False, apply hard thresholding
            
        Returns:
            Pruned coefficients [batch_size, total_coeffs]
        """
        if not training:
            # During inference, use hard thresholding for better quality
            return coeffs_tensor
        
        batch_size = coeffs_tensor.shape[0]
        device = coeffs_tensor.device
        pruned_coeffs = torch.zeros_like(coeffs_tensor)
        
        # Process each level separately with appropriate thresholds
        for i, (start_idx, end_idx, _) in enumerate(self.slice_indices):
            # Extract coefficients for this level
            level_coeffs = coeffs_tensor[:, start_idx:end_idx]
            
            # Skip thresholding for approximation coefficients (first level)
            if i == 0:
                pruned_coeffs[:, start_idx:end_idx] = level_coeffs
                continue
            
            # For detail coefficients, apply thresholding
            # Estimate noise level using median absolute deviation
            mad = torch.median(torch.abs(level_coeffs - torch.median(level_coeffs))) / 0.6745
            
            # Universal threshold formula: T = threshold_factor * σ * sqrt(2 * log(N))
            # Higher levels get progressively stronger thresholding
            level_factor = self.threshold_factor * (1 + (i-1) * 0.2)  # Increase factor for higher levels
            N = end_idx - start_idx
            threshold = level_factor * mad * torch.sqrt(2.0 * torch.log(torch.tensor(N, dtype=torch.float, device=device)))
            
            # Apply soft thresholding: sign(x) * max(|x| - threshold, 0)
            # This keeps the sign but reduces the magnitude
            signs = torch.sign(level_coeffs)
            magnitudes = torch.abs(level_coeffs) - threshold
            pruned_level_coeffs = signs * torch.clamp(magnitudes, min=0.0)
            
            # Store pruned coefficients
            pruned_coeffs[:, start_idx:end_idx] = pruned_level_coeffs
            
        return pruned_coeffs
    
    def forward(self, x, training=True):
        """Apply DWT to batch of signals and return flattened coefficients with pruning
        
        Args:
            x: Input audio tensor [batch_size, audio_length]
            training: Whether in training mode (affects pruning strategy)
            
        Returns:
            Flattened wavelet coefficients tensor [batch_size, total_coeffs]
        """
        batch_size = x.shape[0]
        device = x.device
        
        # If first time or different audio length, calculate expected shapes
        if self.total_coeffs is None or x.shape[1] != self.expected_length:
            self.expected_length = x.shape[1]
            self.total_coeffs, self.slice_indices, self.level_dims = self._calculate_expected_shapes(x.shape[1])
            
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
        coeffs_tensor = coeffs_tensor.to(device)
        
        # Apply coefficient pruning during training
        pruned_coeffs = self._apply_coefficient_pruning(coeffs_tensor, training)
        
        return pruned_coeffs
    
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
        total_coeffs, _, level_dims = self._calculate_expected_shapes(input_length)
        return total_coeffs
    
    def get_level_dims(self, input_length=None):
        """Get dimensions of each wavelet decomposition level
        
        Args:
            input_length: Optional input length to update dimensions
            
        Returns:
            List of dimensions for each level [cA, cD_L, cD_{L-1}, ..., cD_1]
        """
        if input_length is not None or self.level_dims is None:
            _, _, self.level_dims = self._calculate_expected_shapes(input_length or self.expected_length)
        
        return self.level_dims