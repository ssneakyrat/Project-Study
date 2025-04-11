import torch
import numpy as np
import pywt

class WaveletTransformOptimized:
    """Efficient wavelet transform with aggressive coefficient pruning and approximation-focused processing
    
    Key improvements:
    1. More aggressive pruning of detail coefficients
    2. Preservation of approximation coefficients
    3. Optimized batch processing to reduce CPU-GPU transfers
    """
    def __init__(self, wavelet='db4', level=3, threshold_factor=3.0):
        """Initialize wavelet transform with specified wavelet and decomposition level
        
        Args:
            wavelet (str): Wavelet family to use (e.g., 'db4', 'haar')
            level (int): Decomposition level
            threshold_factor (float): Controls sparsity via T_j = threshold_factor·σ·√(2·log N_j)
                                      Higher values → more sparsity → faster training
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
        """Calculate expected coefficient shapes for a given input length"""
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
    
    def _apply_aggressive_pruning(self, coeffs_tensor, training=True):
        """Apply adaptive thresholding focused on detail coefficients
        
        This preserves approximation coefficients (most important) while
        aggressively pruning detail coefficients (less perceptually important).
        
        Mathematical basis: Universal shrinkage threshold
        T_j = threshold_factor * σ * sqrt(2 * log(N_j))
        with level-dependent scaling for perceptual importance.
        """
        if not training:
            # During inference, preserve more coefficients for quality
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
            
            # For detail coefficients, apply more aggressive thresholding for higher levels
            # Higher level detail coefficients (higher frequency) typically contribute less
            # to perceptual quality, so we apply stronger pruning
            
            # Estimate noise level using median absolute deviation
            mad = torch.median(torch.abs(level_coeffs - torch.median(level_coeffs))) / 0.6745
            
            # Universal threshold with level-dependent scaling:
            # Higher i → higher frequency → more aggressive pruning
            level_factor = self.threshold_factor * (1.0 + i * 0.5)  # Increase by 50% per level
            N = end_idx - start_idx
            threshold = level_factor * mad * torch.sqrt(2.0 * torch.log(torch.tensor(N, dtype=torch.float, device=device)))
            
            # Hard thresholding for faster training and better sparsity
            magnitude = torch.abs(level_coeffs)
            mask = magnitude > threshold
            pruned_level_coeffs = level_coeffs * mask.float()
            
            # Store pruned coefficients
            pruned_coeffs[:, start_idx:end_idx] = pruned_level_coeffs
            
        return pruned_coeffs
    
    def forward(self, x, training=True):
        """Apply DWT with batch processing to improve efficiency
        
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
            
        # Process in small batches to reduce memory pressure
        max_batch = 16
        coeffs_tensor = torch.zeros((batch_size, self.total_coeffs), dtype=torch.float32, device='cpu')
        
        # Process in smaller batches to reduce memory transfers
        for i in range(0, batch_size, max_batch):
            end_idx = min(i + max_batch, batch_size)
            sub_batch = x[i:end_idx]
            
            # Move entire sub-batch to CPU at once
            sub_batch_cpu = sub_batch.cpu().numpy()
            
            # Process each sample in the sub-batch
            for j in range(end_idx - i):
                # Apply DWT
                coeffs = pywt.wavedec(sub_batch_cpu[j], self.wavelet, level=self.level)
                
                # Concatenate coefficients
                start_idx = 0
                for c in coeffs:
                    end_idx_c = start_idx + len(c)
                    coeffs_tensor[i+j, start_idx:end_idx_c] = torch.from_numpy(c.astype(np.float32))
                    start_idx = end_idx_c
        
        # Move back to original device (single transfer)
        coeffs_tensor = coeffs_tensor.to(device)
        
        # Apply coefficient pruning during training
        pruned_coeffs = self._apply_aggressive_pruning(coeffs_tensor, training)
        
        return pruned_coeffs
    
    def inverse(self, coeffs_tensor):
        """Reconstruct signals from wavelet coefficients with optimized batch processing"""
        batch_size = coeffs_tensor.shape[0]
        device = coeffs_tensor.device
        
        # Process in smaller batches to reduce memory pressure
        max_batch = 16
        output_length = self.expected_length
        reconstructed = torch.zeros((batch_size, output_length), dtype=torch.float32, device=device)
        
        # Move to CPU for processing with PyWavelets
        coeffs_tensor_cpu = coeffs_tensor.cpu()
        
        for i in range(0, batch_size, max_batch):
            end_idx = min(i + max_batch, batch_size)
            
            # Process each sample in the sub-batch
            for j in range(end_idx - i):
                # Extract coefficients for this sample
                coeffs = []
                for start_idx, end_idx_c, shape in self.slice_indices:
                    coeff = coeffs_tensor_cpu[i+j, start_idx:end_idx_c].detach().numpy().reshape(shape)
                    coeffs.append(coeff)
                
                # Reconstruct signal
                rec_signal = pywt.waverec(coeffs, self.wavelet)
                
                # Trim to expected length
                if len(rec_signal) >= output_length:
                    rec_signal = rec_signal[:output_length]
                else:
                    # Pad if too short
                    rec_signal = np.pad(rec_signal, (0, output_length - len(rec_signal)))
                
                # Copy back to output tensor
                reconstructed[i+j] = torch.tensor(rec_signal, dtype=torch.float32, device=device)
        
        return reconstructed
    
    def get_output_dim(self, input_length):
        """Calculate the dimension of wavelet coefficients for a given input length"""
        total_coeffs, _, _ = self._calculate_expected_shapes(input_length)
        return total_coeffs
    
    def get_level_dims(self, input_length=None):
        """Get dimensions of each wavelet decomposition level"""
        if input_length is not None or self.level_dims is None:
            _, _, self.level_dims = self._calculate_expected_shapes(input_length or self.expected_length)
        
        return self.level_dims