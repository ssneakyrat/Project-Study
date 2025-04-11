import torch
import numpy as np

class WindowProcessor:
    """Handles sliding window processing for continuous audio
    
    Supports:
    - Signal segmentation with overlapping windows
    - Window function application for smooth transitions
    - Overlap-add reconstruction for continuous output
    
    Mathematical formulation:
    - Windowed segments: y[m,n] = x[m·H + n]·w[n]
    - Reconstruction: x[n] = Σ_m y[m,n-m·H]·w[n] / Σ_m w^2[n-m·H]
    """
    def __init__(self, window_size, hop_size, window_type='hann'):
        """Initialize window processor
        
        Args:
            window_size: Size of each window in samples
            hop_size: Hop size between windows in samples
            window_type: Type of window function to use
        """
        self.window_size = window_size
        self.hop_size = hop_size
        self.window_type = window_type
        
        # Create window function
        self.window = self._create_window(window_type, window_size)
    
    def _create_window(self, window_type, window_size):
        """Create window function of specified type and size"""
        # Use numpy to create windows and convert to PyTorch tensor
        # This is more compatible across PyTorch versions
        if window_type == 'hann':
            try:
                # Try to use PyTorch's built-in hann_window function
                window = torch.hann_window(window_size)
            except AttributeError:
                # Fall back to numpy implementation
                window = torch.from_numpy(np.hanning(window_size).astype(np.float32))
        elif window_type == 'hamming':
            try:
                # Try to use PyTorch's built-in hamming_window function
                window = torch.hamming_window(window_size)
            except AttributeError:
                # Fall back to numpy implementation
                window = torch.from_numpy(np.hamming(window_size).astype(np.float32))
        elif window_type == 'blackman':
            try:
                # Try to use PyTorch's built-in blackman_window function
                window = torch.blackman_window(window_size)
            except AttributeError:
                # Fall back to numpy implementation
                window = torch.from_numpy(np.blackman(window_size).astype(np.float32))
        elif window_type == 'rectangular':
            window = torch.ones(window_size)
        else:
            raise ValueError(f"Unsupported window type: {window_type}")
        
        return window
    
    def segment(self, signal):
        """Segment signal into overlapping windows
        
        Args:
            signal: Input signal of shape [batch_size, signal_length]
            
        Returns:
            segments: Tensor of windowed segments [batch_size, n_segments, window_size]
            boundary_indices: List of boundary indices in original signal
        """
        batch_size, signal_length = signal.shape
        device = signal.device
        
        # Calculate number of segments
        n_segments = max(1, 1 + (signal_length - self.window_size) // self.hop_size)
        
        # Pre-allocate output tensor
        segments = torch.zeros((batch_size, n_segments, self.window_size), device=device)
        
        # Store boundary indices for visualization
        boundary_indices = [s * self.hop_size for s in range(n_segments)]
        
        # Extract and window each segment
        window = self.window.to(device)
        
        for i in range(n_segments):
            # Calculate start and end indices for this segment
            start_idx = i * self.hop_size
            end_idx = start_idx + self.window_size
            
            # Handle boundary conditions
            if end_idx <= signal_length:
                # Normal case - full window fits within signal
                segments[:, i, :] = signal[:, start_idx:end_idx] * window
            else:
                # Window extends beyond signal - zero pad
                valid_length = signal_length - start_idx
                segments[:, i, :valid_length] = signal[:, start_idx:] * window[:valid_length]
        
        return segments, boundary_indices
    
    def reconstruct(self, segments):
        """Reconstruct signal from overlapping windowed segments
        
        Uses overlap-add method with proper normalization to 
        ensure perfect reconstruction with appropriate windows.
        
        Args:
            segments: Tensor of processed segments [batch_size, n_segments, window_size]
            
        Returns:
            reconstructed: Reconstructed signal [batch_size, signal_length]
        """
        batch_size, n_segments, window_size = segments.shape
        device = segments.device
        
        # Calculate output signal length
        signal_length = (n_segments - 1) * self.hop_size + window_size
        
        # Pre-allocate output signal and normalization buffer
        reconstructed = torch.zeros((batch_size, signal_length), device=device)
        norm_buffer = torch.zeros((batch_size, signal_length), device=device)
        
        # Get window function on the correct device
        window = self.window.to(device)
        
        # Overlap-add each windowed segment
        for i in range(n_segments):
            start_idx = i * self.hop_size
            end_idx = start_idx + window_size
            
            # Add windowed segment
            reconstructed[:, start_idx:end_idx] += segments[:, i, :] * window
            
            # Add window energy for normalization
            norm_buffer[:, start_idx:end_idx] += window ** 2
        
        # Normalize by window energy (avoid division by zero)
        eps = 1e-10
        # Handle normalization safely - iterate through each batch item
        for b in range(batch_size):
            # Create mask for non-zero normalization values
            mask = norm_buffer[b] > eps
            # Apply normalization only to masked positions
            reconstructed[b, mask] = reconstructed[b, mask] / norm_buffer[b, mask]
        
        return reconstructed