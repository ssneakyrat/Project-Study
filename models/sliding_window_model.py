import torch
import torch.nn as nn
import torch.nn.functional as F
from models.window_processor import WindowProcessor

class SlidingWindowModel(nn.Module):
    """Wrapper for processing continuous audio with sliding windows
    
    Applies the base model to overlapping windows of audio and
    reconstructs the full signal using overlap-add method.
    
    Mathematical basis:
    - Segmentation: y[m,n] = x[m·H + n]·w[n]
    - Processing: z[m,n] = model(y[m,n])
    - Reconstruction: x̂[n] = Σ_m z[m,n-m·H]·w[n] / Σ_m w^2[n-m·H]
    """
    def __init__(self, base_model, window_size, hop_size, window_type='hann'):
        """Initialize sliding window model
        
        Args:
            base_model: Base model to apply to each window
            window_size: Size of processing window in samples
            hop_size: Hop size between windows in samples
            window_type: Type of window function to use
        """
        super().__init__()
        self.base_model = base_model
        self.window_processor = WindowProcessor(window_size, hop_size, window_type)
    
    def forward(self, x):
        """Process continuous audio with sliding windows
        
        Args:
            x: Input audio tensor [batch_size, signal_length]
            
        Returns:
            x_recon: Reconstructed audio tensor [batch_size, signal_length]
            segments: Processed audio segments [batch_size, n_segments, window_size]
            boundary_indices: Window boundary indices
        """
        # Segment the input signal into overlapping windows
        segments, boundary_indices = self.window_processor.segment(x)
        
        # Get dimensions
        batch_size, n_segments, window_size = segments.shape
        
        # Process each segment separately
        processed_segments = torch.zeros_like(segments)
        
        for i in range(n_segments):
            # Extract segment for all batches
            segment = segments[:, i, :]
            
            # Process segment with base model
            with torch.no_grad():  # For inference only
                processed, _, _, _ = self.base_model(segment)
            
            # Store processed segment
            processed_segments[:, i, :] = processed
        
        # Reconstruct the full signal
        x_recon = self.window_processor.reconstruct(processed_segments)
        
        return x_recon, processed_segments, boundary_indices
    
    def process_stream(self, x, buffer=None, return_buffer=False):
        """Process audio stream with sliding windows and state buffer
        
        Maintains a buffer of past samples to handle streaming audio.
        
        Args:
            x: New audio chunk [batch_size, chunk_length]
            buffer: Optional buffer of past samples [batch_size, buffer_length]
            return_buffer: Whether to return the updated buffer
            
        Returns:
            output: Processed output for the new chunk
            new_buffer: Updated buffer (if return_buffer=True)
        """
        batch_size, chunk_length = x.shape
        window_size = self.window_processor.window_size
        hop_size = self.window_processor.hop_size
        
        # Initialize or use provided buffer
        if buffer is None:
            # For proper processing, buffer should be at least window_size - hop_size
            buffer_length = window_size - hop_size
            buffer = torch.zeros((batch_size, buffer_length), device=x.device)
        
        # Concatenate buffer with new chunk
        x_with_buffer = torch.cat([buffer, x], dim=1)
        
        # Process the concatenated signal
        x_recon, _, _ = self.forward(x_with_buffer)
        
        # Extract the portion corresponding to the new chunk
        # (accounting for processing delay)
        output = x_recon[:, buffer.shape[1]:]
        
        # Update buffer for next call (keep the last buffer_length samples)
        new_buffer = x_with_buffer[:, -window_size:]
        
        if return_buffer:
            return output, new_buffer
        else:
            return output