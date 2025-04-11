import torch
import torch.nn as nn
import torch.nn.functional as F

class WaveletLevelProcessor(nn.Module):
    """Efficient wavelet coefficient processor that prioritizes approximation coefficients
    
    Mathematical justification:
    - Energy preservation: cA contains ~80-90% of signal energy
    - Perceptual importance: Low-frequency components dominate audio perception
    - Rate-distortion optimization: allocates bits according to perceptual importance

    This module:
    1. Separates coefficients by decomposition level
    2. Applies high-quality processing to approximation coefficients
    3. Uses minimal processing for detail coefficients
    4. Recombines processed coefficients
    """
    def __init__(self, level_dims):
        """Initialize level processor
        
        Args:
            level_dims: List of dimensions for each level [cA, cD_L, cD_{L-1}, ..., cD_1]
        """
        super().__init__()
        self.level_dims = level_dims
        
        # Calculate total input dimension
        self.input_dim = sum(level_dims)
        
        # Compute level boundaries for slicing
        self.level_slice_indices = []
        start_idx = 0
        for dim in level_dims:
            end_idx = start_idx + dim
            self.level_slice_indices.append((start_idx, end_idx))
            start_idx = end_idx
        
        # Create special processor only for approximation coefficients (level 0)
        # Skip if too small (< 16 coefficients)
        if level_dims and level_dims[0] >= 16:
            approx_dim = level_dims[0]
            # Simple linear layer for approximation coefficients
            self.approx_processor = nn.Sequential(
                nn.Linear(approx_dim, approx_dim),
                nn.LeakyReLU(0.2)
            )
        else:
            self.approx_processor = nn.Identity()
        
        # Use identity mapping for all detail coefficients to save computation
        # This dramatically reduces computation while preserving most perceptual quality
    
    def forward(self, x):
        """Process coefficients by level with focus on approximation coefficients
        
        Args:
            x: Flattened wavelet coefficients [batch_size, total_coeffs]
            
        Returns:
            Processed coefficients [batch_size, total_coeffs]
        """
        # Pre-allocate output tensor (copy input to preserve detail coefficients)
        output = x.clone()
        
        # Process only approximation coefficients (level 0)
        if self.level_dims and self.level_dims[0] >= 16:
            start_idx, end_idx = self.level_slice_indices[0]
            approx_coeffs = x[:, start_idx:end_idx]
            output[:, start_idx:end_idx] = self.approx_processor(approx_coeffs)
        
        return output