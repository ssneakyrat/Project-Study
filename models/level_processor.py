import torch
import torch.nn as nn
import torch.nn.functional as F

class WaveletLevelProcessor(nn.Module):
    """Process wavelet coefficients by level with different compression ratios
    
    Different wavelet decomposition levels have different perceptual importance:
    - Approximation coefficients (cA) need highest quality representation
    - Detail coefficients at different levels need progressively less precision
    
    This module:
    1. Separates coefficients by decomposition level
    2. Applies appropriate compression to each level
    3. Recombines processed coefficients
    """
    def __init__(self, level_dims, base_dim=32):
        """Initialize level processor
        
        Args:
            level_dims: List of dimensions for each level [cA, cD_L, cD_{L-1}, ..., cD_1]
            base_dim: Base dimension for compression calculation
        """
        super().__init__()
        self.level_dims = level_dims
        
        # Calculate total input dimension
        self.input_dim = sum(level_dims)
        
        # Build separate processor for each level
        self.level_processors = nn.ModuleList()
        self.level_slice_indices = []
        
        # Use shared parameters for similar levels to reduce model size
        # Group levels into categories
        approximation_net = None
        detail_nets = {}
        
        start_idx = 0
        for i, dim in enumerate(level_dims):
            end_idx = start_idx + dim
            self.level_slice_indices.append((start_idx, end_idx))
            
            # Skip extremely small dimensions
            if dim <= 4:
                # For very small dimensions, just use identity mapping
                processor = nn.Identity()
                self.level_processors.append(processor)
                start_idx = end_idx
                continue
                
            # For level sizes that are too large, use dimensionality reduction
            if dim > 10000:
                # Use much more aggressive compression for very large dimensions
                if i == 0:
                    # Approximation coefficients
                    compressed_dim = min(128, max(16, dim // 100))
                else:
                    # Detail coefficients
                    compressed_dim = min(64, max(8, dim // 200))
                    
                processor = nn.Sequential(
                    nn.Linear(dim, compressed_dim),
                    nn.LeakyReLU(0.2),
                    nn.Linear(compressed_dim, dim)
                )
            else:
                # For moderate sized dimensions, use level-based processing
                if i == 0:
                    # For approximation coefficients: preserve quality with less compression
                    compressed_dim = max(16, min(64, dim // 4))  # Much more aggressive
                    
                    # Create or reuse approximation network
                    if approximation_net is None:
                        approximation_net = nn.Sequential(
                            nn.Linear(dim, compressed_dim),
                            nn.LeakyReLU(0.2),
                            nn.Linear(compressed_dim, dim)
                        )
                    processor = approximation_net
                else:
                    # For detail coefficients: group by size range to share parameters
                    # Create size buckets: small, medium, large
                    if dim < 100:
                        size_bucket = 'small'
                        compressed_dim = max(4, dim // 4)
                    elif dim < 1000:
                        size_bucket = 'medium'
                        compressed_dim = max(8, dim // 8)
                    else:
                        size_bucket = 'large'
                        compressed_dim = max(16, dim // 16)
                        
                    # Create or reuse networks for this size bucket
                    if (size_bucket, dim) not in detail_nets:
                        detail_nets[(size_bucket, dim)] = nn.Sequential(
                            nn.Linear(dim, compressed_dim),
                            nn.LeakyReLU(0.2),
                            nn.Linear(compressed_dim, dim)
                        )
                    processor = detail_nets[(size_bucket, dim)]
            
            self.level_processors.append(processor)
            start_idx = end_idx
    
    def forward(self, x):
        """Process coefficients by level
        
        Args:
            x: Flattened wavelet coefficients [batch_size, total_coeffs]
            
        Returns:
            Processed coefficients [batch_size, total_coeffs]
        """
        # Pre-allocate output tensor
        batch_size = x.shape[0]
        output = torch.zeros_like(x)
        
        # Process each level separately
        for i, ((start_idx, end_idx), processor) in enumerate(zip(self.level_slice_indices, self.level_processors)):
            # Extract coefficients for this level
            level_coeffs = x[:, start_idx:end_idx]
            
            # Process coefficients
            processed_coeffs = processor(level_coeffs)
            
            # Store processed coefficients
            output[:, start_idx:end_idx] = processed_coeffs
            
        return output