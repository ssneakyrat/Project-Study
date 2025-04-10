import torch
import torch.nn as nn
import torch.nn.functional as F
from kymatio.torch import Scattering1D

class WaveletScatteringTransform(nn.Module):
    def __init__(self, 
                 J=8,  # Increased from 7 to 8 for better low-frequency resolution
                 Q=16,  # Increased from 12 to 16 for finer frequency bands
                 T=16000,  # Signal length
                 max_order=1,  # Reduced order for better time-frequency localization
                 out_type='array',
                 oversampling=4):  # Increased oversampling for better phase reconstruction
        """Wavelet Scattering Transform module with enhanced phase preservation.
        
        Args:
            J: Number of scales 
            Q: Number of wavelets per octave
            T: Signal length
            max_order: Maximum scattering order
            out_type: Output format
            oversampling: Oversampling factor for better time resolution
        """
        super().__init__()
        self.scattering = Scattering1D(J=J, shape=T, Q=Q, max_order=max_order, 
                                        out_type=out_type, oversampling=oversampling)
        self.J = J
        self.Q = Q
        self.T = T
        self.max_order = max_order
        self.out_type = out_type
        self.oversampling = oversampling
        
        # Register normalization parameters as buffers with improved initialization
        self.register_buffer('mean', torch.zeros(1))
        self.register_buffer('std', torch.ones(1))
        self.register_buffer('initialized', torch.tensor(0))
        
        # Register per-channel statistics for better normalization
        dummy_input = torch.zeros(1, 1, T)
        with torch.no_grad():
            dummy_output = self.scattering(dummy_input)
            C = dummy_output.shape[1]
            self.register_buffer('channel_mean', torch.zeros(1, C, 1))
            self.register_buffer('channel_std', torch.ones(1, C, 1))
            self.register_buffer('channel_initialized', torch.tensor(0))
            
        # Register additional frequency band weights for emphasizing perceptually important bands
        # Create a weighted preference for middle frequencies (where human hearing is most sensitive)
        freq_bands = torch.arange(C, device=dummy_output.device).float()
        # Gaussian-like weighting centered on perceptually important frequencies
        alpha = 0.7  # Controls the width of the Gaussian
        mu = C * 0.35  # Center at approximately 1/3 of the frequency range
        band_weights = torch.exp(-alpha * (freq_bands - mu)**2 / C)
        # Normalize weights
        band_weights = band_weights / band_weights.sum()
        # Register as buffer
        self.register_buffer('band_weights', band_weights.view(1, -1, 1))
        
    def _update_normalization_stats(self, Sx):
        """Update running statistics for better normalization using per-channel stats"""
        if not self.training:
            return
            
        with torch.no_grad():
            # Update global stats
            batch_mean = torch.mean(Sx)
            batch_std = torch.std(Sx)
            
            if self.initialized == 0:
                self.mean = batch_mean
                self.std = batch_std
                self.initialized = torch.tensor(1)
            else:
                # Exponential moving average
                momentum = 0.1
                self.mean = (1 - momentum) * self.mean + momentum * batch_mean
                self.std = (1 - momentum) * self.std + momentum * batch_std
                
            # Update per-channel stats for more precise normalization
            batch_channel_mean = torch.mean(Sx, dim=(0, 2), keepdim=True)
            batch_channel_std = torch.std(Sx, dim=(0, 2), keepdim=True)
            
            if self.channel_initialized == 0:
                self.channel_mean = batch_channel_mean
                self.channel_std = batch_channel_std
                self.channel_initialized = torch.tensor(1)
            else:
                # Exponential moving average for channels
                self.channel_mean = (1 - momentum) * self.channel_mean + momentum * batch_channel_mean
                self.channel_std = (1 - momentum) * self.channel_std + momentum * batch_channel_std
    
    def _apply_frequency_band_weighting(self, Sx):
        """Apply perceptual weighting to frequency bands"""
        # Ensure band_weights matches the channel dimension
        if self.band_weights.shape[1] != Sx.shape[1]:
            with torch.no_grad():
                # Resize band_weights to match new channel dimension
                C = Sx.shape[1]
                freq_bands = torch.arange(C, device=Sx.device).float()
                alpha = 0.7
                mu = C * 0.35
                band_weights = torch.exp(-alpha * (freq_bands - mu)**2 / C)
                band_weights = band_weights / band_weights.sum()
                self.band_weights = band_weights.view(1, -1, 1)
        
        # Apply band weights - emphasize perceptually important bands
        # Using a soft weighting approach: Sx * (0.5 + 0.5 * band_weights)
        # This ensures all bands are preserved but important ones are emphasized
        weighted_Sx = Sx * (0.5 + 0.5 * self.band_weights)
        return weighted_Sx
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, 1, T) or (batch_size, T)
        
        Returns:
            Scattering coefficients with enhanced phase preservation
        """
        # Ensure input has the right shape
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        # Apply cosine window (less aggressive than Hann) to reduce edge effects
        # while preserving more signal content near boundaries
        window = torch.cos(torch.linspace(-torch.pi/2, torch.pi/2, self.T, device=x.device)) * 0.5 + 0.5
        x = x * window.view(1, 1, -1)
        
        # Pad or trim input if necessary to match expected length T
        _, _, L = x.shape
        if L != self.T:
            if L < self.T:
                # Use reflection padding to reduce boundary artifacts
                pad_size = self.T - L
                left_pad = pad_size // 2
                right_pad = pad_size - left_pad
                x = F.pad(x, (left_pad, right_pad), mode='reflect')
            else:
                # Center-crop to get most important part of signal
                start = (L - self.T) // 2
                x = x[:, :, start:start+self.T]
        
        # Apply scattering transform
        Sx = self.scattering(x)
        
        # Reshape the output for the complex CNN
        if Sx.dim() == 4:
            # Keep order dimension separate for better interpretability
            B, O, C, T = Sx.shape
            
            # Reshape while preserving order information
            # Each order will be processed separately in the network
            order_means = []
            for o in range(O):
                order_means.append(Sx[:, o].mean(dim=1, keepdim=True))
            
            # Scale each order differently to balance their contribution
            order_scales = [1.0, 0.7, 0.5]  # Decreasing importance for higher orders
            scaled_orders = []
            for o in range(min(O, len(order_scales))):
                scaled_orders.append(Sx[:, o] * order_scales[o])
            
            # Combine orders with their respective scaling
            Sx = torch.cat(scaled_orders, dim=1)
        
        # Apply perceptual frequency band weighting
        Sx = self._apply_frequency_band_weighting(Sx)
        
        # Update normalization stats
        self._update_normalization_stats(Sx)
        
        # Apply per-channel normalization for more balanced learning across coefficients
        epsilon = 1e-8  # Prevent division by zero
        Sx_normalized = (Sx - self.channel_mean) / (self.channel_std + epsilon)
        
        # Apply enhanced coefficient transformation for better phase preservation
        # Instead of logarithmic scaling, use a more phase-preserving transformation
        
        # 1. Split positive and negative coefficients
        Sx_pos = F.relu(Sx_normalized)
        Sx_neg = F.relu(-Sx_normalized)
        
        # 2. Apply different nonlinearities to positive and negative parts
        # Using log1p for positive and log1p for negative (after abs)
        # This better preserves the sign information crucial for phase
        alpha = 0.5  # Control parameter for nonlinearity strength
        transformed_pos = torch.log1p(alpha * Sx_pos) / alpha
        transformed_neg = -torch.log1p(alpha * Sx_neg) / alpha
        
        # 3. Recombine with weighted average
        Sx_transformed = transformed_pos + transformed_neg
        
        # Blend original and transformed coefficients
        blend_ratio = 0.7  # Higher values preserve more of the original information
        Sx_processed = blend_ratio * Sx_normalized + (1.0 - blend_ratio) * Sx_transformed
        
        # Apply temporal smoothing for better phase coherence
        # Small kernel to avoid excessive blurring
        kernel_size = 3
        padding = kernel_size // 2
        Sx_smoothed = F.avg_pool1d(Sx_processed, kernel_size=kernel_size, stride=1, padding=padding)
        
        # Final blend of original and smoothed coefficients
        smooth_ratio = 0.2  # Lower values preserve more temporal detail
        Sx_final = (1.0 - smooth_ratio) * Sx_processed + smooth_ratio * Sx_smoothed
        
        return Sx_final