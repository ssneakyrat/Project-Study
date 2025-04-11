import torch
import torch.nn as nn
import torch.nn.functional as F
from models.dwt import DWT, IDWT

class ResidualBlock(nn.Module):
    """
    Residual block with BatchNorm and LeakyReLU activation
    """
    def __init__(self, channels, kernel_size=3):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(channels)
        self.lrelu1 = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(channels)
        self.lrelu2 = nn.LeakyReLU(0.2)
        
    def forward(self, x):
        residual = x
        out = self.lrelu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual  # Skip connection
        out = self.lrelu2(out)
        return out

class AsymmetricFrequencyBlock(nn.Module):
    """
    Enhanced specialized block for high-frequency detail processing
    with position-aware processing and left/right asymmetry
    """
    def __init__(self, channels, kernel_size=3, level=0):
        super(AsymmetricFrequencyBlock, self).__init__()
        self.level = level  # Store wavelet level for position-aware processing
        self.norm1 = nn.BatchNorm1d(channels)
        
        # Split channels for direction-sensitive processing
        self.channels_half = channels // 2
        
        # Direction-sensitive convolutions with level-specific kernel sizes
        if level == 0:  # Level 1 (highest freq) - needs most detail preservation on left
            left_kernel = 3  # Smaller kernel for better left-side detail
            right_kernel = 7  # Larger kernel for right side
        elif level == 1:  # Level 2 - needs better right-side detail
            left_kernel = 7  # Larger kernel for left side
            right_kernel = 3  # Smaller kernel for better right-side detail
        else:  # Level 3+ - balanced
            left_kernel = 5
            right_kernel = 5
            
        self.conv_left = nn.Conv1d(self.channels_half, self.channels_half, 
                                   kernel_size=left_kernel, padding=left_kernel//2)
        self.conv_right = nn.Conv1d(self.channels_half, self.channels_half, 
                                    kernel_size=right_kernel, padding=right_kernel//2)
        
        self.norm2 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=kernel_size//2)
        
        # Enhanced position-dependent attention mechanism
        self.pos_attention = nn.Sequential(
            nn.Conv1d(channels + 1, channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Gating mechanism for adaptive detail preservation
        self.gate = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Gentler activation for high-frequency details
        self.lrelu = nn.LeakyReLU(0.1)
        
    def forward(self, x):
        residual = x
        batch_size, channels, seq_len = x.shape
        
        # Normalization
        out = self.norm1(x)
        
        # Split channels
        out_left, out_right = torch.split(out, self.channels_half, dim=1)
        
        # Apply different processing to left and right channels
        out_left = self.conv_left(out_left)
        out_right = self.conv_right(out_right)
        
        # Concatenate channels
        out = torch.cat([out_left, out_right], dim=1)
        
        # Generate level-specific position encoding with proper bias correction
        position = torch.linspace(0, 1, seq_len).view(1, 1, -1).to(out.device)
        position_encoding = position.expand(batch_size, 1, -1)  # Expand to match batch size
        
        # Apply level-specific position-dependent scaling
        if self.level == 0:  # Level 1 (highest frequency) - strengthen right side
            # Create position weights that emphasize the right (weak area)
            position_weights = 1.0 + 1.5 * position  # 1.0 at left, 2.5 at right
        elif self.level == 1:  # Level 2 - strengthen left side 
            # Create position weights that emphasize the left (weak area)
            position_weights = 2.5 - 1.5 * position  # 2.5 at left, 1.0 at right
        elif self.level == 2:  # Level 3 - stronger left emphasis
            # Create weights with stronger emphasis on the left side
            position_weights = 3.0 - 2.0 * position  # 3.0 at left, 1.0 at right
        else:
            # Neutral weighting for other levels
            position_weights = torch.ones_like(position)
        
        # Expand position weights to match batch size
        position_weights = position_weights.expand(batch_size, 1, -1)
            
        # Apply position-dependent attention
        position_aware_attention = self.pos_attention(torch.cat([out, position_encoding], dim=1))
        out = out * (position_aware_attention * position_weights)
        
        # Second conv layer
        out = self.lrelu(self.norm2(out))
        out = self.conv2(out)
        
        # Adaptive gating for selective detail preservation
        gate = self.gate(residual)
        out = gate * out + (1 - gate) * residual
        
        out = self.lrelu(out)
        return out

class FrequencyAwareBlock(nn.Module):
    """
    Enhanced specialized block for high-frequency detail processing
    with multi-scale filtering and gating mechanism
    """
    def __init__(self, channels, kernel_size=3):
        super(FrequencyAwareBlock, self).__init__()
        self.norm1 = nn.BatchNorm1d(channels)
        
        # Parallel processing paths with different kernel sizes for multi-scale awareness
        self.conv_small = nn.Conv1d(channels, channels//2, kernel_size=3, padding=1, groups=2)
        self.conv_med = nn.Conv1d(channels, channels//2, kernel_size=5, padding=2)
        
        self.norm2 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=kernel_size//2)
        
        # Gating mechanism for adaptive detail preservation
        self.gate = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Gentler activation for high-frequency details
        self.lrelu = nn.LeakyReLU(0.1)
        
    def forward(self, x):
        residual = x
        
        # Normalization
        out = self.norm1(x)
        
        # Multi-scale parallel paths
        out1 = self.conv_small(out)
        out2 = self.conv_med(out)
        out = torch.cat([out1, out2], dim=1)
        
        # Second conv layer
        out = self.lrelu(self.norm2(out))
        out = self.conv2(out)
        
        # Adaptive gating for selective detail preservation
        gate = self.gate(residual)
        out = gate * out + (1 - gate) * residual
        
        out = self.lrelu(out)
        return out

class UnifiedEncoder(nn.Module):
    """
    Enhanced unified encoder for wavelet coefficients with improved high-frequency processing
    """
    def __init__(self, levels=4, hidden_dim=64):
        super(UnifiedEncoder, self).__init__()
        self.levels = levels
        self.hidden_dim = hidden_dim
        
        # Initial convolutions for each coefficient level
        self.initial_convs = nn.ModuleList()
        for i in range(levels + 1):  # +1 for approximation coefficients
            if i == 0:  # Approximation coefficients
                # Standard processing for low frequencies
                kernel_size = 7
                self.initial_convs.append(
                    nn.Sequential(
                        nn.Conv1d(1, hidden_dim, kernel_size=kernel_size, padding=kernel_size//2),
                        nn.BatchNorm1d(hidden_dim),
                        nn.LeakyReLU(0.2)
                    )
                )
            elif i < 3:  # First 3 detail levels (high frequencies)
                # Enhanced processing for high frequencies
                kernel_size = 3  # Smaller kernel for better detail preservation
                self.initial_convs.append(
                    nn.Sequential(
                        nn.Conv1d(1, hidden_dim, kernel_size=kernel_size, padding=kernel_size//2),
                        nn.BatchNorm1d(hidden_dim),
                        nn.LeakyReLU(0.1),  # Gentler activation
                        nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=kernel_size//2),
                        nn.BatchNorm1d(hidden_dim),
                        nn.LeakyReLU(0.1)
                    )
                )
            else:
                # Standard processing for mid/low frequencies
                kernel_size = 5
                self.initial_convs.append(
                    nn.Sequential(
                        nn.Conv1d(1, hidden_dim, kernel_size=kernel_size, padding=kernel_size//2),
                        nn.BatchNorm1d(hidden_dim),
                        nn.LeakyReLU(0.2)
                    )
                )
        
        # Unified processing blocks with residual connections
        self.encoder_blocks = nn.ModuleList()
        for i in range(levels + 1):
            if i < 3:  # High-frequency specific processing with asymmetric blocks
                self.encoder_blocks.append(
                    nn.Sequential(
                        AsymmetricFrequencyBlock(hidden_dim, kernel_size=3, level=i-1),
                        AsymmetricFrequencyBlock(hidden_dim, kernel_size=3, level=i-1)
                    )
                )
            else:
                self.encoder_blocks.append(
                    nn.Sequential(
                        ResidualBlock(hidden_dim),
                        ResidualBlock(hidden_dim)
                    )
                )
        
        # Progressive downsampling with preserved information for high frequencies
        self.downsample = nn.ModuleList()
        for i in range(levels + 1):
            if i < 3:  # Reduced downsampling for high frequencies
                # Less aggressive downsampling to preserve high-frequency details
                # Mathematical scaling: higher frequencies (j=0,1,2) get smaller downsampling factor
                factor = max(2, min(2**(levels-i-2), 4))
                
                # Enhanced downsampling for high frequencies with extra processing
                self.downsample.append(
                    nn.Sequential(
                        nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                        nn.BatchNorm1d(hidden_dim),
                        nn.LeakyReLU(0.1),
                        nn.Conv1d(hidden_dim, hidden_dim*2, kernel_size=factor+1, stride=factor, padding=1),
                        nn.BatchNorm1d(hidden_dim*2),
                        nn.LeakyReLU(0.1)
                    )
                )
            else:
                factor = min(2**(levels-i), 8)  # Standard downsampling for others
                self.downsample.append(
                    nn.Sequential(
                        nn.Conv1d(hidden_dim, hidden_dim*2, kernel_size=factor+1, stride=factor, padding=1),
                        nn.BatchNorm1d(hidden_dim*2),
                        nn.LeakyReLU(0.2)
                    )
                )
            
        # Fusion layer for combining all encoded features
        fusion_input_dim = hidden_dim*2 * (levels+1)
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_input_dim, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.2)
        )
        
    def forward(self, coeffs):
        """
        Forward pass of unified encoder
        Args:
            coeffs: Dictionary of wavelet coefficients {'a': approx, 'd': [detail1, ..., detailN]}
        Returns:
            z: Latent representation
            encoded_features: Intermediate encoded features for skip connections
        """
        batch_size = coeffs['a'].shape[0]
        encoded_features = []
        feature_vectors = []
        
        # Process approximation coefficients
        a = coeffs['a'].unsqueeze(1)  # [B, 1, T]
        a_enc = self.initial_convs[0](a)
        a_enc = self.encoder_blocks[0](a_enc)
        encoded_features.append(a_enc)
        
        # Downsample and extract feature vector
        a_down = self.downsample[0](a_enc)
        a_feat = F.adaptive_avg_pool1d(a_down, 1).view(batch_size, -1)
        feature_vectors.append(a_feat)
        
        # Process detail coefficients for each level
        for j, d in enumerate(coeffs['d']):
            d = d.unsqueeze(1)  # [B, 1, T]
            d_enc = self.initial_convs[j+1](d)
            d_enc = self.encoder_blocks[j+1](d_enc)
            encoded_features.append(d_enc)
            
            # Downsample and extract feature vector
            d_down = self.downsample[j+1](d_enc)
            d_feat = F.adaptive_avg_pool1d(d_down, 1).view(batch_size, -1)
            feature_vectors.append(d_feat)
        
        # Concatenate all features and apply fusion layer
        concat_features = torch.cat(feature_vectors, dim=1)
        z = self.fusion_layer(concat_features)
        
        return z, encoded_features

class PositionEnhancedLayer(nn.Module):
    """
    New layer to specifically address position-dependent wavelet reconstruction issues
    """
    def __init__(self, channels, level, mode='reconstruct'):
        super(PositionEnhancedLayer, self).__init__()
        self.level = level
        self.mode = mode  # 'encode' or 'reconstruct'
        
        # Position-aware conv to learn position-specific transformations
        self.pos_conv = nn.Conv1d(channels + 1, channels, kernel_size=5, padding=2)
        
        # Level-specific processing parameters
        self.alpha = nn.Parameter(torch.tensor(1.0))  # Learnable scaling parameter
        self.beta = nn.Parameter(torch.tensor(0.5))   # Learnable bias parameter
        
        # Final activation
        self.act = nn.LeakyReLU(0.1)
        
    def forward(self, x):
        batch_size, channels, seq_len = x.shape
        
        # Create position encoding
        position = torch.linspace(0, 1, seq_len).view(1, 1, -1).to(x.device)
        position = position.expand(batch_size, 1, -1)
        
        # Concatenate with input
        x_with_pos = torch.cat([x, position], dim=1)
        
        # Apply position-aware convolution
        pos_features = self.pos_conv(x_with_pos)
        
        # Create level-specific position weights
        if self.level == 0:  # Detail level 1 - boost right side reconstruction
            if self.mode == 'reconstruct':
                # For reconstruction: stronger correction on right side
                pos_weights = 1.0 + self.alpha * position ** 2
            else:
                # For encoding: emphasize left side (where reconstruction is better)
                pos_weights = 1.0 + self.alpha * (1 - position) ** 2
                
        elif self.level == 1:  # Detail level 2 - boost left side reconstruction
            if self.mode == 'reconstruct':
                # For reconstruction: stronger correction on left side
                pos_weights = 1.0 + self.alpha * (1 - position) ** 2
            else:
                # For encoding: emphasize right side (where reconstruction is better)
                pos_weights = 1.0 + self.alpha * position ** 2
                
        elif self.level == 2:  # Detail level 3 - strong left boost
            if self.mode == 'reconstruct':
                # For reconstruction: stronger correction on left side
                pos_weights = 1.0 + 1.5 * self.alpha * (1 - position) ** self.beta
            else:
                # For encoding: stronger right emphasis
                pos_weights = 1.0 + self.alpha * position ** self.beta
        else:
            # Neutral weighting for other levels
            pos_weights = torch.ones_like(position)
        
        # Apply position-dependent weighting
        out = x * pos_weights + pos_features
        
        return self.act(out)

class UnifiedDecoder(nn.Module):
    """
    Enhanced unified decoder with better detail reconstruction capabilities
    and position-aware processing for different frequency bands
    """
    def __init__(self, levels=4, hidden_dim=64, latent_dim=256):
        super(UnifiedDecoder, self).__init__()
        self.levels = levels
        self.hidden_dim = hidden_dim
        
        # Initial projection from latent space
        self.latent_projection = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2)
        )
        
        # Branch-specific feature generators
        self.branch_projections = nn.ModuleList()
        for i in range(levels + 1):
            if i < 3:  # High-frequency branches get more capacity
                # For high frequencies, we use a larger initial feature size (hidden_dim * 4)
                self.branch_projections.append(
                    nn.Sequential(
                        nn.Linear(512, hidden_dim * 4),
                        nn.LayerNorm(hidden_dim * 4),
                        nn.LeakyReLU(0.1)
                    )
                )
            else:
                self.branch_projections.append(
                    nn.Sequential(
                        nn.Linear(512, hidden_dim * 4),
                        nn.LayerNorm(hidden_dim * 4),
                        nn.LeakyReLU(0.2)
                    )
                )
        
        # Position-enhanced layers for targeted detail reconstruction
        self.position_enhance = nn.ModuleList()
        for i in range(levels + 1):
            if i < 3:  # Only for high-frequency detail levels
                self.position_enhance.append(
                    PositionEnhancedLayer(hidden_dim, i, mode='reconstruct')
                )
            else:
                self.position_enhance.append(
                    nn.Identity()  # No special processing for low frequencies
                )
        
        # Unified decoder blocks with residual connections
        self.decoder_blocks = nn.ModuleList()
        for i in range(levels + 1):
            if i < 3:  # High-frequency specific processing with asymmetric blocks
                self.decoder_blocks.append(
                    nn.Sequential(
                        nn.Conv1d(hidden_dim*2, hidden_dim, kernel_size=3, padding=1),
                        nn.BatchNorm1d(hidden_dim),
                        nn.LeakyReLU(0.1),
                        AsymmetricFrequencyBlock(hidden_dim, kernel_size=3, level=i),
                        AsymmetricFrequencyBlock(hidden_dim, kernel_size=3, level=i)
                    )
                )
            else:
                # Standard processing for other frequencies
                self.decoder_blocks.append(
                    nn.Sequential(
                        nn.Conv1d(hidden_dim*2, hidden_dim, kernel_size=3, padding=1),
                        nn.BatchNorm1d(hidden_dim),
                        nn.LeakyReLU(0.2),
                        ResidualBlock(hidden_dim),
                        ResidualBlock(hidden_dim)
                    )
                )
        
        # Final output layers with enhanced detail preservation
        self.output_layers = nn.ModuleList()
        for i in range(levels + 1):
            if i < 3:  # High-frequency detail reconstruction gets enhanced output
                self.output_layers.append(
                    nn.Sequential(
                        nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                        nn.LeakyReLU(0.1),
                        nn.Conv1d(hidden_dim, 1, kernel_size=1)  # Point-wise for detail preservation
                    )
                )
            else:
                self.output_layers.append(
                    nn.Conv1d(hidden_dim, 1, kernel_size=3, padding=1)
                )
    
    def forward(self, z, encoded_features, orig_shapes):
        """
        Forward pass of unified decoder
        Args:
            z: Latent representation
            encoded_features: Intermediate encoded features from encoder (for skip connections)
            orig_shapes: Original shapes of wavelet coefficients
        Returns:
            coeffs: Reconstructed wavelet coefficients
        """
        batch_size = z.shape[0]
        
        # Project latent vector to higher dimension
        h = self.latent_projection(z)
        
        # Reconstruct approximation coefficients
        a_shape = orig_shapes['a']
        a_feat = self.branch_projections[0](h).view(batch_size, self.hidden_dim, 4)
        
        # Upsample to match encoder feature size then concatenate with encoder features
        a_up = F.interpolate(a_feat, size=encoded_features[0].size(2), mode='linear')
        a_concat = torch.cat([a_up, encoded_features[0]], dim=1)
        
        # Apply decoder blocks
        a_dec = self.decoder_blocks[0](a_concat)
        
        # Apply position enhancement if needed
        if isinstance(self.position_enhance[0], PositionEnhancedLayer):
            a_dec = self.position_enhance[0](a_dec)
        
        # Final layer and reshape to match original shape
        a_out = self.output_layers[0](a_dec)
        a_out = F.interpolate(a_out, size=a_shape[1], mode='linear')
        a_out = a_out.squeeze(1)
        
        # Reconstruct detail coefficients
        d_outs = []
        for j in range(self.levels):
            d_shape = orig_shapes['d'][j]
            d_feat = self.branch_projections[j+1](h).view(batch_size, self.hidden_dim, 4)
            
            # Enhanced multi-scale upsampling for high-frequency components
            if j < 3:  # High-frequency specific handling
                # Progressive upsampling for better detail preservation
                d_up1 = F.interpolate(d_feat, size=encoded_features[j+1].size(2)//4, mode='linear')
                d_up2 = F.interpolate(d_up1, size=encoded_features[j+1].size(2)//2, mode='linear')
                d_up = F.interpolate(d_up2, size=encoded_features[j+1].size(2), mode='linear')
            else:
                d_up = F.interpolate(d_feat, size=encoded_features[j+1].size(2))
                
            d_concat = torch.cat([d_up, encoded_features[j+1]], dim=1)
            
            # Apply decoder blocks
            d_dec = self.decoder_blocks[j+1](d_concat)
            
            # Apply position enhancement for detail coefficients
            if j < 3 and isinstance(self.position_enhance[j+1], PositionEnhancedLayer):
                d_dec = self.position_enhance[j+1](d_dec)
            
            # Final layer and reshape to match original shape
            d_out = self.output_layers[j+1](d_dec)
            
            # Enhanced interpolation for high frequencies
            d_out = F.interpolate(d_out, size=d_shape[1], mode='linear')
            
            # Apply position-dependent correction factors based on observed reconstruction biases
            seq_len = d_shape[1]
            position = torch.linspace(0, 1, seq_len, device=d_out.device).view(1, 1, -1)
            
            if j == 0:  # Level 1 (80% left reconstruction) - apply strong correction to right side
                # Corrective mask: increase gain on right side (1.0 → 3.0)
                correction = 1.0 + 2.0 * torch.pow(position, 2.0)  # Quadratic increase to right
            elif j == 1:  # Level 2 (50% right reconstruction) - apply correction to left side
                # Corrective mask: increase gain on left side (2.5 → 1.0)
                correction = 2.5 - 1.5 * position  # Linear decrease to right
            elif j == 2:  # Level 3 (30% right reconstruction) - strong left boost
                # Corrective mask: strong emphasis on left side (3.0 → 1.0)
                correction = 3.0 - 2.0 * torch.pow(position, 0.7)  # Non-linear decrease to right
            else:
                correction = torch.ones_like(position)
                
            # Expand correction to batch dimension
            correction = correction.expand(batch_size, 1, -1)
                
            # Apply the correction factors
            d_out = d_out * correction
            d_out = d_out.squeeze(1)
            d_outs.append(d_out)
        
        # Return as coefficient dictionary
        return {
            'a': a_out,
            'd': d_outs
        }

class WaveletEchoMatrix(nn.Module):
    def __init__(self, wavelet='db4', levels=4, hidden_dim=64, latent_dim=256):
        super(WaveletEchoMatrix, self).__init__()
        self.wavelet = wavelet
        self.levels = levels
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # DWT and IDWT modules
        self.dwt = DWT(wave=wavelet, level=levels)
        self.idwt = IDWT(wave=wavelet)
        
        # Encoder and decoder
        self.encoder = UnifiedEncoder(levels=levels, hidden_dim=hidden_dim)
        self.decoder = UnifiedDecoder(levels=levels, hidden_dim=hidden_dim, latent_dim=latent_dim)
        
    def forward(self, x):
        """
        Forward pass of WaveletEchoMatrix
        Args:
            x: Input audio tensor of shape [B, T]
        Returns:
            output: Dictionary containing reconstructed audio and intermediate representations
        """
        # Apply DWT
        coeffs = self.dwt(x)
        
        # Store original shapes for reconstruction
        orig_shapes = {
            'a': coeffs['a'].shape,
            'd': [d.shape for d in coeffs['d']]
        }
        
        # Normalize coefficients with improved stability
        norm_coeffs, stats = self.dwt.normalize_coeffs(coeffs)
        
        # Apply frequency-aware adaptive thresholding for sparsity
        thresh_coeffs = self.dwt.threshold_coeffs(norm_coeffs)
        
        # Encode to latent space
        z, encoded_features = self.encoder(thresh_coeffs)
        
        # Decode from latent space using skip connections
        rec_coeffs = self.decoder(z, encoded_features, orig_shapes)
        
        # Denormalize coefficients using stored statistics
        denorm_coeffs = {
            'a': rec_coeffs['a'] * stats['stds'][0] + stats['means'][0],
            'd': [rec_coeffs['d'][j] * stats['stds'][j+1] + stats['means'][j+1] for j in range(self.levels)]
        }
        
        # Apply IDWT for reconstruction
        reconstructed = self.idwt(denorm_coeffs)
        
        # Ensure output has same length as input
        if reconstructed.shape[1] != x.shape[1]:
            reconstructed = F.pad(reconstructed, (0, x.shape[1] - reconstructed.shape[1])) if reconstructed.shape[1] < x.shape[1] else reconstructed[:, :x.shape[1]]
        
        # Return all relevant outputs
        output = {
            'reconstructed': reconstructed,
            'latent': z,
            'coeffs': coeffs,
            'rec_coeffs': denorm_coeffs
        }
        
        return output