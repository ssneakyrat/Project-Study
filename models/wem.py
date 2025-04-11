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

class UnifiedEncoder(nn.Module):
    """
    Simplified unified encoder for wavelet coefficients with better gradient flow
    """
    def __init__(self, levels=4, hidden_dim=64):
        super(UnifiedEncoder, self).__init__()
        self.levels = levels
        self.hidden_dim = hidden_dim
        
        # Initial convolutions for each coefficient level
        self.initial_convs = nn.ModuleList()
        for i in range(levels + 1):  # +1 for approximation coefficients
            # Use smaller kernels for higher frequency details
            kernel_size = 5 if i >= levels-1 else 7
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
            self.encoder_blocks.append(
                nn.Sequential(
                    ResidualBlock(hidden_dim),
                    ResidualBlock(hidden_dim)
                )
            )
        
        # Progressive downsampling with preserved information
        self.downsample = nn.ModuleList()
        for i in range(levels + 1):
            # Different downsampling factors based on coefficient length
            factor = min(2**(levels-i), 8)  # Limit maximum downsampling
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

class UnifiedDecoder(nn.Module):
    """
    Simplified unified decoder with better gradient flow and skip connections
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
            self.branch_projections.append(
                nn.Sequential(
                    nn.Linear(512, hidden_dim * 4),
                    nn.LayerNorm(hidden_dim * 4),
                    nn.LeakyReLU(0.2)
                )
            )
        
        # Unified decoder blocks with residual connections
        self.decoder_blocks = nn.ModuleList()
        for i in range(levels + 1):
            # Add extra input channels for skip connections
            self.decoder_blocks.append(
                nn.Sequential(
                    nn.Conv1d(hidden_dim*2, hidden_dim, kernel_size=3, padding=1),
                    nn.BatchNorm1d(hidden_dim),
                    nn.LeakyReLU(0.2),
                    ResidualBlock(hidden_dim),
                    ResidualBlock(hidden_dim)
                )
            )
        
        # Final output layers
        self.output_layers = nn.ModuleList()
        for i in range(levels + 1):
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
        a_up = F.interpolate(a_feat, size=encoded_features[0].size(2))
        a_concat = torch.cat([a_up, encoded_features[0]], dim=1)
        
        # Apply decoder blocks
        a_dec = self.decoder_blocks[0](a_concat)
        
        # Final layer and reshape to match original shape
        a_out = self.output_layers[0](a_dec)
        a_out = F.interpolate(a_out, size=a_shape[1])
        a_out = a_out.squeeze(1)
        
        # Reconstruct detail coefficients
        d_outs = []
        for j in range(self.levels):
            d_shape = orig_shapes['d'][j]
            d_feat = self.branch_projections[j+1](h).view(batch_size, self.hidden_dim, 4)
            
            # Upsample to match encoder feature size then concatenate with encoder features
            d_up = F.interpolate(d_feat, size=encoded_features[j+1].size(2))
            d_concat = torch.cat([d_up, encoded_features[j+1]], dim=1)
            
            # Apply decoder blocks
            d_dec = self.decoder_blocks[j+1](d_concat)
            
            # Final layer and reshape to match original shape
            d_out = self.output_layers[j+1](d_dec)
            d_out = F.interpolate(d_out, size=d_shape[1])
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
        
        # Apply adaptive thresholding for sparsity
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