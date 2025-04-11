import torch
import torch.nn as nn
import torch.nn.functional as F
from models.dwt import DWT, IDWT

class HierarchicalEncoder(nn.Module):
    def __init__(self, levels=6, hidden_dim=128):
        super(HierarchicalEncoder, self).__init__()
        self.levels = levels
        self.hidden_dim = hidden_dim
        
        # Approximation branch encoder (for level 6 approximation coefficients)
        self.approx_encoder = nn.Sequential(
            nn.Conv1d(1, hidden_dim // 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(hidden_dim // 2, hidden_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2)
        )
        
        # Detail branch encoders for each level
        self.detail_encoders = nn.ModuleList()
        for i in range(levels):
            if i < 4:  # Levels 1-4 use larger networks due to more coefficients
                encoder = nn.Sequential(
                    nn.Conv1d(1, hidden_dim // 2, kernel_size=3, padding=1),
                    nn.LeakyReLU(0.2),
                    nn.Conv1d(hidden_dim // 2, hidden_dim, kernel_size=3, padding=1),
                    nn.LeakyReLU(0.2),
                    nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                    nn.LeakyReLU(0.2)
                )
            else:  # Levels 5-6 use smaller networks
                encoder = nn.Sequential(
                    nn.Conv1d(1, hidden_dim // 2, kernel_size=3, padding=1),
                    nn.LeakyReLU(0.2),
                    nn.Conv1d(hidden_dim // 2, hidden_dim, kernel_size=3, padding=1),
                    nn.LeakyReLU(0.2)
                )
            self.detail_encoders.append(encoder)
        
        # Fusion layer (to combine all encoded branches)
        fusion_input_size = hidden_dim * (levels + 1)  # +1 for approximation coeffs
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_size, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2)
        )
        
    def forward(self, coeffs):
        """
        Forward pass of hierarchical encoder
        Args:
            coeffs: Dictionary of wavelet coefficients {'a': approx, 'd': [detail1, ..., detailN]}
        Returns:
            z: Latent representation
        """
        batch_size = coeffs['a'].shape[0]
        
        # Process approximation coefficients
        a = coeffs['a'].unsqueeze(1)  # [B, 1, T]
        encoded_a = self.approx_encoder(a)
        # Global pooling to get fixed-size feature vector
        pooled_a = F.adaptive_avg_pool1d(encoded_a, 1).view(batch_size, -1)
        
        # Process detail coefficients for each level
        pooled_ds = []
        for j, d in enumerate(coeffs['d']):
            d = d.unsqueeze(1)  # [B, 1, T]
            encoded_d = self.detail_encoders[j](d)
            # Global pooling
            pooled_d = F.adaptive_avg_pool1d(encoded_d, 1).view(batch_size, -1)
            pooled_ds.append(pooled_d)
        
        # Concatenate all pooled features
        concat_features = torch.cat([pooled_a] + pooled_ds, dim=1)
        
        # Apply fusion layer to get final latent representation
        z = self.fusion(concat_features)
        
        return z

class HierarchicalDecoder(nn.Module):
    def __init__(self, levels=6, hidden_dim=128, latent_dim=256):
        super(HierarchicalDecoder, self).__init__()
        self.levels = levels
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # Initial expansion from latent to hidden
        self.initial_expansion = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.LeakyReLU(0.2)
        )
        
        # Branch distribution layers - small linear projections to create branch-specific features
        self.approx_features = nn.Linear(512, hidden_dim)
        self.detail_features = nn.ModuleList([nn.Linear(512, hidden_dim) for _ in range(levels)])
        
        # Parameter-efficient decoder for approximation coefficients
        self.approx_decoder = nn.Sequential(
            nn.ConvTranspose1d(hidden_dim, hidden_dim // 2, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(hidden_dim // 2, hidden_dim // 4, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(hidden_dim // 4, 1, kernel_size=4, stride=2, padding=1)
        )
        
        # Create efficient decoders for detail coefficients at each level
        # Each decoder uses a different number of upsampling operations based on its target size
        self.detail_decoders = nn.ModuleList()
        for i in range(levels):
            # Number of upsampling operations needed for this level
            # More upsampling for higher frequency detail coefficients
            num_upsample = min(5, levels - i + 2)  # Empirical formula
            
            layers = []
            in_channels = hidden_dim
            
            for j in range(num_upsample):
                out_channels = max(hidden_dim // (2 ** (j+1)), 8)  # Decrease channels as we upsample
                layers.extend([
                    nn.ConvTranspose1d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
                    nn.LeakyReLU(0.2)
                ])
                in_channels = out_channels
            
            # Final layer to output 1 channel
            layers.append(nn.Conv1d(in_channels, 1, kernel_size=3, padding=1))
            
            self.detail_decoders.append(nn.Sequential(*layers))
        
    def forward(self, z, orig_shapes):
        """
        Forward pass of hierarchical decoder
        Args:
            z: Latent representation
            orig_shapes: Original shapes of wavelet coefficients
        Returns:
            coeffs: Reconstructed wavelet coefficients
        """
        batch_size = z.shape[0]
        
        # Initial expansion
        h = self.initial_expansion(z)
        
        # Generate features for approximation coefficients
        a_features = self.approx_features(h).view(batch_size, -1, 1)  # [B, hidden_dim, 1]
        
        # Decode approximation coefficients
        a_shape = orig_shapes['a']
        a_rec = self.approx_decoder(a_features)
        a_rec = F.interpolate(a_rec, size=a_shape[1])
        a_rec = a_rec.squeeze(1)  # [B, T]
        
        # Reconstruct detail coefficients for each level
        d_rec = []
        for j in range(self.levels):
            # Get level-specific features
            d_features = self.detail_features[j](h).view(batch_size, -1, 1)  # [B, hidden_dim, 1]
            
            # Apply decoder
            d_shape = orig_shapes['d'][j]
            d = self.detail_decoders[j](d_features)
            d = F.interpolate(d, size=d_shape[1])
            d = d.squeeze(1)  # [B, T]
            d_rec.append(d)
        
        # Combine into coefficient dictionary
        reconstructed_coeffs = {
            'a': a_rec,
            'd': d_rec
        }
        
        return reconstructed_coeffs

class WaveletEchoMatrix(nn.Module):
    def __init__(self, wavelet='db4', levels=6, hidden_dim=128, latent_dim=256):
        super(WaveletEchoMatrix, self).__init__()
        self.wavelet = wavelet
        self.levels = levels
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # DWT and IDWT modules
        self.dwt = DWT(wave=wavelet, level=levels)
        self.idwt = IDWT(wave=wavelet)
        
        # Encoder and decoder
        self.encoder = HierarchicalEncoder(levels=levels, hidden_dim=hidden_dim)
        self.decoder = HierarchicalDecoder(levels=levels, hidden_dim=hidden_dim, latent_dim=latent_dim)
        
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
        
        # Normalize coefficients
        norm_coeffs, stats = self.dwt.normalize_coeffs(coeffs)
        
        # Apply thresholding for sparsity
        thresh_coeffs = self.dwt.threshold_coeffs(norm_coeffs)
        
        # Encode to latent space
        z = self.encoder(thresh_coeffs)
        
        # Decode from latent space
        rec_coeffs = self.decoder(z, orig_shapes)
        
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