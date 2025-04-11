import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl
from utils.loss import TriangularWindow

class InverseWaveletTransform(nn.Module):
    def __init__(self, config, wavelet_type="morlet", channels=8, output_size=16000):
        super().__init__()
        self.wavelet_type = wavelet_type
        self.channels = channels
        self.output_size = output_size
        self.kernel_size = 101
        
        # Initialize synthesis filters (for inverse wavelet transform)
        self.synthesis_filters = nn.ConvTranspose1d(
            channels, 1, kernel_size=self.kernel_size, stride=2, padding=self.kernel_size//2, bias=False
        )
        
        # Initialize the synthesis filters with mathematically accurate inverse wavelets
        with torch.no_grad():
            scales = torch.logspace(1, 3, channels)
            for i, scale in enumerate(scales):
                t = torch.linspace(-50, 50, 101)
                
                # Create synthesis filter that forms a dual frame with the analysis filter
                if wavelet_type == "morlet":
                    # Dual frame of Morlet wavelet for perfect reconstruction
                    morlet = torch.exp(-t**2/(2*scale**2)) * torch.cos(5*t/scale)
                    morlet = morlet.flip(0)  # Time reversal for dual frame
                    
                elif wavelet_type == "mexican_hat":
                    # Dual frame of Mexican hat wavelet
                    morlet = (1 - t**2/scale**2) * torch.exp(-t**2/(2*scale**2))
                    morlet = morlet.flip(0)  # Time reversal
                    
                else:  # Default to Morlet
                    morlet = torch.exp(-t**2/(2*scale**2)) * torch.cos(5*t/scale)
                    morlet = morlet.flip(0)
                    
                # Normalize synthesis filter for energy preservation
                # |a|^(-1/2) factor from the wavelet transform definition
                morlet = morlet / torch.sqrt(scale) / torch.norm(morlet)
                
                # Enforce symmetry constraints to maintain wavelet admissibility
                if self.kernel_size % 2 == 1:  # For odd kernel size
                    center = self.kernel_size // 2
                    # Make filter symmetric around center for linear phase
                    symmetric_part = (morlet[:center] + morlet[center+1:].flip(0)) / 2
                    morlet[:center] = symmetric_part
                    morlet[center+1:] = symmetric_part.flip(0)
                
                self.synthesis_filters.weight[i, 0, :] = morlet
        
        # Enforce perfect reconstruction condition: ∑_k h[k]h[2n-k] = δ[n]
        with torch.no_grad():
            # Get filter weights
            weights = self.synthesis_filters.weight.squeeze(1)  # [channels, kernel_size]
            
            # Orthogonalize the filter bank for improved perfect reconstruction
            # Step 1: Compute filter inner products (Gram matrix)
            gram = torch.mm(weights, weights.t())
            
            # Step 2: Apply Cholesky decomposition-based orthogonalization
            # Add small diagonal term for numerical stability
            gram += torch.eye(channels, device=weights.device) * 1e-6
            L = torch.linalg.cholesky(gram)
            
            # Step 3: Solve the system (updated to use the newer API)
            ortho_weights = torch.linalg.solve(L, weights)
            
            # Step 4: Normalize each filter
            for i in range(channels):
                ortho_weights[i] = ortho_weights[i] / torch.norm(ortho_weights[i])
            
            # Update filter weights
            self.synthesis_filters.weight = nn.Parameter(ortho_weights.unsqueeze(1))
        
        # Adaptive modulation for synthesis
        self.modulation = nn.Conv1d(channels, channels, kernel_size=1)
        
        # Flag to indicate if encoder parameters have been set
        self.encoder_params_set = False
    
    def set_encoder_parameters(self, wavelet_params):
        """
        Set the synthesis filters and modulation weights using parameters from the encoder
        
        Args:
            wavelet_params: Dictionary containing 'adaptive_filters' and 'feature_modulation_weights'
        """
        if wavelet_params is None:
            return
        
        adaptive_filters = wavelet_params.get('adaptive_filters')
        feature_modulation_weights = wavelet_params.get('feature_modulation_weights')
        
        if adaptive_filters is not None:
            # Compute synthesis filters that form a biorthogonal system with analysis filters
            # This enforces the perfect reconstruction condition: ∑_n h[n-2k]g[n] = δ[k]
            with torch.no_grad():
                # Extract analysis filters
                analysis_filters = adaptive_filters.squeeze(1)  # [channels, kernel_size]
                
                # To create a biorthogonal system, we need to solve for synthesis filters
                # that satisfy the perfect reconstruction condition
                kernel_size = analysis_filters.shape[1]
                channels = analysis_filters.shape[0]
                
                # Step 1: Compute the cross-correlation matrix of analysis filters at even shifts
                # This represents the inner products between analysis filters at shifts of 2k
                R = torch.zeros((channels, channels), device=analysis_filters.device)
                
                for i in range(channels):
                    for j in range(channels):
                        # Compute cross-correlation (convolution with time-reversed filter)
                        # We only care about even indices for downsampling by 2
                        h_i = analysis_filters[i]
                        h_j = analysis_filters[j].flip(0)  # Time-reverse
                        
                        # Convolve h_i with time-reversed h_j
                        # This is equivalent to cross-correlation between h_i and h_j
                        corr = torch.nn.functional.conv1d(
                            h_i.view(1, 1, -1), 
                            h_j.view(1, 1, -1), 
                            padding=kernel_size-1
                        ).squeeze()
                        
                        # Extract values at even indices (for downsampling by 2)
                        # Center value represents zero shift
                        center = corr.size(0) // 2
                        R[i, j] = corr[center]
                
                # Step 2: Add regularization for numerical stability
                R = R + torch.eye(channels, device=R.device) * 1e-6
                
                # Step 3: Compute synthesis filters by inverting the correlation matrix
                # G = H⁻¹ where H is the matrix of analysis filters
                try:
                    R_inv = torch.inverse(R)
                except:
                    # If inverse fails, use pseudo-inverse with more regularization
                    R = R + torch.eye(channels, device=R.device) * 1e-4
                    R_inv = torch.inverse(R)
                
                # Step 4: Apply the inverse to get synthesis filters
                synthesis_filters = torch.zeros_like(analysis_filters)
                
                for i in range(channels):
                    g_i = torch.zeros(kernel_size, device=analysis_filters.device)
                    for j in range(channels):
                        # Apply R_inv to the time-reversed analysis filters
                        g_i = g_i + R_inv[i, j] * analysis_filters[j].flip(0)
                    
                    # Normalize for energy preservation
                    g_i = g_i / torch.norm(g_i)
                    
                    # Ensure filter has zero mean (wavelet admissibility)
                    g_i = g_i - g_i.mean()
                    
                    # Update synthesis filter
                    synthesis_filters[i] = g_i
                
                # Step 5: Reshape and assign to synthesis_filters.weight
                self.synthesis_filters.weight.copy_(synthesis_filters.unsqueeze(1))
        
        if feature_modulation_weights is not None:
            # Set modulation weights to match encoder's feature modulation
            with torch.no_grad():
                self.modulation.weight.copy_(feature_modulation_weights)
        
        self.encoder_params_set = True
    
    def forward(self, x):
        """
        Inverse wavelet transform using synthesis filter bank
        
        Mathematically implements the inverse continuous wavelet transform:
        x(t) = C_ψ^(-1) ∫∫ W_ψx(a,b) ψ((t-b)/a) da db/a²
        
        where C_ψ is the wavelet admissibility constant.
        
        Args:
            x: Wavelet coefficients [B, C, T]
            
        Returns:
            Reconstructed signal [B, 1, 2*T]
        """
        # Apply adaptive modulation
        x = self.modulation(x)
        
        # Apply synthesis filters
        # This implements the integration over scales and positions
        # The TransposedConv1d with stride=2 provides the necessary upsampling
        output = self.synthesis_filters(x)
        
        # Ensure output size matches expected dimension
        if output.size(2) != self.output_size:
            output = F.interpolate(output, size=self.output_size, mode='linear', align_corners=False)
            
        return output

class WaveletDecoder(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        
        bottleneck_channels = config['model']['bottleneck_channels']
        hidden_channels = config['model']['hidden_channels']
        wavelet_channels = config['model']['wavelet_channels']  # Use the same number of channels as the encoder
        
        # Init for debugging
        self.debug_print = True
        
        # Gradual upsampling as specified in architecture
        # Modified to ensure symmetric alignment with encoder path
        # [B,256] → Linear → [B,256,2000] (matching encoder's bottleneck dimension)
        self.upsampling = nn.Sequential(
            nn.Linear(bottleneck_channels, bottleneck_channels * 2000),
            nn.LeakyReLU(),
            nn.Unflatten(1, (bottleneck_channels, 2000))
        )
        
        # Reconstruction with symmetric transpose convolutions to match encoder path
        # [B,256,2000] → [B,32,2000] (channel reduction)
        self.channel_reduction = nn.Conv1d(
            bottleneck_channels, 
            hidden_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )
        
        # [B,32,2000] → [B,32,4000] (symmetric to encoder's second conv)
        self.upconv1 = nn.ConvTranspose1d(
            in_channels=hidden_channels,  # Explicitly specify in_channels=32
            out_channels=hidden_channels, 
            kernel_size=4, 
            stride=2, 
            padding=1,
            output_padding=0
        )
        
        # [B,32,4000] → [B,16,8000] (symmetric to encoder's first conv)
        self.upconv2 = nn.ConvTranspose1d(
            in_channels=hidden_channels,  # Explicitly specify in_channels=32 
            out_channels=wavelet_channels, 
            kernel_size=4, 
            stride=2, 
            padding=1,
            output_padding=0
        )
        
        # Inverse wavelet transform: [B,wavelet_channels,8000] → [B,1,16000]
        self.inverse_wavelet = InverseWaveletTransform( 
            config,
            wavelet_type=config['model']['wavelet_type'],
            channels=wavelet_channels,
            output_size=config['model']['input_size']
        )
        
        # Triangular window for overlap-add reconstruction
        self.window = TriangularWindow(config['model']['input_size'])
        
        # Track previous frame for overlapping - starts as None
        self.prev_frame = None
        
        # Overlap factor for frame processing
        self.overlap_factor = 0.25  # From architecture specification
    
    def set_encoder_parameters(self, wavelet_params):
        """Set the encoder parameters for the inverse wavelet transform"""
        if wavelet_params is not None:
            self.inverse_wavelet.set_encoder_parameters(wavelet_params)
    
    def forward(self, z):
        """
        Forward pass through the decoder
        z: [B, 256] bottleneck representation
        Returns: [B, 1, 16000] reconstructed audio
        """
        # Upsampling: [B,256] → Reshape/Linear → [B,256,125]
        x = self.upsampling(z)
        
        # [B,256,125] → Transpose-Conv1D → [B,128,500]
        x = F.relu(self.upconv1(x))
        
        # Reconstruction: 3×Transpose-Conv1D as specified in architecture
        # [B,128,500] → [B,64,1000]
        x = F.relu(self.upconv2(x))
        
        # [B,64,1000] → [B,32,2000]
        x = F.relu(self.upconv3(x))
        
        # [B,32,2000] → [B,wavelet_channels,8000] with a single stride 4 convolution
        x = F.relu(self.upconv4(x))
        
        # Inverse wavelet transform: [B,wavelet_channels,8000] → [B,1,16000]
        x = self.inverse_wavelet(x)
        
        return x
    
    def process_overlapping_frames(self, frame):
        """
        Process overlapping frames with triangular windowing for smooth reconstruction
        Implements: x̂ = ∑_t w_t · x̂_t where w_t is a triangular window
        
        Args:
            frame: Current reconstructed frame [B, 1, T]
            
        Returns:
            Processed frame with overlap-add if previous frame exists
        """
        # Apply triangular window to current frame
        windowed_frame = self.window(frame)
        
        # If no previous frame, store current and return as is
        if self.prev_frame is None:
            self.prev_frame = frame.detach()
            return windowed_frame
            
        # Calculate overlap size
        T = frame.size(2)
        overlap_size = int(T * self.overlap_factor)
        
        # Create output buffer
        output = torch.zeros_like(frame)
        
        # Copy previous frame's end (with overlap)
        output[:, :, :overlap_size] = self.prev_frame[:, :, -overlap_size:]
        
        # Add overlapping region with windowing for smooth transition
        output[:, :, :overlap_size] += windowed_frame[:, :, :overlap_size]
        
        # Copy remaining part of current frame
        output[:, :, overlap_size:] = windowed_frame[:, :, overlap_size:]
        
        # Store current frame for next iteration
        self.prev_frame = frame.detach()
        
        return output
    
    def encode_decode(self, z, wavelet_params=None, use_overlap=True):
        """
        Decode with optional overlap-add processing
        
        Args:
            z: Latent vector [B, 256]
            wavelet_params: Wavelet parameters from encoder
            use_overlap: Whether to use overlap-add processing
            
        Returns:
            Reconstructed signal and previous frame for loss calculation
        """
        # Set encoder parameters for the inverse wavelet transform
        if wavelet_params is not None:
            self.set_encoder_parameters(wavelet_params)
        
        # Decode the latent vector
        x_hat = self(z)
        
        # Store previous frame for loss calculation
        prev_frame = self.prev_frame
        
        # Apply overlap-add processing if requested
        if use_overlap:
            x_hat = self.process_overlapping_frames(x_hat)
        
        return x_hat, prev_frame
    
    def training_step(self, batch, batch_idx):
        x = batch
        
        # For standalone testing, we generate random latent vectors
        # In real training this would come from the encoder
        batch_size = x.shape[0]
        z = torch.randn(batch_size, self.config['model']['bottleneck_channels'], device=x.device)
        
        # Decode
        x_hat = self(z)
        
        # MSE reconstruction loss
        loss = F.mse_loss(x_hat, x)
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch
        
        # Generate random latent vectors for standalone testing
        batch_size = x.shape[0]
        z = torch.randn(batch_size, self.config['model']['bottleneck_channels'], device=x.device)
        
        # Decode
        x_hat = self(z)
        
        # MSE reconstruction loss
        loss = F.mse_loss(x_hat, x)
        
        # Log metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
    
    def configure_optimizers(self):
        """Adam optimizer with learning rate from config"""
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.config['training']['learning_rate']
        )
        return optimizer