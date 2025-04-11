import torch
import torch.nn as nn
import torch.nn.functional as F

class WaveletLoss(nn.Module):
    """
    Combined loss function for the wavelet network:
    Loss = MSE(x, x̂) + L1(W_ψ[x], W_ψ[x̂]) + KL(z, N(0,1))
    """
    def __init__(self, wavelet_transform, mse_weight=1.0, wavelet_weight=1.0, kl_weight=0.1):
        super().__init__()
        self.wavelet_transform = wavelet_transform
        self.mse_weight = mse_weight
        self.wavelet_weight = wavelet_weight
        self.kl_weight = kl_weight
    
    def forward(self, x, x_hat, z=None, z_mean=None, z_logvar=None):
        """
        Compute the combined loss.
        
        Args:
            x: Original signal [B, 1, T]
            x_hat: Reconstructed signal [B, 1, T]
            z: Latent vector (optional)
            z_mean: Mean of latent space for KL divergence
            z_logvar: Log variance of latent space for KL divergence
            
        Returns:
            Total loss and dictionary of individual loss components
        """
        # MSE reconstruction loss in time domain
        mse_loss = F.mse_loss(x_hat, x)
        
        # Wavelet domain loss: L1(W_ψ[x], W_ψ[x̂])
        # Apply wavelet transform to both signals
        with torch.no_grad():
            wx = self.wavelet_transform(x)
        wx_hat = self.wavelet_transform(x_hat)
        
        # L1 loss in wavelet domain
        wavelet_loss = F.l1_loss(wx_hat, wx)
        
        # Initialize total loss
        total_loss = self.mse_weight * mse_loss + self.wavelet_weight * wavelet_loss
        
        # Add KL divergence term if provided
        kl_loss = 0.0
        if z_mean is not None and z_logvar is not None:
            # KL divergence: KL(q(z|x) || p(z))
            # For VAE with normal prior: -0.5 * sum(1 + log(σ²) - μ² - σ²)
            kl_loss = -0.5 * torch.sum(1 + z_logvar - z_mean.pow(2) - z_logvar.exp())
            
            # Normalize by batch size
            kl_loss = kl_loss / x.size(0)
            
            # Add to total loss
            total_loss += self.kl_weight * kl_loss
        
        # Return total loss and components
        loss_components = {
            "mse_loss": mse_loss.item(),
            "wavelet_loss": wavelet_loss.item(),
            "kl_loss": kl_loss if isinstance(kl_loss, float) else kl_loss.item()
        }
        
        return total_loss, loss_components