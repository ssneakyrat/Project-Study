import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import numpy as np
import io
from PIL import Image
import torchvision.transforms as T

class EnvelopeLoss(nn.Module):
    """Loss function that compares signal envelopes rather than using FFT."""
    def __init__(self, alpha=0.4, window_size=128):
        super().__init__()
        self.alpha = alpha
        self.window_size = window_size
        
    def forward(self, pred, target):
        # Regular MSE loss
        direct_loss = F.mse_loss(pred, target)
        
        # Compute envelopes (absolute value + smoothing)
        pred_env = self._compute_envelope(pred)
        target_env = self._compute_envelope(target)
        
        # Envelope MSE
        env_loss = F.mse_loss(pred_env, target_env)
        
        # Combined loss
        return (1 - self.alpha) * direct_loss + self.alpha * env_loss
    
    def _compute_envelope(self, signal):
        # Take absolute value
        signal_abs = torch.abs(signal)
        
        # Apply smoothing with average pooling
        # Pad first to maintain signal length
        pad = self.window_size // 2
        padded = F.pad(signal_abs, (pad, pad), mode='reflect')
        
        # Use average pooling for smoothing
        envelope = F.avg_pool1d(
            padded.unsqueeze(1),  # Add channel dim
            kernel_size=self.window_size, 
            stride=1,
            padding=0
        ).squeeze(1)  # Remove channel dim
        
        return envelope


class MultiScaleLoss(nn.Module):
    """Compares signals at multiple resolutions."""
    def __init__(self, scales=[1, 2, 4, 8], weights=None):
        super().__init__()
        self.scales = scales
        
        # Default to equal weighting if not provided
        if weights is None:
            weights = [1.0/len(scales)] * len(scales)
        else:
            # Normalize weights
            total = sum(weights)
            weights = [w/total for w in weights]
            
        self.weights = weights
        
    def forward(self, pred, target):
        total_loss = 0.0
        
        for scale, weight in zip(self.scales, self.weights):
            if scale == 1:
                # Original resolution
                total_loss += weight * F.mse_loss(pred, target)
            else:
                # Downsampled resolution
                pred_down = self._downsample(pred, scale)
                target_down = self._downsample(target, scale)
                total_loss += weight * F.mse_loss(pred_down, target_down)
                
        return total_loss
    
    def _downsample(self, signal, factor):
        # Use average pooling for downsampling
        return F.avg_pool1d(
            signal.unsqueeze(1),  # Add channel dim
            kernel_size=factor,
            stride=factor,
            padding=0
        ).squeeze(1)  # Remove channel dim
    
class Signal2DTo1DModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        
        # Extract config parameters
        self.input_shape = config['model']['input_shape']
        self.output_shape = config['model']['output_shape']
        self.encoder_filters = config['model']['encoder_filters']
        self.decoder_filters = config['model']['decoder_filters']
        self.bottleneck_size = config['model']['bottleneck_size']  # Consider increasing this to 2048
        self.dropout_rate = config['model']['dropout_rate']
        self.lr = config['training']['learning_rate']
        self.weight_decay = config['training']['weight_decay']
        
        # Define encoder blocks to capture intermediate features for skip connections
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, self.encoder_filters[0], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.encoder_filters[0]),
            nn.LeakyReLU(0.2)
        )
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(self.encoder_filters[0], self.encoder_filters[1], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.encoder_filters[1]),
            nn.LeakyReLU(0.2)
        )
        
        self.enc3 = nn.Sequential(
            nn.Conv2d(self.encoder_filters[1], self.encoder_filters[2], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.encoder_filters[2]),
            nn.LeakyReLU(0.2)
        )
        
        self.enc4 = nn.Sequential(
            nn.Conv2d(self.encoder_filters[2], self.encoder_filters[3], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.encoder_filters[3]),
            nn.LeakyReLU(0.2)
        )
        
        # Calculate encoder output dimensions
        self.enc_output_dim = self._get_encoder_output_dim()
        
        # Enhanced bottleneck
        self.bottleneck = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.encoder_filters[3], self.bottleneck_size),
            nn.Dropout(self.dropout_rate),
            nn.LeakyReLU(0.2)
        )
        
        # Decoder input
        self.decoder_input = nn.Linear(self.bottleneck_size, 256)  # 16*16*1
        
        # Decoder blocks with skip connections
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(1, self.decoder_filters[0], kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(self.decoder_filters[0]),
            nn.LeakyReLU(0.2)
        )
        
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(self.decoder_filters[0] + self.encoder_filters[3], 
                              self.decoder_filters[1], kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(self.decoder_filters[1]),
            nn.LeakyReLU(0.2)
        )
        
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(self.decoder_filters[1] + self.encoder_filters[2], 
                              self.decoder_filters[2], kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(self.decoder_filters[2]),
            nn.LeakyReLU(0.2)
        )
        
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(self.decoder_filters[2] + self.encoder_filters[1], 
                              self.decoder_filters[3], kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(self.decoder_filters[3]),
            nn.LeakyReLU(0.2)
        )
        
        # 1D CNN approach for final conversion
        self.final_conv = nn.Sequential(
            nn.AdaptiveAvgPool2d((32, 1)),  # Convert to shape suitable for 1D convs
            nn.Flatten(2),  # Flatten spatial dims but keep batch and channel
            nn.Conv1d(self.decoder_filters[3], 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2)
        )
        
        # Final projection to output size
        self.output_proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 32, 4096),
            nn.LeakyReLU(0.2),
            nn.Dropout(self.dropout_rate),
            nn.Linear(4096, self.output_shape[0])
        )
        
        # Custom loss
         # Initialize loss functions
        self.use_envelope_loss = config['training'].get('use_envelope_loss', True)
        self.use_multiscale_loss = config['training'].get('use_multiscale_loss', True)
        
        # Loss weights (how much each loss contributes)
        self.direct_loss_weight = config['training'].get('direct_loss_weight', 0.3)
        self.envelope_loss_weight = config['training'].get('envelope_loss_weight', 0.4)
        self.multiscale_loss_weight = config['training'].get('multiscale_loss_weight', 0.3)
        
        # Initialize losses
        window_size = config['training'].get('envelope_window_size', 128)
        self.envelope_loss = EnvelopeLoss(
            alpha=0.5,  # Not used in combined mode
            window_size=window_size
        )
        
        scales = config['training'].get('multiscale_scales', [1, 2, 4, 8])
        self.multiscale_loss = MultiScaleLoss(
            scales=scales,
            weights=None  # Equal weighting
        )
        
        # For baseline comparison
        self.mse_loss = nn.MSELoss()
    
    def _get_encoder_output_dim(self):
        # Helper method to calculate the encoder output dimension
        x = torch.zeros(1, 1, self.input_shape[0], self.input_shape[1])
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        return e4.shape
    
    def forward(self, x):
        # Ensure input is correct shape (batch_size, channels, height, width)
        if len(x.shape) == 3:
            x = x.unsqueeze(1)  # Add channel dimension if missing
        
        # Encoder with intermediate features
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        
        # Bottleneck
        z = self.bottleneck(e4)
        
        # Decoder pre-processing
        z = self.decoder_input(z)
        z = z.view(-1, 1, 16, 16)  # Reshape for transpose convolutions
        
        # Decoder with skip connections
        d1 = self.dec1(z)
        
        # Use skip connections - need to handle different spatial dimensions
        # We use adaptive pooling to ensure compatible dimensions
        e4_resized = F.adaptive_avg_pool2d(e4, d1.shape[2:])
        d1_skip = torch.cat([d1, e4_resized], dim=1)
        
        d2 = self.dec2(d1_skip)
        e3_resized = F.adaptive_avg_pool2d(e3, d2.shape[2:])
        d2_skip = torch.cat([d2, e3_resized], dim=1)
        
        d3 = self.dec3(d2_skip)
        e2_resized = F.adaptive_avg_pool2d(e2, d3.shape[2:])
        d3_skip = torch.cat([d3, e2_resized], dim=1)
        
        d4 = self.dec4(d3_skip)
        
        # 1D conversion
        x = self.final_conv(d4)
        
        # Final projection to output
        x = self.output_proj(x)
        
        return x
    
    def configure_optimizers(self):
        # Use AdamW instead of Adam for better weight decay behavior
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        
        if self.hparams['training']['lr_scheduler']:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=self.hparams['training']['lr_factor'],
                patience=self.hparams['training']['lr_patience'],
                verbose=True
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val_loss',
                    'interval': 'epoch',
                    'frequency': 1
                }
            }
        else:
            return optimizer
    
    def _compute_combined_loss(self, pred, target):
        """Compute weighted combination of losses"""
        loss_components = {}
        
        # Basic MSE loss (always computed for comparison)
        direct_loss = self.mse_loss(pred, target)
        loss_components['direct_loss'] = direct_loss
        
        # Initialize combined loss with direct loss component
        combined_loss = self.direct_loss_weight * direct_loss
        
        # Add envelope loss if enabled
        if self.use_envelope_loss:
            # Disable autocast to avoid potential precision issues
            with torch.cuda.amp.autocast(enabled=False):
                env_loss = self.envelope_loss(pred.float(), target.float())
            loss_components['envelope_loss'] = env_loss
            combined_loss += self.envelope_loss_weight * env_loss
            
        # Add multiscale loss if enabled
        if self.use_multiscale_loss:
            # Disable autocast to avoid potential precision issues
            with torch.cuda.amp.autocast(enabled=False):
                ms_loss = self.multiscale_loss(pred.float(), target.float())
            loss_components['multiscale_loss'] = ms_loss
            combined_loss += self.multiscale_loss_weight * ms_loss
            
        # Store all loss components for logging
        loss_components['combined_loss'] = combined_loss
        
        return combined_loss, loss_components
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        
        # Compute combined loss
        loss, loss_components = self._compute_combined_loss(y_pred, y)
        
        # Log all loss components
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        
        # Log individual loss components
        for name, value in loss_components.items():
            if name != 'combined_loss':  # Already logged as train_loss
                self.log(f'train_{name}', value, prog_bar=False, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        
        # Compute combined loss
        loss, loss_components = self._compute_combined_loss(y_pred, y)
        
        # Log all loss components
        self.log('val_loss', loss, prog_bar=True, on_epoch=True)
        
        # Log individual loss components
        for name, value in loss_components.items():
            if name != 'combined_loss':  # Already logged as val_loss
                self.log(f'val_{name}', value, prog_bar=False, on_epoch=True)
        
        # Maintain same visualization as original model
        log_every = self.hparams['logging'].get('log_val_every_epoch', 1)
        if batch_idx == 0 and (self.current_epoch % log_every == 0):
            self._log_predictions(x, y, y_pred, 'val')
        
        return loss
    
    def _log_predictions(self, x, y, y_pred, prefix='train'):
        """Log visualizations to TensorBoard."""
        # Get a few samples
        num_samples = min(4, x.shape[0])
        
        # 1. Log input 2D signals
        fig, axes = plt.subplots(num_samples, 1, figsize=(10, 2*num_samples))
        if num_samples == 1:
            axes = [axes]
        
        for i in range(num_samples):
            # Handle different possible input shapes correctly
            if len(x.shape) == 4:  # [batch, channel, height, width]
                img_data = x[i, 0].cpu().numpy()
            elif len(x.shape) == 3 and x.shape[1] == self.input_shape[0] and x.shape[2] == self.input_shape[1]:
                # [batch, height, width]
                img_data = x[i].cpu().numpy()
            elif len(x.shape) == 3:
                # [batch, channel=1, height*width flattened]
                img_data = x[i].cpu().numpy().reshape(self.input_shape[0], self.input_shape[1])
            elif len(x.shape) == 2:
                # [batch, height*width flattened]
                img_data = x[i].cpu().numpy().reshape(self.input_shape[0], self.input_shape[1])
            else:
                print(f"Warning: Unexpected input shape: {x.shape}, skipping visualization")
                continue
                
            # 2D signal visualization
            im = axes[i].imshow(img_data, aspect='auto', cmap='viridis')
            axes[i].set_title(f"Input 2D Signal {i+1}")
            plt.colorbar(im, ax=axes[i])
        
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img = Image.open(buf)
        img_tensor = T.ToTensor()(img)
        self.logger.experiment.add_image(f'{prefix}_inputs', img_tensor, self.global_step)
        plt.close(fig)
        
        # 2. Log output 1D signals with comparison
        fig, axes = plt.subplots(num_samples, 1, figsize=(10, 2*num_samples))
        if num_samples == 1:
            axes = [axes]
        
        for i in range(num_samples):
            # Predicted vs Ground Truth
            axes[i].plot(y[i].cpu().numpy(), label='Ground Truth', alpha=0.7)
            axes[i].plot(y_pred[i].detach().cpu().numpy(), label='Prediction', alpha=0.7)
            axes[i].set_title(f"Output 1D Signal {i+1}")
            axes[i].legend()
        
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img = Image.open(buf)
        img_tensor = T.ToTensor()(img)
        self.logger.experiment.add_image(f'{prefix}_outputs_comparison', img_tensor, self.global_step)
        plt.close(fig)