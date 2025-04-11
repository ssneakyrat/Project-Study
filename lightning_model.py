import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import os

from models.wavelet_encoder import WaveletEncoder, AdaptiveWaveletTransform
from models.latent_processor import LatentProcessor
from models.wavelet_decoder import WaveletDecoder
from utils.loss import WaveletLoss
from utils.utils import plot_comparison, compute_metrics, figure_to_tensor, plot_spectrogram_comparison

class AdaptiveWaveletNetwork(pl.LightningModule):
    """
    PyTorch Lightning module for the adaptive wavelet network
    """
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        
        # Initialize encoder
        self.encoder = WaveletEncoder(config)
        
        # Initialize latent processor
        self.processor = LatentProcessor(config)
        
        # Initialize decoder
        self.decoder = WaveletDecoder(config)
        
        # Initialize wavelet transform for loss computation
        self.wavelet_transform = AdaptiveWaveletTransform(
            wavelet_type=config['model']['wavelet_type'],
            channels=config['model']['wavelet_channels'],
            input_size=config['model']['input_size']
        )
        
        # Initialize loss function
        self.loss_fn = WaveletLoss(
            wavelet_transform=self.wavelet_transform,
            mse_weight=config.get('training', {}).get('mse_weight', 1.0),
            wavelet_weight=config.get('training', {}).get('wavelet_weight', 1.0),
            kl_weight=config.get('training', {}).get('kl_weight', 0.1)
        )
        
        # Flag for using conditioning
        self.use_conditioning = config.get('data', {}).get('use_conditioning', False)
    
    def forward(self, x, condition=None):
        """
        Forward pass through the full model
        
        Args:
            x: Input audio [B, 1, T]
            condition: Optional conditioning tensor
            
        Returns:
            Reconstructed audio
        """
        # Encode input to latent space
        z = self.encoder(x)
        
        # Process latent with optional conditioning
        z_processed = self.processor(z, condition)
        
        # Decode back to audio
        x_hat = self.decoder(z_processed)
        
        return x_hat, z
    
    def _get_batch_input(self, batch):
        """Parse batch based on conditioning"""
        if self.use_conditioning:
            x, condition = batch
        else:
            x = batch
            condition = None
        return x, condition
    
    def training_step(self, batch, batch_idx):
        """Training step"""
        # Get input
        x, condition = self._get_batch_input(batch)
        
        # Forward pass
        x_hat, z = self(x, condition)
        
        # Compute loss
        loss, loss_components = self.loss_fn(x, x_hat, z)
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        for name, value in loss_components.items():
            self.log(f'train_{name}', value, on_step=False, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step with visual logging"""
        # Get input
        x, condition = self._get_batch_input(batch)
        
        # Forward pass
        x_hat, z = self(x, condition)
        
        # Compute loss
        loss, loss_components = self.loss_fn(x, x_hat, z)
        
        # Log metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        for name, value in loss_components.items():
            self.log(f'val_{name}', value, on_step=False, on_epoch=True)
        
        # Compute audio quality metrics for first sample
        metrics = compute_metrics(x[0], x_hat[0])
        for name, value in metrics.items():
            self.log(f'val_{name}', value, on_step=False, on_epoch=True)
        
        # Log visual comparison for first batch only
        if batch_idx == 0:
            # Create waveform comparison plot for the first sample
            waveform_fig = plot_comparison(
                x[0], 
                x_hat[0], 
                sample_rate=self.config['data']['sample_rate']
            )
            
            # Create spectrogram comparison plot for the first sample
            spectrogram_fig = plot_spectrogram_comparison(
                x[0], 
                x_hat[0], 
                sample_rate=self.config['data']['sample_rate']
            )
            
            # Convert figures to tensors and log to tensorboard
            if self.logger and hasattr(self.logger, 'experiment'):
                waveform_img_tensor = figure_to_tensor(waveform_fig)
                spectrogram_img_tensor = figure_to_tensor(spectrogram_fig)
                
                self.logger.experiment.add_image(
                    'validation_waveform_comparison', 
                    waveform_img_tensor, 
                    self.current_epoch
                )
                
                self.logger.experiment.add_image(
                    'validation_spectrogram_comparison', 
                    spectrogram_img_tensor, 
                    self.current_epoch
                )
        
        return loss
    
    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers"""
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.config['training']['learning_rate']
        )
        
        # Optional learning rate scheduler
        if 'lr_scheduler' in self.config.get('training', {}):
            scheduler_config = self.config['training']['lr_scheduler']
            if scheduler_config['type'] == 'step':
                scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer,
                    step_size=scheduler_config.get('step_size', 10),
                    gamma=scheduler_config.get('gamma', 0.5)
                )
            elif scheduler_config['type'] == 'cosine':
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=scheduler_config.get('T_max', 10)
                )
            else:
                return optimizer
            
            return {
                'optimizer': optimizer,
                'lr_scheduler': scheduler,
                'monitor': 'val_loss'
            }
        
        return optimizer