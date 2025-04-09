import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import numpy as np
import io
from PIL import Image
import torchvision.transforms as T

class Signal2DTo1DModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        
        # Extract config parameters
        self.input_shape = config['model']['input_shape']
        self.output_shape = config['model']['output_shape']
        self.encoder_filters = config['model']['encoder_filters']
        self.decoder_filters = config['model']['decoder_filters']
        self.bottleneck_size = config['model']['bottleneck_size']
        self.dropout_rate = config['model']['dropout_rate']
        self.lr = config['training']['learning_rate']
        self.weight_decay = config['training']['weight_decay']
        
        # Build Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, self.encoder_filters[0], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.encoder_filters[0]),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(self.encoder_filters[0], self.encoder_filters[1], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.encoder_filters[1]),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(self.encoder_filters[1], self.encoder_filters[2], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.encoder_filters[2]),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(self.encoder_filters[2], self.encoder_filters[3], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.encoder_filters[3]),
            nn.LeakyReLU(0.2)
        )
        
        # Calculate encoder output dimensions
        self.enc_output_dim = self._get_encoder_output_dim()
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.encoder_filters[3], self.bottleneck_size),
            nn.Dropout(self.dropout_rate),
            nn.LeakyReLU(0.2)
        )
        
        # Decoder
        self.decoder_input = nn.Linear(self.bottleneck_size, 256)  # 16*16*1
        
        # Transpose convolutions
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1, self.decoder_filters[0], kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(self.decoder_filters[0]),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose2d(self.decoder_filters[0], self.decoder_filters[1], kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(self.decoder_filters[1]),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose2d(self.decoder_filters[1], self.decoder_filters[2], kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(self.decoder_filters[2]),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose2d(self.decoder_filters[2], self.decoder_filters[3], kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(self.decoder_filters[3]),
            nn.LeakyReLU(0.2)
        )
        
        # Calculate the flattened size after decoder
        self.decoder_output_size = self._calculate_decoder_output_size()
        
        # More efficient approach to produce final output
        # Instead of a massive fully connected layer, we use a series of smaller ones
        self.final_layers = nn.Sequential(
            nn.AdaptiveAvgPool2d((16, 16)),  # Reduce spatial dimensions
            nn.Flatten(),
            nn.Linear(16 * 16 * self.decoder_filters[3], 4096),
            nn.LeakyReLU(0.2),
            nn.Dropout(self.dropout_rate),
            nn.Linear(4096, self.output_shape[0])
        )
    
    def _get_encoder_output_dim(self):
        # Helper method to calculate the encoder output dimension
        x = torch.zeros(1, 1, self.input_shape[0], self.input_shape[1])
        x = self.encoder(x)
        return x.shape
    
    def _calculate_decoder_output_size(self):
        # Calculate the output size of the decoder
        x = torch.zeros(1, 1, 16, 16)  # Starting size after reshaping
        x = self.decoder(x)
        return x.numel() // x.shape[0]  # Get the flattened size for one sample
    
    def forward(self, x):
        # Ensure input is correct shape (batch_size, channels, height, width)
        if len(x.shape) == 3:
            x = x.unsqueeze(1)  # Add channel dimension if missing
        
        # Encoder
        x = self.encoder(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder pre-processing
        x = self.decoder_input(x)
        x = x.view(-1, 1, 16, 16)  # Reshape for transpose convolutions
        
        # Decoder
        x = self.decoder(x)
        
        # Final layers (adaptive pooling, flatten, and projection to output size)
        x = self.final_layers(x)
        
        return x
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
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
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = F.mse_loss(y_pred, y)
        
        # Log loss
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        
        # Log visualizations every N steps
        #if batch_idx % self.hparams['logging']['log_every_n_steps'] == 0:
        #    self._log_predictions(x, y, y_pred, 'train')
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        val_loss = F.mse_loss(y_pred, y)
        
        # Log validation loss
        self.log('val_loss', val_loss, prog_bar=True, on_epoch=True)
        
        # Log visualizations
        log_every = self.hparams['logging'].get('log_val_every_epoch', 1)  # Default to 1 if not set
        if batch_idx == 0 and ( self.current_epoch % log_every == 0):
            self._log_predictions(x, y, y_pred, 'val')
        
        return val_loss
    
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