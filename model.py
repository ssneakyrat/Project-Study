import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from wavelet import WaveletTransform
from encoder import ComplexEncoder
from decoder import ComplexDecoder
from loss import AudioReconstructionLoss
from metrics import compute_metrics
from complextensor import ComplexTensor

class WaveletAEModel(LightningModule):
    def __init__(
        self, 
        segment_length=16000,
        sample_rate=16000,
        encoder_dims=[64, 128, 256],
        learning_rate=1e-4
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Wavelet scattering transform
        self.wavelet = WaveletTransform(shape=segment_length, J=8, Q=8)
        
        # Get wavelet output shape
        dummy_input = torch.zeros(1, 1, segment_length)
        with torch.no_grad():
            wavelet_output = self.wavelet(dummy_input)
        wavelet_channels = wavelet_output.shape[1]
        
        # Encoder and decoder
        self.encoder = ComplexEncoder(wavelet_channels, encoder_dims)
        self.decoder = ComplexDecoder(encoder_dims[-1], encoder_dims, output_channels=1)
        
        # Loss function
        self.loss_fn = AudioReconstructionLoss(sample_rate)
        
        # For logging
        self.learning_rate = learning_rate
    
    def forward(self, x):
        """
        x: Audio waveform [batch_size, 1, length]
        Returns:
            x_reconstructed: Reconstructed audio waveform [batch_size, 1, length]
        """
        # Apply wavelet transform
        x_wavelet = self.wavelet(x)
        
        # Make complex by splitting channels
        n_channels = x_wavelet.shape[1]
        half_channels = n_channels // 2
        x_complex = ComplexTensor(
            x_wavelet[:, :half_channels], 
            x_wavelet[:, half_channels:]
        )
        
        # Encode
        z, encoder_features = self.encoder(x_complex)
        
        # Decode
        x_reconstructed = self.decoder(z, encoder_features)
        
        return x_reconstructed
    
    def training_step(self, batch, batch_idx):
        x = batch  # [batch_size, 1, length]
        x_reconstructed = self(x)
        
        # Compute loss
        loss, loss_details = self.loss_fn(x_reconstructed, x)
        
        # Log losses
        self.log('train_loss', loss)
        for k, v in loss_details.items():
            self.log(f'train_{k}', v)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch
        x_reconstructed = self(x)
        
        # Compute loss
        loss, loss_details = self.loss_fn(x_reconstructed, x)
        
        # Compute metrics
        metrics = compute_metrics(x_reconstructed, x)
        
        # Log losses and metrics
        self.log('val_loss', loss)
        for k, v in loss_details.items():
            self.log(f'val_{k}', v)
        for k, v in metrics.items():
            self.log(f'val_{k}', v)
        
        # Save audio samples periodically
        if batch_idx == 0:
            # Log audio samples to tensorboard
            sample_rate = self.hparams.sample_rate
            self.logger.experiment.add_audio(
                'original_audio', 
                x[0].cpu(), 
                self.global_step, 
                sample_rate
            )
            self.logger.experiment.add_audio(
                'reconstructed_audio', 
                x_reconstructed[0].cpu(), 
                self.global_step, 
                sample_rate
            )
        
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', factor=0.5, patience=5, verbose=True
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }