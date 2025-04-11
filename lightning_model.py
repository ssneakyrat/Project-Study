import torch
import pytorch_lightning as pl
from models.wem import WaveletEchoMatrix
from models.loss import CombinedLoss
from models.metric import MultiMetric

class WEMLightningModel(pl.LightningModule):
    def __init__(self, config):
        super(WEMLightningModel, self).__init__()
        self.save_hyperparameters(config)
        self.config = config
        
        # Initialize model
        self.model = WaveletEchoMatrix(
            wavelet=config['wavelet'],
            levels=config['levels'],
            hidden_dim=config['hidden_dim'],
            latent_dim=config['latent_dim']
        )
        
        # Initialize loss function
        self.loss_fn = CombinedLoss(
            wavelet_weight=config['wavelet_loss_weight'],
            time_weight=config['time_loss_weight'],
            levels=config['levels'],
            alpha=config['loss_alpha']
        )
        
        # Initialize metrics
        self.metrics = MultiMetric()
        
    def forward(self, x):
        return self.model(x)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
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
    
    def training_step(self, batch, batch_idx):
        x = batch
        output = self(x)
        
        loss = self.loss_fn(
            output['reconstructed'],
            x,
            output['rec_coeffs'],
            output['coeffs']
        )
        
        # Log training metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        if batch_idx % 100 == 0:  # Log detailed metrics periodically
            metrics = self.metrics(output['reconstructed'], x)
            for name, value in metrics.items():
                self.log(f'train_{name}', value, on_step=True, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch
        output = self(x)
        
        loss = self.loss_fn(
            output['reconstructed'],
            x,
            output['rec_coeffs'],
            output['coeffs']
        )
        
        # Log validation metrics
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        
        metrics = self.metrics(output['reconstructed'], x)
        for name, value in metrics.items():
            self.log(f'val_{name}', value, on_epoch=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        x = batch
        output = self(x)
        
        loss = self.loss_fn(
            output['reconstructed'],
            x,
            output['rec_coeffs'],
            output['coeffs']
        )
        
        # Log test metrics
        self.log('test_loss', loss, on_epoch=True)
        
        metrics = self.metrics(output['reconstructed'], x)
        for name, value in metrics.items():
            self.log(f'test_{name}', value, on_epoch=True)
        
        return loss