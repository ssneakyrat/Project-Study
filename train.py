#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import yaml
import argparse
import copy
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from data.dataset import AudioDataModule
from lightning_module import ComplexAudioEncoderDecoder
from utils.ema import ModelEMA


class EMACallback(pl.Callback):
    """
    PyTorch Lightning callback for Exponential Moving Average (EMA)
    """
    def __init__(self, decay=0.999):
        super().__init__()
        self.decay = decay
        self.ema = None
    
    def on_fit_start(self, trainer, pl_module):
        # Initialize EMA model at the start of training
        self.ema = ModelEMA(pl_module, decay=self.decay, device=None)
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Update EMA parameters after each training batch
        if self.ema is not None:
            self.ema.update(pl_module)
    
    def on_validation_start(self, trainer, pl_module):
        # Store original weights before validation
        if self.ema is not None:
            self.stashed_weights = copy.deepcopy(pl_module.state_dict())
            # Replace model weights with EMA weights for validation
            pl_module.load_state_dict(self.ema.ema.state_dict())
    
    def on_validation_end(self, trainer, pl_module):
        # Restore original weights after validation
        if self.ema is not None and hasattr(self, 'stashed_weights'):
            pl_module.load_state_dict(self.stashed_weights)
            del self.stashed_weights
    
    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        # Save EMA state in checkpoint
        if self.ema is not None:
            checkpoint['ema_state_dict'] = self.ema.ema.state_dict()
    
    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        # Load EMA state from checkpoint
        if 'ema_state_dict' in checkpoint and self.ema is not None:
            self.ema.ema.load_state_dict(checkpoint['ema_state_dict'])


def main(args):
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set seed for reproducibility
    pl.seed_everything(config['seed'])
    
    # Create output directory
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Initialize data module
    data_module = AudioDataModule(
        data_dir=args.data_dir,
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['num_workers'],
        sample_rate=config['data']['sample_rate'],
        audio_length=config['data'].get('audio_length', 2.0),
        cache_size=config['data'].get('cache_size', 100)
    )
    
    # Initialize model
    model = ComplexAudioEncoderDecoder(config)
    
    # Set up callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(config['output_dir'], 'checkpoints'),
        filename='{epoch}-{val_loss:.4f}',
        monitor='val_loss',
        mode='min',
        save_top_k=3
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    # Add EMA callback
    ema_callback = EMACallback(decay=0.999)
    
    # Set up logger
    logger = TensorBoardLogger(
        save_dir=config['output_dir'],
        name=config['model_name']
    )
    
    # Initialize trainer with gradient clipping and accumulation
    trainer = pl.Trainer(
        max_epochs=config['training']['max_epochs'],
        accelerator='gpu',
        devices=1,
        callbacks=[checkpoint_callback, lr_monitor, ema_callback],
        logger=logger,
        precision=config['training']['precision'],
        accumulate_grad_batches=config['training'].get('accumulate_grad_batches', 1),
        gradient_clip_val=config['training'].get('gradient_clip_val', None),
        log_every_n_steps=10
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming training from checkpoint: {args.resume}")
        # Train model
        trainer.fit(model, data_module, ckpt_path=args.resume)
    else:
        # Train model from scratch
        trainer.fit(model, data_module)
    
    # Test model
    trainer.test(model, data_module)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train audio encoder-decoder model')
    parser.add_argument('--config', type=str, default='config/model.yaml', 
                        help='Path to configuration YAML file')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing audio data')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint for resuming training')
    
    args = parser.parse_args()
    
    main(args)