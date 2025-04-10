#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import yaml
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from data.dataset import AudioDataModule
from lightning_module import ComplexAudioEncoderDecoder


# Modify the main function in train.py to include the gradient clipping

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
        **config['data']
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
    
    # Set up logger
    logger = TensorBoardLogger(
        save_dir=config['output_dir'],
        name=config['model_name']
    )
    
    # Initialize trainer with gradient clipping
    trainer = pl.Trainer(
        max_epochs=config['training']['max_epochs'],
        accelerator='gpu',
        devices=1,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=logger,
        precision=config['training']['precision'],
        accumulate_grad_batches=config['training'].get('accumulate_grad_batches', 1),
        gradient_clip_val=config['training'].get('gradient_clip_val', None),  # Add gradient clipping
        log_every_n_steps=10
    )
    
    # Train model
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