import os
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from utils.utils import load_config, set_seed, count_parameters, create_directories
from data.dataset import AudioDataModule
from lightning_model import WEMLightningModel

def main(args):
    # Load configuration
    config = load_config(args.config)
    
    # Set random seed for reproducibility
    set_seed(config['seed'])
    
    # Create directories
    create_directories([
        args.output_dir,
        os.path.join(args.output_dir, 'checkpoints'),
        os.path.join(args.output_dir, 'logs')
    ])
    
    # Create data module
    data_module = AudioDataModule(
        data_dir=args.data_dir,
        batch_size=config['training']['batch_size'],
        sample_rate=config['data']['sample_rate'],
        duration=config['data']['duration'],
        num_workers=config['data']['num_workers']
    )
    
    # Create model
    model_config = {
        'wavelet': config['model']['wavelet'],
        'levels': config['model']['levels'],
        'hidden_dim': config['model']['hidden_dim'],
        'latent_dim': config['model']['latent_dim'],
        'learning_rate': config['training']['learning_rate'],
        'weight_decay': config['training']['weight_decay'],
        'wavelet_loss_weight': config['loss']['wavelet_loss_weight'],
        'time_loss_weight': config['loss']['time_loss_weight'],
        'loss_alpha': config['loss']['loss_alpha']
    }
    
    model = WEMLightningModel(model_config)
    
    # Print model information
    print(f"Model: {model.__class__.__name__}")
    print(f"Total parameters: {count_parameters(model):,}")
    
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.output_dir, 'checkpoints'),
        filename='wem-{epoch:02d}-{val_loss:.4f}',
        monitor='val_loss',
        mode='min',
        save_top_k=3,
        save_last=True
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=config['training']['early_stopping_patience'],
        mode='min'
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    # Setup logger
    logger = TensorBoardLogger(
        save_dir=os.path.join(args.output_dir, 'logs'),
        name='wem'
    )
    
    # Setup trainer
    trainer = pl.Trainer(
        max_epochs=config['training']['max_epochs'],
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
        logger=logger,
        log_every_n_steps=config['logging']['log_every_n_steps'],
        gradient_clip_val=config['training']['gradient_clip_val'],
        precision=config['precision']
    )
    
    # Train model
    trainer.fit(model, data_module)
    
    # Test model
    trainer.test(model, data_module)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train WaveletEchoMatrix model")
    parser.add_argument("--config", type=str, default="config/model.yaml", help="Path to config file")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory containing datasets")
    parser.add_argument("--output_dir", type=str, default="output", help="Output directory")
    
    args = parser.parse_args()
    
    main(args)