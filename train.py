import os
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from utils.utils import load_config, print_gpu_stats, save_config
from data.dataset import get_dataloaders
from lightning_model import WaveletAudioAE

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Train Wavelet Echo Matrix (WEM) model')
    parser.add_argument('--config', default='config/model.yaml', help='Path to config file')
    parser.add_argument('--output_dir', default='outputs', help='Output directory')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Learning rate convertion
    if isinstance(config.learning_rate, str):
        config.learning_rate = float(config.learning_rate)
    
    # Set random seed
    pl.seed_everything(42)
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get dataloaders
    train_loader, test_loader = get_dataloaders(config)
    
    # Initialize model
    model = WaveletAudioAE(config)
    
    # Initialize logger
    logger = TensorBoardLogger('logs', name='wavelet_ae')
    
    # Initialize checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.output_dir, 'checkpoints'),
        filename='wavelet_ae-{epoch:02d}-{val_loss:.4f}',
        monitor='val_loss',
        save_top_k=3,
        mode='min'
    )
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=config.epochs,
        logger=logger,
        log_every_n_steps=1,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        check_val_every_n_epoch=10,
        callbacks=[checkpoint_callback]
    )
    
    # Track GPU memory usage
    print("Memory usage before training:")
    print_gpu_stats()
    
    # Train model
    print("Starting training...")
    trainer.fit(model, train_loader, test_loader)
    
    # Track GPU memory after training
    print("Memory usage after training:")
    print_gpu_stats()
    
    # Save model and configuration
    model_path = os.path.join(args.output_dir, 'wavelet_ae_model.pt')
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    # Save config
    config_path = os.path.join(args.output_dir, 'wavelet_ae_config.yaml')
    save_config(config, config_path)
    print(f"Configuration saved to {config_path}")
    
    # Evaluate model on test data
    print("\nEvaluating model on test data...")
    trainer.test(model, test_loader)
    
    print("\nTraining completed successfully!")

if __name__ == "__main__":
    main()