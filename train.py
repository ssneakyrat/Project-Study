import os
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping

from utils.utils import load_config, print_gpu_stats, save_config
from data.dataset import get_dataloaders
from lightning_model import WaveletAudioAE

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Train Optimized Wavelet Echo Matrix (WEM) model')
    parser.add_argument('--config', default='config/model.yaml', help='Path to config file')
    parser.add_argument('--output_dir', default='outputs_optimized', help='Output directory')
    parser.add_argument('--precision', default='16-mixed', choices=['32', '16-mixed'], 
                        help='Use mixed precision training for better performance')
    parser.add_argument('--gradient_clip_val', type=float, default=1.0, 
                        help='Gradient clipping value to prevent exploding gradients')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Learning rate conversion
    if isinstance(config.learning_rate, str):
        config.learning_rate = float(config.learning_rate)
    
    # Set random seed
    pl.seed_everything(42)
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get dataloaders
    train_loader, test_loader = get_dataloaders(config)
    
    # Initialize optimized model
    model = WaveletAudioAE(config)
    
    # Initialize logger
    logger = TensorBoardLogger('logs', name='wavelet_ae_optimized')
    
    # Initialize callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.output_dir, 'checkpoints'),
        filename='wavelet_ae-{epoch:02d}-{val_loss:.4f}',
        monitor='val_loss',
        save_top_k=3,
        mode='min'
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=50,  # Stop if no improvement for 50 epochs
        mode='min',
        verbose=True
    )
    
    # Initialize trainer with mixed precision and gradient checkpointing
    trainer = pl.Trainer(
        max_epochs=config.epochs,
        logger=logger,
        log_every_n_steps=1,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        precision=args.precision,  # Use mixed precision for better performance
        gradient_clip_val=args.gradient_clip_val,  # Prevent exploding gradients
        check_val_every_n_epoch=5,  # Validate less frequently to speed up training
        callbacks=[checkpoint_callback, lr_monitor, early_stopping]
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
    model_path = os.path.join(args.output_dir, 'wavelet_ae_optimized_model.pt')
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    # Save config
    config_path = os.path.join(args.output_dir, 'wavelet_ae_optimized_config.yaml')
    save_config(config, config_path)
    print(f"Configuration saved to {config_path}")
    
    # Evaluate model on test data
    print("\nEvaluating model on test data...")
    trainer.test(model, test_loader)
    
    print("\nTraining completed successfully!")

if __name__ == "__main__":
    main()