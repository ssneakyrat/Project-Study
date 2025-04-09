import os
import argparse
import yaml
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks import Callback

from model import Signal2DTo1DModel
from dataset import SignalDataModule

# Custom callback to save checkpoints only at specific epoch intervals
class EpochCheckpoint(Callback):
    def __init__(self, dirpath, save_every_n_epochs):
        super().__init__()
        self.dirpath = dirpath
        self.save_every_n_epochs = save_every_n_epochs
        os.makedirs(dirpath, exist_ok=True)
        
    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if (epoch + 1) % self.save_every_n_epochs == 0:
            checkpoint_path = os.path.join(self.dirpath, f"epoch_{epoch+1:04d}.ckpt")
            trainer.save_checkpoint(checkpoint_path)
            print(f"Saved checkpoint at epoch {epoch+1} to {checkpoint_path}")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train 2D to 1D signal conversion model")
    parser.add_argument('--config', type=str, default='model.yaml', help='Path to config file')
    parser.add_argument('--data_file', type=str, help='Path to h5py data file (overrides config)')
    parser.add_argument('--batch_size', type=int, help='Batch size (overrides config)')
    parser.add_argument('--epochs', type=int, help='Number of epochs (overrides config)')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    # Load config file
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override config with command line args if provided
    if args.data_file:
        config['data']['data_file'] = args.data_file
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.epochs:
        config['training']['epochs'] = args.epochs
    
    # Create directories
    os.makedirs(config['logging']['tensorboard_log_dir'], exist_ok=True)
    os.makedirs(config['logging']['save_model_dir'], exist_ok=True)
    
    # Initialize data module
    data_module = SignalDataModule(config)
    
    # Initialize model
    model = Signal2DTo1DModel(config)
    
    # Set up logger
    logger = TensorBoardLogger(
        save_dir=config['logging']['tensorboard_log_dir'],
        name='signal_conversion'
    )
    
    # Set up callbacks
    callbacks = []
    
    # Performance-based checkpointing (best models)
    if config['logging'].get('save_best_models', True):
        checkpoint_callback = ModelCheckpoint(
            dirpath=config['logging']['save_model_dir'],
            filename='{epoch}-{val_loss:.4f}',
            save_top_k=config['logging'].get('save_best_models_count', 3),
            monitor='val_loss',
            mode='min'
        )
        callbacks.append(checkpoint_callback)
    
    # Custom epoch-based checkpointing using our custom callback
    if config['logging'].get('save_every_n_epochs', 0) > 0:
        epoch_checkpoint_callback = EpochCheckpoint(
            dirpath=config['logging']['save_model_dir'],
            save_every_n_epochs=config['logging']['save_every_n_epochs']
        )
        callbacks.append(epoch_checkpoint_callback)

    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks.append(lr_monitor)
    
    # Configure trainer with mixed precision if enabled
    precision = '16-mixed' if config['training']['mixed_precision'] else 32
    
    # Set up trainer
    trainer = Trainer(
        max_epochs=config['training']['epochs'],
        logger=logger,
        callbacks=callbacks,
        gradient_clip_val=config['training']['grad_clip_val'],
        accumulate_grad_batches=config['training']['grad_accumulation_steps'],
        precision=precision,
        devices='auto',
        accelerator='auto',
        strategy='auto'
    )
    
    # Train the model
    if args.resume:
        trainer.fit(model, data_module, ckpt_path=args.resume)
    else:
        trainer.fit(model, data_module)
    
    if config['logging'].get('save_best_models', True):
        print(f"Training complete! Best model saved at: {checkpoint_callback.best_model_path}")
    else:
        print("Training complete!")

if __name__ == "__main__":
    main()