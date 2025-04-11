import os
import torch
import yaml
import argparse
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from data.dataset import AudioDataModule
from lightning_model import AdaptiveWaveletNetwork

def main(args):
    """
    Run end-to-end training test for the Adaptive Wavelet Network
    """
    print("Starting Adaptive Wavelet Network end-to-end test...")
    
    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    # Create directories if they don't exist
    os.makedirs(config["logging"]["tensorboard_dir"], exist_ok=True)
    os.makedirs(config["data"]["train_path"], exist_ok=True)
    os.makedirs(config["data"]["val_path"], exist_ok=True)
    
    # Add wavelet loss weightings to config
    if 'training' not in config:
        config['training'] = {}
    config['training']['mse_weight'] = args.mse_weight
    config['training']['wavelet_weight'] = args.wavelet_weight
    config['training']['kl_weight'] = args.kl_weight
    
    # Configure data module with conditioning if requested
    config['data']['use_conditioning'] = args.use_conditioning
    config['data']['num_conditions'] = args.num_conditions
    
    # Initialize data module
    data_module = AudioDataModule(config)
    
    # Initialize model
    model = AdaptiveWaveletNetwork(config)
    
    # Create logger
    logger = TensorBoardLogger(
        save_dir=config["logging"]["tensorboard_dir"],
        name="wavelet_network"
    )
    
    # Create callbacks
    callbacks = [
        ModelCheckpoint(
            monitor="val_loss",
            dirpath=os.path.join(config["logging"]["tensorboard_dir"], "checkpoints"),
            filename="wavelet-{epoch:02d}-{val_loss:.4f}",
            save_top_k=3,
            mode="min"
        ),
        LearningRateMonitor(logging_interval="epoch")
    ]
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        logger=logger,
        log_every_n_steps=config["logging"]["log_every_n_steps"],
        accumulate_grad_batches=config["training"].get("gradient_accumulation", 1),
        callbacks=callbacks,
    )
    
    # Run training
    print(f"Starting training for {args.epochs} epochs...")
    trainer.fit(model, data_module)
    
    # Save final model
    final_model_path = os.path.join(
        config["logging"]["tensorboard_dir"], 
        "final_model.ckpt"
    )
    trainer.save_checkpoint(final_model_path)
    print(f"Model saved to {final_model_path}")
    
    print("End-to-end test completed successfully!")
    
    return model, trainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Adaptive Wavelet Network")
    parser.add_argument("--config", type=str, default="config/model.yaml", 
                        help="Path to config file")
    parser.add_argument("--epochs", type=int, default=10, 
                        help="Number of epochs to train")
    parser.add_argument("--use-conditioning", action="store_true", 
                        help="Use conditioning in the model")
    parser.add_argument("--num-conditions", type=int, default=10, 
                        help="Number of condition classes")
    parser.add_argument("--mse-weight", type=float, default=1.0, 
                        help="Weight for MSE loss")
    parser.add_argument("--wavelet-weight", type=float, default=0.5, 
                        help="Weight for wavelet domain loss")
    parser.add_argument("--kl-weight", type=float, default=0.5, 
                        help="Weight for KL divergence loss")
    
    args = parser.parse_args()
    main(args)