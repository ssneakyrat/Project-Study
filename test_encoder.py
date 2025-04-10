import os
import torch
import yaml
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import Dataset, DataLoader
from models.wavelet_encoder import WaveletEncoder

class DummyDataset(Dataset):
    """Generate dummy sine waves for testing the encoder"""
    def __init__(self, num_samples=1000, sample_length=16000):
        self.num_samples = num_samples
        self.sample_length = sample_length
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Generate random frequency between 100-4000 Hz
        freq = np.random.uniform(100, 4000)
        
        # Generate time points
        t = np.linspace(0, 1, self.sample_length)
        
        # Create sine wave with random frequency
        wave = np.sin(2 * np.pi * freq * t)
        
        # Add some noise
        noise = np.random.normal(0, 0.01, self.sample_length)
        wave += noise
        
        # Convert to torch tensor and reshape to [1, T]
        return torch.from_numpy(wave).float().unsqueeze(0)

def main():
    # Load configuration
    with open("config/model.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Create tensorboard directory if it doesn't exist
    os.makedirs(config["logging"]["tensorboard_dir"], exist_ok=True)
    
    # Create datasets
    train_dataset = DummyDataset(num_samples=1000)
    val_dataset = DummyDataset(num_samples=200)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"],
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"],
        persistent_workers=True
    )
    
    # Initialize model
    model = WaveletEncoder(config)
    
    # Create logger
    logger = TensorBoardLogger(
        save_dir=config["logging"]["tensorboard_dir"],
        name="wavelet_encoder"
    )
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=config["training"]["max_epochs"],
        logger=logger,
        log_every_n_steps=config["logging"]["log_every_n_steps"],
        accumulate_grad_batches=config["training"]["gradient_accumulation"],
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                monitor="val_loss",
                dirpath=config["logging"]["tensorboard_dir"]+"/checkpoints",
                filename="wavelet-encoder-{epoch:02d}-{val_loss:.4f}",
                save_top_k=3,
                mode="min"
            )
        ]
    )
    
    # Train model
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    main()