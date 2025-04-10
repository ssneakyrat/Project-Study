import os
import torch
import yaml
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import Dataset, DataLoader
from models.wavelet_encoder import WaveletEncoder
from models.latent_processor import LatentProcessor

class ConditionedAudioDataset(Dataset):
    """Generate dummy sine waves with conditions for testing"""
    def __init__(self, num_samples=1000, sample_length=16000, num_conditions=10):
        self.num_samples = num_samples
        self.sample_length = sample_length
        self.num_conditions = num_conditions
    
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
        
        # Create a one-hot condition vector (e.g., representing different classes)
        condition_idx = idx % self.num_conditions
        condition = torch.zeros(self.num_conditions)
        condition[condition_idx] = 1.0
        
        # Convert wave to torch tensor and reshape to [1, T]
        return torch.from_numpy(wave).float().unsqueeze(0), condition

def test_latent_processor():
    """Test the latent processor with different conditioning methods"""
    # Load configuration
    with open("config/model.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Update config with latent processing specific parameters
    config['model']['condition_dim'] = 10  # For one-hot encoded conditions
    config['model']['use_vq'] = True
    config['model']['num_embeddings'] = 512
    
    # Initialize latent processor
    processor = LatentProcessor(config)
    processor.eval()
    
    # Create dummy latent vectors and conditions
    batch_size = 10  # Match to number of available conditions
    z = torch.randn(batch_size, config['model']['bottleneck_channels'])
    condition = torch.eye(10)  # 10x10 one-hot identity matrix
    
    # Test different conditioning methods
    with torch.no_grad():
        # 1. Standard conditioning: z' = z + MLP(condition)
        z_conditioned = processor(z, condition)
        
        # 2. No conditioning
        z_no_condition = processor(z)
        
        # 3. Direct latent manipulation: z' = z + c (simplified version)
        condition_resized = torch.nn.functional.pad(
            condition, (0, config['model']['bottleneck_channels'] - condition.size(1))
        )
        z_direct = z + condition_resized
    
    # Print results
    print("=== Latent Processor Test ===")
    print(f"Input latent shape: {z.shape}")
    print(f"Condition shape: {condition.shape}")
    print(f"Conditioned latent shape: {z_conditioned.shape}")
    print(f"Latent without condition: {z_no_condition.shape}")
    print(f"Direct manipulation latent: {z_direct.shape}")
    
    return processor

def test_encoder_with_processor():
    """Test the encoder + latent processor integration"""
    # Load configuration
    with open("config/model.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Update config for conditioning
    config['model']['condition_dim'] = 10
    config['model']['use_vq'] = True
    config['model']['num_embeddings'] = 512
    
    # Initialize models
    encoder = WaveletEncoder(config)
    processor = LatentProcessor(config)
    
    # Set to evaluation mode
    encoder.eval()
    processor.eval()
    
    # Create dummy audio input and condition
    batch_size = 8
    audio = torch.randn(batch_size, 1, config['model']['input_size'])
    # Generate batch_size random one-hot vectors
    condition = torch.nn.functional.one_hot(
        torch.randint(0, 10, (batch_size,)), 
        num_classes=10
    ).float()
    
    # Process through encoder and latent processor
    with torch.no_grad():
        # Get latent representation from encoder
        z = encoder(audio)
        
        # Process latent with condition
        z_conditioned = processor(z, condition)
    
    # Print results
    print("\n=== Encoder + Processor Integration Test ===")
    print(f"Audio input shape: {audio.shape}")
    print(f"Encoder output shape: {z.shape}")
    print(f"Conditioned latent shape: {z_conditioned.shape}")
    
    # Verify mathematical properties
    # 1. Conditioning should modify the latent space
    diff = torch.norm(z - z_conditioned).item()
    print(f"L2 distance between z and z_conditioned: {diff:.4f}")
    
    # 2. Different conditions should lead to different latent representations
    # Create a different condition by shifting class indices
    condition2 = torch.nn.functional.one_hot(
        (torch.argmax(condition, dim=1) + 1) % 10, 
        num_classes=10
    ).float()
    z_conditioned2 = processor(z, condition2)
    cond_diff = torch.norm(z_conditioned - z_conditioned2).item()
    print(f"L2 distance between different conditions: {cond_diff:.4f}")

def main():
    """Main function to run all tests"""
    # Test latent processor in isolation
    processor = test_latent_processor()
    
    # Test integration with encoder
    test_encoder_with_processor()
    
    # Further testing with actual training
    print("\n=== Setup Training for Latent Processor ===")
    
    # Load configuration
    with open("config/model.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Update config
    config['model']['condition_dim'] = 10
    config['model']['use_vq'] = True
    config['model']['num_embeddings'] = 512
    
    # Create datasets
    train_dataset = ConditionedAudioDataset(num_samples=100)
    val_dataset = ConditionedAudioDataset(num_samples=20)
    
    # Create dataloaders with smaller batch size for testing
    train_loader = DataLoader(
        train_dataset, 
        batch_size=4,
        shuffle=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0
    )
    
    # Load a batch for testing
    audio, condition = next(iter(train_loader))
    print(f"Audio batch shape: {audio.shape}")
    print(f"Condition batch shape: {condition.shape}")
    
    print("\nLatent processing component testing complete.")

if __name__ == "__main__":
    main()