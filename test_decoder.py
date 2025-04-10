import os
import torch
import yaml
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import Dataset, DataLoader
from models.wavelet_encoder import WaveletEncoder
from models.latent_processor import LatentProcessor
from models.wavelet_decoder import WaveletDecoder

class DummyDataset(Dataset):
    """Generate dummy sine waves for testing the decoder"""
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

def test_decoder_in_isolation():
    """Test the decoder in isolation"""
    # Load configuration
    with open("config/model.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Initialize decoder
    decoder = WaveletDecoder(config)
    decoder.eval()
    
    # Create dummy latent vectors
    batch_size = 8
    z = torch.randn(batch_size, config['model']['bottleneck_channels'])
    
    # Test decoding
    with torch.no_grad():
        x_hat = decoder(z)
    
    # Print results
    print("=== Decoder Test ===")
    print(f"Input latent shape: {z.shape}")
    print(f"Output audio shape: {x_hat.shape}")
    
    # Verify output dimensions match expected dimensions
    expected_output_shape = (batch_size, 1, config['model']['input_size'])
    assert x_hat.shape == expected_output_shape, f"Expected {expected_output_shape}, got {x_hat.shape}"
    
    print("Decoder test passed!")
    return decoder

def test_encoder_decoder_pipeline():
    """Test the full encoder-decoder pipeline"""
    # Load configuration
    with open("config/model.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Initialize models
    encoder = WaveletEncoder(config)
    decoder = WaveletDecoder(config)
    
    # Set to evaluation mode
    encoder.eval()
    decoder.eval()
    
    # Create dummy audio input
    batch_size = 8
    audio = torch.randn(batch_size, 1, config['model']['input_size'])
    
    # Process through encoder and decoder
    with torch.no_grad():
        # Get latent representation from encoder
        z = encoder(audio)
        
        # Decode back to audio
        audio_reconstructed = decoder(z)
    
    # Print results
    print("\n=== Encoder-Decoder Pipeline Test ===")
    print(f"Original audio shape: {audio.shape}")
    print(f"Latent representation shape: {z.shape}")
    print(f"Reconstructed audio shape: {audio_reconstructed.shape}")
    
    # Compute reconstruction error
    mse = torch.nn.functional.mse_loss(audio, audio_reconstructed).item()
    print(f"Reconstruction MSE: {mse:.6f}")
    
    return encoder, decoder

def test_full_pipeline_with_latent_processor():
    """Test the full encoder-latent_processor-decoder pipeline"""
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
    decoder = WaveletDecoder(config)
    
    # Set to evaluation mode
    encoder.eval()
    processor.eval()
    decoder.eval()
    
    # Create dummy audio input and condition
    batch_size = 8
    audio = torch.randn(batch_size, 1, config['model']['input_size'])
    condition = torch.nn.functional.one_hot(
        torch.randint(0, 10, (batch_size,)), 
        num_classes=10
    ).float()
    
    # Process through the full pipeline
    with torch.no_grad():
        # Get latent representation from encoder
        z = encoder(audio)
        
        # Process latent with condition
        z_conditioned = processor(z, condition)
        
        # Decode back to audio
        audio_reconstructed = decoder(z_conditioned)
    
    # Print results
    print("\n=== Full Pipeline Test with Latent Processor ===")
    print(f"Original audio shape: {audio.shape}")
    print(f"Latent representation shape: {z.shape}")
    print(f"Conditioned latent shape: {z_conditioned.shape}")
    print(f"Reconstructed audio shape: {audio_reconstructed.shape}")
    
    # Verify mathematical properties of the wavelet transform
    print(f"Testing wavelet transform properties:")
    
    # 1. Check if reconstruction preserves energy
    orig_energy = torch.mean(torch.sum(audio**2, dim=2))
    recon_energy = torch.mean(torch.sum(audio_reconstructed**2, dim=2))
    energy_ratio = recon_energy / orig_energy
    print(f"Energy preservation ratio: {energy_ratio:.4f} (should be close to 1.0)")
    
    # 2. Check effect of different conditions on output
    # Create a different condition
    condition2 = torch.nn.functional.one_hot(
        (torch.randint(0, 10, (batch_size,))), 
        num_classes=10
    ).float()
    
    # Process with different condition
    z_conditioned2 = processor(z, condition2)
    audio_reconstructed2 = decoder(z_conditioned2)
    
    # Calculate difference between outputs with different conditions
    output_diff = torch.mean(torch.norm(audio_reconstructed - audio_reconstructed2, dim=2))
    print(f"Mean difference between outputs with different conditions: {output_diff:.4f}")
    
    return encoder, processor, decoder

def test_wavelet_properties():
    """Test mathematical properties of the wavelet transform"""
    # Load configuration
    with open("config/model.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Initialize decoder with InverseWaveletTransform
    decoder = WaveletDecoder(config)
    
    # Create test signal (sine wave)
    freq = 440  # Hz
    duration = 1.0  # seconds
    sample_rate = config['model']['input_size'] // duration
    t = np.linspace(0, duration, int(sample_rate * duration))
    test_signal = np.sin(2 * np.pi * freq * t)
    test_signal = torch.from_numpy(test_signal).float().unsqueeze(0).unsqueeze(0)
    
    # Create random latent vector
    z = torch.randn(1, config['model']['bottleneck_channels'])
    
    # Decode to get output signal
    with torch.no_grad():
        output_signal = decoder(z)
    
    print("\n=== Wavelet Transform Mathematical Properties ===")
    print(f"Test signal shape: {test_signal.shape}")
    print(f"Output signal shape: {output_signal.shape}")
    
    # Basic spectral analysis
    from scipy import signal as sp_signal
    import matplotlib.pyplot as plt
    
    # Convert torch tensors to numpy for analysis
    output_np = output_signal.squeeze().numpy()
    
    # Compute power spectral density
    f, Pxx = sp_signal.welch(output_np, fs=sample_rate, nperseg=1024)
    
    print(f"Computed spectral density with {len(f)} frequency bins")
    print(f"Peak frequency components: {f[np.argsort(Pxx)[-5:]][::-1]} Hz")
    
    # Theoretical property: Frequency localization in wavelets
    print("The wavelet transform provides frequency localization, with each channel")
    print("responding to different frequency bands in the signal.")

def main():
    """Main function to run all tests"""
    print("Starting Decoder tests...")
    
    # Test decoder in isolation
    decoder = test_decoder_in_isolation()
    
    # Test encoder-decoder pipeline
    encoder, decoder = test_encoder_decoder_pipeline()
    
    # Test full pipeline with latent processor
    encoder, processor, decoder = test_full_pipeline_with_latent_processor()
    
    # Test wavelet mathematical properties
    test_wavelet_properties()
    
    # Test training with lightning
    print("\n=== Training Test ===")
    # Create datasets
    train_dataset = DummyDataset(num_samples=100)
    val_dataset = DummyDataset(num_samples=20)
    
    # Create dataloaders
    with open("config/model.yaml", "r") as f:
        config = yaml.safe_load(f)
    
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
    model = WaveletDecoder(config)
    
    # Create logger
    logger = TensorBoardLogger(
        save_dir=config["logging"]["tensorboard_dir"],
        name="wavelet_decoder"
    )
    
    # Create trainer with limited epochs for testing
    trainer = pl.Trainer(
        max_epochs=2,  # Just run 2 epochs for testing
        logger=logger,
        log_every_n_steps=config["logging"]["log_every_n_steps"],
        accumulate_grad_batches=config["training"]["gradient_accumulation"]
    )
    
    # Train model (this is just a test, so we'll run a very short training)
    print("Running 2 epochs for testing...")
    trainer.fit(model, train_loader, val_loader)
    print("Training test completed.")
    
    print("\nAll tests completed!")

if __name__ == "__main__":
    main()