import torch
import torchaudio
import matplotlib.pyplot as plt
import numpy as np
import os
from collections import defaultdict

from wavelet import WaveletTransform
from encoder import ComplexEncoder
from decoder import ComplexDecoder
from model import WaveletAEModel
from data import AudioDataModule
from complextensor import ComplexTensor
from metrics import compute_metrics

def test_component(name, test_fn):
    """Run a test and log result"""
    results = defaultdict(bool)
    print(f"Testing {name}...")
    try:
        result = test_fn()
        results[name] = result
        print(f"✓ {name} test {'passed' if result else 'failed'}")
    except Exception as e:
        results[name] = False
        print(f"✗ {name} test failed with error: {str(e)}")
    
    return results

def test_data_module():
    """Test the data module generation and loading"""
    data_module = AudioDataModule(batch_size=4, segment_length=8000)
    data_module.prepare_data()
    data_module.setup()
    
    # Check if files were generated
    success = len(data_module.audio_files) > 0
    
    # Test dataloader
    train_loader = data_module.train_dataloader()
    batch = next(iter(train_loader))
    success = success and batch.shape == (4, 1, 8000)
    
    return success

def test_wavelet_transform():
    """Test the wavelet transform"""
    # Create a simple sine wave
    sr = 16000
    t = torch.linspace(0, 1, sr)
    x = torch.sin(2 * np.pi * 440 * t).unsqueeze(0).unsqueeze(0)  # A4 note
    
    # Apply wavelet transform
    wavelet = WaveletTransform(shape=sr, J=6, Q=8)
    Sx = wavelet(x)
    
    # Check output shape and non-zero values
    success = Sx.shape[0] == 1 and Sx.numel() > 0 and torch.sum(Sx != 0).item() > 0
    
    # Visualize (optional)
    plt.figure(figsize=(10, 6))
    plt.imshow(Sx[0].detach().cpu().numpy(), aspect='auto', origin='lower')
    plt.colorbar()
    plt.title('Wavelet Scattering Coefficients')
    plt.savefig('wavelet_output.png')
    plt.close()
    
    return success

def test_complex_encoder():
    """Test the complex encoder"""
    # Create a dummy input
    batch_size = 2
    channels = 16
    length = 1000
    x_real = torch.randn(batch_size, channels, length)
    x_imag = torch.randn(batch_size, channels, length)
    x = ComplexTensor(x_real, x_imag)
    
    # Create encoder
    encoder = ComplexEncoder(channels, hidden_dims=[32, 64, 128])
    
    # Forward pass
    z, features = encoder(x)
    
    # Check output
    success = (
        isinstance(z, ComplexTensor) and 
        z.shape[0] == batch_size and
        z.shape[1] == 128 and
        len(features) == 3
    )
    
    return success

def test_complex_decoder():
    """Test the complex decoder"""
    # Create encoder first to get features
    batch_size = 2
    channels = 16
    length = 1000
    x_real = torch.randn(batch_size, channels, length)
    x_imag = torch.randn(batch_size, channels, length)
    x = ComplexTensor(x_real, x_imag)
    
    encoder = ComplexEncoder(channels, hidden_dims=[32, 64, 128])
    z, features = encoder(x)
    
    # Create decoder and test
    decoder = ComplexDecoder(128, hidden_dims=[128, 64, 32], output_channels=1)
    output = decoder(z, features)
    
    # Check output shape
    success = (
        output.shape[0] == batch_size and
        output.shape[1] == 1 and
        output.ndim == 3
    )
    
    return success

def test_full_model():
    """Test the full model end-to-end"""
    # Create a simple audio input
    sr = 16000
    t = torch.linspace(0, 1, sr)
    x = torch.sin(2 * np.pi * 440 * t).unsqueeze(0).unsqueeze(0)  # A4 note
    
    # Initialize model
    model = WaveletAEModel(segment_length=sr)
    
    # Forward pass
    with torch.no_grad():
        y = model(x)
    
    # Check output shape
    shape_match = y.shape[0] == 1 and y.shape[1] == 1
    
    # Compute metrics
    metrics = compute_metrics(y, x)
    print(f"Model test metrics: {metrics}")
    
    # Basic check - not expecting perfect results without training
    snr_ok = metrics['snr_db'] > -30  # Very basic check
    
    success = shape_match and snr_ok
    
    # Visualize comparison
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(x[0, 0].detach().cpu().numpy())
    plt.title('Original Audio')
    
    plt.subplot(2, 1, 2)
    plt.plot(y[0, 0].detach().cpu().numpy())
    plt.title(f'Reconstructed Audio (SNR: {metrics["snr_db"]:.2f} dB)')
    
    plt.tight_layout()
    plt.savefig('audio_reconstruction.png')
    plt.close()
    
    return success

if __name__ == "__main__":
    # Create output directory for figures
    os.makedirs("test_output", exist_ok=True)
    
    # Run all tests
    results = {}
    
    # Test data module
    results.update(test_component("DataModule", test_data_module))
    
    # Test wavelet transform
    results.update(test_component("WaveletTransform", test_wavelet_transform))
    
    # Test encoder
    results.update(test_component("ComplexEncoder", test_complex_encoder))
    
    # Test decoder
    results.update(test_component("ComplexDecoder", test_complex_decoder))
    
    # Test full model
    results.update(test_component("FullModel", test_full_model))
    
    # Print summary
    print("\nTest Summary:")
    for component, passed in results.items():
        print(f"{component}: {'✓ PASS' if passed else '✗ FAIL'}")
    
    # Overall success
    overall = all(results.values())
    print(f"\nOverall: {'✓ All tests passed' if overall else '✗ Some tests failed'}")