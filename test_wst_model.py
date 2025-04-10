#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import yaml
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from pathlib import Path
import argparse
import time

# Project imports
from models.wavelet import WaveletScatteringTransform
from models.encoder import ComplexEncoder
from models.decoder import ComplexDecoder
from models.complex_layers import ComplexToReal
from lightning_module import ComplexAudioEncoderDecoder


def check_tensor_stats(tensor, name):
    """Print statistics about a tensor"""
    if isinstance(tensor, tuple):
        real, imag = tensor
        print(f"\n{name} - Complex Tensor Statistics:")
        print(f"  Real part - shape: {real.shape}, min: {real.min().item():.6f}, max: {real.max().item():.6f}, mean: {real.mean().item():.6f}, std: {real.std().item():.6f}")
        print(f"  Imag part - shape: {imag.shape}, min: {imag.min().item():.6f}, max: {imag.max().item():.6f}, mean: {imag.mean().item():.6f}, std: {imag.std().item():.6f}")
        
        # Calculate magnitude and phase for complex tensor
        mag = torch.sqrt(real**2 + imag**2 + 1e-8)
        phase = torch.atan2(imag, real)
        print(f"  Magnitude - min: {mag.min().item():.6f}, max: {mag.max().item():.6f}, mean: {mag.mean().item():.6f}, std: {mag.std().item():.6f}")
        print(f"  Phase - min: {phase.min().item():.6f}, max: {phase.max().item():.6f}, mean: {phase.mean().item():.6f}, std: {phase.std().item():.6f}")
        
        # Check for NaN and Inf values
        has_nan_real = torch.isnan(real).any().item()
        has_inf_real = torch.isinf(real).any().item()
        has_nan_imag = torch.isnan(imag).any().item()
        has_inf_imag = torch.isinf(imag).any().item()
        print(f"  Contains NaN - Real: {has_nan_real}, Imag: {has_nan_imag}")
        print(f"  Contains Inf - Real: {has_inf_real}, Imag: {has_inf_imag}")
        
        # Check sparsity
        real_sparsity = (real.abs() < 1e-6).float().mean().item()
        imag_sparsity = (imag.abs() < 1e-6).float().mean().item()
        print(f"  Sparsity - Real: {real_sparsity:.4f}, Imag: {imag_sparsity:.4f}")
    else:
        print(f"\n{name} - Tensor Statistics:")
        print(f"  Shape: {tensor.shape}, min: {tensor.min().item():.6f}, max: {tensor.max().item():.6f}, mean: {tensor.mean().item():.6f}, std: {tensor.std().item():.6f}")
        
        # Check for NaN and Inf values
        has_nan = torch.isnan(tensor).any().item()
        has_inf = torch.isinf(tensor).any().item()
        print(f"  Contains NaN: {has_nan}")
        print(f"  Contains Inf: {has_inf}")
        
        # Check sparsity 
        sparsity = (tensor.abs() < 1e-6).float().mean().item()
        print(f"  Sparsity: {sparsity:.4f}")


def compute_stft(audio, n_fft=1024, hop_length=256):
    """Compute STFT spectrogram"""
    window = torch.hann_window(n_fft, device=audio.device)
    stft = torch.stft(audio, n_fft=n_fft, hop_length=hop_length, 
                     window=window, return_complex=True)
    mag = torch.log10(torch.abs(stft) + 1e-8)
    return mag


def plot_spectrograms(original, reconstructed, title, output_dir):
    """Plot original and reconstructed spectrograms"""
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # Calculate spectrograms
    orig_stft = compute_stft(original[0])
    recon_stft = compute_stft(reconstructed[0])
    
    # Ensure numpy arrays for plotting
    orig_stft = orig_stft.cpu().numpy()
    recon_stft = recon_stft.cpu().detach().numpy()
    
    # Plot original spectrogram
    im0 = axes[0].imshow(orig_stft, aspect='auto', origin='lower', cmap='viridis')
    axes[0].set_title('Input Spectrogram')
    plt.colorbar(im0, ax=axes[0])
    
    # Plot reconstructed spectrogram
    im1 = axes[1].imshow(recon_stft, aspect='auto', origin='lower', cmap='viridis')
    axes[1].set_title('Reconstructed Spectrogram')
    plt.colorbar(im1, ax=axes[1])
    
    plt.tight_layout()
    plt.savefig(output_dir / f"{title}_spectrograms.png")
    plt.close(fig)


def test_model_components(config_path, output_dir='test_results'):
    """Test each component of the model with various input signals"""
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running tests on {device}")
    
    # Configure parameters from config
    sample_rate = config['data']['sample_rate']
    wavelet_T = config['wavelet']['T']
    
    # Generate test signals
    print("\nGenerating test signals...")
    duration = wavelet_T / sample_rate
    
    # Test signals
    test_signals = {}
    
    # 1. Sine wave at 440 Hz
    t = np.linspace(0, duration, wavelet_T)
    sine_wave = np.sin(2 * np.pi * 440 * t)
    test_signals['sine_wave'] = torch.tensor(sine_wave, dtype=torch.float32).to(device).unsqueeze(0)
    
    # 2. Chirp signal (frequency sweep from 100 Hz to 4000 Hz)
    f0, f1 = 100, 4000
    chirp = np.sin(2 * np.pi * t * (f0 + (f1-f0) * t / duration))
    test_signals['chirp'] = torch.tensor(chirp, dtype=torch.float32).to(device).unsqueeze(0)
    
    # 3. White noise
    noise = np.random.randn(wavelet_T)
    noise = noise / np.max(np.abs(noise))  # Normalize to [-1, 1]
    test_signals['noise'] = torch.tensor(noise, dtype=torch.float32).to(device).unsqueeze(0)
    
    # 4. Harmonic signal (fundamental + overtones)
    harmonic = np.sin(2 * np.pi * 220 * t) + 0.5 * np.sin(2 * np.pi * 440 * t) + 0.25 * np.sin(2 * np.pi * 880 * t)
    harmonic = harmonic / np.max(np.abs(harmonic))  # Normalize
    test_signals['harmonic'] = torch.tensor(harmonic, dtype=torch.float32).to(device).unsqueeze(0)
    
    # Print test signal stats
    for name, signal in test_signals.items():
        print(f"\nTest signal '{name}' - shape: {signal.shape}, min: {signal.min().item():.4f}, max: {signal.max().item():.4f}")
    
    # Initialize WST
    print("\n" + "="*50)
    print("TESTING WAVELET SCATTERING TRANSFORM")
    print("="*50)
    
    wst = WaveletScatteringTransform(
        J=config['wavelet']['J'],
        Q=config['wavelet']['Q'],
        T=config['wavelet']['T'],
        sample_rate=config['data']['sample_rate'],
        normalize=config['wavelet'].get('normalize', True),
        max_order=config['wavelet'].get('max_order', 1),
        ensure_output_dim=config['wavelet'].get('ensure_output_dim', True)
    ).to(device)
    
    # Test WST with each signal
    wst_outputs = {}
    
    for name, signal in test_signals.items():
        print(f"\nApplying WST to {name}...")
        start_time = time.time()
        wst_output = wst(signal)
        end_time = time.time()
        wst_outputs[name] = wst_output
        
        # Check output statistics
        check_tensor_stats(wst_output, f"WST Output - {name}")
        print(f"WST processing time: {end_time - start_time:.4f} seconds")
        
        # Check expected dimensions
        real, imag = wst_output
        print(f"Expected time dimension: {wst.expected_output_time_dim}, Actual: {real.shape[2]}")
        print(f"Expected channels: {wst.expected_channels}, Actual: {real.shape[1]}")
        
        # Save magnitude visualization
        mag = torch.sqrt(real**2 + imag**2 + 1e-8)
        plt.figure(figsize=(10, 6))
        plt.imshow(mag[0].cpu().numpy(), aspect='auto', cmap='viridis')
        plt.colorbar()
        plt.title(f"WST Magnitude - {name}")
        plt.savefig(output_dir / f"wst_mag_{name}.png")
        plt.close()
    
    # Initialize Encoder
    print("\n" + "="*50)
    print("TESTING COMPLEX ENCODER")
    print("="*50)
    
    # Get input channels from WST output
    input_channels = wst_outputs[list(test_signals.keys())[0]][0].shape[1]
    
    encoder = ComplexEncoder(
        input_channels=input_channels,
        channels=config['model']['channels'],
        kernel_sizes=config['model']['kernel_sizes'],
        strides=config['model']['strides'],
        paddings=config['model'].get('paddings', None),
        dropout=config['model']['dropout'],
        use_batch_norm=config['model']['use_batch_norm']
    ).to(device)
    
    # Test encoder
    encoder_outputs = {}
    encoder_intermediates = {}
    
    for name, wst_output in wst_outputs.items():
        print(f"\nEncoding {name}...")
        start_time = time.time()
        encoded, intermediates = encoder(wst_output)
        end_time = time.time()
        encoder_outputs[name] = encoded
        encoder_intermediates[name] = intermediates
        
        # Check output statistics
        check_tensor_stats(encoded, f"Encoder Output - {name}")
        print(f"Encoding time: {end_time - start_time:.4f} seconds")
        
        # Print intermediates info for skip connections
        print(f"Intermediate output shapes (for skip connections):")
        for i, inter in enumerate(intermediates):
            real, imag = inter
            print(f"  Layer {i}: {real.shape}")
    
    # Initialize Decoder
    print("\n" + "="*50)
    print("TESTING COMPLEX DECODER")
    print("="*50)
    
    decoder = ComplexDecoder(
        latent_channels=config['model']['channels'][-1],
        channels=config['model']['channels'],
        kernel_sizes=config['model']['kernel_sizes'],
        strides=config['model']['strides'],
        paddings=config['model'].get('paddings', None),
        output_paddings=config['model'].get('output_paddings', None),
        output_channels=1,
        dropout=config['model']['dropout'],
        use_batch_norm=config['model']['use_batch_norm']
    ).to(device)
    
    # Initialize complex to real conversion
    complex_to_real = ComplexToReal(mode=config['model'].get('complex_repr', 'mag_phase')).to(device)
    
    # Test decoder
    for name in test_signals.keys():
        print(f"\nDecoding {name}...")
        encoded = encoder_outputs[name]
        intermediates = encoder_intermediates[name]
        
        # Forward pass through decoder
        start_time = time.time()
        decoded = decoder(encoded, intermediates)
        
        # Convert to real
        output = complex_to_real(decoded)
        end_time = time.time()
        
        # Check statistics
        check_tensor_stats(decoded, f"Decoder Output (Complex) - {name}")
        check_tensor_stats(output, f"Final Output (Real) - {name}")
        print(f"Decoding time: {end_time - start_time:.4f} seconds")
        
        # Resize if necessary
        if output.shape[-1] != test_signals[name].shape[-1]:
            print(f"Resizing output from {output.shape[-1]} to {test_signals[name].shape[-1]}")
            if output.dim() == 2:  # [B, T]
                output = output.unsqueeze(1)  # Add channel dim [B, 1, T]
                
            output = F.interpolate(
                output, 
                size=test_signals[name].shape[-1],
                mode='linear', 
                align_corners=False
            )
            
            if output.dim() == 3 and output.size(1) == 1:
                output = output.squeeze(1)  # Remove channel dim [B, T]
        
        # Plot original and reconstructed signals
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.plot(test_signals[name][0].cpu().numpy())
        plt.title(f"Original {name}")
        plt.subplot(2, 1, 2)
        plt.plot(output[0].cpu().detach().numpy())
        plt.title(f"Reconstructed {name}")
        plt.tight_layout()
        plt.savefig(output_dir / f"reconstruction_{name}.png")
        plt.close()
        
        # Plot spectrograms
        plot_spectrograms(test_signals[name], output, name, output_dir)
        
        # Calculate reconstruction metrics
        mse = F.mse_loss(output, test_signals[name]).item()
        l1 = F.l1_loss(output, test_signals[name]).item()
        
        # Calculate SNR
        noise = test_signals[name] - output
        signal_power = torch.sum(test_signals[name] ** 2).item()
        noise_power = torch.sum(noise ** 2).item() + 1e-10
        snr = 10 * np.log10(signal_power / noise_power)
        
        print(f"Reconstruction metrics:")
        print(f"  MSE: {mse:.6f}")
        print(f"  L1: {l1:.6f}")
        print(f"  SNR: {snr:.2f} dB")
    
    # Test the full model with end-to-end processing
    print("\n" + "="*50)
    print("TESTING FULL MODEL END-TO-END")
    print("="*50)
    
    # Initialize full model
    model = ComplexAudioEncoderDecoder(config).to(device)
    
    # Set to eval mode
    model.eval()
    
    for name, signal in test_signals.items():
        print(f"\nProcessing {name} with full model...")
        
        # Forward pass
        with torch.no_grad():
            start_time = time.time()
            output = model(signal)
            end_time = time.time()
        
        print(f"Full model processing time: {end_time - start_time:.4f} seconds")
        
        # Check output statistics
        check_tensor_stats(output, f"Full Model Output - {name}")
        
        # Calculate metrics
        mse = F.mse_loss(output, signal).item()
        l1 = F.l1_loss(output, signal).item()
        
        # Calculate SNR
        noise = signal - output
        signal_power = torch.sum(signal ** 2).item()
        noise_power = torch.sum(noise ** 2).item() + 1e-10
        snr = 10 * np.log10(signal_power / noise_power)
        
        print(f"Full model reconstruction metrics:")
        print(f"  MSE: {mse:.6f}")
        print(f"  L1: {l1:.6f}")
        print(f"  SNR: {snr:.2f} dB")
        
        # Plot full model reconstruction spectrograms
        plot_spectrograms(signal, output, f"full_model_{name}", output_dir)
    
    # Test gradient flow
    print("\n" + "="*50)
    print("TESTING GRADIENT FLOW")
    print("="*50)
    
    # Set model to train mode
    model.train()
    
    # Use a single test signal
    test_signal = test_signals['harmonic']
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Reset gradients
    optimizer.zero_grad()
    
    # Forward pass
    output = model(test_signal)
    
    # Compute loss (MSE)
    loss = F.mse_loss(output, test_signal)
    print(f"Initial loss: {loss.item():.6f}")
    
    # Backward pass
    loss.backward()
    
    # Check gradients by component
    print("\nAnalyzing gradients:")
    
    def analyze_gradient_norms(model_component, name):
        grad_norms = []
        for param_name, param in model_component.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_norms.append(grad_norm)
                print(f"  {name}.{param_name}: grad_norm={grad_norm:.6e}")
        
        if grad_norms:
            avg_norm = sum(grad_norms) / len(grad_norms)
            min_norm = min(grad_norms)
            max_norm = max(grad_norms)
            print(f"\n  {name} summary - avg: {avg_norm:.6e}, min: {min_norm:.6e}, max: {max_norm:.6e}")
    
    # Analyze gradient norms by component
    analyze_gradient_norms(model.wst, "WST")
    analyze_gradient_norms(model.encoder, "Encoder")
    analyze_gradient_norms(model.decoder, "Decoder")
    
    print("\nTest completed. Results saved to", output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test WST-based audio autoencoder components")
    parser.add_argument("--config", type=str, default="config/model.yaml", help="Path to config file")
    parser.add_argument("--output", type=str, default="test_results", help="Output directory")
    args = parser.parse_args()
    
    test_model_components(args.config, args.output)