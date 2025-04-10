import torch
import torchaudio
import time
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import pandas as pd
from tabulate import tabulate

from model import WaveletAEModel
from metrics import compute_metrics
from wavelet import WaveletTransform

def generate_test_signals(num_signals=5, sample_rate=16000, duration=2.0):
    """Generate a set of test signals with varying complexity"""
    signals = []
    
    # A4 sine wave (440 Hz)
    t = torch.linspace(0, duration, int(sample_rate * duration))
    sine_wave = torch.sin(2 * np.pi * 440 * t).unsqueeze(0)
    signals.append(("sine_440hz", sine_wave))
    
    # Complex tone (multiple frequencies)
    complex_tone = torch.sin(2 * np.pi * 440 * t) + 0.5 * torch.sin(2 * np.pi * 880 * t) + 0.25 * torch.sin(2 * np.pi * 1320 * t)
    complex_tone = complex_tone.unsqueeze(0) / complex_tone.abs().max()
    signals.append(("complex_tone", complex_tone))
    
    # White noise
    white_noise = torch.randn(int(sample_rate * duration))
    white_noise = white_noise.unsqueeze(0) / white_noise.abs().max() * 0.5
    signals.append(("white_noise", white_noise))
    
    # Chirp (frequency sweep)
    start_freq, end_freq = 100, 8000
    phase = torch.cumsum(torch.linspace(start_freq, end_freq, int(sample_rate * duration)) / sample_rate, dim=0) * 2 * np.pi
    chirp = torch.sin(phase).unsqueeze(0)
    signals.append(("chirp", chirp))
    
    # AM tone (amplitude modulation)
    carrier_freq, mod_freq = 1000, 5
    carrier = torch.sin(2 * np.pi * carrier_freq * t)
    modulator = 0.5 + 0.5 * torch.sin(2 * np.pi * mod_freq * t)
    am_tone = (carrier * modulator).unsqueeze(0)
    signals.append(("am_tone", am_tone))
    
    return signals

def benchmark_model(model, test_signals, output_dir="benchmark_results", device="cpu"):
    """Benchmark model on test signals and record metrics"""
    os.makedirs(output_dir, exist_ok=True)
    
    model = model.to(device)
    model.eval()
    
    # Initialize results dictionary
    results = {
        "signal_name": [],
        "mae": [],
        "rmse": [],
        "snr_db": [],
        "lsd": [],
        "inference_time_ms": []
    }
    
    # Process each test signal
    for name, signal in test_signals:
        signal = signal.unsqueeze(0).to(device)  # Add channel dimension and move to device
        
        # Measure inference time
        start_time = time.time()
        with torch.no_grad():
            reconstructed = model(signal)
        inference_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Compute metrics
        metrics = compute_metrics(reconstructed, signal)
        
        # Save results
        results["signal_name"].append(name)
        results["mae"].append(metrics["mae"])
        results["rmse"].append(metrics["rmse"])
        results["snr_db"].append(metrics["snr_db"])
        results["lsd"].append(metrics["lsd"])
        results["inference_time_ms"].append(inference_time)
        
        # Plot and save comparison
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.plot(signal[0, 0].cpu().numpy())
        plt.title(f'Original: {name}')
        plt.grid(True)
        
        plt.subplot(2, 1, 2)
        plt.plot(reconstructed[0, 0].cpu().numpy())
        plt.title(f'Reconstructed (SNR: {metrics["snr_db"]:.2f} dB)')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{name}_comparison.png"))
        plt.close()
        
        # Save audio files
        torchaudio.save(
            os.path.join(output_dir, f"{name}_original.wav"),
            signal[0].cpu(),
            sample_rate=16000
        )
        torchaudio.save(
            os.path.join(output_dir, f"{name}_reconstructed.wav"),
            reconstructed[0].cpu(),
            sample_rate=16000
        )
    
    # Create summary tables
    df = pd.DataFrame(results)
    
    # Print tabular results
    print("\nBenchmark Results:")
    print(tabulate(df, headers='keys', tablefmt='grid', showindex=False))
    
    # Save results to CSV
    df.to_csv(os.path.join(output_dir, "benchmark_results.csv"), index=False)
    
    # Plot metrics comparison
    plt.figure(figsize=(14, 10))
    
    # Plot SNR
    plt.subplot(2, 2, 1)
    plt.bar(df["signal_name"], df["snr_db"])
    plt.title("Signal-to-Noise Ratio (dB)")
    plt.xticks(rotation=45)
    plt.grid(True)
    
    # Plot MAE
    plt.subplot(2, 2, 2)
    plt.bar(df["signal_name"], df["mae"])
    plt.title("Mean Absolute Error")
    plt.xticks(rotation=45)
    plt.grid(True)
    
    # Plot LSD
    plt.subplot(2, 2, 3)
    plt.bar(df["signal_name"], df["lsd"])
    plt.title("Log-Spectral Distance")
    plt.xticks(rotation=45)
    plt.grid(True)
    
    # Plot inference time
    plt.subplot(2, 2, 4)
    plt.bar(df["signal_name"], df["inference_time_ms"])
    plt.title("Inference Time (ms)")
    plt.xticks(rotation=45)
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "metrics_comparison.png"))
    plt.close()
    
    return df

def benchmark_memory_usage(model, segment_lengths=[8000, 16000, 32000, 64000], device="cpu"):
    """Benchmark model memory usage with different input lengths"""
    results = {
        "segment_length": [],
        "gpu_memory_mb": [],
        "parameters": []
    }
    
    # Count model parameters
    model_size = sum(p.numel() for p in model.parameters())
    
    # Test with different segment lengths
    for length in segment_lengths:
        # Generate dummy input
        dummy_input = torch.randn(1, 1, length).to(device)
        
        # Clear cache if using GPU
        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        # Forward pass
        with torch.no_grad():
            _ = model(dummy_input)
        
        # Measure memory usage if on GPU
        if device == "cuda":
            memory_used = torch.cuda.max_memory_allocated() / (1024 ** 2)  # Convert to MB
        else:
            memory_used = float('nan')  # Not available for CPU
        
        # Add to results
        results["segment_length"].append(length)
        results["gpu_memory_mb"].append(memory_used)
        results["parameters"].append(model_size)
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Print results
    print("\nMemory Usage Benchmark:")
    print(tabulate(df, headers='keys', tablefmt='grid', showindex=False))
    
    # Plot memory usage vs segment length
    if device == "cuda":
        plt.figure(figsize=(10, 6))
        plt.plot(df["segment_length"], df["gpu_memory_mb"], marker='o')
        plt.title("GPU Memory Usage vs Input Length")
        plt.xlabel("Segment Length (samples)")
        plt.ylabel("GPU Memory (MB)")
        plt.grid(True)
        plt.savefig("memory_benchmark.png")
        plt.close()
    
    return df

def compare_with_mel_vocoder(wavelet_model, segment_length=16000, sample_rate=16000, device="cpu"):
    """
    Compare wavelet model with traditional mel-spectrogram approach
    Note: This is just a simple comparison without implementing a full mel vocoder
    """
    # Generate test signal
    t = torch.linspace(0, segment_length/sample_rate, segment_length)
    test_signal = torch.sin(2 * np.pi * 440 * t).unsqueeze(0).unsqueeze(0).to(device)
    
    # Wavelet transform process
    wavelet_start = time.time()
    wavelet = WaveletTransform(shape=segment_length, J=8, Q=8).to(device)
    wavelet_coeffs = wavelet(test_signal)
    wavelet_time = time.time() - wavelet_start
    
    # Mel-spectrogram process
    mel_start = time.time()
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=1024,
        hop_length=256,
        n_mels=80
    ).to(device)
    mel_spec = mel_transform(test_signal.squeeze(1))
    mel_time = time.time() - mel_start
    
    # Print results
    print("\nTransform Time Comparison:")
    print(f"Wavelet Transform: {wavelet_time*1000:.2f} ms")
    print(f"Mel-Spectrogram:   {mel_time*1000:.2f} ms")
    print(f"Speedup factor:    {mel_time/wavelet_time:.2f}x")
    
    # Get compression ratios
    wavelet_ratio = test_signal.numel() / wavelet_coeffs.numel()
    mel_ratio = test_signal.numel() / mel_spec.numel()
    
    print("\nCompression Ratio Comparison:")
    print(f"Wavelet Transform: {wavelet_ratio:.2f}:1")
    print(f"Mel-Spectrogram:   {mel_ratio:.2f}:1")
    
    # Create comparison plot
    plt.figure(figsize=(12, 8))
    
    # Plot original signal
    plt.subplot(3, 1, 1)
    plt.plot(test_signal[0, 0].cpu().numpy())
    plt.title("Original Audio Signal")
    plt.grid(True)
    
    # Plot wavelet coefficients
    plt.subplot(3, 1, 2)
    plt.imshow(wavelet_coeffs[0].cpu().numpy(), aspect='auto', origin='lower')
    plt.title(f"Wavelet Scattering Coefficients (Compression: {wavelet_ratio:.2f}:1)")
    plt.colorbar()
    
    # Plot mel-spectrogram
    plt.subplot(3, 1, 3)
    plt.imshow(mel_spec[0].cpu().numpy(), aspect='auto', origin='lower')
    plt.title(f"Mel-Spectrogram (Compression: {mel_ratio:.2f}:1)")
    plt.colorbar()
    
    plt.tight_layout()
    plt.savefig("transform_comparison.png")
    plt.close()
    
    # Return comparison metrics
    return {
        "wavelet_time_ms": wavelet_time * 1000,
        "mel_time_ms": mel_time * 1000,
        "wavelet_compression_ratio": wavelet_ratio,
        "mel_compression_ratio": mel_ratio
    }

if __name__ == "__main__":
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Initialize model
    model = WaveletAEModel(segment_length=16000)
    
    # Generate test signals
    test_signals = generate_test_signals()
    
    # Run benchmarks
    results = benchmark_model(model, test_signals, device=device)
    
    # Memory benchmark
    memory_results = benchmark_memory_usage(model, device=device)
    
    # Compare with mel-spectrogram
    comparison = compare_with_mel_vocoder(model, device=device)