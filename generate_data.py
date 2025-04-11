import os
import numpy as np
import h5py
import argparse
from tqdm import tqdm

def generate_sine_wave(freq, sample_rate, duration):
    """
    Generate sine wave
    Args:
        freq: Frequency in Hz
        sample_rate: Sample rate in Hz
        duration: Duration in seconds
    Returns:
        audio: Sine wave
    """
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio = np.sin(2 * np.pi * freq * t)
    return audio

def generate_white_noise(sample_rate, duration):
    """
    Generate white noise
    Args:
        sample_rate: Sample rate in Hz
        duration: Duration in seconds
    Returns:
        audio: White noise
    """
    return np.random.normal(0, 0.1, int(sample_rate * duration))

def generate_impulse(sample_rate, duration):
    """
    Generate impulse
    Args:
        sample_rate: Sample rate in Hz
        duration: Duration in seconds
    Returns:
        audio: Impulse
    """
    audio = np.zeros(int(sample_rate * duration))
    impulse_pos = np.random.randint(0, len(audio))
    audio[impulse_pos] = 1.0
    return audio

def generate_chirp(sample_rate, duration, f0=100, f1=5000):
    """
    Generate chirp signal
    Args:
        sample_rate: Sample rate in Hz
        duration: Duration in seconds
        f0: Start frequency in Hz
        f1: End frequency in Hz
    Returns:
        audio: Chirp signal
    """
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    # Logarithmic chirp
    audio = np.sin(2 * np.pi * f0 * duration / np.log(f1/f0) * (np.exp(t/duration * np.log(f1/f0)) - 1))
    return audio

def generate_complex_signal(sample_rate, duration):
    """
    Generate complex signal with multiple components
    Args:
        sample_rate: Sample rate in Hz
        duration: Duration in seconds
    Returns:
        audio: Complex signal
    """
    # Generate random frequencies
    freqs = np.random.randint(50, 2000, size=5)
    amplitudes = np.random.uniform(0.1, 1.0, size=5)
    
    # Generate time vector
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    # Generate composite signal
    audio = np.zeros_like(t)
    for freq, amp in zip(freqs, amplitudes):
        audio += amp * np.sin(2 * np.pi * freq * t)
    
    # Add some noise
    audio += 0.05 * np.random.normal(0, 1, len(audio))
    
    # Normalize
    audio = audio / np.max(np.abs(audio))
    
    return audio

def create_dataset(output_dir, num_samples, sample_rate=16000, duration=2.0, split=(0.8, 0.1, 0.1)):
    """
    Create dataset of synthetic audio signals
    Args:
        output_dir: Output directory
        num_samples: Number of samples to generate
        sample_rate: Sample rate in Hz
        duration: Duration in seconds
        split: Train/val/test split
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate split
    num_train = int(num_samples * split[0])
    num_val = int(num_samples * split[1])
    num_test = num_samples - num_train - num_val
    
    # Generate data
    all_audio = []
    print("Generating audio samples...")
    for _ in tqdm(range(num_samples)):
        # Randomly select generation method
        method = np.random.choice([
            'sine', 'noise', 'impulse', 'chirp', 'complex'
        ])
        
        if method == 'sine':
            freq = np.random.uniform(50, 5000)
            audio = generate_sine_wave(freq, sample_rate, duration)
        elif method == 'noise':
            audio = generate_white_noise(sample_rate, duration)
        elif method == 'impulse':
            audio = generate_impulse(sample_rate, duration)
        elif method == 'chirp':
            f0 = np.random.uniform(50, 500)
            f1 = np.random.uniform(1000, 8000)
            audio = generate_chirp(sample_rate, duration, f0, f1)
        elif method == 'complex':
            audio = generate_complex_signal(sample_rate, duration)
        
        # Normalize
        audio = audio / np.max(np.abs(audio))
        
        all_audio.append(audio)
    
    # Shuffle data
    np.random.shuffle(all_audio)
    
    # Split data
    train_audio = all_audio[:num_train]
    val_audio = all_audio[num_train:num_train+num_val]
    test_audio = all_audio[num_train+num_val:]
    
    # Save to H5 files
    print("Saving train data...")
    with h5py.File(os.path.join(output_dir, 'train.h5'), 'w') as f:
        f.create_dataset('audio', data=np.array(train_audio))
    
    print("Saving validation data...")
    with h5py.File(os.path.join(output_dir, 'val.h5'), 'w') as f:
        f.create_dataset('audio', data=np.array(val_audio))
    
    print("Saving test data...")
    with h5py.File(os.path.join(output_dir, 'test.h5'), 'w') as f:
        f.create_dataset('audio', data=np.array(test_audio))
    
    print(f"Dataset created successfully: {num_train} train, {num_val} validation, {num_test} test samples")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic audio dataset")
    parser.add_argument("--output_dir", type=str, default="data", help="Output directory")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of samples to generate")
    parser.add_argument("--sample_rate", type=int, default=16000, help="Sample rate in Hz")
    parser.add_argument("--duration", type=float, default=2.0, help="Duration in seconds")
    
    args = parser.parse_args()
    
    create_dataset(
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        sample_rate=args.sample_rate,
        duration=args.duration
    )