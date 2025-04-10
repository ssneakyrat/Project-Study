#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import h5py
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def generate_sine_sweep(duration, sample_rate):
    """Generate a logarithmic sine sweep"""
    t = np.linspace(0, duration, int(duration * sample_rate))
    f0, f1 = 50, 8000  # Start and end frequencies
    sweep = np.sin(2 * np.pi * f0 * duration / np.log(f1/f0) * (np.exp(t/duration * np.log(f1/f0)) - 1))
    return sweep.astype(np.float32)

def generate_harmonic_series(duration, sample_rate, fundamental=100):
    """Generate harmonic series with decreasing amplitudes"""
    t = np.linspace(0, duration, int(duration * sample_rate))
    signal = np.zeros_like(t, dtype=np.float32)
    
    # Generate 10 harmonics with decreasing amplitude
    for i in range(1, 11):
        amplitude = 1.0 / i
        signal += amplitude * np.sin(2 * np.pi * fundamental * i * t)
    
    # Normalize
    signal /= np.max(np.abs(signal))
    
    return signal

def generate_sample(i, duration, sample_rate):
    """Generate a single audio sample with varied characteristics"""
    # Choose between sine sweep and harmonic series randomly
    if np.random.rand() < 0.5:
        # Generate sine sweep with random amplitude
        amplitude = np.random.uniform(0.5, 1.0)
        audio = amplitude * generate_sine_sweep(duration, sample_rate)
    else:
        # Generate harmonic series with random fundamental
        fundamental = np.random.uniform(80, 400)
        audio = generate_harmonic_series(duration, sample_rate, fundamental)
    
    # Add noise
    noise_level = np.random.uniform(0.001, 0.01)
    noise = np.random.normal(0, noise_level, size=audio.shape)
    audio += noise
    
    # Apply random amplitude modulation
    if np.random.rand() < 0.3:
        mod_freq = np.random.uniform(0.5, 2.0)
        t = np.linspace(0, duration, int(duration * sample_rate))
        mod = 0.5 + 0.5 * np.sin(2 * np.pi * mod_freq * t)
        audio *= mod
    
    # Apply random fade in/out
    if np.random.rand() < 0.5:
        fade_samples = int(sample_rate * np.random.uniform(0.05, 0.2))
        fade_in = np.linspace(0, 1, fade_samples)
        fade_out = np.linspace(1, 0, fade_samples)
        
        audio[:fade_samples] *= fade_in
        audio[-fade_samples:] *= fade_out
    
    # Ensure the audio is in the range [-1, 1]
    audio = np.clip(audio, -1, 1)
    
    return audio

def create_dataset(output_file, num_samples, duration, sample_rate):
    """Create dataset of audio samples and save to H5 file using parallel processing"""
    # Prepare output directory
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)
    
    # Create the dataset dimensions
    audio_samples_shape = (num_samples, int(duration * sample_rate))
    
    # Initialize H5 file
    with h5py.File(output_file, 'w') as f:
        f.create_dataset('audio', shape=audio_samples_shape, dtype=np.float32,
                         chunks=(1, audio_samples_shape[1]), compression='gzip', compression_opts=4)
    
    # Generate audio samples in parallel
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        # Set up a progress bar
        print(f"Generating {num_samples} samples for {output_file}...")
        
        # Submit all tasks
        futures = [executor.submit(generate_sample, i, duration, sample_rate) 
                 for i in range(num_samples)]
        
        # Process results as they complete
        for i, future in tqdm(enumerate(futures), total=num_samples):
            audio = future.result()
            
            # Write to H5 file
            with h5py.File(output_file, 'r+') as f:
                f['audio'][i] = audio
    
    print(f"Created {num_samples} samples in {output_file}")

def main(args):
    print("Generating test data...")
    
    # Create train, validation, and test sets
    create_dataset(
        os.path.join(args.output_dir, 'train.h5'),
        args.train_samples,
        args.duration,
        args.sample_rate
    )
    
    create_dataset(
        os.path.join(args.output_dir, 'val.h5'),
        args.val_samples,
        args.duration,
        args.sample_rate
    )
    
    create_dataset(
        os.path.join(args.output_dir, 'test.h5'),
        args.test_samples,
        args.duration,
        args.sample_rate
    )
    
    print("Done generating test data.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate test audio data')
    parser.add_argument('--output_dir', type=str, default='./data',
                        help='Output directory for H5 files')
    parser.add_argument('--train_samples', type=int, default=1000,
                        help='Number of training samples')
    parser.add_argument('--val_samples', type=int, default=100,
                        help='Number of validation samples')
    parser.add_argument('--test_samples', type=int, default=100,
                        help='Number of test samples')
    parser.add_argument('--duration', type=float, default=2.0,
                        help='Duration of audio in seconds')
    parser.add_argument('--sample_rate', type=int, default=16000,
                        help='Audio sample rate')
    args = parser.parse_args()
    
    main(args)