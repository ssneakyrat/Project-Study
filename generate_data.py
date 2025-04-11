import os
import numpy as np
import h5py
import argparse
from tqdm import tqdm
import yaml
from utils.utils import load_config

def generate_audio_data(n_samples, sample_rate, duration):
    """Generate synthetic audio data for testing
    
    Creates sinusoidal signals with harmonics following the equation:
    s(t) = sin(2πft) + 0.5sin(4πft) + 0.25sin(6πft) + noise
    where f is randomly selected between 200-2000 Hz
    """
    data = []
    for i in tqdm(range(n_samples), desc="Generating audio"):
        # Generate a sinusoidal signal with random frequency
        freq = np.random.uniform(200, 2000)
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        
        # Add some harmonics for complexity
        signal = np.sin(2 * np.pi * freq * t)
        signal += 0.5 * np.sin(2 * np.pi * freq * 2 * t)
        signal += 0.25 * np.sin(2 * np.pi * freq * 3 * t)
        
        # Add some noise
        noise = np.random.normal(0, 0.05, len(signal))
        signal += noise
        
        # Normalize
        signal = signal / np.max(np.abs(signal))
        
        data.append(signal)
    return np.array(data)

def save_to_h5(data, output_path):
    """Save audio data to HDF5 file"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('audio', data=data, dtype='float32', compression='gzip')
    
    print(f"Saved {len(data)} audio samples to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Generate synthetic audio data for testing')
    parser.add_argument('--config', default='config/model.yaml', help='Path to config file')
    parser.add_argument('--output_dir', default='data/audio', help='Output directory for HDF5 files')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate training data
    print("Generating training data...")
    train_data = generate_audio_data(
        config.n_training_samples,
        config.sample_rate,
        config.audio_length / config.sample_rate
    )
    save_to_h5(train_data, os.path.join(args.output_dir, 'train.h5'))
    
    # Generate test data
    print("Generating test data...")
    test_data = generate_audio_data(
        config.n_test_samples,
        config.sample_rate,
        config.audio_length / config.sample_rate
    )
    save_to_h5(test_data, os.path.join(args.output_dir, 'test.h5'))
    
    print("Data generation complete!")

if __name__ == "__main__":
    main()