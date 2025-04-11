import os
import numpy as np
import h5py
import argparse
from tqdm import tqdm
import yaml
from utils.utils import load_config

def generate_audio_data(n_samples, sample_rate, audio_length):
    """Generate synthetic audio data for testing
    
    Creates sinusoidal signals with harmonics following the equation:
    s(t) = sin(2πft) + 0.5sin(4πft) + 0.25sin(6πft) + noise
    where f is randomly selected between 200-2000 Hz
    
    Args:
        n_samples: Number of audio samples to generate
        sample_rate: Sampling rate in Hz
        audio_length: Number of samples per audio clip
    """
    duration = audio_length / sample_rate
    data = []
    for i in tqdm(range(n_samples), desc="Generating audio"):
        # Generate a sinusoidal signal with random frequency
        freq = np.random.uniform(200, 2000)
        t = np.linspace(0, duration, audio_length, endpoint=False)
        
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

def generate_continuous_audio(n_samples, sample_rate, duration, evolve_rate=0.2):
    """Generate continuous audio with evolving characteristics
    
    Creates audio with gradually changing frequency, harmonics, and amplitude
    modulations to test the model's ability to handle transitions.
    
    Args:
        n_samples: Number of audio samples to generate
        sample_rate: Sampling rate in Hz
        duration: Duration in seconds for each continuous clip
        evolve_rate: Rate at which audio characteristics evolve (0-1)
    
    Returns:
        audio_data: Array of shape [n_samples, audio_length]
        markers: Dictionary with window boundary markers for each sample
    """
    audio_length = int(duration * sample_rate)
    data = []
    markers = {}
    
    for i in tqdm(range(n_samples), desc="Generating continuous audio"):
        # Create time array
        t = np.linspace(0, duration, audio_length, endpoint=False)
        
        # Generate varying characteristics
        base_freq = np.random.uniform(300, 1200)
        
        # Create frequency sweep or evolution
        evolution_type = np.random.choice(["sweep", "step", "stable"])
        
        if evolution_type == "sweep":
            # Smooth frequency sweep
            end_freq = np.random.uniform(300, 1200)
            freq_evolution = base_freq + (end_freq - base_freq) * (np.sin(2 * np.pi * evolve_rate * t / duration) + 1) / 2
        
        elif evolution_type == "step":
            # Step changes at specific points (good for testing boundaries)
            freq_evolution = np.ones_like(t) * base_freq
            change_points = np.random.choice(range(1, int(duration)), size=min(3, int(duration)), replace=False)
            
            # Record boundary markers for evaluation
            markers[i] = {"change_points": (change_points * sample_rate).tolist()}
            
            for cp in change_points:
                cp_idx = int(cp * sample_rate)
                freq_evolution[cp_idx:] = np.random.uniform(300, 1200)
        
        else:  # stable
            # Constant frequency
            freq_evolution = np.ones_like(t) * base_freq
            
        # Generate signal with evolving harmonics
        signal = np.sin(2 * np.pi * np.cumsum(freq_evolution / sample_rate))
        
        # Add harmonics with varying strength
        harmonic_strength = 0.5 + 0.3 * np.sin(2 * np.pi * evolve_rate * 0.5 * t)
        signal += harmonic_strength * np.sin(4 * np.pi * np.cumsum(freq_evolution / sample_rate))
        
        # Add third harmonic in some sections
        third_harm_sections = np.random.random(size=int(duration)) > 0.5
        third_harm = np.zeros_like(t)
        for s_idx, active in enumerate(third_harm_sections):
            if active:
                start_idx = s_idx * sample_rate
                end_idx = min((s_idx + 1) * sample_rate, audio_length)
                third_harm[start_idx:end_idx] = 0.25
        
        signal += third_harm * np.sin(6 * np.pi * np.cumsum(freq_evolution / sample_rate))
        
        # Add amplitude modulation
        if np.random.random() > 0.5:
            am_freq = np.random.uniform(0.2, 1.0)
            am = 0.7 + 0.3 * np.sin(2 * np.pi * am_freq * t)
            signal *= am
        
        # Add some noise
        noise_level = np.random.uniform(0.01, 0.05)
        noise = np.random.normal(0, noise_level, len(signal))
        signal += noise
        
        # Normalize
        signal = signal / np.max(np.abs(signal))
        
        data.append(signal)
    
    return np.array(data), markers

def generate_transition_test_audio(sample_rate, window_size, hop_size, n_samples=10):
    """Generate test audio specifically for window transitions
    
    Creates audio that has deliberate transitions at window boundaries
    to test the effectiveness of the windowing approach.
    
    Args:
        sample_rate: Sampling rate in Hz
        window_size: Size of processing window in samples
        hop_size: Hop size between windows in samples
        n_samples: Number of test samples to generate
    
    Returns:
        audio_data: Array of transition test audio
        markers: Dictionary with window boundary information
    """
    # Calculate duration to include multiple windows
    n_windows = 5
    audio_length = window_size + (n_windows - 1) * hop_size
    duration = audio_length / sample_rate
    
    data = []
    markers = {}
    
    for i in range(n_samples):
        # Create time array
        t = np.linspace(0, duration, audio_length, endpoint=False)
        
        # Generate signal with transitions at window boundaries
        signal = np.zeros_like(t)
        
        # Create markers for window boundaries
        window_boundaries = []
        
        for w in range(n_windows):
            # Calculate window boundary
            boundary = w * hop_size
            window_boundaries.append(boundary)
            
            # Define segment for this window
            start_idx = boundary
            end_idx = min(start_idx + window_size, audio_length)
            
            # Generate different content for each window
            window_freq = 300 + w * 200  # Increasing frequency per window
            
            # Phase alignment to avoid discontinuities
            if w > 0:
                # Calculate phase to match at boundary
                prev_freq = 300 + (w-1) * 200
                # Get phase at the end of previous window
                prev_phase = 2 * np.pi * prev_freq * (start_idx / sample_rate)
                phase_offset = prev_phase % (2 * np.pi)
            else:
                phase_offset = 0
                
            # Generate the signal for this window segment
            segment_t = t[start_idx:end_idx] - (start_idx / sample_rate)
            segment = np.sin(2 * np.pi * window_freq * segment_t + phase_offset)
            
            # Apply a fade-in/fade-out within the window
            fade_len = min(int(0.1 * window_size), end_idx - start_idx)
            fade_in = np.linspace(0, 1, fade_len)
            fade_out = np.linspace(1, 0, fade_len)
            
            envelope = np.ones(end_idx - start_idx)
            envelope[:fade_len] = fade_in
            envelope[-fade_len:] = fade_out
            
            segment = segment * envelope
            
            # Add to the full signal
            signal[start_idx:end_idx] += segment
        
        # Add some noise
        noise = np.random.normal(0, 0.02, len(signal))
        signal += noise
        
        # Normalize
        signal = signal / np.max(np.abs(signal))
        
        # Store boundary markers
        markers[i] = {"window_boundaries": window_boundaries}
        
        data.append(signal)
    
    return np.array(data), markers

def save_to_h5(data, output_path, markers=None):
    """Save audio data to HDF5 file
    
    Args:
        data: Audio data array
        output_path: Path to save HDF5 file
        markers: Optional dictionary of markers/metadata to save
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('audio', data=data, dtype='float32', compression='gzip')
        
        # Save markers if provided
        if markers:
            markers_group = f.create_group('markers')
            for sample_idx, marker_dict in markers.items():
                sample_group = markers_group.create_group(str(sample_idx))
                for key, value in marker_dict.items():
                    sample_group.create_dataset(key, data=value)
    
    print(f"Saved {len(data)} audio samples to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Generate synthetic audio data for testing')
    parser.add_argument('--config', default='config/model.yaml', help='Path to config file')
    parser.add_argument('--output_dir', default='data/audio', help='Output directory for HDF5 files')
    parser.add_argument('--continuous', action='store_true', help='Generate additional continuous audio datasets')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate training data (original short segments)
    print("Generating training data...")
    print(f"Audio settings: {config.audio_length} samples at {config.sample_rate} Hz")
    print(f"Duration per clip: {config.audio_length/config.sample_rate:.2f} seconds")
    
    train_data = generate_audio_data(
        config.n_training_samples,
        config.sample_rate,
        config.audio_length
    )
    save_to_h5(train_data, os.path.join(args.output_dir, 'train.h5'))
    
    # Generate test data (original short segments)
    print("Generating test data...")
    test_data = generate_audio_data(
        config.n_test_samples,
        config.sample_rate,
        config.audio_length
    )
    save_to_h5(test_data, os.path.join(args.output_dir, 'test.h5'))
    
    # Generate continuous audio datasets if requested
    if args.continuous:
        print("\nGenerating continuous audio datasets...")
        
        # Medium duration (5 seconds)
        medium_duration = 5.0
        print(f"Generating medium continuous dataset ({medium_duration}s clips)...")
        medium_data, medium_markers = generate_continuous_audio(
            n_samples=20,
            sample_rate=config.sample_rate,
            duration=medium_duration,
            evolve_rate=0.3
        )
        save_to_h5(
            medium_data, 
            os.path.join(args.output_dir, 'continuous_medium.h5'),
            medium_markers
        )
        
        # Long duration (15 seconds)
        long_duration = 15.0
        print(f"Generating long continuous dataset ({long_duration}s clips)...")
        long_data, long_markers = generate_continuous_audio(
            n_samples=10,
            sample_rate=config.sample_rate,
            duration=long_duration,
            evolve_rate=0.2
        )
        save_to_h5(
            long_data, 
            os.path.join(args.output_dir, 'continuous_long.h5'),
            long_markers
        )
        
        # Transition test dataset
        print("Generating transition test dataset...")
        # Assume a window size equal to current audio_length
        window_size = config.audio_length
        # Use 75% overlap for testing
        hop_size = window_size // 4
        
        transition_data, transition_markers = generate_transition_test_audio(
            sample_rate=config.sample_rate,
            window_size=window_size,
            hop_size=hop_size,
            n_samples=15
        )
        save_to_h5(
            transition_data,
            os.path.join(args.output_dir, 'transition_test.h5'),
            transition_markers
        )
        
        print("Continuous audio datasets generated successfully!")
    
    print("Data generation complete!")

if __name__ == "__main__":
    main()