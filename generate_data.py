import numpy as np
import h5py
import argparse
import os

def generate_synthetic_data(num_samples, input_shape, output_shape):
    """
    Generate synthetic 2D to 1D signal data for testing.
    
    Args:
        num_samples (int): Number of samples to generate
        input_shape (tuple): Shape of 2D input signal (x, y)
        output_shape (int): Length of 1D output signal
    
    Returns:
        tuple: (input_2d, output_1d) arrays
    """
    input_2d = np.zeros((num_samples, input_shape[0], input_shape[1]), dtype=np.float32)
    output_1d = np.zeros((num_samples, output_shape), dtype=np.float32)
    
    # Generate synthetic data
    for i in range(num_samples):
        # Generate random 2D signal (e.g., a mixture of Gaussians)
        x = np.linspace(-5, 5, input_shape[0])
        y = np.linspace(-5, 5, input_shape[1])
        xx, yy = np.meshgrid(x, y)
        num_peaks = np.random.randint(1, 5)
        
        signal_2d = np.zeros(input_shape[::-1])  # Note: meshgrid gives transposed coordinates
        
        for _ in range(num_peaks):
            # Random center and width for Gaussian
            x0 = np.random.uniform(-4, 4)
            y0 = np.random.uniform(-4, 4)
            sigma_x = np.random.uniform(0.5, 2.0)
            sigma_y = np.random.uniform(0.5, 2.0)
            amplitude = np.random.uniform(0.5, 1.0)
            
            # Generate 2D Gaussian
            gauss = amplitude * np.exp(-(
                (xx - x0)**2 / (2 * sigma_x**2) + 
                (yy - y0)**2 / (2 * sigma_y**2)
            ))
            
            signal_2d += gauss
        
        # Transpose to match expected dimensions (height, width)
        signal_2d = signal_2d.T
        
        # Normalize 2D signal to [0, 1]
        signal_2d = (signal_2d - signal_2d.min()) / (signal_2d.max() - signal_2d.min() + 1e-8)
        
        # Generate corresponding 1D signal (e.g., some transformation of the 2D signal)
        # For simplicity, we'll create a 1D signal that has some relation to the 2D one
        x_1d = np.linspace(-5, 5, output_shape)
        
        # Create a mixture of sinusoids with frequencies related to the 2D signal
        freqs = [np.random.uniform(0.1, 1.0) for _ in range(num_peaks)]
        phases = [np.random.uniform(0, 2*np.pi) for _ in range(num_peaks)]
        
        signal_1d = np.zeros(output_shape)
        for freq, phase in zip(freqs, phases):
            signal_1d += np.sin(2 * np.pi * freq * x_1d + phase)
        
        # Add some noise
        signal_1d += np.random.normal(0, 0.1, output_shape)
        
        # Normalize 1D signal to [0, 1]
        signal_1d = (signal_1d - signal_1d.min()) / (signal_1d.max() - signal_1d.min() + 1e-8)
        
        # Store the signals
        input_2d[i] = signal_2d
        output_1d[i] = signal_1d
    
    return input_2d, output_1d

def create_h5_dataset(file_path, num_samples, input_shape=(862, 80), output_shape=22050):
    """
    Create an h5py file with synthetic 2D to 1D signal data.
    
    Args:
        file_path (str): Path to save the h5py file
        num_samples (int): Number of samples to generate
        input_shape (tuple): Shape of 2D input signal (x, y)
        output_shape (int): Length of 1D output signal
    """
    # Generate synthetic data
    print(f"Generating {num_samples} synthetic data samples...")
    input_2d, output_1d = generate_synthetic_data(num_samples, input_shape, output_shape)
    
    # Create h5py file and save the data
    print(f"Saving data to {file_path}...")
    with h5py.File(file_path, 'w') as f:
        f.create_dataset('input_2d', data=input_2d, compression='gzip', compression_opts=9)
        f.create_dataset('output_1d', data=output_1d, compression='gzip', compression_opts=9)
    
    print("Done!")
    print(f"Data shapes: input_2d {input_2d.shape}, output_1d {output_1d.shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic 2D to 1D signal data")
    parser.add_argument('--output', type=str, default='signal_data.h5', help='Output h5py file path')
    parser.add_argument('--samples', type=int, default=1000, help='Number of samples to generate')
    parser.add_argument('--input_x', type=int, default=862, help='X dimension of input 2D signal')
    parser.add_argument('--input_y', type=int, default=80, help='Y dimension of input 2D signal')
    parser.add_argument('--output_len', type=int, default=22050, help='Length of output 1D signal')
    args = parser.parse_args()
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    
    # Generate and save the data
    create_h5_dataset(
        args.output, 
        args.samples, 
        input_shape=(args.input_x, args.input_y), 
        output_shape=args.output_len
    )