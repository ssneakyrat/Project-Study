import os
import argparse
import torch
import h5py
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import soundfile as sf  # For audio file loading

from utils.utils import load_config, calculate_snr, compute_spectrogram
from lightning_model import WaveletAudioAE

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Test WaveletEchoMatrix model on audio data')
    parser.add_argument('--config', default='config/model.yaml', help='Path to config file')
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    parser.add_argument('--dataset', default=None, help='Path to test dataset (h5 file)')
    parser.add_argument('--audio_file', default=None, help='Path to individual audio file')
    parser.add_argument('--output_dir', default='results/inference', help='Output directory')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for processing')
    args = parser.parse_args()
    
    # Ensure either dataset or audio file is provided
    if args.dataset is None and args.audio_file is None:
        raise ValueError("Either --dataset or --audio_file must be provided")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and args.gpu else 'cpu')
    print(f"Using device: {device}")
    
    # Load config
    config = load_config(args.config)
    
    # Load model
    print("Loading model from checkpoint:", args.checkpoint)
    model = WaveletAudioAE.load_from_checkpoint(args.checkpoint, config=config)
    model.eval()
    model.to(device)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process dataset or individual file
    if args.dataset:
        process_dataset(args.dataset, model, config, device, args.output_dir)
    
    if args.audio_file:
        process_audio_file(args.audio_file, model, config, device, args.output_dir)
    
    print("Inference completed!")

def process_dataset(dataset_path, model, config, device, output_dir):
    """Process a dataset of audio samples"""
    print(f"Processing dataset: {dataset_path}")
    
    # Create dataset output directory
    dataset_dir = os.path.join(output_dir, os.path.basename(dataset_path).split('.')[0])
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Load dataset
    with h5py.File(dataset_path, 'r') as f:
        audio_data = torch.tensor(f['audio'][:], dtype=torch.float32)
    
    print(f"Loaded {len(audio_data)} audio samples of length {audio_data.shape[1]}")
    
    # Process samples (limit to first 10 to avoid excessive output)
    max_samples = min(10, len(audio_data))
    
    all_snr_values = []
    
    for i in tqdm(range(max_samples), desc="Processing samples"):
        # Get the sample
        audio = audio_data[i:i+1].to(device)
        
        # Process through model
        with torch.no_grad():
            reconstructed, wavelet_coeffs, wavelet_coeffs_recon, latent = model(audio)
        
        # Calculate SNR
        snr = calculate_snr(audio, reconstructed)[0].item()
        all_snr_values.append(snr)
        
        # Create sample output directory
        sample_dir = os.path.join(dataset_dir, f"sample_{i}")
        os.makedirs(sample_dir, exist_ok=True)
        
        # Visualize and save results
        visualize_results(
            audio[0].cpu(),
            reconstructed[0].cpu(),
            wavelet_coeffs[0].cpu(),
            wavelet_coeffs_recon[0].cpu(),
            latent[0].cpu(),
            snr,
            config.sample_rate,
            sample_dir
        )
    
    # Calculate and print average SNR
    avg_snr = sum(all_snr_values) / len(all_snr_values)
    print(f"Average SNR across {len(all_snr_values)} samples: {avg_snr:.2f} dB")
    
    # Save metrics
    with open(os.path.join(dataset_dir, 'metrics.txt'), 'w') as f:
        f.write(f"Average SNR: {avg_snr:.2f} dB\n")
        f.write("Individual SNR values:\n")
        for i, snr in enumerate(all_snr_values):
            f.write(f"Sample {i}: {snr:.2f} dB\n")

def process_audio_file(audio_path, model, config, device, output_dir):
    """Process an individual audio file"""
    print(f"Processing audio file: {audio_path}")
    
    # Create output directory
    file_name = os.path.basename(audio_path).split('.')[0]
    file_dir = os.path.join(output_dir, file_name)
    os.makedirs(file_dir, exist_ok=True)
    
    # Load audio file using soundfile
    try:
        audio_data, sample_rate = sf.read(audio_path)
    except:
        # Fall back to loading using simple numpy
        try:
            import numpy as np
            audio_data = np.fromfile(audio_path, dtype=np.float32)
            sample_rate = config.sample_rate  # Assume config sample rate
        except:
            print(f"Error loading audio file: {audio_path}")
            return
    
    # Convert to mono if stereo
    if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
        audio_data = np.mean(audio_data, axis=1)
    
    # Resample if necessary
    if sample_rate != config.sample_rate:
        print(f"Resampling from {sample_rate} Hz to {config.sample_rate} Hz")
        try:
            from scipy import signal
            audio_data = signal.resample(audio_data, int(len(audio_data) * config.sample_rate / sample_rate))
        except:
            print(f"Warning: Resampling failed, continuing with original sample rate")
    
    # Trim or pad to match model's expected length
    if len(audio_data) > config.audio_length:
        print(f"Trimming audio from {len(audio_data)} to {config.audio_length} samples")
        audio_data = audio_data[:config.audio_length]
    elif len(audio_data) < config.audio_length:
        print(f"Padding audio from {len(audio_data)} to {config.audio_length} samples")
        padding = np.zeros(config.audio_length - len(audio_data))
        audio_data = np.concatenate([audio_data, padding])
    
    # Normalize
    audio_data = audio_data / np.max(np.abs(audio_data))
    
    # Convert to tensor
    audio = torch.tensor(audio_data, dtype=torch.float32).unsqueeze(0).to(device)
    
    # Process through model
    with torch.no_grad():
        reconstructed, wavelet_coeffs, wavelet_coeffs_recon, latent = model(audio)
    
    # Calculate SNR
    snr = calculate_snr(audio, reconstructed)[0].item()
    print(f"SNR: {snr:.2f} dB")
    
    # Visualize and save results
    visualize_results(
        audio[0].cpu(),
        reconstructed[0].cpu(),
        wavelet_coeffs[0].cpu(),
        wavelet_coeffs_recon[0].cpu(),
        latent[0].cpu(),
        snr,
        config.sample_rate,
        file_dir
    )

def visualize_results(original, reconstructed, wavelet_coeffs, wavelet_coeffs_recon, latent, snr, sample_rate, output_dir):
    """Visualize and save inference results"""
    # Convert tensors to numpy
    orig_np = original.numpy()
    recon_np = reconstructed.numpy()
    coeffs_np = wavelet_coeffs.numpy()
    coeffs_recon_np = wavelet_coeffs_recon.numpy()
    latent_np = latent.numpy()
    
    # 1. Waveform comparison
    fig, ax = plt.subplots(figsize=(12, 4))
    
    # Create time axis in seconds
    t = np.arange(len(orig_np)) / sample_rate
    
    # Plot original waveform
    ax.plot(t, orig_np, alpha=0.7, label='Original', color='blue')
    
    # Plot reconstructed waveform
    ax.plot(t, recon_np, alpha=0.7, label='Reconstructed', color='red')
    
    # Plot error region
    error = orig_np - recon_np
    ax.fill_between(t, error, -error, color='grey', alpha=0.3, label='Error')
    
    ax.set_title(f'Waveform Comparison - SNR: {snr:.2f} dB')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Set reasonable y-limits
    max_amp = max(np.max(np.abs(orig_np)), np.max(np.abs(recon_np)))
    ax.set_ylim(-max_amp * 1.1, max_amp * 1.1)
    
    # Add zoom-in panel for detail
    zoom_start = len(orig_np) // 3
    zoom_width = min(1000, len(orig_np) // 10)
    zoom_end = zoom_start + zoom_width
    
    # Create inset axis for zoom
    axins = ax.inset_axes([0.05, 0.05, 0.3, 0.3])
    axins.plot(t[zoom_start:zoom_end], orig_np[zoom_start:zoom_end], 'b-')
    axins.plot(t[zoom_start:zoom_end], recon_np[zoom_start:zoom_end], 'r-')
    axins.set_title('Zoom', fontsize=8)
    axins.grid(True, alpha=0.3)
    
    # Mark zoom region on main plot
    ax.indicate_inset_zoom(axins, edgecolor="black")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'waveform_comparison.png'), dpi=300)
    plt.close(fig)
    
    # 2. Wavelet coefficient comparison
    fig, axs = plt.subplots(3, 1, figsize=(12, 8))
    
    # Determine reasonable viewing window for coefficients
    view_size = min(2000, len(coeffs_np))
    
    # Original coefficients
    axs[0].plot(coeffs_np[:view_size], 'b-', label='Original')
    axs[0].set_title('Original Wavelet Coefficients')
    axs[0].grid(True, alpha=0.3)
    
    # Reconstructed coefficients
    axs[1].plot(coeffs_recon_np[:view_size], 'r-', label='Reconstructed')
    axs[1].set_title('Reconstructed Wavelet Coefficients')
    axs[1].grid(True, alpha=0.3)
    
    # Coefficient difference
    diff = coeffs_np[:view_size] - coeffs_recon_np[:view_size]
    axs[2].plot(diff, 'g-', label='Difference')
    axs[2].set_title('Coefficient Difference')
    axs[2].set_xlabel('Coefficient Index')
    axs[2].grid(True, alpha=0.3)
    
    # Set consistent y-limits for comparison
    coeff_max = max(np.max(np.abs(coeffs_np[:view_size])), np.max(np.abs(coeffs_recon_np[:view_size])))
    axs[0].set_ylim(-coeff_max * 1.1, coeff_max * 1.1)
    axs[1].set_ylim(-coeff_max * 1.1, coeff_max * 1.1)
    
    # Set y-limit for difference plot
    diff_max = np.max(np.abs(diff))
    axs[2].set_ylim(-diff_max * 1.1, diff_max * 1.1)
    
    # Add coefficient statistics
    mse = np.mean((diff) ** 2)
    axs[2].text(0.02, 0.92, f'MSE: {mse:.6f}', transform=axs[2].transAxes, 
               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'wavelet_comparison.png'), dpi=300)
    plt.close(fig)
    
    # 3. Spectrogram comparison
    n_fft = min(1024, len(orig_np) // 4)
    hop_length = n_fft // 4
    
    # Calculate spectrograms
    spec_orig = compute_spectrogram(orig_np, n_fft, hop_length)
    spec_recon = compute_spectrogram(recon_np, n_fft, hop_length)
    
    # Convert to dB scale
    eps = 1e-10
    spec_orig_db = 20 * np.log10(spec_orig + eps)
    spec_recon_db = 20 * np.log10(spec_recon + eps)
    
    # Determine adaptive scaling
    all_values = np.concatenate([spec_orig_db.flatten(), spec_recon_db.flatten()])
    vmin = np.percentile(all_values, 1)
    vmax = np.percentile(all_values, 99)
    
    # Create frequency axis (Hz)
    freqs = np.linspace(0, sample_rate/2, spec_orig.shape[0])
    
    # Create time axis (seconds)
    spec_times = np.linspace(0, len(orig_np)/sample_rate, spec_orig.shape[1])
    
    # Create figure
    fig, axs = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot original spectrogram
    im0 = axs[0].imshow(spec_orig_db, aspect='auto', origin='lower', 
                    cmap='viridis', vmin=vmin, vmax=vmax)
    axs[0].set_title('Original Spectrogram')
    axs[0].set_ylabel('Frequency (Hz)')
    
    # Add frequency axis labels
    n_freq_labels = 6
    freq_indices = np.linspace(0, len(freqs)-1, n_freq_labels, dtype=int)
    axs[0].set_yticks(freq_indices)
    axs[0].set_yticklabels([f'{freqs[i]:.0f}' for i in freq_indices])
    
    # Plot reconstructed spectrogram
    im1 = axs[1].imshow(spec_recon_db, aspect='auto', origin='lower', 
                    cmap='viridis', vmin=vmin, vmax=vmax)
    axs[1].set_title('Reconstructed Spectrogram')
    axs[1].set_ylabel('Frequency (Hz)')
    axs[1].set_xlabel('Time (s)')
    
    # Add frequency axis labels
    axs[1].set_yticks(freq_indices)
    axs[1].set_yticklabels([f'{freqs[i]:.0f}' for i in freq_indices])
    
    # Add time axis labels
    n_time_labels = 6
    time_indices = np.linspace(0, len(spec_times)-1, n_time_labels, dtype=int)
    axs[1].set_xticks(time_indices)
    axs[1].set_xticklabels([f'{spec_times[i]:.2f}' for i in time_indices])
    
    # Add colorbar
    fig.colorbar(im0, ax=axs, orientation='vertical', label='Power (dB)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'spectrogram_comparison.png'), dpi=300)
    plt.close(fig)
    
    # 4. Latent space visualization
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(latent_np)), latent_np)
    ax.set_title('Latent Space Representation')
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Value')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'latent_representation.png'), dpi=300)
    plt.close(fig)
    
    # 5. Save audio files
    try:
        from scipy.io import wavfile
        wavfile.write(os.path.join(output_dir, 'original.wav'), sample_rate, orig_np)
        wavfile.write(os.path.join(output_dir, 'reconstructed.wav'), sample_rate, recon_np)
    except ImportError:
        print("SciPy not available, skipping audio file saving")
    
    # 6. Save metrics text file
    with open(os.path.join(output_dir, 'metrics.txt'), 'w') as f:
        f.write(f"SNR: {snr:.2f} dB\n")
        f.write(f"MSE (wavelet coefficients): {mse:.6f}\n")
        f.write(f"Compression ratio: {len(orig_np) / len(latent_np):.2f}x\n")

if __name__ == "__main__":
    main()