import os
import argparse
import torch
import h5py
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pytorch_lightning as pl

from utils.utils import load_config, calculate_snr, compute_spectrogram
from lightning_model import WaveletAudioAE
from models.sliding_window_model import SlidingWindowModel

def load_continuous_data(file_path):
    """Load continuous audio data from HDF5 file
    
    Returns:
        audio_data: Tensor of audio data [n_samples, audio_length]
        markers: Dictionary of markers/metadata
    """
    with h5py.File(file_path, 'r') as f:
        audio_data = torch.tensor(f['audio'][:], dtype=torch.float32)
        
        # Load markers if available
        markers = {}
        if 'markers' in f:
            for sample_idx in f['markers']:
                markers[int(sample_idx)] = {}
                for key in f['markers'][sample_idx]:
                    markers[int(sample_idx)][key] = f['markers'][sample_idx][key][:]
    
    return audio_data, markers

def visualize_continuous_processing(original, reconstructed, window_boundaries, sample_rate, output_dir):
    """Visualize continuous audio processing results
    
    Args:
        original: Original audio tensor [audio_length]
        reconstructed: Reconstructed audio tensor [audio_length]
        window_boundaries: List of window boundary indices
        sample_rate: Audio sample rate
        output_dir: Output directory for plots
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to numpy
    orig_np = original.cpu().numpy()
    recon_np = reconstructed.cpu().numpy()
    
    # Create time axis in seconds
    t = np.arange(len(orig_np)) / sample_rate
    
    # 1. Waveform comparison with window boundaries
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot original and reconstructed waveforms
    ax.plot(t, orig_np, alpha=0.7, label='Original', color='blue')
    ax.plot(t, recon_np, alpha=0.7, label='Reconstructed', color='red')
    
    # Plot error region
    error = orig_np - recon_np
    ax.fill_between(t, error, -error, color='grey', alpha=0.3, label='Error')
    
    # Add window boundaries
    for boundary in window_boundaries:
        boundary_time = boundary / sample_rate
        ax.axvline(x=boundary_time, color='green', linestyle='--', alpha=0.5, label='Window Boundary' if boundary == window_boundaries[0] else None)
    
    # Calculate SNR per window
    window_snrs = []
    for i in range(len(window_boundaries) - 1):
        start = window_boundaries[i]
        end = window_boundaries[i] + (window_boundaries[1] - window_boundaries[0])  # Assuming equal window sizes
        if end > len(orig_np):
            end = len(orig_np)
        
        window_orig = torch.tensor(orig_np[start:end]).unsqueeze(0)
        window_recon = torch.tensor(recon_np[start:end]).unsqueeze(0)
        
        snr = calculate_snr(window_orig, window_recon)[0].item()
        window_snrs.append(snr)
        
        # Add SNR text for each window
        mid_point = (start + min(end, len(orig_np) - 1)) / 2
        mid_time = mid_point / sample_rate
        ax.text(mid_time, 0.9 * ax.get_ylim()[1], f"SNR: {snr:.1f} dB", 
                horizontalalignment='center', fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))
    
    # Calculate overall SNR
    overall_snr = calculate_snr(
        torch.tensor(orig_np).unsqueeze(0), 
        torch.tensor(recon_np).unsqueeze(0)
    )[0].item()
    
    ax.set_title(f'Continuous Audio Processing - Overall SNR: {overall_snr:.2f} dB')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Set reasonable y-limits
    max_amp = max(np.max(np.abs(orig_np)), np.max(np.abs(recon_np)))
    ax.set_ylim(-max_amp * 1.1, max_amp * 1.1)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'continuous_waveform.png'), dpi=300)
    plt.close(fig)
    
    # 2. Spectrogram comparison with window boundaries
    # Compute spectrograms
    n_fft = min(1024, len(orig_np) // 10)  # Adjust based on signal length
    hop_length = n_fft // 4
    
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
    
    # Create spectrogram figure
    fig, axs = plt.subplots(2, 1, figsize=(14, 10))
    
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
    
    # Add window boundary markers
    for boundary in window_boundaries:
        # Convert sample positions to spectrogram frame indices
        frame_idx = int(boundary / hop_length)
        if frame_idx < spec_orig.shape[1]:
            axs[0].axvline(x=frame_idx, color='white', linestyle='--', alpha=0.5)
            axs[1].axvline(x=frame_idx, color='white', linestyle='--', alpha=0.5)
    
    # Add colorbar
    fig.colorbar(im0, ax=axs, orientation='vertical', label='Power (dB)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'continuous_spectrogram.png'), dpi=300)
    plt.close(fig)
    
    # 3. SNR across windows chart
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Plot SNR for each window
    window_indices = np.arange(len(window_snrs))
    ax.bar(window_indices, window_snrs, alpha=0.7)
    ax.axhline(y=overall_snr, color='r', linestyle='--', label=f'Overall SNR: {overall_snr:.2f} dB')
    
    ax.set_title('SNR Across Windows')
    ax.set_xlabel('Window Index')
    ax.set_ylabel('SNR (dB)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'window_snr.png'), dpi=300)
    plt.close(fig)
    
    # 4. Save audio files
    try:
        from scipy.io import wavfile
        wavfile.write(os.path.join(output_dir, 'original.wav'), sample_rate, orig_np)
        wavfile.write(os.path.join(output_dir, 'reconstructed.wav'), sample_rate, recon_np)
    except ImportError:
        print("SciPy not available, skipping audio file saving")

def main():
    parser = argparse.ArgumentParser(description='Test continuous audio processing with sliding windows')
    parser.add_argument('--config', default='config/model.yaml', help='Path to config file')
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    parser.add_argument('--data', default='data/audio/continuous_medium.h5', help='Path to continuous audio data')
    parser.add_argument('--output_dir', default='results/continuous', help='Output directory')
    parser.add_argument('--overlap', type=float, default=0.75, help='Window overlap ratio (0-1)')
    parser.add_argument('--window_type', default='hann', help='Window function type')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for processing')
    args = parser.parse_args()
    
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
    
    # Load continuous audio data
    print(f"Loading continuous audio data from {args.data}")
    audio_data, markers = load_continuous_data(args.data)
    print(f"Loaded {len(audio_data)} audio samples with {audio_data.shape[1]} samples each")
    
    # Calculate window parameters
    window_size = config.audio_length  # Use model's native audio length as window size
    hop_size = int(window_size * (1.0 - args.overlap))  # Calculate hop size from overlap
    print(f"Using window size: {window_size} samples, hop size: {hop_size} samples")
    print(f"Each window is {window_size/config.sample_rate:.2f} seconds, with {args.overlap:.0%} overlap")
    
    # Create sliding window model
    sliding_model = SlidingWindowModel(model, window_size, hop_size, args.window_type)
    sliding_model.eval()
    sliding_model.to(device)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process each audio sample
    for sample_idx in tqdm(range(min(5, len(audio_data))), desc="Processing audio samples"):
        # Get audio sample
        audio = audio_data[sample_idx:sample_idx+1].to(device)
        
        # Process with sliding window
        with torch.no_grad():
            reconstructed, segments, boundary_indices = sliding_model(audio)
        
        # Evaluate SNR
        snr = calculate_snr(audio, reconstructed)[0].item()
        print(f"Sample {sample_idx} - SNR: {snr:.2f} dB")
        
        # Create sample output directory
        sample_dir = os.path.join(args.output_dir, f"sample_{sample_idx}")
        
        # Visualize results
        visualize_continuous_processing(
            audio[0], 
            reconstructed[0], 
            boundary_indices, 
            config.sample_rate,
            sample_dir
        )
    
    print("Continuous audio processing test completed!")

if __name__ == "__main__":
    main()