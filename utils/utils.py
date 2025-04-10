import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import io
from PIL import Image

def plot_waveform(waveform, sample_rate=16000, title="Waveform"):
    """
    Plot a waveform
    Args:
        waveform: Tensor of shape [1, T] or [T]
        sample_rate: Audio sample rate
        title: Plot title
    Returns:
        Matplotlib Figure
    """
    waveform = waveform.squeeze().cpu().numpy()
    
    fig = Figure(figsize=(10, 4))
    ax = fig.add_subplot(1, 1, 1)
    
    time_axis = np.arange(0, len(waveform)) / sample_rate
    ax.plot(time_axis, waveform)
    
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    ax.grid(True)
    
    return fig

def plot_spectrogram(waveform, sample_rate=16000, title="Spectrogram"):
    """
    Plot a spectrogram
    Args:
        waveform: Tensor of shape [1, T] or [T]
        sample_rate: Audio sample rate
        title: Plot title
    Returns:
        Matplotlib Figure
    """
    waveform = waveform.squeeze().cpu().numpy()
    
    fig = Figure(figsize=(10, 4))
    ax = fig.add_subplot(1, 1, 1)
    
    # Compute spectrogram with matplotlib
    ax.specgram(
        waveform,
        NFFT=1024,
        Fs=sample_rate,
        noverlap=512,
        scale='dB'
    )
    
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(title)
    
    return fig

def plot_comparison(original, reconstructed, sample_rate=16000):
    """
    Plot comparison between original and reconstructed waveforms
    Args:
        original: Original audio tensor [1, T]
        reconstructed: Reconstructed audio tensor [1, T]
        sample_rate: Audio sample rate
    Returns:
        Matplotlib Figure
    """
    original = original.squeeze().cpu().numpy()
    reconstructed = reconstructed.squeeze().cpu().numpy()
    
    time_axis = np.arange(0, len(original)) / sample_rate
    
    fig = Figure(figsize=(10, 8))
    
    # Plot waveforms
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(time_axis, original, label='Original')
    ax1.plot(time_axis, reconstructed, label='Reconstructed', alpha=0.7)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
    ax1.set_title("Waveform Comparison")
    ax1.legend()
    ax1.grid(True)
    
    # Plot difference
    ax2 = fig.add_subplot(2, 1, 2)
    difference = original - reconstructed
    ax2.plot(time_axis, difference)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Difference")
    ax2.set_title("Reconstruction Error")
    ax2.grid(True)
    
    fig.tight_layout()
    return fig

def figure_to_tensor(figure):
    """Convert matplotlib figure to tensor for tensorboard"""
    buf = io.BytesIO()
    figure.savefig(buf, format='png')
    buf.seek(0)
    
    # Open image with PIL and convert to tensor
    image = Image.open(buf)
    image = np.array(image)
    # Remove alpha channel if it exists
    if image.shape[2] == 4:
        image = image[:, :, :3]
    
    # Convert to torch tensor
    image = torch.from_numpy(image).permute(2, 0, 1)
    
    plt.close(figure)
    buf.close()
    
    return image

def compute_metrics(original, reconstructed):
    """
    Compute audio quality metrics
    Args:
        original: Original audio tensor
        reconstructed: Reconstructed audio tensor
    Returns:
        Dictionary of metrics
    """
    # Ensure tensors are flat
    original = original.view(-1)
    reconstructed = reconstructed.view(-1)
    
    # Mean squared error
    mse = torch.nn.functional.mse_loss(original, reconstructed).item()
    
    # Signal-to-noise ratio (SNR)
    noise = original - reconstructed
    signal_power = torch.sum(original**2).item()
    noise_power = torch.sum(noise**2).item()
    
    # Avoid division by zero
    if noise_power < 1e-10:
        snr = 100.0  # High value for very low noise
    else:
        snr = 10 * np.log10(signal_power / noise_power)
    
    # Energy preservation
    energy_ratio = torch.sum(reconstructed**2).item() / max(torch.sum(original**2).item(), 1e-10)
    
    return {
        "mse": mse,
        "snr": snr,
        "energy_ratio": energy_ratio
    }