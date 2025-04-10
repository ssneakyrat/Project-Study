import torch
import numpy as np
import matplotlib.pyplot as plt
from lightning_module import ComplexAudioEncoderDecoder
import yaml

def debug_tensor_dimensions(model, audio_input):
    """
    Debug function to trace tensor dimensions through the network
    """
    print("=" * 50)
    print("DEBUGGING TENSOR DIMENSIONS")
    print("=" * 50)
    
    # Store original shape
    print(f"Input audio shape: {audio_input.shape}")
    
    # Apply wavelet transform
    wst_output = model.wst(audio_input)
    
    # Print WST output shape
    if isinstance(wst_output, tuple):
        print(f"WST output (real) shape: {wst_output[0].shape}")
        print(f"WST output (imag) shape: {wst_output[1].shape}")
        
        # Check for potential dimension issue
        if wst_output[0].size(1) < wst_output[0].size(2):
            print("POTENTIAL DIMENSION ISSUE: Channels < Time bins")
            print("This would cause a dimensional rotation in convolutional layers!")
    else:
        print(f"WST output shape: {wst_output.shape}")
        
        # Check for potential dimension issue
        if wst_output.size(1) < wst_output.size(2):
            print("POTENTIAL DIMENSION ISSUE: Channels < Time bins")
            print("This would cause a dimensional rotation in convolutional layers!")
    
    # Pass through encoder
    encoded, intermediates = model.encoder(wst_output)
    
    # Print encoder output shape
    if isinstance(encoded, tuple):
        print(f"Encoder output (real) shape: {encoded[0].shape}")
        print(f"Encoder output (imag) shape: {encoded[1].shape}")
    else:
        print(f"Encoder output shape: {encoded.shape}")
    
    # Print intermediates shapes
    print("\nIntermediate shapes:")
    for i, inter in enumerate(intermediates):
        if isinstance(inter, tuple):
            print(f"  Layer {i} (real): {inter[0].shape}")
            print(f"  Layer {i} (imag): {inter[1].shape}")
        else:
            print(f"  Layer {i}: {inter.shape}")
    
    # Pass through decoder
    decoded = model.decoder(encoded, intermediates)
    
    # Print decoder output shape
    if isinstance(decoded, tuple):
        print(f"Decoder output (real) shape: {decoded[0].shape}")
        print(f"Decoder output (imag) shape: {decoded[1].shape}")
    else:
        print(f"Decoder output shape: {decoded.shape}")
    
    # Convert to real
    output = model.to_real(decoded)
    print(f"Final output shape: {output.shape}")
    
    print("\nDIMENSION CHECK COMPLETE")
    print("=" * 50)
    
    return output


def visualize_spectrograms(audio_input, audio_output, n_fft=1024, hop_length=256, sample_rate=16000):
    """
    Visualize input and output spectrograms
    """
    # Compute STFTs
    input_stft = torch.stft(audio_input.squeeze(), n_fft, hop_length, 
                           window=torch.hann_window(n_fft), 
                           return_complex=True)
    output_stft = torch.stft(audio_output.squeeze(), n_fft, hop_length, 
                            window=torch.hann_window(n_fft), 
                            return_complex=True)
    
    # Convert to magnitude in dB
    input_mag = 20 * torch.log10(torch.abs(input_stft) + 1e-8).numpy()
    output_mag = 20 * torch.log10(torch.abs(output_stft) + 1e-8).numpy()
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Plot input spectrogram
    plt.subplot(2, 1, 1)
    plt.imshow(input_mag, aspect='auto', origin='lower', cmap='viridis')
    plt.title('Input Spectrogram')
    plt.ylabel('Frequency bin')
    plt.colorbar()
    
    # Plot output spectrogram
    plt.subplot(2, 1, 2)
    plt.imshow(output_mag, aspect='auto', origin='lower', cmap='viridis')
    plt.title('Reconstructed Spectrogram')
    plt.xlabel('Time frame')
    plt.ylabel('Frequency bin')
    plt.colorbar()
    
    plt.tight_layout()
    plt.savefig('spectrograms_debug.png')
    plt.close()
    
    # Create waveform comparison
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(audio_input.squeeze().numpy())
    plt.title('Input Waveform')
    plt.subplot(2, 1, 2)
    plt.plot(audio_output.squeeze().numpy())
    plt.title('Reconstructed Waveform')
    plt.tight_layout()
    plt.savefig('waveforms_debug.png')
    plt.close()
    
    print("Visualizations saved to 'spectrograms_debug.png' and 'waveforms_debug.png'")


if __name__ == "__main__":
    # Load configuration
    with open('config/model.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize model
    model = ComplexAudioEncoderDecoder(config)
    
    # Generate a test frequency sweep
    sample_rate = config['data']['sample_rate']
    duration = config['data']['audio_length']
    t = torch.linspace(0, duration, int(duration * sample_rate))
    
    # Create logarithmic frequency sweep from 50 Hz to 8000 Hz
    f0, f1 = 50, 8000
    sweep = torch.sin(2 * np.pi * f0 * duration / np.log(f1/f0) * 
                     (torch.exp(t/duration * np.log(f1/f0)) - 1))
    
    # Add batch dimension
    sweep = sweep.unsqueeze(0)
    
    # Run debug
    with torch.no_grad():
        output = debug_tensor_dimensions(model, sweep)
    
    # Visualize results
    visualize_spectrograms(sweep, output, sample_rate=sample_rate)