import argparse
import os
import torch
import torchaudio
import matplotlib.pyplot as plt
import time
from model import WaveletAEModel
from metrics import compute_metrics

def process_audio(model, audio_path, output_dir, sample_rate=16000, device="cpu"):
    """Process audio file with model and save results"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Load audio file
    waveform, sr = torchaudio.load(audio_path)
    
    # Convert to mono if necessary
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Resample if necessary
    if sr != sample_rate:
        waveform = torchaudio.transforms.Resample(sr, sample_rate)(waveform)
    
    # Process in chunks if needed
    model = model.to(device)
    model.eval()
    
    # Get filename
    filename = os.path.basename(audio_path).split('.')[0]
    
    # Measure time
    start_time = time.time()
    
    # Process entire waveform
    waveform = waveform.to(device)
    
    with torch.no_grad():
        reconstructed = model(waveform.unsqueeze(0))
    
    processing_time = time.time() - start_time
    
    # Calculate metrics
    metrics = compute_metrics(reconstructed, waveform.unsqueeze(0))
    
    # Save reconstructed audio
    output_path = os.path.join(output_dir, f"{filename}_reconstructed.wav")
    torchaudio.save(output_path, reconstructed[0].cpu(), sample_rate)
    
    # Plot comparison
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.title("Original Audio")
    plt.plot(waveform[0].cpu().numpy())
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.title(f"Reconstructed Audio (SNR: {metrics['snr_db']:.2f} dB)")
    plt.plot(reconstructed[0, 0].cpu().numpy())
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{filename}_comparison.png"))
    
    # Generate a log of results
    with open(os.path.join(output_dir, f"{filename}_log.txt"), "w") as f:
        f.write(f"Processing results for: {audio_path}\n")
        f.write(f"Processing time: {processing_time:.4f} seconds\n\n")
        f.write("Metrics:\n")
        for key, value in metrics.items():
            f.write(f"  {key}: {value:.4f}\n")
    
    return metrics, processing_time, output_path

def main(args):
    """Main inference function"""
    # Load model
    if args.checkpoint:
        model = WaveletAEModel.load_from_checkpoint(args.checkpoint)
        print(f"Loaded model from checkpoint: {args.checkpoint}")
    else:
        # Create new model with default parameters
        model = WaveletAEModel(segment_length=args.segment_length)
        print("Using untrained model with default parameters")
    
    # Set device
    device = "cuda" if torch.cuda.is_available() and args.gpu else "cpu"
    print(f"Using device: {device}")
    
    # Process input
    if os.path.isdir(args.input):
        # Process all audio files in directory
        audio_files = [
            os.path.join(args.input, f) for f in os.listdir(args.input)
            if f.endswith(('.wav', '.mp3', '.flac', '.ogg'))
        ]
        
        results = []
        for audio_file in audio_files:
            print(f"Processing: {audio_file}")
            metrics, time_taken, output_path = process_audio(
                model, audio_file, args.output_dir, args.sample_rate, device
            )
            results.append({
                "file": audio_file,
                "output": output_path,
                "metrics": metrics,
                "time": time_taken
            })
            
        # Print summary
        print("\nProcessing Summary:")
        print(f"{'File':<30} {'SNR (dB)':<10} {'MAE':<10} {'Time (s)':<10}")
        print("-" * 60)
        for result in results:
            filename = os.path.basename(result["file"])
            print(f"{filename:<30} {result['metrics']['snr_db']:<10.2f} {result['metrics']['mae']:<10.4f} {result['time']:<10.4f}")
    
    else:
        # Process single file
        print(f"Processing: {args.input}")
        metrics, time_taken, output_path = process_audio(
            model, args.input, args.output_dir, args.sample_rate, device
        )
        
        # Print results
        print("\nProcessing Results:")
        print(f"Output saved to: {output_path}")
        print(f"Processing time: {time_taken:.4f} seconds")
        print("\nMetrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audio Wavelet Encoder-Decoder Inference")
    
    parser.add_argument("input", type=str, help="Input audio file or directory")
    parser.add_argument("--output_dir", type=str, default="inference_output", 
                        help="Output directory for processed files")
    parser.add_argument("--checkpoint", type=str, default=None, 
                        help="Path to model checkpoint file (.ckpt)")
    parser.add_argument("--sample_rate", type=int, default=16000, 
                        help="Audio sample rate")
    parser.add_argument("--segment_length", type=int, default=16000, 
                        help="Audio segment length for processing")
    parser.add_argument("--gpu", action="store_true", 
                        help="Use GPU for inference if available")
    
    args = parser.parse_args()
    main(args)