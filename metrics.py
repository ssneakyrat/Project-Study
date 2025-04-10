import torch
import torch.nn.functional as F
import torchaudio
import numpy as np

def compute_metrics(y_pred, y_true):
    """
    Compute audio quality metrics between predicted and true audio
    
    Args:
        y_pred: Predicted audio [batch_size, 1, length]
        y_true: Target audio [batch_size, 1, length]
        
    Returns:
        Dictionary of metrics
    """
    # Ensure same length
    min_len = min(y_pred.shape[-1], y_true.shape[-1])
    y_pred = y_pred[..., :min_len]
    y_true = y_true[..., :min_len]
    
    # Move to CPU for numpy calculations
    y_pred_np = y_pred.detach().cpu().numpy()
    y_true_np = y_true.detach().cpu().numpy()
    
    # Signal-to-Noise Ratio (SNR)
    def compute_snr(pred, true):
        noise = pred - true
        snr_ratio = np.sum(true**2) / (np.sum(noise**2) + 1e-10)
        return 10 * np.log10(snr_ratio + 1e-10)
    
    # Mean absolute error
    mae = F.l1_loss(y_pred, y_true).item()
    
    # Root mean square error
    rmse = torch.sqrt(F.mse_loss(y_pred, y_true)).item()
    
    # Batch SNR
    snr_values = [
        compute_snr(y_pred_np[i, 0], y_true_np[i, 0])
        for i in range(y_pred.shape[0])
    ]
    snr = np.mean(snr_values)
    
    # Calculate LSD (Log-Spectral Distance) if batch size is reasonable
    if y_pred.shape[0] <= 16:  # Skip for very large batches
        try:
            # Use STFT to get spectrograms
            n_fft = 1024
            window = torch.hann_window(n_fft).to(y_pred.device)
            
            # Compute STFT for predicted and true audio
            spec_pred = torch.stft(
                y_pred.squeeze(1), 
                n_fft=n_fft, 
                hop_length=n_fft//4,
                window=window,
                return_complex=True
            )
            spec_true = torch.stft(
                y_true.squeeze(1), 
                n_fft=n_fft, 
                hop_length=n_fft//4,
                window=window,
                return_complex=True
            )
            
            # Compute magnitude spectrograms and convert to log scale
            log_spec_pred = torch.log10(torch.abs(spec_pred) + 1e-7)
            log_spec_true = torch.log10(torch.abs(spec_true) + 1e-7)
            
            # Compute LSD (squared difference of log spectra)
            lsd = torch.sqrt(torch.mean((log_spec_pred - log_spec_true)**2, dim=1))
            lsd = torch.mean(lsd).item()
        except Exception as e:
            # Fallback if STFT computation fails
            lsd = float('nan')
    else:
        lsd = float('nan')
    
    # Return metrics dictionary
    return {
        'mae': mae,
        'rmse': rmse,
        'snr_db': snr,
        'lsd': lsd
    }