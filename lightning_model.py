import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import numpy as np
from models.wem import WaveletEchoMatrix
from models.loss import CombinedLoss
from models.metric import MultiMetric

class WEMLightningModel(pl.LightningModule):
    def __init__(self, config):
        super(WEMLightningModel, self).__init__()
        self.save_hyperparameters(config)
        self.config = config
        
        # Initialize model
        self.model = WaveletEchoMatrix(
            wavelet=config['wavelet'],
            levels=config['levels'],
            hidden_dim=config['hidden_dim'],
            latent_dim=config['latent_dim']
        )
        
        # Initialize loss function
        self.loss_fn = CombinedLoss(
            wavelet_weight=config['wavelet_loss_weight'],
            time_weight=config['time_loss_weight'],
            levels=config['levels'],
            alpha=config['loss_alpha']
        )
        
        # Initialize metrics
        self.metrics = MultiMetric()
        
    def forward(self, x):
        return self.model(x)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(  # Changed from Adam to AdamW for better weight decay handling
            self.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }
    
    def training_step(self, batch, batch_idx):
        x = batch
        output = self(x)
        
        loss = self.loss_fn(
            output['reconstructed'],
            x,
            output['rec_coeffs'],
            output['coeffs']
        )
        
        # Log training metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        if batch_idx % 100 == 0:  # Log detailed metrics periodically
            metrics = self.metrics(output['reconstructed'], x)
            for name, value in metrics.items():
                self.log(f'train_{name}', value, on_step=True, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch
        output = self(x)
        
        loss = self.loss_fn(
            output['reconstructed'],
            x,
            output['rec_coeffs'],
            output['coeffs']
        )
        
        # Log validation metrics
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        
        metrics = self.metrics(output['reconstructed'], x)
        for name, value in metrics.items():
            self.log(f'val_{name}', value, on_epoch=True)
        
        # Visual comparison at specified frequency
        viz_frequency = self.config.get('viz_every_n_epochs', 5)  # Default to 5 if not specified
        current_epoch = self.current_epoch
        
        if current_epoch % viz_frequency == 0 and batch_idx == 0:
            # Generate visualizations for the first batch only
            viz_num_samples = self.config.get('viz_num_samples', 3)  # Get number of samples to visualize
            self._log_audio_comparison(x, output['reconstructed'], max_samples=viz_num_samples)
            self._log_wavelet_comparison(output['coeffs'], output['rec_coeffs'])
        
        return loss
    
    def _log_audio_comparison(self, original, reconstructed, max_samples=None):
        """
        Create and log visualizations comparing original and reconstructed audio
        
        Args:
            original: Original audio tensor [B, T]
            reconstructed: Reconstructed audio tensor [B, T]
            max_samples: Maximum number of samples to visualize (defaults to config value)
        """
        # Get max samples from config if not specified
        if max_samples is None:
            max_samples = self.config.get('viz_num_samples', 3)  # Default to 3 if not in config
            
        # Select subset of samples to visualize
        n_samples = min(original.size(0), max_samples)
        sample_rate = self.config.get('sample_rate', 16000)
        
        for i in range(n_samples):
            # Get sample data
            orig = original[i].cpu().numpy()
            recon = reconstructed[i].cpu().numpy()
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 4))
            
            # Plot both waveforms
            time_axis = np.arange(len(orig)) / sample_rate
            ax.plot(time_axis, orig, alpha=0.7, label='Ground Truth', color='blue')
            ax.plot(time_axis, recon, alpha=0.7, label='Reconstructed', color='red')
            
            # Add labels and legend
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Amplitude')
            ax.set_title(f'Audio Comparison - Sample {i+1}')
            ax.legend()
            ax.grid(True)
            
            # Log figure to TensorBoard
            self.logger.experiment.add_figure(
                f'audio_comparison/sample_{i}', 
                fig, 
                global_step=self.current_epoch
            )
            plt.close(fig)
            
            # Also log spectrograms
            self._log_spectrogram_comparison(orig, recon, i)
            
            # Add playable audio to TensorBoard
            # Normalize audio to [-1, 1] for playback
            if np.max(np.abs(orig)) > 0:
                orig_normalized = orig / np.max(np.abs(orig))
            else:
                orig_normalized = orig
                
            if np.max(np.abs(recon)) > 0:
                recon_normalized = recon / np.max(np.abs(recon))
            else:
                recon_normalized = recon
            
            # Log original audio
            self.logger.experiment.add_audio(
                f'audio_playback/original_{i}',
                orig_normalized,
                global_step=self.current_epoch,
                sample_rate=sample_rate
            )
            
            # Log reconstructed audio
            self.logger.experiment.add_audio(
                f'audio_playback/reconstructed_{i}',
                recon_normalized,
                global_step=self.current_epoch,
                sample_rate=sample_rate
            )
            
            # Log difference audio (error) - this is valuable for hearing artifacts
            diff_audio = orig_normalized - recon_normalized
            # Normalize difference for better audibility
            if np.max(np.abs(diff_audio)) > 0:
                diff_normalized = diff_audio / np.max(np.abs(diff_audio)) * 0.9  # Scale to avoid clipping
            else:
                diff_normalized = diff_audio
                
            self.logger.experiment.add_audio(
                f'audio_playback/difference_{i}',
                diff_normalized,
                global_step=self.current_epoch,
                sample_rate=sample_rate
            )
    
    def _log_spectrogram_comparison(self, original, reconstructed, sample_idx):
        """
        Create and log visualizations comparing original and reconstructed spectrograms
        
        Args:
            original: Original audio numpy array
            reconstructed: Reconstructed audio numpy array
            sample_idx: Sample index for logging
        """
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        
        # Compute spectrograms
        D_orig = self._compute_spectrogram(original)
        D_recon = self._compute_spectrogram(reconstructed)
        
        # Get min/max values from ground truth for consistent scale
        vmin = np.min(D_orig)
        vmax = np.max(D_orig)
        
        # Original spectrogram
        axes[0].imshow(D_orig, aspect='auto', origin='lower', vmin=vmin, vmax=vmax)
        axes[0].set_title('Ground Truth Spectrogram')
        axes[0].set_ylabel('Frequency bin')
        
        # Reconstructed spectrogram with same scale as ground truth
        axes[1].imshow(D_recon, aspect='auto', origin='lower', vmin=vmin, vmax=vmax)
        axes[1].set_title('Reconstructed Spectrogram')
        axes[1].set_xlabel('Time frame')
        axes[1].set_ylabel('Frequency bin')
        
        plt.tight_layout()
        
        # Log figure to TensorBoard
        self.logger.experiment.add_figure(
            f'spectrogram_comparison/sample_{sample_idx}', 
            fig, 
            global_step=self.current_epoch
        )
        plt.close(fig)
        
        # Also create and log the spectrogram difference (error visualization)
        fig, ax = plt.subplots(figsize=(10, 4))
        
        # Compute error spectrogram (absolute difference)
        D_error = np.abs(D_orig - D_recon)
        
        # Use a different colormap for error
        im = ax.imshow(D_error, aspect='auto', origin='lower', cmap='hot')
        ax.set_title('Spectrogram Error (absolute difference)')
        ax.set_xlabel('Time frame')
        ax.set_ylabel('Frequency bin')
        
        # Add colorbar
        plt.colorbar(im, ax=ax)
        
        plt.tight_layout()
        
        # Log figure to TensorBoard
        self.logger.experiment.add_figure(
            f'spectrogram_error/sample_{sample_idx}', 
            fig, 
            global_step=self.current_epoch
        )
        plt.close(fig)
    
    def _compute_spectrogram(self, audio):
        """
        Compute log-magnitude spectrogram
        
        Args:
            audio: Audio numpy array
        
        Returns:
            spectrogram: Log-magnitude spectrogram
        """
        # Short-time Fourier transform (STFT)
        n_fft = 1024
        hop_length = 256
        
        # Compute STFT
        S = np.abs(np.lib.stride_tricks.sliding_window_view(
            np.pad(audio, (n_fft//2, n_fft//2)), n_fft
        )[::hop_length]).T
        
        # Convert to log-magnitude in dB
        S_db = 20 * np.log10(S + 1e-8)
        
        return S_db
    
    def _log_wavelet_comparison(self, original_coeffs, reconstructed_coeffs):
        """
        Create and log visualizations comparing original and reconstructed wavelet coefficients
        
        Args:
            original_coeffs: Original coefficient dictionary
            reconstructed_coeffs: Reconstructed coefficient dictionary
        """
        # Plot comparison for approximation coefficients
        fig, ax = plt.subplots(figsize=(10, 4))
        
        # Get first sample
        orig_a = original_coeffs['a'][0].cpu().numpy()
        recon_a = reconstructed_coeffs['a'][0].cpu().numpy()
        
        # Plot both
        ax.plot(orig_a, alpha=0.7, label='Ground Truth', color='blue')
        ax.plot(recon_a, alpha=0.7, label='Reconstructed', color='red')
        
        # Add labels and legend
        ax.set_title('Approximation Coefficients')
        ax.legend()
        ax.grid(True)
        
        # Log figure to TensorBoard
        self.logger.experiment.add_figure(
            'wavelet_comparison/approximation', 
            fig, 
            global_step=self.current_epoch
        )
        plt.close(fig)
        
        # Plot comparison for detail coefficients (one level for brevity)
        for level in range(min(3, len(original_coeffs['d']))):  # Plot first 3 levels only
            fig, ax = plt.subplots(figsize=(10, 4))
            
            # Get first sample
            orig_d = original_coeffs['d'][level][0].cpu().numpy()
            recon_d = reconstructed_coeffs['d'][level][0].cpu().numpy()
            
            # Plot both
            ax.plot(orig_d, alpha=0.7, label='Ground Truth', color='blue')
            ax.plot(recon_d, alpha=0.7, label='Reconstructed', color='red')
            
            # Add labels and legend
            ax.set_title(f'Detail Coefficients (Level {level+1})')
            ax.legend()
            ax.grid(True)
            
            # Log figure to TensorBoard
            self.logger.experiment.add_figure(
                f'wavelet_comparison/detail_level_{level+1}', 
                fig, 
                global_step=self.current_epoch
            )
            plt.close(fig)
            
            # Calculate and display accuracy metrics for this level
            # Using coefficient of determination (R²) to quantify match quality
            r2 = self._calculate_r2(orig_d, recon_d)
            mse = np.mean((orig_d - recon_d)**2)
            
            # Log metrics
            self.logger.experiment.add_scalar(
                f'wavelet_metrics/detail_level_{level+1}_r2', 
                r2,
                global_step=self.current_epoch
            )
            
            self.logger.experiment.add_scalar(
                f'wavelet_metrics/detail_level_{level+1}_mse', 
                mse,
                global_step=self.current_epoch
            )
    
    def _calculate_r2(self, y_true, y_pred):
        """
        Calculate coefficient of determination (R²)
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            
        Returns:
            r2: R² score between 0 and 1
        """
        # Total sum of squares
        ss_tot = np.sum((y_true - np.mean(y_true))**2)
        # Residual sum of squares
        ss_res = np.sum((y_true - y_pred)**2)
        
        # Handle edge case where all y_true values are identical
        if ss_tot < 1e-10:
            return 1.0 if ss_res < 1e-10 else 0.0
            
        # Calculate R²
        r2 = 1 - (ss_res / ss_tot)
        # Clip to [0,1] since we're using as accuracy percentage
        return max(0, min(1, r2))
    
    def test_step(self, batch, batch_idx):
        x = batch
        output = self(x)
        
        loss = self.loss_fn(
            output['reconstructed'],
            x,
            output['rec_coeffs'],
            output['coeffs']
        )
        
        # Log test metrics
        self.log('test_loss', loss, on_epoch=True)
        
        metrics = self.metrics(output['reconstructed'], x)
        for name, value in metrics.items():
            self.log(f'test_{name}', value, on_epoch=True)
        
        return loss