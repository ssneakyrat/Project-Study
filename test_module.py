import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from tqdm import tqdm

# Import model components
from models.dwt import DWT, IDWT
from models.wem import WaveletEchoMatrix, UnifiedEncoder, UnifiedDecoder, ResidualBlock
from models.loss import WaveletMSELoss, TimeDomainMSELoss, CombinedLoss
from models.metric import SignalToNoiseRatio, SpectralConvergence, LogSpectralDistance, MultiMetric

# Import utilities
from utils.utils import load_config, set_seed

class ModelTester:
    def __init__(self, config_path="config/model.yaml"):
        # Load configuration
        self.config = load_config(config_path)
        set_seed(self.config["seed"])
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Model hyperparameters
        self.wavelet = self.config["model"]["wavelet"]
        self.levels = self.config["model"]["levels"]
        self.hidden_dim = self.config["model"]["hidden_dim"]
        self.latent_dim = self.config["model"]["latent_dim"]
        
        # Test hyperparameters
        self.sample_rate = self.config["data"]["sample_rate"]
        self.duration = self.config["data"]["duration"]
        self.batch_size = 4  # Small batch for testing
        
        # Create test data
        self.sample_length = int(self.sample_rate * self.duration)
        self.test_data = self._create_test_data()
        
    def _create_test_data(self):
        """Create synthetic test data"""
        print("Creating test data...")
        # Create synthetic sinusoidal data with multiple frequencies
        t = torch.linspace(0, self.duration, self.sample_length)
        
        # Create batch of synthetic signals with varying frequencies
        signals = []
        for i in range(self.batch_size):
            # Mix of frequencies to create complex signals
            f1 = 440 * (1 + i * 0.2)  # Base frequency (A4) with variations
            f2 = 880 * (1 + i * 0.1)  # Octave up
            f3 = 220 * (1 + i * 0.15)  # Octave down
            
            # Create signal with decay
            signal = 0.5 * torch.sin(2 * np.pi * f1 * t) + \
                     0.3 * torch.sin(2 * np.pi * f2 * t) + \
                     0.2 * torch.sin(2 * np.pi * f3 * t)
            
            # Add some noise
            noise = 0.05 * torch.randn_like(signal)
            signal = signal + noise
            
            # Normalize
            signal = signal / signal.abs().max()
            
            signals.append(signal)
        
        # Stack to create batch
        signals = torch.stack(signals)
        return signals.to(self.device)
    
    def test_dwt_idwt(self):
        """Test DWT and IDWT components"""
        print("\n===== Testing DWT and IDWT =====")
        
        # Initialize DWT and IDWT
        dwt = DWT(wave=self.wavelet, level=self.levels).to(self.device)
        idwt = IDWT(wave=self.wavelet).to(self.device)
        
        # Forward DWT
        print("Applying DWT...")
        coeffs = dwt(self.test_data)
        
        # Check coefficient shapes
        print(f"Approximation coefficients shape: {coeffs['a'].shape}")
        for i, d in enumerate(coeffs['d']):
            print(f"Detail coefficients level {i+1} shape: {d.shape}")
        
        # Test coefficient normalization
        print("\nTesting coefficient normalization...")
        norm_coeffs, stats = dwt.normalize_coeffs(coeffs)
        
        # Check for NaNs in normalized coefficients
        a_has_nan = torch.isnan(norm_coeffs['a']).any().item()
        d_has_nan = any(torch.isnan(d).any().item() for d in norm_coeffs['d'])
        
        if a_has_nan or d_has_nan:
            print("❌ WARNING: NaN values detected in normalized coefficients!")
        else:
            print("✅ Normalized coefficients are valid (no NaNs).")
            
        # Check normalization stats
        print(f"Approximation mean range: [{stats['means'][0].min().item():.4f}, {stats['means'][0].max().item():.4f}]")
        print(f"Approximation std range: [{stats['stds'][0].min().item():.4f}, {stats['stds'][0].max().item():.4f}]")
        
        # Test thresholding
        print("\nTesting coefficient thresholding...")
        thresh_coeffs = dwt.threshold_coeffs(norm_coeffs)
        
        # Calculate sparsity after thresholding
        a_sparsity = (thresh_coeffs['a'] == 0).float().mean().item() * 100
        d_sparsity = [(d == 0).float().mean().item() * 100 for d in thresh_coeffs['d']]
        
        print(f"Approximation coefficient sparsity: {a_sparsity:.2f}%")
        for i, sp in enumerate(d_sparsity):
            print(f"Detail coefficient level {i+1} sparsity: {sp:.2f}%")
            
        # Reconstruct with IDWT
        print("\nApplying IDWT for reconstruction...")
        reconstructed = idwt(coeffs)
        
        # Check reconstruction shape
        print(f"Original data shape: {self.test_data.shape}")
        print(f"Reconstructed data shape: {reconstructed.shape}")
        
        # Calculate reconstruction error
        mse = torch.mean((self.test_data - reconstructed) ** 2).item()
        psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float('inf')
        
        print(f"Reconstruction MSE: {mse:.6f}")
        print(f"Reconstruction PSNR: {psnr:.2f} dB")
        
        if psnr > 40:
            print("✅ DWT-IDWT reconstruction is excellent (PSNR > 40dB)")
        elif psnr > 30:
            print("✅ DWT-IDWT reconstruction is good (PSNR > 30dB)")
        else:
            print("❌ DWT-IDWT reconstruction is poor (PSNR < 30dB)")
        
        return {
            "coeffs": coeffs,
            "norm_coeffs": norm_coeffs,
            "thresh_coeffs": thresh_coeffs,
            "reconstructed": reconstructed,
            "mse": mse,
            "psnr": psnr
        }
    
    def test_encoder(self, coeffs=None):
        """Test the Unified Encoder"""
        print("\n===== Testing Unified Encoder =====")
        
        if coeffs is None:
            # Get coefficients from DWT
            dwt = DWT(wave=self.wavelet, level=self.levels).to(self.device)
            coeffs = dwt(self.test_data)
            _, stats = dwt.normalize_coeffs(coeffs)
            coeffs = dwt.threshold_coeffs(coeffs)
        
        # Initialize encoder
        encoder = UnifiedEncoder(levels=self.levels, hidden_dim=self.hidden_dim).to(self.device)
        
        # Forward pass
        print("Encoding coefficients to latent space...")
        z, encoded_features = encoder(coeffs)
        
        # Check latent shape
        print(f"Latent space shape: {z.shape}")
        expected_latent_dim = 256
        
        if z.shape[1] == expected_latent_dim:
            print(f"✅ Latent dimension is correct ({z.shape[1]} = {expected_latent_dim})")
        else:
            print(f"❌ Latent dimension mismatch: {z.shape[1]} != {expected_latent_dim}")
        
        # Check for NaNs in latent space
        has_nan = torch.isnan(z).any().item()
        if has_nan:
            print("❌ WARNING: NaN values detected in latent space!")
        else:
            print("✅ Latent space values are valid (no NaNs).")
        
        # Check latent space statistics
        z_mean = z.mean().item()
        z_std = z.std().item()
        z_min = z.min().item()
        z_max = z.max().item()
        
        print(f"Latent space statistics: mean={z_mean:.4f}, std={z_std:.4f}, min={z_min:.4f}, max={z_max:.4f}")
        
        # Check for dead neurons
        dead_neurons = (z.abs().sum(dim=0) < 1e-6).sum().item()
        dead_percentage = dead_neurons / z.shape[1] * 100
        
        if dead_neurons > 0:
            print(f"❌ {dead_neurons} dead neurons detected ({dead_percentage:.2f}% of latent space)")
        else:
            print("✅ No dead neurons detected in latent space.")
            
        # Check encoded features for skip connections
        print(f"\nVerifying encoded features for skip connections:")
        for i, feat in enumerate(encoded_features):
            print(f"  Feature {i} shape: {feat.shape}")
            if torch.isnan(feat).any().item():
                print(f"  ❌ NaNs detected in feature {i}")
            else:
                print(f"  ✅ Feature {i} is valid (no NaNs)")
        
        return {
            "latent": z,
            "encoded_features": encoded_features
        }
    
    def test_decoder(self, coeffs=None, latent=None, encoded_features=None):
        """Test the Unified Decoder"""
        print("\n===== Testing Unified Decoder =====")
        
        if coeffs is None or latent is None or encoded_features is None:
            # Get coefficients and latent from previous steps
            dwt = DWT(wave=self.wavelet, level=self.levels).to(self.device)
            coeffs = dwt(self.test_data)
            encoder = UnifiedEncoder(levels=self.levels, hidden_dim=self.hidden_dim).to(self.device)
            latent, encoded_features = encoder(coeffs)
        
        # Original shapes for reconstruction
        orig_shapes = {
            'a': coeffs['a'].shape,
            'd': [d.shape for d in coeffs['d']]
        }
        
        # Initialize decoder
        decoder = UnifiedDecoder(
            levels=self.levels, 
            hidden_dim=self.hidden_dim, 
            latent_dim=latent.shape[1]
        ).to(self.device)
        
        # Forward pass
        print("Decoding latent space to coefficients...")
        rec_coeffs = decoder(latent, encoded_features, orig_shapes)
        
        # Check coefficient shapes
        print(f"Original approximation shape: {coeffs['a'].shape}")
        print(f"Reconstructed approximation shape: {rec_coeffs['a'].shape}")
        
        shape_mismatch = False
        
        if rec_coeffs['a'].shape != coeffs['a'].shape:
            print(f"❌ Approximation coefficient shape mismatch!")
            shape_mismatch = True
        
        for i, (orig_d, rec_d) in enumerate(zip(coeffs['d'], rec_coeffs['d'])):
            print(f"Original detail level {i+1} shape: {orig_d.shape}")
            print(f"Reconstructed detail level {i+1} shape: {rec_d.shape}")
            
            if rec_d.shape != orig_d.shape:
                print(f"❌ Detail coefficient shape mismatch at level {i+1}!")
                shape_mismatch = True
        
        if not shape_mismatch:
            print("✅ All coefficient shapes match between original and reconstructed.")
        
        # Check for NaNs
        a_has_nan = torch.isnan(rec_coeffs['a']).any().item()
        d_has_nan = any(torch.isnan(d).any().item() for d in rec_coeffs['d'])
        
        if a_has_nan or d_has_nan:
            print("❌ WARNING: NaN values detected in reconstructed coefficients!")
        else:
            print("✅ Reconstructed coefficients are valid (no NaNs).")
        
        # Calculate reconstruction error for coefficients
        a_mse = torch.mean((coeffs['a'] - rec_coeffs['a']) ** 2).item()
        d_mse = [torch.mean((d1 - d2) ** 2).item() for d1, d2 in zip(coeffs['d'], rec_coeffs['d'])]
        
        print(f"Approximation coefficient MSE: {a_mse:.6f}")
        for i, mse in enumerate(d_mse):
            print(f"Detail coefficient level {i+1} MSE: {mse:.6f}")
        
        return {
            "rec_coeffs": rec_coeffs,
            "a_mse": a_mse,
            "d_mse": d_mse
        }
    
    def test_residual_block(self):
        """Test ResidualBlock functionality"""
        print("\n===== Testing ResidualBlock Component =====")
        
        # Create a test input
        batch_size = self.test_data.shape[0]
        channels = 16
        seq_len = 100
        test_input = torch.randn(batch_size, channels, seq_len).to(self.device)
        
        # Create residual block
        res_block = ResidualBlock(channels).to(self.device)
        
        # Forward pass
        output = res_block(test_input)
        
        # Check output shape
        if output.shape == test_input.shape:
            print("✅ ResidualBlock preserves input shape correctly")
        else:
            print(f"❌ ResidualBlock shape mismatch: {output.shape} != {test_input.shape}")
        
        # Check for NaNs
        if torch.isnan(output).any().item():
            print("❌ WARNING: NaN values detected in ResidualBlock output!")
        else:
            print("✅ ResidualBlock output is valid (no NaNs).")
        
        # Check if residual connections are working (output should not be identical to input)
        diff = torch.mean(torch.abs(output - test_input)).item()
        if diff < 1e-6:
            print("❌ WARNING: Output is nearly identical to input, residual connection may not be working!")
        else:
            print(f"✅ ResidualBlock is modifying the input (mean abs diff: {diff:.6f})")
        
        # Test gradient flow through residual block
        test_input.requires_grad_(True)
        output = res_block(test_input)
        loss = output.mean()
        loss.backward()
        
        if test_input.grad is None or torch.all(test_input.grad == 0):
            print("❌ WARNING: No gradient flowing through ResidualBlock!")
        else:
            mean_grad = torch.mean(torch.abs(test_input.grad)).item()
            print(f"✅ Gradient is flowing through ResidualBlock (mean abs grad: {mean_grad:.6f})")
        
        return {"output": output, "input": test_input}
    
    def test_full_model(self):
        """Test the full WaveletEchoMatrix model"""
        print("\n===== Testing Full WaveletEchoMatrix Model =====")
        
        # Initialize model
        model = WaveletEchoMatrix(
            wavelet=self.wavelet,
            levels=self.levels,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim
        ).to(self.device)
        
        # Check parameter count
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        # Forward pass
        print("Running full model forward pass...")
        output = model(self.test_data)
        
        # Check output keys
        expected_keys = ["reconstructed", "latent", "coeffs", "rec_coeffs"]
        all_keys_present = all(k in output for k in expected_keys)
        
        if all_keys_present:
            print("✅ All expected output keys are present.")
        else:
            missing_keys = [k for k in expected_keys if k not in output]
            print(f"❌ Missing output keys: {missing_keys}")
        
        # Check reconstruction shape
        if output["reconstructed"].shape == self.test_data.shape:
            print("✅ Reconstruction shape matches input shape.")
        else:
            print(f"❌ Reconstruction shape mismatch: {output['reconstructed'].shape} != {self.test_data.shape}")
        
        # Check for NaNs in reconstruction
        has_nan = torch.isnan(output["reconstructed"]).any().item()
        if has_nan:
            print("❌ WARNING: NaN values detected in final reconstruction!")
        else:
            print("✅ Final reconstruction is valid (no NaNs).")
        
        # Calculate reconstruction metrics
        mse = torch.mean((self.test_data - output["reconstructed"]) ** 2).item()
        psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float('inf')
        
        # Compute additional metrics
        snr_metric = SignalToNoiseRatio().to(self.device)
        sc_metric = SpectralConvergence().to(self.device)
        lsd_metric = LogSpectralDistance().to(self.device)
        
        snr = snr_metric(output["reconstructed"], self.test_data).item()
        sc = sc_metric(output["reconstructed"], self.test_data).item()
        lsd = lsd_metric(output["reconstructed"], self.test_data).item()
        
        print(f"Reconstruction MSE: {mse:.6f}")
        print(f"Reconstruction PSNR: {psnr:.2f} dB")
        print(f"Signal-to-Noise Ratio: {snr:.2f} dB")
        print(f"Spectral Convergence: {sc:.6f}")
        print(f"Log Spectral Distance: {lsd:.6f}")
        
        if psnr > 20:
            print("✅ Model reconstruction is good (PSNR > 20dB)")
        elif psnr > 10:
            print("⚠️ Model reconstruction is mediocre (PSNR > 10dB)")
        else:
            print("❌ Model reconstruction is poor (PSNR < 10dB)")
        
        return {
            "output": output,
            "mse": mse,
            "psnr": psnr,
            "snr": snr,
            "sc": sc,
            "lsd": lsd
        }
    
    def test_gradient_flow(self):
        """Test gradient flow through the model"""
        print("\n===== Testing Gradient Flow =====")
        
        # Initialize model and optimizer
        model = WaveletEchoMatrix(
            wavelet=self.wavelet,
            levels=self.levels,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim
        ).to(self.device)
        
        # Set model to train mode
        model.train()
        
        # Initialize loss
        loss_fn = CombinedLoss(
            wavelet_weight=self.config["loss"]["wavelet_loss_weight"],
            time_weight=self.config["loss"]["time_loss_weight"],
            levels=self.levels,
            alpha=self.config["loss"]["loss_alpha"]
        ).to(self.device)
        
        # Initialize optimizer
        optimizer = torch.optim.AdamW(  # Changed from Adam to AdamW
            model.parameters(),
            lr=self.config["training"]["learning_rate"],
            weight_decay=self.config["training"]["weight_decay"]
        )
        
        # Store initial parameters
        initial_params = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                initial_params[name] = param.clone().detach()
        
        # Training loop (just a few iterations for testing)
        print("Testing gradient flow with 5 iterations...")
        losses = []
        
        for i in range(5):
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            output = model(self.test_data)
            
            # Calculate loss
            loss = loss_fn(
                output["reconstructed"],
                self.test_data,
                output["rec_coeffs"],
                output["coeffs"]
            )
            
            # Backward pass
            loss.backward()
            
            # Optimizer step
            optimizer.step()
            
            # Store loss
            losses.append(loss.item())
            print(f"Iteration {i+1}, Loss: {loss.item():.6f}")
        
        # Check if loss decreased
        if losses[-1] < losses[0]:
            print(f"✅ Loss decreased from {losses[0]:.6f} to {losses[-1]:.6f}")
        else:
            print(f"❌ Loss did not decrease! Initial: {losses[0]:.6f}, Final: {losses[-1]:.6f}")
        
        # Check parameter updates
        all_unchanged = True
        unchanged_count = 0
        total_params = 0
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                total_params += 1
                # Check if parameter changed
                is_unchanged = torch.allclose(param, initial_params[name], rtol=1e-4, atol=1e-4)
                
                if is_unchanged:
                    unchanged_count += 1
                    all_unchanged = all_unchanged and is_unchanged
        
        if all_unchanged:
            print("❌ CRITICAL: All parameters remained unchanged after training!")
        else:
            unchanged_percentage = (unchanged_count / total_params) * 100
            print(f"📊 {unchanged_count} out of {total_params} parameters unchanged ({unchanged_percentage:.2f}%)")
            
            if unchanged_percentage > 50:
                print("❌ WARNING: Over 50% of parameters did not change during training!")
            elif unchanged_percentage > 20:
                print("⚠️ Significant portion of parameters did not change during training.")
            else:
                print("✅ Most parameters were updated during training.")
        
        # Check for vanishing/exploding gradients
        has_vanishing_grads = False
        has_exploding_grads = False
        
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad_mean = param.grad.abs().mean().item()
                
                if grad_mean < 1e-7:
                    has_vanishing_grads = True
                    print(f"❌ Possible vanishing gradient in {name}: mean absolute gradient = {grad_mean:.10f}")
                
                if grad_mean > 1e2:
                    has_exploding_grads = True
                    print(f"❌ Possible exploding gradient in {name}: mean absolute gradient = {grad_mean:.6f}")
        
        if not has_vanishing_grads and not has_exploding_grads:
            print("✅ No obvious vanishing or exploding gradients detected.")
        
        return {
            "losses": losses,
            "unchanged_percentage": (unchanged_count / total_params) * 100
        }
    
    def analyze_model_issues(self):
        """Analyze potential issues with the model learning"""
        print("\n===== Model Learning Analysis =====")
        
        # First test the residual block specifically
        self.test_residual_block()
        
        # Run all tests
        dwt_results = self.test_dwt_idwt()
        encoder_results = self.test_encoder(dwt_results["thresh_coeffs"])
        decoder_results = self.test_decoder(
            dwt_results["thresh_coeffs"], 
            encoder_results["latent"],
            encoder_results["encoded_features"]
        )
        full_model_results = self.test_full_model()
        gradient_results = self.test_gradient_flow()
        
        # Combined analysis
        print("\n===== Combined Analysis =====")
        
        # Issue 1: DWT/IDWT Reconstruction
        if dwt_results["psnr"] < 30:
            print("⚠️ The DWT/IDWT reconstruction has high error, which could limit overall model performance.")
        
        # Issue 2: Coefficient shapes and levels
        level_issues = False
        for i, d in enumerate(dwt_results["coeffs"]["d"]):
            if d.shape[1] < 100:  # Very small detail coefficient arrays
                print(f"⚠️ Level {i+1} detail coefficients are very small ({d.shape[1]} coefficients), may not capture sufficient detail.")
                level_issues = True
        
        if level_issues:
            print("   Consider reducing the number of wavelet levels or using a different wavelet.")
        
        # Issue 3: Latent space
        if encoder_results["latent"].std().item() < 0.1:
            print("⚠️ Latent space has low variance, indicating potential information bottleneck.")
        
        # Issue 4: Decoder coefficient reconstruction
        avg_d_mse = sum(decoder_results["d_mse"]) / len(decoder_results["d_mse"])
        if avg_d_mse > 0.1:
            print("⚠️ Unified decoder has high error in coefficient reconstruction.")
        
        # Issue 5: Gradient flow
        if gradient_results["unchanged_percentage"] > 20:
            print("⚠️ Significant portion of parameters not updating during training.")
            print("   Possible causes:")
            print("   - Learning rate too small")
            print("   - Gradient not flowing through certain parts of network")
            print("   - Model architecture bottlenecks")
        
        # Issue 6: Full model metrics
        if full_model_results["psnr"] < 15:
            print("⚠️ Full model reconstruction quality is poor.")
        
        # Overall assessment
        print("\n===== Overall Assessment =====")
        major_issues = []
        
        if dwt_results["psnr"] < 25:
            major_issues.append("DWT/IDWT reconstruction errors")
        
        if gradient_results["unchanged_percentage"] > 40:
            major_issues.append("Parameter update issues")
        
        if full_model_results["psnr"] < 10:
            major_issues.append("Very poor reconstruction quality")
        
        if has_vanishing_grads := gradient_results.get("has_vanishing_grads", False):
            major_issues.append("Vanishing gradients")
        
        if len(major_issues) > 0:
            print("❌ Major issues detected:")
            for issue in major_issues:
                print(f"   - {issue}")
        else:
            print("✅ No major issues detected. If the model is not learning well, consider:")
            print("   - Increasing model capacity (larger hidden_dim)")
            print("   - Adjusting learning rate")
            print("   - Using a different wavelet type")
            print("   - Adding more training data")
            print("   - Adjusting loss function weights")
        
        # Recommendations
        print("\n===== Recommendations =====")
        print("1. Model Architecture:")
        #print(f"   - Current parameter count: ~{total_params:,} parameters")
        
        if gradient_results["unchanged_percentage"] > 30:
            print("   - Consider adjusting residual connections for better gradient flow")
            print("   - Add more BatchNorm layers or try LayerNorm instead")
        else:
            print("   - Architecture seems to be facilitating good gradient flow")
        
        print("\n2. Training Process:")
        print("   - Test with smaller batch sizes for more frequent updates")
        print("   - Consider learning rate schedule with warm-up")
        
        print("\n3. Wavelet Configuration:")
        print(f"   - Current wavelet: {self.wavelet}, Levels: {self.levels}")
        
        if dwt_results["psnr"] < 35:
            print("   - Try different wavelet types (e.g., 'haar', 'sym4', 'coif3')")
        
        print("\n4. Loss Function:")
        print(f"   - Current weights: Wavelet = {self.config['loss']['wavelet_loss_weight']}, Time = {self.config['loss']['time_loss_weight']}")
        print("   - Experiment with different loss function weights")
        
        print("\nFinal Note: Remember the model is designed for compression efficiency (125:1 ratio),")
        print("which naturally limits reconstruction quality. Trade-offs are expected.")


def main():
    tester = ModelTester()
    tester.analyze_model_issues()


if __name__ == "__main__":
    main()