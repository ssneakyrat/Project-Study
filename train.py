import os
import argparse
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from model import WaveletAEModel
from data import AudioDataModule

def main(args):
    """Main function to run training or testing"""
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Initialize data module
    data_module = AudioDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        sample_rate=args.sample_rate,
        segment_length=args.segment_length
    )
    
    # Prepare data (generate test data if needed)
    data_module.prepare_data()
    data_module.setup()
    
    # Initialize model
    model = WaveletAEModel(
        segment_length=args.segment_length,
        sample_rate=args.sample_rate,
        encoder_dims=args.encoder_dims,
        learning_rate=args.learning_rate
    )
    
    # Logger
    logger = TensorBoardLogger(args.save_dir, name="wavelet_ae")
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(logger.log_dir, "checkpoints"),
        filename="wavelet_ae-{epoch:02d}-{val_loss:.4f}",
        save_top_k=3,
        monitor="val_loss",
        mode="min"
    )
    
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=args.patience,
        mode="min"
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    # Trainer
    trainer = Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() and args.gpu else "cpu",
        devices=1,
        logger=logger,
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
        precision=16 if args.mixed_precision else 32,
        gradient_clip_val=1.0,
        log_every_n_steps=10
    )
    
    if args.test_only:
        # Run component tests
        from test_components import test_component, test_data_module, test_wavelet_transform, \
            test_complex_encoder, test_complex_decoder, test_full_model
            
        results = {}
        results.update(test_component("DataModule", test_data_module))
        results.update(test_component("WaveletTransform", test_wavelet_transform))
        results.update(test_component("ComplexEncoder", test_complex_encoder))
        results.update(test_component("ComplexDecoder", test_complex_decoder))
        results.update(test_component("FullModel", test_full_model))
        
        # Print summary
        print("\nTest Summary:")
        for component, passed in results.items():
            print(f"{component}: {'✓ PASS' if passed else '✗ FAIL'}")
    else:
        # Train the model
        trainer.fit(model, data_module)
    
    return model, trainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wavelet Audio Encoder-Decoder")
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, default=None, 
                        help="Directory with audio files (if None, generates test data)")
    parser.add_argument("--save_dir", type=str, default="results", 
                        help="Directory to save results")
    parser.add_argument("--batch_size", type=int, default=16, 
                        help="Batch size")
    parser.add_argument("--sample_rate", type=int, default=16000, 
                        help="Audio sample rate")
    parser.add_argument("--segment_length", type=int, default=16000, 
                        help="Audio segment length (~1 second at 16kHz)")
    
    # Model arguments
    parser.add_argument("--encoder_dims", type=int, nargs="+", default=[64, 128, 256], 
                        help="Encoder hidden dimensions")
    
    # Training arguments
    parser.add_argument("--learning_rate", type=float, default=1e-4, 
                        help="Learning rate")
    parser.add_argument("--max_epochs", type=int, default=100, 
                        help="Maximum epochs")
    parser.add_argument("--patience", type=int, default=10, 
                        help="Patience for early stopping")
    parser.add_argument("--mixed_precision", action="store_true", 
                        help="Use mixed precision training")
    parser.add_argument("--gpu", action="store_true", 
                        help="Use GPU for training")
    
    # Mode
    parser.add_argument("--test_only", action="store_true", 
                        help="Run component tests only")
    
    args = parser.parse_args()
    
    model, trainer = main(args)