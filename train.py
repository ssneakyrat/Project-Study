import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from data_module import AudioDataModule
from model import WSTVocoder
import argparse

def main(args):
    # Set up data module
    data_module = AudioDataModule(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_samples=args.num_samples,
        sample_length=args.sample_length,
        sr=args.sample_rate
    )
    
    # Convert kernel_size and stride to tuples if needed
    kernel_size = args.kernel_size
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    else:
        h, w = kernel_size.split('x')
        kernel_size = (int(h), int(w))
        
    stride = args.stride
    if isinstance(stride, int):
        stride = (stride, stride)
    else:
        h, w = stride.split('x')
        stride = (int(h), int(w))
    
    # Create model with reduced dimensions
    model = WSTVocoder(
        sample_rate=args.sample_rate,
        wst_J=args.wst_J,
        wst_Q=args.wst_Q,
        channels=args.channels,
        latent_dim=args.latent_dim,
        kernel_size=kernel_size,
        stride=stride,
        compression_factor=args.compression_factor,
        learning_rate=args.learning_rate
    )
    
    # Set up logger
    logger = TensorBoardLogger("logs", name="wst_vocoder")
    
    # Set up checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        filename="wst_vocoder-{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        mode="min"
    )
    
    # Set up LR monitor
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    # Memory saving strategies
    torch.backends.cudnn.benchmark = True  # Optimize CUDA operations
    
    if args.accumulate_grad_batches > 1:
        effective_batch_size = args.batch_size * args.accumulate_grad_batches
        print(f"Using gradient accumulation. Effective batch size: {effective_batch_size}")
    
    # Set up trainer with memory optimizations
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor],
        accelerator="auto",
        devices=1,
        gradient_clip_val=1.0,
        precision=16 if args.use_amp else 32,  # Use mixed precision for memory savings
        accumulate_grad_batches=args.accumulate_grad_batches,  # Gradient accumulation
        strategy="auto" if not args.use_ddp else "ddp",  # DDP for memory optimization
    )
    
    # Train
    trainer.fit(model, data_module)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Data args
    parser.add_argument("--batch_size", type=int, default=16)  # Reduced from 16
    parser.add_argument("--num_workers", type=int, default=4)  # Reduced from 4
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--sample_length", type=int, default=16000)  # Reduced from 32000 (1 sec instead of 2)
    parser.add_argument("--sample_rate", type=int, default=16000)
    
    # Model args
    parser.add_argument("--wst_J", type=int, default=6)  # Reduced from 8
    parser.add_argument("--wst_Q", type=int, default=6)  # Reduced from 8
    parser.add_argument("--channels", type=int, nargs="+", default=[32, 64, 128])  # Reduced from [64, 128, 256]
    parser.add_argument("--latent_dim", type=int, default=32)  # Reduced from 64
    parser.add_argument("--kernel_size", default=3, type=int,
                       help="Kernel size (single int or 'HxW' format)")
    parser.add_argument("--stride", default=2, type=int,
                       help="Stride (single int or 'HxW' format)")
    parser.add_argument("--compression_factor", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    
    # Training args
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--use_amp", action="store_true", help="Use mixed precision training")
    parser.add_argument("--use_ddp", action="store_true", help="Use distributed data parallel")
    parser.add_argument("--accumulate_grad_batches", type=int, default=1, 
                       help="Accumulate gradients over N batches")
    
    args = parser.parse_args()
    
    main(args)