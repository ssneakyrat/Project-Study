import os
import torch
import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

class AudioDataset(Dataset):
    def __init__(self, h5_file, sample_rate=16000, duration=2.0, transform=None):
        """
        Dataset for loading audio samples from H5 file
        Args:
            h5_file: Path to H5 file
            sample_rate: Sample rate of audio
            duration: Duration of audio samples in seconds
            transform: Optional transform to apply to audio samples
        """
        self.h5_file = h5_file
        self.sample_rate = sample_rate
        self.duration = duration
        self.transform = transform
        
        # Calculate sample length
        self.sample_length = int(sample_rate * duration)
        
        # Open H5 file and get number of samples
        with h5py.File(h5_file, 'r') as f:
            self.num_samples = len(f['audio'])
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Open H5 file and get audio sample
        with h5py.File(self.h5_file, 'r') as f:
            audio = f['audio'][idx]
        
        # Convert to torch tensor
        audio = torch.tensor(audio, dtype=torch.float32)
        
        # Ensure correct length
        if audio.shape[0] < self.sample_length:
            # Pad if too short
            audio = torch.nn.functional.pad(audio, (0, self.sample_length - audio.shape[0]))
        elif audio.shape[0] > self.sample_length:
            # Randomly crop if too long
            start = torch.randint(0, audio.shape[0] - self.sample_length, (1,))
            audio = audio[start:start + self.sample_length]
        
        # Apply transform if provided
        if self.transform:
            audio = self.transform(audio)
        
        return audio

class AudioDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=32, sample_rate=16000, duration=2.0, num_workers=4):
        """
        PyTorch Lightning data module for audio data
        Args:
            data_dir: Directory containing H5 files
            batch_size: Batch size
            sample_rate: Sample rate of audio
            duration: Duration of audio samples in seconds
            num_workers: Number of workers for data loading
        """
        super(AudioDataModule, self).__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.sample_rate = sample_rate
        self.duration = duration
        self.num_workers = num_workers
        
    def setup(self, stage=None):
        # Assign datasets for train, val, test
        if stage == 'fit' or stage is None:
            self.train_dataset = AudioDataset(
                h5_file=os.path.join(self.data_dir, 'train.h5'),
                sample_rate=self.sample_rate,
                duration=self.duration
            )
            
            self.val_dataset = AudioDataset(
                h5_file=os.path.join(self.data_dir, 'val.h5'),
                sample_rate=self.sample_rate,
                duration=self.duration
            )
        
        if stage == 'test' or stage is None:
            self.test_dataset = AudioDataset(
                h5_file=os.path.join(self.data_dir, 'test.h5'),
                sample_rate=self.sample_rate,
                duration=self.duration
            )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True
        )