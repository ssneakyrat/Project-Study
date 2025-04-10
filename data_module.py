import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import numpy as np

class RandomAudioDataset(Dataset):
    def __init__(self, num_samples=1000, sample_length=16000, sr=16000):
        """Generate random audio samples for PoC testing.
        
        Args:
            num_samples: Number of samples in the dataset
            sample_length: Length of each audio sample
            sr: Sampling rate
        """
        self.num_samples = num_samples
        self.sample_length = sample_length
        self.sr = sr
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Generate random audio as a mixture of sine waves
        t = np.arange(self.sample_length) / self.sr
        # Generate a random fundamental frequency between 80 and 800 Hz
        f0 = np.random.uniform(80, 800)
        # Generate a mixture of harmonics
        audio = np.zeros(self.sample_length)
        for i in range(1, 6):  # 5 harmonics
            # Each harmonic has decreasing amplitude
            amplitude = 1.0 / i
            # Add some random phase
            phase = np.random.uniform(0, 2 * np.pi)
            audio += amplitude * np.sin(2 * np.pi * i * f0 * t + phase)
        
        # Normalize the audio
        audio = audio / np.max(np.abs(audio))
        
        # Add some noise
        noise = np.random.normal(0, 0.01, self.sample_length)
        audio = audio + noise
        
        # Convert to tensor
        audio_tensor = torch.from_numpy(audio).float()
        
        return {"audio": audio_tensor}

class AudioDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=16, num_workers=4, num_samples=1000, sample_length=16000, sr=16000):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_samples = num_samples
        self.sample_length = sample_length
        self.sr = sr
        
    def setup(self, stage=None):
        self.train_dataset = RandomAudioDataset(
            num_samples=self.num_samples, 
            sample_length=self.sample_length,
            sr=self.sr
        )
        self.val_dataset = RandomAudioDataset(
            num_samples=self.num_samples // 10,  # Smaller validation set
            sample_length=self.sample_length,
            sr=self.sr
        )
        
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True
        )