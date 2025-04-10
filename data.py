import os
import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule

class AudioDataset(Dataset):
    def __init__(self, audio_files, sample_rate=16000, segment_length=16000):
        self.audio_files = audio_files
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        waveform, sr = torchaudio.load(audio_path)
        
        # Convert to mono if necessary
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample if necessary
        if sr != self.sample_rate:
            waveform = torchaudio.transforms.Resample(sr, self.sample_rate)(waveform)
            
        # Ensure segment length
        if waveform.shape[1] >= self.segment_length:
            max_start = waveform.shape[1] - self.segment_length
            start = torch.randint(0, max_start + 1, (1,)).item()
            waveform = waveform[:, start:start + self.segment_length]
        else:
            # Pad if too short
            padding = self.segment_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
            
        return waveform

class AudioDataModule(LightningDataModule):
    def __init__(self, data_dir=None, batch_size=16, sample_rate=16000, segment_length=16000):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        
    def prepare_data(self):
        # Generate tiny test dataset if no data_dir is provided
        if self.data_dir is None:
            os.makedirs("tiny_dataset", exist_ok=True)
            # Generate 10 test audio files with sine waves at different frequencies
            for i in range(10):
                freq = 440 * (i + 1)  # A4 × n
                duration = 2  # seconds
                t = torch.linspace(0, duration, int(self.sample_rate * duration))
                waveform = torch.sin(2 * np.pi * freq * t).unsqueeze(0)
                path = f"tiny_dataset/sine_{freq}hz.wav"
                torchaudio.save(path, waveform, self.sample_rate)
            self.data_dir = "tiny_dataset"
    
    def setup(self, stage=None):
        # Find all audio files
        self.audio_files = [
            os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir)
            if f.endswith('.wav') or f.endswith('.mp3')
        ]
        
        # Split into train/val sets (80/20)
        train_size = int(0.8 * len(self.audio_files))
        self.train_files = self.audio_files[:train_size]
        self.val_files = self.audio_files[train_size:]
    
    def train_dataloader(self):
        dataset = AudioDataset(self.train_files, self.sample_rate, self.segment_length)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
    
    def val_dataloader(self):
        dataset = AudioDataset(self.val_files, self.sample_rate, self.segment_length)
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=4)