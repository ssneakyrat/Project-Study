import os
import torch
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader

class AudioDataset(Dataset):
    """Dataset for audio data with optional conditioning"""
    def __init__(self, 
                 data_path, 
                 sample_rate=16000, 
                 sample_length=16000, 
                 use_conditioning=False,
                 num_conditions=0,
                 dataset_size=None):
        self.data_path = data_path
        self.sample_rate = sample_rate
        self.sample_length = sample_length
        self.use_conditioning = use_conditioning
        self.num_conditions = num_conditions
        self.dataset_size = dataset_size
        
        # Get list of audio files
        self.files = self._get_files()
        
    def _get_files(self):
        """Get list of audio files in data_path"""
        if not os.path.exists(self.data_path):
            # Generate dummy data instead
            return []
        
        files = []
        for root, _, filenames in os.walk(self.data_path):
            for filename in filenames:
                if filename.endswith(('.wav', '.mp3', '.flac')):
                    files.append(os.path.join(root, filename))
        
        return files
    
    def __len__(self):
        if len(self.files) == 0:
            # Return configured size or default for generated data
            return self.dataset_size if self.dataset_size is not None else 1000
        return len(self.files)
    
    def __getitem__(self, idx):
        if len(self.files) == 0:
            # Generate dummy sine waves for testing
            return self._generate_dummy_item(idx)
        
        # In a real implementation, use torchaudio to load files
        # For now, just return dummy item
        return self._generate_dummy_item(idx)
    
    def _generate_dummy_item(self, idx):
        """Generate a dummy sine wave for testing"""
        # Generate random frequency between 100-4000 Hz
        freq = np.random.uniform(100, 4000)
        
        # Generate time points
        t = np.linspace(0, 1, self.sample_length)
        
        # Create sine wave with random frequency
        wave = np.sin(2 * np.pi * freq * t)
        
        # Add some noise
        noise = np.random.normal(0, 0.01, self.sample_length)
        wave += noise
        
        # Convert to torch tensor and reshape to [1, T]
        audio = torch.from_numpy(wave).float().unsqueeze(0)
        
        if self.use_conditioning:
            # Generate a one-hot condition vector
            condition_idx = idx % self.num_conditions
            condition = torch.zeros(self.num_conditions)
            condition[condition_idx] = 1.0
            return audio, condition
        
        return audio

class AudioDataModule(pl.LightningDataModule):
    """PyTorch Lightning data module for audio data"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.batch_size = config['training']['batch_size']
        self.num_workers = config['training']['num_workers']
        
        # Extract data parameters
        self.train_path = config['data']['train_path']
        self.val_path = config['data']['val_path']
        self.sample_rate = config['data']['sample_rate']
        
        # Dataset size parameters
        self.train_size = config['data'].get('train_size', None)
        self.val_size = config['data'].get('val_size', None)
        
        # Optional conditioning parameters
        self.use_conditioning = config.get('data', {}).get('use_conditioning', False)
        self.num_conditions = config.get('data', {}).get('num_conditions', 10)
    
    def setup(self, stage=None):
        """Setup train and validation datasets"""
        self.train_dataset = AudioDataset(
            data_path=self.train_path,
            sample_rate=self.sample_rate,
            sample_length=self.config['model']['input_size'],
            use_conditioning=self.use_conditioning,
            num_conditions=self.num_conditions,
            dataset_size=self.train_size
        )
        
        self.val_dataset = AudioDataset(
            data_path=self.val_path,
            sample_rate=self.sample_rate,
            sample_length=self.config['model']['input_size'],
            use_conditioning=self.use_conditioning,
            num_conditions=self.num_conditions,
            dataset_size=self.val_size
        )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0
        )