import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

class SignalDataset(Dataset):
    def __init__(self, h5_file, split='train', train_ratio=0.8, log_scale_normalize=True):
        """
        Dataset for loading 2D and 1D signal data from h5py file.
        
        Args:
            h5_file (str): Path to h5py file
            split (str): 'train' or 'val'
            train_ratio (float): Ratio of data to use for training
            log_scale_normalize (bool): Whether to apply log-scale normalization
        """
        self.h5_file = h5_file
        self.split = split
        self.train_ratio = train_ratio
        self.log_scale_normalize = log_scale_normalize
        
        # Open the file to get dataset size and prepare indices
        with h5py.File(self.h5_file, 'r') as f:
            self.total_samples = len(f['input_2d'])
            
            # Create train/val indices
            indices = np.arange(self.total_samples)
            np.random.seed(42)  # For reproducibility
            np.random.shuffle(indices)
            
            train_size = int(self.train_ratio * self.total_samples)
            
            if self.split == 'train':
                self.indices = indices[:train_size]
            else:  # val
                self.indices = indices[train_size:]
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        # Map the index to the shuffled indices
        real_idx = self.indices[idx]
        
        # Open h5 file and get data
        with h5py.File(self.h5_file, 'r') as f:
            # Get input and output signals
            input_2d = f['input_2d'][real_idx]
            output_1d = f['output_1d'][real_idx]
            
            # Convert to tensors and ensure correct shape [height, width]
            input_2d = torch.tensor(input_2d, dtype=torch.float32)
            output_1d = torch.tensor(output_1d, dtype=torch.float32)
            
            # Ensure input is in the expected shape
            if len(input_2d.shape) != 2:
                raise ValueError(f"Expected input_2d to have shape [height, width], got {input_2d.shape}")
            
            # Apply log scale normalization if needed
            if self.log_scale_normalize:
                # Adding small constant to avoid log(0)
                input_2d = torch.log(input_2d + 1e-8)
                output_1d = torch.log(output_1d + 1e-8)
                
                # Re-normalize to [0, 1]
                input_2d = (input_2d - input_2d.min()) / (input_2d.max() - input_2d.min() + 1e-8)
                output_1d = (output_1d - output_1d.min()) / (output_1d.max() - output_1d.min() + 1e-8)
            
        return input_2d, output_1d

class SignalDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.data_file = config['data']['data_file']
        self.batch_size = config['training']['batch_size']
        self.train_val_split = config['data']['train_val_split']
        self.num_workers = config['data']['num_workers']
        self.persistent_workers = config['data']['persistent_workers']
        self.prefetch_factor = config['data']['prefetch_factor']
        self.log_scale_normalize = config['data']['log_scale_normalize']
    
    def setup(self, stage=None):
        # Create the datasets
        self.train_dataset = SignalDataset(
            self.data_file, 
            split='train', 
            train_ratio=self.train_val_split,
            log_scale_normalize=self.log_scale_normalize
        )
        
        self.val_dataset = SignalDataset(
            self.data_file, 
            split='val', 
            train_ratio=self.train_val_split,
            log_scale_normalize=self.log_scale_normalize
        )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
            pin_memory=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
            pin_memory=True
        )