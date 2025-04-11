import torch
import numpy as np
import h5py
from torch.utils.data import Dataset, DataLoader

def worker_init_fn(worker_id):
    """Initialize h5py file in each worker process"""
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset  # The dataset copy in this worker process
    
    # Open the file in the worker
    dataset.h5_file = h5py.File(dataset.h5_path, 'r')
    dataset.data = dataset.h5_file['audio']

class AudioDataset(Dataset):
    """Dataset for audio samples stored in HDF5 format"""
    def __init__(self, h5_path, transform=None):
        self.h5_path = h5_path
        self.transform = transform
        
        # For main process, open temporarily to get length
        with h5py.File(h5_path, 'r') as f:
            self.data_len = len(f['audio'])
        
        # These will be set in each worker by worker_init_fn
        self.h5_file = None
        self.data = None
    
    def __len__(self):
        return self.data_len
    
    def __getitem__(self, idx):
        # Check if we're in the main process or a worker
        if self.data is None:
            # Main process or not using workers, open file temporarily
            with h5py.File(self.h5_path, 'r') as h5_file:
                audio = torch.FloatTensor(h5_file['audio'][idx])
        else:
            # Worker process, file is already open
            audio = torch.FloatTensor(self.data[idx])
        
        if self.transform:
            audio = self.transform(audio)
            
        return audio
    
    def __del__(self):
        # Clean up the file handle if it exists
        if hasattr(self, 'h5_file') and self.h5_file is not None:
            self.h5_file.close()

def get_dataloaders(config):
    """Create train and test dataloaders"""
    train_dataset = AudioDataset(f"{config.data_dir}/train.h5")
    test_dataset = AudioDataset(f"{config.data_dir}/test.h5")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
        worker_init_fn=worker_init_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        persistent_workers=True,
        worker_init_fn=worker_init_fn
    )
    
    return train_loader, test_loader