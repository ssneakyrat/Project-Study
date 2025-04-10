#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import h5py
import torch
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from concurrent.futures import ThreadPoolExecutor


class AudioDataset(Dataset):
    def __init__(self, h5_file, transform=None, cache_size=100, preload_workers=4):
        """
        Audio dataset loading from H5 files with multi-threading and caching
        
        Args:
            h5_file (str): Path to H5 file containing audio data
            transform (callable, optional): Optional transform to be applied to audio
            cache_size (int): Number of samples to cache in memory
            preload_workers (int): Number of workers for preloading data
        """
        self.h5_file = h5_file
        self.transform = transform
        self.cache_size = cache_size
        self.preload_workers = preload_workers
        self.cache = {}  # In-memory LRU cache
        
        # Get dataset size
        with h5py.File(self.h5_file, 'r') as f:
            self.length = len(f['audio'])
        
        # Preload a portion of the data if caching is enabled
        if self.cache_size > 0:
            self._preload_data()
    
    def _preload_data(self):
        """Preload a subset of data into memory"""
        indices = list(range(min(self.cache_size, self.length)))
        
        print(f"Preloading {len(indices)} samples into memory...")
        
        with ThreadPoolExecutor(max_workers=self.preload_workers) as executor:
            # Create a dictionary of index -> audio tensor
            results = list(executor.map(self._load_single_item, indices))
            for idx, audio in zip(indices, results):
                self.cache[idx] = audio
        
        print(f"Preloaded {len(self.cache)} samples into memory")
    
    def _load_single_item(self, idx):
        """Load a single audio sample from the H5 file"""
        with h5py.File(self.h5_file, 'r') as f:
            audio = f['audio'][idx]
        return torch.from_numpy(audio).float()
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        # Check if the sample is in cache
        if idx in self.cache:
            audio = self.cache[idx]
        else:
            # Load from disk
            audio = self._load_single_item(idx)
            
            # Update cache (simple LRU implementation)
            if len(self.cache) >= self.cache_size and self.cache_size > 0:
                # Remove the first item (oldest)
                self.cache.pop(next(iter(self.cache)))
            
            # Add to cache if caching is enabled
            if self.cache_size > 0:
                self.cache[idx] = audio
        
        # Apply transform if provided
        if self.transform:
            audio = self.transform(audio)
            
        # Ensure audio is [T] (time only, no batch or channel dimension)
        if audio.dim() > 1:
            audio = audio.squeeze()
            
        return audio


class ThreadedH5Writer:
    """Helper class for efficient multi-threaded h5py writing"""
    def __init__(self, h5_file, dataset_name, shape, dtype=np.float32, chunks=True, compression="gzip", compression_opts=4):
        self.h5_file = h5_file
        self.dataset_name = dataset_name
        self.shape = shape
        self.dtype = dtype
        
        # Create h5 file and dataset
        with h5py.File(self.h5_file, 'w') as f:
            f.create_dataset(
                self.dataset_name, 
                shape=self.shape,
                dtype=self.dtype,
                chunks=chunks,
                compression=compression,
                compression_opts=compression_opts
            )
    
    def write_item(self, item_tuple):
        """Write a single item to the dataset"""
        idx, data = item_tuple
        with h5py.File(self.h5_file, 'r+') as f:
            f[self.dataset_name][idx] = data
    
    def write_batch(self, items):
        """Write multiple items using ThreadPoolExecutor"""
        with ThreadPoolExecutor() as executor:
            executor.map(self.write_item, items)
    
    def write_data(self, data_list):
        """Write a list of data with automatic indexing"""
        items = [(i, data) for i, data in enumerate(data_list)]
        self.write_batch(items)


class AudioDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=16, num_workers=4, 
                 sample_rate=16000, audio_length=2.0, cache_size=100, **kwargs):
        """
        Audio data module for PyTorch Lightning
        
        Args:
            data_dir (str): Directory containing H5 files
            batch_size (int): Batch size for training
            num_workers (int): Number of workers for data loading
            sample_rate (int): Audio sample rate
            audio_length (float): Length of audio in seconds
            cache_size (int): Number of samples to cache in memory
        """
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.sample_rate = sample_rate
        self.audio_length = audio_length
        self.cache_size = cache_size
        
        # Set up file paths
        self.train_file = os.path.join(data_dir, 'train.h5')
        self.val_file = os.path.join(data_dir, 'val.h5')
        self.test_file = os.path.join(data_dir, 'test.h5')
    
    def prepare_data(self):
        """Check if data files exist, otherwise raise error"""
        for file_path in [self.train_file, self.val_file, self.test_file]:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Data file {file_path} not found. Run generate_test_data.py first.")
    
    def setup(self, stage=None):
        """Create datasets for training, validation and testing"""
        if stage == 'fit' or stage is None:
            # Larger cache for training data, smaller for validation
            self.train_dataset = AudioDataset(
                self.train_file, 
                cache_size=self.cache_size,
                preload_workers=self.num_workers
            )
            
            self.val_dataset = AudioDataset(
                self.val_file,
                cache_size=min(self.cache_size, 50),  # Smaller cache for validation
                preload_workers=self.num_workers
            )
        
        if stage == 'test' or stage is None:
            self.test_dataset = AudioDataset(
                self.test_file,
                cache_size=min(self.cache_size, 50),  # Smaller cache for test
                preload_workers=self.num_workers
            )
    
    def train_dataloader(self):
        """Returns the training dataloader with [B, T] shaped tensors"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
            prefetch_factor=2 if self.num_workers > 0 else None
        )
    
    def val_dataloader(self):
        """Returns the validation dataloader with [B, T] shaped tensors"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
            prefetch_factor=2 if self.num_workers > 0 else None
        )
    
    def test_dataloader(self):
        """Returns the test dataloader with [B, T] shaped tensors"""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
            prefetch_factor=2 if self.num_workers > 0 else None
        )