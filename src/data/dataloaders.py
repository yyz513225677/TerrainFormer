"""
DataLoader utilities for TerrainFormer
"""

import torch
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Optional
import numpy as np


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for variable-size point clouds.
    
    Args:
        batch: List of sample dictionaries
        
    Returns:
        Batched dictionary
    """
    collated = {}
    
    for key in batch[0].keys():
        values = [sample[key] for sample in batch]
        
        if isinstance(values[0], torch.Tensor):
            # Stack tensors
            try:
                collated[key] = torch.stack(values)
            except:
                # If sizes don't match, pad
                max_size = max(v.shape[0] for v in values)
                padded = []
                for v in values:
                    if v.shape[0] < max_size:
                        pad_size = max_size - v.shape[0]
                        padding = torch.zeros(pad_size, *v.shape[1:], dtype=v.dtype)
                        v = torch.cat([v, padding])
                    padded.append(v)
                collated[key] = torch.stack(padded)
        else:
            collated[key] = values
            
    return collated


def create_dataloader(dataset: Dataset,
                      batch_size: int = 16,
                      shuffle: bool = True,
                      num_workers: int = 8,
                      pin_memory: bool = True,
                      drop_last: bool = True) -> DataLoader:
    """
    Create a DataLoader with appropriate settings.
    
    Args:
        dataset: PyTorch dataset
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of worker processes
        pin_memory: Pin memory for CUDA
        drop_last: Drop incomplete last batch
        
    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate_fn,
    )


class BalancedSampler(torch.utils.data.Sampler):
    """
    Sampler that balances action distribution.
    """
    
    def __init__(self, dataset: Dataset, num_samples: Optional[int] = None):
        self.dataset = dataset
        self.num_samples = num_samples or len(dataset)
        
        # Get action distribution
        actions = [dataset.samples[i].get('action', 0) for i in range(len(dataset))]
        self.actions = np.array(actions)
        
        # Compute weights
        unique, counts = np.unique(self.actions, return_counts=True)
        weights = 1.0 / counts
        self.sample_weights = weights[self.actions]
        self.sample_weights /= self.sample_weights.sum()
        
    def __iter__(self):
        indices = np.random.choice(
            len(self.dataset),
            size=self.num_samples,
            replace=True,
            p=self.sample_weights
        )
        return iter(indices.tolist())
    
    def __len__(self):
        return self.num_samples


def create_balanced_dataloader(dataset: Dataset,
                               batch_size: int = 16,
                               num_workers: int = 8,
                               pin_memory: bool = True) -> DataLoader:
    """Create dataloader with balanced action sampling."""
    sampler = BalancedSampler(dataset)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )
