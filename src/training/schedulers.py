"""Learning Rate Schedulers"""

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from typing import Optional


def create_scheduler(optimizer: torch.optim.Optimizer,
                     scheduler_type: str = 'cosine',
                     num_epochs: int = 100,
                     warmup_epochs: int = 5,
                     min_lr: float = 1e-6):
    """
    Create learning rate scheduler.
    
    Args:
        optimizer: PyTorch optimizer
        scheduler_type: Type of scheduler ('cosine', 'linear', 'step')
        num_epochs: Total training epochs
        warmup_epochs: Number of warmup epochs
        min_lr: Minimum learning rate
        
    Returns:
        Learning rate scheduler
    """
    if scheduler_type == 'cosine':
        # Warmup + cosine annealing
        warmup = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
        cosine = CosineAnnealingLR(optimizer, T_max=num_epochs - warmup_epochs, eta_min=min_lr)
        scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_epochs])
        
    elif scheduler_type == 'linear':
        scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=min_lr, total_iters=num_epochs)
        
    elif scheduler_type == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
        
    return scheduler
