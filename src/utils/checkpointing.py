"""Checkpointing utilities"""

import torch
from pathlib import Path
from typing import Dict, Any


def save_checkpoint(model, optimizer, epoch: int, metrics: Dict, 
                   path: str, scheduler=None):
    """Save training checkpoint."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
    torch.save(checkpoint, path)


def load_checkpoint(path: str, model, optimizer=None, scheduler=None, 
                   device='cuda') -> Dict[str, Any]:
    """Load training checkpoint."""
    checkpoint = torch.load(path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
    return checkpoint
