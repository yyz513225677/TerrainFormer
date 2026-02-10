"""
World Model Trainer

Phase 1: Self-supervised pretraining of world model.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional
from tqdm import tqdm
import os
import sys
from pathlib import Path

# Import BEV projection
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils.bev_projection_fast import UltraFastBEV


class WorldModelTrainer:
    """Trainer for world model pretraining."""
    
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 optimizer: torch.optim.Optimizer,
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
                 loss_fn: nn.Module,
                 device: str = 'cuda',
                 output_dir: str = 'outputs',
                 fp16: bool = True):
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.device = device
        self.output_dir = output_dir
        self.fp16 = fp16
        
        self.scaler = torch.cuda.amp.GradScaler() if fp16 else None
        os.makedirs(output_dir, exist_ok=True)

        # Create BEV projection module
        self.bev_projection = UltraFastBEV(
            x_range=(-50, 50),
            y_range=(-50, 50),
            z_range=(-3, 5),
            bev_size=256,
            num_features=64
        ).to(device)

        self.best_val_loss = float('inf')
        self.epoch = 0
        
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        loss_components = {}
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.epoch}')
        for batch in pbar:
            # Move to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            self.optimizer.zero_grad()
            
            with torch.amp.autocast('cuda', enabled=self.fp16):
                # Forward pass through encoder + world model
                # In full implementation, this would process through the actual model
                outputs = self._forward_pass(batch)
                loss, components = self.loss_fn(outputs, batch)
                
            if self.fp16:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
                
            total_loss += loss.item()
            for k, v in components.items():
                loss_components[k] = loss_components.get(k, 0) + v
                
            pbar.set_postfix({'loss': loss.item()})
            
        n_batches = len(self.train_loader)
        metrics = {'train/loss': total_loss / n_batches}
        for k, v in loss_components.items():
            metrics[f'train/{k}'] = v / n_batches
            
        return metrics
    
    def validate(self) -> Dict[str, float]:
        """Validate model."""
        self.model.eval()
        total_loss = 0
        loss_components = {}
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                outputs = self._forward_pass(batch)
                loss, components = self.loss_fn(outputs, batch)
                
                total_loss += loss.item()
                for k, v in components.items():
                    loss_components[k] = loss_components.get(k, 0) + v
                    
        n_batches = len(self.val_loader)
        metrics = {'val/loss': total_loss / n_batches}
        for k, v in loss_components.items():
            metrics[f'val/{k}'] = v / n_batches
            
        return metrics
    
    def _forward_pass(self, batch: Dict) -> Dict:
        """Forward pass through model."""
        points = batch['point_cloud']  # (B, N, 4) [x, y, z, intensity]

        # Proper BEV projection from point cloud
        bev = self.bev_projection(points)  # (B, 64, 256, 256)

        # World model forward
        outputs = self.model(bev, return_latent=True)
        return outputs
    
    def train(self, num_epochs: int):
        """Full training loop."""
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            train_metrics = self.train_epoch()
            val_metrics = self.validate()
            
            if self.scheduler:
                self.scheduler.step()
                
            # Logging
            all_metrics = {**train_metrics, **val_metrics}
            print(f"Epoch {epoch}: " + ", ".join(f"{k}={v:.4f}" for k, v in all_metrics.items()))
            
            # Save best model only (delete old to save space)
            if val_metrics['val/loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['val/loss']
                self.save_checkpoint('best_model.pt')
                
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        path = os.path.join(self.output_dir, filename)
        torch.save({
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
        }, path)
        print(f"Saved checkpoint to {path}")
        
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        print(f"Loaded checkpoint from {path}")
