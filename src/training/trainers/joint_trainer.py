"""
Joint Trainer

Phase 3: Joint training of world model and decision transformer.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional
from tqdm import tqdm
import os


class JointTrainer:
    """Trainer for joint world model + decision transformer training."""

    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 optimizer: torch.optim.Optimizer,
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
                 world_loss_fn: nn.Module,
                 decision_loss_fn: nn.Module,
                 device: str = 'cuda',
                 output_dir: str = 'outputs',
                 fp16: bool = True,
                 world_loss_weight: float = 0.3,
                 decision_loss_weight: float = 1.0):

        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.world_loss_fn = world_loss_fn
        self.decision_loss_fn = decision_loss_fn
        self.device = device
        self.output_dir = output_dir
        self.fp16 = fp16
        self.world_loss_weight = world_loss_weight
        self.decision_loss_weight = decision_loss_weight

        self.scaler = torch.cuda.amp.GradScaler() if fp16 else None
        os.makedirs(output_dir, exist_ok=True)

        self.best_score = -float('inf')
        self.epoch = 0

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        metrics = {
            'total_loss': 0,
            'world_loss': 0,
            'decision_loss': 0,
            'accuracy': 0,
        }
        total_samples = 0

        pbar = tqdm(self.train_loader, desc=f'Epoch {self.epoch}')
        for batch in pbar:
            # Move to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            self.optimizer.zero_grad()

            with torch.amp.autocast('cuda', enabled=self.fp16):
                # Forward pass
                outputs = self.model(
                    points=batch['point_cloud'],
                    state=batch['vehicle_state'],
                    goal=batch['goal_direction'],
                    action_history=batch['action_sequence'],
                    return_world_predictions=True
                )

                # Compute losses
                world_loss, _ = self.world_loss_fn(outputs, batch)
                decision_loss = self.decision_loss_fn(outputs['action_logits'], batch['expert_action'])

                loss = self.world_loss_weight * world_loss + self.decision_loss_weight * decision_loss

            if self.fp16:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            B = batch['point_cloud'].shape[0]
            metrics['total_loss'] += loss.item() * B
            metrics['world_loss'] += world_loss.item() * B
            metrics['decision_loss'] += decision_loss.item() * B

            preds = outputs['action_logits'].argmax(dim=-1)
            metrics['accuracy'] += (preds == batch['expert_action']).sum().item()
            total_samples += B

            pbar.set_postfix({'loss': loss.item(), 'acc': metrics['accuracy'] / total_samples})

        for k in metrics:
            metrics[k] /= total_samples

        return {f'train/{k}': v for k, v in metrics.items()}

    def validate(self) -> Dict[str, float]:
        """Validate model."""
        self.model.eval()

        metrics = {
            'total_loss': 0,
            'world_loss': 0,
            'decision_loss': 0,
            'accuracy': 0,
        }
        total_samples = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}

                outputs = self.model(
                    points=batch['point_cloud'],
                    state=batch['vehicle_state'],
                    goal=batch['goal_direction'],
                    action_history=batch['action_sequence'],
                    return_world_predictions=True
                )

                world_loss, _ = self.world_loss_fn(outputs, batch)
                decision_loss = self.decision_loss_fn(outputs['action_logits'], batch['expert_action'])

                loss = self.world_loss_weight * world_loss + self.decision_loss_weight * decision_loss

                B = batch['point_cloud'].shape[0]
                metrics['total_loss'] += loss.item() * B
                metrics['world_loss'] += world_loss.item() * B
                metrics['decision_loss'] += decision_loss.item() * B

                preds = outputs['action_logits'].argmax(dim=-1)
                metrics['accuracy'] += (preds == batch['expert_action']).sum().item()
                total_samples += B

        for k in metrics:
            metrics[k] /= total_samples

        return {f'val/{k}': v for k, v in metrics.items()}

    def train(self, num_epochs: int):
        """Full training loop."""
        for epoch in range(num_epochs):
            self.epoch = epoch

            train_metrics = self.train_epoch()
            val_metrics = self.validate()

            if self.scheduler:
                self.scheduler.step()

            all_metrics = {**train_metrics, **val_metrics}
            print(f"Epoch {epoch}: " + ", ".join(f"{k}={v:.4f}" for k, v in list(all_metrics.items())[:6]))

            # Combined score
            score = val_metrics['val/accuracy'] - 0.1 * val_metrics['val/total_loss']

            if score > self.best_score:
                self.best_score = score
                self.save_checkpoint('best_model.pt')

    def save_checkpoint(self, filename: str):
        """Save checkpoint."""
        path = os.path.join(self.output_dir, filename)
        torch.save({
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_score': self.best_score,
        }, path)
        print(f"Saved checkpoint to {path}")

    def load_checkpoint(self, filename: str):
        """Load checkpoint."""
        path = os.path.join(self.output_dir, filename)
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.best_score = checkpoint['best_score']
        print(f"Loaded checkpoint from {path}")
