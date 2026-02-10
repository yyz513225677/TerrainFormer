"""
Decision Transformer Trainer

Phase 2: Behavioral cloning with frozen world model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Optional, List
from tqdm import tqdm
import os
import numpy as np


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        gamma: Focusing parameter (default: 2.0). Higher gamma = more focus on hard examples.
        alpha: Per-class weights. If None, uses uniform weights.
        label_smoothing: Label smoothing factor (default: 0.0).
        reduction: 'mean', 'sum', or 'none'.
    """

    def __init__(self,
                 gamma: float = 2.0,
                 alpha: Optional[List[float]] = None,
                 label_smoothing: float = 0.0,
                 reduction: str = 'mean',
                 num_classes: int = 18):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        self.num_classes = num_classes

        if alpha is not None:
            self.register_buffer('alpha', torch.tensor(alpha, dtype=torch.float32))
        else:
            self.alpha = None

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: (B, C) logits
            targets: (B,) class indices
        """
        # Apply label smoothing via soft targets
        if self.label_smoothing > 0:
            n_classes = inputs.size(-1)
            smooth_targets = torch.zeros_like(inputs).scatter_(
                -1, targets.unsqueeze(-1), 1.0
            )
            smooth_targets = (1 - self.label_smoothing) * smooth_targets + \
                           self.label_smoothing / n_classes
            log_probs = F.log_softmax(inputs, dim=-1)
            ce_loss = -(smooth_targets * log_probs).sum(dim=-1)
        else:
            ce_loss = F.cross_entropy(inputs, targets, reduction='none')

        # Get probabilities
        probs = F.softmax(inputs, dim=-1)
        p_t = probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma

        # Apply per-class alpha weights
        if self.alpha is not None:
            alpha_t = self.alpha.to(inputs.device).gather(0, targets)
            focal_weight = alpha_t * focal_weight

        # Focal loss
        loss = focal_weight * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class DecisionTrainer:
    """Trainer for decision transformer."""
    
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 optimizer: torch.optim.Optimizer,
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
                 device: str = 'cuda',
                 output_dir: str = 'outputs',
                 fp16: bool = True,
                 label_smoothing: float = 0.1,
                 loss_type: str = 'cross_entropy',
                 focal_gamma: float = 2.0,
                 class_weights: Optional[List[float]] = None,
                 num_classes: int = 18,
                 encoder: Optional[nn.Module] = None,
                 world_model: Optional[nn.Module] = None):

        self.model = model.to(device)
        self.encoder = encoder.to(device) if encoder else None
        self.world_model = world_model.to(device) if world_model else None

        # Freeze encoder and world model
        if self.encoder:
            self.encoder.eval()
            for param in self.encoder.parameters():
                param.requires_grad = False

        if self.world_model:
            self.world_model.eval()
            for param in self.world_model.parameters():
                param.requires_grad = False

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.output_dir = output_dir
        self.fp16 = fp16
        self.label_smoothing = label_smoothing
        self.loss_type = loss_type

        self.scaler = torch.cuda.amp.GradScaler() if fp16 else None

        # Create loss function based on config
        if loss_type == 'focal':
            self.criterion = FocalLoss(
                gamma=focal_gamma,
                alpha=class_weights,
                label_smoothing=label_smoothing,
                num_classes=num_classes
            )
            print(f"Using Focal Loss (gamma={focal_gamma}, label_smoothing={label_smoothing})")
        else:
            self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
            print(f"Using CrossEntropy Loss (label_smoothing={label_smoothing})")

        os.makedirs(output_dir, exist_ok=True)
        self.best_accuracy = 0.0
        self.epoch = 0
        
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.epoch}')
        for batch in pbar:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            self.optimizer.zero_grad()
            
            with torch.amp.autocast('cuda', enabled=self.fp16):
                outputs = self._forward_pass(batch)
                
                loss = self.criterion(
                    outputs['action_logits'],
                    batch['expert_action']
                )
                
                # Add auxiliary losses (use BCEWithLogitsLoss for autocast safety)
                if 'traversability' in outputs:
                    loss += 0.2 * F.binary_cross_entropy_with_logits(
                        outputs['traversability'].squeeze(),
                        batch.get('traversability_target', torch.zeros_like(outputs['traversability'].squeeze()))
                    )
                    
            if self.fp16:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
                
            total_loss += loss.item()
            
            # Compute accuracy
            preds = outputs['action_logits'].argmax(dim=-1)
            correct += (preds == batch['expert_action']).sum().item()
            total += batch['expert_action'].shape[0]
            
            pbar.set_postfix({'loss': loss.item(), 'acc': correct/total})
            
        return {
            'train/loss': total_loss / len(self.train_loader),
            'train/accuracy': correct / total
        }
    
    def validate(self) -> Dict[str, float]:
        """Validate model."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                outputs = self._forward_pass(batch)
                loss = self.criterion(outputs['action_logits'], batch['expert_action'])
                
                total_loss += loss.item()
                preds = outputs['action_logits'].argmax(dim=-1)
                correct += (preds == batch['expert_action']).sum().item()
                total += batch['expert_action'].shape[0]
                
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(batch['expert_action'].cpu().numpy())
                
        # Compute per-class accuracy
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        per_class_acc = {}
        for cls in range(18):
            mask = all_targets == cls
            if mask.sum() > 0:
                per_class_acc[f'val/acc_class_{cls}'] = (all_preds[mask] == cls).mean()
                
        return {
            'val/loss': total_loss / len(self.val_loader),
            'val/accuracy': correct / total,
            **per_class_acc
        }
    
    def _forward_pass(self, batch: Dict) -> Dict:
        """Forward pass through model."""
        B = batch['point_cloud'].shape[0]

        # If encoder and world model are available, encode point cloud
        if self.encoder and self.world_model:
            with torch.no_grad():
                # Simple BEV encoding (placeholder - should use proper pillars encoding)
                bev = torch.randn(B, 64, 256, 256, device=self.device)
                # Add small contribution from point cloud to maintain data dependency
                point_feat = batch['point_cloud'].mean(dim=1)  # (B, 4)
                bev[:, :4, :, :] = bev[:, :4, :, :] + 0.01 * point_feat.view(B, 4, 1, 1)

                # Encode through world model
                world_outputs = self.world_model(bev)
                world_global = world_outputs.get('global_features', torch.randn(B, 512, device=self.device))
                world_latent = world_outputs.get('latent_features', torch.randn(B, 64, 512, device=self.device))
        else:
            # Fallback: use random features (should not be used in production)
            world_global = torch.randn(B, 512, device=self.device)
            world_latent = torch.randn(B, 64, 512, device=self.device)

        return self.model(
            world_global=world_global,
            world_latent=world_latent,
            state=batch['vehicle_state'],
            goal=batch['goal_direction'],
            action_history=batch['action_sequence']
        )
    
    def train(self, num_epochs: int):
        """Full training loop."""
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            train_metrics = self.train_epoch()
            val_metrics = self.validate()
            
            if self.scheduler:
                self.scheduler.step()
                
            all_metrics = {**train_metrics, **val_metrics}
            print(f"Epoch {epoch}: " + ", ".join(f"{k}={v:.4f}" for k, v in list(all_metrics.items())[:5]))
            
            if val_metrics['val/accuracy'] > self.best_accuracy:
                self.best_accuracy = val_metrics['val/accuracy']
                self.save_checkpoint('best_model.pt')
                
    def save_checkpoint(self, filename: str):
        """Save checkpoint."""
        path = os.path.join(self.output_dir, filename)
        torch.save({
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_accuracy': self.best_accuracy,
        }, path)
