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

ACTION_NAMES = {
    0: 'stop', 1: 'fwd_slow', 2: 'fwd_med', 3: 'fwd_fast',
    4: 'L_sharp', 5: 'L_med', 6: 'L_slight',
    7: 'R_slight', 8: 'R_med', 9: 'R_sharp',
    10: 'fwd_L', 11: 'fwd_R',
}


class TemporalEnsemble:
    """
    Temporal ensemble for action chunking at inference time.

    At each frame, the model predicts a chunk of K future actions.
    This class accumulates overlapping chunk predictions from the last K frames
    and combines them with exponential decay weighting (more recent = higher weight).

    Usage:
        ensemble = TemporalEnsemble(chunk_size=5, num_actions=12, decay=0.9)
        for frame in video:
            chunk_logits = model_chunk_output  # numpy (K, num_actions)
            action = ensemble.update(chunk_logits)
        ensemble.reset()  # call between sequences
    """

    def __init__(self, chunk_size: int = 5, num_actions: int = 12, decay: float = 0.9):
        self.chunk_size = chunk_size
        self.num_actions = num_actions
        self.decay = decay
        self.buffer: List[np.ndarray] = []  # list of (K, num_actions) arrays

    def update(self, chunk_logits: np.ndarray) -> int:
        """
        Add new chunk prediction and return ensembled action for the current step.

        Args:
            chunk_logits: (K, num_actions) logits from the current frame

        Returns:
            Predicted action index (int)
        """
        self.buffer.append(chunk_logits.copy())
        if len(self.buffer) > self.chunk_size:
            self.buffer.pop(0)

        # For the current frame T, aggregate predictions from past buffers:
        # buffer from k frames ago → use its k-th step prediction, weight = decay^k
        accum = np.zeros(self.num_actions, dtype=np.float64)
        w_sum = 0.0
        for k, chunk in enumerate(reversed(self.buffer)):
            if k < len(chunk):
                w = self.decay ** k
                accum += w * chunk[k]
                w_sum += w

        return int(np.argmax(accum / w_sum if w_sum > 0 else accum))

    def reset(self):
        """Clear buffer between sequences."""
        self.buffer.clear()


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
                 num_classes: int = 12):
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
                 num_classes: int = 12,
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
        self.num_classes = num_classes

        self.scaler = torch.amp.GradScaler('cuda') if fp16 else None

        # Create loss function based on config
        if loss_type == 'focal':
            self.criterion = FocalLoss(
                gamma=focal_gamma,
                alpha=class_weights,
                label_smoothing=label_smoothing,
                num_classes=num_classes
            )
            weights_str = "with class weights" if class_weights else "no class weights"
            print(f"Using Focal Loss (gamma={focal_gamma}, label_smoothing={label_smoothing}, {weights_str})")
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
                
                # Auxiliary traversability loss: use BEV traversability map mean as target
                if 'decision_traversability' in outputs and 'traversability_map' in batch:
                    trav_target = batch['traversability_map'].mean(dim=(-1, -2))  # (B,)
                    with torch.amp.autocast('cuda', enabled=False):
                        loss += 0.2 * F.binary_cross_entropy(
                            outputs['decision_traversability'].squeeze(-1).float(),
                            trav_target.float(),
                        )

                # Action chunk loss: average classification loss over K future steps
                if 'action_chunk_logits' in outputs and 'action_chunk' in batch:
                    cl = outputs['action_chunk_logits']   # (B, K, num_actions)
                    ct = batch['action_chunk']            # (B, K)
                    B_c, K, C = cl.shape
                    chunk_loss = self.criterion(cl.reshape(B_c * K, C), ct.reshape(B_c * K))
                    loss = loss + 0.5 * chunk_loss
                    
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
        chunk_correct = 0.0
        chunk_total = 0

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
                B = batch['expert_action'].shape[0]
                correct += (preds == batch['expert_action']).sum().item()
                total += B

                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(batch['expert_action'].cpu().numpy())

                # Chunk accuracy: fraction of all K future steps predicted correctly
                if 'action_chunk_logits' in outputs and 'action_chunk' in batch:
                    chunk_preds = outputs['action_chunk_logits'].argmax(dim=-1)  # (B, K)
                    chunk_correct += (chunk_preds == batch['action_chunk']).float().mean().item() * B
                    chunk_total += B
                
        # Compute per-class accuracy
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        per_class_acc = {}
        class_lines = []
        for cls in range(self.num_classes):
            mask = all_targets == cls
            if mask.sum() > 0:
                acc = float((all_preds[mask] == cls).mean())
                name = ACTION_NAMES.get(cls, str(cls))
                per_class_acc[f'val/acc_{name}'] = acc
                class_lines.append(f'{name}={acc:.2f}({mask.sum()})')

        if class_lines:
            print('  per-class: ' + '  '.join(class_lines))

        metrics = {
            'val/loss': total_loss / len(self.val_loader),
            'val/accuracy': correct / total,
            **per_class_acc
        }
        if chunk_total > 0:
            metrics['val/chunk_accuracy'] = chunk_correct / chunk_total
        return metrics
    
    def _forward_pass(self, batch: Dict) -> Dict:
        """Forward pass through TerrainFormer model."""
        # TerrainFormer takes: points, state, goal, action_history
        # and internally handles encoding through lidar_encoder -> world_model -> decision_transformer
        return self.model(
            points=batch['point_cloud'],
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
