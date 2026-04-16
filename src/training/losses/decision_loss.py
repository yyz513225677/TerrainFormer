"""Decision Transformer Loss Functions"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance in action prediction.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self,
                 gamma: float = 2.0,
                 alpha: Optional[torch.Tensor] = None,
                 label_smoothing: float = 0.0,
                 reduction: str = 'mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.size(-1)

        if self.label_smoothing > 0:
            with torch.no_grad():
                targets_smooth = torch.zeros_like(logits)
                targets_smooth.fill_(self.label_smoothing / (num_classes - 1))
                targets_smooth.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
        else:
            targets_smooth = F.one_hot(targets, num_classes).float()

        probs = F.softmax(logits, dim=-1)
        pt = (probs * targets_smooth).sum(dim=-1)
        focal_weight = (1 - pt) ** self.gamma
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        focal_loss = focal_weight * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class DecisionLoss(nn.Module):
    """
    Combined loss for decision transformer training.

    Includes:
    - Action prediction loss (focal loss for class imbalance)
    - Confidence calibration loss
    - Auxiliary losses (traversability, collision prediction)
    """

    def __init__(self,
                 num_actions: int = 18,
                 focal_gamma: float = 2.0,
                 label_smoothing: float = 0.1,
                 action_weight: float = 1.0,
                 confidence_weight: float = 0.1,
                 traversability_weight: float = 0.2,
                 collision_weight: float = 0.3,
                 class_weights: Optional[torch.Tensor] = None):
        super().__init__()

        self.num_actions = num_actions
        self.action_weight = action_weight
        self.confidence_weight = confidence_weight
        self.traversability_weight = traversability_weight
        self.collision_weight = collision_weight

        # Main action loss
        self.action_loss = FocalLoss(
            gamma=focal_gamma,
            alpha=class_weights,
            label_smoothing=label_smoothing
        )

        # Confidence calibration
        self.confidence_loss = nn.BCEWithLogitsLoss()

        # Auxiliary losses
        self.traversability_loss = nn.BCEWithLogitsLoss()
        self.collision_loss = nn.BCEWithLogitsLoss()

    def forward(self,
                outputs: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute all losses.

        Args:
            outputs: Model outputs with 'action_logits', 'confidence', etc.
            targets: Target values with 'action', 'traversability', etc.

        Returns:
            Dictionary with individual losses and total loss
        """
        losses = {}
        total_loss = 0.0

        # Action prediction loss
        if 'action_logits' in outputs and 'action' in targets:
            action_logits = outputs['action_logits']
            action_targets = targets['action']

            action_loss = self.action_loss(action_logits, action_targets)
            losses['action'] = action_loss
            total_loss = total_loss + self.action_weight * action_loss

            # Compute accuracy for logging
            with torch.no_grad():
                predictions = action_logits.argmax(dim=-1)
                accuracy = (predictions == action_targets).float().mean()
                losses['accuracy'] = accuracy

        # Confidence calibration loss
        if 'confidence' in outputs and 'action' in targets:
            confidence = outputs['confidence']
            action_logits = outputs['action_logits']

            with torch.no_grad():
                predictions = action_logits.argmax(dim=-1)
                correct = (predictions == targets['action']).float()

            conf_loss = self.confidence_loss(confidence.squeeze(-1), correct)
            losses['confidence'] = conf_loss
            total_loss = total_loss + self.confidence_weight * conf_loss

        # Traversability auxiliary loss
        if 'pred_traversability' in outputs and 'traversability' in targets:
            trav_loss = self.traversability_loss(
                outputs['pred_traversability'],
                targets['traversability']
            )
            losses['traversability'] = trav_loss
            total_loss = total_loss + self.traversability_weight * trav_loss

        # Collision auxiliary loss
        if 'pred_collision' in outputs and 'collision' in targets:
            collision_loss = self.collision_loss(
                outputs['pred_collision'],
                targets['collision']
            )
            losses['collision'] = collision_loss
            total_loss = total_loss + self.collision_weight * collision_loss

        losses['total'] = total_loss
        return losses
