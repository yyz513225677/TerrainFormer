"""Decision Transformer Loss Functions"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class DecisionLoss(nn.Module):
    """Combined loss for decision transformer training."""
    
    def __init__(self,
                 action_weight: float = 1.0,
                 auxiliary_weight: float = 0.2,
                 label_smoothing: float = 0.1):
        super().__init__()
        
        self.action_weight = action_weight
        self.auxiliary_weight = auxiliary_weight
        
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.bce_loss = nn.BCELoss()
        
    def forward(self, outputs: Dict, targets: Dict) -> Tuple[torch.Tensor, Dict]:
        """Compute combined loss."""
        components = {}
        total_loss = 0
        
        # Action prediction loss
        action_loss = self.ce_loss(
            outputs['action_logits'],
            targets['expert_action']
        )
        components['action'] = action_loss.item()
        total_loss += self.action_weight * action_loss
        
        # Auxiliary losses
        if 'collision' in outputs:
            collision_target = targets.get('collision_label', 
                torch.zeros_like(outputs['collision']))
            collision_loss = self.bce_loss(outputs['collision'], collision_target)
            components['collision'] = collision_loss.item()
            total_loss += self.auxiliary_weight * collision_loss
            
        components['total'] = total_loss.item()
        return total_loss, components
