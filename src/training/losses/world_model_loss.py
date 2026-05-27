"""World Model Loss Functions"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class WorldModelLoss(nn.Module):
    """Combined loss for world model training."""
    
    def __init__(self,
                 reconstruction_weight: float = 1.0,
                 traversability_weight: float = 0.5,
                 elevation_weight: float = 0.3,
                 semantics_weight: float = 0.5):
        super().__init__()
        
        self.reconstruction_weight = reconstruction_weight
        self.traversability_weight = traversability_weight
        self.elevation_weight = elevation_weight
        self.semantics_weight = semantics_weight
        
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-1)
        self.bce_loss = nn.BCEWithLogitsLoss()  # Use BCEWithLogitsLoss for autocast safety
        self.mse_loss = nn.MSELoss()
        
    def forward(self, outputs: Dict, targets: Dict) -> Tuple[torch.Tensor, Dict]:
        """
        Compute combined loss.

        Args:
            outputs: Model outputs (keys: 'traversability', 'elevation', 'semantics', etc.)
            targets: Ground truth targets

        Returns:
            total_loss: Combined loss
            components: Dictionary of individual losses
        """
        components = {}
        total_loss = None

        # Traversability loss
        if 'traversability' in outputs and 'traversability_map' in targets:
            # BCEWithLogitsLoss applies sigmoid internally, so use raw logits
            trav_pred = outputs['traversability']
            trav_target = targets['traversability_map']

            # Ensure target has correct shape
            if trav_target.dim() == 3:  # (B, H, W)
                trav_target = trav_target.unsqueeze(1)  # (B, 1, H, W)

            # Resize if needed
            if trav_pred.shape != trav_target.shape:
                trav_target = F.interpolate(trav_target, size=trav_pred.shape[-2:], mode='nearest')

            trav_loss = self.bce_loss(trav_pred, trav_target)
            components['traversability'] = trav_loss.item()

            if total_loss is None:
                total_loss = self.traversability_weight * trav_loss
            else:
                total_loss += self.traversability_weight * trav_loss

        # Elevation loss
        if 'elevation' in outputs and 'elevation_map' in targets:
            elev_pred = outputs['elevation']
            elev_target = targets['elevation_map']

            # Handle shape mismatches
            if elev_pred.dim() == 4:  # (B, 1, H, W)
                elev_pred = elev_pred.squeeze(1)  # (B, H, W)

            # Resize if needed
            if elev_pred.shape[-2:] != elev_target.shape[-2:]:
                elev_target = F.interpolate(
                    elev_target.unsqueeze(1),
                    size=elev_pred.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                ).squeeze(1)

            elev_loss = self.mse_loss(elev_pred, elev_target)
            components['elevation'] = elev_loss.item()

            if total_loss is None:
                total_loss = self.elevation_weight * elev_loss
            else:
                total_loss += self.elevation_weight * elev_loss

        # Semantic loss
        if 'semantics' in outputs and 'terrain_labels' in targets:
            sem_pred = outputs['semantics']  # (B, num_classes, H, W)
            sem_target = targets['terrain_labels']  # (B, H, W)

            # Resize target if needed
            if sem_pred.shape[-2:] != sem_target.shape[-2:]:
                sem_target = F.interpolate(
                    sem_target.unsqueeze(1).float(),
                    size=sem_pred.shape[-2:],
                    mode='nearest'
                ).squeeze(1).long()

            sem_loss = self.ce_loss(sem_pred, sem_target)
            components['semantics'] = sem_loss.item()

            if total_loss is None:
                total_loss = self.semantics_weight * sem_loss
            else:
                total_loss += self.semantics_weight * sem_loss

        # If no valid losses computed, create a minimal loss from model outputs
        if total_loss is None:
            # Use a small regularization loss to ensure gradients flow
            if outputs:
                # Take any tensor from outputs and compute a small regularization
                first_output = next(iter(outputs.values()))
                if not isinstance(first_output, torch.Tensor):
                    # Skip non-tensor outputs
                    for v in outputs.values():
                        if isinstance(v, torch.Tensor):
                            first_output = v
                            break
                total_loss = 1e-6 * first_output.mean().abs()
                components['total'] = total_loss.item()
            else:
                # Fallback - this shouldn't happen
                raise ValueError("No model outputs available to compute loss")
        else:
            components['total'] = total_loss.item()

        return total_loss, components
