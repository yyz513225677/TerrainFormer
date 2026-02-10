"""
TerrainFormer: Unified Model

Combines LiDAR encoder, world model, and decision transformer
into a single end-to-end architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from ..lidar_encoder import PointNet2Backbone, BEVProjection, PointPillarProjection
from ..world_model import WorldModel
from ..decision import DecisionTransformer


class TerrainFormer(nn.Module):
    """
    TerrainFormer: Unified architecture for autonomous off-road navigation.
    """

    def __init__(self,
                 lidar_in_channels: int = 4,
                 pointnet_global_dim: int = 1024,
                 bev_channels: int = 64,
                 bev_size: int = 256,
                 world_embed_dim: int = 512,
                 world_patch_size: int = 16,
                 world_num_layers: int = 6,
                 world_num_heads: int = 8,
                 num_latent_tokens: int = 64,
                 num_semantic_classes: int = 20,
                 num_future_frames: int = 5,
                 decision_hidden_dim: int = 384,
                 decision_num_layers: int = 4,
                 decision_num_heads: int = 6,
                 num_actions: int = 18,
                 state_dim: int = 6,
                 goal_dim: int = 2,
                 action_embed_dim: int = 128,
                 max_action_history: int = 10,
                 encoder_type: str = 'pointpillars',  # 'pointpillars' or 'pointnet2'
                 max_pillars: int = 12000,
                 max_points_per_pillar: int = 32):
        super().__init__()

        self.bev_size = bev_size
        self.num_actions = num_actions
        self.encoder_type = encoder_type

        # LiDAR Encoder - choose between PointPillars (fast) or PointNet++ (accurate)
        if encoder_type == 'pointpillars':
            # PointPillars: Direct point cloud to BEV (fast, ~5ms)
            self.lidar_encoder = PointPillarProjection(
                in_channels=lidar_in_channels,
                pillar_channels=64,
                out_channels=bev_channels,
                height=bev_size,
                width=bev_size,
                max_pillars=max_pillars,
                max_points_per_pillar=max_points_per_pillar
            )
            self.bev_projection = None  # Not needed for PointPillars
            self._global_pool = nn.AdaptiveAvgPool2d(1)
            self._global_proj = nn.Linear(bev_channels, pointnet_global_dim)
        else:
            # PointNet++: Hierarchical point features (slower, ~25ms)
            self.lidar_encoder = PointNet2Backbone(
                in_channels=lidar_in_channels,
                global_dim=pointnet_global_dim
            )
            self.bev_projection = BEVProjection(
                in_channels=self.lidar_encoder.output_dim,
                out_channels=bev_channels,
                height=bev_size,
                width=bev_size
            )
            self._global_pool = None
            self._global_proj = None
        
        # World Model
        self.world_model = WorldModel(
            bev_channels=bev_channels,
            embed_dim=world_embed_dim,
            patch_size=world_patch_size,
            img_size=bev_size,
            num_transformer_layers=world_num_layers,
            num_heads=world_num_heads,
            num_latent_tokens=num_latent_tokens,
            num_classes=num_semantic_classes,
            num_future_frames=num_future_frames
        )
        
        # Decision Transformer
        self.decision_transformer = DecisionTransformer(
            world_model_dim=world_embed_dim,
            hidden_dim=decision_hidden_dim,
            num_layers=decision_num_layers,
            num_heads=decision_num_heads,
            num_actions=num_actions,
            state_dim=state_dim,
            goal_dim=goal_dim,
            action_embed_dim=action_embed_dim,
            max_action_history=max_action_history
        )
        
    def encode_lidar(self, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode LiDAR point cloud to BEV features.

        Args:
            points: (B, N, C) point cloud with xyz + features

        Returns:
            bev: (B, bev_channels, H, W) BEV feature map
            global_feat: (B, global_dim) global feature vector
        """
        if self.encoder_type == 'pointpillars':
            # PointPillars: direct to BEV
            bev = self.lidar_encoder(points)
            # Extract global feature from BEV
            global_feat = self._global_pool(bev).flatten(1)
            global_feat = self._global_proj(global_feat)
        else:
            # PointNet++: hierarchical encoding then BEV projection
            point_features, global_feat = self.lidar_encoder(points)
            xyz = points[:, :, :3]
            bev = self.bev_projection(xyz, point_features)
        return bev, global_feat
    
    def forward(self,
                points: torch.Tensor,
                state: torch.Tensor,
                goal: torch.Tensor,
                action_history: torch.Tensor,
                return_world_predictions: bool = True) -> Dict[str, torch.Tensor]:
        bev, _ = self.encode_lidar(points)
        world_outputs = self.world_model(bev, return_latent=True)
        
        world_global = world_outputs['global_feature']
        world_latent = world_outputs['latent']
        
        decision_outputs = self.decision_transformer(
            world_global, world_latent, state, goal, action_history
        )
        
        outputs = {
            'action_logits': decision_outputs['action_logits'],
            'confidence': decision_outputs['confidence'],
        }
        
        if 'traversability' in decision_outputs:
            outputs['decision_traversability'] = decision_outputs['traversability']
        if 'collision' in decision_outputs:
            outputs['collision'] = decision_outputs['collision']
            
        if return_world_predictions:
            outputs['world_traversability'] = world_outputs['traversability']
            outputs['world_elevation'] = world_outputs['elevation']
            outputs['world_semantics'] = world_outputs['semantics']
            outputs['world_future'] = world_outputs['future']
            
        return outputs
    
    def predict(self,
                points: torch.Tensor,
                state: torch.Tensor,
                goal: torch.Tensor,
                action_history: torch.Tensor,
                temperature: float = 1.0,
                sample: bool = False,
                action_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        bev, _ = self.encode_lidar(points)
        world_outputs = self.world_model(bev, return_latent=True)
        
        actions, confidence = self.decision_transformer.predict(
            world_outputs['global_feature'],
            world_outputs['latent'],
            state, goal, action_history,
            temperature=temperature,
            sample=sample,
            action_mask=action_mask
        )
        return actions, confidence
    
    def freeze_world_model(self):
        for param in self.world_model.parameters():
            param.requires_grad = False
            
    def unfreeze_world_model(self):
        for param in self.world_model.parameters():
            param.requires_grad = True
            
    def freeze_encoder(self):
        for param in self.lidar_encoder.parameters():
            param.requires_grad = False
        if self.bev_projection is not None:
            for param in self.bev_projection.parameters():
                param.requires_grad = False
        if self._global_proj is not None:
            for param in self._global_proj.parameters():
                param.requires_grad = False

    def unfreeze_encoder(self):
        for param in self.lidar_encoder.parameters():
            param.requires_grad = True
        if self.bev_projection is not None:
            for param in self.bev_projection.parameters():
                param.requires_grad = True
        if self._global_proj is not None:
            for param in self._global_proj.parameters():
                param.requires_grad = True
            
    def get_parameter_groups(self, lr_config: Dict[str, float]) -> list:
        encoder_params = list(self.lidar_encoder.parameters())
        if self.bev_projection is not None:
            encoder_params += list(self.bev_projection.parameters())
        if self._global_proj is not None:
            encoder_params += list(self._global_proj.parameters())

        return [
            {'params': encoder_params,
             'lr': lr_config.get('encoder', 1e-5)},
            {'params': self.world_model.parameters(),
             'lr': lr_config.get('world_model', 3e-5)},
            {'params': self.decision_transformer.parameters(),
             'lr': lr_config.get('decision', 1e-4)},
        ]


def test_terrainformer():
    print("Testing TerrainFormer...")

    # Test PointPillars (default, fast)
    print("\n--- Testing with PointPillars encoder ---")
    model = TerrainFormer(
        lidar_in_channels=4, bev_size=256, num_actions=18,
        encoder_type='pointpillars'
    )
    
    B, N = 2, 16384
    points = torch.randn(B, N, 4)
    points[:, :, :3] *= 30
    state = torch.randn(B, 6)
    goal = torch.randn(B, 2)
    action_history = torch.randint(0, 18, (B, 10))

    outputs = model(points, state, goal, action_history)
    print(f"Action logits: {outputs['action_logits'].shape}")

    actions, confidence = model.predict(points, state, goal, action_history)
    print(f"Predicted actions: {actions}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters (PointPillars): {total_params / 1e6:.2f}M")

    # Test PointNet++ encoder
    print("\n--- Testing with PointNet++ encoder ---")
    model_pn2 = TerrainFormer(
        lidar_in_channels=4, bev_size=256, num_actions=18,
        encoder_type='pointnet2'
    )

    outputs_pn2 = model_pn2(points, state, goal, action_history)
    print(f"Action logits: {outputs_pn2['action_logits'].shape}")

    total_params_pn2 = sum(p.numel() for p in model_pn2.parameters())
    print(f"Total parameters (PointNet++): {total_params_pn2 / 1e6:.2f}M")

    print("\nTerrainFormer test passed!")


if __name__ == "__main__":
    test_terrainformer()
