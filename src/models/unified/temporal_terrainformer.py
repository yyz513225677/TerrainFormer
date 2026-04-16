"""
TemporalTerrainFormer: Unified Model with Temporal Context

Uses past 10 + current + future 10 predicted frames for decision making.
PointPillars encoder for real-time BEV projection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
from collections import deque

from ..lidar_encoder import PointPillarProjection
from ..world_model import WorldModel
from ..decision import TemporalDecisionTransformer


class TemporalTerrainFormer(nn.Module):
    """
    TemporalTerrainFormer: Unified architecture with temporal context.

    Uses 21 frames for decision making:
    - Past 10 frames (from buffer)
    - Current frame
    - Future 10 frames (from world model prediction)

    PointPillars encoder enables real-time inference at >30 FPS.
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
                 num_future_frames: int = 10,
                 num_past_frames: int = 10,
                 decision_hidden_dim: int = 384,
                 decision_num_layers: int = 4,
                 decision_num_heads: int = 6,
                 num_actions: int = 12,  # Simplified: 12 forward-only actions
                 state_dim: int = 6,
                 goal_dim: int = 2,
                 action_embed_dim: int = 128,
                 max_action_history: int = 10,
                 encoder_type: str = 'pointpillars',  # Only 'pointpillars' supported
                 max_pillars: int = 12000,
                 max_points_per_pillar: int = 32):
        super().__init__()

        self.bev_size = bev_size
        self.num_actions = num_actions
        self.encoder_type = 'pointpillars'  # Always use PointPillars
        self.num_past_frames = num_past_frames
        self.num_future_frames = num_future_frames
        self.world_embed_dim = world_embed_dim
        self.num_latent_tokens = num_latent_tokens

        # PointPillars LiDAR Encoder: Direct point cloud to BEV (~5ms)
        self.lidar_encoder = PointPillarProjection(
            in_channels=lidar_in_channels,
            pillar_channels=64,
            out_channels=bev_channels,
            height=bev_size,
            width=bev_size,
            max_pillars=max_pillars,
            max_points_per_pillar=max_points_per_pillar
        )
        self._global_pool = nn.AdaptiveAvgPool2d(1)
        self._global_proj = nn.Linear(bev_channels, pointnet_global_dim)

        # World Model with future prediction
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

        # Future predictor: predicts future world features from current
        self.future_predictor = FuturePredictor(
            embed_dim=world_embed_dim,
            num_future_frames=num_future_frames,
            num_latent_tokens=num_latent_tokens
        )

        # Temporal Decision Transformer
        self.decision_transformer = TemporalDecisionTransformer(
            world_model_dim=world_embed_dim,
            hidden_dim=decision_hidden_dim,
            num_layers=decision_num_layers,
            num_heads=decision_num_heads,
            num_actions=num_actions,
            state_dim=state_dim,
            goal_dim=goal_dim,
            action_embed_dim=action_embed_dim,
            max_action_history=max_action_history,
            num_past_frames=num_past_frames,
            num_future_frames=num_future_frames
        )

    def encode_lidar(self, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode LiDAR point cloud to BEV features using PointPillars."""
        bev = self.lidar_encoder(points)
        global_feat = self._global_pool(bev).flatten(1)
        global_feat = self._global_proj(global_feat)
        return bev, global_feat

    def encode_frame(self, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode a single frame to world model features.

        Returns:
            global_feat: (B, world_embed_dim)
            latent: (B, num_latent_tokens, world_embed_dim)
        """
        bev, _ = self.encode_lidar(points)
        latent, global_feat = self.world_model.encode(bev)
        return global_feat, latent

    def encode_multiple_frames(self, points_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode multiple frames.

        Args:
            points_seq: (B, T, N, C) sequence of point clouds

        Returns:
            global_feats: (B, T, world_embed_dim)
            latents: (B, T, num_latent_tokens, world_embed_dim)
        """
        B, T, N, C = points_seq.shape

        global_feats = []
        latents = []

        for t in range(T):
            global_feat, latent = self.encode_frame(points_seq[:, t])
            global_feats.append(global_feat)
            latents.append(latent)

        global_feats = torch.stack(global_feats, dim=1)  # (B, T, D)
        latents = torch.stack(latents, dim=1)  # (B, T, N_latent, D)

        return global_feats, latents

    def forward(self,
                current_points: torch.Tensor,
                past_points: Optional[torch.Tensor],
                state: torch.Tensor,
                goal: torch.Tensor,
                action_history: torch.Tensor,
                return_world_predictions: bool = True) -> Dict[str, torch.Tensor]:
        """
        Forward pass with temporal context.

        Args:
            current_points: (B, N, C) current point cloud
            past_points: (B, T_past, N, C) past point clouds or None
            state: (B, state_dim) vehicle state
            goal: (B, goal_dim) goal direction
            action_history: (B, T_action) past action IDs
            return_world_predictions: whether to return world model predictions

        Returns:
            Dictionary containing action logits and predictions
        """
        B = current_points.shape[0]
        device = current_points.device

        # Encode current frame
        bev, _ = self.encode_lidar(current_points)
        world_outputs = self.world_model(bev, return_latent=True)

        current_global = world_outputs['global_feature']
        current_latent = world_outputs['latent']

        # Encode past frames if available
        if past_points is not None and past_points.shape[1] > 0:
            past_global, past_latent = self.encode_multiple_frames(past_points)
        else:
            # Use same dtype as current_global for AMP compatibility
            past_global = torch.zeros(B, 0, self.world_embed_dim, device=device, dtype=current_global.dtype)
            past_latent = None

        # Predict future frames from current latent
        future_global, future_latent = self.future_predictor(current_global, current_latent)

        # Decision with temporal context
        decision_outputs = self.decision_transformer(
            past_global, past_latent,
            current_global, current_latent,
            future_global, future_latent,
            state, goal, action_history
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
            outputs['predicted_future_global'] = future_global
            outputs['predicted_future_latent'] = future_latent

        return outputs

    def forward_single_frame(self,
                             points: torch.Tensor,
                             state: torch.Tensor,
                             goal: torch.Tensor,
                             action_history: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Backward compatible single-frame forward."""
        return self.forward(
            current_points=points,
            past_points=None,
            state=state,
            goal=goal,
            action_history=action_history
        )

    def predict(self,
                current_points: torch.Tensor,
                past_points: Optional[torch.Tensor],
                state: torch.Tensor,
                goal: torch.Tensor,
                action_history: torch.Tensor,
                temperature: float = 1.0,
                sample: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict action with temporal context."""
        outputs = self.forward(
            current_points, past_points,
            state, goal, action_history,
            return_world_predictions=False
        )

        logits = outputs['action_logits'] / temperature

        if sample:
            probs = F.softmax(logits, dim=-1)
            actions = torch.multinomial(probs, 1).squeeze(-1)
        else:
            actions = logits.argmax(dim=-1)

        confidence = outputs['confidence'].squeeze(-1)

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
        for param in self._global_proj.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        for param in self.lidar_encoder.parameters():
            param.requires_grad = True
        for param in self._global_proj.parameters():
            param.requires_grad = True


class FuturePredictor(nn.Module):
    """
    Predicts future world features from current frame.

    Uses autoregressive prediction to generate future latent representations.
    """

    def __init__(self,
                 embed_dim: int = 512,
                 num_future_frames: int = 10,
                 num_latent_tokens: int = 64):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_future_frames = num_future_frames
        self.num_latent_tokens = num_latent_tokens

        # Temporal transformer for future prediction
        self.temporal_embed = nn.Parameter(torch.randn(1, num_future_frames, embed_dim) * 0.02)

        # Global feature predictor
        self.global_predictor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim * num_future_frames),
        )

        # Latent predictor with temporal dynamics
        self.latent_predictor = TemporalLatentPredictor(
            embed_dim=embed_dim,
            num_future_frames=num_future_frames,
            num_latent_tokens=num_latent_tokens
        )

    def forward(self, current_global: torch.Tensor,
                current_latent: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict future features.

        Args:
            current_global: (B, embed_dim) current global feature
            current_latent: (B, num_latent_tokens, embed_dim) current latent

        Returns:
            future_global: (B, T_future, embed_dim)
            future_latent: (B, T_future, num_latent_tokens, embed_dim)
        """
        B = current_global.shape[0]

        # Predict future global features
        future_global = self.global_predictor(current_global)
        future_global = future_global.view(B, self.num_future_frames, self.embed_dim)
        future_global = future_global + self.temporal_embed

        # Predict future latent representations
        future_latent = self.latent_predictor(current_latent)

        return future_global, future_latent


class TemporalLatentPredictor(nn.Module):
    """Predicts future latent representations using temporal dynamics."""

    def __init__(self,
                 embed_dim: int = 512,
                 num_future_frames: int = 10,
                 num_latent_tokens: int = 64):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_future_frames = num_future_frames
        self.num_latent_tokens = num_latent_tokens

        # GRU for temporal dynamics
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=embed_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )

        # Per-token prediction
        self.token_predictor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Temporal positional encoding
        self.temporal_pos = nn.Parameter(
            torch.randn(1, num_future_frames, 1, embed_dim) * 0.02
        )

    def forward(self, current_latent: torch.Tensor) -> torch.Tensor:
        """
        Predict future latent representations.

        Args:
            current_latent: (B, N_tokens, embed_dim)

        Returns:
            future_latent: (B, T_future, N_tokens, embed_dim)
        """
        B, N, D = current_latent.shape

        future_latents = []

        # Initialize hidden state from current latent mean
        h = current_latent.mean(dim=1).unsqueeze(0).repeat(2, 1, 1)  # (2, B, D)

        # Current input
        x = current_latent  # (B, N, D)

        for t in range(self.num_future_frames):
            # Process each token through GRU
            x_flat = x.view(B * N, 1, D).contiguous()
            h_expanded = h.unsqueeze(2).expand(-1, -1, N, -1).reshape(2, B * N, D).contiguous()

            out, h_new = self.gru(x_flat, h_expanded)
            out = out.view(B, N, D)
            h = h_new.view(2, B, N, D).mean(dim=2).contiguous()

            # Predict next latent
            next_latent = self.token_predictor(out)
            future_latents.append(next_latent)

            # Use prediction for next step
            x = next_latent

        # Stack and add temporal positional encoding
        future_latent = torch.stack(future_latents, dim=1)  # (B, T, N, D)
        future_latent = future_latent + self.temporal_pos

        return future_latent


class TemporalFrameBuffer:
    """
    Buffer for storing past frame features for real-time inference.
    """

    def __init__(self, max_frames: int = 10, device: str = 'cuda'):
        self.max_frames = max_frames
        self.device = device
        self.global_buffer: deque = deque(maxlen=max_frames)
        self.latent_buffer: deque = deque(maxlen=max_frames)

    def add(self, global_feat: torch.Tensor, latent: torch.Tensor):
        """Add a frame's features to the buffer."""
        self.global_buffer.append(global_feat.detach())
        self.latent_buffer.append(latent.detach())

    def get_past_features(self) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Get past features as tensors.

        Returns:
            past_global: (B, T, D)
            past_latent: (B, T, N, D) or None
        """
        if len(self.global_buffer) == 0:
            return None, None

        past_global = torch.stack(list(self.global_buffer), dim=1)

        if len(self.latent_buffer) > 0:
            past_latent = torch.stack(list(self.latent_buffer), dim=1)
        else:
            past_latent = None

        return past_global, past_latent

    def clear(self):
        """Clear the buffer."""
        self.global_buffer.clear()
        self.latent_buffer.clear()

    def __len__(self):
        return len(self.global_buffer)


def test_temporal_terrainformer():
    print("Testing TemporalTerrainFormer with PointPillars encoder...")

    model = TemporalTerrainFormer(
        lidar_in_channels=4,
        bev_size=256,
        num_actions=12,
        num_past_frames=10,
        num_future_frames=10
    )

    B, N = 2, 16384
    T_past = 5

    current_points = torch.randn(B, N, 4)
    current_points[:, :, :3] *= 30

    past_points = torch.randn(B, T_past, N, 4)
    past_points[:, :, :, :3] *= 30

    state = torch.randn(B, 6)
    goal = torch.randn(B, 2)
    action_history = torch.randint(0, 12, (B, 10))

    # Full temporal forward
    print("\n--- Testing temporal forward ---")
    outputs = model(current_points, past_points, state, goal, action_history)
    print(f"Action logits: {outputs['action_logits'].shape}")
    print(f"Confidence: {outputs['confidence'].shape}")
    print(f"Predicted future global: {outputs['predicted_future_global'].shape}")
    print(f"Predicted future latent: {outputs['predicted_future_latent'].shape}")

    # Single frame forward (backward compatible)
    print("\n--- Testing single frame forward ---")
    outputs_single = model.forward_single_frame(current_points, state, goal, action_history)
    print(f"Action logits: {outputs_single['action_logits'].shape}")

    # Predict
    print("\n--- Testing predict ---")
    actions, confidence = model.predict(current_points, past_points, state, goal, action_history)
    print(f"Predicted actions: {actions}")
    print(f"Confidence: {confidence}")

    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params / 1e6:.2f}M")

    print("\nTemporalTerrainFormer test passed!")


if __name__ == "__main__":
    test_temporal_terrainformer()
