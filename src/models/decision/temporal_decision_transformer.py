"""
Temporal Decision Transformer

Uses temporal context (past + current + future predicted frames) for decision making.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List

from .action_tokenizer import ActionTokenizer
from .output_heads import OutputHeads


class TemporalPositionalEncoding(nn.Module):
    """Learnable positional encoding for temporal sequence."""

    def __init__(self, embed_dim: int, max_seq_len: int = 21):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, embed_dim) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        B, T, D = x.shape
        return x + self.pos_embed[:, :T, :]


class TemporalTransformerBlock(nn.Module):
    """Transformer block with temporal attention."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.temporal_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Temporal self-attention
        normed = self.norm1(x)
        attn_out, _ = self.temporal_attn(normed, normed, normed, attn_mask=mask)
        x = x + attn_out

        # MLP
        x = x + self.mlp(self.norm2(x))

        return x


class TemporalFrameEncoder(nn.Module):
    """Encodes a single frame's world model features."""

    def __init__(self, world_model_dim: int, hidden_dim: int, num_latent_tokens: int = 64):
        super().__init__()

        self.world_model_dim = world_model_dim
        self.hidden_dim = hidden_dim

        # Global feature projection
        self.global_proj = nn.Linear(world_model_dim, hidden_dim)

        # Latent token aggregation
        self.latent_attn = nn.MultiheadAttention(
            hidden_dim, num_heads=4, dropout=0.1, batch_first=True
        )
        self.latent_proj = nn.Linear(world_model_dim, hidden_dim)
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, world_global: torch.Tensor,
                world_latent: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Encode a single frame's features.

        Args:
            world_global: (B, world_model_dim)
            world_latent: (B, N, world_model_dim) or None

        Returns:
            frame_embed: (B, hidden_dim)
        """
        B = world_global.shape[0]

        # Project global feature
        global_embed = self.global_proj(world_global)

        if world_latent is not None:
            # Aggregate latent tokens via attention
            latent_proj = self.latent_proj(world_latent)
            query = self.query.expand(B, -1, -1)
            latent_agg, _ = self.latent_attn(query, latent_proj, latent_proj)
            latent_agg = latent_agg.squeeze(1)

            # Fuse global and latent
            frame_embed = self.fusion(torch.cat([global_embed, latent_agg], dim=-1))
        else:
            frame_embed = global_embed

        return frame_embed


class TemporalDecisionTransformer(nn.Module):
    """
    Temporal Decision Transformer for action prediction.

    Uses temporal context:
    - Past 10 frames (from buffer)
    - Current frame
    - Future 10 frames (from world model prediction)

    Total: 21 frames of context for decision making.
    """

    def __init__(self,
                 world_model_dim: int = 512,
                 hidden_dim: int = 384,
                 num_layers: int = 4,
                 num_heads: int = 6,
                 num_actions: int = 18,
                 state_dim: int = 6,
                 goal_dim: int = 2,
                 action_embed_dim: int = 128,
                 max_action_history: int = 10,
                 num_past_frames: int = 10,
                 num_future_frames: int = 10,
                 dropout: float = 0.1):
        """
        Args:
            world_model_dim: World model feature dimension
            hidden_dim: Transformer hidden dimension
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            num_actions: Number of discrete actions
            state_dim: Vehicle state dimension
            goal_dim: Goal direction dimension
            action_embed_dim: Action embedding dimension
            max_action_history: Maximum action history length
            num_past_frames: Number of past frames in context
            num_future_frames: Number of future frames to predict
            dropout: Dropout rate
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_actions = num_actions
        self.num_past_frames = num_past_frames
        self.num_future_frames = num_future_frames
        self.total_frames = num_past_frames + 1 + num_future_frames  # 21 total

        # Frame encoder (shared for all frames)
        self.frame_encoder = TemporalFrameEncoder(
            world_model_dim, hidden_dim
        )

        # Temporal position encoding
        self.temporal_pos = TemporalPositionalEncoding(hidden_dim, self.total_frames)

        # Frame type embeddings (past, current, future)
        self.frame_type_embed = nn.Embedding(3, hidden_dim)  # 0=past, 1=current, 2=future

        # Action tokenizer
        self.action_tokenizer = ActionTokenizer(
            num_actions=num_actions,
            embed_dim=action_embed_dim,
            max_history=max_action_history
        )

        # State and goal encoders
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.goal_encoder = nn.Sequential(
            nn.Linear(goal_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.action_proj = nn.Linear(action_embed_dim, hidden_dim)

        # Context tokens (state, goal, action history)
        self.context_pos = nn.Parameter(torch.randn(1, 3, hidden_dim) * 0.02)

        # Temporal transformer layers
        self.transformer_blocks = nn.ModuleList([
            TemporalTransformerBlock(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # Output layer norm
        self.norm = nn.LayerNorm(hidden_dim)

        # Decision token for aggregating temporal information
        self.decision_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        # Output heads
        self.output_heads = OutputHeads(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim // 2,
            num_actions=num_actions,
            use_auxiliary=True
        )

    def forward(self,
                past_global: torch.Tensor,
                past_latent: Optional[torch.Tensor],
                current_global: torch.Tensor,
                current_latent: Optional[torch.Tensor],
                future_global: torch.Tensor,
                future_latent: Optional[torch.Tensor],
                state: torch.Tensor,
                goal: torch.Tensor,
                action_history: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            past_global: Past frame global features (B, T_past, world_model_dim)
            past_latent: Past frame latent tokens (B, T_past, N, world_model_dim) or None
            current_global: Current frame global feature (B, world_model_dim)
            current_latent: Current frame latent tokens (B, N, world_model_dim) or None
            future_global: Future frame global features (B, T_future, world_model_dim)
            future_latent: Future frame latent tokens (B, T_future, N, world_model_dim) or None
            state: Vehicle state (B, state_dim)
            goal: Goal direction (B, goal_dim)
            action_history: Past action IDs (B, T_action)

        Returns:
            Dictionary containing action logits, confidence, and auxiliary predictions
        """
        B = current_global.shape[0]
        device = current_global.device

        frame_embeds = []
        frame_types = []

        # Encode past frames
        T_past = past_global.shape[1] if past_global.dim() > 1 else 0
        if T_past > 0:
            for t in range(T_past):
                past_lat_t = past_latent[:, t] if past_latent is not None else None
                embed = self.frame_encoder(past_global[:, t], past_lat_t)
                frame_embeds.append(embed)
                frame_types.append(0)  # past

        # Encode current frame
        current_embed = self.frame_encoder(current_global, current_latent)
        frame_embeds.append(current_embed)
        frame_types.append(1)  # current

        # Encode future frames
        T_future = future_global.shape[1] if future_global.dim() > 1 else 0
        if T_future > 0:
            for t in range(T_future):
                future_lat_t = future_latent[:, t] if future_latent is not None else None
                embed = self.frame_encoder(future_global[:, t], future_lat_t)
                frame_embeds.append(embed)
                frame_types.append(2)  # future

        # Stack frame embeddings: (B, T_total, hidden_dim)
        temporal_seq = torch.stack(frame_embeds, dim=1)
        T_total = temporal_seq.shape[1]

        # Add temporal position encoding
        temporal_seq = self.temporal_pos(temporal_seq)

        # Add frame type embeddings
        frame_type_ids = torch.tensor(frame_types, device=device).unsqueeze(0).expand(B, -1)
        temporal_seq = temporal_seq + self.frame_type_embed(frame_type_ids)

        # Encode context (state, goal, action history)
        state_embed = self.state_encoder(state).unsqueeze(1)
        goal_embed = self.goal_encoder(goal).unsqueeze(1)

        action_embeds = self.action_tokenizer.embed_history(action_history)
        action_agg = self.action_proj(action_embeds.mean(dim=1)).unsqueeze(1)

        context_tokens = torch.cat([state_embed, goal_embed, action_agg], dim=1)
        context_tokens = context_tokens + self.context_pos

        # Add decision token
        decision_token = self.decision_token.expand(B, -1, -1)

        # Full sequence: [decision_token, context_tokens, temporal_frames]
        full_seq = torch.cat([decision_token, context_tokens, temporal_seq], dim=1)

        # Process through transformer
        for block in self.transformer_blocks:
            full_seq = block(full_seq)

        full_seq = self.norm(full_seq)

        # Use decision token for output
        output = full_seq[:, 0]  # (B, hidden_dim)

        # Get predictions
        predictions = self.output_heads(output)

        return predictions

    def forward_single_frame(self,
                             world_global: torch.Tensor,
                             world_latent: Optional[torch.Tensor],
                             state: torch.Tensor,
                             goal: torch.Tensor,
                             action_history: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Simplified forward for single frame (backward compatible).
        Uses only current frame without temporal context.
        """
        B = world_global.shape[0]
        device = world_global.device

        # Create empty past and future (use same dtype as input for AMP compatibility)
        empty_global = torch.zeros(B, 0, world_global.shape[-1], device=device, dtype=world_global.dtype)

        return self.forward(
            past_global=empty_global,
            past_latent=None,
            current_global=world_global,
            current_latent=world_latent,
            future_global=empty_global,
            future_latent=None,
            state=state,
            goal=goal,
            action_history=action_history
        )

    def predict(self,
                past_global: torch.Tensor,
                past_latent: Optional[torch.Tensor],
                current_global: torch.Tensor,
                current_latent: Optional[torch.Tensor],
                future_global: torch.Tensor,
                future_latent: Optional[torch.Tensor],
                state: torch.Tensor,
                goal: torch.Tensor,
                action_history: torch.Tensor,
                temperature: float = 1.0,
                sample: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict next action from temporal context."""
        outputs = self.forward(
            past_global, past_latent,
            current_global, current_latent,
            future_global, future_latent,
            state, goal, action_history
        )

        logits = outputs['action_logits'] / temperature

        if sample:
            probs = F.softmax(logits, dim=-1)
            actions = torch.multinomial(probs, 1).squeeze(-1)
        else:
            actions = logits.argmax(dim=-1)

        confidence = outputs['confidence'].squeeze(-1)

        return actions, confidence


def test_temporal_decision_transformer():
    print("Testing Temporal Decision Transformer...")

    model = TemporalDecisionTransformer(
        world_model_dim=512,
        hidden_dim=384,
        num_layers=4,
        num_heads=6,
        num_actions=18,
        num_past_frames=10,
        num_future_frames=10
    )

    B = 2
    T_past = 10
    T_future = 10
    N_latent = 64

    # Create test inputs
    past_global = torch.randn(B, T_past, 512)
    past_latent = torch.randn(B, T_past, N_latent, 512)
    current_global = torch.randn(B, 512)
    current_latent = torch.randn(B, N_latent, 512)
    future_global = torch.randn(B, T_future, 512)
    future_latent = torch.randn(B, T_future, N_latent, 512)
    state = torch.randn(B, 6)
    goal = torch.randn(B, 2)
    action_history = torch.randint(0, 18, (B, 10))

    # Full temporal forward
    outputs = model(
        past_global, past_latent,
        current_global, current_latent,
        future_global, future_latent,
        state, goal, action_history
    )

    print(f"Action logits shape: {outputs['action_logits'].shape}")
    print(f"Confidence shape: {outputs['confidence'].shape}")

    # Single frame forward (backward compatible)
    outputs_single = model.forward_single_frame(
        current_global, current_latent,
        state, goal, action_history
    )
    print(f"Single frame action logits: {outputs_single['action_logits'].shape}")

    # Predict
    actions, confidence = model.predict(
        past_global, past_latent,
        current_global, current_latent,
        future_global, future_latent,
        state, goal, action_history
    )
    print(f"Predicted actions: {actions}")
    print(f"Confidence: {confidence}")

    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params / 1e6:.2f}M")

    print("Temporal Decision Transformer test passed!")


if __name__ == "__main__":
    test_temporal_decision_transformer()
