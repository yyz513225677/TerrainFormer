"""
Context Aggregator for Decision Transformer

Fuses world model features with vehicle state, goal, and action history.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class StateEncoder(nn.Module):
    """Encodes vehicle state to embedding."""
    
    def __init__(self, state_dim: int = 6, embed_dim: int = 128):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
        )
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.encoder(state)


class GoalEncoder(nn.Module):
    """Encodes goal direction to embedding."""
    
    def __init__(self, goal_dim: int = 2, embed_dim: int = 128):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(goal_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
        )
        
    def forward(self, goal: torch.Tensor) -> torch.Tensor:
        return self.encoder(goal)


class ContextAggregator(nn.Module):
    """
    Aggregates multi-modal context for decision making.
    
    Combines:
    - World model latent state
    - World model global feature
    - Vehicle state (velocity, orientation)
    - Goal direction
    - Action history
    """
    
    def __init__(self,
                 world_model_dim: int = 512,
                 state_dim: int = 6,
                 goal_dim: int = 2,
                 action_embed_dim: int = 128,
                 output_dim: int = 384):
        """
        Args:
            world_model_dim: World model feature dimension
            state_dim: Vehicle state dimension
            goal_dim: Goal direction dimension
            action_embed_dim: Action embedding dimension
            output_dim: Output context dimension
        """
        super().__init__()
        
        self.world_model_dim = world_model_dim
        self.output_dim = output_dim
        
        # Encoders for each modality
        self.state_encoder = StateEncoder(state_dim, action_embed_dim)
        self.goal_encoder = GoalEncoder(goal_dim, action_embed_dim)
        
        # Project world model features
        self.world_proj = nn.Linear(world_model_dim, output_dim)
        
        # Fusion layer
        # Input: world_model + state + goal + action_history
        fusion_input_dim = output_dim + action_embed_dim * 3
        
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, output_dim * 2),
            nn.LayerNorm(output_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
        )
        
        # Cross-attention for world model tokens
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Query projection
        self.query_proj = nn.Linear(output_dim, output_dim)
        
        # Final layer norm
        self.norm = nn.LayerNorm(output_dim)
        
    def forward(self,
                world_global: torch.Tensor,
                world_latent: Optional[torch.Tensor],
                state: torch.Tensor,
                goal: torch.Tensor,
                action_history_embed: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            world_global: Global feature from world model (B, world_model_dim)
            world_latent: Latent tokens from world model (B, N, world_model_dim) or None
            state: Vehicle state (B, state_dim)
            goal: Goal direction (B, goal_dim)
            action_history_embed: Aggregated action history (B, action_embed_dim)
            
        Returns:
            context: Fused context (B, output_dim)
            attended_world: Context with world model attention (B, output_dim)
        """
        B = world_global.shape[0]
        
        # Encode modalities
        state_embed = self.state_encoder(state)
        goal_embed = self.goal_encoder(goal)
        world_proj = self.world_proj(world_global)
        
        # Concatenate for fusion
        fused_input = torch.cat([
            world_proj,
            state_embed,
            goal_embed,
            action_history_embed
        ], dim=-1)
        
        context = self.fusion(fused_input)
        
        # Cross-attention with world model latent tokens if available
        if world_latent is not None:
            # Project world latent
            world_latent_proj = self.world_proj(world_latent)
            
            # Use fused context as query
            query = self.query_proj(context).unsqueeze(1)
            
            attended, _ = self.cross_attn(query, world_latent_proj, world_latent_proj)
            attended = attended.squeeze(1)
            
            # Residual connection
            attended_world = self.norm(context + attended)
        else:
            attended_world = context
            
        return context, attended_world


def test_context_aggregator():
    print("Testing Context Aggregator...")
    
    aggregator = ContextAggregator(
        world_model_dim=512,
        state_dim=6,
        goal_dim=2,
        action_embed_dim=128,
        output_dim=384
    )
    
    B = 4
    world_global = torch.randn(B, 512)
    world_latent = torch.randn(B, 64, 512)
    state = torch.randn(B, 6)
    goal = torch.randn(B, 2)
    action_hist = torch.randn(B, 128)
    
    context, attended = aggregator(
        world_global, world_latent, state, goal, action_hist
    )
    
    print(f"Context shape: {context.shape}")
    print(f"Attended shape: {attended.shape}")
    
    assert context.shape == (B, 384)
    assert attended.shape == (B, 384)
    
    print("Context Aggregator test passed!")


if __name__ == "__main__":
    test_context_aggregator()
