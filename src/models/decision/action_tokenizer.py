"""
Action Tokenizer for Decision Transformer

Converts discrete actions to embeddings and vice versa.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class ActionVocabulary:
    """
    Defines the discrete action space for off-road navigation.
    """
    
    # Steering actions
    STRAIGHT = 0
    LEFT_5 = 1
    RIGHT_5 = 2
    LEFT_10 = 3
    RIGHT_10 = 4
    LEFT_20 = 5
    RIGHT_20 = 6
    LEFT_30 = 7
    RIGHT_30 = 8
    LEFT_45 = 9
    RIGHT_45 = 10
    
    # Speed actions
    MAINTAIN = 11
    SLOW_DOWN = 12
    SPEED_UP = 13
    STOP = 14
    
    # Reverse actions
    REVERSE_STRAIGHT = 15
    REVERSE_LEFT = 16
    REVERSE_RIGHT = 17
    
    NUM_ACTIONS = 18
    
    @classmethod
    def action_to_name(cls, action_id: int) -> str:
        """Convert action ID to human-readable name."""
        names = {
            0: "straight",
            1: "left_5deg", 2: "right_5deg",
            3: "left_10deg", 4: "right_10deg",
            5: "left_20deg", 6: "right_20deg",
            7: "left_30deg", 8: "right_30deg",
            9: "left_45deg", 10: "right_45deg",
            11: "maintain_speed", 12: "slow_down",
            13: "speed_up", 14: "stop",
            15: "reverse_straight",
            16: "reverse_left", 17: "reverse_right",
        }
        return names.get(action_id, "unknown")
    
    @classmethod
    def get_steering_actions(cls) -> List[int]:
        """Get list of steering action IDs."""
        return list(range(11))
    
    @classmethod
    def get_speed_actions(cls) -> List[int]:
        """Get list of speed action IDs."""
        return [11, 12, 13, 14]
    
    @classmethod
    def get_reverse_actions(cls) -> List[int]:
        """Get list of reverse action IDs."""
        return [15, 16, 17]
    
    @classmethod
    def action_to_steering_angle(cls, action_id: int) -> float:
        """Convert steering action to angle in degrees."""
        angles = {
            0: 0, 1: -5, 2: 5, 3: -10, 4: 10,
            5: -20, 6: 20, 7: -30, 8: 30, 9: -45, 10: 45,
            15: 0, 16: -20, 17: 20
        }
        return angles.get(action_id, 0)


class ActionTokenizer(nn.Module):
    """
    Tokenizes actions for the decision transformer.
    
    Converts discrete action IDs to embeddings and provides
    utilities for action sequence processing.
    """
    
    def __init__(self,
                 num_actions: int = 18,
                 embed_dim: int = 128,
                 max_history: int = 10):
        """
        Args:
            num_actions: Size of action vocabulary
            embed_dim: Embedding dimension
            max_history: Maximum action history length
        """
        super().__init__()
        
        self.num_actions = num_actions
        self.embed_dim = embed_dim
        self.max_history = max_history
        
        # Action embedding
        self.action_embed = nn.Embedding(num_actions + 1, embed_dim, padding_idx=num_actions)
        
        # Temporal position embedding for action history
        self.temporal_embed = nn.Embedding(max_history, embed_dim)
        
        # Project to output dimension
        self.proj = nn.Linear(embed_dim, embed_dim)
        
        # Layer norm
        self.norm = nn.LayerNorm(embed_dim)
        
    def embed_single(self, action: torch.Tensor) -> torch.Tensor:
        """
        Embed a single action.
        
        Args:
            action: Action IDs (B,) or (B, 1)
            
        Returns:
            Embeddings (B, embed_dim)
        """
        if action.dim() == 2:
            action = action.squeeze(-1)
        return self.action_embed(action)
    
    def embed_history(self, actions: torch.Tensor,
                      mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Embed action history with temporal encoding.
        
        Args:
            actions: Action history (B, T) where T <= max_history
            mask: Optional mask for padding (B, T)
            
        Returns:
            History embeddings (B, T, embed_dim)
        """
        B, T = actions.shape
        device = actions.device
        
        # Embed actions
        action_embeds = self.action_embed(actions)
        
        # Add temporal position
        positions = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        temporal_embeds = self.temporal_embed(positions)
        
        embeds = action_embeds + temporal_embeds
        
        # Apply mask if provided
        if mask is not None:
            embeds = embeds * mask.unsqueeze(-1)
            
        # Normalize
        embeds = self.norm(embeds)
        
        return embeds
    
    def aggregate_history(self, actions: torch.Tensor,
                          method: str = 'last') -> torch.Tensor:
        """
        Aggregate action history to single vector.
        
        Args:
            actions: Action history (B, T)
            method: 'last', 'mean', or 'attention'
            
        Returns:
            Aggregated embedding (B, embed_dim)
        """
        embeds = self.embed_history(actions)
        
        if method == 'last':
            return embeds[:, -1]
        elif method == 'mean':
            return embeds.mean(dim=1)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
    
    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Default forward pass - embed history.
        
        Args:
            actions: Action history (B, T)
            
        Returns:
            History embeddings (B, T, embed_dim)
        """
        return self.embed_history(actions)


def test_action_tokenizer():
    print("Testing Action Tokenizer...")
    
    tokenizer = ActionTokenizer(
        num_actions=18,
        embed_dim=128,
        max_history=10
    )
    
    # Test vocabulary
    print(f"Action 0: {ActionVocabulary.action_to_name(0)}")
    print(f"Action 5: {ActionVocabulary.action_to_name(5)}")
    
    # Test embedding
    actions = torch.randint(0, 18, (4, 10))
    embeds = tokenizer(actions)
    print(f"History embeddings shape: {embeds.shape}")
    assert embeds.shape == (4, 10, 128)
    
    # Test aggregation
    agg = tokenizer.aggregate_history(actions, method='last')
    print(f"Aggregated shape: {agg.shape}")
    assert agg.shape == (4, 128)
    
    print("Action Tokenizer test passed!")


if __name__ == "__main__":
    test_action_tokenizer()
