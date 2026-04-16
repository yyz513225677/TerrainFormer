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

    12-action forward-only space (turn-first classification):
    - Actions 0-3: Forward speed (minimal turn |omega| < 0.1)
    - Actions 4-9: Turn actions (any velocity, classified by turn rate)
    - Actions 10-11: Fast forward with sharp turn
    """

    # Forward speed actions (minimal turn)
    STOP = 0
    FORWARD_SLOW = 1
    FORWARD_MEDIUM = 2
    FORWARD_FAST = 3

    # Turn actions (any velocity)
    TURN_LEFT_SHARP = 4
    TURN_LEFT_MEDIUM = 5
    TURN_LEFT_SLIGHT = 6
    TURN_RIGHT_SLIGHT = 7
    TURN_RIGHT_MEDIUM = 8
    TURN_RIGHT_SHARP = 9

    # Combined fast forward + sharp turn
    FORWARD_LEFT = 10
    FORWARD_RIGHT = 11

    NUM_ACTIONS = 12

    @classmethod
    def action_to_name(cls, action_id: int) -> str:
        """Convert action ID to human-readable name."""
        names = {
            0: "stop",
            1: "forward_slow",
            2: "forward_medium",
            3: "forward_fast",
            4: "turn_left_sharp",
            5: "turn_left_medium",
            6: "turn_left_slight",
            7: "turn_right_slight",
            8: "turn_right_medium",
            9: "turn_right_sharp",
            10: "forward_left",
            11: "forward_right",
        }
        return names.get(action_id, "unknown")

    @classmethod
    def get_forward_actions(cls) -> List[int]:
        """Get list of forward speed action IDs."""
        return [0, 1, 2, 3]

    @classmethod
    def get_turn_actions(cls) -> List[int]:
        """Get list of turn action IDs."""
        return [4, 5, 6, 7, 8, 9]

    @classmethod
    def get_combined_actions(cls) -> List[int]:
        """Get list of combined forward+turn action IDs."""
        return [10, 11]

    @classmethod
    def action_to_steering_angle(cls, action_id: int) -> float:
        """Convert action to approximate steering angle in degrees."""
        angles = {
            0: 0, 1: 0, 2: 0, 3: 0,  # Forward actions
            4: -45, 5: -25, 6: -10,   # Left turns
            7: 10, 8: 25, 9: 45,      # Right turns
            10: -45, 11: 45,          # Combined
        }
        return angles.get(action_id, 0)


class ActionTokenizer(nn.Module):
    """
    Tokenizes actions for the decision transformer.
    
    Converts discrete action IDs to embeddings and provides
    utilities for action sequence processing.
    """
    
    def __init__(self,
                 num_actions: int = 12,
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
