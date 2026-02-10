"""
Decision Transformer

End-to-end decision making from world model features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from .action_tokenizer import ActionTokenizer, ActionVocabulary
from .context_aggregator import ContextAggregator
from .output_heads import OutputHeads


class DecisionTransformerBlock(nn.Module):
    """Transformer block for decision making."""
    
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
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
        # Self-attention
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed, attn_mask=mask)
        x = x + attn_out
        
        # MLP
        x = x + self.mlp(self.norm2(x))
        
        return x


class DecisionTransformer(nn.Module):
    """
    Decision Transformer for action prediction.
    
    Takes world model features, vehicle state, goal, and action history
    to predict the next action.
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
            dropout: Dropout rate
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions
        
        # Action tokenizer
        self.action_tokenizer = ActionTokenizer(
            num_actions=num_actions,
            embed_dim=action_embed_dim,
            max_history=max_action_history
        )
        
        # Context aggregator
        self.context_aggregator = ContextAggregator(
            world_model_dim=world_model_dim,
            state_dim=state_dim,
            goal_dim=goal_dim,
            action_embed_dim=action_embed_dim,
            output_dim=hidden_dim
        )
        
        # Transformer layers
        self.transformer_blocks = nn.ModuleList([
            DecisionTransformerBlock(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Output layer norm
        self.norm = nn.LayerNorm(hidden_dim)
        
        # Output heads
        self.output_heads = OutputHeads(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim // 2,
            num_actions=num_actions,
            use_auxiliary=True
        )
        
    def forward(self,
                world_global: torch.Tensor,
                world_latent: Optional[torch.Tensor],
                state: torch.Tensor,
                goal: torch.Tensor,
                action_history: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            world_global: Global feature from world model (B, world_model_dim)
            world_latent: Latent tokens from world model (B, N, world_model_dim)
            state: Vehicle state (B, state_dim)
            goal: Goal direction (B, goal_dim)
            action_history: Past action IDs (B, T)
            
        Returns:
            Dictionary containing action logits, confidence, and auxiliary predictions
        """
        B = world_global.shape[0]
        
        # Encode action history
        action_embeds = self.action_tokenizer.embed_history(action_history)
        action_agg = action_embeds.mean(dim=1)  # Simple aggregation
        
        # Aggregate context
        _, context = self.context_aggregator(
            world_global, world_latent, state, goal, action_agg
        )
        
        # Reshape for transformer: create sequence
        # We use the aggregated context as a single token
        x = context.unsqueeze(1)  # (B, 1, hidden_dim)
        
        # Process through transformer
        for block in self.transformer_blocks:
            x = block(x)
            
        x = self.norm(x)
        
        # Use the output token for prediction
        output = x.squeeze(1)  # (B, hidden_dim)
        
        # Get predictions
        predictions = self.output_heads(output)
        
        return predictions
    
    def predict(self,
                world_global: torch.Tensor,
                world_latent: Optional[torch.Tensor],
                state: torch.Tensor,
                goal: torch.Tensor,
                action_history: torch.Tensor,
                temperature: float = 1.0,
                sample: bool = False,
                action_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict next action.
        
        Args:
            world_global: Global feature from world model (B, world_model_dim)
            world_latent: Latent tokens from world model (B, N, world_model_dim)
            state: Vehicle state (B, state_dim)
            goal: Goal direction (B, goal_dim)
            action_history: Past action IDs (B, T)
            temperature: Softmax temperature
            sample: If True, sample from distribution
            action_mask: Optional mask for valid actions (B, num_actions)
            
        Returns:
            actions: Predicted action IDs (B,)
            confidence: Confidence scores (B,)
        """
        outputs = self.forward(world_global, world_latent, state, goal, action_history)
        
        logits = outputs['action_logits'] / temperature
        
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, float('-inf'))
            
        if sample:
            probs = F.softmax(logits, dim=-1)
            actions = torch.multinomial(probs, 1).squeeze(-1)
        else:
            actions = logits.argmax(dim=-1)
            
        confidence = outputs['confidence'].squeeze(-1)
        
        return actions, confidence
    
    def get_action_probabilities(self,
                                  world_global: torch.Tensor,
                                  world_latent: Optional[torch.Tensor],
                                  state: torch.Tensor,
                                  goal: torch.Tensor,
                                  action_history: torch.Tensor) -> torch.Tensor:
        """
        Get probability distribution over actions.
        
        Returns:
            Action probabilities (B, num_actions)
        """
        outputs = self.forward(world_global, world_latent, state, goal, action_history)
        return F.softmax(outputs['action_logits'], dim=-1)


def test_decision_transformer():
    print("Testing Decision Transformer...")
    
    model = DecisionTransformer(
        world_model_dim=512,
        hidden_dim=384,
        num_layers=4,
        num_heads=6,
        num_actions=18,
        state_dim=6,
        goal_dim=2,
        action_embed_dim=128,
        max_action_history=10
    )
    
    B = 4
    world_global = torch.randn(B, 512)
    world_latent = torch.randn(B, 64, 512)
    state = torch.randn(B, 6)
    goal = torch.randn(B, 2)
    action_history = torch.randint(0, 18, (B, 10))
    
    # Forward pass
    outputs = model(world_global, world_latent, state, goal, action_history)
    
    print(f"Action logits shape: {outputs['action_logits'].shape}")
    print(f"Confidence shape: {outputs['confidence'].shape}")
    
    # Prediction
    actions, confidence = model.predict(
        world_global, world_latent, state, goal, action_history
    )
    
    print(f"Predicted actions: {actions}")
    print(f"Confidence: {confidence}")
    
    # Action probabilities
    probs = model.get_action_probabilities(
        world_global, world_latent, state, goal, action_history
    )
    print(f"Action probabilities shape: {probs.shape}")
    print(f"Sum of probs: {probs.sum(dim=-1)}")
    
    print("Decision Transformer test passed!")


if __name__ == "__main__":
    test_decision_transformer()
