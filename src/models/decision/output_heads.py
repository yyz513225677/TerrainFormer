"""
Output Heads for Decision Transformer

Action prediction, confidence estimation, and auxiliary heads.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class ActionHead(nn.Module):
    """
    Predicts action distribution over discrete action space.
    """
    
    def __init__(self,
                 input_dim: int = 384,
                 hidden_dim: int = 256,
                 num_actions: int = 18):
        super().__init__()
        
        self.num_actions = num_actions
        
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_actions),
        )
        
    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            context: Context features (B, input_dim)
            
        Returns:
            Action logits (B, num_actions)
        """
        return self.head(context)
    
    def predict(self, context: torch.Tensor,
                temperature: float = 1.0,
                sample: bool = False) -> torch.Tensor:
        """
        Predict action with optional sampling.
        
        Args:
            context: Context features (B, input_dim)
            temperature: Softmax temperature
            sample: If True, sample from distribution; else argmax
            
        Returns:
            Predicted actions (B,)
        """
        logits = self.forward(context) / temperature
        
        if sample:
            probs = F.softmax(logits, dim=-1)
            return torch.multinomial(probs, 1).squeeze(-1)
        else:
            return logits.argmax(dim=-1)


class ConfidenceHead(nn.Module):
    """
    Estimates prediction confidence.
    """
    
    def __init__(self, input_dim: int = 384, hidden_dim: int = 128):
        super().__init__()
        
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        
    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            context: Context features (B, input_dim)
            
        Returns:
            Confidence scores (B, 1)
        """
        return self.head(context)


class AuxiliaryHeads(nn.Module):
    """
    Auxiliary prediction heads for better representations.
    
    Predicts:
    - Traversability from decision features
    - Collision probability
    - Speed recommendation
    """
    
    def __init__(self, input_dim: int = 384, hidden_dim: int = 128):
        super().__init__()
        
        # Traversability prediction
        self.traversability = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        
        # Collision prediction
        self.collision = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        
        # Speed recommendation (3 classes: slow, medium, fast)
        self.speed = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 3),
        )
        
    def forward(self, context: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            context: Context features (B, input_dim)
            
        Returns:
            Dictionary of auxiliary predictions
        """
        return {
            'traversability': self.traversability(context),
            'collision': self.collision(context),
            'speed': self.speed(context),
        }


class OutputHeads(nn.Module):
    """
    Combined output heads for decision transformer.
    """
    
    def __init__(self,
                 input_dim: int = 384,
                 hidden_dim: int = 256,
                 num_actions: int = 18,
                 use_auxiliary: bool = True):
        super().__init__()
        
        self.action_head = ActionHead(input_dim, hidden_dim, num_actions)
        self.confidence_head = ConfidenceHead(input_dim, hidden_dim // 2)
        
        self.use_auxiliary = use_auxiliary
        if use_auxiliary:
            self.auxiliary_heads = AuxiliaryHeads(input_dim, hidden_dim // 2)
            
    def forward(self, context: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            context: Context features (B, input_dim)
            
        Returns:
            Dictionary of all predictions
        """
        outputs = {
            'action_logits': self.action_head(context),
            'confidence': self.confidence_head(context),
        }
        
        if self.use_auxiliary:
            aux = self.auxiliary_heads(context)
            outputs.update(aux)
            
        return outputs
    
    def predict_action(self, context: torch.Tensor,
                       temperature: float = 1.0,
                       sample: bool = False,
                       mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Predict action with optional masking of invalid actions.
        
        Args:
            context: Context features (B, input_dim)
            temperature: Softmax temperature
            sample: If True, sample; else argmax
            mask: Optional mask for invalid actions (B, num_actions)
            
        Returns:
            Predicted actions (B,)
        """
        logits = self.action_head(context) / temperature
        
        if mask is not None:
            logits = logits.masked_fill(~mask, float('-inf'))
            
        if sample:
            probs = F.softmax(logits, dim=-1)
            return torch.multinomial(probs, 1).squeeze(-1)
        else:
            return logits.argmax(dim=-1)


def test_output_heads():
    print("Testing Output Heads...")
    
    heads = OutputHeads(
        input_dim=384,
        hidden_dim=256,
        num_actions=18,
        use_auxiliary=True
    )
    
    context = torch.randn(4, 384)
    
    outputs = heads(context)
    
    print(f"Action logits shape: {outputs['action_logits'].shape}")
    print(f"Confidence shape: {outputs['confidence'].shape}")
    print(f"Traversability shape: {outputs['traversability'].shape}")
    print(f"Collision shape: {outputs['collision'].shape}")
    print(f"Speed shape: {outputs['speed'].shape}")
    
    assert outputs['action_logits'].shape == (4, 18)
    assert outputs['confidence'].shape == (4, 1)
    
    # Test action prediction
    actions = heads.predict_action(context, sample=False)
    print(f"Predicted actions: {actions}")
    
    print("Output Heads test passed!")


if __name__ == "__main__":
    test_output_heads()
