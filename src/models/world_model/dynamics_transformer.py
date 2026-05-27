"""
Dynamics Transformer for World Model

Learns terrain dynamics and predicts future states.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with optional causal masking."""
    
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 dropout: float = 0.1,
                 bias: bool = True):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, C = x.shape
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
            
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        
        return x


class MLP(nn.Module):
    """Feed-forward network."""
    
    def __init__(self,
                 in_features: int,
                 hidden_features: Optional[int] = None,
                 out_features: Optional[int] = None,
                 dropout: float = 0.1):
        super().__init__()
        
        hidden_features = hidden_features or in_features * 4
        out_features = out_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer block with pre-norm."""
    
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 mlp_ratio: float = 4.0,
                 dropout: float = 0.1):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, int(embed_dim * mlp_ratio), dropout=dropout)
        
    def forward(self, x: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.mlp(self.norm2(x))
        return x


class DynamicsTransformer(nn.Module):
    """
    Transformer for learning terrain dynamics.
    
    Processes tokenized BEV features and learns to predict
    future terrain states.
    """
    
    def __init__(self,
                 embed_dim: int = 512,
                 num_layers: int = 6,
                 num_heads: int = 8,
                 mlp_ratio: float = 4.0,
                 dropout: float = 0.1,
                 use_causal_mask: bool = False):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.use_causal_mask = use_causal_mask
        
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
    def _create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create causal attention mask."""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
        
    def forward(self, tokens: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            tokens: Input tokens (B, N, embed_dim)
            mask: Optional attention mask
            
        Returns:
            Transformed tokens (B, N, embed_dim)
        """
        if self.use_causal_mask and mask is None:
            mask = self._create_causal_mask(tokens.size(1), tokens.device)
            
        for block in self.blocks:
            tokens = block(tokens, mask)
            
        tokens = self.norm(tokens)
        return tokens
    
    def get_attention_weights(self, tokens: torch.Tensor) -> list:
        """Extract attention weights from all layers."""
        attention_weights = []
        
        for block in self.blocks:
            # Forward through attention to get weights
            B, N, C = tokens.shape
            qkv = block.attn.qkv(block.norm1(tokens))
            qkv = qkv.reshape(B, N, 3, block.attn.num_heads, block.attn.head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            
            attn = (q @ k.transpose(-2, -1)) * block.attn.scale
            attn = attn.softmax(dim=-1)
            attention_weights.append(attn.detach())
            
            # Continue forward pass
            tokens = block(tokens)
            
        return attention_weights


def test_dynamics_transformer():
    print("Testing Dynamics Transformer...")
    
    transformer = DynamicsTransformer(
        embed_dim=512,
        num_layers=6,
        num_heads=8
    )
    
    tokens = torch.randn(2, 257, 512)
    output = transformer(tokens)
    print(f"Output shape: {output.shape}")
    assert output.shape == tokens.shape
    
    print("Dynamics Transformer test passed!")


if __name__ == "__main__":
    test_dynamics_transformer()
