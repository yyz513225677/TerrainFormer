"""
Latent State Representation for World Model

Manages the learned latent state that captures terrain dynamics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class LatentStateEncoder(nn.Module):
    """
    Encodes transformer outputs into compact latent state.
    """
    
    def __init__(self,
                 input_dim: int = 512,
                 latent_dim: int = 512,
                 num_latent_tokens: int = 64):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.num_latent_tokens = num_latent_tokens
        
        # Learnable latent queries
        self.latent_queries = nn.Parameter(
            torch.randn(1, num_latent_tokens, latent_dim) * 0.02
        )
        
        # Cross-attention to aggregate information
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Project input if dimensions differ
        self.input_proj = nn.Linear(input_dim, latent_dim) if input_dim != latent_dim else nn.Identity()
        
        self.norm = nn.LayerNorm(latent_dim)
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 4),
            nn.GELU(),
            nn.Linear(latent_dim * 4, latent_dim),
        )
        
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: Transformer outputs (B, N, input_dim)
            
        Returns:
            Latent state (B, num_latent_tokens, latent_dim)
        """
        B = tokens.shape[0]
        
        # Project inputs
        tokens = self.input_proj(tokens)
        
        # Expand queries for batch
        queries = self.latent_queries.expand(B, -1, -1)
        
        # Cross-attention
        latent, _ = self.cross_attn(queries, tokens, tokens)
        
        # MLP
        latent = latent + self.mlp(self.norm(latent))
        
        return latent


class LatentState(nn.Module):
    """
    Complete latent state management including encoding and decoding.
    """
    
    def __init__(self,
                 input_dim: int = 512,
                 latent_dim: int = 512,
                 num_latent_tokens: int = 64,
                 num_output_tokens: int = 256):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.num_latent_tokens = num_latent_tokens
        self.num_output_tokens = num_output_tokens
        
        # Encoder
        self.encoder = LatentStateEncoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            num_latent_tokens=num_latent_tokens
        )
        
        # Decoder queries
        self.output_queries = nn.Parameter(
            torch.randn(1, num_output_tokens, latent_dim) * 0.02
        )
        
        # Decoder cross-attention
        self.decoder_attn = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        self.decoder_norm = nn.LayerNorm(latent_dim)
        self.decoder_mlp = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 4),
            nn.GELU(),
            nn.Linear(latent_dim * 4, latent_dim),
        )
        
        # Global pooling for single vector representation
        self.global_pool = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )
        
    def encode(self, tokens: torch.Tensor) -> torch.Tensor:
        """Encode to latent state."""
        return self.encoder(tokens)
    
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode latent state to output tokens.
        
        Args:
            latent: Latent state (B, num_latent_tokens, latent_dim)
            
        Returns:
            Output tokens (B, num_output_tokens, latent_dim)
        """
        B = latent.shape[0]
        
        queries = self.output_queries.expand(B, -1, -1)
        output, _ = self.decoder_attn(queries, latent, latent)
        output = output + self.decoder_mlp(self.decoder_norm(output))
        
        return output
    
    def get_global(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Get global latent vector from latent tokens.
        
        Args:
            latent: Latent state (B, num_latent_tokens, latent_dim)
            
        Returns:
            Global vector (B, latent_dim)
        """
        # Mean pooling
        pooled = latent.mean(dim=1)
        return self.global_pool(pooled)
    
    def forward(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass.
        
        Args:
            tokens: Input tokens (B, N, input_dim)
            
        Returns:
            latent: Latent state (B, num_latent_tokens, latent_dim)
            decoded: Decoded tokens (B, num_output_tokens, latent_dim)
            global_feat: Global feature (B, latent_dim)
        """
        latent = self.encode(tokens)
        decoded = self.decode(latent)
        global_feat = self.get_global(latent)
        
        return latent, decoded, global_feat


def test_latent_state():
    print("Testing Latent State...")
    
    model = LatentState(
        input_dim=512,
        latent_dim=512,
        num_latent_tokens=64,
        num_output_tokens=256
    )
    
    tokens = torch.randn(2, 257, 512)
    latent, decoded, global_feat = model(tokens)
    
    print(f"Latent shape: {latent.shape}")
    print(f"Decoded shape: {decoded.shape}")
    print(f"Global feature shape: {global_feat.shape}")
    
    assert latent.shape == (2, 64, 512)
    assert decoded.shape == (2, 256, 512)
    assert global_feat.shape == (2, 512)
    
    print("Latent State test passed!")


if __name__ == "__main__":
    test_latent_state()
