"""
Terrain Tokenizer for World Model

Converts BEV features into discrete tokens for transformer processing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class PatchEmbedding(nn.Module):
    """
    Converts BEV feature map into patch embeddings.
    """
    
    def __init__(self,
                 in_channels: int = 64,
                 embed_dim: int = 512,
                 patch_size: int = 16,
                 img_size: int = 256):
        super().__init__()
        
        self.patch_size = patch_size
        self.img_size = img_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class LearnedPositionalEncoding(nn.Module):
    """Learned positional encoding for BEV patches."""
    
    def __init__(self, embed_dim: int, num_patches: int):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pos_embed[:, :x.size(1)]


class TerrainTokenizer(nn.Module):
    """
    Tokenizes BEV features for transformer processing.
    """
    
    def __init__(self,
                 in_channels: int = 64,
                 embed_dim: int = 512,
                 patch_size: int = 16,
                 img_size: int = 256,
                 max_temporal_len: int = 10):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        self.patch_embed = PatchEmbedding(
            in_channels=in_channels,
            embed_dim=embed_dim,
            patch_size=patch_size,
            img_size=img_size
        )
        
        self.pos_embed = LearnedPositionalEncoding(embed_dim, self.num_patches)
        self.temporal_embed = nn.Parameter(torch.zeros(1, max_temporal_len, 1, embed_dim))
        nn.init.trunc_normal_(self.temporal_embed, std=0.02)
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, bev: torch.Tensor, 
                temporal_idx: Optional[torch.Tensor] = None) -> torch.Tensor:
        is_sequence = bev.dim() == 5
        
        if is_sequence:
            B, T, C, H, W = bev.shape
            bev = bev.view(B * T, C, H, W)
        else:
            B = bev.shape[0]
            T = 1
            
        tokens = self.patch_embed(bev)
        tokens = self.pos_embed(tokens)
        
        cls_tokens = self.cls_token.expand(B * T, -1, -1)
        tokens = torch.cat([cls_tokens, tokens], dim=1)
        
        if is_sequence:
            tokens = tokens.view(B, T, -1, self.embed_dim)
            tokens = tokens + self.temporal_embed[:, :T]
            tokens = tokens.view(B, T * (1 + self.num_patches), self.embed_dim)
            
        tokens = self.dropout(tokens)
        return tokens
    
    def decode_to_bev(self, tokens: torch.Tensor) -> torch.Tensor:
        patch_tokens = tokens[:, 1:]
        B = patch_tokens.shape[0]
        grid_size = int(math.sqrt(self.num_patches))
        bev = patch_tokens.transpose(1, 2)
        bev = bev.view(B, self.embed_dim, grid_size, grid_size)
        return bev


def test_terrain_tokenizer():
    print("Testing Terrain Tokenizer...")
    tokenizer = TerrainTokenizer(in_channels=64, embed_dim=512, patch_size=16, img_size=256)
    bev = torch.randn(2, 64, 256, 256)
    tokens = tokenizer(bev)
    print(f"Single frame tokens shape: {tokens.shape}")
    print("Terrain Tokenizer test passed!")


if __name__ == "__main__":
    test_terrain_tokenizer()
