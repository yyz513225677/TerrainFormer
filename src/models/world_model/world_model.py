"""
Complete World Model

Combines all components: tokenizer, dynamics transformer, latent state,
and prediction heads.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from .terrain_tokenizer import TerrainTokenizer
from .dynamics_transformer import DynamicsTransformer
from .latent_state import LatentState
from .prediction_heads import PredictionHeads


class WorldModel(nn.Module):
    """
    World Model for terrain dynamics prediction.
    
    Architecture:
    1. TerrainTokenizer: BEV -> Tokens
    2. DynamicsTransformer: Process tokens
    3. LatentState: Compress to latent representation
    4. PredictionHeads: Predict future states and terrain properties
    """
    
    def __init__(self,
                 bev_channels: int = 64,
                 embed_dim: int = 512,
                 patch_size: int = 16,
                 img_size: int = 256,
                 num_transformer_layers: int = 6,
                 num_heads: int = 8,
                 num_latent_tokens: int = 64,
                 num_classes: int = 20,
                 num_future_frames: int = 5):
        """
        Args:
            bev_channels: Input BEV feature channels
            embed_dim: Transformer embedding dimension
            patch_size: Tokenizer patch size
            img_size: BEV image size
            num_transformer_layers: Number of transformer layers
            num_heads: Number of attention heads
            num_latent_tokens: Number of latent tokens
            num_classes: Number of semantic classes
            num_future_frames: Number of future frames to predict
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) ** 2
        
        # Tokenizer
        self.tokenizer = TerrainTokenizer(
            in_channels=bev_channels,
            embed_dim=embed_dim,
            patch_size=patch_size,
            img_size=img_size
        )
        
        # Dynamics Transformer
        self.transformer = DynamicsTransformer(
            embed_dim=embed_dim,
            num_layers=num_transformer_layers,
            num_heads=num_heads
        )
        
        # Latent State
        self.latent_state = LatentState(
            input_dim=embed_dim,
            latent_dim=embed_dim,
            num_latent_tokens=num_latent_tokens,
            num_output_tokens=self.num_patches
        )
        
        # Prediction Heads
        grid_size = img_size // patch_size
        self.prediction_heads = PredictionHeads(
            in_channels=embed_dim,
            hidden_channels=256,
            out_size=(img_size, img_size),
            num_classes=num_classes,
            num_future_frames=num_future_frames,
            bev_channels=bev_channels
        )
        
        # Feature reshape for prediction heads
        self.grid_size = grid_size
        
    def forward(self, bev: torch.Tensor,
                return_latent: bool = False) -> Dict[str, torch.Tensor]:
        """
        Args:
            bev: BEV features (B, C, H, W)
            return_latent: Whether to return latent state
            
        Returns:
            Dictionary containing predictions and optionally latent state
        """
        B = bev.shape[0]
        
        # Tokenize
        tokens = self.tokenizer(bev)
        
        # Process through transformer
        tokens = self.transformer(tokens)
        
        # Get latent state
        latent, decoded, global_feat = self.latent_state(tokens)
        
        # Reshape decoded tokens to spatial format for prediction heads
        decoded_spatial = decoded.transpose(1, 2).view(
            B, self.embed_dim, self.grid_size, self.grid_size
        )
        
        # Get predictions
        predictions = self.prediction_heads(decoded_spatial)
        predictions['global_feature'] = global_feat
        
        if return_latent:
            predictions['latent'] = latent
            predictions['tokens'] = tokens
            
        return predictions
    
    def encode(self, bev: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode BEV to latent state.
        
        Args:
            bev: BEV features (B, C, H, W)
            
        Returns:
            latent: Latent state (B, num_latent_tokens, embed_dim)
            global_feat: Global feature (B, embed_dim)
        """
        tokens = self.tokenizer(bev)
        tokens = self.transformer(tokens)
        latent = self.latent_state.encode(tokens)
        global_feat = self.latent_state.get_global(latent)
        
        return latent, global_feat
    
    def predict_from_latent(self, latent: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Predict from latent state.
        
        Args:
            latent: Latent state (B, num_latent_tokens, embed_dim)
            
        Returns:
            Predictions dictionary
        """
        B = latent.shape[0]
        
        decoded = self.latent_state.decode(latent)
        decoded_spatial = decoded.transpose(1, 2).view(
            B, self.embed_dim, self.grid_size, self.grid_size
        )
        
        predictions = self.prediction_heads(decoded_spatial)
        predictions['global_feature'] = self.latent_state.get_global(latent)
        
        return predictions


def test_world_model():
    print("Testing World Model...")
    
    model = WorldModel(
        bev_channels=64,
        embed_dim=512,
        patch_size=16,
        img_size=256,
        num_transformer_layers=6,
        num_heads=8,
        num_latent_tokens=64,
        num_classes=20,
        num_future_frames=5
    )
    
    bev = torch.randn(2, 64, 256, 256)
    
    # Full forward pass
    outputs = model(bev, return_latent=True)
    
    print(f"Traversability: {outputs['traversability'].shape}")
    print(f"Elevation: {outputs['elevation'].shape}")
    print(f"Semantics: {outputs['semantics'].shape}")
    print(f"Future: {outputs['future'].shape}")
    print(f"Global feature: {outputs['global_feature'].shape}")
    print(f"Latent: {outputs['latent'].shape}")
    
    # Test encode
    latent, global_feat = model.encode(bev)
    print(f"Encoded latent: {latent.shape}")
    print(f"Encoded global: {global_feat.shape}")
    
    # Test predict from latent
    preds = model.predict_from_latent(latent)
    print(f"Predictions from latent - traversability: {preds['traversability'].shape}")
    
    print("World Model test passed!")


if __name__ == "__main__":
    test_world_model()
