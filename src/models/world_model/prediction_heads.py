"""
Prediction Heads for World Model

Various prediction heads for terrain understanding:
- Future LiDAR prediction
- Traversability estimation
- Elevation prediction
- Semantic segmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import math


class TraversabilityHead(nn.Module):
    """
    Predicts traversability scores for each BEV cell.
    Output is probability of being traversable.
    """
    
    def __init__(self,
                 in_channels: int = 512,
                 hidden_channels: int = 256,
                 out_size: Tuple[int, int] = (256, 256)):
        super().__init__()
        
        self.out_size = out_size
        
        # Upsampling decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels, hidden_channels, 4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(hidden_channels, hidden_channels, 4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(hidden_channels, hidden_channels // 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_channels // 2),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(hidden_channels // 2, hidden_channels // 4, 4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_channels // 4),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(hidden_channels // 4, 1, 1),
        )
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: BEV features (B, C, H, W)
            
        Returns:
            Traversability map (B, 1, out_H, out_W)
        """
        out = self.decoder(features)
        out = F.interpolate(out, size=self.out_size, mode='bilinear', align_corners=False)
        return torch.sigmoid(out)


from typing import Tuple


class ElevationHead(nn.Module):
    """
    Predicts elevation map from BEV features.
    """
    
    def __init__(self,
                 in_channels: int = 512,
                 hidden_channels: int = 256,
                 out_size: Tuple[int, int] = (256, 256)):
        super().__init__()
        
        self.out_size = out_size
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels, hidden_channels, 4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(hidden_channels, hidden_channels, 4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(hidden_channels, hidden_channels // 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_channels // 2),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(hidden_channels // 2, hidden_channels // 4, 4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_channels // 4),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(hidden_channels // 4, 1, 1),
        )
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: BEV features (B, C, H, W)
            
        Returns:
            Elevation map (B, 1, out_H, out_W)
        """
        out = self.decoder(features)
        out = F.interpolate(out, size=self.out_size, mode='bilinear', align_corners=False)
        return out


class SemanticHead(nn.Module):
    """
    Predicts semantic segmentation map.
    """
    
    def __init__(self,
                 in_channels: int = 512,
                 hidden_channels: int = 256,
                 num_classes: int = 20,
                 out_size: Tuple[int, int] = (256, 256)):
        super().__init__()
        
        self.out_size = out_size
        self.num_classes = num_classes
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels, hidden_channels, 4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(hidden_channels, hidden_channels, 4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(hidden_channels, hidden_channels // 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_channels // 2),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(hidden_channels // 2, hidden_channels // 4, 4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_channels // 4),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(hidden_channels // 4, num_classes, 1),
        )
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: BEV features (B, C, H, W)
            
        Returns:
            Semantic logits (B, num_classes, out_H, out_W)
        """
        out = self.decoder(features)
        out = F.interpolate(out, size=self.out_size, mode='bilinear', align_corners=False)
        return out


class FuturePredictionHead(nn.Module):
    """
    Predicts future BEV states.
    """
    
    def __init__(self,
                 in_channels: int = 512,
                 out_channels: int = 64,
                 hidden_channels: int = 256,
                 num_future_frames: int = 5):
        super().__init__()
        
        self.num_future_frames = num_future_frames
        
        # Predict all future frames at once
        self.predictor = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(hidden_channels, out_channels * num_future_frames, 1),
        )
        
        self.out_channels = out_channels
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: BEV features (B, C, H, W)
            
        Returns:
            Future BEV predictions (B, num_frames, out_C, H, W)
        """
        B, C, H, W = features.shape
        out = self.predictor(features)
        out = out.view(B, self.num_future_frames, self.out_channels, H, W)
        return out


class PredictionHeads(nn.Module):
    """
    Combined prediction heads for world model.
    """
    
    def __init__(self,
                 in_channels: int = 512,
                 hidden_channels: int = 256,
                 out_size: Tuple[int, int] = (256, 256),
                 num_classes: int = 20,
                 num_future_frames: int = 5,
                 bev_channels: int = 64):
        super().__init__()
        
        self.traversability = TraversabilityHead(
            in_channels, hidden_channels, out_size
        )
        
        self.elevation = ElevationHead(
            in_channels, hidden_channels, out_size
        )
        
        self.semantics = SemanticHead(
            in_channels, hidden_channels, num_classes, out_size
        )
        
        self.future = FuturePredictionHead(
            in_channels, bev_channels, hidden_channels, num_future_frames
        )
        
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: BEV features (B, C, H, W)
            
        Returns:
            Dictionary of predictions
        """
        return {
            'traversability': self.traversability(features),
            'elevation': self.elevation(features),
            'semantics': self.semantics(features),
            'future': self.future(features),
        }


def test_prediction_heads():
    print("Testing Prediction Heads...")
    
    heads = PredictionHeads(
        in_channels=512,
        hidden_channels=256,
        out_size=(256, 256),
        num_classes=20,
        num_future_frames=5,
        bev_channels=64
    )
    
    features = torch.randn(2, 512, 16, 16)
    outputs = heads(features)
    
    print(f"Traversability shape: {outputs['traversability'].shape}")
    print(f"Elevation shape: {outputs['elevation'].shape}")
    print(f"Semantics shape: {outputs['semantics'].shape}")
    print(f"Future shape: {outputs['future'].shape}")
    
    assert outputs['traversability'].shape == (2, 1, 256, 256)
    assert outputs['elevation'].shape == (2, 1, 256, 256)
    assert outputs['semantics'].shape == (2, 20, 256, 256)
    assert outputs['future'].shape == (2, 5, 64, 16, 16)
    
    print("Prediction Heads test passed!")


if __name__ == "__main__":
    test_prediction_heads()
