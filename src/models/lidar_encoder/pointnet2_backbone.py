"""
PointNet++ Backbone for LiDAR Point Cloud Processing

Implements Set Abstraction and Feature Propagation layers for hierarchical
point cloud feature learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import math


def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """
    Farthest point sampling to select representative points.
    
    Args:
        xyz: Point cloud coordinates (B, N, 3)
        npoint: Number of points to sample
        
    Returns:
        Indices of sampled points (B, npoint)
    """
    device = xyz.device
    B, N, C = xyz.shape
    
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10
    
    # Random initial point
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)
    
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)
        distance = torch.min(distance, dist)
        farthest = torch.max(distance, dim=-1)[1]
        
    return centroids


def query_ball_point(radius: float, nsample: int, xyz: torch.Tensor, 
                      new_xyz: torch.Tensor) -> torch.Tensor:
    """
    Ball query to find neighbors within radius.
    
    Args:
        radius: Search radius
        nsample: Maximum number of neighbors
        xyz: All points (B, N, 3)
        new_xyz: Query points (B, S, 3)
        
    Returns:
        Indices of neighbors (B, S, nsample)
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    
    group_idx = torch.arange(N, dtype=torch.long, device=device).view(1, 1, N).repeat(B, S, 1)
    
    # Compute distances
    sqrdists = torch.sum((new_xyz.unsqueeze(2) - xyz.unsqueeze(1)) ** 2, dim=-1)
    
    # Mask points outside radius
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    
    # Handle case where fewer than nsample points in ball
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat(1, 1, nsample)
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    
    return group_idx


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    Index into point cloud using given indices.
    
    Args:
        points: Point cloud (B, N, C)
        idx: Indices (B, S) or (B, S, K)
        
    Returns:
        Indexed points (B, S, C) or (B, S, K, C)
    """
    device = points.device
    B = points.shape[0]
    
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    
    batch_indices = torch.arange(B, dtype=torch.long, device=device).view(view_shape).repeat(repeat_shape)
    
    new_points = points[batch_indices, idx, :]
    return new_points


class SharedMLP(nn.Module):
    """Shared MLP applied to each point independently."""
    
    def __init__(self, channels: List[int], bn: bool = True):
        super().__init__()
        
        layers = []
        for i in range(len(channels) - 1):
            layers.append(nn.Conv1d(channels[i], channels[i + 1], 1))
            if bn:
                layers.append(nn.BatchNorm1d(channels[i + 1]))
            layers.append(nn.ReLU(inplace=True))
            
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features (B, C, N)
        Returns:
            Output features (B, C', N)
        """
        return self.mlp(x)


class SetAbstraction(nn.Module):
    """
    Set Abstraction module from PointNet++.
    
    Samples points, groups neighbors, and applies PointNet to each group.
    """
    
    def __init__(self, npoint: int, radius: float, nsample: int, 
                 in_channel: int, mlp: List[int], group_all: bool = False):
        """
        Args:
            npoint: Number of points to sample
            radius: Ball query radius
            nsample: Number of samples in each ball
            in_channel: Input feature dimension
            mlp: MLP channel dimensions
            group_all: If True, group all points (global feature)
        """
        super().__init__()
        
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all
        
        # MLP layers
        self.mlp = SharedMLP([in_channel + 3] + mlp)
        
    def forward(self, xyz: torch.Tensor, 
                features: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            xyz: Point coordinates (B, N, 3)
            features: Point features (B, N, C) or None
            
        Returns:
            new_xyz: Sampled point coordinates (B, npoint, 3)
            new_features: Aggregated features (B, npoint, C')
        """
        B, N, _ = xyz.shape
        
        if self.group_all:
            # Global feature
            new_xyz = torch.zeros(B, 1, 3, device=xyz.device)
            grouped_xyz = xyz.view(B, 1, N, 3)
            
            if features is not None:
                grouped_features = features.view(B, 1, N, -1)
                grouped_features = torch.cat([grouped_xyz, grouped_features], dim=-1)
            else:
                grouped_features = grouped_xyz
        else:
            # Sample and group
            fps_idx = farthest_point_sample(xyz, self.npoint)
            new_xyz = index_points(xyz, fps_idx)
            
            idx = query_ball_point(self.radius, self.nsample, xyz, new_xyz)
            grouped_xyz = index_points(xyz, idx)
            grouped_xyz = grouped_xyz - new_xyz.unsqueeze(2)  # Normalize to local coords
            
            if features is not None:
                grouped_features = index_points(features, idx)
                grouped_features = torch.cat([grouped_xyz, grouped_features], dim=-1)
            else:
                grouped_features = grouped_xyz
        
        # Apply MLP: (B, npoint, nsample, C) -> (B, C', npoint, nsample) -> (B, C', npoint)
        grouped_features = grouped_features.permute(0, 3, 1, 2).contiguous()
        B, C, S, K = grouped_features.shape
        grouped_features = grouped_features.view(B, C, S * K)
        
        new_features = self.mlp(grouped_features)
        new_features = new_features.view(B, -1, S, K)
        new_features = torch.max(new_features, dim=-1)[0]  # Max pooling
        new_features = new_features.permute(0, 2, 1).contiguous()
        
        return new_xyz, new_features


class FeaturePropagation(nn.Module):
    """
    Feature Propagation module from PointNet++.
    
    Propagates features from subsampled points back to original points.
    """
    
    def __init__(self, in_channel: int, mlp: List[int]):
        """
        Args:
            in_channel: Input feature dimension (from both levels)
            mlp: MLP channel dimensions
        """
        super().__init__()
        self.mlp = SharedMLP([in_channel] + mlp)
        
    def forward(self, xyz1: torch.Tensor, xyz2: torch.Tensor,
                features1: Optional[torch.Tensor], 
                features2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            xyz1: Target points (B, N, 3)
            xyz2: Source points (B, S, 3)
            features1: Target features (B, N, C1) or None
            features2: Source features (B, S, C2)
            
        Returns:
            Interpolated features (B, N, C')
        """
        B, N, _ = xyz1.shape
        _, S, _ = xyz2.shape
        
        if S == 1:
            # Global feature - repeat for all points
            interpolated = features2.repeat(1, N, 1)
        else:
            # Distance-weighted interpolation
            dists = torch.sum((xyz1.unsqueeze(2) - xyz2.unsqueeze(1)) ** 2, dim=-1)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # 3 nearest neighbors
            
            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=-1, keepdim=True)
            weight = dist_recip / norm
            
            interpolated = torch.sum(
                index_points(features2, idx) * weight.unsqueeze(-1), dim=2
            )
        
        # Concatenate with skip connection
        if features1 is not None:
            interpolated = torch.cat([interpolated, features1], dim=-1)
            
        # Apply MLP
        interpolated = interpolated.permute(0, 2, 1).contiguous()
        interpolated = self.mlp(interpolated)
        interpolated = interpolated.permute(0, 2, 1).contiguous()
        
        return interpolated


class PointNet2Backbone(nn.Module):
    """
    PointNet++ backbone for LiDAR point cloud encoding.
    
    Produces both point-wise features and a global feature vector.
    """
    
    def __init__(self, 
                 in_channels: int = 4,
                 sa_configs: Optional[List[dict]] = None,
                 fp_configs: Optional[List[dict]] = None,
                 global_dim: int = 1024):
        """
        Args:
            in_channels: Input feature dimension (default 4 for x,y,z,intensity)
            sa_configs: List of Set Abstraction layer configs
            fp_configs: List of Feature Propagation layer configs
            global_dim: Global feature dimension
        """
        super().__init__()
        
        # Default configurations
        if sa_configs is None:
            sa_configs = [
                {"npoint": 16384, "radius": 0.5, "nsample": 32, "mlp": [32, 32, 64]},
                {"npoint": 4096, "radius": 1.0, "nsample": 32, "mlp": [64, 64, 128]},
                {"npoint": 1024, "radius": 2.0, "nsample": 32, "mlp": [128, 128, 256]},
                {"npoint": 256, "radius": 4.0, "nsample": 32, "mlp": [256, 256, 512]},
            ]
            
        if fp_configs is None:
            fp_configs = [
                {"mlp": [256, 256]},
                {"mlp": [256, 128]},
                {"mlp": [128, 128, 128]},
                {"mlp": [128, 128, 128]},
            ]
        
        self.in_channels = in_channels
        
        # Build Set Abstraction layers
        self.sa_layers = nn.ModuleList()
        in_ch = in_channels - 3  # Exclude xyz
        
        for i, cfg in enumerate(sa_configs):
            self.sa_layers.append(SetAbstraction(
                npoint=cfg["npoint"],
                radius=cfg["radius"],
                nsample=cfg["nsample"],
                in_channel=in_ch,
                mlp=cfg["mlp"]
            ))
            in_ch = cfg["mlp"][-1]
            
        # Global feature extraction
        self.sa_global = SetAbstraction(
            npoint=None,
            radius=None,
            nsample=None,
            in_channel=in_ch,
            mlp=[512, global_dim],
            group_all=True
        )
        
        # Build Feature Propagation layers
        self.fp_layers = nn.ModuleList()
        
        # Reverse order for upsampling
        sa_out_channels = [in_channels - 3] + [cfg["mlp"][-1] for cfg in sa_configs]
        
        prev_ch = global_dim
        for i, cfg in enumerate(fp_configs):
            skip_ch = sa_out_channels[-(i + 2)]
            in_ch = prev_ch + skip_ch
            self.fp_layers.append(FeaturePropagation(
                in_channel=in_ch,
                mlp=cfg["mlp"]
            ))
            prev_ch = cfg["mlp"][-1]
            
        self.output_dim = prev_ch
        self.global_dim = global_dim
        
    def forward(self, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            points: Input point cloud (B, N, C) where C >= 4
            
        Returns:
            point_features: Per-point features (B, N, D)
            global_features: Global feature vector (B, global_dim)
        """
        B, N, C = points.shape
        
        xyz = points[:, :, :3]
        features = points[:, :, 3:] if C > 3 else None
        
        # Encoder: hierarchical feature learning
        xyz_list = [xyz]
        features_list = [features]
        
        for sa_layer in self.sa_layers:
            xyz, features = sa_layer(xyz, features)
            xyz_list.append(xyz)
            features_list.append(features)
            
        # Global feature
        _, global_features = self.sa_global(xyz, features)
        global_features = global_features.squeeze(1)
        
        # Decoder: feature propagation
        for i, fp_layer in enumerate(self.fp_layers):
            # Get source and target
            src_idx = -(i + 1)
            tgt_idx = -(i + 2)
            
            xyz_src = xyz_list[src_idx]
            xyz_tgt = xyz_list[tgt_idx]
            feat_src = features_list[src_idx]
            feat_tgt = features_list[tgt_idx]
            
            features_list[tgt_idx] = fp_layer(xyz_tgt, xyz_src, feat_tgt, feat_src)
            
        point_features = features_list[0]
        
        return point_features, global_features
    
    
def test_pointnet2():
    """Test PointNet++ backbone."""
    print("Testing PointNet++ backbone...")
    
    model = PointNet2Backbone(in_channels=4)
    
    # Random point cloud
    B, N = 2, 16384
    points = torch.randn(B, N, 4)
    
    point_feat, global_feat = model(points)
    
    print(f"Input shape: {points.shape}")
    print(f"Point features shape: {point_feat.shape}")
    print(f"Global features shape: {global_feat.shape}")
    
    assert point_feat.shape == (B, N, 128), f"Expected (2, 16384, 128), got {point_feat.shape}"
    assert global_feat.shape == (B, 1024), f"Expected (2, 1024), got {global_feat.shape}"
    
    print("PointNet++ backbone test passed!")
    

if __name__ == "__main__":
    test_pointnet2()
