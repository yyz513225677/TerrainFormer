"""
Voxel-based LiDAR Encoder

Alternative encoding path using voxelization for efficient processing
of large point clouds.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List
import math


class VoxelFeatureExtractor(nn.Module):
    """
    Extracts features from voxelized point clouds using PointNet-style processing.
    """
    
    def __init__(self, 
                 in_channels: int = 4,
                 hidden_channels: int = 64,
                 out_channels: int = 128,
                 max_points_per_voxel: int = 32):
        """
        Args:
            in_channels: Input feature dimension per point
            hidden_channels: Hidden layer dimension
            out_channels: Output feature dimension per voxel
            max_points_per_voxel: Maximum points to consider per voxel
        """
        super().__init__()
        
        self.max_points = max_points_per_voxel
        
        # Point-wise MLP
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, out_channels),
        )
        
        self.out_channels = out_channels
        
    def forward(self, voxel_features: torch.Tensor, 
                voxel_num_points: torch.Tensor) -> torch.Tensor:
        """
        Args:
            voxel_features: Features of points in each voxel (num_voxels, max_points, C)
            voxel_num_points: Number of points in each voxel (num_voxels,)
            
        Returns:
            Voxel features (num_voxels, out_channels)
        """
        num_voxels, max_points, C = voxel_features.shape
        
        # Flatten for batch processing
        flat_features = voxel_features.view(-1, C)
        flat_features = self.mlp(flat_features)
        flat_features = flat_features.view(num_voxels, max_points, -1)
        
        # Create mask for valid points
        mask = torch.arange(max_points, device=voxel_features.device)
        mask = mask.unsqueeze(0) < voxel_num_points.unsqueeze(1)
        mask = mask.unsqueeze(-1).float()
        
        # Masked max pooling
        flat_features = flat_features * mask
        flat_features[~mask.bool().expand_as(flat_features)] = float('-inf')
        voxel_out = flat_features.max(dim=1)[0]
        
        return voxel_out


class SparseConv3D(nn.Module):
    """
    Sparse 3D convolution for processing voxelized point clouds.
    Uses scatter operations for efficiency.
    """
    
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: int = 1):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Convolution weights
        self.weight = nn.Parameter(
            torch.randn(kernel_size ** 3, in_channels, out_channels) * 0.01
        )
        self.bias = nn.Parameter(torch.zeros(out_channels))
        
        self.bn = nn.BatchNorm1d(out_channels)
        
    def forward(self, features: torch.Tensor, 
                coords: torch.Tensor,
                spatial_shape: Tuple[int, int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features: Voxel features (N, C)
            coords: Voxel coordinates (N, 4) - batch_idx, z, y, x
            spatial_shape: (D, H, W) spatial dimensions
            
        Returns:
            out_features: Output features (M, C')
            out_coords: Output coordinates (M, 4)
        """
        # For simplicity, convert to dense, apply conv, convert back
        # In production, use spconv or MinkowskiEngine
        
        device = features.device
        N, C = features.shape
        batch_size = coords[:, 0].max().item() + 1
        D, H, W = spatial_shape
        
        # Create dense tensor
        dense = torch.zeros(batch_size, C, D, H, W, device=device)
        
        for i in range(N):
            b, z, y, x = coords[i].long()
            dense[b, :, z, y, x] = features[i]
            
        # Apply 3D convolution
        conv = nn.Conv3d(
            self.in_channels, self.out_channels,
            self.kernel_size, self.stride, self.padding
        ).to(device)
        
        with torch.no_grad():
            conv.weight.copy_(self.weight.view(
                self.out_channels, self.in_channels,
                self.kernel_size, self.kernel_size, self.kernel_size
            ))
            conv.bias.copy_(self.bias)
            
        dense_out = conv(dense)
        dense_out = F.relu(dense_out)
        
        # Convert back to sparse
        out_coords_list = []
        out_features_list = []
        
        B, C_out, D_out, H_out, W_out = dense_out.shape
        
        for b in range(B):
            for z in range(D_out):
                for y in range(H_out):
                    for x in range(W_out):
                        feat = dense_out[b, :, z, y, x]
                        if feat.abs().sum() > 1e-6:
                            out_coords_list.append(torch.tensor([b, z, y, x]))
                            out_features_list.append(feat)
                            
        if len(out_features_list) > 0:
            out_features = torch.stack(out_features_list)
            out_coords = torch.stack(out_coords_list).to(device)
        else:
            out_features = torch.zeros(0, self.out_channels, device=device)
            out_coords = torch.zeros(0, 4, device=device)
            
        return out_features, out_coords


class VoxelEncoder(nn.Module):
    """
    Complete voxel-based encoder for LiDAR point clouds.
    
    Voxelizes point cloud, extracts voxel features, and applies
    sparse 3D convolutions.
    """
    
    def __init__(self,
                 voxel_size: List[float] = [0.2, 0.2, 0.2],
                 point_cloud_range: List[float] = [-50, -50, -3, 50, 50, 5],
                 max_voxels: int = 40000,
                 max_points_per_voxel: int = 32,
                 in_channels: int = 4,
                 hidden_channels: int = 64,
                 out_channels: int = 128):
        """
        Args:
            voxel_size: Voxel dimensions [x, y, z] in meters
            point_cloud_range: [x_min, y_min, z_min, x_max, y_max, z_max]
            max_voxels: Maximum number of voxels
            max_points_per_voxel: Maximum points per voxel
            in_channels: Input point features
            hidden_channels: Hidden dimension
            out_channels: Output voxel features
        """
        super().__init__()
        
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.max_voxels = max_voxels
        self.max_points_per_voxel = max_points_per_voxel
        
        # Compute spatial shape
        self.spatial_shape = [
            int((point_cloud_range[i + 3] - point_cloud_range[i]) / voxel_size[i])
            for i in range(3)
        ]
        
        # Voxel feature extractor
        self.vfe = VoxelFeatureExtractor(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            max_points_per_voxel=max_points_per_voxel
        )
        
        self.out_channels = out_channels
        
    def voxelize(self, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Voxelize point cloud.
        
        Args:
            points: Point cloud (N, C)
            
        Returns:
            voxel_features: (num_voxels, max_points, C)
            voxel_coords: (num_voxels, 3) - z, y, x indices
            voxel_num_points: (num_voxels,)
        """
        device = points.device
        N, C = points.shape
        
        # Compute voxel indices
        points_xyz = points[:, :3]
        
        voxel_idx = ((points_xyz - torch.tensor(self.point_cloud_range[:3], device=device)) 
                     / torch.tensor(self.voxel_size, device=device)).long()
        
        # Filter out-of-range points
        valid_mask = (
            (voxel_idx[:, 0] >= 0) & (voxel_idx[:, 0] < self.spatial_shape[0]) &
            (voxel_idx[:, 1] >= 0) & (voxel_idx[:, 1] < self.spatial_shape[1]) &
            (voxel_idx[:, 2] >= 0) & (voxel_idx[:, 2] < self.spatial_shape[2])
        )
        
        valid_points = points[valid_mask]
        valid_idx = voxel_idx[valid_mask]
        
        # Create unique voxel keys
        D, H, W = self.spatial_shape
        voxel_keys = valid_idx[:, 0] * H * W + valid_idx[:, 1] * W + valid_idx[:, 2]
        
        # Get unique voxels
        unique_keys, inverse_indices = torch.unique(voxel_keys, return_inverse=True)
        num_voxels = min(len(unique_keys), self.max_voxels)
        
        # Initialize outputs
        voxel_features = torch.zeros(
            num_voxels, self.max_points_per_voxel, C, device=device
        )
        voxel_num_points = torch.zeros(num_voxels, dtype=torch.long, device=device)
        
        # Fill voxels
        for i in range(len(valid_points)):
            voxel_id = inverse_indices[i].item()
            if voxel_id >= num_voxels:
                continue
                
            point_idx = voxel_num_points[voxel_id].item()
            if point_idx < self.max_points_per_voxel:
                voxel_features[voxel_id, point_idx] = valid_points[i]
                voxel_num_points[voxel_id] += 1
                
        # Compute voxel coordinates
        voxel_coords = torch.zeros(num_voxels, 3, dtype=torch.long, device=device)
        for i, key in enumerate(unique_keys[:num_voxels]):
            key = key.item()
            z = key // (H * W)
            y = (key % (H * W)) // W
            x = key % W
            voxel_coords[i] = torch.tensor([z, y, x])
            
        return voxel_features, voxel_coords, voxel_num_points
    
    def forward(self, points: torch.Tensor, 
                batch_indices: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            points: Point cloud (N, C) or batched (B, N, C)
            batch_indices: Batch index for each point (N,) if points is (N, C)
            
        Returns:
            voxel_features: (num_voxels, out_channels)
            voxel_coords: (num_voxels, 4) - batch_idx, z, y, x
        """
        if points.dim() == 3:
            # Batched input
            B, N, C = points.shape
            device = points.device
            
            all_features = []
            all_coords = []
            
            for b in range(B):
                vf, vc, vnp = self.voxelize(points[b])
                features = self.vfe(vf, vnp)
                
                # Add batch index
                batch_col = torch.full((len(vc), 1), b, dtype=torch.long, device=device)
                coords = torch.cat([batch_col, vc], dim=1)
                
                all_features.append(features)
                all_coords.append(coords)
                
            voxel_features = torch.cat(all_features, dim=0)
            voxel_coords = torch.cat(all_coords, dim=0)
            
        else:
            # Single batch
            vf, vc, vnp = self.voxelize(points)
            voxel_features = self.vfe(vf, vnp)
            
            batch_col = torch.zeros(len(vc), 1, dtype=torch.long, device=points.device)
            voxel_coords = torch.cat([batch_col, vc], dim=1)
            
        return voxel_features, voxel_coords


def test_voxel_encoder():
    """Test voxel encoder."""
    print("Testing Voxel Encoder...")
    
    encoder = VoxelEncoder(
        voxel_size=[0.5, 0.5, 0.5],
        point_cloud_range=[-50, -50, -3, 50, 50, 5],
        max_voxels=10000,
        max_points_per_voxel=32,
        in_channels=4,
        out_channels=128
    )
    
    # Random point cloud
    B, N = 2, 16384
    points = torch.randn(B, N, 4)
    points[:, :, :3] *= 30  # Scale to reasonable range
    
    features, coords = encoder(points)
    
    print(f"Input shape: {points.shape}")
    print(f"Output features shape: {features.shape}")
    print(f"Output coords shape: {coords.shape}")
    print(f"Number of voxels: {len(features)}")
    
    print("Voxel Encoder test passed!")
    

if __name__ == "__main__":
    test_voxel_encoder()
