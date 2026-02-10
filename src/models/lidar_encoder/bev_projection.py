"""
Bird's Eye View (BEV) Projection for LiDAR Point Clouds

Projects 3D point cloud features into 2D BEV representation for
spatial reasoning in the world model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional


class BEVProjection(nn.Module):
    """
    Projects 3D point features to Bird's Eye View representation.
    
    Uses pillar-style projection where points are scattered into BEV grid
    and aggregated via learned pooling.
    """
    
    def __init__(self,
                 in_channels: int = 128,
                 out_channels: int = 64,
                 height: int = 256,
                 width: int = 256,
                 x_range: Tuple[float, float] = (-50, 50),
                 y_range: Tuple[float, float] = (-50, 50),
                 z_range: Tuple[float, float] = (-3, 5)):
        """
        Args:
            in_channels: Input point feature dimension
            out_channels: Output BEV feature dimension
            height: BEV grid height
            width: BEV grid width
            x_range: X axis range in meters
            y_range: Y axis range in meters
            z_range: Z axis range (for height encoding)
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.height = height
        self.width = width
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range
        
        # Compute resolution
        self.x_res = (x_range[1] - x_range[0]) / width
        self.y_res = (y_range[1] - y_range[0]) / height
        
        # Point feature transform before projection
        self.point_mlp = nn.Sequential(
            nn.Linear(in_channels + 3, out_channels),  # +3 for xyz encoding
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels, out_channels),
        )
        
        # BEV feature refinement
        self.bev_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        
    def points_to_bev_indices(self, xyz: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert 3D coordinates to BEV grid indices.
        
        Args:
            xyz: Point coordinates (N, 3)
            
        Returns:
            bev_indices: Grid indices (N, 2) - row, col
            valid_mask: Boolean mask for points within range (N,)
        """
        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        
        # Check range
        valid_mask = (
            (x >= self.x_range[0]) & (x < self.x_range[1]) &
            (y >= self.y_range[0]) & (y < self.y_range[1]) &
            (z >= self.z_range[0]) & (z < self.z_range[1])
        )
        
        # Compute grid indices
        col = ((x - self.x_range[0]) / self.x_res).long()
        row = ((y - self.y_range[0]) / self.y_res).long()
        
        # Clamp to valid range
        col = col.clamp(0, self.width - 1)
        row = row.clamp(0, self.height - 1)
        
        bev_indices = torch.stack([row, col], dim=1)
        
        return bev_indices, valid_mask
    
    def forward(self, points: torch.Tensor, 
                features: torch.Tensor) -> torch.Tensor:
        """
        Project point features to BEV.
        
        Args:
            points: Point coordinates (B, N, 3)
            features: Point features (B, N, C)
            
        Returns:
            bev_features: BEV feature map (B, out_channels, H, W)
        """
        B, N, C = features.shape
        device = features.device
        
        # Initialize BEV
        bev = torch.zeros(B, self.out_channels, self.height, self.width, device=device)
        
        for b in range(B):
            pts = points[b]  # (N, 3)
            feat = features[b]  # (N, C)
            
            # Get BEV indices
            bev_idx, valid = self.points_to_bev_indices(pts)
            
            # Filter valid points
            valid_pts = pts[valid]
            valid_feat = feat[valid]
            valid_idx = bev_idx[valid]
            
            if len(valid_pts) == 0:
                continue
                
            # Encode position
            pos_enc = self._position_encoding(valid_pts)
            combined = torch.cat([valid_feat, pos_enc], dim=-1)
            
            # Transform features
            transformed = self.point_mlp(combined)
            
            # Scatter to BEV with max pooling
            for i in range(len(transformed)):
                r, c = valid_idx[i]
                bev[b, :, r, c] = torch.max(bev[b, :, r, c], transformed[i])
                
        # Refine BEV features
        bev = self.bev_conv(bev)
        
        return bev
    
    def _position_encoding(self, xyz: torch.Tensor) -> torch.Tensor:
        """
        Create position encoding for points.
        
        Args:
            xyz: Point coordinates (N, 3)
            
        Returns:
            Position encoding (N, 3)
        """
        # Normalize to [0, 1]
        x_norm = (xyz[:, 0] - self.x_range[0]) / (self.x_range[1] - self.x_range[0])
        y_norm = (xyz[:, 1] - self.y_range[0]) / (self.y_range[1] - self.y_range[0])
        z_norm = (xyz[:, 2] - self.z_range[0]) / (self.z_range[1] - self.z_range[0])
        
        return torch.stack([x_norm, y_norm, z_norm], dim=-1)


class PointPillarProjection(nn.Module):
    """
    PointPillars-style BEV projection (optimized).

    Creates vertical pillars and encodes all points within each pillar
    using a simplified PointNet. Uses vectorized operations for speed.

    Reference: PointPillars (Lang et al., CVPR 2019)
    """

    def __init__(self,
                 in_channels: int = 4,
                 pillar_channels: int = 64,
                 out_channels: int = 64,
                 height: int = 256,
                 width: int = 256,
                 x_range: Tuple[float, float] = (-50, 50),
                 y_range: Tuple[float, float] = (-50, 50),
                 max_points_per_pillar: int = 32,
                 max_pillars: int = 12000):
        """
        Args:
            in_channels: Input point features (xyz + intensity = 4)
            pillar_channels: Pillar feature dimension
            out_channels: Output BEV channels
            height, width: BEV grid size
            x_range, y_range: Spatial range
            max_points_per_pillar: Maximum points per pillar
            max_pillars: Maximum number of pillars
        """
        super().__init__()

        self.in_channels = in_channels
        self.pillar_channels = pillar_channels
        self.out_channels = out_channels
        self.height = height
        self.width = width
        self.x_range = x_range
        self.y_range = y_range
        self.max_points_per_pillar = max_points_per_pillar
        self.max_pillars = max_pillars

        self.x_res = (x_range[1] - x_range[0]) / width
        self.y_res = (y_range[1] - y_range[0]) / height

        # Pillar feature network (simplified PointNet)
        # Input: x, y, z, intensity, x_c, y_c, z_c, x_p, y_p (9 channels for 4-channel input)
        augmented_channels = in_channels + 5

        self.pfn = nn.Sequential(
            nn.Linear(augmented_channels, pillar_channels),
            nn.ReLU(inplace=True),
            nn.Linear(pillar_channels, pillar_channels),
        )

        # 2D backbone on BEV
        self.backbone = nn.Sequential(
            nn.Conv2d(pillar_channels, out_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        Vectorized PointPillars forward pass.

        Args:
            points: Point cloud (B, N, C)

        Returns:
            bev: BEV feature map (B, out_channels, H, W)
        """
        B, N, C = points.shape
        device = points.device

        bev = torch.zeros(B, self.pillar_channels, self.height, self.width, device=device)

        for b in range(B):
            pts = points[b]  # (N, C)

            # Compute pillar indices
            x, y = pts[:, 0], pts[:, 1]
            col = ((x - self.x_range[0]) / self.x_res).long()
            row = ((y - self.y_range[0]) / self.y_res).long()

            # Filter valid points
            valid = (col >= 0) & (col < self.width) & (row >= 0) & (row < self.height)
            valid_pts = pts[valid]
            valid_col = col[valid]
            valid_row = row[valid]

            if len(valid_pts) == 0:
                continue

            # Create pillar keys and find unique pillars
            pillar_keys = valid_row * self.width + valid_col
            unique_keys, inverse_indices = torch.unique(pillar_keys, return_inverse=True)

            num_pillars = min(len(unique_keys), self.max_pillars)
            if num_pillars < len(unique_keys):
                # Keep only first max_pillars
                keep_mask = inverse_indices < num_pillars
                valid_pts = valid_pts[keep_mask]
                valid_col = valid_col[keep_mask]
                valid_row = valid_row[keep_mask]
                pillar_keys = pillar_keys[keep_mask]
                unique_keys = unique_keys[:num_pillars]
                _, inverse_indices = torch.unique(pillar_keys, return_inverse=True)

            # Compute pillar centers using scatter_reduce
            pillar_x_center = torch.zeros(num_pillars, device=device)
            pillar_y_center = torch.zeros(num_pillars, device=device)
            pillar_z_center = torch.zeros(num_pillars, device=device)
            pillar_counts = torch.zeros(num_pillars, device=device)

            pillar_x_center.scatter_add_(0, inverse_indices, valid_pts[:, 0])
            pillar_y_center.scatter_add_(0, inverse_indices, valid_pts[:, 1])
            pillar_z_center.scatter_add_(0, inverse_indices, valid_pts[:, 2])
            pillar_counts.scatter_add_(0, inverse_indices, torch.ones_like(valid_pts[:, 0]))

            pillar_x_center = pillar_x_center / pillar_counts.clamp(min=1)
            pillar_y_center = pillar_y_center / pillar_counts.clamp(min=1)
            pillar_z_center = pillar_z_center / pillar_counts.clamp(min=1)

            # Compute pillar geometric centers
            pillar_row = unique_keys // self.width
            pillar_col = unique_keys % self.width
            x_p = self.x_range[0] + (pillar_col.float() + 0.5) * self.x_res
            y_p = self.y_range[0] + (pillar_row.float() + 0.5) * self.y_res

            # Create augmented features for each point
            x_c = pillar_x_center[inverse_indices]
            y_c = pillar_y_center[inverse_indices]
            z_c = pillar_z_center[inverse_indices]
            x_p_pts = x_p[inverse_indices]
            y_p_pts = y_p[inverse_indices]

            augmented = torch.cat([
                valid_pts,                                    # Original features
                (valid_pts[:, 0] - x_c).unsqueeze(1),        # x offset from centroid
                (valid_pts[:, 1] - y_c).unsqueeze(1),        # y offset from centroid
                (valid_pts[:, 2] - z_c).unsqueeze(1),        # z offset from centroid
                (valid_pts[:, 0] - x_p_pts).unsqueeze(1),    # x offset from pillar center
                (valid_pts[:, 1] - y_p_pts).unsqueeze(1),    # y offset from pillar center
            ], dim=1)

            # Apply PFN to all points
            point_features = self.pfn(augmented)  # (num_valid_points, pillar_channels)

            # Max pooling per pillar using scatter_reduce
            pillar_features = torch.full(
                (num_pillars, self.pillar_channels), float('-inf'), device=device
            )

            # Scatter max for each channel
            for c in range(self.pillar_channels):
                pillar_features[:, c].scatter_reduce_(
                    0, inverse_indices, point_features[:, c], reduce='amax'
                )

            # Replace -inf with 0 for empty positions
            pillar_features = pillar_features.clamp(min=0)

            # Scatter to BEV
            bev[b, :, pillar_row, pillar_col] = pillar_features.T

        # Apply 2D backbone
        bev = self.backbone(bev)

        return bev


def test_bev_projection():
    """Test BEV projection modules."""
    print("Testing BEV Projection...")
    
    B, N = 2, 16384
    
    # Test BEVProjection
    proj = BEVProjection(
        in_channels=128,
        out_channels=64,
        height=256,
        width=256
    )
    
    points = torch.randn(B, N, 3) * 30
    features = torch.randn(B, N, 128)
    
    bev = proj(points, features)
    print(f"BEVProjection output shape: {bev.shape}")
    assert bev.shape == (B, 64, 256, 256)
    
    # Test PointPillarProjection
    pillar_proj = PointPillarProjection(
        in_channels=4,
        pillar_channels=64,
        out_channels=64,
        height=256,
        width=256
    )

    points = torch.randn(B, N, 4)
    points[:, :, :3] *= 30

    bev = pillar_proj(points)
    print(f"PointPillarProjection output shape: {bev.shape}")
    assert bev.shape == (B, 64, 256, 256)
    
    print("BEV Projection test passed!")


if __name__ == "__main__":
    test_bev_projection()
