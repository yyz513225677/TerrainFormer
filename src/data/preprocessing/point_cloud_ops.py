"""
Point Cloud Operations

Common operations for LiDAR point cloud preprocessing.
"""

import torch
import numpy as np
from typing import Tuple, Optional


class PointCloudOps:
    """Collection of point cloud operations."""
    
    @staticmethod
    def range_filter(points: np.ndarray,
                     min_range: float = 0.5,
                     max_range: float = 70.0) -> np.ndarray:
        """Filter points by range from origin."""
        distances = np.linalg.norm(points[:, :3], axis=1)
        mask = (distances >= min_range) & (distances <= max_range)
        return points[mask]
    
    @staticmethod
    def height_filter(points: np.ndarray,
                      min_height: float = -3.0,
                      max_height: float = 5.0) -> np.ndarray:
        """Filter points by height (z-axis)."""
        mask = (points[:, 2] >= min_height) & (points[:, 2] <= max_height)
        return points[mask]
    
    @staticmethod
    def voxel_downsample(points: np.ndarray,
                         voxel_size: float = 0.1) -> np.ndarray:
        """Voxel downsampling of point cloud."""
        # Compute voxel indices
        voxel_idx = (points[:, :3] / voxel_size).astype(np.int32)
        
        # Get unique voxels
        _, unique_indices = np.unique(
            voxel_idx[:, 0] * 1000000 + voxel_idx[:, 1] * 1000 + voxel_idx[:, 2],
            return_index=True
        )
        
        return points[unique_indices]
    
    @staticmethod
    def normalize_intensity(points: np.ndarray,
                           intensity_range: Tuple[float, float] = (0, 255)) -> np.ndarray:
        """Normalize intensity to [0, 1]."""
        points = points.copy()
        min_i, max_i = intensity_range
        points[:, 3] = (points[:, 3] - min_i) / (max_i - min_i + 1e-6)
        points[:, 3] = np.clip(points[:, 3], 0, 1)
        return points
    
    @staticmethod
    def center_point_cloud(points: np.ndarray,
                           center: Optional[np.ndarray] = None) -> np.ndarray:
        """Center point cloud at origin or specified center."""
        points = points.copy()
        if center is None:
            center = points[:, :3].mean(axis=0)
        points[:, :3] -= center
        return points
    
    @staticmethod
    def random_sample(points: np.ndarray, n_points: int) -> np.ndarray:
        """Randomly sample n_points from point cloud."""
        N = points.shape[0]
        if N >= n_points:
            indices = np.random.choice(N, n_points, replace=False)
        else:
            indices = np.random.choice(N, n_points, replace=True)
        return points[indices]
    
    @staticmethod
    def farthest_point_sample(points: np.ndarray, n_points: int) -> np.ndarray:
        """Farthest point sampling."""
        N = points.shape[0]
        if N <= n_points:
            return points
            
        xyz = points[:, :3]
        selected = np.zeros(n_points, dtype=np.int64)
        distances = np.full(N, np.inf)
        
        # Start from random point
        selected[0] = np.random.randint(N)
        
        for i in range(1, n_points):
            dist = np.linalg.norm(xyz - xyz[selected[i-1]], axis=1)
            distances = np.minimum(distances, dist)
            selected[i] = np.argmax(distances)
            
        return points[selected]


class PointCloudTransform:
    """Callable transform for point cloud preprocessing pipeline."""
    
    def __init__(self,
                 min_range: float = 0.5,
                 max_range: float = 70.0,
                 min_height: float = -3.0,
                 max_height: float = 5.0,
                 voxel_size: Optional[float] = None,
                 normalize_intensity: bool = True,
                 center: bool = True,
                 n_points: Optional[int] = None):
        self.min_range = min_range
        self.max_range = max_range
        self.min_height = min_height
        self.max_height = max_height
        self.voxel_size = voxel_size
        self.normalize_intensity = normalize_intensity
        self.center = center
        self.n_points = n_points
        
    def __call__(self, points: np.ndarray) -> np.ndarray:
        # Range filter
        points = PointCloudOps.range_filter(points, self.min_range, self.max_range)
        
        # Height filter
        points = PointCloudOps.height_filter(points, self.min_height, self.max_height)
        
        # Voxel downsample
        if self.voxel_size is not None:
            points = PointCloudOps.voxel_downsample(points, self.voxel_size)
            
        # Normalize intensity
        if self.normalize_intensity and points.shape[1] > 3:
            points = PointCloudOps.normalize_intensity(points)
            
        # Center
        if self.center:
            points = PointCloudOps.center_point_cloud(points)
            
        # Sample
        if self.n_points is not None:
            points = PointCloudOps.random_sample(points, self.n_points)
            
        return points
