"""
Ground Segmentation for LiDAR Point Clouds

Separates ground plane from obstacles.
"""

import numpy as np
from typing import Tuple, Optional


class GroundSegmentation:
    """
    Ground segmentation using RANSAC or other methods.
    """
    
    def __init__(self,
                 method: str = 'ransac',
                 distance_threshold: float = 0.2,
                 max_iterations: int = 1000,
                 seed_height: float = -1.5):
        """
        Args:
            method: Segmentation method ('ransac', 'height', 'progressive')
            distance_threshold: Distance threshold for inliers
            max_iterations: Maximum RANSAC iterations
            seed_height: Initial height estimate for ground
        """
        self.method = method
        self.distance_threshold = distance_threshold
        self.max_iterations = max_iterations
        self.seed_height = seed_height
        
    def segment(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Segment ground from point cloud.
        
        Args:
            points: Point cloud (N, 4)
            
        Returns:
            ground_points: Ground points
            non_ground_points: Non-ground points
        """
        if self.method == 'ransac':
            return self._ransac_segment(points)
        elif self.method == 'height':
            return self._height_segment(points)
        else:
            return self._progressive_segment(points)
            
    def _ransac_segment(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """RANSAC-based ground plane fitting."""
        N = points.shape[0]
        if N < 3:
            return points, np.zeros((0, points.shape[1]))
            
        xyz = points[:, :3]
        best_inliers = np.zeros(N, dtype=bool)
        best_count = 0
        
        for _ in range(self.max_iterations):
            # Sample 3 random points
            idx = np.random.choice(N, 3, replace=False)
            p1, p2, p3 = xyz[idx]
            
            # Compute plane normal
            v1 = p2 - p1
            v2 = p3 - p1
            normal = np.cross(v1, v2)
            norm = np.linalg.norm(normal)
            
            if norm < 1e-6:
                continue
                
            normal /= norm
            d = -np.dot(normal, p1)
            
            # Check if plane is roughly horizontal (z-component dominant)
            if abs(normal[2]) < 0.7:
                continue
                
            # Count inliers
            distances = np.abs(np.dot(xyz, normal) + d)
            inliers = distances < self.distance_threshold
            count = inliers.sum()
            
            if count > best_count:
                best_count = count
                best_inliers = inliers
                
        ground = points[best_inliers]
        non_ground = points[~best_inliers]
        
        return ground, non_ground
    
    def _height_segment(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Simple height-based segmentation."""
        ground_mask = points[:, 2] < self.seed_height
        return points[ground_mask], points[~ground_mask]
    
    def _progressive_segment(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Progressive morphological filter."""
        # Divide into grid cells and find lowest points
        cell_size = 1.0
        xyz = points[:, :3]
        
        # Compute cell indices
        cell_x = (xyz[:, 0] / cell_size).astype(np.int32)
        cell_y = (xyz[:, 1] / cell_size).astype(np.int32)
        
        # Find minimum height in each cell
        cells = {}
        for i, (cx, cy) in enumerate(zip(cell_x, cell_y)):
            key = (cx, cy)
            if key not in cells:
                cells[key] = []
            cells[key].append(i)
            
        # Get ground estimate
        ground_heights = {}
        for key, indices in cells.items():
            heights = xyz[indices, 2]
            ground_heights[key] = np.percentile(heights, 10)
            
        # Classify points
        ground_mask = np.zeros(len(points), dtype=bool)
        for i, (cx, cy) in enumerate(zip(cell_x, cell_y)):
            key = (cx, cy)
            if key in ground_heights:
                if xyz[i, 2] < ground_heights[key] + self.distance_threshold:
                    ground_mask[i] = True
                    
        return points[ground_mask], points[~ground_mask]
    
    def __call__(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make callable."""
        return self.segment(points)
