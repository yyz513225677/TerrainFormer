"""
Base Dataset for TerrainFormer

Abstract base class defining the interface for all datasets.
"""

import torch
from torch.utils.data import Dataset
from typing import Dict, Optional, List, Tuple
from abc import ABC, abstractmethod
import numpy as np


class BaseDataset(Dataset, ABC):
    """
    Abstract base dataset for off-road navigation data.
    
    All datasets should inherit from this class and implement
    the required methods.
    """
    
    def __init__(self,
                 root_path: str,
                 split: str = 'train',
                 max_points: int = 65536,
                 history_frames: int = 5,
                 future_frames: int = 5,
                 transform=None):
        """
        Args:
            root_path: Path to dataset root
            split: Dataset split ('train', 'val', 'test')
            max_points: Maximum points per cloud
            history_frames: Number of history frames
            future_frames: Number of future frames
            transform: Optional data transforms
        """
        super().__init__()
        
        self.root_path = root_path
        self.split = split
        self.max_points = max_points
        self.history_frames = history_frames
        self.future_frames = future_frames
        self.transform = transform
        
        # To be populated by subclasses
        self.samples: List[Dict] = []
        
    @abstractmethod
    def _load_samples(self):
        """Load sample metadata. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def _load_point_cloud(self, path: str) -> np.ndarray:
        """Load point cloud from file."""
        pass
    
    @abstractmethod
    def _load_labels(self, path: str) -> Dict:
        """Load labels/annotations from file."""
        pass
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.
        
        Returns dictionary with:
            - point_cloud: (N, 4) - x, y, z, intensity
            - point_cloud_history: (T, N, 4)
            - future_point_clouds: (K, N, 4) 
            - terrain_labels: (H, W)
            - traversability_map: (H, W)
            - elevation_map: (H, W)
            - expert_action: int
            - action_sequence: (S,)
            - vehicle_state: (6,)
            - goal_direction: (2,)
        """
        sample_info = self.samples[idx]
        
        # Load current point cloud
        point_cloud = self._load_point_cloud(sample_info['point_cloud_path'])
        
        # Subsample/pad to max_points
        point_cloud = self._normalize_point_count(point_cloud)
        
        # Load history frames
        history = []
        for path in sample_info.get('history_paths', []):
            pc = self._load_point_cloud(path)
            pc = self._normalize_point_count(pc)
            history.append(pc)
        
        # Pad history if needed
        while len(history) < self.history_frames:
            history.insert(0, np.zeros_like(point_cloud))
        history = np.stack(history[-self.history_frames:])
        
        # Load future frames
        future = []
        for path in sample_info.get('future_paths', []):
            pc = self._load_point_cloud(path)
            pc = self._normalize_point_count(pc)
            future.append(pc)
            
        while len(future) < self.future_frames:
            future.append(np.zeros_like(point_cloud))
        future = np.stack(future[:self.future_frames])
        
        # Load labels
        labels = self._load_labels(sample_info.get('label_path', ''))

        # Compute traversability from point cloud geometry when labels lack real values.
        # Height variance per BEV cell: flat cells = traversable (1.0),
        # rough/obstacle cells = not traversable (0.0), empty = uncertain (0.5).
        trav_label = labels.get('traversability', None)
        if trav_label is None or (np.abs(trav_label).max() < 1e-6):
            trav_label = self._compute_traversability_from_points(point_cloud)

        # Build output dictionary
        output = {
            'point_cloud': torch.from_numpy(point_cloud).float(),
            'point_cloud_history': torch.from_numpy(history).float(),
            'future_point_clouds': torch.from_numpy(future).float(),
            'terrain_labels': torch.from_numpy(labels.get('terrain', np.zeros((256, 256)))).long(),
            'traversability_map': torch.from_numpy(trav_label).float(),
            'elevation_map': torch.from_numpy(labels.get('elevation', np.zeros((256, 256)))).float(),
            'expert_action': torch.tensor(sample_info.get('action', 0)).long(),
            'action_sequence': torch.tensor(sample_info.get('action_history', [0]*10)).long(),
            'action_chunk': torch.tensor(
                sample_info.get('action_chunk', [sample_info.get('action', 0)])
            ).long(),
            'vehicle_state': torch.tensor(sample_info.get('state', [0]*6)).float(),
            'goal_direction': torch.tensor(sample_info.get('goal', [1, 0])).float(),
        }
        
        if self.transform:
            output = self.transform(output)
            
        return output
    
    @staticmethod
    def _compute_traversability_from_points(points: np.ndarray,
                                             bev_size: int = 256,
                                             x_range: tuple = (-50, 50),
                                             y_range: tuple = (-50, 50)) -> np.ndarray:
        """
        Compute BEV traversability map from point cloud geometry.

        Flat, dense areas (low height variance) → 1.0 (traversable).
        Rough or obstacle areas (high height variance) → 0.0.
        Cells with no points → 0.5 (uncertain).

        Args:
            points: (N, 4+) array [x, y, z, ...]

        Returns:
            (bev_size, bev_size) float32 array in [0, 1]
        """
        x_res = (x_range[1] - x_range[0]) / bev_size
        y_res = (y_range[1] - y_range[0]) / bev_size

        xs = ((points[:, 0] - x_range[0]) / x_res).astype(np.int32)
        ys = ((points[:, 1] - y_range[0]) / y_res).astype(np.int32)
        zs = points[:, 2].astype(np.float32)

        valid = (xs >= 0) & (xs < bev_size) & (ys >= 0) & (ys < bev_size)
        xs, ys, zs = xs[valid], ys[valid], zs[valid]

        count = np.zeros((bev_size, bev_size), dtype=np.float32)
        z_sum = np.zeros((bev_size, bev_size), dtype=np.float32)
        z_sq_sum = np.zeros((bev_size, bev_size), dtype=np.float32)

        np.add.at(count, (ys, xs), 1)
        np.add.at(z_sum, (ys, xs), zs)

        has_pts = count > 0
        z_mean = np.where(has_pts, z_sum / np.where(has_pts, count, 1), 0.0)

        z_dev = zs - z_mean[ys, xs]
        np.add.at(z_sq_sum, (ys, xs), z_dev ** 2)
        z_var = np.where(has_pts, z_sq_sum / np.where(has_pts, count, 1), 0.0)

        flatness = np.exp(-z_var * 10.0)
        trav = np.where(has_pts, flatness, 0.5)
        return trav.astype(np.float32)

    def _normalize_point_count(self, points: np.ndarray) -> np.ndarray:
        """Subsample or pad point cloud to max_points."""
        N = points.shape[0]
        
        if N > self.max_points:
            # Random subsample
            indices = np.random.choice(N, self.max_points, replace=False)
            points = points[indices]
        elif N < self.max_points:
            # Pad with zeros
            padding = np.zeros((self.max_points - N, points.shape[1]))
            points = np.vstack([points, padding])
            
        return points
    
    def get_action_statistics(self) -> Dict:
        """Get statistics about action distribution."""
        actions = [s.get('action', 0) for s in self.samples]
        unique, counts = np.unique(actions, return_counts=True)
        return dict(zip(unique.tolist(), counts.tolist()))
