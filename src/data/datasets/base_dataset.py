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
        
        # Build output dictionary
        output = {
            'point_cloud': torch.from_numpy(point_cloud).float(),
            'point_cloud_history': torch.from_numpy(history).float(),
            'future_point_clouds': torch.from_numpy(future).float(),
            'terrain_labels': torch.from_numpy(labels.get('terrain', np.zeros((256, 256)))).long(),
            'traversability_map': torch.from_numpy(labels.get('traversability', np.zeros((256, 256)))).float(),
            'elevation_map': torch.from_numpy(labels.get('elevation', np.zeros((256, 256)))).float(),
            'expert_action': torch.tensor(sample_info.get('action', 0)).long(),
            'action_sequence': torch.tensor(sample_info.get('action_history', [0]*10)).long(),
            'vehicle_state': torch.tensor(sample_info.get('state', [0]*6)).float(),
            'goal_direction': torch.tensor(sample_info.get('goal', [1, 0])).float(),
        }
        
        if self.transform:
            output = self.transform(output)
            
        return output
    
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
