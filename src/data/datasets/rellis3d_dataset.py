"""
RELLIS-3D Dataset Implementation

RELLIS-3D is an off-road terrain dataset with semantic annotations.
"""

import os
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path
import json

from .base_dataset import BaseDataset


# RELLIS-3D label mapping
RELLIS_LABELS = {
    0: 'void',
    1: 'grass',
    3: 'tree',
    4: 'pole',
    5: 'water',
    6: 'sky',
    7: 'vehicle',
    8: 'object',
    9: 'asphalt',
    10: 'building',
    12: 'log',
    15: 'person',
    17: 'fence',
    18: 'bush',
    19: 'concrete',
    23: 'barrier',
    27: 'puddle',
    31: 'mud',
    33: 'rubble',
    34: 'gravel',
}

# Traversability scores (0 = not traversable, 1 = traversable)
RELLIS_TRAVERSABILITY = {
    0: 0.5,   # void - unknown
    1: 0.9,   # grass - traversable
    3: 0.1,   # tree - obstacle
    4: 0.0,   # pole - obstacle
    5: 0.3,   # water - caution
    6: 0.0,   # sky - N/A
    7: 0.0,   # vehicle - obstacle
    8: 0.2,   # object - mostly obstacle
    9: 1.0,   # asphalt - highly traversable
    10: 0.0,  # building - obstacle
    12: 0.2,  # log - obstacle
    15: 0.0,  # person - obstacle
    17: 0.1,  # fence - obstacle
    18: 0.4,  # bush - partially traversable
    19: 1.0,  # concrete - highly traversable
    23: 0.0,  # barrier - obstacle
    27: 0.5,  # puddle - caution
    31: 0.6,  # mud - traversable with caution
    33: 0.7,  # rubble - traversable with caution
    34: 0.8,  # gravel - traversable
}


class RELLIS3DDataset(BaseDataset):
    """
    RELLIS-3D dataset for off-road semantic understanding.
    """

    def __init__(self,
                 root_path: str,
                 split: str = 'train',
                 sequences: Optional[List[int]] = None,
                 max_points: int = 65536,
                 history_frames: int = 5,
                 future_frames: int = 5,
                 transform=None,
                 train_ratio: float = 0.70,
                 val_ratio: float = 0.15,
                 test_ratio: float = 0.15,
                 frame_stride: int = 1,
                 num_actions: int = 12,
                 chunk_size: int = 5):
        """
        Args:
            root_path: Path to RELLIS-3D bin directory (e.g., ../RELLIS/bin/)
            split: Dataset split ('train', 'val', 'test')
            sequences: Specific sequence IDs to use (None for all: [0,1,2,3,4])
            max_points: Maximum points per cloud
            history_frames: Number of history frames
            future_frames: Number of future frames
            transform: Optional transforms
            train_ratio: Ratio of data for training (default: 0.70)
            val_ratio: Ratio of data for validation (default: 0.15)
            test_ratio: Ratio of data for testing (default: 0.15)
            frame_stride: Use every Nth frame to reduce sequential redundancy (default: 1)
            num_actions: Number of action classes (default: 12 for forward-only)
            chunk_size: Number of future actions to label per sample (action chunking)
        """
        super().__init__(root_path, split, max_points, history_frames, future_frames, transform)

        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.frame_stride = frame_stride
        self.num_actions = num_actions
        self.chunk_size = chunk_size

        # Use all sequences if not specified
        self.sequences = sequences if sequences is not None else [0, 1, 2, 3, 4]
        self._load_samples()
        
    def _load_samples(self):
        """Load sample metadata from dataset with 70/15/15 split."""
        self.samples = []

        for seq_id in self.sequences:
            # RELLIS-3D format: root_path/seq_id/velodyne/*.bin
            seq_path = Path(self.root_path) / f'{seq_id:05d}'

            if not seq_path.exists():
                print(f"Warning: Sequence directory not found: {seq_path}")
                continue

            # Bin files are in velodyne/ subfolder (RELLIS-3D / SemanticKITTI format)
            velodyne_path = seq_path / 'velodyne'
            bin_dir = velodyne_path if velodyne_path.exists() else seq_path

            # Get all point cloud files (sorted by filename)
            pcd_files = sorted(bin_dir.glob('*.bin'))

            if len(pcd_files) == 0:
                print(f"Warning: No .bin files found in {seq_path}")
                continue

            # Calculate split indices for this sequence
            total_frames = len(pcd_files)
            train_end = int(total_frames * self.train_ratio)
            val_end = train_end + int(total_frames * self.val_ratio)

            # Select frames based on split
            if self.split == 'train':
                selected_files = pcd_files[:train_end]
            elif self.split == 'val':
                selected_files = pcd_files[train_end:val_end]
            elif self.split == 'test':
                selected_files = pcd_files[val_end:]
            else:
                raise ValueError(f"Unknown split: {self.split}. Use 'train', 'val', or 'test'.")

            # Load action labels from generated actions.npy
            actions_dir = Path(self.root_path).parent / 'actions' / f'{seq_id:05d}'
            actions_file = actions_dir / 'actions.npy'
            actions = self._load_actions(actions_file) if actions_file.exists() else None

            if actions is None:
                print(f"Warning: No action labels found at {actions_file}. Using default action 0.")
                actions = np.zeros(len(pcd_files), dtype=np.int64)

            # Load goal directions from generated goals.npy (computed from poses)
            goals_file = actions_dir / 'goals.npy'
            goals = self._load_goals(goals_file) if goals_file.exists() else None

            if goals is None:
                # Default to forward direction
                goals = np.tile([1.0, 0.0], (len(pcd_files), 1))

            # Apply frame stride to reduce sequential redundancy
            if self.frame_stride > 1:
                selected_files = selected_files[::self.frame_stride]

            # Create samples from selected files
            for i, pcd_file in enumerate(selected_files):
                frame_id = int(pcd_file.stem)

                # Get actual index in full sequence for history/future
                full_seq_idx = pcd_files.index(pcd_file)

                # Build sample info
                sample = {
                    'sequence': seq_id,
                    'frame_id': frame_id,
                    'point_cloud_path': str(pcd_file),
                    'history_paths': [str(pcd_files[max(0, full_seq_idx-j-1)]) for j in range(self.history_frames)],
                    'future_paths': [str(pcd_files[min(len(pcd_files)-1, full_seq_idx+j+1)]) for j in range(self.future_frames)],
                }

                # Add label path if exists (RELLIS-3D: labels/ inside sequence dir)
                label_dir = seq_path / 'labels'
                label_file = label_dir / f'{frame_id:06d}.label'
                if label_file.exists():
                    sample['label_path'] = str(label_file)

                # Vehicle state (dummy for now)
                sample['state'] = [0, 0, 0, 0, 0, 0]

                # Load real action label (clamped to valid range)
                max_action = self.num_actions - 1
                if full_seq_idx < len(actions):
                    sample['action'] = int(min(actions[full_seq_idx], max_action))
                else:
                    sample['action'] = 0

                # Action chunk: current + next (chunk_size-1) actions for action chunking
                chunk = []
                for k in range(self.chunk_size):
                    idx = full_seq_idx + k
                    if idx < len(actions):
                        chunk.append(int(min(actions[idx], max_action)))
                    else:
                        chunk.append(chunk[-1] if chunk else 0)
                sample['action_chunk'] = chunk

                # Action history (last 10 actions, clamped to valid range)
                action_hist = []
                for j in range(10):
                    hist_idx = full_seq_idx - j - 1
                    if hist_idx >= 0 and hist_idx < len(actions):
                        action_hist.append(int(min(actions[hist_idx], max_action)))
                    else:
                        action_hist.append(0)
                sample['action_history'] = action_hist[::-1]  # Reverse to chronological order

                # Goal direction from actual movement (computed from poses)
                if full_seq_idx < len(goals):
                    sample['goal'] = goals[full_seq_idx].tolist()
                else:
                    sample['goal'] = [1.0, 0.0]

                self.samples.append(sample)

        # Print statistics
        total = len(self.samples)
        print(f"Loaded {total} samples from RELLIS-3D ({self.split} split)")
        print(f"  Split ratios: train={self.train_ratio:.0%}, val={self.val_ratio:.0%}, test={self.test_ratio:.0%}")
        
    def _load_actions(self, actions_file: Path) -> Optional[np.ndarray]:
        """Load action labels from numpy file with clamping to valid range."""
        try:
            actions = np.load(actions_file)
            # Clamp actions to valid range [0, num_actions-1]
            # This handles old labels generated with different action spaces
            max_action = self.num_actions - 1
            if np.any(actions > max_action):
                num_invalid = np.sum(actions > max_action)
                print(f"Warning: {num_invalid} actions > {max_action} found, clamping to valid range")
                actions = np.clip(actions, 0, max_action)
            return actions
        except Exception as e:
            print(f"Warning: Could not load actions from {actions_file}: {e}")
            return None

    def _load_goals(self, goals_file: Path) -> Optional[np.ndarray]:
        """Load goal directions from numpy file."""
        try:
            goals = np.load(goals_file)
            return goals
        except Exception as e:
            # Goals file is optional, don't warn
            return None
            
    def _load_point_cloud(self, path: str) -> np.ndarray:
        """Load KITTI-format binary point cloud."""
        try:
            points = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
            return points
        except Exception as e:
            print(f"Warning: Could not load point cloud {path}: {e}")
            return np.zeros((1000, 4), dtype=np.float32)
            
    def _load_labels(self, path: str) -> Dict:
        """Load semantic labels and compute traversability."""
        labels = {
            'terrain': np.zeros((256, 256), dtype=np.int64),
            'traversability': np.zeros((256, 256), dtype=np.float32),
            'elevation': np.zeros((256, 256), dtype=np.float32),
        }
        
        if not path or not os.path.exists(path):
            return labels
            
        try:
            # Load semantic labels
            label_data = np.fromfile(path, dtype=np.uint32).reshape(-1)
            semantic = label_data & 0xFFFF
            
            # This is point-wise labels; for BEV, we'd need to project
            # For simplicity, return placeholder
            # In full implementation, project to BEV grid
            
        except Exception as e:
            print(f"Warning: Could not load labels {path}: {e}")
            
        return labels
    
    @staticmethod
    def get_label_name(label_id: int) -> str:
        """Get human-readable label name."""
        return RELLIS_LABELS.get(label_id, 'unknown')
    
    @staticmethod
    def get_traversability(label_id: int) -> float:
        """Get traversability score for label."""
        return RELLIS_TRAVERSABILITY.get(label_id, 0.5)
