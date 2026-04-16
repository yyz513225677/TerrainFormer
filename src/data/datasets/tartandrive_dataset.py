"""
TartanDrive Dataset Implementation

TartanDrive contains diverse off-road driving data with aggressive maneuvers.
"""

import os
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path
import json

from .base_dataset import BaseDataset


class TartanDriveDataset(BaseDataset):
    """
    TartanDrive dataset for off-road driving.
    """
    
    def __init__(self,
                 root_path: str,
                 split: str = 'train',
                 max_points: int = 65536,
                 history_frames: int = 5,
                 future_frames: int = 5,
                 transform=None,
                 terrain_types: Optional[List[str]] = None):
        """
        Args:
            root_path: Path to TartanDrive root directory
            split: Dataset split
            max_points: Maximum points per cloud
            history_frames: Number of history frames
            future_frames: Number of future frames
            transform: Optional transforms
            terrain_types: Filter by terrain type (None for all)
        """
        super().__init__(root_path, split, max_points, history_frames, future_frames, transform)
        
        self.terrain_types = terrain_types
        self._load_samples()
        
    def _load_samples(self):
        """Load sample metadata from dataset."""
        self.samples = []
        root = Path(self.root_path)
        
        # TartanDrive structure varies; handle multiple layouts
        # Common structure: root/terrain_type/sequence/...
        
        # Get all sequences
        sequences = []
        for item in root.iterdir():
            if item.is_dir():
                # Check if it's a terrain type folder or sequence folder
                if (item / 'lidar').exists():
                    sequences.append(item)
                else:
                    # Check subdirectories
                    for subitem in item.iterdir():
                        if subitem.is_dir() and (subitem / 'lidar').exists():
                            sequences.append(subitem)
                            
        # Filter by terrain type if specified
        if self.terrain_types:
            sequences = [s for s in sequences 
                        if any(t in str(s) for t in self.terrain_types)]
            
        # Split sequences
        np.random.seed(42)
        np.random.shuffle(sequences)
        
        n_train = int(0.8 * len(sequences))
        n_val = int(0.1 * len(sequences))
        
        if self.split == 'train':
            sequences = sequences[:n_train]
        elif self.split == 'val':
            sequences = sequences[n_train:n_train + n_val]
        else:
            sequences = sequences[n_train + n_val:]
            
        # Load samples from each sequence
        for seq_path in sequences:
            self._load_sequence(seq_path)
            
        print(f"Loaded {len(self.samples)} samples from TartanDrive ({self.split})")
        
    def _load_sequence(self, seq_path: Path):
        """Load samples from a single sequence."""
        lidar_dir = seq_path / 'lidar'
        if not lidar_dir.exists():
            return
            
        # Get point cloud files
        pcd_files = sorted(lidar_dir.glob('*.bin')) + sorted(lidar_dir.glob('*.npy'))
        
        # Load IMU/state data if available
        imu_file = seq_path / 'imu.txt'
        imu_data = self._load_imu(imu_file) if imu_file.exists() else {}
        
        # Load odometry for action labels
        odom_file = seq_path / 'odom.txt'
        odom_data = self._load_odom(odom_file) if odom_file.exists() else {}
        
        for i, pcd_file in enumerate(pcd_files):
            frame_id = i
            
            sample = {
                'sequence': str(seq_path),
                'frame_id': frame_id,
                'point_cloud_path': str(pcd_file),
                'history_paths': [str(pcd_files[max(0, i-j-1)]) for j in range(self.history_frames)],
                'future_paths': [str(pcd_files[min(len(pcd_files)-1, i+j+1)]) for j in range(self.future_frames)],
            }
            
            # Add state from IMU
            if frame_id in imu_data:
                sample['state'] = imu_data[frame_id]
            else:
                sample['state'] = [0, 0, 0, 0, 0, 0]
                
            # Compute action from odometry
            if frame_id in odom_data and (frame_id + 1) in odom_data:
                sample['action'] = self._compute_action_from_odom(
                    odom_data[frame_id], odom_data[frame_id + 1]
                )
            else:
                sample['action'] = 0
                
            sample['action_history'] = [0] * 10
            sample['goal'] = [1.0, 0.0]
            
            self.samples.append(sample)
            
    def _load_imu(self, path: Path) -> Dict:
        """Load IMU data."""
        imu = {}
        try:
            data = np.loadtxt(path)
            for i, row in enumerate(data):
                if len(row) >= 6:
                    imu[i] = row[:6].tolist()
        except:
            pass
        return imu
    
    def _load_odom(self, path: Path) -> Dict:
        """Load odometry data."""
        odom = {}
        try:
            data = np.loadtxt(path)
            for i, row in enumerate(data):
                odom[i] = row.tolist()
        except:
            pass
        return odom
    
    def _compute_action_from_odom(self, odom1: List[float], odom2: List[float]) -> int:
        """Compute action from odometry change."""
        # Compute heading change
        if len(odom1) >= 6 and len(odom2) >= 6:
            yaw_change = odom2[5] - odom1[5]
        else:
            yaw_change = 0
            
        yaw_deg = np.degrees(yaw_change)
        
        # Also consider velocity for speed actions
        vel1 = np.sqrt(odom1[3]**2 + odom1[4]**2) if len(odom1) >= 5 else 0
        vel2 = np.sqrt(odom2[3]**2 + odom2[4]**2) if len(odom2) >= 5 else 0
        
        # Determine steering action
        if abs(yaw_deg) < 2.5:
            steering = 0
        elif yaw_deg < -35:
            steering = 9
        elif yaw_deg < -25:
            steering = 7
        elif yaw_deg < -15:
            steering = 5
        elif yaw_deg < -7.5:
            steering = 3
        elif yaw_deg < 0:
            steering = 1
        elif yaw_deg < 7.5:
            steering = 2
        elif yaw_deg < 15:
            steering = 4
        elif yaw_deg < 25:
            steering = 6
        elif yaw_deg < 35:
            steering = 8
        else:
            steering = 10
            
        return steering
    
    def _load_point_cloud(self, path: str) -> np.ndarray:
        """Load point cloud from binary or numpy file."""
        try:
            if path.endswith('.npy'):
                points = np.load(path)
            else:
                points = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
            return points
        except Exception as e:
            print(f"Warning: Could not load {path}: {e}")
            return np.zeros((1000, 4), dtype=np.float32)
            
    def _load_labels(self, path: str) -> Dict:
        """Load labels (TartanDrive has limited labels)."""
        return {
            'terrain': np.zeros((256, 256), dtype=np.int64),
            'traversability': np.ones((256, 256), dtype=np.float32) * 0.5,
            'elevation': np.zeros((256, 256), dtype=np.float32),
        }
