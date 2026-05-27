"""
LidarDustX Dataset Implementation

Multi-sensor LiDAR dataset for dusty off-road environments.
Used for Phase 1 (World Model pretraining) only -- no pose/action data available.

6 sensors: ls128, ls64, ly150, ly300, m1, ouster
Each sensor folder contains multiple recording sequences identified by sequence_XXXX in filenames.
"""

import os
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path

from .base_dataset import BaseDataset


# All sensor folder names
SENSOR_FOLDERS = ['ls128', 'ls64', 'ly150', 'ly300', 'm1', 'ouster']

# Sensors with known NaN issues in coordinates
SENSORS_WITH_NAN = {'m1'}

# Sensor-specific max intensity for normalization
SENSOR_INTENSITY_MAX = {
    'ls128': 255,
    'ls64': 255,
    'ly150': 255,
    'ly300': 255,
    'm1': 255,
    'ouster': 2701,
}

# LidarDustX semantic label mapping (uint8 per-point)
LIDARDUSTX_LABELS = {
    0: 'background',
    1: 'dust',
    2: 'ground',
    3: 'mound',
    4: 'stone',
    5: 'obstacle',
    6: 'engineering',
    7: 'car',
    8: 'truck',
    9: 'trailer',
    10: 'pedestrian',
}

# Traversability scores (0 = not traversable, 1 = traversable)
LIDARDUSTX_TRAVERSABILITY = {
    0: 0.5,   # background - unknown
    1: 0.3,   # dust - reduced visibility
    2: 0.9,   # ground - traversable
    3: 0.4,   # mound - partially traversable
    4: 0.2,   # stone - mostly obstacle
    5: 0.1,   # obstacle - not traversable
    6: 0.0,   # engineering - obstacle
    7: 0.0,   # car - obstacle
    8: 0.0,   # truck - obstacle
    9: 0.0,   # trailer - obstacle
    10: 0.0,  # pedestrian - obstacle
}

DEFAULT_TRAVERSABILITY = 0.5


class LidarDustXDataset(BaseDataset):
    """
    LidarDustX dataset for dusty off-road environments.

    Phase 1 only (World Model pretraining): no pose/action data.
    Each (sensor, sequence_XXXX) pair is treated as a distinct sequence.
    Frames are sorted by timestamp for correct temporal ordering.
    """

    def __init__(self,
                 root_path: str,
                 split: str = 'train',
                 sensors: Optional[List[str]] = None,
                 max_points: int = 65536,
                 history_frames: int = 5,
                 future_frames: int = 5,
                 transform=None,
                 train_ratio: float = 0.70,
                 val_ratio: float = 0.15,
                 test_ratio: float = 0.15,
                 max_temporal_gap: float = 3.0,
                 frame_stride: int = 1):
        """
        Args:
            root_path: Path to LidarDustX root (contains ls128/, ls64/, etc.)
            split: Dataset split ('train', 'val', 'test')
            sensors: List of sensor folders to use (None for all 6)
            max_points: Maximum points per cloud
            history_frames: Number of history frames
            future_frames: Number of future frames
            transform: Optional transforms
            train_ratio: Training split ratio
            val_ratio: Validation split ratio
            test_ratio: Test split ratio
            max_temporal_gap: Max seconds between consecutive frames for
                             history/future chains (breaks chain at larger gaps)
            frame_stride: Use every Nth frame to reduce sequential redundancy (default: 1)
        """
        super().__init__(root_path, split, max_points,
                         history_frames, future_frames, transform)

        self.sensors = sensors if sensors is not None else SENSOR_FOLDERS
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.max_temporal_gap = max_temporal_gap
        self.frame_stride = frame_stride

        self._load_samples()

    @staticmethod
    def _parse_filename(filepath: str) -> Dict:
        """Parse LidarDustX filename into components.

        Filename pattern: sequence_XXXX_SENSOR_YYYY_TIMESTAMP_MICROSECONDS.ext
        Example: sequence_0000_ls128_0001_1718681522_2011757.bin
        """
        basename = Path(filepath).stem
        parts = basename.split('_')
        # parts: ['sequence', 'XXXX', 'SENSOR', 'YYYY', 'TIMESTAMP', 'MICROSECONDS']
        seq_id = parts[1]
        sensor = parts[2]
        frame_id = int(parts[3])
        timestamp = float(parts[4]) + float(parts[5]) / 10000000.0

        return {
            'seq_id': seq_id,
            'sensor': sensor,
            'frame_id': frame_id,
            'timestamp': timestamp,
        }

    def _load_samples(self):
        """Load sample metadata from LidarDustX dataset."""
        self.samples = []
        root = Path(self.root_path)

        for sensor in self.sensors:
            sensor_dir = root / sensor
            if not sensor_dir.exists():
                print(f"Warning: Sensor directory not found: {sensor_dir}")
                continue

            # Discover all .bin files
            bin_files = list(sensor_dir.glob('*.bin'))
            if not bin_files:
                print(f"Warning: No .bin files in {sensor_dir}")
                continue

            # Parse filenames and group by sequence ID
            seq_frames = {}  # {seq_id: [(timestamp, frame_id, bin_path), ...]}
            for bf in bin_files:
                meta = self._parse_filename(str(bf))
                seq_key = meta['seq_id']
                seq_frames.setdefault(seq_key, []).append(
                    (meta['timestamp'], meta['frame_id'], str(bf))
                )

            # Process each sequence
            for seq_id in sorted(seq_frames.keys()):
                frames = seq_frames[seq_id]

                # Sort by timestamp for correct temporal ordering
                frames.sort(key=lambda x: x[0])

                if len(frames) < 2:
                    continue

                # Apply train/val/test split
                total = len(frames)
                train_end = int(total * self.train_ratio)
                val_end = train_end + int(total * self.val_ratio)

                if self.split == 'train':
                    selected_indices = list(range(0, train_end))
                elif self.split == 'val':
                    selected_indices = list(range(train_end, val_end))
                elif self.split == 'test':
                    selected_indices = list(range(val_end, total))
                else:
                    raise ValueError(f"Unknown split: {self.split}")

                # Apply frame stride to reduce sequential redundancy
                if self.frame_stride > 1:
                    selected_indices = selected_indices[::self.frame_stride]

                # Build sample dicts for selected frames
                for idx in selected_indices:
                    ts, fid, bin_path = frames[idx]

                    # Build history paths (respecting temporal gap)
                    history_paths = []
                    for j in range(1, self.history_frames + 1):
                        hist_idx = idx - j
                        if hist_idx < 0:
                            break
                        # Check gap between consecutive frames in the chain
                        gap_ok = True
                        for k in range(hist_idx, idx):
                            if frames[k + 1][0] - frames[k][0] > self.max_temporal_gap:
                                gap_ok = False
                                break
                        if gap_ok:
                            history_paths.append(frames[hist_idx][2])
                        else:
                            break
                    history_paths.reverse()  # Chronological order

                    # Build future paths (respecting temporal gap)
                    future_paths = []
                    for j in range(1, self.future_frames + 1):
                        fut_idx = idx + j
                        if fut_idx >= len(frames):
                            break
                        gap_ok = True
                        for k in range(idx, fut_idx):
                            if frames[k + 1][0] - frames[k][0] > self.max_temporal_gap:
                                gap_ok = False
                                break
                        if gap_ok:
                            future_paths.append(frames[fut_idx][2])
                        else:
                            break

                    # Derive label path from bin path
                    label_path = bin_path.replace('.bin', '.label')

                    sample = {
                        'sequence': f"{sensor}_{seq_id}",
                        'frame_id': fid,
                        'point_cloud_path': bin_path,
                        'history_paths': history_paths,
                        'future_paths': future_paths,
                        'label_path': label_path if os.path.exists(label_path) else '',
                        'sensor': sensor,
                        # Defaults for missing fields (no poses in LidarDustX)
                        'state': [0, 0, 0, 0, 0, 0],
                        'action': 0,
                        'action_history': [0] * 10,
                        'goal': [1.0, 0.0],
                    }

                    self.samples.append(sample)

        print(f"Loaded {len(self.samples)} samples from LidarDustX ({self.split} split)")
        print(f"  Sensors: {self.sensors}")

    def _load_point_cloud(self, path: str) -> np.ndarray:
        """Load point cloud from binary file.

        Handles NaN replacement (M1 sensor) and intensity normalization (OUSTER).
        """
        try:
            points = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
        except Exception as e:
            print(f"Warning: Could not load point cloud {path}: {e}")
            return np.zeros((1000, 4), dtype=np.float32)

        # Detect sensor from filename
        meta = self._parse_filename(path)
        sensor = meta['sensor']

        # Handle NaN values (prevalent in M1 sensor)
        if sensor in SENSORS_WITH_NAN:
            nan_mask = np.isnan(points)
            if nan_mask.any():
                points[nan_mask] = 0.0

        # Normalize OUSTER intensity from [0, 2701] to [0, 255]
        if sensor == 'ouster':
            max_intensity = SENSOR_INTENSITY_MAX['ouster']
            points[:, 3] = np.clip(points[:, 3] / max_intensity * 255.0, 0, 255)

        return points

    def _load_labels(self, path: str) -> Dict:
        """Load semantic labels (uint8) and compute traversability."""
        labels = {
            'terrain': np.zeros((256, 256), dtype=np.int64),
            'traversability': np.zeros((256, 256), dtype=np.float32),
            'elevation': np.zeros((256, 256), dtype=np.float32),
        }

        if not path or not os.path.exists(path):
            return labels

        try:
            # LidarDustX labels are uint8 (NOT uint32 like RELLIS)
            label_data = np.fromfile(path, dtype=np.uint8)
            # Point-wise labels available; BEV projection done separately
        except Exception as e:
            print(f"Warning: Could not load labels {path}: {e}")

        return labels

    @staticmethod
    def get_label_name(label_id: int) -> str:
        """Get human-readable label name."""
        return LIDARDUSTX_LABELS.get(label_id, f'unknown_{label_id}')

    @staticmethod
    def get_traversability(label_id: int) -> float:
        """Get traversability score for label."""
        return LIDARDUSTX_TRAVERSABILITY.get(label_id, DEFAULT_TRAVERSABILITY)
