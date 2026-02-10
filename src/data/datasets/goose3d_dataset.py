"""
GOOSE-3D Dataset Implementation

Large-scale 3D semantic segmentation dataset for outdoor unstructured environments.
Used for Phase 1 (World Model pretraining) only -- no pose/odometry data available.

23 scenes captured with Velodyne VLS128 (128 channels).
Each scene directory contains frames with timestamps embedded in filenames.
"""

import os
import numpy as np
from typing import Dict, Optional
from pathlib import Path

from .base_dataset import BaseDataset


# GOOSE-3D semantic label mapping (64 classes, from goose_label_mapping.csv)
GOOSE3D_LABELS = {
    0: 'undefined',
    1: 'traffic_cone',
    2: 'snow',
    3: 'cobble',
    4: 'obstacle',
    5: 'leaves',
    6: 'street_light',
    7: 'bikeway',
    8: 'ego_vehicle',
    9: 'pedestrian_crossing',
    10: 'road_block',
    11: 'road_marking',
    12: 'car',
    13: 'bicycle',
    14: 'person',
    15: 'bus',
    16: 'forest',
    17: 'bush',
    18: 'moss',
    19: 'traffic_light',
    20: 'motorcycle',
    21: 'sidewalk',
    22: 'curb',
    23: 'asphalt',
    24: 'gravel',
    25: 'boom_barrier',
    26: 'rail_track',
    27: 'tree_crown',
    28: 'tree_trunk',
    29: 'debris',
    30: 'crops',
    31: 'soil',
    32: 'rider',
    33: 'animal',
    34: 'truck',
    35: 'on_rails',
    36: 'caravan',
    37: 'trailer',
    38: 'building',
    39: 'wall',
    40: 'rock',
    41: 'fence',
    42: 'guard_rail',
    43: 'bridge',
    44: 'tunnel',
    45: 'pole',
    46: 'traffic_sign',
    47: 'misc_sign',
    48: 'barrier_tape',
    49: 'kick_scooter',
    50: 'low_grass',
    51: 'high_grass',
    52: 'scenery_vegetation',
    53: 'sky',
    54: 'water',
    55: 'wire',
    56: 'outlier',
    57: 'heavy_machinery',
    58: 'container',
    59: 'hedge',
    60: 'barrel',
    61: 'pipe',
    62: 'tree_root',
    63: 'military_vehicle',
}

# Traversability scores (0 = not traversable, 1 = traversable)
GOOSE3D_TRAVERSABILITY = {
    # Highly traversable surfaces
    23: 1.0,   # asphalt
    7: 0.9,    # bikeway
    21: 0.9,   # sidewalk
    9: 0.9,    # pedestrian_crossing
    11: 0.9,   # road_marking
    43: 0.8,   # bridge
    24: 0.8,   # gravel
    3: 0.8,    # cobble

    # Moderately traversable terrain
    50: 0.7,   # low_grass
    31: 0.7,   # soil
    5: 0.7,    # leaves
    30: 0.6,   # crops
    22: 0.6,   # curb
    2: 0.6,    # snow
    51: 0.6,   # high_grass
    18: 0.5,   # moss
    62: 0.5,   # tree_root
    29: 0.5,   # debris
    0: 0.5,    # undefined

    # Low traversability
    40: 0.3,   # rock
    17: 0.4,   # bush
    59: 0.4,   # hedge
    52: 0.3,   # scenery_vegetation
    54: 0.3,   # water
    56: 0.3,   # outlier
    4: 0.2,    # obstacle
    26: 0.2,   # rail_track

    # Not traversable - structures
    38: 0.0,   # building
    39: 0.0,   # wall
    44: 0.0,   # tunnel
    41: 0.1,   # fence
    42: 0.1,   # guard_rail
    45: 0.0,   # pole
    6: 0.0,    # street_light
    19: 0.0,   # traffic_light
    46: 0.0,   # traffic_sign
    47: 0.0,   # misc_sign
    10: 0.0,   # road_block
    25: 0.0,   # boom_barrier
    48: 0.0,   # barrier_tape
    55: 0.0,   # wire
    61: 0.0,   # pipe

    # Not traversable - vegetation
    16: 0.1,   # forest
    27: 0.1,   # tree_crown
    28: 0.1,   # tree_trunk

    # Not traversable - dynamic objects
    1: 0.0,    # traffic_cone
    8: 0.0,    # ego_vehicle
    12: 0.0,   # car
    13: 0.0,   # bicycle
    14: 0.0,   # person
    15: 0.0,   # bus
    20: 0.0,   # motorcycle
    32: 0.0,   # rider
    33: 0.0,   # animal
    34: 0.0,   # truck
    35: 0.0,   # on_rails
    36: 0.0,   # caravan
    37: 0.0,   # trailer
    49: 0.0,   # kick_scooter
    57: 0.0,   # heavy_machinery
    58: 0.0,   # container
    60: 0.0,   # barrel
    63: 0.0,   # military_vehicle

    # Sky - N/A
    53: 0.0,   # sky
}

DEFAULT_TRAVERSABILITY = 0.5


class GOOSE3DDataset(BaseDataset):
    """
    GOOSE-3D dataset for outdoor unstructured environments.

    Phase 1 only (World Model pretraining): no pose/odometry data.
    Each scene directory is treated as a distinct sequence (23 total).
    Frames are sorted by timestamp for correct temporal ordering.
    """

    def __init__(self,
                 root_path: str,
                 split: str = 'train',
                 max_points: int = 65536,
                 history_frames: int = 5,
                 future_frames: int = 5,
                 transform=None,
                 train_ratio: float = 0.70,
                 val_ratio: float = 0.15,
                 test_ratio: float = 0.15,
                 max_temporal_gap: float = 15.0,
                 frame_stride: int = 1):
        """
        Args:
            root_path: Path to goose_3d_train/ root (contains lidar/, labels/)
            split: Dataset split ('train', 'val', 'test')
            max_points: Maximum points per cloud
            history_frames: Number of history frames
            future_frames: Number of future frames
            transform: Optional transforms
            train_ratio: Training split ratio
            val_ratio: Validation split ratio
            test_ratio: Test split ratio
            max_temporal_gap: Max seconds between consecutive frames for
                             history/future chains (breaks chain at larger gaps).
                             Default 15.0s since GOOSE-3D frames are sparse.
            frame_stride: Use every Nth frame to reduce sequential redundancy (default: 1)
        """
        super().__init__(root_path, split, max_points,
                         history_frames, future_frames, transform)

        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.max_temporal_gap = max_temporal_gap
        self.frame_stride = frame_stride

        self._load_samples()

    @staticmethod
    def _parse_filename(filepath: str) -> Dict:
        """Parse GOOSE-3D filename into components.

        Filename pattern: {scene_name}__{frame_id}_{timestamp_ns}_{sensor}.ext
        The scene_name may contain single underscores.
        The double underscore (__) separates scene name from frame metadata.

        Example: 2022-10-14_hohenbrunn_feldwege_waldwege__0004_1665754901907541929_vls128.bin
        """
        basename = Path(filepath).stem
        parts = basename.split('__', 1)  # Split on FIRST double-underscore only
        scene_name = parts[0]
        remainder = parts[1]

        rem_parts = remainder.split('_')
        frame_id = int(rem_parts[0])
        timestamp_ns = int(rem_parts[1])
        sensor = rem_parts[2]
        timestamp = timestamp_ns / 1e9  # Convert to seconds

        return {
            'scene_name': scene_name,
            'frame_id': frame_id,
            'timestamp_ns': timestamp_ns,
            'timestamp': timestamp,
            'sensor': sensor,
        }

    def _load_samples(self):
        """Load sample metadata from GOOSE-3D dataset."""
        self.samples = []
        root = Path(self.root_path)
        lidar_dir = root / 'lidar' / 'train'

        if not lidar_dir.exists():
            print(f"Warning: Lidar directory not found: {lidar_dir}")
            return

        for scene_dir in sorted(lidar_dir.iterdir()):
            if not scene_dir.is_dir():
                continue

            scene_name = scene_dir.name

            # Discover all .bin files in this scene
            bin_files = list(scene_dir.glob('*.bin'))
            if not bin_files:
                print(f"Warning: No .bin files in {scene_dir}")
                continue

            # Parse filenames and collect (timestamp, frame_id, path)
            frames = []
            for bf in bin_files:
                meta = self._parse_filename(str(bf))
                frames.append((meta['timestamp'], meta['frame_id'], str(bf)))

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

                # Derive label path: lidar/ -> labels/, _vls128.bin -> _goose.label
                label_path = bin_path.replace('/lidar/', '/labels/').replace(
                    '_vls128.bin', '_goose.label')

                sample = {
                    'sequence': scene_name,
                    'frame_id': fid,
                    'point_cloud_path': bin_path,
                    'history_paths': history_paths,
                    'future_paths': future_paths,
                    'label_path': label_path if os.path.exists(label_path) else '',
                    # Defaults for missing fields (no poses in GOOSE-3D)
                    'state': [0, 0, 0, 0, 0, 0],
                    'action': 0,
                    'action_history': [0] * 10,
                    'goal': [1.0, 0.0],
                }

                self.samples.append(sample)

        print(f"Loaded {len(self.samples)} samples from GOOSE-3D ({self.split} split)")
        print(f"  Scenes: {len(set(s['sequence'] for s in self.samples))}")

    def _load_point_cloud(self, path: str) -> np.ndarray:
        """Load point cloud from binary file.

        GOOSE-3D uses Velodyne VLS128: float32[N, 4] = (x, y, z, intensity).
        No special sensor handling needed.
        """
        try:
            points = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
        except Exception as e:
            print(f"Warning: Could not load point cloud {path}: {e}")
            return np.zeros((1000, 4), dtype=np.float32)

        return points

    def _load_labels(self, path: str) -> Dict:
        """Load semantic labels (uint32) and compute traversability.

        GOOSE-3D labels are uint32: semantic_class = label & 0xFF,
        instance_id = label >> 8.
        """
        labels = {
            'terrain': np.zeros((256, 256), dtype=np.int64),
            'traversability': np.zeros((256, 256), dtype=np.float32),
            'elevation': np.zeros((256, 256), dtype=np.float32),
        }

        if not path or not os.path.exists(path):
            return labels

        try:
            label_data = np.fromfile(path, dtype=np.uint32)
            semantic = label_data & 0xFF  # Lower 8 bits = semantic class (0-63)
            # Point-wise labels available; BEV projection done separately
        except Exception as e:
            print(f"Warning: Could not load labels {path}: {e}")

        return labels

    @staticmethod
    def get_label_name(label_id: int) -> str:
        """Get human-readable label name."""
        return GOOSE3D_LABELS.get(label_id, f'unknown_{label_id}')

    @staticmethod
    def get_traversability(label_id: int) -> float:
        """Get traversability score for label."""
        return GOOSE3D_TRAVERSABILITY.get(label_id, DEFAULT_TRAVERSABILITY)
