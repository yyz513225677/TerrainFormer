#!/usr/bin/env python3
"""
TerrainFormer Real-Time Inference with Velodyne LiDAR

Connects to Velodyne LiDAR sensor and runs real-time decision making.
Supports:
1. ROS (sensor_msgs/PointCloud2)
2. Direct UDP connection (Velodyne native protocol)
3. Folder mode (load from .bin/.npy files with navigation)

Usage:
    # ROS mode (requires ROS environment)
    python realtime_inference.py --mode ros --topic /velodyne_points

    # Direct UDP mode
    python realtime_inference.py --mode udp --port 2368

    # Simulation mode (for testing without hardware)
    python realtime_inference.py --mode sim

    # Folder mode (load from RELLIS-3D sequence directory)
    python realtime_inference.py --mode folder --data-dir /home/rickslab3/Documents/Datasets/RELLIS/dataset/sequences/00000
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import torch
import numpy as np
import time
import threading
import queue
import socket
import json
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List
import argparse
import glob

# Visualization
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Button
import tkinter as tk
from tkinter import filedialog

# TerrainFormer
from models.unified.terrainformer import TerrainFormer
from models.unified.temporal_terrainformer import TemporalTerrainFormer, TemporalFrameBuffer


@dataclass
class VehicleState:
    """Vehicle state for decision making."""
    vx: float = 0.0      # Forward velocity (m/s)
    vy: float = 0.0      # Lateral velocity (m/s)
    vz: float = 0.0      # Vertical velocity (m/s)
    roll: float = 0.0    # Roll angle (rad)
    pitch: float = 0.0   # Pitch angle (rad)
    yaw: float = 0.0     # Yaw angle (rad)

    def to_tensor(self, device='cuda') -> torch.Tensor:
        return torch.tensor(
            [[self.vx, self.vy, self.vz, self.roll, self.pitch, self.yaw]],
            dtype=torch.float32, device=device
        )


@dataclass
class NavigationGoal:
    """Navigation goal direction."""
    dx: float = 1.0  # Forward direction
    dy: float = 0.0  # Lateral direction

    def to_tensor(self, device='cuda') -> torch.Tensor:
        # Normalize
        norm = np.sqrt(self.dx**2 + self.dy**2)
        if norm > 0:
            return torch.tensor([[self.dx/norm, self.dy/norm]],
                              dtype=torch.float32, device=device)
        return torch.tensor([[1.0, 0.0]], dtype=torch.float32, device=device)

    @staticmethod
    def from_action(action: int) -> 'NavigationGoal':
        """
        Compute goal direction from action label.
        Maps action to approximate goal direction.
        """
        # Action to goal direction mapping
        action_to_goal = {
            0: (0.0, 0.0),      # STOP
            1: (1.0, 0.0),      # FORWARD_SLOW
            2: (1.0, 0.0),      # FORWARD_MEDIUM
            3: (1.0, 0.0),      # FORWARD_FAST
            4: (-1.0, 0.0),     # BACKWARD_SLOW
            5: (-1.0, 0.0),     # BACKWARD_MEDIUM
            6: (-1.0, 0.0),     # BACKWARD_FAST
            7: (0.0, 1.0),      # TURN_LEFT_SHARP
            8: (0.5, 0.87),     # TURN_LEFT_MEDIUM
            9: (0.87, 0.5),     # TURN_LEFT_SLIGHT
            10: (1.0, 0.0),     # STRAIGHT
            11: (0.87, -0.5),   # TURN_RIGHT_SLIGHT
            12: (0.5, -0.87),   # TURN_RIGHT_MEDIUM
            13: (0.0, -1.0),    # TURN_RIGHT_SHARP
            14: (0.87, 0.5),    # FORWARD_LEFT
            15: (0.87, -0.5),   # FORWARD_RIGHT
            16: (-0.87, 0.5),   # BACKWARD_LEFT
            17: (-0.87, -0.5),  # BACKWARD_RIGHT
        }
        dx, dy = action_to_goal.get(action, (1.0, 0.0))
        return NavigationGoal(dx=dx, dy=dy)


class GoalProvider:
    """
    Provides goal directions for inference.

    Modes:
    - 'manual': Use fixed goal direction
    - 'auto': Use ground truth goals from dataset
    - 'velocity': Infer goal from vehicle velocity
    """

    def __init__(self, mode: str = 'manual', default_goal: NavigationGoal = None):
        self.mode = mode
        self.default_goal = default_goal or NavigationGoal(1.0, 0.0)
        self.goals: Optional[np.ndarray] = None

    def load_goals(self, goals_file: str) -> bool:
        """Load goals from numpy file."""
        try:
            self.goals = np.load(goals_file)
            print(f"Loaded {len(self.goals)} goal directions from {goals_file}")
            return True
        except Exception as e:
            print(f"Could not load goals: {e}")
            return False

    def load_from_sequence(self, bin_folder: str) -> bool:
        """Auto-detect and load goals for a RELLIS-3D sequence."""
        bin_path = Path(bin_folder)
        seq_id = bin_path.name

        # Try to find goals.npy
        goals_path = bin_path.parent.parent / 'actions' / seq_id / 'goals.npy'
        if goals_path.exists():
            return self.load_goals(str(goals_path))
        return False

    def get_goal(self, frame_idx: int = None, state: VehicleState = None) -> NavigationGoal:
        """Get goal direction for current frame."""
        if self.mode == 'auto' and self.goals is not None and frame_idx is not None:
            if 0 <= frame_idx < len(self.goals):
                goal = self.goals[frame_idx]
                return NavigationGoal(dx=float(goal[0]), dy=float(goal[1]))

        if self.mode == 'velocity' and state is not None:
            # Use velocity direction as goal
            vx, vy = state.vx, state.vy
            norm = np.sqrt(vx**2 + vy**2)
            if norm > 0.1:  # Only if moving
                return NavigationGoal(dx=vx/norm, dy=vy/norm)

        return self.default_goal


# Action labels — 12-action forward-only space (must match generate_action_labels.py)
ACTION_LABELS = {
    0:  "Stop",
    1:  "Fwd Slow",
    2:  "Fwd Med",
    3:  "Fwd Fast",
    4:  "Left Sharp",
    5:  "Left Med",
    6:  "Left Slight",
    7:  "Right Slight",
    8:  "Right Med",
    9:  "Right Sharp",
    10: "Fwd+Left",
    11: "Fwd+Right",
}

# Category color per action (red=stop, green=fwd, blue=left, orange=right, teal=fwd+turn)
ACTION_COLORS = {
    0:  '#e74c3c',   # Stop         — red
    1:  '#2ecc71',   # Fwd Slow     — light green
    2:  '#27ae60',   # Fwd Med      — green
    3:  '#1a8040',   # Fwd Fast     — dark green
    4:  '#2980b9',   # Left Sharp   — blue
    5:  '#5dade2',   # Left Med     — medium blue
    6:  '#85c1e9',   # Left Slight  — light blue
    7:  '#f39c12',   # Right Slight — light orange
    8:  '#e67e22',   # Right Med    — orange
    9:  '#d35400',   # Right Sharp  — dark orange
    10: '#1abc9c',   # Fwd+Left     — teal
    11: '#16a085',   # Fwd+Right    — dark teal
}

# Direction arrow symbol per action
ACTION_ARROWS = {
    0:  '■',   # Stop
    1:  '↑',   # Fwd Slow
    2:  '↑',   # Fwd Med
    3:  '⇑',   # Fwd Fast  (double arrow = speed)
    4:  '↺',   # Left Sharp
    5:  '←',   # Left Med
    6:  '↖',   # Left Slight
    7:  '↗',   # Right Slight
    8:  '→',   # Right Med
    9:  '↻',   # Right Sharp
    10: '↰',   # Fwd+Left
    11: '↱',   # Fwd+Right
}


class DecisionPublisher:
    """
    Publishes decision results to external UI system via UDP.

    Sends JSON messages with action, confidence, and metadata.
    UI system should listen on the specified port.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 9999, enabled: bool = True):
        self.host = host
        self.port = port
        self.enabled = enabled
        self.socket = None

        if enabled:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            print(f"Decision publisher enabled: UDP {host}:{port}")

    def publish(self, result: Dict):
        """
        Publish decision result to UI.

        Args:
            result: Dictionary with action, confidence, etc.
        """
        if not self.enabled or self.socket is None:
            return

        try:
            # Create message with essential fields
            message = {
                "timestamp": time.time(),
                "action": int(result.get('action', 0)),
                "action_name": result.get('action_name', 'Unknown'),
                "confidence": float(result.get('confidence', 0.0)),
                "inference_ms": float(result.get('inference_time_ms', 0.0)),
                # Include top-3 actions with probabilities
                "action_probs": {},
            }

            # Add top action probabilities
            if 'action_probs' in result:
                probs = result['action_probs']
                top_idx = np.argsort(probs)[-3:][::-1]
                for idx in top_idx:
                    message["action_probs"][ACTION_LABELS.get(idx, f"A{idx}")] = float(probs[idx])

            # Add ground truth if available
            if 'gt_action' in result:
                message["gt_action"] = int(result['gt_action'])
                message["gt_action_name"] = result.get('gt_action_name', '')
                message["correct"] = result.get('correct', False)

            # Send UDP packet
            data = json.dumps(message).encode('utf-8')
            self.socket.sendto(data, (self.host, self.port))

        except Exception as e:
            print(f"Decision publish error: {e}")

    def close(self):
        if self.socket:
            self.socket.close()


class VelodyneUDPReceiver:
    """Direct UDP receiver for Velodyne LiDAR."""

    def __init__(self, port: int = 2368, device_ip: str = ''):
        self.port = port
        self.device_ip = device_ip
        self.socket = None
        self.running = False
        self.point_queue = queue.Queue(maxsize=10)

        # Velodyne VLP-16 parameters
        self.num_lasers = 16
        self.laser_angles = np.array([
            -15, 1, -13, 3, -11, 5, -9, 7,
            -7, 9, -5, 11, -3, 13, -1, 15
        ]) * np.pi / 180

    def start(self):
        """Start receiving UDP packets."""
        import socket
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind(('', self.port))
        self.socket.settimeout(0.1)
        self.running = True

        self.thread = threading.Thread(target=self._receive_loop, daemon=True)
        self.thread.start()
        print(f"Velodyne UDP receiver started on port {self.port}")

    def stop(self):
        """Stop receiving."""
        self.running = False
        if self.socket:
            self.socket.close()

    def _receive_loop(self):
        """Background thread for receiving packets."""
        points_buffer = []
        last_azimuth = 0

        while self.running:
            try:
                data, addr = self.socket.recvfrom(1248)
                if len(data) == 1206:  # Velodyne data packet
                    points = self._parse_packet(data)
                    points_buffer.extend(points)

                    # Check for new rotation (azimuth wrap)
                    if len(points_buffer) > 10000:
                        # Full scan ready
                        if not self.point_queue.full():
                            self.point_queue.put(np.array(points_buffer))
                        points_buffer = []

            except Exception as e:
                if self.running:
                    continue

    def _parse_packet(self, data: bytes) -> list:
        """Parse Velodyne VLP-16 packet."""
        points = []
        # Simplified parser - for production use velodyne_decoder
        for block in range(12):  # 12 data blocks per packet
            offset = block * 100
            azimuth = (data[offset+2] + data[offset+3] * 256) / 100.0 * np.pi / 180

            for laser in range(32):  # 2 firings x 16 lasers
                laser_id = laser % 16
                idx = offset + 4 + laser * 3
                distance = (data[idx] + data[idx+1] * 256) * 0.002  # meters
                intensity = data[idx+2]

                if distance > 0.5 and distance < 100:
                    vert_angle = self.laser_angles[laser_id]
                    x = distance * np.cos(vert_angle) * np.sin(azimuth)
                    y = distance * np.cos(vert_angle) * np.cos(azimuth)
                    z = distance * np.sin(vert_angle)
                    points.append([x, y, z, intensity / 255.0])

        return points

    def get_points(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """Get latest point cloud."""
        try:
            return self.point_queue.get(timeout=timeout)
        except queue.Empty:
            return None


class ROSReceiver:
    """ROS point cloud subscriber."""

    def __init__(self, topic: str = '/velodyne_points'):
        self.topic = topic
        self.point_queue = queue.Queue(maxsize=10)
        self.subscriber = None

    def start(self):
        """Start ROS subscriber."""
        try:
            import rospy
            from sensor_msgs.msg import PointCloud2
            import sensor_msgs.point_cloud2 as pc2

            if not rospy.core.is_initialized():
                rospy.init_node('terrainformer_inference', anonymous=True)

            self.subscriber = rospy.Subscriber(
                self.topic, PointCloud2, self._callback, queue_size=1
            )
            print(f"ROS subscriber started on topic {self.topic}")

        except ImportError:
            raise RuntimeError("ROS not available. Install ros-noetic-desktop or use --mode udp")

    def _callback(self, msg):
        """Process incoming point cloud."""
        import sensor_msgs.point_cloud2 as pc2

        points = []
        for p in pc2.read_points(msg, field_names=("x", "y", "z", "intensity"), skip_nans=True):
            points.append([p[0], p[1], p[2], p[3] / 255.0 if p[3] > 1 else p[3]])

        if points and not self.point_queue.full():
            self.point_queue.put(np.array(points))

    def stop(self):
        """Stop subscriber."""
        if self.subscriber:
            self.subscriber.unregister()

    def get_points(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """Get latest point cloud."""
        try:
            return self.point_queue.get(timeout=timeout)
        except queue.Empty:
            return None


class SimulatedLiDAR:
    """Simulated LiDAR for testing."""

    def __init__(self, hz: float = 10):
        self.hz = hz
        self.running = False

    def start(self):
        self.running = True
        print("Simulated LiDAR started")

    def stop(self):
        self.running = False

    def get_points(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """Generate synthetic point cloud."""
        time.sleep(1.0 / self.hz)

        N = 65536
        x = np.random.uniform(-40, 40, N)
        y = np.random.uniform(-40, 40, N)
        z = 0.1 * np.sin(x * 0.1) * np.cos(y * 0.1) + np.random.normal(0, 0.1, N)

        # Add obstacles
        for _ in range(10):
            ox, oy = np.random.uniform(-30, 30, 2)
            mask = ((x - ox)**2 + (y - oy)**2) < 4
            z[mask] += np.random.uniform(0.5, 2.0)

        intensity = (z - z.min()) / (z.max() - z.min() + 1e-6)
        return np.stack([x, y, z, intensity], axis=1)


class FolderDataSource:
    """Load point clouds from a folder with navigation support."""

    SUPPORTED_EXTENSIONS = ['.bin', '.npy', '.pcd', '.ply']

    def __init__(self, folder_path: Optional[str] = None):
        self.folder_path = folder_path
        self.files: List[Path] = []
        self.current_index = 0
        self.running = False
        self.pose_file: Optional[str] = None

        if folder_path:
            self.load_folder(folder_path)

    def load_folder(self, folder_path: str) -> int:
        """Load all point cloud files from folder."""
        self.folder_path = Path(folder_path)
        self.files = []
        self.pose_file = None

        if not self.folder_path.exists():
            print(f"Folder not found: {folder_path}")
            return 0

        # RELLIS-3D / SemanticKITTI: if a velodyne/ subfolder exists, load from there
        velodyne_dir = self.folder_path / 'velodyne'
        scan_dir = velodyne_dir if velodyne_dir.exists() else self.folder_path

        # Find all supported files (non-recursive from scan_dir)
        for ext in self.SUPPORTED_EXTENSIONS:
            self.files.extend(sorted(scan_dir.glob(f'*{ext}')))

        # Remove duplicates and sort
        self.files = sorted(list(set(self.files)))
        self.current_index = 0

        # Determine sequence directory (the folder containing poses.txt / labels)
        # If we stepped into velodyne/, go up one level; otherwise use folder_path itself
        seq_dir = self.folder_path if scan_dir == self.folder_path else self.folder_path
        seq_id = seq_dir.name  # e.g., "00000"

        # Try to find action labels file (RELLIS-3D format)
        # Expected: <dataset_root>/actions/{seq_id}/actions.npy
        self.actions_file = None
        try:
            actions_path = seq_dir.parent.parent / 'actions' / seq_id / 'actions.npy'
            if actions_path.exists():
                self.actions_file = str(actions_path)
                print(f"Found action labels: {actions_path}")
            else:
                # Look for poses.txt in sequence directory (RELLIS-3D structure)
                pose_candidates = [
                    seq_dir / 'poses.txt',                              # <seq>/poses.txt
                    seq_dir.parent / seq_id / 'poses.txt',              # alternate
                    seq_dir.parent.parent / 'calib' / seq_id / 'poses.txt',  # legacy
                ]
                for pose_path in pose_candidates:
                    if pose_path.exists():
                        self.pose_file = str(pose_path)
                        print(f"Found pose file: {pose_path}")
                        break
        except Exception as e:
            print(f"Could not find action labels: {e}")

        print(f"Loaded {len(self.files)} point cloud files from {folder_path}")
        return len(self.files)

    def load_file(self, file_path: Path) -> Optional[np.ndarray]:
        """Load a single point cloud file."""
        try:
            suffix = file_path.suffix.lower()

            if suffix == '.bin':
                # KITTI/RELLIS format: N x 4 float32
                points = np.fromfile(str(file_path), dtype=np.float32)
                if len(points) % 4 == 0:
                    points = points.reshape(-1, 4)
                else:
                    # Try 3-channel
                    points = points.reshape(-1, 3)
                    # Add dummy intensity
                    intensity = np.zeros((len(points), 1), dtype=np.float32)
                    points = np.hstack([points, intensity])

            elif suffix == '.npy':
                points = np.load(str(file_path))
                if points.ndim == 1:
                    points = points.reshape(-1, 4)
                if points.shape[1] == 3:
                    intensity = np.zeros((len(points), 1), dtype=np.float32)
                    points = np.hstack([points, intensity])

            elif suffix == '.pcd':
                points = self._load_pcd(file_path)

            elif suffix == '.ply':
                points = self._load_ply(file_path)

            else:
                print(f"Unsupported format: {suffix}")
                return None

            return points.astype(np.float32)

        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None

    def _load_pcd(self, file_path: Path) -> np.ndarray:
        """Load PCD file (basic ASCII/binary support)."""
        points = []
        with open(file_path, 'rb') as f:
            header_done = False
            is_binary = False
            num_points = 0

            for line in f:
                line_str = line.decode('utf-8', errors='ignore').strip()

                if not header_done:
                    if line_str.startswith('DATA'):
                        is_binary = 'binary' in line_str.lower()
                        header_done = True
                        if is_binary:
                            # Read binary data
                            data = f.read()
                            points = np.frombuffer(data, dtype=np.float32).reshape(-1, 4)
                            break
                    elif line_str.startswith('POINTS'):
                        num_points = int(line_str.split()[-1])
                else:
                    # ASCII mode
                    vals = line_str.split()
                    if len(vals) >= 3:
                        x, y, z = float(vals[0]), float(vals[1]), float(vals[2])
                        intensity = float(vals[3]) if len(vals) > 3 else 0.0
                        points.append([x, y, z, intensity])

        return np.array(points, dtype=np.float32)

    def _load_ply(self, file_path: Path) -> np.ndarray:
        """Load PLY file (basic ASCII support)."""
        points = []
        with open(file_path, 'r') as f:
            header_done = False
            for line in f:
                line = line.strip()
                if not header_done:
                    if line == 'end_header':
                        header_done = True
                else:
                    vals = line.split()
                    if len(vals) >= 3:
                        x, y, z = float(vals[0]), float(vals[1]), float(vals[2])
                        intensity = float(vals[3]) if len(vals) > 3 else 0.0
                        points.append([x, y, z, intensity])

        return np.array(points, dtype=np.float32)

    def start(self):
        """Start data source."""
        self.running = True
        if len(self.files) > 0:
            print(f"Folder data source ready: {len(self.files)} files")
        else:
            print("Folder data source started (no files loaded)")

    def stop(self):
        """Stop data source."""
        self.running = False

    def get_points(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """Get current point cloud."""
        if len(self.files) == 0:
            return None
        return self.load_file(self.files[self.current_index])

    def next(self) -> bool:
        """Move to next file."""
        if len(self.files) == 0:
            return False
        self.current_index = (self.current_index + 1) % len(self.files)
        return True

    def prev(self) -> bool:
        """Move to previous file."""
        if len(self.files) == 0:
            return False
        self.current_index = (self.current_index - 1) % len(self.files)
        return True

    def goto(self, index: int) -> bool:
        """Go to specific index."""
        if 0 <= index < len(self.files):
            self.current_index = index
            return True
        return False

    def get_current_filename(self) -> str:
        """Get current filename."""
        if len(self.files) == 0:
            return "No files loaded"
        return self.files[self.current_index].name

    def get_progress(self) -> Tuple[int, int]:
        """Get current progress (index, total)."""
        return self.current_index + 1, len(self.files)


class PoseLoader:
    """Load pre-generated action labels for ground truth comparison."""

    def __init__(self, actions_file: Optional[str] = None):
        self.actions = None
        if actions_file:
            self.load_actions(actions_file)

    def load_actions(self, actions_file: str):
        """Load pre-generated action labels from .npy file."""
        try:
            self.actions = np.load(actions_file)
            print(f"Loaded {len(self.actions)} action labels from {actions_file}")
        except Exception as e:
            print(f"Error loading actions: {e}")
            self.actions = None

    def load_from_sequence(self, bin_folder: str):
        """Auto-detect and load action labels for a RELLIS-3D sequence."""
        bin_path = Path(bin_folder)
        seq_id = bin_path.name  # e.g., "00000"

        # Try: ../actions/{seq_id}/actions.npy
        actions_path = bin_path.parent.parent / 'actions' / seq_id / 'actions.npy'
        if actions_path.exists():
            self.load_actions(str(actions_path))
            return True

        # Try: ../../actions/{seq_id}/actions.npy (if bin_folder is deeper)
        actions_path = bin_path.parent.parent.parent / 'actions' / seq_id / 'actions.npy'
        if actions_path.exists():
            self.load_actions(str(actions_path))
            return True

        print(f"No action labels found for sequence {seq_id}")
        return False

    def get_action(self, frame_idx: int) -> Optional[int]:
        """Get ground truth action for frame index."""
        if self.actions is None or frame_idx >= len(self.actions):
            return None
        return self.actions[frame_idx]

    def get_action_name(self, frame_idx: int) -> str:
        """Get ground truth action name for frame index."""
        action = self.get_action(frame_idx)
        if action is None:
            return "N/A"
        return ACTION_LABELS.get(action, f"Action {action}")


class TerrainFormerInference:
    """Real-time TerrainFormer inference engine with temporal context."""

    def __init__(self,
                 checkpoint_path: Optional[str] = None,
                 world_model_checkpoint: Optional[str] = None,
                 device: str = 'cuda',
                 encoder_type: str = 'pointpillars',
                 use_temporal: bool = True,
                 temporal_mode: str = 'voting',  # 'voting', 'new_model', or 'none'
                 num_past_frames: int = 10,
                 num_future_frames: int = 10,
                 publish_decisions: bool = False,
                 publish_host: str = "127.0.0.1",
                 publish_port: int = 9999):
        self.device = device
        self.encoder_type = encoder_type
        self.use_temporal = use_temporal
        self.temporal_mode = temporal_mode
        self.num_past_frames = num_past_frames
        self.num_future_frames = num_future_frames

        # Decision publisher for external UI
        self.publisher = DecisionPublisher(
            host=publish_host,
            port=publish_port,
            enabled=publish_decisions
        )

        # Always load the base TerrainFormer model (works with existing checkpoint)
        print("Loading TerrainFormer model...")
        self.model = TerrainFormer(
            lidar_in_channels=4,
            bev_size=256,
            num_actions=12,  # Simplified forward-only action space
            encoder_type=encoder_type,
        ).to(device)
        self.model.eval()

        if checkpoint_path and Path(checkpoint_path).exists():
            print(f"Loading checkpoint: {checkpoint_path}")
            ckpt = torch.load(checkpoint_path, map_location=device)
            self.model.load_state_dict(ckpt['model_state_dict'], strict=False)

        # Load pretrained world model weights (for traversability visualization)
        # Auto-detect if not specified: look alongside the main checkpoint
        wm_path = world_model_checkpoint
        if wm_path is None and checkpoint_path:
            # Try outputs/world_model_pretrain/best_model.pt relative to project root
            project_root = Path(checkpoint_path).parent.parent
            candidate = project_root / 'world_model_pretrain' / 'best_model.pt'
            if candidate.exists():
                wm_path = str(candidate)
                print(f"Auto-detected world model checkpoint: {wm_path}")

        if wm_path and Path(wm_path).exists():
            print(f"Loading pretrained world model: {wm_path}")
            wm_ckpt = torch.load(wm_path, map_location=device)
            wm_state = wm_ckpt['model_state_dict']
            # Standalone WorldModel checkpoint keys lack the 'world_model.' prefix
            # that TerrainFormer uses for its submodule — remap them
            if not any(k.startswith('world_model.') for k in wm_state):
                wm_state = {f'world_model.{k}': v for k, v in wm_state.items()}
            missing, _ = self.model.load_state_dict(wm_state, strict=False)
            loaded = len(wm_state) - len(missing)
            print(f"World model weights loaded: {loaded}/{len(wm_state)} keys")
        else:
            print("Warning: No world model checkpoint found — traversability uses random weights")

        # Action history buffer
        self.action_history = deque(maxlen=10)
        for _ in range(10):
            self.action_history.append(1)  # Start with FORWARD_SLOW (action 1)

        # Frame buffer for temporal context (stores raw points)
        self.frame_buffer: deque = deque(maxlen=num_past_frames)

        # Prediction buffer for temporal voting
        self.pred_buffer: deque = deque(maxlen=num_past_frames)

        # Performance tracking
        self.inference_times = deque(maxlen=100)

        # Action label loader for ground truth
        self.pose_loader: Optional[PoseLoader] = None

        mode_str = f"temporal={use_temporal}, mode={temporal_mode}" if use_temporal else "single-frame"
        print(f"Model loaded ({encoder_type} encoder, {mode_str})")

    def load_actions(self, actions_file: str):
        """Load action labels file for ground truth comparison."""
        self.pose_loader = PoseLoader(actions_file)

    def load_actions_from_sequence(self, bin_folder: str):
        """Auto-detect and load action labels for a RELLIS-3D sequence."""
        self.pose_loader = PoseLoader()
        return self.pose_loader.load_from_sequence(bin_folder)

    def load_poses(self, pose_file: str):
        """Deprecated: Use load_actions or load_actions_from_sequence instead."""
        # For backwards compatibility, try to find action labels from pose file path
        pose_path = Path(pose_file)
        seq_id = pose_path.parent.name  # e.g., "00000"
        actions_path = pose_path.parent.parent.parent / 'actions' / seq_id / 'actions.npy'
        if actions_path.exists():
            self.load_actions(str(actions_path))
        else:
            print(f"Warning: Could not find action labels at {actions_path}")
            print("Run 'python scripts/generate_action_labels.py' to generate them")

    def preprocess_points(self, points: np.ndarray) -> torch.Tensor:
        """Preprocess point cloud for inference with deterministic sampling."""
        # Filter range
        distances = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
        valid = (distances > 0.5) & (distances < 70)
        points = points[valid]

        # Deterministic subsample to fixed size (evenly spaced indices for reproducibility)
        target_size = 65536
        if len(points) > target_size:
            # Use evenly spaced indices instead of random sampling
            idx = np.linspace(0, len(points) - 1, target_size, dtype=np.int64)
            points = points[idx]
        elif len(points) < target_size:
            # Pad by repeating points cyclically (deterministic)
            pad_size = target_size - len(points)
            if len(points) > 0:
                # Repeat points cyclically to fill
                repeat_idx = np.arange(pad_size) % len(points)
                points = np.vstack([points, points[repeat_idx]])
            else:
                points = np.zeros((target_size, 4))

        return torch.tensor(points, dtype=torch.float32, device=self.device).unsqueeze(0)

    def add_to_buffer(self, points: np.ndarray):
        """Add preprocessed points to frame buffer."""
        points_tensor = self.preprocess_points(points)
        self.frame_buffer.append(points_tensor)

    def get_past_points(self) -> Optional[torch.Tensor]:
        """Get past frames from buffer as tensor."""
        if len(self.frame_buffer) == 0:
            return None
        # Stack past frames: (1, T, N, C)
        past = torch.cat(list(self.frame_buffer), dim=0).unsqueeze(0)
        return past

    @torch.no_grad()
    def infer(self,
              points: np.ndarray,
              state: VehicleState,
              goal: NavigationGoal,
              frame_idx: Optional[int] = None) -> Dict:
        """Run inference on point cloud with temporal context."""
        t0 = time.perf_counter()

        # Preprocess current frame
        current_tensor = self.preprocess_points(points)
        state_tensor = state.to_tensor(self.device)
        goal_tensor = goal.to_tensor(self.device)
        action_history_tensor = torch.tensor(
            [list(self.action_history)], dtype=torch.long, device=self.device
        )

        # Run base model inference (explicitly request world predictions for traversability)
        outputs = self.model(current_tensor, state_tensor, goal_tensor, action_history_tensor,
                            return_world_predictions=True)

        # Debug: log available output keys once
        if not hasattr(self, '_logged_output_keys'):
            print(f"Model output keys: {list(outputs.keys())}")
            self._logged_output_keys = True

        # Get current frame action
        action_logits = outputs['action_logits']
        action_probs = torch.softmax(action_logits, dim=-1)
        current_action = action_logits.argmax(dim=-1).item()
        current_confidence = action_probs[0, current_action].item()
        current_probs = action_probs[0].cpu().numpy()

        # Temporal voting: aggregate predictions from past frames
        if self.use_temporal and self.temporal_mode == 'voting':
            # Store current prediction in buffer
            self.pred_buffer.append({
                'action': current_action,
                'confidence': current_confidence,
                'probs': current_probs.copy()
            })

            # Aggregate predictions using confidence-weighted voting
            if len(self.pred_buffer) >= 3:  # Need at least 3 frames
                # Method 1: Confidence-weighted voting
                action_scores = np.zeros(12)  # 12 forward-only actions
                total_weight = 0

                for pred in self.pred_buffer:
                    weight = pred['confidence']
                    action_scores[pred['action']] += weight
                    total_weight += weight

                # Also add probability mass from all predictions
                avg_probs = np.zeros(12)  # 12 forward-only actions
                for pred in self.pred_buffer:
                    avg_probs += pred['probs'] * pred['confidence']
                avg_probs /= (total_weight + 1e-6)

                # Combine voting score and average probability
                combined_scores = 0.5 * (action_scores / (total_weight + 1e-6)) + 0.5 * avg_probs

                # Select action with highest combined score
                action = int(np.argmax(combined_scores))
                confidence = combined_scores[action]
                action_probs_final = combined_scores
            else:
                # Not enough frames yet, use current prediction
                action = current_action
                confidence = current_confidence
                action_probs_final = current_probs
        else:
            # No temporal voting, use current frame only
            action = current_action
            confidence = current_confidence
            action_probs_final = current_probs

        # Update history and buffer
        self.action_history.append(action)
        self.add_to_buffer(points)

        # Timing
        t1 = time.perf_counter()
        inference_time = (t1 - t0) * 1000
        self.inference_times.append(inference_time)

        result = {
            'action': action,
            'action_name': ACTION_LABELS.get(action, f"Action {action}"),
            'confidence': confidence,
            'action_probs': action_probs_final,
            'inference_time_ms': inference_time,
            'avg_inference_ms': np.mean(self.inference_times),
            'fps': 1000 / inference_time,
            'bev': outputs.get('world_traversability'),
            'traversability': outputs.get('world_traversability'),
            'temporal_context': len(self.pred_buffer) if self.use_temporal else 0,
            'raw_action': current_action,  # Original single-frame prediction
            'raw_confidence': current_confidence,
        }

        # Add ground truth if available
        if frame_idx is not None and self.pose_loader is not None:
            gt_action = self.pose_loader.get_action(frame_idx)
            gt_name = self.pose_loader.get_action_name(frame_idx)
            result['gt_action'] = gt_action
            result['gt_action_name'] = gt_name
            result['correct'] = (gt_action == action) if gt_action is not None else None
            # Also track raw (non-voting) accuracy
            result['raw_correct'] = (gt_action == current_action) if gt_action is not None else None

        # Publish to external UI system
        self.publisher.publish(result)

        return result

    def clear_buffer(self):
        """Clear the frame buffer."""
        self.frame_buffer.clear()


class RealTimeVisualizer:
    """Real-time visualization of inference results."""

    def __init__(self, inference_engine: TerrainFormerInference, folder_mode: bool = False,
                 goal_mode: str = 'manual'):
        self.engine = inference_engine
        self.lidar_source = None
        self.state = VehicleState()
        self.goal = NavigationGoal()
        self.folder_mode = folder_mode

        # Goal provider for automatic goal direction
        self.goal_provider = GoalProvider(mode=goal_mode)
        self.goal_mode = goal_mode

        # Visualization state
        self.latest_points = None
        self.latest_result = None
        self.running = False
        self.auto_play = False
        self.needs_update = True

        # Setup figure
        plt.ion()
        self.fig = plt.figure(figsize=(22, 10))
        self.fig.suptitle('TerrainFormer Real-Time Inference', fontsize=23)

        # Layout: Large point cloud on left, smaller panels on right
        # Column widths: [2, 1, 1, 1, 1] — point cloud takes 2x width; 5th col = trajectory
        if folder_mode:
            self.gs = GridSpec(3, 5, figure=self.fig, hspace=0.3, wspace=0.28,
                             height_ratios=[1, 1, 0.12], width_ratios=[2, 1, 1, 1, 1])
        else:
            self.gs = GridSpec(2, 5, figure=self.fig, hspace=0.3, wspace=0.28,
                             width_ratios=[2, 1, 1, 1, 1])

        # Main point cloud plot (large, spans 2 rows and 2 columns)
        self.ax_points = self.fig.add_subplot(self.gs[0:2, 0:2])

        # Smaller panels on the right
        self.ax_trav = self.fig.add_subplot(self.gs[0, 2])      # Traversability
        self.ax_action = self.fig.add_subplot(self.gs[0, 3])    # Decision card
        self.ax_probs = self.fig.add_subplot(self.gs[1, 2])     # Action probabilities
        self.ax_info = self.fig.add_subplot(self.gs[1, 3])      # Info panel
        self.ax_traj = self.fig.add_subplot(self.gs[0:2, 4])    # Trajectory (spans both rows)

        # Trajectory data (loaded from poses.txt)
        self.traj_positions = None   # (N, 2) float32 [x, y]

        # Navigation buttons (folder mode only)
        self.btn_load = None
        self.btn_prev = None
        self.btn_next = None
        self.btn_play = None

        if folder_mode:
            self._setup_navigation_buttons()

        # Initialize plots
        self._init_plots()

    def _setup_navigation_buttons(self):
        """Setup navigation buttons for folder mode."""
        # Button axes
        btn_width = 0.12
        btn_height = 0.04
        btn_y = 0.02
        spacing = 0.02

        # Load button
        ax_load = self.fig.add_axes([0.1, btn_y, btn_width, btn_height])
        self.btn_load = Button(ax_load, 'Load Folder', color='lightblue', hovercolor='deepskyblue')
        self.btn_load.on_clicked(self._on_load_clicked)

        # Previous button
        ax_prev = self.fig.add_axes([0.1 + btn_width + spacing, btn_y, btn_width * 0.8, btn_height])
        self.btn_prev = Button(ax_prev, '< Prev', color='lightgray', hovercolor='silver')
        self.btn_prev.on_clicked(self._on_prev_clicked)

        # Play/Pause button
        ax_play = self.fig.add_axes([0.1 + btn_width * 1.8 + spacing * 2, btn_y, btn_width * 0.8, btn_height])
        self.btn_play = Button(ax_play, 'Play', color='lightgreen', hovercolor='limegreen')
        self.btn_play.on_clicked(self._on_play_clicked)

        # Next button
        ax_next = self.fig.add_axes([0.1 + btn_width * 2.6 + spacing * 3, btn_y, btn_width * 0.8, btn_height])
        self.btn_next = Button(ax_next, 'Next >', color='lightgray', hovercolor='silver')
        self.btn_next.on_clicked(self._on_next_clicked)

        # File info text
        self.ax_file_info = self.fig.add_axes([0.5, btn_y, 0.45, btn_height])
        self.ax_file_info.axis('off')
        self.file_info_text = self.ax_file_info.text(
            0.5, 0.5, 'No folder loaded',
            ha='center', va='center', fontsize=18,
            transform=self.ax_file_info.transAxes
        )

    def _on_load_clicked(self, event):
        """Handle load button click."""
        # Open folder dialog
        root = tk.Tk()
        root.withdraw()  # Hide main window
        folder_path = filedialog.askdirectory(title='Select Point Cloud Folder')
        root.destroy()

        if folder_path and isinstance(self.lidar_source, FolderDataSource):
            num_files = self.lidar_source.load_folder(folder_path)
            if num_files > 0:
                # Load action labels if available
                if self.lidar_source.actions_file:
                    self.engine.load_actions(self.lidar_source.actions_file)
                elif self.lidar_source.pose_file:
                    self.engine.load_poses(self.lidar_source.pose_file)
                # Load trajectory (XY positions) for the trajectory panel
                if self.lidar_source.pose_file:
                    self.load_trajectory(self.lidar_source.pose_file)
                self.engine.clear_buffer()  # Reset frame buffer
                self._update_file_info()
                self._update_frame(None)  # Force immediate update

    def _on_prev_clicked(self, event):
        """Handle previous button click."""
        if isinstance(self.lidar_source, FolderDataSource):
            self.lidar_source.prev()
            self._update_file_info()
            self._update_frame(None)  # Force immediate update

    def _on_next_clicked(self, event):
        """Handle next button click."""
        if isinstance(self.lidar_source, FolderDataSource):
            self.lidar_source.next()
            self._update_file_info()
            self._update_frame(None)  # Force immediate update

    def _on_play_clicked(self, event):
        """Handle play/pause button click."""
        self.auto_play = not self.auto_play
        if self.btn_play:
            self.btn_play.label.set_text('Pause' if self.auto_play else 'Play')
            self.btn_play.color = 'lightyellow' if self.auto_play else 'lightgreen'
            self.btn_play.hovercolor = 'yellow' if self.auto_play else 'limegreen'

    def _update_file_info(self):
        """Update file info display."""
        if hasattr(self, 'file_info_text') and isinstance(self.lidar_source, FolderDataSource):
            current, total = self.lidar_source.get_progress()
            filename = self.lidar_source.get_current_filename()
            self.file_info_text.set_text(f'{filename}  [{current}/{total}]')
            self.fig.canvas.draw_idle()

    def load_trajectory(self, pose_file: str) -> bool:
        """Load trajectory XY positions from a poses.txt file (3×4 [R|t] per line)."""
        try:
            positions = []
            with open(pose_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    vals = list(map(float, line.split()))
                    if len(vals) == 12:
                        mat = np.array(vals).reshape(3, 4)
                        positions.append([mat[0, 3], mat[1, 3]])  # X, Y
            if positions:
                self.traj_positions = np.array(positions, dtype=np.float32)
                print(f"Loaded {len(positions)} trajectory positions from {pose_file}")
                return True
            print(f"No valid poses found in {pose_file}")
        except Exception as e:
            print(f"Could not load trajectory from {pose_file}: {e}")
        return False

    @staticmethod
    def compute_traversability(points: np.ndarray,
                                bev_size: int = 256,
                                x_range: tuple = (-50, 50),
                                y_range: tuple = (-50, 50)) -> np.ndarray:
        """
        Compute BEV traversability map from raw point cloud geometry.

        Flat, dense areas (low height variance) → traversable (green).
        Rough / obstacle areas (high height variance) → not traversable (red).
        Empty cells → uncertain (0.5, yellow).

        Args:
            points: (N, 4) point cloud [x, y, z, intensity]

        Returns:
            (bev_size, bev_size) float32 array in [0, 1]
        """
        x_res = (x_range[1] - x_range[0]) / bev_size
        y_res = (y_range[1] - y_range[0]) / bev_size

        xs = ((points[:, 0] - x_range[0]) / x_res).astype(np.int32)
        ys = ((points[:, 1] - y_range[0]) / y_res).astype(np.int32)
        zs = points[:, 2]

        valid = (xs >= 0) & (xs < bev_size) & (ys >= 0) & (ys < bev_size)
        xs, ys, zs = xs[valid], ys[valid], zs[valid]

        count = np.zeros((bev_size, bev_size), dtype=np.float32)
        z_sum = np.zeros((bev_size, bev_size), dtype=np.float32)
        z_sq_sum = np.zeros((bev_size, bev_size), dtype=np.float32)

        np.add.at(count, (ys, xs), 1)
        np.add.at(z_sum, (ys, xs), zs)

        has_pts = count > 0
        z_mean = np.where(has_pts, z_sum / np.where(has_pts, count, 1), 0)

        z_dev = zs - z_mean[ys, xs]
        np.add.at(z_sq_sum, (ys, xs), z_dev ** 2)
        z_var = np.where(has_pts, z_sq_sum / np.where(has_pts, count, 1), 0)

        # Flat = traversable (1.0), rough = obstacle (0.0), empty = uncertain (0.5)
        flatness = np.exp(-z_var * 10.0)
        trav = np.where(has_pts, flatness, 0.5)
        return trav.astype(np.float32)

    def _init_plots(self):
        """Initialize plot elements."""
        # Main point cloud plot (large)
        self.ax_points.set_xlim(-50, 50)
        self.ax_points.set_ylim(-50, 50)
        self.ax_points.set_aspect('equal')
        self.ax_points.set_title('LiDAR Point Cloud (BEV)', fontsize=20, fontweight='bold')
        self.ax_points.set_xlabel('X (m)')
        self.ax_points.set_ylabel('Y (m)')
        self.ax_points.grid(True, alpha=0.3)
        self.scatter = self.ax_points.scatter([], [], s=0.5, c=[], cmap='terrain')

        # Traversability BEV panel
        self.ax_trav.set_title('Traversability', fontsize=18, fontweight='bold')
        self.trav_img = self.ax_trav.imshow(
            np.zeros((256, 256)), cmap='RdYlGn',
            extent=[-50, 50, -50, 50], vmin=0, vmax=1,
            origin='lower'
        )
        self.ax_trav.set_aspect('equal')
        self.ax_trav.set_xlabel('X (m)', fontsize=14)
        self.ax_trav.set_ylabel('Y (m)', fontsize=14)
        self.ax_trav.tick_params(labelsize=10)

        # Ego vehicle marker at origin
        self.ax_trav.plot(0, 0, marker='^', color='white', markersize=7,
                          markeredgecolor='black', markeredgewidth=1, zorder=5)

        # Colorbar: red=obstacle, yellow=unknown, green=safe
        cbar = self.fig.colorbar(self.trav_img, ax=self.ax_trav,
                                  orientation='vertical', fraction=0.046, pad=0.04)
        cbar.set_ticks([0.0, 0.5, 1.0])
        cbar.set_ticklabels(['Obstacle', 'Unknown', 'Safe'], fontsize=14)

        # Decision card (right panel)
        self.ax_action.set_xlim(0, 1)
        self.ax_action.set_ylim(0, 1)
        self.ax_action.set_title('Decision', fontsize=19, fontweight='bold')
        self.ax_action.axis('off')

        # Action probability bars
        self.ax_probs.set_title('Action Probs', fontsize=18, fontweight='bold')
        self.ax_probs.set_xlim(0, 1)

        # Info panel
        self.ax_info.axis('off')

        # Trajectory panel
        self.ax_traj.set_title('Trajectory', fontsize=18, fontweight='bold')
        self.ax_traj.set_facecolor('white')
        self.ax_traj.grid(True, alpha=0.4, color='#dddddd')
        self.ax_traj.set_aspect('equal')
        self.ax_traj.tick_params(labelsize=9)
        self.ax_traj.text(0.5, 0.5, 'No trajectory\n(load poses.txt)',
                          ha='center', va='center', fontsize=16,
                          color='#aaaaaa', transform=self.ax_traj.transAxes)

    def set_lidar_source(self, source):
        """Set LiDAR data source."""
        self.lidar_source = source

    def set_state(self, state: VehicleState):
        """Update vehicle state."""
        self.state = state

    def set_goal(self, goal: NavigationGoal):
        """Update navigation goal."""
        self.goal = goal

    def _update_frame(self, frame):
        """Update visualization frame."""
        if self.lidar_source is None:
            return

        # Get point cloud
        points = self.lidar_source.get_points(timeout=0.1)
        if points is None:
            return

        self.latest_points = points

        # Get frame index for ground truth
        frame_idx = None
        if isinstance(self.lidar_source, FolderDataSource):
            frame_idx = self.lidar_source.current_index

        # Get goal direction (auto or manual)
        if self.goal_mode == 'auto':
            current_goal = self.goal_provider.get_goal(frame_idx=frame_idx, state=self.state)
        else:
            current_goal = self.goal

        # Run inference with frame index for ground truth comparison
        result = self.engine.infer(points, self.state, current_goal, frame_idx=frame_idx)
        result['goal'] = current_goal  # Store goal for visualization
        self.latest_result = result

        # Update main point cloud plot (large)
        self.ax_points.clear()
        self.ax_points.set_xlim(-50, 50)
        self.ax_points.set_ylim(-50, 50)
        self.ax_points.set_aspect('equal')
        self.ax_points.grid(True, alpha=0.3)
        self.ax_points.set_xlabel('X (m)')
        self.ax_points.set_ylabel('Y (m)')

        # Show filename in folder mode
        if self.folder_mode and isinstance(self.lidar_source, FolderDataSource):
            current, total = self.lidar_source.get_progress()
            filename = self.lidar_source.get_current_filename()
            self.ax_points.set_title(f'{filename}  [{current}/{total}]  ({len(points):,} pts)',
                                    fontsize=19, fontweight='bold')
        else:
            self.ax_points.set_title(f'LiDAR Point Cloud ({len(points):,} pts)',
                                    fontsize=19, fontweight='bold')

        # Display more points in the larger plot (deterministic subsampling)
        max_display_points = 20000
        if len(points) > max_display_points:
            # Use evenly spaced indices for deterministic display
            idx = np.linspace(0, len(points) - 1, max_display_points, dtype=np.int64)
            disp_pts = points[idx]
        else:
            disp_pts = points

        # Color by height
        z_norm = (disp_pts[:, 2] - disp_pts[:, 2].min()) / (disp_pts[:, 2].max() - disp_pts[:, 2].min() + 1e-6)
        colors = plt.cm.viridis(z_norm)
        self.ax_points.scatter(disp_pts[:, 0], disp_pts[:, 1], s=0.5, c=colors, alpha=0.8)

        # Draw ego vehicle marker
        self.ax_points.plot(0, 0, 'r^', markersize=12, markeredgecolor='black', markeredgewidth=1)
        self.ax_points.annotate('EGO', (0, 0), textcoords="offset points", xytext=(10, 10),
                               fontsize=15, color='red', fontweight='bold')

        # Update traversability: height-variance proxy (flat=safe, rough=obstacle, empty=unknown)
        if self.latest_points is not None:
            trav = self.compute_traversability(self.latest_points)
            self.trav_img.set_data(trav)
        self.ax_trav.set_title('Traversability  (height variance)',
                                fontsize=16, fontweight='bold')

        # ── Decision card ──────────────────────────────────────────────────
        self.ax_action.clear()
        self.ax_action.set_xlim(0, 1)
        self.ax_action.set_ylim(0, 1)
        self.ax_action.axis('off')

        action_id   = result['action']
        action_name = result['action_name']
        confidence  = result['confidence']
        color       = ACTION_COLORS.get(action_id, '#888888')
        arrow       = ACTION_ARROWS.get(action_id, '?')

        # Tinted background
        self.ax_action.set_facecolor(color + '22')

        # "Vote" / "Pred" tag (top-left, small)
        label_prefix = "Vote" if result.get('temporal_context', 0) >= 3 else "Pred"
        self.ax_action.text(0.05, 0.96, label_prefix,
                            fontsize=15, ha='left', va='top', color='#888888',
                            style='italic', transform=self.ax_action.transAxes)

        # Goal direction tag (top-right, small green)
        current_goal = result.get('goal', self.goal)
        goal_dx = current_goal.dx if hasattr(current_goal, 'dx') else current_goal.get('dx', 1.0)
        goal_dy = current_goal.dy if hasattr(current_goal, 'dy') else current_goal.get('dy', 0.0)
        if abs(goal_dy) > 0.3:
            goal_label = "← goal" if goal_dy > 0 else "goal →"
        elif goal_dx < 0:
            goal_label = "↓ goal"
        else:
            goal_label = "↑ goal"
        self.ax_action.text(0.95, 0.96, goal_label,
                            fontsize=15, ha='right', va='top', color='#27ae60',
                            transform=self.ax_action.transAxes)

        # Large direction arrow (center-top)
        self.ax_action.text(0.5, 0.72, arrow,
                            fontsize=68, ha='center', va='center', color=color,
                            transform=self.ax_action.transAxes)

        # Action name (large, color-coded for GT match)
        pred_color = '#222222'
        if 'gt_action_name' in result and result.get('correct') is not None:
            pred_color = '#27ae60' if result['correct'] else '#c0392b'
        self.ax_action.text(0.5, 0.47, action_name,
                            fontsize=22, ha='center', va='center',
                            color=pred_color, fontweight='bold',
                            transform=self.ax_action.transAxes)

        # Confidence progress bar
        bar_l, bar_r = 0.08, 0.92
        bar_y, bar_h = 0.32, 0.055
        # Gray track
        self.ax_action.add_patch(
            mpatches.Rectangle((bar_l, bar_y), bar_r - bar_l, bar_h,
                           transform=self.ax_action.transAxes,
                           color='#dddddd', zorder=1, clip_on=False))
        # Filled portion
        self.ax_action.add_patch(
            mpatches.Rectangle((bar_l, bar_y), (bar_r - bar_l) * confidence, bar_h,
                           transform=self.ax_action.transAxes,
                           color=color, zorder=2, clip_on=False))
        # Confidence value
        self.ax_action.text(0.5, 0.22, f'{confidence:.0%}',
                            fontsize=19, ha='center', va='center', color='#555555',
                            transform=self.ax_action.transAxes)

        # Raw (single-frame) if different from voted
        if result.get('raw_action') is not None and result['raw_action'] != result['action']:
            raw_name = ACTION_LABELS.get(result['raw_action'], f"A{result['raw_action']}")
            raw_color = '#aaaaaa'
            if 'gt_action' in result and result['gt_action'] is not None:
                raw_color = '#27ae60' if result['raw_action'] == result['gt_action'] else '#c0392b'
            self.ax_action.text(0.5, 0.12, f'raw: {raw_name}',
                                fontsize=15, ha='center', va='center',
                                color=raw_color, style='italic',
                                transform=self.ax_action.transAxes)

        # Ground truth row at bottom
        if 'gt_action_name' in result:
            match_sym = '✓' if result.get('correct') else '✗'
            gt_color  = '#2980b9' if result.get('correct') else '#c0392b'
            self.ax_action.text(0.5, 0.04, f'GT: {result["gt_action_name"]}  {match_sym}',
                                fontsize=16, ha='center', va='bottom',
                                color=gt_color, fontweight='bold',
                                transform=self.ax_action.transAxes)

        self.ax_action.set_title('Decision', fontsize=19, fontweight='bold')

        # ── Action probability bars (all 12 actions) ───────────────────────
        self.ax_probs.clear()
        probs   = result['action_probs']
        n_act   = len(ACTION_LABELS)   # 12
        bar_colors = []
        for i in range(n_act):
            c = ACTION_COLORS.get(i, '#888888')
            # Dim unselected actions so the winner stands out
            bar_colors.append(c if i == result['action'] else c + '66')
        labels = [ACTION_LABELS.get(i, f"A{i}") for i in range(n_act)]

        bars = self.ax_probs.barh(range(n_act), probs[:n_act],
                                   color=bar_colors, height=0.7,
                                   edgecolor='none')

        # Percentage label on each bar (skip very small ones)
        for i, (bar, p) in enumerate(zip(bars, probs[:n_act])):
            if p >= 0.04:
                self.ax_probs.text(p + 0.01, i, f'{p:.0%}',
                                   va='center', fontsize=14, color='#333333')

        self.ax_probs.set_yticks(range(n_act))
        self.ax_probs.set_yticklabels(labels, fontsize=15)
        self.ax_probs.set_xlim(0, 1.18)
        self.ax_probs.tick_params(axis='x', labelsize=10)
        self.ax_probs.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        self.ax_probs.set_xticklabels(['0', '25%', '50%', '75%', '100%'], fontsize=14)
        self.ax_probs.spines[['top', 'right']].set_visible(False)
        self.ax_probs.set_title('Action Probs', fontsize=18, fontweight='bold')

        # ── Info panel ─────────────────────────────────────────────────────
        self.ax_info.clear()
        self.ax_info.axis('off')

        # Build rows: (label, value, value_color)
        rows = [
            ('Latency', f"{result['inference_time_ms']:.1f} ms", '#333333'),
            ('FPS',     f"{result['fps']:.0f}",                  '#333333'),
        ]
        if 'temporal_context' in result and result['temporal_context'] > 0:
            rows.append(('Vote ctx', f"{result['temporal_context']} frames", '#555555'))

        if self.folder_mode and isinstance(self.lidar_source, FolderDataSource):
            play_val   = 'ON' if self.auto_play else 'OFF'
            play_color = '#27ae60' if self.auto_play else '#888888'
            rows.append(('Playback', play_val, play_color))

        if 'correct' in result and result['correct'] is not None:
            voted_ok = result['correct']
            raw_ok   = result.get('raw_correct')
            if raw_ok is not None and raw_ok != voted_ok:
                # Voting changed the outcome — show both
                rows.append(('Voted',  '✓ correct' if voted_ok else '✗ wrong',
                              '#27ae60' if voted_ok else '#c0392b'))
                rows.append(('Raw',    '✓ correct' if raw_ok   else '✗ wrong',
                              '#27ae60' if raw_ok   else '#c0392b'))
            else:
                rows.append(('Match',  '✓ correct' if voted_ok else '✗ wrong',
                              '#27ae60' if voted_ok else '#c0392b'))

        # Render as a simple two-column table
        row_h = 0.13
        y0    = 0.95
        for i, (lbl, val, vcol) in enumerate(rows):
            y = y0 - i * row_h
            self.ax_info.text(0.05, y, lbl,
                              fontsize=16, ha='left', va='top', color='#777777',
                              transform=self.ax_info.transAxes)
            self.ax_info.text(0.95, y, val,
                              fontsize=16, ha='right', va='top', color=vcol,
                              fontweight='bold', transform=self.ax_info.transAxes)
            # Thin separator line
            if i < len(rows) - 1:
                self.ax_info.axhline(y - row_h * 0.85,
                                     xmin=0.02, xmax=0.98,
                                     color='#eeeeee', linewidth=0.6)

        # ── Trajectory panel ────────────────────────────────────────────────
        self.ax_traj.clear()
        self.ax_traj.set_title('Trajectory', fontsize=18, fontweight='bold')
        self.ax_traj.set_facecolor('white')
        self.ax_traj.grid(True, alpha=0.4, color='#dddddd')

        if self.traj_positions is not None and len(self.traj_positions) > 0:
            n_traj = len(self.traj_positions)
            f_idx = (frame_idx if frame_idx is not None else 0)
            f_idx = min(f_idx, n_traj - 1)

            xs = self.traj_positions[:, 0]
            ys = self.traj_positions[:, 1]

            # Full gray path
            self.ax_traj.plot(xs, ys, '-', color='#cccccc', linewidth=1.5, zorder=1)

            # Colored dots by GT action (if available)
            traj_actions = (self.engine.pose_loader.actions
                            if self.engine.pose_loader is not None else None)
            if traj_actions is not None:
                for i in range(n_traj):
                    a = int(traj_actions[i]) if i < len(traj_actions) else -1
                    c = ACTION_COLORS.get(a, '#aaaaaa') if a >= 0 else '#cccccc'
                    self.ax_traj.plot(xs[i], ys[i], 'o', color=c,
                                      markersize=3, zorder=2, alpha=0.7)
            else:
                # No labels — just draw dots along the path
                self.ax_traj.plot(xs, ys, 'o', color='#aaaaaa',
                                  markersize=2, zorder=2, alpha=0.5)

            # Current position — red triangle
            cx, cy = xs[f_idx], ys[f_idx]
            self.ax_traj.plot(cx, cy, '^', color='#e74c3c', markersize=10,
                              markeredgecolor='#333333', markeredgewidth=0.7, zorder=5)

            # Center view ±30 m around current position
            margin = 30.0
            self.ax_traj.set_xlim(cx - margin, cx + margin)
            self.ax_traj.set_ylim(cy - margin, cy + margin)
            self.ax_traj.set_xlabel('X (m)', fontsize=14)
            self.ax_traj.set_ylabel('Y (m)', fontsize=14)
            self.ax_traj.tick_params(labelsize=9)
            self.ax_traj.set_aspect('equal')

            # Frame counter (top-left)
            self.ax_traj.text(0.04, 0.97, f'{f_idx + 1} / {n_traj}',
                              fontsize=14, ha='left', va='top',
                              color='#555555', transform=self.ax_traj.transAxes)
        else:
            self.ax_traj.text(0.5, 0.5, 'No trajectory\n(load poses.txt)',
                              ha='center', va='center', fontsize=16,
                              color='#aaaaaa', transform=self.ax_traj.transAxes)
            self.ax_traj.set_xlim(-1, 1)
            self.ax_traj.set_ylim(-1, 1)
            self.ax_traj.set_aspect('equal')

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def run(self, hz: float = 10):
        """Run visualization loop."""
        self.running = True

        if self.folder_mode:
            print("Starting folder mode visualization...")
            print("Use buttons to navigate: Load Folder, < Prev, Play/Pause, Next >")
            print("Press Ctrl+C to stop")
            self._update_file_info()
        else:
            print(f"Starting visualization at {hz} Hz...")
            print("Press Ctrl+C to stop")

        frame_interval = 1.0 / hz
        last_update_time = 0

        try:
            while self.running:
                current_time = time.time()

                if self.folder_mode:
                    # Folder mode: update on demand or auto-play
                    if self.needs_update:
                        self._update_frame(None)
                        self.needs_update = False
                    elif self.auto_play and (current_time - last_update_time) >= frame_interval:
                        if isinstance(self.lidar_source, FolderDataSource):
                            self.lidar_source.next()
                            self._update_file_info()
                        self._update_frame(None)
                        last_update_time = current_time
                else:
                    # Real-time mode: continuous updates
                    self._update_frame(None)

                plt.pause(0.01)
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            self.running = False


def main():
    parser = argparse.ArgumentParser(description='TerrainFormer Real-Time Inference')
    parser.add_argument('--mode', type=str, default='folder',
                        choices=['ros', 'udp', 'sim', 'folder'],
                        help='LiDAR connection mode')
    parser.add_argument('--topic', type=str, default='/velodyne_points',
                        help='ROS topic for point cloud (ros mode)')
    parser.add_argument('--port', type=int, default=2368,
                        help='UDP port for Velodyne (udp mode)')
    parser.add_argument('--data-dir', type=str,
                        default='/home/rickslab3/Documents/Datasets/RELLIS/dataset/sequences/00003',
                        help='Path to point cloud folder or RELLIS-3D sequence dir (folder mode)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint')
    parser.add_argument('--world-model-checkpoint', type=str, default=None,
                        help='Path to pretrained world model checkpoint (for traversability). '
                             'Auto-detected from outputs/world_model_pretrain/best_model.pt if not specified.')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device for inference')
    parser.add_argument('--encoder', type=str, default='pointpillars',
                        choices=['pointpillars', 'pointnet2'],
                        help='LiDAR encoder type')
    parser.add_argument('--hz', type=float, default=10,
                        help='Target inference rate')
    parser.add_argument('--goal-x', type=float, default=1.0,
                        help='Goal direction X')
    parser.add_argument('--goal-y', type=float, default=0.0,
                        help='Goal direction Y')
    parser.add_argument('--temporal', action='store_true',
                        help='Use temporal voting (aggregate predictions from past frames)')
    parser.add_argument('--no-temporal', dest='temporal', action='store_false',
                        help='Disable temporal voting (single frame only)')
    parser.add_argument('--temporal-mode', type=str, default='voting',
                        choices=['voting', 'none'],
                        help='Temporal mode: voting (aggregate past predictions)')
    parser.add_argument('--vote-window', type=int, default=10,
                        help='Number of past frames to use for voting (default: 10)')
    parser.add_argument('--pose-file', type=str,
                        default='/home/rickslab3/Documents/Datasets/RELLIS/dataset/sequences/00003/poses.txt',
                        help='Path to pose file or action labels for ground truth comparison')
    parser.add_argument('--num-past', type=int, default=10,
                        help='Number of past frames for temporal context')
    parser.add_argument('--num-future', type=int, default=10,
                        help='Number of future frames to predict (unused in voting mode)')
    # Decision publishing for external UI
    parser.add_argument('--publish', action='store_true',
                        help='Publish decisions via UDP for external UI')
    parser.add_argument('--publish-host', type=str, default='127.0.0.1',
                        help='UDP host for decision publishing')
    parser.add_argument('--publish-port', type=int, default=9999,
                        help='UDP port for decision publishing')
    # Goal direction mode
    parser.add_argument('--goal-mode', type=str, default='auto',
                        choices=['manual', 'auto', 'velocity'],
                        help='Goal direction mode: manual (use --goal-x/y), auto (from dataset), velocity (from state)')
    parser.set_defaults(temporal=True)
    args = parser.parse_args()

    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'

    # Create LiDAR source
    folder_mode = False
    if args.mode == 'ros':
        lidar = ROSReceiver(topic=args.topic)
    elif args.mode == 'udp':
        lidar = VelodyneUDPReceiver(port=args.port)
    elif args.mode == 'folder':
        lidar = FolderDataSource(folder_path=args.data_dir)
        folder_mode = True
    else:
        lidar = SimulatedLiDAR(hz=args.hz)

    # Create inference engine
    temporal_mode = args.temporal_mode if args.temporal else 'none'
    engine = TerrainFormerInference(
        checkpoint_path=args.checkpoint,
        world_model_checkpoint=args.world_model_checkpoint,
        device=args.device,
        encoder_type=args.encoder,
        use_temporal=args.temporal,
        temporal_mode=temporal_mode,
        num_past_frames=args.vote_window,  # Use vote_window for prediction buffer size
        num_future_frames=args.num_future,
        publish_decisions=args.publish,
        publish_host=args.publish_host,
        publish_port=args.publish_port,
    )

    # Load action labels if available (auto-detected or specified)
    if args.pose_file:
        # User specified a file - could be pose or action labels
        if args.pose_file.endswith('.npy'):
            engine.load_actions(args.pose_file)
        else:
            engine.load_poses(args.pose_file)
    elif folder_mode and isinstance(lidar, FolderDataSource):
        if lidar.actions_file:
            engine.load_actions(lidar.actions_file)
        elif lidar.pose_file:
            engine.load_poses(lidar.pose_file)

    # Create visualizer with goal mode
    viz = RealTimeVisualizer(engine, folder_mode=folder_mode, goal_mode=args.goal_mode)
    viz.set_lidar_source(lidar)

    # Load trajectory positions (for the trajectory panel)
    # From explicit --pose-file (if it looks like a poses.txt)
    if args.pose_file and not args.pose_file.endswith('.npy'):
        viz.load_trajectory(args.pose_file)
    # Auto-detect from folder's poses.txt
    elif folder_mode and isinstance(lidar, FolderDataSource) and lidar.pose_file:
        viz.load_trajectory(lidar.pose_file)

    # Set manual goal if in manual mode
    if args.goal_mode == 'manual':
        viz.set_goal(NavigationGoal(dx=args.goal_x, dy=args.goal_y))
    elif args.goal_mode == 'auto' and folder_mode and isinstance(lidar, FolderDataSource):
        # Load goals from dataset
        viz.goal_provider.load_from_sequence(args.data_dir)
        print(f"Goal mode: auto (using goals from dataset)")
    else:
        print(f"Goal mode: {args.goal_mode}")

    # Start LiDAR
    lidar.start()

    try:
        # Run visualization
        viz.run(hz=args.hz)
    finally:
        lidar.stop()
        plt.close('all')


if __name__ == '__main__':
    main()
