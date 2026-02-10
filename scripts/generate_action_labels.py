"""
Generate action labels from RELLIS-3D pose files.

Converts continuous poses into discrete action labels for navigation.
"""
import numpy as np
import os
from pathlib import Path
import json
from typing import List, Tuple

# Action space (18 actions to match model configuration)
ACTIONS = {
    0: 'STOP',
    1: 'FORWARD_SLOW',
    2: 'FORWARD_MEDIUM',
    3: 'FORWARD_FAST',
    4: 'BACKWARD_SLOW',
    5: 'BACKWARD_MEDIUM',
    6: 'BACKWARD_FAST',
    7: 'TURN_LEFT_SHARP',
    8: 'TURN_LEFT_MEDIUM',
    9: 'TURN_LEFT_SLIGHT',
    10: 'STRAIGHT',
    11: 'TURN_RIGHT_SLIGHT',
    12: 'TURN_RIGHT_MEDIUM',
    13: 'TURN_RIGHT_SHARP',
    14: 'FORWARD_LEFT',
    15: 'FORWARD_RIGHT',
    16: 'BACKWARD_LEFT',
    17: 'BACKWARD_RIGHT',
}


def load_poses(poses_file: str) -> np.ndarray:
    """Load poses from file. Each line is a 3x4 transformation matrix."""
    poses = []
    with open(poses_file, 'r') as f:
        for line in f:
            values = [float(x) for x in line.strip().split()]
            if len(values) == 12:
                # Reshape to 3x4 matrix [R|t]
                pose = np.array(values).reshape(3, 4)
                poses.append(pose)
    return np.array(poses)


def pose_to_position_yaw(pose: np.ndarray) -> Tuple[np.ndarray, float]:
    """Extract position (x, y, z) and yaw angle from pose matrix."""
    position = pose[:, 3]  # Translation vector

    # Extract yaw from rotation matrix (assuming vehicle moves in XY plane)
    # yaw = atan2(r21, r11)
    yaw = np.arctan2(pose[1, 0], pose[0, 0])

    return position, yaw


def compute_velocity_and_turn_rate(poses: np.ndarray, dt: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute linear velocity and turn rate from consecutive poses.

    Args:
        poses: (N, 3, 4) array of poses
        dt: time step between poses in seconds

    Returns:
        velocities: (N-1,) array of forward velocities (m/s)
        turn_rates: (N-1,) array of turn rates (rad/s)
    """
    N = len(poses)
    velocities = np.zeros(N - 1)
    turn_rates = np.zeros(N - 1)

    for i in range(N - 1):
        pos1, yaw1 = pose_to_position_yaw(poses[i])
        pos2, yaw2 = pose_to_position_yaw(poses[i + 1])

        # Compute displacement
        displacement = pos2 - pos1
        distance = np.linalg.norm(displacement[:2])  # XY distance

        # Compute forward velocity (signed based on direction)
        # Positive if moving forward, negative if backward
        forward_dir = np.array([np.cos(yaw1), np.sin(yaw1)])
        velocity_sign = np.sign(np.dot(displacement[:2], forward_dir))
        velocities[i] = velocity_sign * distance / dt

        # Compute turn rate (handle angle wrapping)
        dyaw = yaw2 - yaw1
        dyaw = np.arctan2(np.sin(dyaw), np.cos(dyaw))  # Wrap to [-pi, pi]
        turn_rates[i] = dyaw / dt

    return velocities, turn_rates


def discretize_actions(velocities: np.ndarray, turn_rates: np.ndarray) -> np.ndarray:
    """
    Convert continuous velocity and turn rate to discrete actions.

    Velocity thresholds (m/s):
        - Stop: |v| < 0.1
        - Slow: 0.1 <= |v| < 0.5
        - Medium: 0.5 <= |v| < 1.0
        - Fast: |v| >= 1.0

    Turn rate thresholds (rad/s):
        - Straight: |omega| < 0.1
        - Slight: 0.1 <= |omega| < 0.3
        - Medium: 0.3 <= |omega| < 0.6
        - Sharp: |omega| >= 0.6
    """
    N = len(velocities)
    actions = np.zeros(N, dtype=np.int64)

    for i in range(N):
        v = velocities[i]
        omega = turn_rates[i]

        # Classify velocity
        if abs(v) < 0.1:
            vel_class = 'STOP'
        elif v >= 1.0:
            vel_class = 'FORWARD_FAST'
        elif v >= 0.5:
            vel_class = 'FORWARD_MEDIUM'
        elif v >= 0.1:
            vel_class = 'FORWARD_SLOW'
        elif v <= -1.0:
            vel_class = 'BACKWARD_FAST'
        elif v <= -0.5:
            vel_class = 'BACKWARD_MEDIUM'
        else:
            vel_class = 'BACKWARD_SLOW'

        # Classify turn rate
        if abs(omega) < 0.1:
            turn_class = 'STRAIGHT'
        elif omega >= 0.6:
            turn_class = 'LEFT_SHARP'
        elif omega >= 0.3:
            turn_class = 'LEFT_MEDIUM'
        elif omega >= 0.1:
            turn_class = 'LEFT_SLIGHT'
        elif omega <= -0.6:
            turn_class = 'RIGHT_SHARP'
        elif omega <= -0.3:
            turn_class = 'RIGHT_MEDIUM'
        else:
            turn_class = 'RIGHT_SLIGHT'

        # Map to action
        if vel_class == 'STOP':
            actions[i] = 0  # STOP
        elif vel_class == 'FORWARD_FAST':
            if turn_class in ['STRAIGHT', 'LEFT_SLIGHT', 'RIGHT_SLIGHT']:
                actions[i] = 3  # FORWARD_FAST
            elif turn_class in ['LEFT_MEDIUM', 'LEFT_SHARP']:
                actions[i] = 14  # FORWARD_LEFT
            else:
                actions[i] = 15  # FORWARD_RIGHT
        elif vel_class == 'FORWARD_MEDIUM':
            if turn_class in ['STRAIGHT', 'LEFT_SLIGHT', 'RIGHT_SLIGHT']:
                actions[i] = 2  # FORWARD_MEDIUM
            elif turn_class in ['LEFT_MEDIUM', 'LEFT_SHARP']:
                actions[i] = 14  # FORWARD_LEFT
            else:
                actions[i] = 15  # FORWARD_RIGHT
        elif vel_class == 'FORWARD_SLOW':
            if turn_class in ['STRAIGHT', 'LEFT_SLIGHT', 'RIGHT_SLIGHT']:
                actions[i] = 1  # FORWARD_SLOW
            elif turn_class in ['LEFT_MEDIUM', 'LEFT_SHARP']:
                actions[i] = 14  # FORWARD_LEFT
            else:
                actions[i] = 15  # FORWARD_RIGHT
        elif vel_class.startswith('BACKWARD'):
            if turn_class in ['LEFT_MEDIUM', 'LEFT_SHARP']:
                actions[i] = 16  # BACKWARD_LEFT
            elif turn_class in ['RIGHT_MEDIUM', 'RIGHT_SHARP']:
                actions[i] = 17  # BACKWARD_RIGHT
            elif vel_class == 'BACKWARD_FAST':
                actions[i] = 6
            elif vel_class == 'BACKWARD_MEDIUM':
                actions[i] = 5
            else:
                actions[i] = 4
        else:
            # Pure turning (no forward/backward motion)
            if turn_class == 'LEFT_SHARP':
                actions[i] = 7
            elif turn_class == 'LEFT_MEDIUM':
                actions[i] = 8
            elif turn_class == 'LEFT_SLIGHT':
                actions[i] = 9
            elif turn_class == 'STRAIGHT':
                actions[i] = 10
            elif turn_class == 'RIGHT_SLIGHT':
                actions[i] = 11
            elif turn_class == 'RIGHT_MEDIUM':
                actions[i] = 12
            else:
                actions[i] = 13

    return actions


def generate_action_labels_for_sequence(sequence_path: str, calib_path: str, output_path: str):
    """Generate and save action labels for a single sequence."""
    poses_file = os.path.join(calib_path, 'poses.txt')

    if not os.path.exists(poses_file):
        print(f"Warning: No poses file found for {sequence_path}")
        return

    # Load poses
    poses = load_poses(poses_file)
    print(f"Loaded {len(poses)} poses from {poses_file}")

    # Compute velocities and turn rates
    velocities, turn_rates = compute_velocity_and_turn_rate(poses, dt=0.1)

    # Discretize to actions (pad last frame with same action)
    actions = discretize_actions(velocities, turn_rates)
    actions = np.concatenate([actions, [actions[-1]]])  # Same action for last frame

    # Statistics
    unique, counts = np.unique(actions, return_counts=True)
    print(f"\nAction distribution:")
    for action_id, count in zip(unique, counts):
        print(f"  {action_id:2d} ({ACTIONS[action_id]:20s}): {count:4d} ({100*count/len(actions):.1f}%)")

    # Save as numpy file
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, 'actions.npy')
    np.save(output_file, actions)
    print(f"\nSaved actions to {output_file}")

    # Also save metadata
    metadata = {
        'num_frames': len(actions),
        'action_counts': {int(k): int(v) for k, v in zip(unique, counts)},
        'action_names': ACTIONS,
    }
    metadata_file = os.path.join(output_path, 'actions_metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    return actions


def main():
    rellis_root = Path('../RELLIS')
    sequences = ['00000', '00001', '00002', '00003', '00004']

    print("="*70)
    print("Generating Action Labels from RELLIS-3D Poses")
    print("="*70)

    for seq in sequences:
        print(f"\n{'='*70}")
        print(f"Processing sequence {seq}")
        print('='*70)

        sequence_path = rellis_root / 'bin' / seq
        calib_path = rellis_root / 'calib' / seq
        output_path = rellis_root / 'actions' / seq

        generate_action_labels_for_sequence(
            str(sequence_path),
            str(calib_path),
            str(output_path)
        )

    print("\n" + "="*70)
    print("Action label generation complete!")
    print("="*70)


if __name__ == '__main__':
    main()
