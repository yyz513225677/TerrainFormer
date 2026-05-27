"""
Generate action labels from RELLIS-3D pose files.

Converts continuous poses into discrete action labels for navigation.

Coordinate System Notes:
- RELLIS-3D uses a right-handed coordinate system where:
  - X points forward (vehicle heading direction)
  - Y points left
  - Z points up
- The front_direction_offset allows calibration for different datasets
- RELLIS-3D contains only forward motion (no backward driving)
"""
import numpy as np
import os
from pathlib import Path
import json
from typing import List, Tuple, Optional
import argparse

# Simplified action space (12 actions - forward motion only)
# RELLIS-3D only contains forward driving, no backward motion
ACTIONS = {
    0: 'STOP',
    1: 'FORWARD_SLOW',
    2: 'FORWARD_MEDIUM',
    3: 'FORWARD_FAST',
    4: 'TURN_LEFT_SHARP',
    5: 'TURN_LEFT_MEDIUM',
    6: 'TURN_LEFT_SLIGHT',
    7: 'TURN_RIGHT_SLIGHT',
    8: 'TURN_RIGHT_MEDIUM',
    9: 'TURN_RIGHT_SHARP',
    10: 'FORWARD_LEFT',
    11: 'FORWARD_RIGHT',
}

# Dataset-specific configurations
DATASET_CONFIGS = {
    'rellis3d': {
        'front_direction_offset': 0.0,  # X-forward coordinate system
        'dt': 0.1,  # 10 Hz LiDAR
        'velocity_thresholds': {
            'stop': 0.1,
            'slow': 0.5,
            'medium': 1.0,
        },
        'turn_rate_thresholds': {
            'straight': 0.1,
            'slight': 0.3,
            'medium': 0.6,
        },
    },
    'custom': {
        'front_direction_offset': 0.0,
        'dt': 0.1,
        'velocity_thresholds': {
            'stop': 0.1,
            'slow': 0.5,
            'medium': 1.0,
        },
        'turn_rate_thresholds': {
            'straight': 0.1,
            'slight': 0.3,
            'medium': 0.6,
        },
    },
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


def pose_to_position_yaw(pose: np.ndarray, front_offset: float = 0.0) -> Tuple[np.ndarray, float]:
    """
    Extract position (x, y, z) and yaw angle from pose matrix.

    Args:
        pose: 3x4 transformation matrix [R|t]
        front_offset: Rotation offset (radians) to align front direction.
                     Use this to calibrate different coordinate conventions.
                     Positive = rotate CCW (front shifts left)

    Returns:
        position: (x, y, z) translation vector
        yaw: Heading angle in radians (with front_offset applied)
    """
    position = pose[:, 3]  # Translation vector

    # Extract yaw from rotation matrix (assuming vehicle moves in XY plane)
    # For RELLIS-3D: X is forward, Y is left, Z is up
    # yaw = atan2(r21, r11) gives angle of X-axis in world frame
    yaw = np.arctan2(pose[1, 0], pose[0, 0])

    # Apply front direction offset for coordinate system calibration
    yaw = yaw + front_offset

    return position, yaw


def compute_motion_direction(pos1: np.ndarray, pos2: np.ndarray,
                            yaw: float) -> Tuple[float, float]:
    """
    Compute motion direction relative to vehicle heading.

    Args:
        pos1: Current position (x, y, z)
        pos2: Next position (x, y, z)
        yaw: Vehicle heading angle

    Returns:
        forward_component: Movement in forward direction (positive = forward)
        lateral_component: Movement in lateral direction (positive = left)
    """
    # Compute displacement in world frame
    displacement = pos2[:2] - pos1[:2]

    # Get forward and left unit vectors based on yaw
    forward_dir = np.array([np.cos(yaw), np.sin(yaw)])
    left_dir = np.array([-np.sin(yaw), np.cos(yaw)])

    # Project displacement onto vehicle axes
    forward_component = np.dot(displacement, forward_dir)
    lateral_component = np.dot(displacement, left_dir)

    return forward_component, lateral_component


def compute_goal_directions(poses: np.ndarray, lookahead: int = 5,
                            front_offset: float = 0.0) -> np.ndarray:
    """
    Compute goal directions from consecutive poses.

    For each frame, the goal direction is the normalized displacement vector
    to a future frame (lookahead frames ahead), expressed in the vehicle's
    local coordinate frame.

    Args:
        poses: (N, 3, 4) array of poses
        lookahead: Number of frames to look ahead for goal direction
        front_offset: Front direction offset for coordinate calibration

    Returns:
        goals: (N, 2) array of normalized goal directions (dx, dy) in local frame
               dx > 0: goal is ahead
               dy > 0: goal is to the left
               dy < 0: goal is to the right
    """
    N = len(poses)
    goals = np.zeros((N, 2))

    for i in range(N):
        # Look ahead to future frame
        future_idx = min(i + lookahead, N - 1)

        # Get current and future positions
        pos_curr, yaw_curr = pose_to_position_yaw(poses[i], front_offset)
        pos_future, _ = pose_to_position_yaw(poses[future_idx], front_offset)

        # Compute displacement in world frame
        displacement = pos_future[:2] - pos_curr[:2]

        # If stationary (same frame or no movement), default to forward
        if np.linalg.norm(displacement) < 0.01:
            goals[i] = [1.0, 0.0]
            continue

        # Transform to local (vehicle) frame
        # Rotate by -yaw to get local coordinates
        cos_yaw = np.cos(-yaw_curr)
        sin_yaw = np.sin(-yaw_curr)
        local_x = displacement[0] * cos_yaw - displacement[1] * sin_yaw
        local_y = displacement[0] * sin_yaw + displacement[1] * cos_yaw

        # Normalize
        norm = np.sqrt(local_x**2 + local_y**2)
        goals[i] = [local_x / norm, local_y / norm]

    return goals


def compute_velocity_and_turn_rate(poses: np.ndarray, dt: float = 0.1,
                                   front_offset: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute linear velocity and turn rate from consecutive poses.

    Args:
        poses: (N, 3, 4) array of poses
        dt: time step between poses in seconds
        front_offset: Front direction offset for coordinate calibration

    Returns:
        velocities: (N-1,) array of forward velocities (m/s)
                   Positive = moving forward, negative = moving backward
        turn_rates: (N-1,) array of turn rates (rad/s)
                   Positive = turning left, negative = turning right
    """
    N = len(poses)
    velocities = np.zeros(N - 1)
    turn_rates = np.zeros(N - 1)

    for i in range(N - 1):
        pos1, yaw1 = pose_to_position_yaw(poses[i], front_offset)
        pos2, yaw2 = pose_to_position_yaw(poses[i + 1], front_offset)

        # Compute motion components relative to vehicle heading
        forward_comp, lateral_comp = compute_motion_direction(pos1, pos2, yaw1)

        # Forward velocity: positive = forward, negative = backward
        velocities[i] = forward_comp / dt

        # Compute turn rate (handle angle wrapping)
        dyaw = yaw2 - yaw1
        dyaw = np.arctan2(np.sin(dyaw), np.cos(dyaw))  # Wrap to [-pi, pi]
        turn_rates[i] = dyaw / dt

    return velocities, turn_rates


def discretize_actions(velocities: np.ndarray, turn_rates: np.ndarray,
                       config: Optional[dict] = None) -> np.ndarray:
    """
    Convert continuous velocity and turn rate to discrete actions.

    Logic (turn-first approach for medium/sharp turns):
        Step 1: Check turn rate magnitude first
        Step 2: Then check velocity for forward actions

    12-Action Space (forward motion only, indices 0-11):
        Speed-based (moving with minimal turn |omega| < 0.1):
            0: STOP              - Not moving (v < 0.1 m/s)
            1: FORWARD_SLOW      - Slow speed (0.1 <= v < 0.5), minimal turn
            2: FORWARD_MEDIUM    - Medium speed (0.5 <= v < 1.0), minimal turn
            3: FORWARD_FAST      - High speed (v >= 1.0), minimal turn

        Turn-based (any velocity, classified by turn rate):
            4: TURN_LEFT_SHARP   - Sharp left turn (omega >= 0.6)
            5: TURN_LEFT_MEDIUM  - Medium left turn (0.3 <= omega < 0.6)
            6: TURN_LEFT_SLIGHT  - Slight left turn (0.1 <= omega < 0.3)
            7: TURN_RIGHT_SLIGHT - Slight right turn
            8: TURN_RIGHT_MEDIUM - Medium right turn
            9: TURN_RIGHT_SHARP  - Sharp right turn

        Combined (moving fast with sharp turn |omega| >= 0.6):
            10: FORWARD_LEFT     - Fast forward with sharp left turn
            11: FORWARD_RIGHT    - Fast forward with sharp right turn

    Args:
        velocities: Forward velocities (m/s), positive = forward
        turn_rates: Turn rates (rad/s), positive = left
        config: Optional thresholds configuration

    Returns:
        actions: Discrete action indices (0-11)
    """
    if config is None:
        config = DATASET_CONFIGS['rellis3d']

    vel_thresh = config['velocity_thresholds']
    turn_thresh = config['turn_rate_thresholds']

    N = len(velocities)
    actions = np.zeros(N, dtype=np.int64)

    for i in range(N):
        v = velocities[i]
        omega = turn_rates[i]

        # For RELLIS-3D: all motion is forward, so treat any backward as forward
        # This handles coordinate system artifacts
        v = abs(v)

        abs_omega = abs(omega)
        is_moving = v >= vel_thresh['stop']

        # STEP 1: Check turn rate first (turn-first approach)
        if abs_omega >= turn_thresh['medium']:
            # Sharp turn (>= 0.6 rad/s)
            if is_moving and v >= vel_thresh['medium']:
                # Fast forward with sharp turn
                actions[i] = 10 if omega > 0 else 11  # FORWARD_LEFT/RIGHT
            else:
                # Sharp turn (stopped or slow)
                actions[i] = 4 if omega > 0 else 9  # TURN_LEFT/RIGHT_SHARP
        elif abs_omega >= turn_thresh['slight']:
            # Medium turn (0.3-0.6 rad/s)
            actions[i] = 5 if omega > 0 else 8  # TURN_LEFT/RIGHT_MEDIUM
        elif abs_omega >= turn_thresh['straight']:
            # Slight turn (0.1-0.3 rad/s)
            actions[i] = 6 if omega > 0 else 7  # TURN_LEFT/RIGHT_SLIGHT
        else:
            # Minimal turn (< 0.1 rad/s) - classify by speed
            if not is_moving:
                actions[i] = 0  # STOP
            elif v >= vel_thresh['medium']:
                actions[i] = 3  # FORWARD_FAST
            elif v >= vel_thresh['slow']:
                actions[i] = 2  # FORWARD_MEDIUM
            else:
                actions[i] = 1  # FORWARD_SLOW

    return actions


def analyze_motion_statistics(velocities: np.ndarray, turn_rates: np.ndarray,
                              goals: np.ndarray) -> dict:
    """Analyze motion statistics for debugging coordinate systems."""
    stats = {
        'velocity': {
            'mean': float(np.mean(velocities)),
            'std': float(np.std(velocities)),
            'min': float(np.min(velocities)),
            'max': float(np.max(velocities)),
            'positive_ratio': float(np.mean(velocities > 0)),
            'negative_ratio': float(np.mean(velocities < 0)),
        },
        'turn_rate': {
            'mean': float(np.mean(turn_rates)),
            'std': float(np.std(turn_rates)),
            'min': float(np.min(turn_rates)),
            'max': float(np.max(turn_rates)),
            'left_ratio': float(np.mean(turn_rates > 0.1)),
            'right_ratio': float(np.mean(turn_rates < -0.1)),
        },
        'goal': {
            'mean_x': float(np.mean(goals[:, 0])),
            'mean_y': float(np.mean(goals[:, 1])),
            'forward_ratio': float(np.mean(goals[:, 0] > 0)),
            'left_ratio': float(np.mean(goals[:, 1] > 0.1)),
            'right_ratio': float(np.mean(goals[:, 1] < -0.1)),
        },
    }
    return stats


def generate_action_labels_for_sequence(sequence_path: str, calib_path: str, output_path: str,
                                        dataset_config: str = 'rellis3d',
                                        front_offset: Optional[float] = None):
    """
    Generate and save action labels for a single sequence.

    Args:
        sequence_path: Path to sequence data
        calib_path: Path to calibration/pose files
        output_path: Path to save outputs
        dataset_config: Configuration name ('rellis3d', 'custom')
        front_offset: Override front direction offset (radians)
    """
    poses_file = os.path.join(calib_path, 'poses.txt')

    if not os.path.exists(poses_file):
        print(f"Warning: No poses file found for {sequence_path}")
        return

    # Get configuration
    config = DATASET_CONFIGS.get(dataset_config, DATASET_CONFIGS['rellis3d']).copy()
    if front_offset is not None:
        config['front_direction_offset'] = front_offset

    # Load poses
    poses = load_poses(poses_file)
    print(f"Loaded {len(poses)} poses from {poses_file}")
    print(f"Using front_direction_offset: {config['front_direction_offset']:.3f} rad "
          f"({np.degrees(config['front_direction_offset']):.1f} deg)")

    # Compute velocities and turn rates
    velocities, turn_rates = compute_velocity_and_turn_rate(
        poses, dt=config['dt'], front_offset=config['front_direction_offset']
    )

    # Compute goal directions from poses (where the vehicle will move)
    goals = compute_goal_directions(
        poses, lookahead=5, front_offset=config['front_direction_offset']
    )

    # Analyze motion for debugging
    motion_stats = analyze_motion_statistics(velocities, turn_rates, goals)
    print(f"\nMotion statistics:")
    print(f"  Velocity: mean={motion_stats['velocity']['mean']:.2f} m/s, "
          f"forward={100*motion_stats['velocity']['positive_ratio']:.1f}%, "
          f"backward={100*motion_stats['velocity']['negative_ratio']:.1f}%")
    print(f"  Turn rate: mean={motion_stats['turn_rate']['mean']:.3f} rad/s, "
          f"left={100*motion_stats['turn_rate']['left_ratio']:.1f}%, "
          f"right={100*motion_stats['turn_rate']['right_ratio']:.1f}%")
    print(f"  Goal direction: mean_x={motion_stats['goal']['mean_x']:.2f}, "
          f"mean_y={motion_stats['goal']['mean_y']:.2f}")

    # Discretize to actions (pad last frame with same action)
    actions = discretize_actions(velocities, turn_rates, config)
    actions = np.concatenate([actions, [actions[-1]]])  # Same action for last frame

    # Statistics
    unique, counts = np.unique(actions, return_counts=True)
    print(f"\nAction distribution ({len(ACTIONS)} possible actions):")
    for action_id, count in zip(unique, counts):
        action_name = ACTIONS.get(action_id, f'UNKNOWN_{action_id}')
        print(f"  {action_id:2d} ({action_name:20s}): {count:4d} ({100*count/len(actions):.1f}%)")

    # Goal direction statistics
    left_goals = np.sum(goals[:, 1] > 0.1)
    right_goals = np.sum(goals[:, 1] < -0.1)
    straight_goals = np.sum(np.abs(goals[:, 1]) <= 0.1)
    print(f"\nGoal direction distribution:")
    print(f"  Left turns (y > 0.1):   {left_goals:4d} ({100*left_goals/len(goals):.1f}%)")
    print(f"  Straight (|y| <= 0.1):  {straight_goals:4d} ({100*straight_goals/len(goals):.1f}%)")
    print(f"  Right turns (y < -0.1): {right_goals:4d} ({100*right_goals/len(goals):.1f}%)")

    # Save as numpy files
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, 'actions.npy')
    np.save(output_file, actions)
    print(f"\nSaved actions to {output_file}")

    goals_file = os.path.join(output_path, 'goals.npy')
    np.save(goals_file, goals)
    print(f"Saved goals to {goals_file}")

    # Also save metadata
    metadata = {
        'num_frames': len(actions),
        'num_actions': len(ACTIONS),
        'action_counts': {int(k): int(v) for k, v in zip(unique, counts)},
        'action_names': ACTIONS,
        'config': {
            'dataset': dataset_config,
            'front_direction_offset': config['front_direction_offset'],
            'dt': config['dt'],
            'velocity_thresholds': config['velocity_thresholds'],
            'turn_rate_thresholds': config['turn_rate_thresholds'],
        },
        'motion_statistics': motion_stats,
    }
    metadata_file = os.path.join(output_path, 'actions_metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {metadata_file}")

    return actions


def calibrate_front_direction(poses: np.ndarray, test_offsets: List[float] = None) -> float:
    """
    Find the optimal front direction offset by checking motion statistics.

    For a forward-only dataset like RELLIS-3D, the correct offset should result in:
    - Most velocities being positive (forward motion)
    - Goal directions predominantly forward (positive x)

    Args:
        poses: Array of pose matrices
        test_offsets: List of offset angles to test (radians)

    Returns:
        Best front direction offset (radians)
    """
    if test_offsets is None:
        # Test every 15 degrees from -180 to 180
        test_offsets = [np.radians(a) for a in range(-180, 181, 15)]

    best_offset = 0.0
    best_forward_ratio = 0.0

    print("\nCalibrating front direction offset...")
    print("-" * 50)

    for offset in test_offsets:
        velocities, _ = compute_velocity_and_turn_rate(poses, dt=0.1, front_offset=offset)
        forward_ratio = np.mean(velocities > 0)

        if forward_ratio > best_forward_ratio:
            best_forward_ratio = forward_ratio
            best_offset = offset

    print(f"Best offset: {best_offset:.3f} rad ({np.degrees(best_offset):.1f} deg)")
    print(f"Forward motion ratio: {100*best_forward_ratio:.1f}%")
    print("-" * 50)

    return best_offset


def main():
    parser = argparse.ArgumentParser(
        description='Generate action labels from pose files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate labels for RELLIS-3D with default settings
  python generate_action_labels.py

  # Auto-calibrate front direction
  python generate_action_labels.py --calibrate

  # Use specific front direction offset (in degrees)
  python generate_action_labels.py --front-offset 90

  # Process specific sequences
  python generate_action_labels.py --sequences 00003 00004
        """
    )
    parser.add_argument('--root', type=str, default='../RELLIS',
                        help='Root path to RELLIS-3D dataset')
    parser.add_argument('--sequences', nargs='+', default=None,
                        help='Specific sequences to process (default: all)')
    parser.add_argument('--front-offset', type=float, default=None,
                        help='Front direction offset in degrees')
    parser.add_argument('--calibrate', action='store_true',
                        help='Auto-calibrate front direction offset')
    parser.add_argument('--dataset', type=str, default='rellis3d',
                        choices=list(DATASET_CONFIGS.keys()),
                        help='Dataset configuration to use')

    args = parser.parse_args()

    rellis_root = Path(args.root)
    sequences = args.sequences or ['00000', '00001', '00002', '00003', '00004']

    # Convert front offset from degrees to radians
    front_offset = None
    if args.front_offset is not None:
        front_offset = np.radians(args.front_offset)

    print("=" * 70)
    print("Generating Action Labels from RELLIS-3D Poses")
    print("=" * 70)
    print(f"Dataset root: {rellis_root}")
    print(f"Sequences: {sequences}")
    print(f"Dataset config: {args.dataset}")
    if front_offset is not None:
        print(f"Front offset: {np.degrees(front_offset):.1f} degrees")

    # Auto-calibrate if requested
    if args.calibrate and front_offset is None:
        # Load first sequence for calibration
        first_seq = sequences[0]
        calib_path = rellis_root / 'calib' / first_seq / 'poses.txt'
        if calib_path.exists():
            print(f"\nCalibrating using sequence {first_seq}...")
            poses = load_poses(str(calib_path))
            front_offset = calibrate_front_direction(poses)
        else:
            print(f"Warning: Cannot calibrate - no poses file at {calib_path}")

    for seq in sequences:
        print(f"\n{'=' * 70}")
        print(f"Processing sequence {seq}")
        print('=' * 70)

        sequence_path = rellis_root / 'bin' / seq
        calib_path = rellis_root / 'calib' / seq
        output_path = rellis_root / 'actions' / seq

        generate_action_labels_for_sequence(
            str(sequence_path),
            str(calib_path),
            str(output_path),
            dataset_config=args.dataset,
            front_offset=front_offset
        )

    print("\n" + "=" * 70)
    print("Action label generation complete!")
    print("=" * 70)
    print(f"\nSimplified action space ({len(ACTIONS)} actions):")
    for idx, name in ACTIONS.items():
        print(f"  {idx:2d}: {name}")


if __name__ == '__main__':
    main()
