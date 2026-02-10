#!/usr/bin/env python3
"""
TerrainFormer Inference Demo

Demonstrates real-time inference with the trained model.

Usage:
    python scripts/inference_demo.py --checkpoint path/to/model.pt
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import torch
import numpy as np
import time

from models import TerrainFormer
from models.unified import InferencePipeline, SafetyConfig
from models.decision.action_tokenizer import ActionVocabulary


def parse_args():
    parser = argparse.ArgumentParser(description='TerrainFormer Inference Demo')
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--num_steps', type=int, default=100, help='Number of inference steps')
    return parser.parse_args()


def generate_dummy_lidar():
    """Generate synthetic LiDAR point cloud for demo."""
    # Simulate ground plane
    n_ground = 10000
    x = np.random.uniform(-50, 50, n_ground)
    y = np.random.uniform(-50, 50, n_ground)
    z = np.random.normal(-1.5, 0.1, n_ground)
    intensity = np.random.uniform(0, 1, n_ground)
    ground = np.stack([x, y, z, intensity], axis=1)
    
    # Simulate some obstacles
    n_obstacles = 2000
    for _ in range(5):
        cx, cy = np.random.uniform(-30, 30, 2)
        ox = np.random.normal(cx, 1, n_obstacles // 5)
        oy = np.random.normal(cy, 1, n_obstacles // 5)
        oz = np.random.uniform(-1, 2, n_obstacles // 5)
        oi = np.random.uniform(0.5, 1, n_obstacles // 5)
        obstacle = np.stack([ox, oy, oz, oi], axis=1)
        ground = np.vstack([ground, obstacle])
    
    return ground.astype(np.float32)


def main():
    args = parse_args()
    
    print("=" * 60)
    print("TerrainFormer Inference Demo")
    print("=" * 60)
    
    # Create model
    print("\nLoading model...")
    model = TerrainFormer(
        lidar_in_channels=4,
        bev_size=256,
        num_actions=18,
    )
    
    # Load checkpoint if exists
    device = args.device if torch.cuda.is_available() else 'cpu'
    if Path(args.checkpoint).exists():
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from {args.checkpoint}")
    else:
        print(f"Checkpoint not found, using random weights for demo")
    
    # Create inference pipeline
    safety_config = SafetyConfig(
        min_confidence=0.7,
        emergency_stop_collision_prob=0.8,
    )
    
    pipeline = InferencePipeline(model, safety_config, device=device)
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print(f"Device: {device}")
    
    # Run inference
    print(f"\nRunning {args.num_steps} inference steps...")
    print("-" * 60)
    
    action_counts = {}
    total_time = 0
    
    for step in range(args.num_steps):
        # Generate synthetic data
        points = torch.from_numpy(generate_dummy_lidar())
        state = torch.randn(6)  # velocity, angular_vel, pitch, roll, x, y
        state[0] = np.random.uniform(2, 8)  # Forward velocity
        goal = torch.tensor([1.0, np.random.uniform(-0.3, 0.3)])  # Goal direction
        
        # Inference
        start_time = time.time()
        result = pipeline.step(points, state, goal)
        inference_time = time.time() - start_time
        total_time += inference_time
        
        # Track actions
        action = result['action']
        action_name = ActionVocabulary.action_to_name(action)
        action_counts[action_name] = action_counts.get(action_name, 0) + 1
        
        # Print every 10 steps
        if (step + 1) % 10 == 0:
            print(f"Step {step+1:3d}: action={action_name:12s}, "
                  f"confidence={result['confidence']:.3f}, "
                  f"time={result['inference_time_ms']:.1f}ms")
    
    # Statistics
    print("-" * 60)
    print("\nInference Statistics:")
    stats = pipeline.get_statistics()
    print(f"  Mean inference time: {stats['mean_inference_ms']:.2f}ms")
    print(f"  Std inference time:  {stats['std_inference_ms']:.2f}ms")
    print(f"  Max inference time:  {stats['max_inference_ms']:.2f}ms")
    print(f"  Total steps:         {stats['total_steps']}")
    
    print("\nAction Distribution:")
    for action, count in sorted(action_counts.items(), key=lambda x: -x[1]):
        pct = count / args.num_steps * 100
        print(f"  {action:15s}: {count:3d} ({pct:5.1f}%)")
    
    print("\n" + "=" * 60)
    print("Demo complete!")


if __name__ == '__main__':
    main()
