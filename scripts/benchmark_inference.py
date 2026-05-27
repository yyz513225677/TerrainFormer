#!/usr/bin/env python3
"""
Benchmark inference latency for TerrainFormer pipeline.

Measures:
1. LiDAR encoding time (PointNet++ + BEV projection)
2. World model prediction time
3. Decision transformer time
4. Total end-to-end latency
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import torch
import time
import numpy as np
from collections import defaultdict

from models.unified.terrainformer import TerrainFormer


def benchmark_inference(num_iterations=100, warmup=10, device='cuda', encoder_type='pointpillars'):
    """Benchmark inference latency."""

    print("=" * 70)
    print("TERRAINFORMER INFERENCE BENCHMARK")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Encoder: {encoder_type}")
    print(f"Warmup iterations: {warmup}")
    print(f"Benchmark iterations: {num_iterations}")
    print()

    # Create unified model
    print("Loading model...")

    model = TerrainFormer(
        lidar_in_channels=4,
        bev_size=256,
        num_actions=18,
        encoder_type=encoder_type,
    ).to(device)

    model.eval()

    # Model sizes
    encoder_params = sum(p.numel() for p in model.lidar_encoder.parameters()) / 1e6
    encoder_params += sum(p.numel() for p in model.bev_projection.parameters()) / 1e6
    world_params = sum(p.numel() for p in model.world_model.parameters()) / 1e6
    decision_params = sum(p.numel() for p in model.decision_transformer.parameters()) / 1e6
    total_params = sum(p.numel() for p in model.parameters()) / 1e6

    print(f"\nModel parameters:")
    print(f"  LiDAR Encoder:   {encoder_params:.2f}M")
    print(f"  World Model:     {world_params:.2f}M")
    print(f"  Decision Model:  {decision_params:.2f}M")
    print(f"  Total:           {total_params:.2f}M")

    # Create dummy inputs (simulating real data shapes)
    batch_size = 1  # Single sample for real-time inference

    # Point cloud: (B, N, 4) - x, y, z, intensity
    points = torch.randn(batch_size, 65536, 4, device=device)
    points[:, :, :3] *= 30  # Scale xyz to realistic range

    # Vehicle state: (B, 6)
    state = torch.randn(batch_size, 6, device=device)

    # Goal direction: (B, 2)
    goal = torch.randn(batch_size, 2, device=device)

    # Action history: (B, 10)
    action_history = torch.randint(0, 18, (batch_size, 10), device=device)

    print(f"\nInput shapes:")
    print(f"  Point cloud:     {tuple(points.shape)} ({points.numel() * 4 / 1024:.1f} KB)")
    print(f"  Vehicle state:   {tuple(state.shape)}")
    print(f"  Goal direction:  {tuple(goal.shape)}")
    print(f"  Action history:  {tuple(action_history.shape)}")

    # Timing storage
    times = defaultdict(list)

    print(f"\nRunning benchmark...")

    with torch.no_grad():
        # Warmup
        for _ in range(warmup):
            _ = model(points, state, goal, action_history)
            if device == 'cuda':
                torch.cuda.synchronize()

        # Benchmark with detailed timing
        for i in range(num_iterations):
            if device == 'cuda':
                torch.cuda.synchronize()

            # === Stage 1: LiDAR Encoding ===
            t0 = time.perf_counter()

            bev, lidar_global = model.encode_lidar(points)

            if device == 'cuda':
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            # === Stage 2: World Model ===
            world_outputs = model.world_model(bev, return_latent=True)

            if device == 'cuda':
                torch.cuda.synchronize()
            t2 = time.perf_counter()

            # === Stage 3: Decision Transformer ===
            world_global = world_outputs['global_feature']
            world_latent = world_outputs['latent']

            decision_outputs = model.decision_transformer(
                world_global, world_latent, state, goal, action_history
            )

            if device == 'cuda':
                torch.cuda.synchronize()
            t3 = time.perf_counter()

            # Record times
            times['lidar_encode'].append((t1 - t0) * 1000)  # ms
            times['world_model'].append((t2 - t1) * 1000)  # ms
            times['decision'].append((t3 - t2) * 1000)  # ms
            times['total'].append((t3 - t0) * 1000)  # ms

    # Compute statistics
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print()
    print("Latency (milliseconds):")
    print("-" * 70)
    print(f"{'Stage':<30} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print("-" * 70)

    encoder_name = 'PointPillars' if encoder_type == 'pointpillars' else 'PointNet++'
    labels = {
        'lidar_encode': f'1. LiDAR Encode ({encoder_name})',
        'world_model': '2. World Model (predict)',
        'decision': '3. Decision Transformer',
        'total': 'TOTAL END-TO-END',
    }

    for stage in ['lidar_encode', 'world_model', 'decision', 'total']:
        t = np.array(times[stage])
        label = labels[stage]
        print(f"{label:<30} {t.mean():>10.2f} {t.std():>10.2f} {t.min():>10.2f} {t.max():>10.2f}")

    print("-" * 70)

    # Throughput
    total_mean = np.mean(times['total'])
    fps = 1000.0 / total_mean

    print()
    print(f"Throughput: {fps:.1f} FPS")
    print(f"Latency:    {total_mean:.2f} ms per frame")
    print()

    # Real-time feasibility
    lidar_hz = 10  # LiDAR typically runs at 10 Hz
    frame_budget = 1000.0 / lidar_hz  # 100 ms

    print("Real-time Feasibility (10 Hz LiDAR):")
    print("-" * 70)
    print(f"  Frame budget:     {frame_budget:.1f} ms")
    print(f"  Inference time:   {total_mean:.2f} ms")
    print(f"  Margin:           {frame_budget - total_mean:.2f} ms")

    if total_mean < frame_budget:
        print(f"  Status:           REAL-TIME CAPABLE")
    else:
        print(f"  Status:           TOO SLOW (need {total_mean/frame_budget:.1f}x speedup)")

    print()
    print("=" * 70)

    # Breakdown
    lidar_pct = np.mean(times['lidar_encode']) / total_mean * 100
    world_pct = np.mean(times['world_model']) / total_mean * 100
    decision_pct = np.mean(times['decision']) / total_mean * 100

    print()
    print("Time Breakdown:")
    print(f"  LiDAR Encode:    {lidar_pct:5.1f}%  ({np.mean(times['lidar_encode']):.2f} ms)")
    print(f"  World Model:     {world_pct:5.1f}%  ({np.mean(times['world_model']):.2f} ms)")
    print(f"  Decision:        {decision_pct:5.1f}%  ({np.mean(times['decision']):.2f} ms)")
    print()

    # Pipeline visualization
    print("Pipeline Flow:")
    print("  Point Cloud (65536 x 4)")
    print("       |")
    print(f"       v  [{np.mean(times['lidar_encode']):.1f} ms]")
    print(f"  LiDAR Encoder ({encoder_name})")
    print("       |")
    print("  BEV Features (64 x 256 x 256)")
    print("       |")
    print(f"       v  [{np.mean(times['world_model']):.1f} ms]")
    print("  World Model (Transformer)")
    print("       |")
    print("  Latent + Future Predictions")
    print("       |")
    print(f"       v  [{np.mean(times['decision']):.1f} ms]")
    print("  Decision Transformer")
    print("       |")
    print("  Action (18 classes)")
    print()
    print(f"  Total: {total_mean:.1f} ms ({fps:.1f} FPS)")
    print()

    return times


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--encoder', type=str, default='pointpillars',
                        choices=['pointpillars', 'pointnet2'],
                        help='LiDAR encoder type (pointpillars=fast, pointnet2=accurate)')
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    benchmark_inference(
        num_iterations=args.iterations,
        warmup=args.warmup,
        device=device,
        encoder_type=args.encoder,
    )
