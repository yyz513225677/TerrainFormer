#!/usr/bin/env python3
"""
TerrainFormer Pipeline Visualization

Visualizes:
1. Input point cloud (BEV and 3D views)
2. World model predictions (traversability, elevation, semantics, future)
3. Decision output (action distribution and selected action)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches

from models.unified.terrainformer import TerrainFormer


# Action labels for visualization
ACTION_LABELS = {
    0: "Forward",
    1: "Forward-Left",
    2: "Forward-Right",
    3: "Left",
    4: "Right",
    5: "Slow Forward",
    6: "Slow Left",
    7: "Slow Right",
    8: "Backward",
    9: "Backward-Left",
    10: "Backward-Right",
    11: "Stop",
    12: "Sharp Left",
    13: "Sharp Right",
    14: "Accelerate",
    15: "Decelerate",
    16: "Maintain",
    17: "Emergency Stop",
}

# Semantic class colors
SEMANTIC_COLORS = [
    [0, 0, 0],        # 0: void
    [0, 128, 0],      # 1: grass
    [139, 69, 19],    # 2: dirt
    [128, 128, 128],  # 3: gravel
    [64, 64, 64],     # 4: asphalt
    [0, 0, 255],      # 5: water
    [0, 255, 0],      # 6: vegetation
    [255, 255, 0],    # 7: obstacle
    [255, 0, 0],      # 8: danger
    [128, 0, 128],    # 9: building
]


def create_sample_data(device='cuda'):
    """Create sample data for visualization."""
    B = 1
    N = 65536

    # Generate realistic point cloud
    # Ground plane with terrain variations
    x = np.random.uniform(-40, 40, N)
    y = np.random.uniform(-40, 40, N)

    # Add terrain height variation
    z = 0.1 * np.sin(x * 0.1) * np.cos(y * 0.1) + np.random.normal(0, 0.1, N)

    # Add some obstacles (trees, rocks)
    num_obstacles = 20
    for _ in range(num_obstacles):
        ox, oy = np.random.uniform(-30, 30, 2)
        mask = ((x - ox)**2 + (y - oy)**2) < 4
        z[mask] += np.random.uniform(0.5, 2.0)

    # Intensity based on height
    intensity = (z - z.min()) / (z.max() - z.min() + 1e-6)

    points = torch.tensor(
        np.stack([x, y, z, intensity], axis=1),
        dtype=torch.float32, device=device
    ).unsqueeze(0)

    # Vehicle state: [vx, vy, vz, roll, pitch, yaw]
    state = torch.tensor([[2.0, 0.0, 0.0, 0.0, 0.0, 0.3]], device=device)

    # Goal direction: [dx, dy] normalized
    goal = torch.tensor([[0.8, 0.2]], device=device)
    goal = goal / goal.norm(dim=-1, keepdim=True)

    # Action history
    action_history = torch.tensor([[5, 5, 5, 0, 0, 0, 5, 5, 0, 0]], device=device)

    return points, state, goal, action_history


def visualize_point_cloud_bev(ax, points, title="Point Cloud (BEV)"):
    """Visualize point cloud from bird's eye view."""
    pts = points[0].cpu().numpy()
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]

    # Color by height
    colors = plt.cm.terrain((z - z.min()) / (z.max() - z.min() + 1e-6))

    ax.scatter(x, y, c=colors, s=0.1, alpha=0.5)
    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(title)
    ax.set_aspect('equal')

    # Add vehicle marker
    ax.plot(0, 0, 'r^', markersize=10, label='Vehicle')
    ax.legend(loc='upper right')


def visualize_point_cloud_3d(ax, points, title="Point Cloud (3D)"):
    """Visualize point cloud in 3D."""
    pts = points[0].cpu().numpy()

    # Downsample for 3D visualization
    idx = np.random.choice(len(pts), min(10000, len(pts)), replace=False)
    x, y, z = pts[idx, 0], pts[idx, 1], pts[idx, 2]

    colors = plt.cm.terrain((z - z.min()) / (z.max() - z.min() + 1e-6))

    ax.scatter(x, y, z, c=colors, s=0.5, alpha=0.5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)

    # Set view angle
    ax.view_init(elev=30, azim=-60)


def visualize_bev_features(ax, bev, title="BEV Features"):
    """Visualize BEV feature map."""
    # Take first 3 channels as RGB
    bev_np = bev[0, :3].cpu().numpy()
    bev_np = (bev_np - bev_np.min()) / (bev_np.max() - bev_np.min() + 1e-6)
    bev_np = np.transpose(bev_np, (1, 2, 0))

    ax.imshow(bev_np, extent=[-50, 50, -50, 50], origin='lower')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(title)

    # Vehicle marker
    ax.plot(0, 0, 'r^', markersize=8)


def visualize_traversability(ax, traversability, title="Traversability"):
    """Visualize traversability prediction."""
    trav = traversability[0, 0].cpu().numpy()

    im = ax.imshow(trav, cmap='RdYlGn', extent=[-50, 50, -50, 50],
                   origin='lower', vmin=0, vmax=1)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(title)
    ax.plot(0, 0, 'k^', markersize=8)

    # Colorbar
    plt.colorbar(im, ax=ax, label='Traversability', shrink=0.8)


def visualize_elevation(ax, elevation, title="Elevation"):
    """Visualize elevation prediction."""
    elev = elevation[0, 0].cpu().numpy()

    im = ax.imshow(elev, cmap='terrain', extent=[-50, 50, -50, 50], origin='lower')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(title)
    ax.plot(0, 0, 'r^', markersize=8)

    plt.colorbar(im, ax=ax, label='Height (m)', shrink=0.8)


def visualize_semantics(ax, semantics, title="Semantics"):
    """Visualize semantic prediction."""
    sem = semantics[0].argmax(dim=0).cpu().numpy()

    # Create colored image
    h, w = sem.shape
    sem_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for i, color in enumerate(SEMANTIC_COLORS):
        sem_rgb[sem == i] = color

    ax.imshow(sem_rgb, extent=[-50, 50, -50, 50], origin='lower')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(title)
    ax.plot(0, 0, 'w^', markersize=8)


def visualize_action_distribution(ax, logits, selected_action, title="Action Distribution"):
    """Visualize action probability distribution."""
    probs = torch.softmax(logits[0], dim=-1).cpu().numpy()

    # Get top actions
    top_k = min(8, len(probs))
    top_indices = np.argsort(probs)[-top_k:][::-1]
    top_probs = probs[top_indices]
    top_labels = [ACTION_LABELS.get(i, f"Action {i}") for i in top_indices]

    colors = ['green' if i == selected_action else 'steelblue' for i in top_indices]

    bars = ax.barh(range(top_k), top_probs, color=colors)
    ax.set_yticks(range(top_k))
    ax.set_yticklabels(top_labels)
    ax.set_xlabel('Probability')
    ax.set_title(title)
    ax.set_xlim(0, 1)

    # Add probability values
    for i, (bar, prob) in enumerate(zip(bars, top_probs)):
        ax.text(prob + 0.02, i, f'{prob:.2f}', va='center', fontsize=9)


def visualize_decision(ax, action, confidence, goal, state):
    """Visualize the decision with vehicle and trajectory."""
    ax.set_xlim(-5, 5)
    ax.set_ylim(-2, 8)

    # Draw vehicle
    vehicle = plt.Polygon([[-0.5, -0.3], [0.5, -0.3], [0.5, 0.3], [0, 0.6], [-0.5, 0.3]],
                          color='blue', alpha=0.8)
    ax.add_patch(vehicle)

    # Draw goal direction
    goal_np = goal[0].cpu().numpy()
    ax.arrow(0, 0.3, goal_np[0] * 4, goal_np[1] * 4,
             head_width=0.3, head_length=0.2, fc='green', ec='green', alpha=0.7)
    ax.text(goal_np[0] * 4.5, goal_np[1] * 4.5 + 0.5, 'Goal', fontsize=10, color='green')

    # Draw action direction
    action_idx = action.item() if isinstance(action, torch.Tensor) else action
    action_name = ACTION_LABELS.get(action_idx, f"Action {action_idx}")

    # Simple action to direction mapping
    action_dirs = {
        0: (0, 1),      # Forward
        1: (-0.5, 1),   # Forward-Left
        2: (0.5, 1),    # Forward-Right
        3: (-1, 0),     # Left
        4: (1, 0),      # Right
        5: (0, 0.5),    # Slow Forward
        11: (0, 0),     # Stop
    }

    dx, dy = action_dirs.get(action_idx, (0, 0.5))
    if dx != 0 or dy != 0:
        ax.arrow(0, 0.3, dx * 3, dy * 3,
                 head_width=0.2, head_length=0.15, fc='red', ec='red', linewidth=2)

    # Add text
    conf = confidence[0].item() if isinstance(confidence, torch.Tensor) else confidence
    ax.text(0, -1.5, f"Action: {action_name}", fontsize=12, ha='center', fontweight='bold')
    ax.text(0, -2.2, f"Confidence: {conf:.1%}", fontsize=10, ha='center')

    # Velocity
    vx = state[0, 0].item()
    ax.text(3, 7, f"Speed: {vx:.1f} m/s", fontsize=9)

    ax.set_aspect('equal')
    ax.set_title("Decision Output")
    ax.axis('off')


def visualize_future_frames(axes, future_bev, title_prefix="Future"):
    """Visualize predicted future BEV frames."""
    num_frames = min(len(axes), future_bev.shape[1])

    for i, ax in enumerate(axes[:num_frames]):
        if i < future_bev.shape[1]:
            frame = future_bev[0, i, :3].cpu().numpy()
            frame = (frame - frame.min()) / (frame.max() - frame.min() + 1e-6)
            frame = np.transpose(frame, (1, 2, 0))

            ax.imshow(frame, extent=[-50, 50, -50, 50], origin='lower')
            ax.set_title(f"{title_prefix} t+{i+1}")
            ax.plot(0, 0, 'r^', markersize=5)
        ax.axis('off')


def run_visualization(device='cuda', save_path=None):
    """Run full pipeline visualization."""
    print("=" * 60)
    print("TERRAINFORMER PIPELINE VISUALIZATION")
    print("=" * 60)

    # Create model
    print("Loading model...")
    model = TerrainFormer(
        lidar_in_channels=4,
        bev_size=256,
        num_actions=18,
        encoder_type='pointpillars',
    ).to(device)
    model.eval()

    # Create sample data
    print("Generating sample data...")
    points, state, goal, action_history = create_sample_data(device)

    # Run inference
    print("Running inference...")
    with torch.no_grad():
        # Get BEV features
        bev, global_feat = model.encode_lidar(points)

        # Get world model predictions
        world_outputs = model.world_model(bev, return_latent=True)

        # Get decision
        outputs = model(points, state, goal, action_history)
        action = outputs['action_logits'].argmax(dim=-1)
        confidence = torch.softmax(outputs['action_logits'], dim=-1).max(dim=-1)[0]

    print(f"Selected action: {ACTION_LABELS.get(action.item(), action.item())}")
    print(f"Confidence: {confidence.item():.2%}")

    # Create figure
    print("Creating visualization...")
    fig = plt.figure(figsize=(20, 14))
    fig.suptitle('TerrainFormer: World Model + Decision Transformer Pipeline',
                 fontsize=16, fontweight='bold')

    gs = GridSpec(4, 5, figure=fig, hspace=0.3, wspace=0.3)

    # Row 1: Input
    ax1 = fig.add_subplot(gs[0, 0])
    visualize_point_cloud_bev(ax1, points, "Input: Point Cloud (BEV)")

    ax2 = fig.add_subplot(gs[0, 1], projection='3d')
    visualize_point_cloud_3d(ax2, points, "Input: Point Cloud (3D)")

    ax3 = fig.add_subplot(gs[0, 2])
    visualize_bev_features(ax3, bev, "Encoder: BEV Features")

    # Row 1: Arrow and info
    ax_arrow1 = fig.add_subplot(gs[0, 3:])
    ax_arrow1.text(0.5, 0.7, "LiDAR Point Cloud", fontsize=14, ha='center', fontweight='bold')
    ax_arrow1.text(0.5, 0.5, "65,536 points (x, y, z, intensity)", fontsize=11, ha='center')
    ax_arrow1.annotate('', xy=(0.9, 0.3), xytext=(0.1, 0.3),
                       arrowprops=dict(arrowstyle='->', lw=2, color='blue'))
    ax_arrow1.text(0.5, 0.15, "PointPillars Encoder", fontsize=12, ha='center', color='blue')
    ax_arrow1.axis('off')

    # Row 2: World Model Predictions
    ax4 = fig.add_subplot(gs[1, 0])
    visualize_traversability(ax4, world_outputs['traversability'], "World Model: Traversability")

    ax5 = fig.add_subplot(gs[1, 1])
    visualize_elevation(ax5, world_outputs['elevation'], "World Model: Elevation")

    ax6 = fig.add_subplot(gs[1, 2])
    visualize_semantics(ax6, world_outputs['semantics'], "World Model: Semantics")

    # Future frames
    ax_future = [fig.add_subplot(gs[1, 3]), fig.add_subplot(gs[1, 4])]
    if 'future' in world_outputs and world_outputs['future'] is not None:
        visualize_future_frames(ax_future, world_outputs['future'], "Future BEV")
    else:
        for ax in ax_future:
            ax.text(0.5, 0.5, "Future\nPrediction", ha='center', va='center', fontsize=12)
            ax.axis('off')

    # Row 3: Decision
    ax7 = fig.add_subplot(gs[2, :2])
    visualize_action_distribution(ax7, outputs['action_logits'], action.item(),
                                  "Decision: Action Probabilities")

    ax8 = fig.add_subplot(gs[2, 2:4])
    visualize_decision(ax8, action, confidence, goal, state)

    # Info panel
    ax_info = fig.add_subplot(gs[2, 4])
    info_text = [
        "Model Info:",
        f"  Encoder: PointPillars",
        f"  BEV Size: 256x256",
        f"  Actions: 18 classes",
        "",
        "Input:",
        f"  Points: {points.shape[1]:,}",
        f"  Speed: {state[0,0].item():.1f} m/s",
        "",
        "Output:",
        f"  Action: {ACTION_LABELS.get(action.item(), '?')}",
        f"  Confidence: {confidence.item():.1%}",
    ]
    ax_info.text(0.1, 0.95, '\n'.join(info_text), transform=ax_info.transAxes,
                 fontsize=10, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    ax_info.axis('off')

    # Row 4: Pipeline flow
    ax_flow = fig.add_subplot(gs[3, :])
    draw_pipeline_flow(ax_flow)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")

    plt.show()
    print("Done!")


def draw_pipeline_flow(ax):
    """Draw the pipeline flow diagram."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1.5)

    # Boxes
    boxes = [
        (0.5, 0.5, "Point Cloud\n(65K x 4)", 'lightblue'),
        (2.5, 0.5, "PointPillars\nEncoder", 'lightgreen'),
        (4.5, 0.5, "BEV\n(64x256x256)", 'lightyellow'),
        (6.5, 0.5, "World Model\nTransformer", 'lightcoral'),
        (8.5, 0.5, "Decision\nTransformer", 'plum'),
    ]

    for x, y, text, color in boxes:
        rect = plt.Rectangle((x-0.6, y-0.35), 1.2, 0.7,
                              facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(x, y, text, ha='center', va='center', fontsize=9, fontweight='bold')

    # Arrows
    for i in range(len(boxes) - 1):
        x1 = boxes[i][0] + 0.6
        x2 = boxes[i+1][0] - 0.6
        ax.annotate('', xy=(x2, 0.5), xytext=(x1, 0.5),
                   arrowprops=dict(arrowstyle='->', lw=2, color='gray'))

    # Output
    ax.annotate('', xy=(9.5, 0.5), xytext=(9.1, 0.5),
               arrowprops=dict(arrowstyle='->', lw=2, color='gray'))
    ax.text(9.7, 0.5, "Action", fontsize=10, va='center', fontweight='bold')

    # Timing
    ax.text(2.5, 0.05, "~5ms", ha='center', fontsize=8, color='blue')
    ax.text(6.5, 0.05, "~10ms", ha='center', fontsize=8, color='blue')
    ax.text(8.5, 0.05, "~5ms", ha='center', fontsize=8, color='blue')

    ax.text(5, 1.3, "TerrainFormer Pipeline (~20ms total, 50 FPS)",
            ha='center', fontsize=12, fontweight='bold')

    ax.axis('off')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Visualize TerrainFormer pipeline')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--save', type=str, default=None, help='Save path for figure')
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    run_visualization(device=device, save_path=args.save)
