"""Visualization utilities"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional


def visualize_predictions(point_cloud: np.ndarray,
                          traversability: np.ndarray,
                          action_probs: np.ndarray,
                          predicted_action: int,
                          save_path: Optional[str] = None):
    """
    Visualize model predictions.
    
    Args:
        point_cloud: Input point cloud (N, 4)
        traversability: Traversability map (H, W)
        action_probs: Action probability distribution (num_actions,)
        predicted_action: Predicted action ID
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Point cloud BEV
    ax = axes[0]
    ax.scatter(point_cloud[:, 0], point_cloud[:, 1], c=point_cloud[:, 2], 
               s=0.1, cmap='viridis')
    ax.set_title('Point Cloud (BEV)')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_aspect('equal')
    
    # Traversability map
    ax = axes[1]
    im = ax.imshow(traversability, cmap='RdYlGn', vmin=0, vmax=1)
    ax.set_title('Traversability Map')
    plt.colorbar(im, ax=ax)
    
    # Action distribution
    ax = axes[2]
    action_names = ['straight', 'L5', 'R5', 'L10', 'R10', 'L20', 'R20',
                    'L30', 'R30', 'L45', 'R45', 'maintain', 'slow', 'fast',
                    'stop', 'rev', 'revL', 'revR']
    ax.bar(range(len(action_probs)), action_probs)
    ax.axvline(predicted_action, color='r', linestyle='--', label='Predicted')
    ax.set_xticks(range(len(action_names)))
    ax.set_xticklabels(action_names, rotation=45, ha='right')
    ax.set_title('Action Distribution')
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()


def visualize_bev_features(bev: np.ndarray, save_path: Optional[str] = None):
    """Visualize BEV feature channels."""
    n_channels = min(16, bev.shape[0])
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    
    for i, ax in enumerate(axes.flat):
        if i < n_channels:
            ax.imshow(bev[i], cmap='viridis')
            ax.set_title(f'Channel {i}')
        ax.axis('off')
        
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()
