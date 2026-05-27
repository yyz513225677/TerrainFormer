"""
Example usage of RELLIS-3D data loader with SemanticKITTI format

This example shows how to load and iterate through the RELLIS-3D dataset
with the 70/15/15 train/val/test split.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data.datasets.rellis3d_dataset import RELLIS3DDataset
from torch.utils.data import DataLoader


def main():
    # Configuration
    RELLIS_ROOT = '../RELLIS/bin/'
    BATCH_SIZE = 4
    NUM_WORKERS = 4

    # Create datasets for each split
    train_dataset = RELLIS3DDataset(
        root_path=RELLIS_ROOT,
        split='train',
        max_points=65536,
        history_frames=5,
        future_frames=5,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15
    )

    val_dataset = RELLIS3DDataset(
        root_path=RELLIS_ROOT,
        split='val',
        max_points=65536,
        history_frames=5,
        future_frames=5,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15
    )

    test_dataset = RELLIS3DDataset(
        root_path=RELLIS_ROOT,
        split='test',
        max_points=65536,
        history_frames=5,
        future_frames=5,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15
    )

    print(f"Dataset sizes:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val:   {len(val_dataset)} samples")
    print(f"  Test:  {len(test_dataset)} samples")
    print()

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    # Example: Iterate through training data
    print("Example batch from training set:")
    for batch_idx, batch in enumerate(train_loader):
        print(f"\nBatch {batch_idx}:")
        print(f"  Point cloud shape: {batch['point_cloud'].shape}")
        print(f"  History shape: {batch['point_cloud_history'].shape}")
        print(f"  Future shape: {batch['future_point_clouds'].shape}")
        print(f"  Terrain labels shape: {batch['terrain_labels'].shape}")
        print(f"  Traversability map shape: {batch['traversability_map'].shape}")
        print(f"  Expert actions: {batch['expert_action']}")
        print(f"  Vehicle states shape: {batch['vehicle_state'].shape}")

        # Only show first batch
        break

    # Example: Access action statistics
    print("\nAction distribution in training set:")
    action_stats = train_dataset.get_action_statistics()
    for action, count in sorted(action_stats.items()):
        print(f"  Action {action}: {count} samples")


if __name__ == '__main__':
    main()
