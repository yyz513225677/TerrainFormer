#!/usr/bin/env python3
"""
TerrainFormer Training Script

Usage:
    python scripts/train.py --config configs/training/pretrain_world_model.yaml
    python scripts/train.py --config configs/training/train_decision.yaml
    python scripts/train.py --config configs/training/finetune_joint.yaml
"""

import argparse
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import torch
import yaml
import numpy as np
from omegaconf import OmegaConf
from collections import Counter

from models import TerrainFormer
from models.world_model import WorldModel
from models.decision import DecisionTransformer
from data import create_dataloader
from data.datasets import RELLIS3DDataset, TartanDriveDataset, LidarDustXDataset, GOOSE3DDataset
from data.preprocessing import Augmentation
from training import WorldModelTrainer, DecisionTrainer, JointTrainer
from training.schedulers import create_scheduler
from training.losses.world_model_loss import WorldModelLoss
from utils import setup_logging


def compute_class_weights(dataset, num_classes: int = 12):
    """
    Compute class weights from action distribution in dataset.
    Uses inverse frequency weighting: weight = total / (num_classes * count)

    Args:
        dataset: Training dataset
        num_classes: Number of action classes (default: 12)

    Returns:
        List of class weights
    """
    print("Computing class weights from action distribution...")

    # Count actions across all samples
    action_counts = Counter()

    # Handle ConcatDataset
    from torch.utils.data import ConcatDataset
    if isinstance(dataset, ConcatDataset):
        datasets = dataset.datasets
    else:
        datasets = [dataset]

    # Iterate through all datasets and count actions
    for ds in datasets:
        if hasattr(ds, 'samples'):
            for sample in ds.samples:
                action = sample.get('action', 0)
                action_counts[action] += 1

    total = sum(action_counts.values())

    # Print distribution
    print(f"Action distribution (total {total} samples):")
    for action in sorted(action_counts.keys()):
        count = action_counts[action]
        pct = 100 * count / total
        print(f"  Action {action:2d}: {count:6d} ({pct:5.1f}%)")

    # Compute inverse frequency weights
    # Formula: weight = total / (num_classes * count)
    # This gives higher weight to rare classes
    weights = []
    for i in range(num_classes):
        count = action_counts.get(i, 1)  # Avoid division by zero
        weight = total / (num_classes * count)
        weights.append(weight)

    # Normalize so weights sum to num_classes (average weight = 1)
    weight_sum = sum(weights)
    weights = [w * num_classes / weight_sum for w in weights]

    print(f"Computed class weights: {[f'{w:.2f}' for w in weights]}")

    return weights


def parse_args():
    parser = argparse.ArgumentParser(description='Train TerrainFormer')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    return parser.parse_args()


def load_config(config_path: str):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return OmegaConf.create(config)


def create_datasets(config):
    """Create train and validation datasets."""
    datasets = {'train': [], 'val': []}

    # Load dataset config to get paths
    dataset_config_path = config.data.get('config', 'configs/data/dataset_config.yaml')
    with open(dataset_config_path, 'r') as f:
        dataset_config = yaml.safe_load(f)

    data_config = config.data

    # Augmentation for training only (reduces overfitting on sequential data)
    train_augmentation = Augmentation(
        random_rotation=True,
        rotation_range=(-45, 45),
        random_flip=True,
        flip_prob=0.5,
        random_scale=True,
        scale_range=(0.95, 1.05),
        random_translation=True,
        translation_range=(-0.5, 0.5),
        random_dropout=False,  # Disabled: changes point count, breaks batching
        random_noise=True,
        noise_std=0.02,
        intensity_jitter=True,
        intensity_range=(-0.1, 0.1),
    )

    # Frame stride to reduce sequential redundancy (training only)
    train_frame_stride = data_config.get('frame_stride', 3)
    chunk_size = data_config.get('chunk_size', 1)

    for split in ['train', 'val']:
        split_config = data_config.get(split, {})
        dataset_names = split_config.get('datasets', ['rellis3d'])
        is_train = (split == 'train')

        for name in dataset_names:
            if name == 'rellis3d':
                rellis_path = dataset_config['datasets']['rellis3d']['path']
                rellis3d_sequences = data_config.get('rellis3d_sequences', None)
                dataset = RELLIS3DDataset(
                    root_path=rellis_path,
                    split=split,
                    sequences=rellis3d_sequences,
                    max_points=65536,
                    transform=train_augmentation if is_train else None,
                    frame_stride=train_frame_stride if is_train else 1,
                    chunk_size=chunk_size,
                )
            elif name == 'tartandrive':
                tartandrive_path = dataset_config['datasets']['tartandrive']['path']
                dataset = TartanDriveDataset(
                    root_path=tartandrive_path,
                    split=split,
                    max_points=65536,
                )
            elif name == 'lidardustx':
                lidardustx_path = dataset_config['datasets']['lidardustx']['path']
                sensors = dataset_config['datasets']['lidardustx'].get('sensors', None)
                dataset = LidarDustXDataset(
                    root_path=lidardustx_path,
                    split=split,
                    sensors=sensors,
                    max_points=65536,
                    transform=train_augmentation if is_train else None,
                    frame_stride=train_frame_stride if is_train else 1,
                )
            elif name == 'goose3d':
                goose3d_path = dataset_config['datasets']['goose3d']['path']
                dataset = GOOSE3DDataset(
                    root_path=goose3d_path,
                    split=split,
                    max_points=65536,
                    transform=train_augmentation if is_train else None,
                    frame_stride=train_frame_stride if is_train else 1,
                )
            else:
                print(f"Unknown dataset: {name}")
                continue

            datasets[split].append(dataset)

    # Concatenate datasets
    from torch.utils.data import ConcatDataset

    train_dataset = ConcatDataset(datasets['train']) if datasets['train'] else None
    val_dataset = ConcatDataset(datasets['val']) if datasets['val'] else None

    return train_dataset, val_dataset


def create_model(config, phase: str):
    """Create model based on training phase."""
    # Get encoder type from config or use default
    encoder_type = config.model.get('encoder_type', 'pointpillars')
    chunk_size = config.data.get('chunk_size', 1)

    if phase == 'world_model_pretrain':
        return WorldModel(
            bev_channels=64,
            embed_dim=512,
            patch_size=16,
            img_size=256,
        )
    elif phase == 'decision_train':
        # For decision training, we need the full TerrainFormer model
        # (encoder + world model + decision transformer)
        # The encoder and world model will be frozen during training
        return TerrainFormer(
            lidar_in_channels=4,
            bev_size=256,
            num_actions=12,
            encoder_type=encoder_type,
            decision_chunk_size=chunk_size,
        )
    else:  # joint
        return TerrainFormer(
            lidar_in_channels=4,
            bev_size=256,
            num_actions=12,
            encoder_type=encoder_type,
            decision_chunk_size=chunk_size,
        )


def main():
    args = parse_args()
    config = load_config(args.config)
    
    # Setup
    output_dir = config.experiment.output_dir
    os.makedirs(output_dir, exist_ok=True)
    logger = setup_logging(output_dir, config.experiment.name)
    
    logger.info(f"Configuration: {config}")
    
    # Set seed
    torch.manual_seed(config.experiment.seed)
    
    # Device
    device = args.device if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Create datasets
    logger.info("Creating datasets...")
    train_dataset, val_dataset = create_datasets(config)
    
    if train_dataset is None:
        logger.warning("No training data found. Using dummy data for demonstration.")
        from torch.utils.data import TensorDataset
        dummy_points = torch.randn(100, 16384, 4)
        dummy_labels = torch.randint(0, 12, (100,))
        train_dataset = TensorDataset(dummy_points, dummy_labels)
        val_dataset = train_dataset
        
    # Create dataloaders
    train_loader = create_dataloader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.data.get('num_workers', 4),
    )
    
    val_loader = create_dataloader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.data.get('num_workers', 4),
    )
    
    # Determine training phase
    exp_name = config.experiment.name
    if 'pretrain' in exp_name:
        phase = 'world_model_pretrain'
    elif 'decision' in exp_name:
        phase = 'decision_train'
    else:
        phase = 'joint'
        
    logger.info(f"Training phase: {phase}")
    
    # Create model
    model = create_model(config, phase)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # Load pretrained weights if specified
    if hasattr(config.model, 'pretrained') and config.model.pretrained:
        pretrained_path = config.model.pretrained.get('world_model')
        if pretrained_path and os.path.exists(pretrained_path):
            logger.info(f"Loading pretrained world model weights from {pretrained_path}")
            checkpoint = torch.load(pretrained_path, map_location=device)
            state_dict = checkpoint['model_state_dict']
            # If loading a standalone WorldModel checkpoint into TerrainFormer,
            # remap keys by adding 'world_model.' prefix so they match submodule names
            if hasattr(model, 'world_model') and not any(k.startswith('world_model.') for k in state_dict):
                state_dict = {f'world_model.{k}': v for k, v in state_dict.items()}
                logger.info("Remapped checkpoint keys: added 'world_model.' prefix")
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            loaded = len(state_dict) - len(missing)
            logger.info(f"Loaded {loaded}/{len(state_dict)} pretrained parameters")
            if missing:
                logger.warning(f"  {len(missing)} missing keys (expected if only world model is pretrained)")
            
    # Create optimizer
    opt_config = config.training.optimizer
    if hasattr(config.training.optimizer, 'lr_config'):
        # Layer-wise learning rates
        param_groups = model.get_parameter_groups(dict(opt_config.lr_config))
        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=float(opt_config.weight_decay),
            betas=tuple(opt_config.betas),
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(opt_config.lr),
            weight_decay=float(opt_config.weight_decay),
            betas=tuple(opt_config.betas),
        )

    # Create scheduler
    scheduler = create_scheduler(
        optimizer,
        scheduler_type=config.training.scheduler.type,
        num_epochs=config.training.epochs,
        warmup_epochs=config.training.scheduler.warmup_epochs,
        min_lr=float(config.training.scheduler.min_lr),
    )
    
    # Create trainer based on phase
    if phase == 'world_model_pretrain':
        # Create loss function for world model
        loss_config = config.get('loss', {})
        loss_fn = WorldModelLoss(
            reconstruction_weight=loss_config.get('reconstruction', {}).get('weight', 1.0),
            traversability_weight=loss_config.get('traversability', {}).get('weight', 0.5),
            elevation_weight=loss_config.get('elevation', {}).get('weight', 0.3),
            semantics_weight=loss_config.get('semantics', {}).get('weight', 0.5),
        )

        trainer = WorldModelTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn,
            device=device,
            output_dir=output_dir,
            fp16=config.training.fp16,
        )
    elif phase == 'decision_train':
        # Extract loss configuration
        loss_config = config.get('loss', {}).get('action', {})
        loss_type = loss_config.get('type', 'focal')
        label_smoothing = loss_config.get('label_smoothing', 0.1)
        focal_gamma = loss_config.get('gamma', 2.0)
        class_weights = loss_config.get('class_weights', None)

        # Auto-compute class weights if not provided
        if class_weights is None or class_weights == 'auto':
            class_weights = compute_class_weights(train_dataset, num_classes=12)

        trainer = DecisionTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            output_dir=output_dir,
            fp16=config.training.fp16,
            loss_type=loss_type,
            label_smoothing=label_smoothing,
            focal_gamma=focal_gamma,
            class_weights=class_weights,
            num_classes=12,
        )
    else:
        trainer = JointTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            output_dir=output_dir,
            fp16=config.training.fp16,
        )

    # Train
    logger.info("Starting training...")
    trainer.train(config.training.epochs)
    
    logger.info("Training complete!")


if __name__ == '__main__':
    main()
