#!/usr/bin/env python3
"""
Decision Transformer Evaluation Script

Usage:
    python scripts/evaluate_decision.py --checkpoint outputs/decision_train/best_model.pt
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import torch
import numpy as np
from tqdm import tqdm

from models.decision.decision_transformer import DecisionTransformer
from models.world_model.world_model import WorldModel
from data.datasets.rellis3d_dataset import RELLIS3DDataset
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from utils.bev_projection_fast import UltraFastBEV
import matplotlib.pyplot as plt
import seaborn as sns


def create_models(device='cuda'):
    """Create world model and decision transformer"""

    # World model (for encoding)
    world_model = WorldModel(
        bev_channels=64,
        embed_dim=512,
        patch_size=16,
        img_size=256,
    ).to(device)

    # Decision transformer
    decision_model = DecisionTransformer(
        world_model_dim=512,
        hidden_dim=384,
        num_layers=4,
        num_heads=6,
        num_actions=18,
    ).to(device)

    # BEV projection
    bev_projection = UltraFastBEV(
        x_range=(-50, 50),
        y_range=(-50, 50),
        z_range=(-3, 5),
        bev_size=256,
        num_features=64
    ).to(device)

    return bev_projection, world_model, decision_model


def load_checkpoints(world_model, decision_model, decision_ckpt_path, device='cuda'):
    """Load trained weights"""

    # Load decision transformer
    print(f"Loading decision checkpoint: {decision_ckpt_path}")
    decision_ckpt = torch.load(decision_ckpt_path, map_location=device, weights_only=False)
    decision_model.load_state_dict(decision_ckpt['model_state_dict'])

    # Load world model
    world_ckpt_path = Path('outputs/world_model_pretrain/best_model.pt')
    if world_ckpt_path.exists():
        print(f"Loading world model checkpoint: {world_ckpt_path}")
        world_ckpt = torch.load(world_ckpt_path, map_location=device, weights_only=False)
        world_model.load_state_dict(world_ckpt['model_state_dict'])
    else:
        print(f"Warning: World model checkpoint not found at {world_ckpt_path}")

    return decision_ckpt


def evaluate(bev_projection, world_model, decision_model, test_loader, device='cuda'):
    """Run evaluation on test set"""

    bev_projection.eval()
    world_model.eval()
    decision_model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            # Move to device
            point_cloud = batch['point_cloud'].to(device)
            vehicle_state = batch['vehicle_state'].to(device)
            goal_direction = batch['goal_direction'].to(device)
            action_sequence = batch['action_sequence'].to(device)
            expert_action = batch['expert_action'].to(device)

            # Forward pass: BEV -> world model -> decision
            bev_features = bev_projection(point_cloud)

            # Get world model features using encode
            world_latent, world_global = world_model.encode(bev_features)

            # Decision transformer forward
            outputs = decision_model(
                world_global=world_global,
                world_latent=world_latent,
                state=vehicle_state,
                goal=goal_direction,
                action_history=action_sequence
            )
            action_logits = outputs['action_logits']

            # Get predictions
            predicted_actions = torch.argmax(action_logits, dim=-1)

            # Collect results
            all_preds.extend(predicted_actions.cpu().numpy())
            all_targets.extend(expert_action.cpu().numpy())

    return np.array(all_preds), np.array(all_targets)


def compute_metrics(preds, targets):
    """Compute evaluation metrics"""

    accuracy = accuracy_score(targets, preds)

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        targets, preds, average=None, zero_division=0
    )

    # Macro averages
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        targets, preds, average='macro', zero_division=0
    )

    # Confusion matrix
    cm = confusion_matrix(targets, preds)

    return {
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_per_class': precision,
        'recall_per_class': recall,
        'f1_per_class': f1,
        'support_per_class': support,
        'confusion_matrix': cm
    }


def plot_confusion_matrix(cm, output_path='outputs/confusion_matrix.png'):
    """Plot and save confusion matrix"""

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', square=True)
    plt.xlabel('Predicted Action')
    plt.ylabel('True Action')
    plt.title('Decision Transformer - Confusion Matrix')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"\nConfusion matrix saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='outputs/decision_train/best_model.pt')
    parser.add_argument('--data', type=str, default='../RELLIS/bin/')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")

    # Create models
    print("Creating models...")
    bev_projection, world_model, decision_model = create_models(device)

    # Load checkpoints
    print("\nLoading checkpoints...")
    ckpt = load_checkpoints(world_model, decision_model, args.checkpoint, device)
    print(f"\nCheckpoint info:")
    print(f"  - Epoch: {ckpt.get('epoch', 'N/A')}")
    print(f"  - Best Accuracy: {ckpt.get('best_accuracy', 'N/A'):.4f}")

    # Create test dataset
    print("\nLoading test dataset...")
    test_dataset = RELLIS3DDataset(root_path=args.data, split='test')
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    print(f"Test samples: {len(test_dataset)}")

    # Evaluate
    print("\nEvaluating...")
    preds, targets = evaluate(bev_projection, world_model, decision_model, test_loader, device)

    # Compute metrics
    metrics = compute_metrics(preds, targets)

    # Build results string
    output_dir = Path('outputs/decision_train')
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    results.append("=" * 60)
    results.append("EVALUATION RESULTS")
    results.append("=" * 60)
    results.append(f"Overall Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    results.append(f"Precision (macro): {metrics['precision_macro']:.4f}")
    results.append(f"Recall (macro):    {metrics['recall_macro']:.4f}")
    results.append(f"F1 Score (macro):  {metrics['f1_macro']:.4f}")
    results.append("")
    results.append("=" * 60)
    results.append("PER-CLASS METRICS")
    results.append("=" * 60)
    results.append(f"{'Action':<10} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<10}")
    results.append("-" * 60)

    unique_actions = np.unique(targets)
    for action in unique_actions:
        if action < len(metrics['precision_per_class']):
            results.append(f"{action:<10} {metrics['precision_per_class'][action]:<12.4f} "
                  f"{metrics['recall_per_class'][action]:<12.4f} "
                  f"{metrics['f1_per_class'][action]:<12.4f} "
                  f"{int(metrics['support_per_class'][action]):<10}")

    results.append("")
    results.append("=" * 60)
    results.append("ACTION DISTRIBUTION (Test Set)")
    results.append("=" * 60)
    for action in unique_actions:
        count = np.sum(targets == action)
        pct = 100 * count / len(targets)
        results.append(f"Action {action}: {count:5d} samples ({pct:5.2f}%)")

    results.append("")
    results.append("=" * 60)
    results.append("Evaluation complete!")
    results.append("=" * 60)

    # Print results
    for line in results:
        print(line)

    # Save to file
    results_file = output_dir / 'evaluation_results.txt'
    with open(results_file, 'w') as f:
        f.write('\n'.join(results))
    print(f"\nResults saved to: {results_file}")

    # Plot confusion matrix
    print("Generating confusion matrix...")
    plot_confusion_matrix(metrics['confusion_matrix'], output_dir / 'confusion_matrix.png')


if __name__ == '__main__':
    main()
