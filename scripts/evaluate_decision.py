#!/usr/bin/env python3
"""
Decision Transformer Evaluation Script

Loads the full TerrainFormer checkpoint and evaluates action prediction
accuracy on the RELLIS-3D test split.

Usage:
    python scripts/evaluate_decision.py
    python scripts/evaluate_decision.py --checkpoint outputs/decision_train/best_model.pt
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import torch
import numpy as np
from tqdm import tqdm

from models.unified.terrainformer import TerrainFormer
from data.datasets.rellis3d_dataset import RELLIS3DDataset
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

ACTION_NAMES = {
    0: 'stop', 1: 'fwd_slow', 2: 'fwd_med', 3: 'fwd_fast',
    4: 'L_sharp', 5: 'L_med', 6: 'L_slight',
    7: 'R_slight', 8: 'R_med', 9: 'R_sharp',
    10: 'fwd_L', 11: 'fwd_R',
}


def create_model(device='cuda'):
    model = TerrainFormer(
        lidar_in_channels=4,
        bev_size=256,
        num_actions=12,
        decision_chunk_size=5,
    ).to(device)
    return model


def load_checkpoint(model, checkpoint_path, device='cuda'):
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"  Epoch: {ckpt.get('epoch', 'N/A')}")
    print(f"  Best accuracy: {ckpt.get('best_accuracy', 0.0):.4f}")
    return ckpt


def evaluate(model, test_loader, device='cuda'):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            outputs = model(
                points=batch['point_cloud'],
                state=batch['vehicle_state'],
                goal=batch['goal_direction'],
                action_history=batch['action_sequence'],
                return_world_predictions=False,
            )
            preds = outputs['action_logits'].argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(batch['expert_action'].cpu().numpy())

    return np.array(all_preds), np.array(all_targets)


def compute_metrics(preds, targets, num_classes=12):
    accuracy = accuracy_score(targets, preds)
    precision, recall, f1, support = precision_recall_fscore_support(
        targets, preds, average=None, labels=list(range(num_classes)), zero_division=0
    )
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        targets, preds, average='macro', zero_division=0
    )
    cm = confusion_matrix(targets, preds, labels=list(range(num_classes)))
    return {
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_per_class': precision,
        'recall_per_class': recall,
        'f1_per_class': f1,
        'support_per_class': support,
        'confusion_matrix': cm,
    }


def plot_confusion_matrix(cm, output_path, normalize=False):
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    labels = [ACTION_NAMES.get(i, str(i)) for i in range(cm.shape[0])]

    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', square=True,
                xticklabels=labels, yticklabels=labels, ax=axes[0])
    axes[0].set_xlabel('Predicted Action')
    axes[0].set_ylabel('True Action')
    axes[0].set_title('Confusion Matrix (counts)')

    # Normalized (row = per-class recall)
    cm_norm = cm.astype(float)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # avoid div-by-zero for empty classes
    cm_norm = cm_norm / row_sums
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', square=True,
                xticklabels=labels, yticklabels=labels, vmin=0, vmax=1, ax=axes[1])
    axes[1].set_xlabel('Predicted Action')
    axes[1].set_ylabel('True Action')
    axes[1].set_title('Confusion Matrix (row-normalized recall)')

    plt.suptitle('Decision Transformer — Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to: {output_path}")


def plot_per_class_f1(metrics, output_path):
    labels = [ACTION_NAMES.get(i, str(i)) for i in range(len(metrics['f1_per_class']))]
    f1_vals = metrics['f1_per_class']
    support = metrics['support_per_class']

    colors = ['#2ecc71' if f >= 0.5 else '#e67e22' if f >= 0.2 else '#e74c3c'
              for f in f1_vals]

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(labels, f1_vals, color=colors)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('F1 Score')
    ax.set_title('Per-Class F1 Score (green ≥ 0.5, orange ≥ 0.2, red < 0.2)')
    ax.axhline(metrics['f1_macro'], color='navy', linestyle='--',
               label=f'Macro F1 = {metrics["f1_macro"]:.3f}')
    ax.legend()

    for bar, sup in zip(bars, support):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'n={int(sup)}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Per-class F1 chart saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str,
                        default='outputs/decision_train/best_model.pt')
    parser.add_argument('--data', type=str, default='../RELLIS/bin/')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")

    # Model
    print("Creating TerrainFormer...")
    model = create_model(device)

    # Checkpoint
    print("\nLoading checkpoint...")
    load_checkpoint(model, args.checkpoint, device)

    # Dataset
    print("\nLoading test dataset...")
    test_dataset = RELLIS3DDataset(root_path=args.data, split='test', chunk_size=5)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=4, pin_memory=True)
    print(f"Test samples: {len(test_dataset)}")

    # Evaluate
    print("\nEvaluating...")
    preds, targets = evaluate(model, test_loader, device)

    # Metrics
    metrics = compute_metrics(preds, targets)

    output_dir = Path('outputs/decision_train')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build results text
    lines = []
    lines.append("=" * 60)
    lines.append("EVALUATION RESULTS")
    lines.append("=" * 60)
    lines.append(f"Accuracy:          {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    lines.append(f"Precision (macro): {metrics['precision_macro']:.4f}")
    lines.append(f"Recall (macro):    {metrics['recall_macro']:.4f}")
    lines.append(f"F1 Score (macro):  {metrics['f1_macro']:.4f}")
    lines.append("")
    lines.append("=" * 60)
    lines.append("PER-CLASS METRICS")
    lines.append("=" * 60)
    lines.append(f"{'ID':<4} {'Action':<12} {'Prec':>8} {'Rec':>8} {'F1':>8} {'Support':>8}")
    lines.append("-" * 54)
    for i in range(12):
        name = ACTION_NAMES.get(i, str(i))
        lines.append(f"{i:<4} {name:<12} "
                     f"{metrics['precision_per_class'][i]:>8.4f} "
                     f"{metrics['recall_per_class'][i]:>8.4f} "
                     f"{metrics['f1_per_class'][i]:>8.4f} "
                     f"{int(metrics['support_per_class'][i]):>8}")
    lines.append("")
    lines.append("=" * 60)
    lines.append("ACTION DISTRIBUTION (Test Set)")
    lines.append("=" * 60)
    for i in range(12):
        count = int(metrics['support_per_class'][i])
        total = len(targets)
        pct = 100 * count / total if total > 0 else 0
        lines.append(f"  {ACTION_NAMES.get(i, str(i)):<12}: {count:5d} ({pct:.1f}%)")

    for line in lines:
        print(line)

    results_file = output_dir / 'evaluation_results.txt'
    results_file.write_text('\n'.join(lines))
    print(f"\nResults saved to: {results_file}")

    # Plots
    print("Generating plots...")
    plot_confusion_matrix(metrics['confusion_matrix'], output_dir / 'confusion_matrix.png')
    plot_per_class_f1(metrics, output_dir / 'per_class_f1.png')


if __name__ == '__main__':
    main()
