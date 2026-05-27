#!/usr/bin/env python3
"""
Predictive Evaluation Script

Compares action predictions from:
  [1] Current frame  — world model features from ground-truth observation
  [2] Predicted frame — world model features from the world model's own future prediction

Usage:
    python scripts/evaluate_predictive.py
    python scripts/evaluate_predictive.py --checkpoint outputs/decision_train/best_model.pt
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from models.unified.terrainformer import TerrainFormer
from data.datasets.rellis3d_dataset import RELLIS3DDataset
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt


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


def evaluate_predictive(model, test_loader, device='cuda'):
    """
    For each batch:
      - Current path:   BEV → WorldModel.encode() → DecisionTransformer
      - Predicted path: BEV → WorldModel() → future[0] → upsample →
                        WorldModel.encode(future_bev) → DecisionTransformer
    """
    model.eval()
    all_preds_current = []
    all_preds_predicted = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating (Predictive)"):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            points        = batch['point_cloud']
            state         = batch['vehicle_state']
            goal          = batch['goal_direction']
            action_history = batch['action_sequence']
            expert_action  = batch['expert_action']

            # ── Current frame path ──────────────────────────────────────────
            bev, _ = model.encode_lidar(points)
            world_out_cur = model.world_model(bev, return_latent=True)
            dec_out_cur = model.decision_transformer(
                world_out_cur['global_feature'],
                world_out_cur['latent'],
                state, goal, action_history,
            )
            preds_cur = dec_out_cur['action_logits'].argmax(dim=-1)

            # ── Predicted future frame path ──────────────────────────────────
            # future shape: (B, num_future_frames, C, patch_h, patch_w)
            future_patches = world_out_cur['future']   # (B, 5, 64, 16, 16)
            future_step0   = future_patches[:, 0]      # (B, 64, 16, 16)
            # Upsample patch-resolution prediction back to full BEV
            future_bev = F.interpolate(future_step0.float(), size=(256, 256),
                                       mode='bilinear', align_corners=False)
            world_out_pred = model.world_model(future_bev, return_latent=True)
            dec_out_pred = model.decision_transformer(
                world_out_pred['global_feature'],
                world_out_pred['latent'],
                state, goal, action_history,
            )
            preds_pred = dec_out_pred['action_logits'].argmax(dim=-1)

            all_preds_current.extend(preds_cur.cpu().numpy())
            all_preds_predicted.extend(preds_pred.cpu().numpy())
            all_targets.extend(expert_action.cpu().numpy())

    return (np.array(all_preds_current),
            np.array(all_preds_predicted),
            np.array(all_targets))


def compute_metrics(preds, targets):
    accuracy = accuracy_score(targets, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        targets, preds, average='macro', zero_division=0
    )
    return {'accuracy': accuracy, 'precision_macro': prec,
            'recall_macro': rec, 'f1_macro': f1}


def plot_comparison(metrics_cur, metrics_pred, output_path):
    labels = ['Accuracy', 'Precision', 'Recall', 'F1']
    cur_vals  = [metrics_cur['accuracy'],  metrics_cur['precision_macro'],
                 metrics_cur['recall_macro'],  metrics_cur['f1_macro']]
    pred_vals = [metrics_pred['accuracy'], metrics_pred['precision_macro'],
                 metrics_pred['recall_macro'], metrics_pred['f1_macro']]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, cur_vals,  width, label='Current Frame',   color='steelblue')
    bars2 = ax.bar(x + width/2, pred_vals, width, label='Predicted Frame',  color='coral')

    ax.set_ylabel('Score')
    ax.set_title('Decision Accuracy: Current Frame vs. Predicted Future Frame')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_ylim(0, 1.05)

    for bar in bars1 + bars2:
        h = bar.get_height()
        ax.annotate(f'{h:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords='offset points',
                    ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Comparison plot saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str,
                        default='outputs/decision_train/best_model.pt')
    parser.add_argument('--data', type=str, default='../RELLIS/bin/')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=16)
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
    preds_cur, preds_pred, targets = evaluate_predictive(model, test_loader, device)

    metrics_cur  = compute_metrics(preds_cur,  targets)
    metrics_pred = compute_metrics(preds_pred, targets)

    acc_diff  = metrics_cur['accuracy'] - metrics_pred['accuracy']
    agreement = float(np.mean(preds_cur == preds_pred))

    output_dir = Path('outputs/decision_train')
    output_dir.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("=" * 70)
    lines.append("PREDICTIVE EVALUATION RESULTS")
    lines.append("=" * 70)
    lines.append("")
    lines.append("[1] CURRENT FRAME (ground-truth observation):")
    lines.append(f"    Accuracy:  {metrics_cur['accuracy']:.4f} ({metrics_cur['accuracy']*100:.2f}%)")
    lines.append(f"    Precision: {metrics_cur['precision_macro']:.4f}")
    lines.append(f"    Recall:    {metrics_cur['recall_macro']:.4f}")
    lines.append(f"    F1 Score:  {metrics_cur['f1_macro']:.4f}")
    lines.append("")
    lines.append("[2] PREDICTED FUTURE FRAME (world model prediction):")
    lines.append(f"    Accuracy:  {metrics_pred['accuracy']:.4f} ({metrics_pred['accuracy']*100:.2f}%)")
    lines.append(f"    Precision: {metrics_pred['precision_macro']:.4f}")
    lines.append(f"    Recall:    {metrics_pred['recall_macro']:.4f}")
    lines.append(f"    F1 Score:  {metrics_pred['f1_macro']:.4f}")
    lines.append("")
    lines.append("-" * 70)
    lines.append(f"Accuracy difference (current − predicted): {acc_diff:+.4f} ({acc_diff*100:+.2f}%)")
    if acc_diff > 0:
        lines.append(f"  → Prediction quality causes {acc_diff*100:.1f}% accuracy drop")
    else:
        lines.append(f"  → Predicted future actually helps! ({abs(acc_diff)*100:.1f}% improvement)")
    lines.append(f"Prediction agreement (same action both paths): {agreement:.4f} ({agreement*100:.2f}%)")
    lines.append("=" * 70)

    for line in lines:
        print(line)

    results_file = output_dir / 'predictive_results.txt'
    results_file.write_text('\n'.join(lines))
    print(f"\nResults saved to: {results_file}")

    # Plot
    plot_comparison(metrics_cur, metrics_pred, output_dir / 'predictive_comparison.png')
    print("\nEvaluation complete!")


if __name__ == '__main__':
    main()
