#!/usr/bin/env python3
"""
Predictive Evaluation Script

Evaluates the full pipeline:
1. World Model predicts next frame from current frame
2. Decision Transformer uses PREDICTED frame (not ground truth)
3. Compare action predictions vs ground truth from poses

Usage:
    python scripts/evaluate_predictive.py
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import torch
import torch.nn.functional as F
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

    world_model = WorldModel(
        bev_channels=64,
        embed_dim=512,
        patch_size=16,
        img_size=256,
    ).to(device)

    decision_model = DecisionTransformer(
        world_model_dim=512,
        hidden_dim=384,
        num_layers=4,
        num_heads=6,
        num_actions=18,
    ).to(device)

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

    print(f"Loading decision checkpoint: {decision_ckpt_path}")
    decision_ckpt = torch.load(decision_ckpt_path, map_location=device, weights_only=False)
    decision_model.load_state_dict(decision_ckpt['model_state_dict'])

    world_ckpt_path = Path('outputs/world_model_pretrain/best_model.pt')
    if world_ckpt_path.exists():
        print(f"Loading world model checkpoint: {world_ckpt_path}")
        world_ckpt = torch.load(world_ckpt_path, map_location=device, weights_only=False)
        world_model.load_state_dict(world_ckpt['model_state_dict'])

    return decision_ckpt


def evaluate_with_prediction(bev_projection, world_model, decision_model, test_loader, device='cuda'):
    """
    Evaluate using world model's predicted future instead of ground truth.

    Pipeline:
    1. Current frame → BEV projection
    2. BEV → World Model → Encode current + Predict future
    3. Predicted future features → Decision Transformer → Action
    4. Compare predicted action vs ground truth action
    """

    bev_projection.eval()
    world_model.eval()
    decision_model.eval()

    all_preds_current = []  # Using current frame encoding
    all_preds_predicted = []  # Using predicted future
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating (Predictive)"):
            point_cloud = batch['point_cloud'].to(device)
            vehicle_state = batch['vehicle_state'].to(device)
            goal_direction = batch['goal_direction'].to(device)
            action_sequence = batch['action_sequence'].to(device)
            expert_action = batch['expert_action'].to(device)

            # 1. Project current point cloud to BEV
            current_bev = bev_projection(point_cloud)

            # 2. World model: encode current + predict future
            world_outputs = world_model(current_bev)

            # Get current encoding
            world_latent_current, world_global_current = world_model.encode(current_bev)

            # Get predicted future features
            # future shape: (B, num_steps, C, H, W) = (B, 5, 64, 16, 16)
            future_pred = world_outputs['future']  # Predicted future BEV features

            # Use first predicted future step
            # Reshape and upsample to match expected dimensions
            B = future_pred.shape[0]
            future_step1 = future_pred[:, 0]  # (B, 64, 16, 16)

            # Upsample to full BEV size for re-encoding
            future_bev = F.interpolate(future_step1, size=(256, 256), mode='bilinear', align_corners=False)

            # Encode the predicted future
            world_latent_predicted, world_global_predicted = world_model.encode(future_bev)

            # 3a. Decision from CURRENT frame (baseline)
            outputs_current = decision_model(
                world_global=world_global_current,
                world_latent=world_latent_current,
                state=vehicle_state,
                goal=goal_direction,
                action_history=action_sequence
            )
            preds_current = torch.argmax(outputs_current['action_logits'], dim=-1)

            # 3b. Decision from PREDICTED future frame
            outputs_predicted = decision_model(
                world_global=world_global_predicted,
                world_latent=world_latent_predicted,
                state=vehicle_state,
                goal=goal_direction,
                action_history=action_sequence
            )
            preds_predicted = torch.argmax(outputs_predicted['action_logits'], dim=-1)

            # Collect results
            all_preds_current.extend(preds_current.cpu().numpy())
            all_preds_predicted.extend(preds_predicted.cpu().numpy())
            all_targets.extend(expert_action.cpu().numpy())

    return (np.array(all_preds_current),
            np.array(all_preds_predicted),
            np.array(all_targets))


def compute_metrics(preds, targets):
    """Compute evaluation metrics"""

    accuracy = accuracy_score(targets, preds)

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        targets, preds, average='macro', zero_division=0
    )

    cm = confusion_matrix(targets, preds)

    return {
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'confusion_matrix': cm
    }


def plot_comparison(metrics_current, metrics_predicted, output_path):
    """Plot comparison bar chart"""

    labels = ['Accuracy', 'Precision', 'Recall', 'F1']
    current_vals = [
        metrics_current['accuracy'],
        metrics_current['precision_macro'],
        metrics_current['recall_macro'],
        metrics_current['f1_macro']
    ]
    predicted_vals = [
        metrics_predicted['accuracy'],
        metrics_predicted['precision_macro'],
        metrics_predicted['recall_macro'],
        metrics_predicted['f1_macro']
    ]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, current_vals, width, label='Current Frame', color='steelblue')
    bars2 = ax.bar(x + width/2, predicted_vals, width, label='Predicted Frame', color='coral')

    ax.set_ylabel('Score')
    ax.set_title('Decision Accuracy: Current Frame vs Predicted Future Frame')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_ylim(0, 1)

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"\nComparison plot saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='outputs/decision_train/best_model.pt')
    parser.add_argument('--data', type=str, default='../RELLIS/bin/')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=16)
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")

    # Create models
    print("Creating models...")
    bev_projection, world_model, decision_model = create_models(device)

    # Load checkpoints
    print("\nLoading checkpoints...")
    ckpt = load_checkpoints(world_model, decision_model, args.checkpoint, device)
    print(f"Checkpoint info: Epoch {ckpt.get('epoch')}, Accuracy {ckpt.get('best_accuracy'):.4f}")

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
    preds_current, preds_predicted, targets = evaluate_with_prediction(
        bev_projection, world_model, decision_model, test_loader, device
    )

    # Compute metrics
    metrics_current = compute_metrics(preds_current, targets)
    metrics_predicted = compute_metrics(preds_predicted, targets)

    # Build results
    output_dir = Path('outputs/decision_train')
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    results.append("=" * 70)
    results.append("PREDICTIVE EVALUATION RESULTS")
    results.append("=" * 70)
    results.append("")
    results.append("[1] Using CURRENT FRAME (baseline):")
    results.append(f"    Accuracy:  {metrics_current['accuracy']:.4f} ({metrics_current['accuracy']*100:.2f}%)")
    results.append(f"    Precision: {metrics_current['precision_macro']:.4f}")
    results.append(f"    Recall:    {metrics_current['recall_macro']:.4f}")
    results.append(f"    F1 Score:  {metrics_current['f1_macro']:.4f}")
    results.append("")
    results.append("[2] Using PREDICTED FUTURE FRAME (world model prediction):")
    results.append(f"    Accuracy:  {metrics_predicted['accuracy']:.4f} ({metrics_predicted['accuracy']*100:.2f}%)")
    results.append(f"    Precision: {metrics_predicted['precision_macro']:.4f}")
    results.append(f"    Recall:    {metrics_predicted['recall_macro']:.4f}")
    results.append(f"    F1 Score:  {metrics_predicted['f1_macro']:.4f}")

    # Calculate degradation
    acc_diff = metrics_current['accuracy'] - metrics_predicted['accuracy']
    results.append("")
    results.append("-" * 70)
    results.append(f"Accuracy Difference (Current - Predicted): {acc_diff:.4f} ({acc_diff*100:.2f}%)")
    if acc_diff > 0:
        results.append(f"  -> Prediction quality causes {acc_diff*100:.1f}% accuracy drop")
    else:
        results.append(f"  -> Predicted future actually helps! ({abs(acc_diff)*100:.1f}% improvement)")

    # Agreement between methods
    agreement = np.mean(preds_current == preds_predicted)
    results.append(f"")
    results.append(f"Prediction Agreement (same action chosen): {agreement:.4f} ({agreement*100:.2f}%)")
    results.append("=" * 70)

    # Print results
    for line in results:
        print(line)

    # Save to file
    results_file = output_dir / 'predictive_results.txt'
    with open(results_file, 'w') as f:
        f.write('\n'.join(results))
    print(f"\nResults saved to: {results_file}")

    # Plot comparison
    plot_comparison(metrics_current, metrics_predicted, output_dir / 'predictive_comparison.png')

    print("\nEvaluation complete!")


if __name__ == '__main__':
    main()
