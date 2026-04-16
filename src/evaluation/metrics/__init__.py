"""Evaluation Metrics"""

import torch
import numpy as np
from typing import Dict
from sklearn.metrics import confusion_matrix, classification_report


def compute_metrics(predictions: np.ndarray, targets: np.ndarray, num_classes: int = 18) -> Dict:
    """
    Compute evaluation metrics.
    
    Args:
        predictions: Predicted action IDs
        targets: Ground truth action IDs
        num_classes: Number of action classes
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Overall accuracy
    metrics['accuracy'] = (predictions == targets).mean()
    
    # Top-3 accuracy (if we have logits)
    # metrics['top3_accuracy'] = ...
    
    # Per-class accuracy
    for cls in range(num_classes):
        mask = targets == cls
        if mask.sum() > 0:
            metrics[f'accuracy_class_{cls}'] = (predictions[mask] == cls).mean()
            
    # Confusion matrix
    cm = confusion_matrix(targets, predictions, labels=range(num_classes))
    metrics['confusion_matrix'] = cm
    
    # Classification report
    report = classification_report(targets, predictions, output_dict=True, zero_division=0)
    metrics['precision_macro'] = report['macro avg']['precision']
    metrics['recall_macro'] = report['macro avg']['recall']
    metrics['f1_macro'] = report['macro avg']['f1-score']
    
    return metrics


def compute_trajectory_metrics(predicted_trajectory: np.ndarray, 
                               target_trajectory: np.ndarray) -> Dict:
    """Compute trajectory quality metrics."""
    # ADE (Average Displacement Error)
    ade = np.mean(np.linalg.norm(predicted_trajectory - target_trajectory, axis=-1))
    
    # FDE (Final Displacement Error)
    fde = np.linalg.norm(predicted_trajectory[-1] - target_trajectory[-1])
    
    return {'ade': ade, 'fde': fde}
