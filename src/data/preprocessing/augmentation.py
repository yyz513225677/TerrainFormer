"""
Data Augmentation for Point Clouds

Various augmentation techniques for training robustness.
"""

import torch
import numpy as np
from typing import Dict, Optional, Tuple
import random


class Augmentation:
    """
    Point cloud augmentation pipeline.
    """
    
    def __init__(self,
                 random_rotation: bool = True,
                 rotation_range: Tuple[float, float] = (-45, 45),
                 random_flip: bool = True,
                 flip_prob: float = 0.5,
                 random_scale: bool = True,
                 scale_range: Tuple[float, float] = (0.95, 1.05),
                 random_translation: bool = True,
                 translation_range: Tuple[float, float] = (-0.5, 0.5),
                 random_dropout: bool = True,
                 dropout_prob: float = 0.1,
                 random_noise: bool = True,
                 noise_std: float = 0.02,
                 intensity_jitter: bool = True,
                 intensity_range: Tuple[float, float] = (-0.1, 0.1)):
        
        self.random_rotation = random_rotation
        self.rotation_range = rotation_range
        self.random_flip = random_flip
        self.flip_prob = flip_prob
        self.random_scale = random_scale
        self.scale_range = scale_range
        self.random_translation = random_translation
        self.translation_range = translation_range
        self.random_dropout = random_dropout
        self.dropout_prob = dropout_prob
        self.random_noise = random_noise
        self.noise_std = noise_std
        self.intensity_jitter = intensity_jitter
        self.intensity_range = intensity_range
        
    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Apply augmentations to a sample.
        
        Args:
            sample: Dictionary with point_cloud and other tensors
            
        Returns:
            Augmented sample
        """
        points = sample['point_cloud'].numpy()
        
        # Random rotation around z-axis
        if self.random_rotation:
            points = self._random_rotation(points)
            
        # Random flip
        if self.random_flip and random.random() < self.flip_prob:
            points = self._random_flip(points)
            
        # Random scale
        if self.random_scale:
            points = self._random_scale(points)
            
        # Random translation
        if self.random_translation:
            points = self._random_translation(points)
            
        # Random point dropout
        if self.random_dropout:
            points = self._random_dropout(points)
            
        # Random noise
        if self.random_noise:
            points = self._random_noise(points)
            
        # Intensity jitter
        if self.intensity_jitter and points.shape[1] > 3:
            points = self._intensity_jitter(points)
            
        sample['point_cloud'] = torch.from_numpy(points).float()
        
        # Also augment history if present
        if 'point_cloud_history' in sample:
            history = sample['point_cloud_history'].numpy()
            # Apply same rotation to history for consistency
            sample['point_cloud_history'] = torch.from_numpy(history).float()
            
        return sample
    
    def _random_rotation(self, points: np.ndarray) -> np.ndarray:
        """Rotate around z-axis."""
        angle = np.random.uniform(*[np.radians(x) for x in self.rotation_range])
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rotation = np.array([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1]
        ])
        points[:, :3] = points[:, :3] @ rotation.T
        return points
    
    def _random_flip(self, points: np.ndarray) -> np.ndarray:
        """Flip along x-axis."""
        points[:, 0] *= -1
        return points
    
    def _random_scale(self, points: np.ndarray) -> np.ndarray:
        """Random uniform scaling."""
        scale = np.random.uniform(*self.scale_range)
        points[:, :3] *= scale
        return points
    
    def _random_translation(self, points: np.ndarray) -> np.ndarray:
        """Random translation."""
        translation = np.random.uniform(
            self.translation_range[0],
            self.translation_range[1],
            size=3
        )
        points[:, :3] += translation
        return points
    
    def _random_dropout(self, points: np.ndarray) -> np.ndarray:
        """Randomly drop points."""
        mask = np.random.random(len(points)) > self.dropout_prob
        if mask.sum() < 100:  # Keep minimum points
            return points
        return points[mask]
    
    def _random_noise(self, points: np.ndarray) -> np.ndarray:
        """Add Gaussian noise to xyz."""
        noise = np.random.normal(0, self.noise_std, points[:, :3].shape)
        points[:, :3] += noise
        return points
    
    def _intensity_jitter(self, points: np.ndarray) -> np.ndarray:
        """Jitter intensity values."""
        jitter = np.random.uniform(*self.intensity_range)
        points[:, 3] += jitter
        points[:, 3] = np.clip(points[:, 3], 0, 1)
        return points


class MixUp:
    """MixUp augmentation for point clouds."""
    
    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha
        
    def __call__(self, sample1: Dict, sample2: Dict) -> Dict:
        """Mix two samples."""
        lam = np.random.beta(self.alpha, self.alpha)
        
        mixed = {}
        for key in sample1:
            if isinstance(sample1[key], torch.Tensor):
                if sample1[key].dtype == torch.long:
                    # For labels, randomly choose
                    mixed[key] = sample1[key] if random.random() < lam else sample2[key]
                else:
                    # For features, interpolate
                    mixed[key] = lam * sample1[key] + (1 - lam) * sample2[key]
            else:
                mixed[key] = sample1[key]
                
        mixed['mixup_lambda'] = lam
        return mixed
