"""Geometry utilities"""

import numpy as np
import torch


def rotation_matrix_z(angle: float) -> np.ndarray:
    """Create rotation matrix around z-axis."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def transform_points(points: np.ndarray, transformation: np.ndarray) -> np.ndarray:
    """Apply 4x4 transformation to points."""
    ones = np.ones((points.shape[0], 1))
    points_h = np.hstack([points[:, :3], ones])
    transformed = (transformation @ points_h.T).T
    result = np.hstack([transformed[:, :3], points[:, 3:]])
    return result
