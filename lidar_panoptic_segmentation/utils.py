"""
Utility Functions for LiDAR Panoptic Segmentation

Provides common utilities for data processing, I/O, and visualization.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


# ==============================================================================
# Random Seed Utilities
# ==============================================================================

def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


# ==============================================================================
# File I/O Utilities
# ==============================================================================

def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, creating if necessary.

    Args:
        path: Directory path

    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_file_hash(path: Union[str, Path], algorithm: str = "md5") -> str:
    """
    Compute hash of file contents.

    Args:
        path: File path
        algorithm: Hash algorithm (md5, sha256, etc.)

    Returns:
        Hexadecimal hash string
    """
    hasher = hashlib.new(algorithm)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def atomic_write(path: Union[str, Path], content: Union[str, bytes]) -> None:
    """
    Write file atomically using temporary file and rename.

    Args:
        path: Target file path
        content: Content to write
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    mode = "wb" if isinstance(content, bytes) else "w"

    with tempfile.NamedTemporaryFile(
        mode=mode,
        dir=path.parent,
        delete=False,
    ) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    shutil.move(tmp_path, path)


def load_json(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load JSON file.

    Args:
        path: JSON file path

    Returns:
        Parsed JSON data
    """
    with open(path, "r") as f:
        return json.load(f)


def save_json(
    data: Any,
    path: Union[str, Path],
    indent: int = 2,
) -> None:
    """
    Save data to JSON file.

    Args:
        data: Data to save
        path: Output path
        indent: Indentation level
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(data, f, indent=indent, default=str)


# ==============================================================================
# Numpy Utilities
# ==============================================================================

def normalize_points(
    points: np.ndarray,
    center: bool = True,
    scale: bool = True,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Normalize point cloud coordinates.

    Args:
        points: (N, 3) point coordinates
        center: Center to origin
        scale: Scale to unit sphere

    Returns:
        Tuple of (normalized_points, normalization_params)
    """
    params = {}

    if center:
        centroid = points.mean(axis=0)
        points = points - centroid
        params["centroid"] = centroid

    if scale:
        max_dist = np.linalg.norm(points, axis=1).max()
        if max_dist > 0:
            points = points / max_dist
        params["scale"] = max_dist

    return points, params


def denormalize_points(
    points: np.ndarray,
    params: Dict[str, np.ndarray],
) -> np.ndarray:
    """
    Denormalize point cloud coordinates.

    Args:
        points: (N, 3) normalized coordinates
        params: Normalization parameters from normalize_points

    Returns:
        Denormalized points
    """
    if "scale" in params:
        points = points * params["scale"]

    if "centroid" in params:
        points = points + params["centroid"]

    return points


def random_subsample(
    points: np.ndarray,
    n_samples: int,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Randomly subsample points.

    Args:
        points: (N, D) points
        n_samples: Number of samples
        seed: Random seed

    Returns:
        Tuple of (sampled_points, indices)
    """
    if seed is not None:
        np.random.seed(seed)

    n_points = len(points)

    if n_samples >= n_points:
        indices = np.arange(n_points)
    else:
        indices = np.random.choice(n_points, n_samples, replace=False)

    return points[indices], indices


def farthest_point_sampling(
    points: np.ndarray,
    n_samples: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Farthest point sampling for uniform coverage.

    Args:
        points: (N, 3) point coordinates
        n_samples: Number of samples

    Returns:
        Tuple of (sampled_points, indices)
    """
    n_points = len(points)
    if n_samples >= n_points:
        return points, np.arange(n_points)

    indices = np.zeros(n_samples, dtype=np.int64)
    distances = np.full(n_points, np.inf)

    # Start with random point
    indices[0] = np.random.randint(n_points)

    for i in range(1, n_samples):
        last_point = points[indices[i - 1]]
        dist_to_last = np.linalg.norm(points - last_point, axis=1)
        distances = np.minimum(distances, dist_to_last)
        indices[i] = np.argmax(distances)

    return points[indices], indices


# ==============================================================================
# Point Cloud Utilities
# ==============================================================================

def compute_normals(
    points: np.ndarray,
    k_neighbors: int = 30,
) -> np.ndarray:
    """
    Compute point cloud normals using PCA.

    Args:
        points: (N, 3) point coordinates
        k_neighbors: Number of neighbors for normal estimation

    Returns:
        (N, 3) normal vectors
    """
    from scipy.spatial import cKDTree

    tree = cKDTree(points)
    _, indices = tree.query(points, k=k_neighbors)

    normals = np.zeros_like(points)

    for i, neighbors in enumerate(indices):
        neighbor_points = points[neighbors]
        centered = neighbor_points - neighbor_points.mean(axis=0)

        # PCA via SVD
        _, _, vh = np.linalg.svd(centered)
        normal = vh[-1]

        # Orient normals consistently (pointing up)
        if normal[2] < 0:
            normal = -normal

        normals[i] = normal

    return normals


def compute_height_above_ground(
    points: np.ndarray,
    ground_percentile: float = 5.0,
) -> np.ndarray:
    """
    Compute height above ground for each point.

    Args:
        points: (N, 3) point coordinates
        ground_percentile: Percentile for ground estimation

    Returns:
        (N,) height values
    """
    ground_z = np.percentile(points[:, 2], ground_percentile)
    return points[:, 2] - ground_z


def crop_point_cloud(
    points: np.ndarray,
    bounds: np.ndarray,
    features: Optional[Dict[str, np.ndarray]] = None,
) -> Tuple[np.ndarray, Optional[Dict[str, np.ndarray]], np.ndarray]:
    """
    Crop point cloud to bounding box.

    Args:
        points: (N, 3) point coordinates
        bounds: (2, 3) min/max bounds
        features: Optional dictionary of features

    Returns:
        Tuple of (cropped_points, cropped_features, mask)
    """
    mask = (
        (points[:, 0] >= bounds[0, 0]) &
        (points[:, 0] <= bounds[1, 0]) &
        (points[:, 1] >= bounds[0, 1]) &
        (points[:, 1] <= bounds[1, 1]) &
        (points[:, 2] >= bounds[0, 2]) &
        (points[:, 2] <= bounds[1, 2])
    )

    cropped_features = None
    if features is not None:
        cropped_features = {k: v[mask] for k, v in features.items() if v is not None}

    return points[mask], cropped_features, mask


# ==============================================================================
# Metrics Utilities
# ==============================================================================

def compute_iou(pred: np.ndarray, target: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Compute per-class Intersection over Union.

    Args:
        pred: (N,) predicted labels
        target: (N,) ground truth labels
        num_classes: Number of classes

    Returns:
        (num_classes,) IoU values
    """
    ious = np.zeros(num_classes)

    for c in range(num_classes):
        pred_c = pred == c
        target_c = target == c

        intersection = (pred_c & target_c).sum()
        union = (pred_c | target_c).sum()

        if union > 0:
            ious[c] = intersection / union
        else:
            ious[c] = float("nan")

    return ious


def compute_instance_metrics(
    pred_instances: np.ndarray,
    gt_instances: np.ndarray,
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute instance segmentation metrics.

    Args:
        pred_instances: (N,) predicted instance IDs
        gt_instances: (N,) ground truth instance IDs
        iou_threshold: IoU threshold for matching

    Returns:
        Dictionary of metrics (precision, recall, f1)
    """
    pred_ids = np.unique(pred_instances)
    pred_ids = pred_ids[pred_ids >= 0]

    gt_ids = np.unique(gt_instances)
    gt_ids = gt_ids[gt_ids >= 0]

    # Compute IoU matrix
    iou_matrix = np.zeros((len(pred_ids), len(gt_ids)))

    for i, pred_id in enumerate(pred_ids):
        pred_mask = pred_instances == pred_id
        for j, gt_id in enumerate(gt_ids):
            gt_mask = gt_instances == gt_id

            intersection = (pred_mask & gt_mask).sum()
            union = (pred_mask | gt_mask).sum()

            if union > 0:
                iou_matrix[i, j] = intersection / union

    # Match instances
    n_matched = 0
    matched_gt = set()

    for i in range(len(pred_ids)):
        best_j = iou_matrix[i].argmax()
        if iou_matrix[i, best_j] >= iou_threshold and best_j not in matched_gt:
            n_matched += 1
            matched_gt.add(best_j)

    precision = n_matched / len(pred_ids) if len(pred_ids) > 0 else 0
    recall = n_matched / len(gt_ids) if len(gt_ids) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n_pred": len(pred_ids),
        "n_gt": len(gt_ids),
        "n_matched": n_matched,
    }


# ==============================================================================
# Timing Utilities
# ==============================================================================

class Timer:
    """Context manager for timing code blocks."""

    def __init__(self, name: str = "", log: bool = True):
        self.name = name
        self.log = log
        self.start_time = None
        self.elapsed = None

    def __enter__(self):
        self.start_time = datetime.now()
        return self

    def __exit__(self, *args):
        self.elapsed = (datetime.now() - self.start_time).total_seconds()
        if self.log:
            logger.info(f"{self.name}: {self.elapsed:.2f}s")


# ==============================================================================
# Environment Utilities
# ==============================================================================

def is_databricks() -> bool:
    """Check if running on Databricks."""
    return (
        "DATABRICKS_RUNTIME_VERSION" in os.environ or
        os.path.exists("/databricks")
    )


def get_gpu_memory_usage() -> Dict[str, float]:
    """
    Get GPU memory usage.

    Returns:
        Dictionary with memory stats in GB
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return {}

        device = torch.cuda.current_device()
        return {
            "allocated": torch.cuda.memory_allocated(device) / 1e9,
            "cached": torch.cuda.memory_reserved(device) / 1e9,
            "max_allocated": torch.cuda.max_memory_allocated(device) / 1e9,
        }
    except Exception:
        return {}


def clear_gpu_cache() -> None:
    """Clear GPU memory cache."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


# ==============================================================================
# Visualization Utilities
# ==============================================================================

def get_color_map(num_colors: int) -> np.ndarray:
    """
    Generate distinct colors for visualization.

    Args:
        num_colors: Number of colors needed

    Returns:
        (num_colors, 3) RGB colors in [0, 255]
    """
    import colorsys

    colors = []
    for i in range(num_colors):
        hue = i / num_colors
        rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
        colors.append([int(c * 255) for c in rgb])

    return np.array(colors, dtype=np.uint8)


def colorize_labels(
    labels: np.ndarray,
    color_map: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Convert labels to RGB colors.

    Args:
        labels: (N,) label array
        color_map: Optional (num_classes, 3) color map

    Returns:
        (N, 3) RGB colors
    """
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels >= 0]
    num_labels = len(unique_labels)

    if color_map is None:
        color_map = get_color_map(max(num_labels, labels.max() + 1))

    colors = np.zeros((len(labels), 3), dtype=np.uint8)
    colors[labels < 0] = [128, 128, 128]  # Gray for background

    for label in unique_labels:
        mask = labels == label
        colors[mask] = color_map[label % len(color_map)]

    return colors


# Convenience exports
__all__ = [
    # Seed
    "set_seed",
    # I/O
    "ensure_dir",
    "get_file_hash",
    "atomic_write",
    "load_json",
    "save_json",
    # Numpy
    "normalize_points",
    "denormalize_points",
    "random_subsample",
    "farthest_point_sampling",
    # Point cloud
    "compute_normals",
    "compute_height_above_ground",
    "crop_point_cloud",
    # Metrics
    "compute_iou",
    "compute_instance_metrics",
    # Timing
    "Timer",
    # Environment
    "is_databricks",
    "get_gpu_memory_usage",
    "clear_gpu_cache",
    # Visualization
    "get_color_map",
    "colorize_labels",
]
