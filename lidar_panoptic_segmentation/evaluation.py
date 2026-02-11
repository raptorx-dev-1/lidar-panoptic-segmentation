"""
Evaluation Module for LiDAR Panoptic Segmentation

Provides comprehensive metrics for semantic segmentation, instance segmentation,
and panoptic segmentation evaluation, including:

- Semantic IoU (Intersection over Union)
- Confusion Matrix
- Instance Segmentation: Precision, Recall, F1 Score
- Panoptic Quality (PQ), Segmentation Quality (SQ), Recognition Quality (RQ)
- Polygon-based evaluation (IoU between predicted and ground truth polygons)
- Coverage metrics (MUCov, MWCov)
- Multi-threshold AP (Average Precision) and mAP
- Segmentation error analysis (USE, OSE, Boundary IoU)
- Tree-level geometric metrics (height, location, crown diameter)
- Forestry metrics (commission, omission, detection by size)

Reference:
    Kirillov et al., "Panoptic Segmentation", CVPR 2019
    https://arxiv.org/abs/1801.00868
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes for Results
# =============================================================================

@dataclass
class SemanticMetrics:
    """Semantic segmentation metrics."""

    # Per-class metrics
    class_names: List[str]
    iou_per_class: np.ndarray  # (C,) IoU per class
    accuracy_per_class: np.ndarray  # (C,) accuracy per class
    precision_per_class: np.ndarray  # (C,) precision per class
    recall_per_class: np.ndarray  # (C,) recall per class

    # Aggregate metrics
    miou: float  # Mean IoU
    overall_accuracy: float
    mean_accuracy: float

    # Confusion matrix
    confusion_matrix: np.ndarray  # (C, C) confusion matrix


@dataclass
class InstanceMetrics:
    """Instance segmentation metrics."""

    # Detection metrics
    precision: float
    recall: float
    f1_score: float

    # True/False positives
    true_positives: int
    false_positives: int
    false_negatives: int

    # Coverage metrics
    mean_coverage: float  # MUCov
    weighted_mean_coverage: float  # MWCov

    # IoU threshold used
    iou_threshold: float


@dataclass
class PanopticMetrics:
    """Panoptic segmentation metrics."""

    # Per-class PQ, SQ, RQ
    pq_per_class: np.ndarray  # (C,) PQ per class
    sq_per_class: np.ndarray  # (C,) SQ per class
    rq_per_class: np.ndarray  # (C,) RQ per class

    # Aggregate metrics
    pq: float  # Panoptic Quality
    sq: float  # Segmentation Quality
    rq: float  # Recognition Quality

    # Thing vs Stuff breakdown
    pq_things: float
    pq_stuff: float

    # Class names
    class_names: List[str]
    thing_classes: List[int]
    stuff_classes: List[int]


@dataclass
class DetectionMetrics:
    """Detection metrics at multiple IoU thresholds."""

    # Average Precision at different thresholds
    ap_25: float  # AP at IoU >= 0.25
    ap_50: float  # AP at IoU >= 0.50
    ap_75: float  # AP at IoU >= 0.75
    map: float  # Mean AP across thresholds

    # Precision-Recall data for curves
    precisions: np.ndarray  # Precision values
    recalls: np.ndarray  # Recall values
    iou_thresholds: np.ndarray  # IoU thresholds used

    # F1 at different thresholds
    f1_25: float
    f1_50: float
    f1_75: float


@dataclass
class SegmentationErrorMetrics:
    """Segmentation error analysis metrics."""

    # Over/Under segmentation
    over_segmentation_error: float  # OSE - trees split into multiple
    under_segmentation_error: float  # USE - multiple trees merged
    segmentation_accuracy: float  # Achievable Segmentation Accuracy

    # Boundary metrics
    boundary_iou: float  # IoU focused on boundaries
    boundary_precision: float
    boundary_recall: float

    # Per-instance errors
    split_instances: int  # GT instances split into multiple pred
    merged_instances: int  # Multiple GT merged into one pred


@dataclass
class TreeGeometricMetrics:
    """Tree-level geometric accuracy metrics."""

    # Height metrics
    height_rmse: float  # Root Mean Square Error
    height_mae: float  # Mean Absolute Error
    height_bias: float  # Systematic bias
    height_r2: float  # Correlation coefficient

    # Location metrics
    location_rmse: float  # Centroid distance RMSE
    location_mae: float  # Centroid distance MAE

    # Crown diameter metrics
    crown_diameter_rmse: float
    crown_diameter_mae: float

    # Count metrics
    count_error: int  # Predicted - GT count
    count_error_pct: float  # Percentage error

    # Per-tree data for analysis
    gt_heights: np.ndarray
    pred_heights: np.ndarray
    height_errors: np.ndarray


@dataclass
class ForestryMetrics:
    """Forestry-specific evaluation metrics."""

    # Detection rates
    detection_rate: float  # Overall detection rate
    commission_rate: float  # False positive rate
    omission_rate: float  # False negative rate

    # Detection by tree size
    detection_rate_small: float  # Trees < 5m
    detection_rate_medium: float  # Trees 5-15m
    detection_rate_large: float  # Trees > 15m

    # Point-level forestry metrics
    tree_point_accuracy: float  # Accuracy on tree points only
    ground_point_accuracy: float  # Accuracy on ground points

    # Canopy metrics
    canopy_cover_gt: float  # Ground truth canopy cover %
    canopy_cover_pred: float  # Predicted canopy cover %
    canopy_cover_error: float  # Absolute error


@dataclass
class EvaluationResult:
    """Complete evaluation result."""

    semantic: SemanticMetrics
    instance: InstanceMetrics
    panoptic: PanopticMetrics

    # Metadata
    num_points: int
    num_gt_instances: int
    num_pred_instances: int

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 60,
            "EVALUATION RESULTS",
            "=" * 60,
            "",
            "Semantic Segmentation:",
            f"  mIoU:              {self.semantic.miou:.4f}",
            f"  Overall Accuracy:  {self.semantic.overall_accuracy:.4f}",
            f"  Mean Accuracy:     {self.semantic.mean_accuracy:.4f}",
            "",
            "  Per-class IoU:",
        ]

        for name, iou in zip(self.semantic.class_names, self.semantic.iou_per_class):
            lines.append(f"    {name}: {iou:.4f}")

        lines.extend([
            "",
            "Instance Segmentation:",
            f"  Precision:         {self.instance.precision:.4f}",
            f"  Recall:            {self.instance.recall:.4f}",
            f"  F1 Score:          {self.instance.f1_score:.4f}",
            f"  MUCov:             {self.instance.mean_coverage:.4f}",
            f"  MWCov:             {self.instance.weighted_mean_coverage:.4f}",
            f"  (IoU threshold:    {self.instance.iou_threshold:.2f})",
            "",
            "Panoptic Segmentation:",
            f"  PQ:                {self.panoptic.pq:.4f}",
            f"  SQ:                {self.panoptic.sq:.4f}",
            f"  RQ:                {self.panoptic.rq:.4f}",
            f"  PQ (things):       {self.panoptic.pq_things:.4f}",
            f"  PQ (stuff):        {self.panoptic.pq_stuff:.4f}",
            "",
            f"Statistics:",
            f"  Total points:      {self.num_points:,}",
            f"  GT instances:      {self.num_gt_instances}",
            f"  Pred instances:    {self.num_pred_instances}",
            "=" * 60,
        ])

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/saving."""
        return {
            "semantic": {
                "miou": self.semantic.miou,
                "overall_accuracy": self.semantic.overall_accuracy,
                "mean_accuracy": self.semantic.mean_accuracy,
                "iou_per_class": {
                    name: float(iou)
                    for name, iou in zip(
                        self.semantic.class_names, self.semantic.iou_per_class
                    )
                },
            },
            "instance": {
                "precision": self.instance.precision,
                "recall": self.instance.recall,
                "f1_score": self.instance.f1_score,
                "mean_coverage": self.instance.mean_coverage,
                "weighted_mean_coverage": self.instance.weighted_mean_coverage,
                "true_positives": self.instance.true_positives,
                "false_positives": self.instance.false_positives,
                "false_negatives": self.instance.false_negatives,
            },
            "panoptic": {
                "pq": self.panoptic.pq,
                "sq": self.panoptic.sq,
                "rq": self.panoptic.rq,
                "pq_things": self.panoptic.pq_things,
                "pq_stuff": self.panoptic.pq_stuff,
            },
            "statistics": {
                "num_points": self.num_points,
                "num_gt_instances": self.num_gt_instances,
                "num_pred_instances": self.num_pred_instances,
            },
        }


# =============================================================================
# Semantic Segmentation Evaluation
# =============================================================================

def compute_confusion_matrix(
    pred: np.ndarray,
    gt: np.ndarray,
    num_classes: int,
    ignore_label: int = -1,
) -> np.ndarray:
    """
    Compute confusion matrix.

    Args:
        pred: (N,) predicted labels
        gt: (N,) ground truth labels
        num_classes: Number of classes
        ignore_label: Label to ignore (e.g., unlabeled points)

    Returns:
        (C, C) confusion matrix where [i, j] = count of GT=i predicted as j
    """
    # Filter out ignored labels
    valid_mask = gt != ignore_label
    pred = pred[valid_mask]
    gt = gt[valid_mask]

    # Clamp to valid range
    pred = np.clip(pred, 0, num_classes - 1)
    gt = np.clip(gt, 0, num_classes - 1)

    # Compute confusion matrix
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for i in range(num_classes):
        gt_mask = gt == i
        for j in range(num_classes):
            cm[i, j] = np.sum((pred == j) & gt_mask)

    return cm


def compute_semantic_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    num_classes: int = 2,
    class_names: Optional[List[str]] = None,
    ignore_label: int = -1,
) -> SemanticMetrics:
    """
    Compute semantic segmentation metrics.

    Args:
        pred: (N,) predicted semantic labels
        gt: (N,) ground truth semantic labels
        num_classes: Number of classes
        class_names: Optional class names
        ignore_label: Label to ignore

    Returns:
        SemanticMetrics object
    """
    if class_names is None:
        class_names = ["non-tree", "tree"][:num_classes]

    # Compute confusion matrix
    cm = compute_confusion_matrix(pred, gt, num_classes, ignore_label)

    # Per-class metrics
    iou_per_class = np.zeros(num_classes)
    accuracy_per_class = np.zeros(num_classes)
    precision_per_class = np.zeros(num_classes)
    recall_per_class = np.zeros(num_classes)

    for i in range(num_classes):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp

        # IoU = TP / (TP + FP + FN)
        union = tp + fp + fn
        iou_per_class[i] = tp / union if union > 0 else 0.0

        # Accuracy = TP / (TP + FN)
        gt_count = tp + fn
        accuracy_per_class[i] = tp / gt_count if gt_count > 0 else 0.0

        # Precision = TP / (TP + FP)
        pred_count = tp + fp
        precision_per_class[i] = tp / pred_count if pred_count > 0 else 0.0

        # Recall = TP / (TP + FN)
        recall_per_class[i] = tp / gt_count if gt_count > 0 else 0.0

    # Aggregate metrics
    miou = np.mean(iou_per_class)
    overall_accuracy = np.trace(cm) / np.sum(cm) if np.sum(cm) > 0 else 0.0
    mean_accuracy = np.mean(accuracy_per_class)

    return SemanticMetrics(
        class_names=class_names,
        iou_per_class=iou_per_class,
        accuracy_per_class=accuracy_per_class,
        precision_per_class=precision_per_class,
        recall_per_class=recall_per_class,
        miou=miou,
        overall_accuracy=overall_accuracy,
        mean_accuracy=mean_accuracy,
        confusion_matrix=cm,
    )


# =============================================================================
# Instance Segmentation Evaluation
# =============================================================================

def compute_instance_iou(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
) -> float:
    """
    Compute IoU between two instance masks.

    Args:
        pred_mask: (N,) boolean mask for predicted instance
        gt_mask: (N,) boolean mask for ground truth instance

    Returns:
        IoU value
    """
    intersection = np.sum(pred_mask & gt_mask)
    union = np.sum(pred_mask | gt_mask)
    return intersection / union if union > 0 else 0.0


def compute_instance_metrics(
    pred_instances: np.ndarray,
    gt_instances: np.ndarray,
    semantic_pred: Optional[np.ndarray] = None,
    semantic_gt: Optional[np.ndarray] = None,
    iou_threshold: float = 0.5,
    tree_class: int = 1,
) -> InstanceMetrics:
    """
    Compute instance segmentation metrics.

    Args:
        pred_instances: (N,) predicted instance IDs (-1 for background)
        gt_instances: (N,) ground truth instance IDs (-1 for background)
        semantic_pred: Optional (N,) predicted semantic labels
        semantic_gt: Optional (N,) ground truth semantic labels
        iou_threshold: IoU threshold for matching (default 0.5)
        tree_class: Semantic class for trees (default 1)

    Returns:
        InstanceMetrics object
    """
    # Get unique instance IDs
    pred_ids = np.unique(pred_instances[pred_instances >= 0])
    gt_ids = np.unique(gt_instances[gt_instances >= 0])

    n_pred = len(pred_ids)
    n_gt = len(gt_ids)

    if n_gt == 0:
        logger.warning("No ground truth instances found")
        return InstanceMetrics(
            precision=1.0 if n_pred == 0 else 0.0,
            recall=1.0,
            f1_score=1.0 if n_pred == 0 else 0.0,
            true_positives=0,
            false_positives=n_pred,
            false_negatives=0,
            mean_coverage=0.0,
            weighted_mean_coverage=0.0,
            iou_threshold=iou_threshold,
        )

    if n_pred == 0:
        return InstanceMetrics(
            precision=0.0,
            recall=0.0,
            f1_score=0.0,
            true_positives=0,
            false_positives=0,
            false_negatives=n_gt,
            mean_coverage=0.0,
            weighted_mean_coverage=0.0,
            iou_threshold=iou_threshold,
        )

    # Compute IoU matrix between all pred and gt instances
    iou_matrix = np.zeros((n_pred, n_gt))

    for i, pred_id in enumerate(pred_ids):
        pred_mask = pred_instances == pred_id
        for j, gt_id in enumerate(gt_ids):
            gt_mask = gt_instances == gt_id
            iou_matrix[i, j] = compute_instance_iou(pred_mask, gt_mask)

    # Match instances using greedy matching
    matched_pred = set()
    matched_gt = set()
    tp = 0
    iou_sum = 0.0

    # Sort by IoU descending
    iou_flat = iou_matrix.flatten()
    sorted_indices = np.argsort(-iou_flat)

    for idx in sorted_indices:
        i = idx // n_gt
        j = idx % n_gt
        iou = iou_matrix[i, j]

        if iou < iou_threshold:
            break

        if i not in matched_pred and j not in matched_gt:
            matched_pred.add(i)
            matched_gt.add(j)
            tp += 1
            iou_sum += iou

    fp = n_pred - len(matched_pred)
    fn = n_gt - len(matched_gt)

    # Precision, Recall, F1
    precision = tp / n_pred if n_pred > 0 else 0.0
    recall = tp / n_gt if n_gt > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Coverage metrics (MUCov, MWCov)
    coverage_per_gt = np.zeros(n_gt)
    gt_sizes = np.zeros(n_gt)

    for j, gt_id in enumerate(gt_ids):
        gt_mask = gt_instances == gt_id
        gt_sizes[j] = np.sum(gt_mask)

        # Find best covering prediction
        max_iou = 0.0
        for i, pred_id in enumerate(pred_ids):
            pred_mask = pred_instances == pred_id
            iou = compute_instance_iou(pred_mask, gt_mask)
            max_iou = max(max_iou, iou)
        coverage_per_gt[j] = max_iou

    mean_coverage = np.mean(coverage_per_gt)
    weighted_mean_coverage = (
        np.sum(coverage_per_gt * gt_sizes) / np.sum(gt_sizes)
        if np.sum(gt_sizes) > 0
        else 0.0
    )

    return InstanceMetrics(
        precision=precision,
        recall=recall,
        f1_score=f1_score,
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        mean_coverage=mean_coverage,
        weighted_mean_coverage=weighted_mean_coverage,
        iou_threshold=iou_threshold,
    )


# =============================================================================
# Panoptic Segmentation Evaluation
# =============================================================================

def compute_panoptic_metrics(
    pred_semantic: np.ndarray,
    gt_semantic: np.ndarray,
    pred_instances: np.ndarray,
    gt_instances: np.ndarray,
    num_classes: int = 2,
    class_names: Optional[List[str]] = None,
    thing_classes: Optional[List[int]] = None,
    stuff_classes: Optional[List[int]] = None,
    iou_threshold: float = 0.5,
    ignore_label: int = -1,
) -> PanopticMetrics:
    """
    Compute Panoptic Quality (PQ) and related metrics.

    PQ = SQ * RQ where:
    - SQ = mean IoU of matched segments
    - RQ = F1 score of segment matching

    Args:
        pred_semantic: (N,) predicted semantic labels
        gt_semantic: (N,) ground truth semantic labels
        pred_instances: (N,) predicted instance IDs
        gt_instances: (N,) ground truth instance IDs
        num_classes: Number of semantic classes
        class_names: Optional class names
        thing_classes: Classes that have instances (e.g., trees)
        stuff_classes: Classes without instances (e.g., ground)
        iou_threshold: IoU threshold for matching
        ignore_label: Label to ignore

    Returns:
        PanopticMetrics object
    """
    if class_names is None:
        class_names = ["non-tree", "tree"][:num_classes]

    if thing_classes is None:
        thing_classes = [1]  # Tree class has instances

    if stuff_classes is None:
        stuff_classes = [0]  # Non-tree is stuff

    # Initialize per-class metrics
    pq_per_class = np.zeros(num_classes)
    sq_per_class = np.zeros(num_classes)
    rq_per_class = np.zeros(num_classes)

    # Filter valid points
    valid_mask = gt_semantic != ignore_label

    # Evaluate stuff classes (semantic IoU)
    for cls in stuff_classes:
        if cls >= num_classes:
            continue

        pred_mask = (pred_semantic == cls) & valid_mask
        gt_mask = (gt_semantic == cls) & valid_mask

        intersection = np.sum(pred_mask & gt_mask)
        union = np.sum(pred_mask | gt_mask)

        if union > 0:
            iou = intersection / union
            if iou >= iou_threshold:
                sq_per_class[cls] = iou
                rq_per_class[cls] = 1.0
                pq_per_class[cls] = iou
            else:
                sq_per_class[cls] = 0.0
                rq_per_class[cls] = 0.0
                pq_per_class[cls] = 0.0

    # Evaluate thing classes (instance matching)
    for cls in thing_classes:
        if cls >= num_classes:
            continue

        # Get instances for this class
        cls_mask = (gt_semantic == cls) & valid_mask
        pred_cls_mask = (pred_semantic == cls) & valid_mask

        # Get instance IDs within this class
        gt_inst_ids = np.unique(gt_instances[cls_mask])
        gt_inst_ids = gt_inst_ids[gt_inst_ids >= 0]

        pred_inst_ids = np.unique(pred_instances[pred_cls_mask])
        pred_inst_ids = pred_inst_ids[pred_inst_ids >= 0]

        n_gt = len(gt_inst_ids)
        n_pred = len(pred_inst_ids)

        if n_gt == 0 and n_pred == 0:
            # No instances - perfect match
            pq_per_class[cls] = 1.0
            sq_per_class[cls] = 1.0
            rq_per_class[cls] = 1.0
            continue

        if n_gt == 0 or n_pred == 0:
            # One side empty
            pq_per_class[cls] = 0.0
            sq_per_class[cls] = 0.0
            rq_per_class[cls] = 0.0
            continue

        # Compute IoU matrix
        iou_matrix = np.zeros((n_pred, n_gt))

        for i, pred_id in enumerate(pred_inst_ids):
            pred_mask = pred_instances == pred_id
            for j, gt_id in enumerate(gt_inst_ids):
                gt_mask = gt_instances == gt_id
                iou_matrix[i, j] = compute_instance_iou(pred_mask, gt_mask)

        # Match instances (greedy)
        matched_pred = set()
        matched_gt = set()
        tp = 0
        iou_sum = 0.0

        iou_flat = iou_matrix.flatten()
        sorted_indices = np.argsort(-iou_flat)

        for idx in sorted_indices:
            i = idx // n_gt
            j = idx % n_gt
            iou = iou_matrix[i, j]

            if iou < iou_threshold:
                break

            if i not in matched_pred and j not in matched_gt:
                matched_pred.add(i)
                matched_gt.add(j)
                tp += 1
                iou_sum += iou

        fp = n_pred - len(matched_pred)
        fn = n_gt - len(matched_gt)

        # Compute PQ, SQ, RQ for this class
        if tp > 0:
            sq_per_class[cls] = iou_sum / tp
        else:
            sq_per_class[cls] = 0.0

        if tp + fp / 2 + fn / 2 > 0:
            rq_per_class[cls] = tp / (tp + fp / 2 + fn / 2)
        else:
            rq_per_class[cls] = 0.0

        pq_per_class[cls] = sq_per_class[cls] * rq_per_class[cls]

    # Aggregate metrics
    # Mean PQ/SQ/RQ over all classes with ground truth
    valid_classes = []
    for i in range(num_classes):
        # Class has ground truth points
        if np.sum((gt_semantic == i) & valid_mask) > 0:
            valid_classes.append(i)

    if len(valid_classes) > 0:
        pq = np.mean(pq_per_class[valid_classes])
        sq = np.mean(sq_per_class[valid_classes])
        rq = np.mean(rq_per_class[valid_classes])
    else:
        pq = sq = rq = 0.0

    # PQ for things and stuff separately
    thing_pq = (
        np.mean(pq_per_class[thing_classes])
        if len(thing_classes) > 0
        else 0.0
    )
    stuff_pq = (
        np.mean(pq_per_class[stuff_classes])
        if len(stuff_classes) > 0
        else 0.0
    )

    return PanopticMetrics(
        pq_per_class=pq_per_class,
        sq_per_class=sq_per_class,
        rq_per_class=rq_per_class,
        pq=pq,
        sq=sq,
        rq=rq,
        pq_things=thing_pq,
        pq_stuff=stuff_pq,
        class_names=class_names,
        thing_classes=thing_classes,
        stuff_classes=stuff_classes,
    )


# =============================================================================
# Complete Evaluation
# =============================================================================

def evaluate(
    pred_semantic: np.ndarray,
    gt_semantic: np.ndarray,
    pred_instances: np.ndarray,
    gt_instances: np.ndarray,
    num_classes: int = 2,
    class_names: Optional[List[str]] = None,
    thing_classes: Optional[List[int]] = None,
    stuff_classes: Optional[List[int]] = None,
    iou_threshold: float = 0.5,
    ignore_label: int = -1,
) -> EvaluationResult:
    """
    Run complete evaluation.

    Args:
        pred_semantic: (N,) predicted semantic labels
        gt_semantic: (N,) ground truth semantic labels
        pred_instances: (N,) predicted instance IDs
        gt_instances: (N,) ground truth instance IDs
        num_classes: Number of semantic classes
        class_names: Optional class names
        thing_classes: Classes with instances (default: [1] for trees)
        stuff_classes: Classes without instances (default: [0] for non-trees)
        iou_threshold: IoU threshold for instance matching
        ignore_label: Label to ignore

    Returns:
        EvaluationResult with all metrics
    """
    if class_names is None:
        class_names = ["non-tree", "tree"][:num_classes]

    if thing_classes is None:
        thing_classes = [1]

    if stuff_classes is None:
        stuff_classes = [0]

    # Semantic metrics
    semantic = compute_semantic_metrics(
        pred=pred_semantic,
        gt=gt_semantic,
        num_classes=num_classes,
        class_names=class_names,
        ignore_label=ignore_label,
    )

    # Instance metrics (for thing classes only)
    # Filter to thing class points
    thing_mask = np.isin(gt_semantic, thing_classes) & (gt_semantic != ignore_label)

    if thing_mask.any():
        instance = compute_instance_metrics(
            pred_instances=pred_instances[thing_mask],
            gt_instances=gt_instances[thing_mask],
            iou_threshold=iou_threshold,
        )
    else:
        instance = InstanceMetrics(
            precision=0.0,
            recall=0.0,
            f1_score=0.0,
            true_positives=0,
            false_positives=0,
            false_negatives=0,
            mean_coverage=0.0,
            weighted_mean_coverage=0.0,
            iou_threshold=iou_threshold,
        )

    # Panoptic metrics
    panoptic = compute_panoptic_metrics(
        pred_semantic=pred_semantic,
        gt_semantic=gt_semantic,
        pred_instances=pred_instances,
        gt_instances=gt_instances,
        num_classes=num_classes,
        class_names=class_names,
        thing_classes=thing_classes,
        stuff_classes=stuff_classes,
        iou_threshold=iou_threshold,
        ignore_label=ignore_label,
    )

    # Count instances
    n_gt = len(np.unique(gt_instances[gt_instances >= 0]))
    n_pred = len(np.unique(pred_instances[pred_instances >= 0]))

    return EvaluationResult(
        semantic=semantic,
        instance=instance,
        panoptic=panoptic,
        num_points=len(gt_semantic),
        num_gt_instances=n_gt,
        num_pred_instances=n_pred,
    )


# =============================================================================
# Polygon-based Evaluation
# =============================================================================

def compute_polygon_iou(
    pred_polygon: Any,
    gt_polygon: Any,
) -> float:
    """
    Compute IoU between two Shapely polygons.

    Args:
        pred_polygon: Predicted polygon
        gt_polygon: Ground truth polygon

    Returns:
        IoU value
    """
    try:
        intersection = pred_polygon.intersection(gt_polygon).area
        union = pred_polygon.union(gt_polygon).area
        return intersection / union if union > 0 else 0.0
    except Exception:
        return 0.0


def evaluate_polygons(
    pred_polygons: List[Any],
    gt_polygons: List[Any],
    iou_threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Evaluate predicted polygons against ground truth polygons.

    Args:
        pred_polygons: List of predicted Shapely polygons
        gt_polygons: List of ground truth Shapely polygons
        iou_threshold: IoU threshold for matching

    Returns:
        Dictionary with polygon evaluation metrics
    """
    n_pred = len(pred_polygons)
    n_gt = len(gt_polygons)

    if n_gt == 0 and n_pred == 0:
        return {
            "precision": 1.0,
            "recall": 1.0,
            "f1_score": 1.0,
            "matched_iou_mean": 0.0,
            "true_positives": 0,
            "false_positives": 0,
            "false_negatives": 0,
        }

    if n_gt == 0:
        return {
            "precision": 0.0,
            "recall": 1.0,
            "f1_score": 0.0,
            "matched_iou_mean": 0.0,
            "true_positives": 0,
            "false_positives": n_pred,
            "false_negatives": 0,
        }

    if n_pred == 0:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "matched_iou_mean": 0.0,
            "true_positives": 0,
            "false_positives": 0,
            "false_negatives": n_gt,
        }

    # Compute IoU matrix
    iou_matrix = np.zeros((n_pred, n_gt))

    for i, pred_poly in enumerate(pred_polygons):
        for j, gt_poly in enumerate(gt_polygons):
            iou_matrix[i, j] = compute_polygon_iou(pred_poly, gt_poly)

    # Match polygons (greedy)
    matched_pred = set()
    matched_gt = set()
    matched_ious = []

    iou_flat = iou_matrix.flatten()
    sorted_indices = np.argsort(-iou_flat)

    for idx in sorted_indices:
        i = idx // n_gt
        j = idx % n_gt
        iou = iou_matrix[i, j]

        if iou < iou_threshold:
            break

        if i not in matched_pred and j not in matched_gt:
            matched_pred.add(i)
            matched_gt.add(j)
            matched_ious.append(iou)

    tp = len(matched_pred)
    fp = n_pred - tp
    fn = n_gt - len(matched_gt)

    precision = tp / n_pred if n_pred > 0 else 0.0
    recall = tp / n_gt if n_gt > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    matched_iou_mean = np.mean(matched_ious) if matched_ious else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "matched_iou_mean": matched_iou_mean,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
    }


# =============================================================================
# Multi-Threshold Detection Metrics (AP, mAP)
# =============================================================================

def compute_detection_metrics(
    pred_instances: np.ndarray,
    gt_instances: np.ndarray,
    iou_thresholds: Optional[List[float]] = None,
) -> DetectionMetrics:
    """
    Compute detection metrics at multiple IoU thresholds.

    Args:
        pred_instances: (N,) predicted instance IDs
        gt_instances: (N,) ground truth instance IDs
        iou_thresholds: List of IoU thresholds (default: [0.25, 0.5, 0.75])

    Returns:
        DetectionMetrics object
    """
    if iou_thresholds is None:
        iou_thresholds = [0.25, 0.5, 0.75]

    pred_ids = np.unique(pred_instances[pred_instances >= 0])
    gt_ids = np.unique(gt_instances[gt_instances >= 0])

    n_pred, n_gt = len(pred_ids), len(gt_ids)

    if n_gt == 0 or n_pred == 0:
        return DetectionMetrics(
            ap_25=0.0, ap_50=0.0, ap_75=0.0, map=0.0,
            precisions=np.array([]), recalls=np.array([]),
            iou_thresholds=np.array(iou_thresholds),
            f1_25=0.0, f1_50=0.0, f1_75=0.0,
        )

    # Compute IoU matrix
    iou_matrix = np.zeros((n_pred, n_gt))
    for i, pred_id in enumerate(pred_ids):
        pred_mask = pred_instances == pred_id
        for j, gt_id in enumerate(gt_ids):
            gt_mask = gt_instances == gt_id
            iou_matrix[i, j] = compute_instance_iou(pred_mask, gt_mask)

    # Compute AP at each threshold
    aps = {}
    f1s = {}
    all_precisions = []
    all_recalls = []

    for thresh in iou_thresholds:
        # Match at this threshold
        matched_pred, matched_gt = set(), set()
        sorted_indices = np.argsort(-iou_matrix.flatten())

        for idx in sorted_indices:
            i, j = idx // n_gt, idx % n_gt
            if iou_matrix[i, j] < thresh:
                break
            if i not in matched_pred and j not in matched_gt:
                matched_pred.add(i)
                matched_gt.add(j)

        tp = len(matched_pred)
        fp = n_pred - tp
        fn = n_gt - len(matched_gt)

        precision = tp / n_pred if n_pred > 0 else 0.0
        recall = tp / n_gt if n_gt > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        aps[thresh] = precision * recall  # Simplified AP
        f1s[thresh] = f1
        all_precisions.append(precision)
        all_recalls.append(recall)

    return DetectionMetrics(
        ap_25=aps.get(0.25, 0.0),
        ap_50=aps.get(0.5, 0.0),
        ap_75=aps.get(0.75, 0.0),
        map=np.mean(list(aps.values())),
        precisions=np.array(all_precisions),
        recalls=np.array(all_recalls),
        iou_thresholds=np.array(iou_thresholds),
        f1_25=f1s.get(0.25, 0.0),
        f1_50=f1s.get(0.5, 0.0),
        f1_75=f1s.get(0.75, 0.0),
    )


# =============================================================================
# Segmentation Error Analysis
# =============================================================================

def compute_segmentation_errors(
    pred_instances: np.ndarray,
    gt_instances: np.ndarray,
    pred_semantic: Optional[np.ndarray] = None,
    gt_semantic: Optional[np.ndarray] = None,
    iou_threshold: float = 0.5,
    boundary_tolerance: int = 3,
) -> SegmentationErrorMetrics:
    """
    Compute segmentation error metrics (over/under segmentation).

    Args:
        pred_instances: (N,) predicted instance IDs
        gt_instances: (N,) ground truth instance IDs
        pred_semantic: Optional (N,) predicted semantic labels
        gt_semantic: Optional (N,) ground truth semantic labels
        iou_threshold: IoU threshold for matching
        boundary_tolerance: Tolerance for boundary detection (in points)

    Returns:
        SegmentationErrorMetrics object
    """
    pred_ids = np.unique(pred_instances[pred_instances >= 0])
    gt_ids = np.unique(gt_instances[gt_instances >= 0])

    n_pred, n_gt = len(pred_ids), len(gt_ids)

    if n_gt == 0:
        return SegmentationErrorMetrics(
            over_segmentation_error=0.0,
            under_segmentation_error=0.0,
            segmentation_accuracy=1.0,
            boundary_iou=0.0,
            boundary_precision=0.0,
            boundary_recall=0.0,
            split_instances=0,
            merged_instances=0,
        )

    # Compute IoU matrix
    iou_matrix = np.zeros((n_pred, n_gt))
    for i, pred_id in enumerate(pred_ids):
        pred_mask = pred_instances == pred_id
        for j, gt_id in enumerate(gt_ids):
            gt_mask = gt_instances == gt_id
            iou_matrix[i, j] = compute_instance_iou(pred_mask, gt_mask)

    # Over-segmentation: GT instances covered by multiple predictions
    split_instances = 0
    for j in range(n_gt):
        covering_preds = np.sum(iou_matrix[:, j] > 0.1)  # Multiple preds overlap
        if covering_preds > 1:
            split_instances += 1

    # Under-segmentation: Predictions covering multiple GT
    merged_instances = 0
    for i in range(n_pred):
        covered_gts = np.sum(iou_matrix[i, :] > 0.1)
        if covered_gts > 1:
            merged_instances += 1

    # Over-segmentation error (OSE)
    ose = split_instances / n_gt if n_gt > 0 else 0.0

    # Under-segmentation error (USE)
    use = merged_instances / n_pred if n_pred > 0 else 0.0

    # Achievable Segmentation Accuracy (ASA)
    # Best possible accuracy given the segmentation
    best_matches = np.max(iou_matrix, axis=0) if n_pred > 0 else np.zeros(n_gt)
    asa = np.mean(best_matches) if len(best_matches) > 0 else 0.0

    # Boundary metrics (simplified - based on point disagreement at edges)
    # In a full implementation, this would use actual boundary detection
    boundary_iou = 0.0
    boundary_precision = 0.0
    boundary_recall = 0.0

    if gt_semantic is not None and pred_semantic is not None:
        # Simple boundary approximation: points where neighbors have different labels
        gt_boundary = np.zeros(len(gt_semantic), dtype=bool)
        pred_boundary = np.zeros(len(pred_semantic), dtype=bool)

        # For now, use semantic boundaries as proxy
        gt_tree_mask = gt_semantic == 1
        pred_tree_mask = pred_semantic == 1

        # Boundary = disagreement between semantic and instance
        gt_boundary = gt_tree_mask & (gt_instances > 0)
        pred_boundary = pred_tree_mask & (pred_instances > 0)

        if np.sum(gt_boundary) > 0 and np.sum(pred_boundary) > 0:
            intersection = np.sum(gt_boundary & pred_boundary)
            union = np.sum(gt_boundary | pred_boundary)
            boundary_iou = intersection / union if union > 0 else 0.0
            boundary_precision = intersection / np.sum(pred_boundary) if np.sum(pred_boundary) > 0 else 0.0
            boundary_recall = intersection / np.sum(gt_boundary) if np.sum(gt_boundary) > 0 else 0.0

    return SegmentationErrorMetrics(
        over_segmentation_error=ose,
        under_segmentation_error=use,
        segmentation_accuracy=asa,
        boundary_iou=boundary_iou,
        boundary_precision=boundary_precision,
        boundary_recall=boundary_recall,
        split_instances=split_instances,
        merged_instances=merged_instances,
    )


# =============================================================================
# Tree Geometric Metrics
# =============================================================================

def compute_tree_geometric_metrics(
    pos: np.ndarray,
    pred_instances: np.ndarray,
    gt_instances: np.ndarray,
    iou_threshold: float = 0.5,
) -> TreeGeometricMetrics:
    """
    Compute tree-level geometric metrics (height, location, crown diameter).

    Args:
        pos: (N, 3) point positions
        pred_instances: (N,) predicted instance IDs
        gt_instances: (N,) ground truth instance IDs
        iou_threshold: IoU threshold for matching trees

    Returns:
        TreeGeometricMetrics object
    """
    pred_ids = np.unique(pred_instances[pred_instances >= 0])
    gt_ids = np.unique(gt_instances[gt_instances >= 0])

    n_pred, n_gt = len(pred_ids), len(gt_ids)

    if n_gt == 0:
        return TreeGeometricMetrics(
            height_rmse=0.0, height_mae=0.0, height_bias=0.0, height_r2=0.0,
            location_rmse=0.0, location_mae=0.0,
            crown_diameter_rmse=0.0, crown_diameter_mae=0.0,
            count_error=n_pred, count_error_pct=100.0,
            gt_heights=np.array([]), pred_heights=np.array([]),
            height_errors=np.array([]),
        )

    # Compute tree properties
    def get_tree_props(mask):
        tree_points = pos[mask]
        if len(tree_points) == 0:
            return None
        height = tree_points[:, 2].max() - tree_points[:, 2].min()
        centroid = tree_points.mean(axis=0)
        # Crown diameter as max horizontal extent
        xy_extent = tree_points[:, :2].max(axis=0) - tree_points[:, :2].min(axis=0)
        crown_diameter = np.mean(xy_extent)
        return {'height': height, 'centroid': centroid, 'crown_diameter': crown_diameter}

    gt_props = {}
    for gt_id in gt_ids:
        props = get_tree_props(gt_instances == gt_id)
        if props:
            gt_props[gt_id] = props

    pred_props = {}
    for pred_id in pred_ids:
        props = get_tree_props(pred_instances == pred_id)
        if props:
            pred_props[pred_id] = props

    # Match trees using IoU
    iou_matrix = np.zeros((n_pred, n_gt))
    for i, pred_id in enumerate(pred_ids):
        pred_mask = pred_instances == pred_id
        for j, gt_id in enumerate(gt_ids):
            gt_mask = gt_instances == gt_id
            iou_matrix[i, j] = compute_instance_iou(pred_mask, gt_mask)

    # Greedy matching
    matched_pairs = []
    matched_pred, matched_gt = set(), set()
    sorted_indices = np.argsort(-iou_matrix.flatten())

    for idx in sorted_indices:
        i, j = idx // n_gt, idx % n_gt
        if iou_matrix[i, j] < iou_threshold:
            break
        if i not in matched_pred and j not in matched_gt:
            matched_pred.add(i)
            matched_gt.add(j)
            matched_pairs.append((pred_ids[i], gt_ids[j]))

    # Compute errors for matched pairs
    height_errors = []
    location_errors = []
    crown_errors = []
    gt_heights = []
    pred_heights = []

    for pred_id, gt_id in matched_pairs:
        if pred_id in pred_props and gt_id in gt_props:
            pp, gp = pred_props[pred_id], gt_props[gt_id]

            height_errors.append(pp['height'] - gp['height'])
            gt_heights.append(gp['height'])
            pred_heights.append(pp['height'])

            loc_error = np.linalg.norm(pp['centroid'][:2] - gp['centroid'][:2])
            location_errors.append(loc_error)

            crown_errors.append(pp['crown_diameter'] - gp['crown_diameter'])

    height_errors = np.array(height_errors)
    location_errors = np.array(location_errors)
    crown_errors = np.array(crown_errors)
    gt_heights = np.array(gt_heights)
    pred_heights = np.array(pred_heights)

    # Compute metrics
    if len(height_errors) > 0:
        height_rmse = np.sqrt(np.mean(height_errors ** 2))
        height_mae = np.mean(np.abs(height_errors))
        height_bias = np.mean(height_errors)

        if len(gt_heights) > 1 and np.std(gt_heights) > 0:
            correlation = np.corrcoef(gt_heights, pred_heights)[0, 1]
            height_r2 = correlation ** 2 if not np.isnan(correlation) else 0.0
        else:
            height_r2 = 0.0
    else:
        height_rmse = height_mae = height_bias = height_r2 = 0.0

    if len(location_errors) > 0:
        location_rmse = np.sqrt(np.mean(location_errors ** 2))
        location_mae = np.mean(location_errors)
    else:
        location_rmse = location_mae = 0.0

    if len(crown_errors) > 0:
        crown_diameter_rmse = np.sqrt(np.mean(crown_errors ** 2))
        crown_diameter_mae = np.mean(np.abs(crown_errors))
    else:
        crown_diameter_rmse = crown_diameter_mae = 0.0

    count_error = n_pred - n_gt
    count_error_pct = abs(count_error) / n_gt * 100 if n_gt > 0 else 0.0

    return TreeGeometricMetrics(
        height_rmse=height_rmse,
        height_mae=height_mae,
        height_bias=height_bias,
        height_r2=height_r2,
        location_rmse=location_rmse,
        location_mae=location_mae,
        crown_diameter_rmse=crown_diameter_rmse,
        crown_diameter_mae=crown_diameter_mae,
        count_error=count_error,
        count_error_pct=count_error_pct,
        gt_heights=gt_heights,
        pred_heights=pred_heights,
        height_errors=height_errors,
    )


# =============================================================================
# Forestry-Specific Metrics
# =============================================================================

def compute_forestry_metrics(
    pos: np.ndarray,
    pred_semantic: np.ndarray,
    gt_semantic: np.ndarray,
    pred_instances: np.ndarray,
    gt_instances: np.ndarray,
    iou_threshold: float = 0.5,
    small_tree_height: float = 5.0,
    large_tree_height: float = 15.0,
) -> ForestryMetrics:
    """
    Compute forestry-specific metrics.

    Args:
        pos: (N, 3) point positions
        pred_semantic: (N,) predicted semantic labels
        gt_semantic: (N,) ground truth semantic labels
        pred_instances: (N,) predicted instance IDs
        gt_instances: (N,) ground truth instance IDs
        iou_threshold: IoU threshold for detection
        small_tree_height: Height threshold for small trees
        large_tree_height: Height threshold for large trees

    Returns:
        ForestryMetrics object
    """
    pred_ids = np.unique(pred_instances[pred_instances >= 0])
    gt_ids = np.unique(gt_instances[gt_instances >= 0])

    n_pred, n_gt = len(pred_ids), len(gt_ids)

    # Compute tree heights
    def get_tree_height(mask):
        if np.sum(mask) == 0:
            return 0.0
        return pos[mask, 2].max() - pos[mask, 2].min()

    gt_heights = {gt_id: get_tree_height(gt_instances == gt_id) for gt_id in gt_ids}

    # Categorize GT trees by size
    small_gt = [gid for gid, h in gt_heights.items() if h < small_tree_height]
    medium_gt = [gid for gid, h in gt_heights.items() if small_tree_height <= h < large_tree_height]
    large_gt = [gid for gid, h in gt_heights.items() if h >= large_tree_height]

    # Compute IoU matrix for matching
    if n_pred > 0 and n_gt > 0:
        iou_matrix = np.zeros((n_pred, n_gt))
        for i, pred_id in enumerate(pred_ids):
            pred_mask = pred_instances == pred_id
            for j, gt_id in enumerate(gt_ids):
                gt_mask = gt_instances == gt_id
                iou_matrix[i, j] = compute_instance_iou(pred_mask, gt_mask)

        # Find detected GT trees
        detected_gt = set()
        for j in range(n_gt):
            if np.max(iou_matrix[:, j]) >= iou_threshold:
                detected_gt.add(gt_ids[j])
    else:
        detected_gt = set()

    # Detection rates
    detection_rate = len(detected_gt) / n_gt if n_gt > 0 else 0.0
    omission_rate = 1.0 - detection_rate

    # Commission rate (false positives)
    if n_pred > 0 and n_gt > 0:
        matched_pred = set()
        for i in range(n_pred):
            if np.max(iou_matrix[i, :]) >= iou_threshold:
                matched_pred.add(i)
        commission_rate = (n_pred - len(matched_pred)) / n_pred if n_pred > 0 else 0.0
    else:
        commission_rate = 1.0 if n_pred > 0 else 0.0

    # Detection by size
    def detection_rate_for_subset(subset):
        if len(subset) == 0:
            return 0.0
        detected = len([gid for gid in subset if gid in detected_gt])
        return detected / len(subset)

    detection_rate_small = detection_rate_for_subset(small_gt)
    detection_rate_medium = detection_rate_for_subset(medium_gt)
    detection_rate_large = detection_rate_for_subset(large_gt)

    # Point-level accuracy
    tree_mask_gt = gt_semantic == 1
    tree_mask_pred = pred_semantic == 1
    ground_mask_gt = gt_semantic == 0

    tree_point_accuracy = (
        np.sum((pred_semantic == gt_semantic) & tree_mask_gt) / np.sum(tree_mask_gt)
        if np.sum(tree_mask_gt) > 0 else 0.0
    )

    ground_point_accuracy = (
        np.sum((pred_semantic == gt_semantic) & ground_mask_gt) / np.sum(ground_mask_gt)
        if np.sum(ground_mask_gt) > 0 else 0.0
    )

    # Canopy cover (simplified as % of tree points)
    total_points = len(gt_semantic)
    canopy_cover_gt = np.sum(tree_mask_gt) / total_points * 100 if total_points > 0 else 0.0
    canopy_cover_pred = np.sum(tree_mask_pred) / total_points * 100 if total_points > 0 else 0.0
    canopy_cover_error = abs(canopy_cover_pred - canopy_cover_gt)

    return ForestryMetrics(
        detection_rate=detection_rate,
        commission_rate=commission_rate,
        omission_rate=omission_rate,
        detection_rate_small=detection_rate_small,
        detection_rate_medium=detection_rate_medium,
        detection_rate_large=detection_rate_large,
        tree_point_accuracy=tree_point_accuracy,
        ground_point_accuracy=ground_point_accuracy,
        canopy_cover_gt=canopy_cover_gt,
        canopy_cover_pred=canopy_cover_pred,
        canopy_cover_error=canopy_cover_error,
    )


# =============================================================================
# Extended Evaluation (All Metrics)
# =============================================================================

def evaluate_extended(
    pos: np.ndarray,
    pred_semantic: np.ndarray,
    gt_semantic: np.ndarray,
    pred_instances: np.ndarray,
    gt_instances: np.ndarray,
    num_classes: int = 2,
    iou_threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Run extended evaluation with all metrics.

    Args:
        pos: (N, 3) point positions
        pred_semantic: (N,) predicted semantic labels
        gt_semantic: (N,) ground truth semantic labels
        pred_instances: (N,) predicted instance IDs
        gt_instances: (N,) ground truth instance IDs
        num_classes: Number of semantic classes
        iou_threshold: IoU threshold for instance matching

    Returns:
        Dictionary with all evaluation metrics
    """
    # Basic evaluation
    basic_result = evaluate(
        pred_semantic=pred_semantic,
        gt_semantic=gt_semantic,
        pred_instances=pred_instances,
        gt_instances=gt_instances,
        num_classes=num_classes,
        iou_threshold=iou_threshold,
    )

    # Detection metrics at multiple thresholds
    thing_mask = gt_semantic == 1
    if thing_mask.any():
        detection = compute_detection_metrics(
            pred_instances[thing_mask],
            gt_instances[thing_mask],
        )
    else:
        detection = None

    # Segmentation errors
    seg_errors = compute_segmentation_errors(
        pred_instances=pred_instances,
        gt_instances=gt_instances,
        pred_semantic=pred_semantic,
        gt_semantic=gt_semantic,
        iou_threshold=iou_threshold,
    )

    # Tree geometric metrics
    tree_geometry = compute_tree_geometric_metrics(
        pos=pos,
        pred_instances=pred_instances,
        gt_instances=gt_instances,
        iou_threshold=iou_threshold,
    )

    # Forestry metrics
    forestry = compute_forestry_metrics(
        pos=pos,
        pred_semantic=pred_semantic,
        gt_semantic=gt_semantic,
        pred_instances=pred_instances,
        gt_instances=gt_instances,
        iou_threshold=iou_threshold,
    )

    return {
        "basic": basic_result,
        "detection": detection,
        "segmentation_errors": seg_errors,
        "tree_geometry": tree_geometry,
        "forestry": forestry,
    }


# =============================================================================
# Utility Functions
# =============================================================================

def sanitize_for_json(obj: Any) -> Any:
    """Recursively sanitize an object for JSON serialization.

    Converts NaN/Inf to None and numpy types to Python types.
    """
    import math

    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return sanitize_for_json(obj.tolist())
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, (int, str, bool, type(None))):
        return obj
    else:
        return str(obj)


def print_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
) -> str:
    """
    Format confusion matrix as printable string.

    Args:
        cm: (C, C) confusion matrix
        class_names: Class names

    Returns:
        Formatted string
    """
    n = len(class_names)

    # Header
    max_name_len = max(len(name) for name in class_names)
    header = " " * (max_name_len + 2) + "Predicted".center(n * 8)
    lines = [header, "-" * (max_name_len + 2 + n * 8)]

    # Column headers
    col_header = " " * (max_name_len + 2) + "".join(f"{name[:6]:>8}" for name in class_names)
    lines.append(col_header)

    # Rows
    for i, row_name in enumerate(class_names):
        row_vals = "".join(f"{cm[i, j]:>8d}" for j in range(n))
        lines.append(f"{row_name:>{max_name_len}} | {row_vals}")

    return "\n".join(lines)


def save_evaluation_results(
    result: EvaluationResult,
    output_path: Union[str, Path],
    format: str = "json",
) -> None:
    """
    Save evaluation results to file.

    Args:
        result: EvaluationResult object
        output_path: Output file path
        format: Output format (json, txt)
    """
    import json

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == "json":
        with open(output_path, "w") as f:
            json.dump(sanitize_for_json(result.to_dict()), f, indent=2)
    elif format == "txt":
        with open(output_path, "w") as f:
            f.write(result.summary())
    else:
        raise ValueError(f"Unknown format: {format}")

    logger.info(f"Saved evaluation results to {output_path}")


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Metrics dataclasses
    "SemanticMetrics",
    "InstanceMetrics",
    "PanopticMetrics",
    "DetectionMetrics",
    "SegmentationErrorMetrics",
    "TreeGeometricMetrics",
    "ForestryMetrics",
    "EvaluationResult",
    # Core evaluation functions
    "compute_confusion_matrix",
    "compute_semantic_metrics",
    "compute_instance_metrics",
    "compute_panoptic_metrics",
    "evaluate",
    # Multi-threshold detection
    "compute_detection_metrics",
    # Segmentation error analysis
    "compute_segmentation_errors",
    # Tree geometric metrics
    "compute_tree_geometric_metrics",
    # Forestry metrics
    "compute_forestry_metrics",
    # Extended evaluation (all metrics)
    "evaluate_extended",
    # Polygon evaluation
    "compute_polygon_iou",
    "evaluate_polygons",
    # Utilities
    "sanitize_for_json",
    "print_confusion_matrix",
    "save_evaluation_results",
]
