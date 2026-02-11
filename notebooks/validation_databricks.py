# Databricks notebook source
# MAGIC %md
# MAGIC # SegmentAnyTree - Model Validation with Metrics & Visualizations
# MAGIC
# MAGIC Comprehensive model evaluation with:
# MAGIC - **Semantic Metrics**: IoU, Confusion Matrix, Accuracy
# MAGIC - **Instance Metrics**: Precision, Recall, F1, MUCov, MWCov
# MAGIC - **Panoptic Metrics**: PQ, SQ, RQ
# MAGIC - **Visualizations**: Heatmaps, Bar Charts, Point Cloud Plots

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

# Paths - UPDATE THESE
MODEL_PATH = "/Volumes/your_catalog/your_schema/models/PointGroup-PAPER.pt"
VALIDATION_DATA_PATH = "/Volumes/your_catalog/your_schema/validation_data"
OUTPUT_PATH = "/Volumes/your_catalog/your_schema/validation_results"

# Evaluation parameters
IOU_THRESHOLD = 0.5
CLASS_NAMES = ["non-tree", "tree"]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

import numpy as np
import torch
import json
import math
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field


def sanitize_for_json(obj: Any) -> Any:
    """Recursively sanitize an object for JSON serialization.

    Converts NaN/Inf to None and numpy types to Python types.
    """
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return sanitize_for_json(obj.tolist())
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, float)):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, (int, str, bool, type(None))):
        return obj
    else:
        return str(obj)

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluation Metrics Classes

# COMMAND ----------

@dataclass
class SemanticMetrics:
    """Semantic segmentation metrics."""
    class_names: List[str]
    iou_per_class: np.ndarray
    accuracy_per_class: np.ndarray
    precision_per_class: np.ndarray
    recall_per_class: np.ndarray
    miou: float
    overall_accuracy: float
    mean_accuracy: float
    confusion_matrix: np.ndarray


@dataclass
class InstanceMetrics:
    """Instance segmentation metrics."""
    precision: float
    recall: float
    f1_score: float
    true_positives: int
    false_positives: int
    false_negatives: int
    mean_coverage: float
    weighted_mean_coverage: float
    iou_threshold: float


@dataclass
class PanopticMetrics:
    """Panoptic segmentation metrics."""
    pq_per_class: np.ndarray
    sq_per_class: np.ndarray
    rq_per_class: np.ndarray
    pq: float
    sq: float
    rq: float
    pq_things: float
    pq_stuff: float
    class_names: List[str]
    thing_classes: List[int]
    stuff_classes: List[int]


@dataclass
class EvaluationResult:
    """Complete evaluation result."""
    semantic: SemanticMetrics
    instance: InstanceMetrics
    panoptic: PanopticMetrics
    num_points: int
    num_gt_instances: int
    num_pred_instances: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "semantic": {
                "miou": self.semantic.miou,
                "overall_accuracy": self.semantic.overall_accuracy,
                "iou_per_class": {n: float(v) for n, v in zip(self.semantic.class_names, self.semantic.iou_per_class)},
            },
            "instance": {
                "precision": self.instance.precision,
                "recall": self.instance.recall,
                "f1_score": self.instance.f1_score,
                "mean_coverage": self.instance.mean_coverage,
                "weighted_mean_coverage": self.instance.weighted_mean_coverage,
            },
            "panoptic": {
                "pq": self.panoptic.pq,
                "sq": self.panoptic.sq,
                "rq": self.panoptic.rq,
            },
            "statistics": {
                "num_points": self.num_points,
                "num_gt_instances": self.num_gt_instances,
                "num_pred_instances": self.num_pred_instances,
            },
        }

# COMMAND ----------

# MAGIC %md
# MAGIC ## Metric Computation Functions

# COMMAND ----------

def compute_confusion_matrix(pred: np.ndarray, gt: np.ndarray, num_classes: int) -> np.ndarray:
    """Compute confusion matrix."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for i in range(num_classes):
        gt_mask = gt == i
        for j in range(num_classes):
            cm[i, j] = np.sum((pred == j) & gt_mask)
    return cm


def compute_semantic_metrics(pred: np.ndarray, gt: np.ndarray, num_classes: int = 2,
                            class_names: List[str] = None) -> SemanticMetrics:
    """Compute semantic segmentation metrics."""
    if class_names is None:
        class_names = CLASS_NAMES

    cm = compute_confusion_matrix(pred, gt, num_classes)

    iou_per_class = np.zeros(num_classes)
    accuracy_per_class = np.zeros(num_classes)
    precision_per_class = np.zeros(num_classes)
    recall_per_class = np.zeros(num_classes)

    for i in range(num_classes):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp

        union = tp + fp + fn
        iou_per_class[i] = tp / union if union > 0 else 0.0

        gt_count = tp + fn
        accuracy_per_class[i] = tp / gt_count if gt_count > 0 else 0.0
        recall_per_class[i] = tp / gt_count if gt_count > 0 else 0.0

        pred_count = tp + fp
        precision_per_class[i] = tp / pred_count if pred_count > 0 else 0.0

    return SemanticMetrics(
        class_names=class_names,
        iou_per_class=iou_per_class,
        accuracy_per_class=accuracy_per_class,
        precision_per_class=precision_per_class,
        recall_per_class=recall_per_class,
        miou=np.mean(iou_per_class),
        overall_accuracy=np.trace(cm) / np.sum(cm) if np.sum(cm) > 0 else 0.0,
        mean_accuracy=np.mean(accuracy_per_class),
        confusion_matrix=cm,
    )


def compute_instance_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Compute IoU between two instance masks."""
    intersection = np.sum(pred_mask & gt_mask)
    union = np.sum(pred_mask | gt_mask)
    return intersection / union if union > 0 else 0.0


def compute_instance_metrics(pred_instances: np.ndarray, gt_instances: np.ndarray,
                            iou_threshold: float = 0.5) -> InstanceMetrics:
    """Compute instance segmentation metrics."""
    pred_ids = np.unique(pred_instances[pred_instances >= 0])
    gt_ids = np.unique(gt_instances[gt_instances >= 0])

    n_pred, n_gt = len(pred_ids), len(gt_ids)

    if n_gt == 0 or n_pred == 0:
        return InstanceMetrics(
            precision=0.0, recall=0.0, f1_score=0.0,
            true_positives=0, false_positives=n_pred, false_negatives=n_gt,
            mean_coverage=0.0, weighted_mean_coverage=0.0, iou_threshold=iou_threshold,
        )

    # Compute IoU matrix
    iou_matrix = np.zeros((n_pred, n_gt))
    for i, pred_id in enumerate(pred_ids):
        pred_mask = pred_instances == pred_id
        for j, gt_id in enumerate(gt_ids):
            gt_mask = gt_instances == gt_id
            iou_matrix[i, j] = compute_instance_iou(pred_mask, gt_mask)

    # Greedy matching
    matched_pred, matched_gt = set(), set()
    tp, iou_sum = 0, 0.0

    sorted_indices = np.argsort(-iou_matrix.flatten())
    for idx in sorted_indices:
        i, j = idx // n_gt, idx % n_gt
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

    precision = tp / n_pred if n_pred > 0 else 0.0
    recall = tp / n_gt if n_gt > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Coverage metrics
    coverage_per_gt = np.zeros(n_gt)
    gt_sizes = np.zeros(n_gt)
    for j, gt_id in enumerate(gt_ids):
        gt_mask = gt_instances == gt_id
        gt_sizes[j] = np.sum(gt_mask)
        coverage_per_gt[j] = np.max(iou_matrix[:, j])

    mean_coverage = np.mean(coverage_per_gt)
    weighted_mean_coverage = np.sum(coverage_per_gt * gt_sizes) / np.sum(gt_sizes) if np.sum(gt_sizes) > 0 else 0.0

    return InstanceMetrics(
        precision=precision, recall=recall, f1_score=f1_score,
        true_positives=tp, false_positives=fp, false_negatives=fn,
        mean_coverage=mean_coverage, weighted_mean_coverage=weighted_mean_coverage,
        iou_threshold=iou_threshold,
    )


def compute_panoptic_metrics(pred_sem: np.ndarray, gt_sem: np.ndarray,
                            pred_inst: np.ndarray, gt_inst: np.ndarray,
                            num_classes: int = 2, class_names: List[str] = None,
                            thing_classes: List[int] = None, stuff_classes: List[int] = None,
                            iou_threshold: float = 0.5) -> PanopticMetrics:
    """Compute Panoptic Quality metrics."""
    if class_names is None:
        class_names = CLASS_NAMES
    if thing_classes is None:
        thing_classes = [1]
    if stuff_classes is None:
        stuff_classes = [0]

    pq_per_class = np.zeros(num_classes)
    sq_per_class = np.zeros(num_classes)
    rq_per_class = np.zeros(num_classes)

    # Stuff classes (semantic IoU)
    for cls in stuff_classes:
        if cls >= num_classes:
            continue
        pred_mask = pred_sem == cls
        gt_mask = gt_sem == cls
        intersection = np.sum(pred_mask & gt_mask)
        union = np.sum(pred_mask | gt_mask)
        if union > 0:
            iou = intersection / union
            if iou >= iou_threshold:
                sq_per_class[cls] = iou
                rq_per_class[cls] = 1.0
                pq_per_class[cls] = iou

    # Thing classes (instance matching)
    for cls in thing_classes:
        if cls >= num_classes:
            continue
        cls_mask = gt_sem == cls
        pred_cls_mask = pred_sem == cls

        gt_inst_ids = np.unique(gt_inst[cls_mask])
        gt_inst_ids = gt_inst_ids[gt_inst_ids >= 0]
        pred_inst_ids = np.unique(pred_inst[pred_cls_mask])
        pred_inst_ids = pred_inst_ids[pred_inst_ids >= 0]

        n_gt, n_pred = len(gt_inst_ids), len(pred_inst_ids)

        if n_gt == 0 and n_pred == 0:
            pq_per_class[cls] = sq_per_class[cls] = rq_per_class[cls] = 1.0
            continue
        if n_gt == 0 or n_pred == 0:
            continue

        iou_matrix = np.zeros((n_pred, n_gt))
        for i, pred_id in enumerate(pred_inst_ids):
            pred_mask = pred_inst == pred_id
            for j, gt_id in enumerate(gt_inst_ids):
                gt_mask = gt_inst == gt_id
                iou_matrix[i, j] = compute_instance_iou(pred_mask, gt_mask)

        matched_pred, matched_gt = set(), set()
        tp, iou_sum = 0, 0.0

        sorted_indices = np.argsort(-iou_matrix.flatten())
        for idx in sorted_indices:
            i, j = idx // n_gt, idx % n_gt
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

        sq_per_class[cls] = iou_sum / tp if tp > 0 else 0.0
        rq_per_class[cls] = tp / (tp + fp/2 + fn/2) if (tp + fp/2 + fn/2) > 0 else 0.0
        pq_per_class[cls] = sq_per_class[cls] * rq_per_class[cls]

    valid_classes = [i for i in range(num_classes) if np.sum(gt_sem == i) > 0]
    pq = np.mean(pq_per_class[valid_classes]) if valid_classes else 0.0
    sq = np.mean(sq_per_class[valid_classes]) if valid_classes else 0.0
    rq = np.mean(rq_per_class[valid_classes]) if valid_classes else 0.0

    return PanopticMetrics(
        pq_per_class=pq_per_class, sq_per_class=sq_per_class, rq_per_class=rq_per_class,
        pq=pq, sq=sq, rq=rq,
        pq_things=np.mean(pq_per_class[thing_classes]) if thing_classes else 0.0,
        pq_stuff=np.mean(pq_per_class[stuff_classes]) if stuff_classes else 0.0,
        class_names=class_names, thing_classes=thing_classes, stuff_classes=stuff_classes,
    )


def evaluate(pred_sem: np.ndarray, gt_sem: np.ndarray,
            pred_inst: np.ndarray, gt_inst: np.ndarray,
            num_classes: int = 2, iou_threshold: float = 0.5) -> EvaluationResult:
    """Run complete evaluation."""
    semantic = compute_semantic_metrics(pred_sem, gt_sem, num_classes)

    thing_mask = gt_sem == 1
    if thing_mask.any():
        instance = compute_instance_metrics(pred_inst[thing_mask], gt_inst[thing_mask], iou_threshold)
    else:
        instance = InstanceMetrics(0, 0, 0, 0, 0, 0, 0, 0, iou_threshold)

    panoptic = compute_panoptic_metrics(pred_sem, gt_sem, pred_inst, gt_inst, num_classes)

    return EvaluationResult(
        semantic=semantic, instance=instance, panoptic=panoptic,
        num_points=len(gt_sem),
        num_gt_instances=len(np.unique(gt_inst[gt_inst >= 0])),
        num_pred_instances=len(np.unique(pred_inst[pred_inst >= 0])),
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Visualization Functions

# COMMAND ----------

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def plot_confusion_matrix(result: EvaluationResult, ax=None):
    """Plot confusion matrix heatmap."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    cm = result.semantic.confusion_matrix
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)

    im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label='Proportion')

    class_names = result.semantic.class_names
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Ground Truth')
    ax.set_title('Confusion Matrix (Normalized)')

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            text = f"{cm[i, j]:,}\n({cm_norm[i, j]:.1%})"
            color = 'white' if cm_norm[i, j] > 0.5 else 'black'
            ax.text(j, i, text, ha='center', va='center', color=color, fontsize=10)

    return ax


def plot_semantic_iou(result: EvaluationResult, ax=None):
    """Plot semantic IoU bar chart."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    class_names = result.semantic.class_names
    x = np.arange(len(class_names))
    bars = ax.bar(x, result.semantic.iou_per_class, color=['#2ecc71', '#3498db'])
    ax.axhline(y=result.semantic.miou, color='red', linestyle='--',
               label=f'mIoU: {result.semantic.miou:.3f}')

    ax.set_xlabel('Class')
    ax.set_ylabel('IoU')
    ax.set_title('Semantic Segmentation IoU by Class')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.set_ylim(0, 1)
    ax.legend()

    for bar, iou in zip(bars, result.semantic.iou_per_class):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{iou:.3f}', ha='center', va='bottom', fontsize=12)

    return ax


def plot_instance_metrics(result: EvaluationResult, ax=None):
    """Plot instance metrics bar chart."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    metrics = ['Precision', 'Recall', 'F1 Score', 'MUCov', 'MWCov']
    values = [
        result.instance.precision, result.instance.recall, result.instance.f1_score,
        result.instance.mean_coverage, result.instance.weighted_mean_coverage,
    ]
    colors = ['#e74c3c', '#f39c12', '#9b59b6', '#1abc9c', '#34495e']

    bars = ax.bar(metrics, values, color=colors)
    ax.set_ylabel('Score')
    ax.set_title(f'Instance Segmentation Metrics (IoU ≥ {result.instance.iou_threshold})')
    ax.set_ylim(0, 1)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=11)

    return ax


def plot_panoptic_metrics(result: EvaluationResult, axes=None):
    """Plot panoptic quality metrics."""
    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    class_names = result.panoptic.class_names

    # Per-class
    ax = axes[0]
    x = np.arange(len(class_names))
    width = 0.25
    ax.bar(x - width, result.panoptic.pq_per_class, width, label='PQ', color='#3498db')
    ax.bar(x, result.panoptic.sq_per_class, width, label='SQ', color='#2ecc71')
    ax.bar(x + width, result.panoptic.rq_per_class, width, label='RQ', color='#e74c3c')
    ax.set_xlabel('Class')
    ax.set_ylabel('Score')
    ax.set_title('Panoptic Quality by Class')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.set_ylim(0, 1)
    ax.legend()

    # Overall
    ax = axes[1]
    metrics = ['PQ', 'SQ', 'RQ', 'PQ\n(things)', 'PQ\n(stuff)']
    values = [result.panoptic.pq, result.panoptic.sq, result.panoptic.rq,
              result.panoptic.pq_things, result.panoptic.pq_stuff]
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6', '#f39c12']
    bars = ax.bar(metrics, values, color=colors)
    ax.set_ylabel('Score')
    ax.set_title('Overall Panoptic Metrics')
    ax.set_ylim(0, 1)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)

    return axes


def plot_point_cloud_comparison(pos: np.ndarray, gt_sem: np.ndarray, pred_sem: np.ndarray,
                                gt_inst: np.ndarray, pred_inst: np.ndarray,
                                max_points: int = 50000):
    """Plot point cloud comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Subsample if needed
    n = len(pos)
    if n > max_points:
        idx = np.random.choice(n, max_points, replace=False)
    else:
        idx = np.arange(n)

    pos_v = pos[idx]
    gt_sem_v = gt_sem[idx]
    pred_sem_v = pred_sem[idx]
    gt_inst_v = gt_inst[idx]
    pred_inst_v = pred_inst[idx]

    # GT semantic
    ax = axes[0, 0]
    ax.scatter(pos_v[:, 0], pos_v[:, 1], c=gt_sem_v, cmap='RdYlGn', s=1, alpha=0.7)
    ax.set_title('Ground Truth Semantic')
    ax.set_aspect('equal')

    # Pred semantic
    ax = axes[0, 1]
    ax.scatter(pos_v[:, 0], pos_v[:, 1], c=pred_sem_v, cmap='RdYlGn', s=1, alpha=0.7)
    ax.set_title('Predicted Semantic')
    ax.set_aspect('equal')

    # GT instances
    ax = axes[1, 0]
    np.random.seed(42)
    n_inst = max(gt_inst_v.max() + 1, 1)
    colors = np.random.rand(n_inst, 3)
    colors[0] = [0.7, 0.7, 0.7]
    inst_colors = colors[np.clip(gt_inst_v, 0, n_inst-1)]
    ax.scatter(pos_v[:, 0], pos_v[:, 1], c=inst_colors, s=1, alpha=0.7)
    ax.set_title(f'Ground Truth Instances ({gt_inst.max()} trees)')
    ax.set_aspect('equal')

    # Pred instances
    ax = axes[1, 1]
    n_inst = max(pred_inst_v.max() + 1, 1)
    colors = np.random.rand(n_inst, 3)
    colors[0] = [0.7, 0.7, 0.7]
    inst_colors = colors[np.clip(pred_inst_v, 0, n_inst-1)]
    ax.scatter(pos_v[:, 0], pos_v[:, 1], c=inst_colors, s=1, alpha=0.7)
    ax.set_title(f'Predicted Instances ({pred_inst.max()} trees)')
    ax.set_aspect('equal')

    plt.tight_layout()
    return fig

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Model

# COMMAND ----------

# Check if MinkowskiEngine is available
try:
    import MinkowskiEngine
    print(f"MinkowskiEngine: {MinkowskiEngine.__version__}")
    MINKOWSKI_AVAILABLE = True
except ImportError:
    print("MinkowskiEngine not available - install with init script")
    MINKOWSKI_AVAILABLE = False

# COMMAND ----------

if MINKOWSKI_AVAILABLE:
    from omegaconf import OmegaConf
    from torch_points3d.models.model_factory import instantiate_model

    print(f"Loading model from {MODEL_PATH}...")
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    dataset_props = checkpoint.get("dataset_properties", {})

    class MockDataset:
        def __init__(self):
            self.feature_dimension = dataset_props.get("feature_dimension", 4)
            self.num_classes = dataset_props.get("num_classes", 2)
            self.stuff_classes = torch.tensor([0])
            self.thing_classes = torch.tensor([1])

    model = instantiate_model(OmegaConf.create(checkpoint["run_config"]), MockDataset())
    model.load_state_dict(checkpoint["models"].get("best_miou", checkpoint["models"]["latest"]), strict=False)
    model = model.cuda().eval()

    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define Inference Function

# COMMAND ----------

if MINKOWSKI_AVAILABLE:
    from torch_geometric.data import Data
    from torch_points3d.core.data_transform import GridSampling3D

    def run_inference(pos: np.ndarray, grid_size: float = 0.2):
        """Run inference on point cloud."""
        # Normalize
        offset = pos.min(axis=0)
        pos_norm = (pos - offset).astype(np.float32)
        center = pos_norm.mean(axis=0)

        # Features
        features = np.column_stack([
            pos_norm[:, 0] - center[0],
            pos_norm[:, 1] - center[1],
            pos_norm[:, 2] - center[2],
            pos_norm[:, 2]
        ]).astype(np.float32)

        n = len(pos_norm)
        data = Data(
            pos=torch.from_numpy(pos_norm),
            x=torch.from_numpy(features),
            center_label=torch.zeros(n, 3, dtype=torch.float32),
            y=torch.zeros(n, dtype=torch.long),
            num_instances=torch.tensor([0]),
            instance_labels=torch.zeros(n, dtype=torch.long),
            instance_mask=torch.zeros(n, dtype=torch.bool),
            vote_label=torch.zeros(n, 3, dtype=torch.float32),
        )

        sampler = GridSampling3D(size=grid_size, quantize_coords=True, mode="last")
        data = sampler(data)
        data.batch = torch.zeros(data.pos.shape[0], dtype=torch.long)

        data = data.cuda()
        with torch.no_grad():
            model.set_input(data, torch.device("cuda"))
            model.forward(epoch=999)
            output = model.get_output()

        sem_pred = torch.argmax(output.semantic_logits, dim=1).cpu().numpy()
        inst_pred = np.zeros(len(sem_pred), dtype=np.int32)
        if hasattr(output, "clusters") and output.clusters:
            for i, cl in enumerate(output.clusters):
                if cl is not None:
                    inst_pred[cl.cpu().numpy()] = i + 1

        out_pos = data.pos.cpu().numpy() + offset
        return out_pos, sem_pred, inst_pred

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Validation Data

# COMMAND ----------

from plyfile import PlyData
from scipy.spatial import cKDTree

def load_labeled_ply(path: str):
    """Load PLY with ground truth labels."""
    ply = PlyData.read(path)
    v = ply['vertex']

    pos = np.column_stack([v['x'], v['y'], v['z']]).astype(np.float32)

    # Try different label field names
    sem_gt = None
    for name in ['semantic_seg', 'label', 'classification']:
        if name in v.data.dtype.names:
            sem_gt = v[name].astype(np.int32)
            break

    inst_gt = None
    for name in ['treeID', 'instance_id', 'tree_id']:
        if name in v.data.dtype.names:
            inst_gt = v[name].astype(np.int32)
            break

    return pos, sem_gt, inst_gt

# COMMAND ----------

# Find validation files
import os

val_path = Path(VALIDATION_DATA_PATH)
val_files = list(val_path.glob("*.ply"))
print(f"Found {len(val_files)} validation files")
for f in val_files[:5]:
    print(f"  - {f.name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run Validation

# COMMAND ----------

all_results = []

for file_path in val_files:
    print(f"\n{'='*60}")
    print(f"Processing: {file_path.name}")
    print(f"{'='*60}")

    # Load ground truth
    pos, gt_sem, gt_inst = load_labeled_ply(str(file_path))
    print(f"Loaded {len(pos):,} points, {gt_inst.max() if gt_inst is not None else 0} GT instances")

    if gt_sem is None:
        print("  SKIPPED: No ground truth labels")
        continue

    if gt_inst is None:
        gt_inst = np.zeros(len(pos), dtype=np.int32)

    # Run inference
    if MINKOWSKI_AVAILABLE:
        pred_pos, pred_sem, pred_inst = run_inference(pos)
        print(f"Predictions: {len(pred_pos):,} points, {pred_inst.max()} instances")

        # Interpolate to original points if needed
        if len(pred_pos) != len(pos):
            tree = cKDTree(pred_pos)
            _, indices = tree.query(pos, k=1)
            pred_sem = pred_sem[indices]
            pred_inst = pred_inst[indices]
    else:
        print("  SKIPPED: MinkowskiEngine not available")
        continue

    # Evaluate
    result = evaluate(pred_sem, gt_sem, pred_inst, gt_inst, num_classes=2, iou_threshold=IOU_THRESHOLD)

    # Print summary
    print(f"\nResults for {file_path.name}:")
    print(f"  Semantic mIoU:     {result.semantic.miou:.4f}")
    print(f"  Overall Accuracy:  {result.semantic.overall_accuracy:.4f}")
    print(f"  Instance F1:       {result.instance.f1_score:.4f}")
    print(f"  Panoptic Quality:  {result.panoptic.pq:.4f}")

    all_results.append({
        "file": file_path.name,
        "pos": pos,
        "gt_sem": gt_sem,
        "pred_sem": pred_sem,
        "gt_inst": gt_inst,
        "pred_inst": pred_inst,
        "result": result,
    })

# COMMAND ----------

# MAGIC %md
# MAGIC ## Visualize Results

# COMMAND ----------

if all_results:
    # Use first result for visualization
    r = all_results[0]
    result = r["result"]

    # Create dashboard
    fig = plt.figure(figsize=(16, 12))

    # Confusion matrix
    ax1 = fig.add_subplot(2, 3, 1)
    plot_confusion_matrix(result, ax1)

    # Semantic IoU
    ax2 = fig.add_subplot(2, 3, 2)
    plot_semantic_iou(result, ax2)

    # Instance metrics
    ax3 = fig.add_subplot(2, 3, 3)
    plot_instance_metrics(result, ax3)

    # Panoptic metrics
    ax4 = fig.add_subplot(2, 3, 4)
    ax5 = fig.add_subplot(2, 3, 5)
    plot_panoptic_metrics(result, [ax4, ax5])

    # Metrics table
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    table_data = [
        ['Metric', 'Value'],
        ['mIoU', f'{result.semantic.miou:.4f}'],
        ['Accuracy', f'{result.semantic.overall_accuracy:.4f}'],
        ['Instance F1', f'{result.instance.f1_score:.4f}'],
        ['PQ', f'{result.panoptic.pq:.4f}'],
        ['SQ', f'{result.panoptic.sq:.4f}'],
        ['RQ', f'{result.panoptic.rq:.4f}'],
    ]
    table = ax6.table(cellText=table_data, loc='center', cellLoc='center', colWidths=[0.4, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.5)
    for i in range(2):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(color='white', fontweight='bold')

    plt.suptitle(f'Validation Results: {r["file"]}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    display(fig)
    plt.close()

# COMMAND ----------

# Point cloud visualization
if all_results:
    r = all_results[0]
    fig = plot_point_cloud_comparison(
        r["pos"], r["gt_sem"], r["pred_sem"], r["gt_inst"], r["pred_inst"]
    )
    plt.suptitle(f'Point Cloud Comparison: {r["file"]}', fontsize=14)
    display(fig)
    plt.close()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary Statistics

# COMMAND ----------

if all_results:
    import pandas as pd

    summary_data = []
    for r in all_results:
        res = r["result"]
        summary_data.append({
            "File": r["file"],
            "Points": res.num_points,
            "GT Trees": res.num_gt_instances,
            "Pred Trees": res.num_pred_instances,
            "mIoU": res.semantic.miou,
            "Accuracy": res.semantic.overall_accuracy,
            "Inst Precision": res.instance.precision,
            "Inst Recall": res.instance.recall,
            "Inst F1": res.instance.f1_score,
            "PQ": res.panoptic.pq,
            "SQ": res.panoptic.sq,
            "RQ": res.panoptic.rq,
        })

    df = pd.DataFrame(summary_data)
    display(df)

    # Averages
    print("\n" + "="*60)
    print("AVERAGE METRICS ACROSS ALL FILES")
    print("="*60)
    print(f"mIoU:      {df['mIoU'].mean():.4f} ± {df['mIoU'].std():.4f}")
    print(f"Accuracy:  {df['Accuracy'].mean():.4f} ± {df['Accuracy'].std():.4f}")
    print(f"Inst F1:   {df['Inst F1'].mean():.4f} ± {df['Inst F1'].std():.4f}")
    print(f"PQ:        {df['PQ'].mean():.4f} ± {df['PQ'].std():.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save Results

# COMMAND ----------

if all_results:
    output_path = Path(OUTPUT_PATH)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save metrics JSON
    metrics_output = []
    for r in all_results:
        metrics_output.append({
            "file": r["file"],
            **r["result"].to_dict()
        })

    with open(output_path / "validation_metrics.json", "w") as f:
        json.dump(sanitize_for_json(metrics_output), f, indent=2)

    # Save summary CSV
    df.to_csv(output_path / "validation_summary.csv", index=False)

    print(f"Results saved to {OUTPUT_PATH}")
