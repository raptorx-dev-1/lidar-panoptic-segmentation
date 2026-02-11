#!/usr/bin/env python3
"""
Model Validation Script with Metrics and Visualizations

Evaluates the SegmentAnyTree model on labeled data and generates:
- Semantic IoU and confusion matrix
- Instance segmentation metrics (Precision, Recall, F1, MUCov, MWCov)
- Panoptic Quality (PQ, SQ, RQ)
- Visualizations: confusion matrix heatmap, bar charts, point cloud plots

Usage:
    python scripts/validate_model.py \
        --input /path/to/labeled/data \
        --output /path/to/results \
        --model model_file/PointGroup-PAPER.pt

Input data format:
    PLY files with fields: x, y, z, semantic_seg (0/1), treeID (instance ID)
"""

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


class NaNSafeJSONEncoder(json.JSONEncoder):
    """JSON encoder that handles NaN, Inf, and numpy types."""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            if math.isnan(obj) or math.isinf(obj):
                return None
            return float(obj)
        return super().default(obj)


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

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_labeled_ply(path: Path) -> Dict[str, np.ndarray]:
    """Load PLY file with ground truth labels."""
    from plyfile import PlyData

    ply = PlyData.read(str(path))
    v = ply['vertex']

    data = {
        'pos': np.column_stack([v['x'], v['y'], v['z']]).astype(np.float32),
    }

    # Load semantic labels if present
    if 'semantic_seg' in v.data.dtype.names:
        data['semantic_gt'] = v['semantic_seg'].astype(np.int32)
    elif 'label' in v.data.dtype.names:
        data['semantic_gt'] = v['label'].astype(np.int32)
    elif 'classification' in v.data.dtype.names:
        data['semantic_gt'] = v['classification'].astype(np.int32)
    else:
        logger.warning(f"No semantic labels found in {path.name}")
        data['semantic_gt'] = None

    # Load instance labels if present
    if 'treeID' in v.data.dtype.names:
        data['instance_gt'] = v['treeID'].astype(np.int32)
    elif 'instance_id' in v.data.dtype.names:
        data['instance_gt'] = v['instance_id'].astype(np.int32)
    elif 'tree_id' in v.data.dtype.names:
        data['instance_gt'] = v['tree_id'].astype(np.int32)
    else:
        logger.warning(f"No instance labels found in {path.name}")
        data['instance_gt'] = None

    return data


def create_visualizations(
    result: Any,
    pred_semantic: np.ndarray,
    gt_semantic: np.ndarray,
    pred_instances: np.ndarray,
    gt_instances: np.ndarray,
    pos: np.ndarray,
    output_dir: Path,
    filename_prefix: str = "",
) -> List[Path]:
    """
    Create evaluation visualizations.

    Returns list of saved file paths.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
    except ImportError:
        logger.warning("matplotlib not available, skipping visualizations")
        return []

    saved_files = []
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = f"{filename_prefix}_" if filename_prefix else ""

    # 1. Confusion Matrix Heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    cm = result.semantic.confusion_matrix
    cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    cm_normalized = np.nan_to_num(cm_normalized)

    im = ax.imshow(cm_normalized, cmap='Blues', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label='Proportion')

    # Add labels
    class_names = result.semantic.class_names
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Ground Truth')
    ax.set_title('Confusion Matrix (Normalized)')

    # Add text annotations
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            text = f"{cm[i, j]:,}\n({cm_normalized[i, j]:.1%})"
            color = 'white' if cm_normalized[i, j] > 0.5 else 'black'
            ax.text(j, i, text, ha='center', va='center', color=color, fontsize=10)

    plt.tight_layout()
    cm_path = output_dir / f"{prefix}confusion_matrix.png"
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    plt.close()
    saved_files.append(cm_path)
    logger.info(f"Saved: {cm_path}")

    # 2. Semantic IoU Bar Chart
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(class_names))
    bars = ax.bar(x, result.semantic.iou_per_class, color=['#2ecc71', '#3498db'])
    ax.axhline(y=result.semantic.miou, color='red', linestyle='--', label=f'mIoU: {result.semantic.miou:.3f}')

    ax.set_xlabel('Class')
    ax.set_ylabel('IoU')
    ax.set_title('Semantic Segmentation IoU by Class')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.set_ylim(0, 1)
    ax.legend()

    # Add value labels
    for bar, iou in zip(bars, result.semantic.iou_per_class):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{iou:.3f}', ha='center', va='bottom', fontsize=12)

    plt.tight_layout()
    iou_path = output_dir / f"{prefix}semantic_iou.png"
    plt.savefig(iou_path, dpi=150, bbox_inches='tight')
    plt.close()
    saved_files.append(iou_path)
    logger.info(f"Saved: {iou_path}")

    # 3. Instance Metrics Bar Chart
    fig, ax = plt.subplots(figsize=(10, 5))
    metrics = ['Precision', 'Recall', 'F1 Score', 'MUCov', 'MWCov']
    values = [
        result.instance.precision,
        result.instance.recall,
        result.instance.f1_score,
        result.instance.mean_coverage,
        result.instance.weighted_mean_coverage,
    ]
    colors = ['#e74c3c', '#f39c12', '#9b59b6', '#1abc9c', '#34495e']

    bars = ax.bar(metrics, values, color=colors)
    ax.set_ylabel('Score')
    ax.set_title(f'Instance Segmentation Metrics (IoU threshold: {result.instance.iou_threshold})')
    ax.set_ylim(0, 1)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=11)

    plt.tight_layout()
    inst_path = output_dir / f"{prefix}instance_metrics.png"
    plt.savefig(inst_path, dpi=150, bbox_inches='tight')
    plt.close()
    saved_files.append(inst_path)
    logger.info(f"Saved: {inst_path}")

    # 4. Panoptic Quality Chart
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Per-class PQ
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

    # Overall PQ metrics
    ax = axes[1]
    metrics = ['PQ', 'SQ', 'RQ', 'PQ\n(things)', 'PQ\n(stuff)']
    values = [
        result.panoptic.pq,
        result.panoptic.sq,
        result.panoptic.rq,
        result.panoptic.pq_things,
        result.panoptic.pq_stuff,
    ]
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6', '#f39c12']
    bars = ax.bar(metrics, values, color=colors)
    ax.set_ylabel('Score')
    ax.set_title('Overall Panoptic Metrics')
    ax.set_ylim(0, 1)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    pq_path = output_dir / f"{prefix}panoptic_quality.png"
    plt.savefig(pq_path, dpi=150, bbox_inches='tight')
    plt.close()
    saved_files.append(pq_path)
    logger.info(f"Saved: {pq_path}")

    # 5. Point Cloud Visualization (2D projection)
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Subsample for visualization if too many points
    n_points = len(pos)
    if n_points > 50000:
        idx = np.random.choice(n_points, 50000, replace=False)
    else:
        idx = np.arange(n_points)

    pos_vis = pos[idx]
    gt_sem_vis = gt_semantic[idx]
    pred_sem_vis = pred_semantic[idx]
    gt_inst_vis = gt_instances[idx]
    pred_inst_vis = pred_instances[idx]

    # Ground truth semantic
    ax = axes[0, 0]
    scatter = ax.scatter(pos_vis[:, 0], pos_vis[:, 1], c=gt_sem_vis,
                         cmap='RdYlGn', s=1, alpha=0.7)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Ground Truth Semantic')
    ax.set_aspect('equal')
    plt.colorbar(scatter, ax=ax, label='Class')

    # Predicted semantic
    ax = axes[0, 1]
    scatter = ax.scatter(pos_vis[:, 0], pos_vis[:, 1], c=pred_sem_vis,
                         cmap='RdYlGn', s=1, alpha=0.7)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Predicted Semantic')
    ax.set_aspect('equal')
    plt.colorbar(scatter, ax=ax, label='Class')

    # Ground truth instances
    ax = axes[1, 0]
    # Use random colors for instances
    n_gt_inst = gt_inst_vis.max() + 1
    colors_gt = np.zeros((len(gt_inst_vis), 3))
    np.random.seed(42)
    inst_colors = np.random.rand(n_gt_inst, 3)
    inst_colors[0] = [0.7, 0.7, 0.7]  # Background gray
    for i in range(n_gt_inst):
        mask = gt_inst_vis == i
        colors_gt[mask] = inst_colors[i]

    ax.scatter(pos_vis[:, 0], pos_vis[:, 1], c=colors_gt, s=1, alpha=0.7)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'Ground Truth Instances ({result.num_gt_instances} trees)')
    ax.set_aspect('equal')

    # Predicted instances
    ax = axes[1, 1]
    n_pred_inst = pred_inst_vis.max() + 1
    colors_pred = np.zeros((len(pred_inst_vis), 3))
    np.random.seed(42)
    inst_colors = np.random.rand(max(n_pred_inst, 1), 3)
    inst_colors[0] = [0.7, 0.7, 0.7]  # Background gray
    for i in range(n_pred_inst):
        mask = pred_inst_vis == i
        colors_pred[mask] = inst_colors[min(i, len(inst_colors)-1)]

    ax.scatter(pos_vis[:, 0], pos_vis[:, 1], c=colors_pred, s=1, alpha=0.7)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'Predicted Instances ({result.num_pred_instances} trees)')
    ax.set_aspect('equal')

    plt.tight_layout()
    pc_path = output_dir / f"{prefix}point_cloud_comparison.png"
    plt.savefig(pc_path, dpi=150, bbox_inches='tight')
    plt.close()
    saved_files.append(pc_path)
    logger.info(f"Saved: {pc_path}")

    # 6. Semantic Errors Visualization
    fig, ax = plt.subplots(figsize=(10, 8))

    errors = (pred_sem_vis != gt_sem_vis).astype(int)
    colors_err = np.array(['#2ecc71', '#e74c3c'])  # Green=correct, Red=error

    ax.scatter(pos_vis[:, 0], pos_vis[:, 1], c=[colors_err[e] for e in errors],
               s=1, alpha=0.7)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    accuracy = 1 - errors.mean()
    ax.set_title(f'Semantic Prediction Errors (Green=Correct, Red=Error)\nAccuracy: {accuracy:.1%}')
    ax.set_aspect('equal')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', label='Correct'),
        Patch(facecolor='#e74c3c', label='Error'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    err_path = output_dir / f"{prefix}semantic_errors.png"
    plt.savefig(err_path, dpi=150, bbox_inches='tight')
    plt.close()
    saved_files.append(err_path)
    logger.info(f"Saved: {err_path}")

    # 7. Summary Dashboard
    fig = plt.figure(figsize=(16, 10))

    # Create grid
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

    # Title
    fig.suptitle('SegmentAnyTree Validation Results', fontsize=16, fontweight='bold')

    # Confusion matrix (top left)
    ax = fig.add_subplot(gs[0, 0:2])
    im = ax.imshow(cm_normalized, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Ground Truth')
    ax.set_title('Confusion Matrix')
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            color = 'white' if cm_normalized[i, j] > 0.5 else 'black'
            ax.text(j, i, f"{cm_normalized[i, j]:.1%}", ha='center', va='center', color=color)

    # Metrics table (top right)
    ax = fig.add_subplot(gs[0, 2:4])
    ax.axis('off')
    table_data = [
        ['Metric', 'Value'],
        ['mIoU', f'{result.semantic.miou:.4f}'],
        ['Overall Accuracy', f'{result.semantic.overall_accuracy:.4f}'],
        ['Instance Precision', f'{result.instance.precision:.4f}'],
        ['Instance Recall', f'{result.instance.recall:.4f}'],
        ['Instance F1', f'{result.instance.f1_score:.4f}'],
        ['Panoptic Quality (PQ)', f'{result.panoptic.pq:.4f}'],
        ['Segmentation Quality (SQ)', f'{result.panoptic.sq:.4f}'],
        ['Recognition Quality (RQ)', f'{result.panoptic.rq:.4f}'],
    ]
    table = ax.table(cellText=table_data, loc='center', cellLoc='center',
                     colWidths=[0.4, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    # Header styling
    for i in range(2):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    ax.set_title('Key Metrics', pad=20)

    # IoU bars (middle left)
    ax = fig.add_subplot(gs[1, 0:2])
    bars = ax.bar(class_names, result.semantic.iou_per_class, color=['#2ecc71', '#3498db'])
    ax.axhline(y=result.semantic.miou, color='red', linestyle='--', label=f'mIoU: {result.semantic.miou:.3f}')
    ax.set_ylabel('IoU')
    ax.set_title('Semantic IoU by Class')
    ax.set_ylim(0, 1)
    ax.legend()

    # Instance metrics (middle right)
    ax = fig.add_subplot(gs[1, 2:4])
    inst_metrics = ['Precision', 'Recall', 'F1', 'MUCov', 'MWCov']
    inst_values = [result.instance.precision, result.instance.recall,
                   result.instance.f1_score, result.instance.mean_coverage,
                   result.instance.weighted_mean_coverage]
    ax.bar(inst_metrics, inst_values, color=['#e74c3c', '#f39c12', '#9b59b6', '#1abc9c', '#34495e'])
    ax.set_ylabel('Score')
    ax.set_title('Instance Segmentation Metrics')
    ax.set_ylim(0, 1)

    # Point cloud (bottom)
    ax = fig.add_subplot(gs[2, 0:2])
    ax.scatter(pos_vis[:, 0], pos_vis[:, 1], c=pred_sem_vis, cmap='RdYlGn', s=0.5, alpha=0.5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Predicted Semantic Segmentation')
    ax.set_aspect('equal')

    ax = fig.add_subplot(gs[2, 2:4])
    ax.scatter(pos_vis[:, 0], pos_vis[:, 1], c=colors_pred, s=0.5, alpha=0.5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'Predicted Instances ({result.num_pred_instances} trees)')
    ax.set_aspect('equal')

    plt.tight_layout()
    dashboard_path = output_dir / f"{prefix}validation_dashboard.png"
    plt.savefig(dashboard_path, dpi=150, bbox_inches='tight')
    plt.close()
    saved_files.append(dashboard_path)
    logger.info(f"Saved: {dashboard_path}")

    return saved_files


def run_inference_docker(
    input_file: Path,
    model_path: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run inference using Docker container.

    Returns:
        Tuple of (positions, semantic_pred, instance_pred)
    """
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        output_file = tmpdir / "output.ply"

        # Create inference script
        script = f'''
import sys
sys.path.insert(0, "/workspace")

import torch
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf
from plyfile import PlyData, PlyElement
from torch_geometric.data import Data
from torch_points3d.core.data_transform import GridSampling3D
from torch_points3d.models.model_factory import instantiate_model

# Load model
checkpoint = torch.load("/workspace/{model_path}", map_location="cpu")
dataset_props = checkpoint.get("dataset_properties", {{}})

class MockDataset:
    def __init__(self):
        self.feature_dimension = dataset_props.get("feature_dimension", 4)
        self.num_classes = dataset_props.get("num_classes", 2)
        self.stuff_classes = torch.tensor([0])
        self.thing_classes = torch.tensor([1])

model = instantiate_model(OmegaConf.create(checkpoint["run_config"]), MockDataset())
model.load_state_dict(checkpoint["models"].get("best_miou", checkpoint["models"]["latest"]), strict=False)
model = model.cuda().eval()

# Load input
ply = PlyData.read("/input/input.ply")
v = ply["vertex"]
pos = np.column_stack([v["x"], v["y"], v["z"]]).astype(np.float32)

# Preprocess
offset = pos.min(axis=0)
pos_norm = pos - offset
center = pos_norm.mean(axis=0)
features = np.column_stack([
    pos_norm[:, 0] - center[0],
    pos_norm[:, 1] - center[1],
    pos_norm[:, 2] - center[2],
    pos_norm[:, 2]
]).astype(np.float32)

n_points = len(pos_norm)
data = Data(
    pos=torch.from_numpy(pos_norm),
    x=torch.from_numpy(features),
    center_label=torch.zeros(n_points, 3, dtype=torch.float32),
    y=torch.zeros(n_points, dtype=torch.long),
    num_instances=torch.tensor([0]),
    instance_labels=torch.zeros(n_points, dtype=torch.long),
    instance_mask=torch.zeros(n_points, dtype=torch.bool),
    vote_label=torch.zeros(n_points, 3, dtype=torch.float32),
)

sampler = GridSampling3D(size=0.2, quantize_coords=True, mode="last")
data = sampler(data)
data.batch = torch.zeros(data.pos.shape[0], dtype=torch.long)

# Inference
data = data.cuda()
with torch.no_grad():
    model.set_input(data, torch.device("cuda"))
    model.forward(epoch=999)
    output = model.get_output()

# Get predictions
sem_pred = torch.argmax(output.semantic_logits, dim=1).cpu().numpy()
instance_pred = np.zeros(len(sem_pred), dtype=np.int32)
if hasattr(output, "clusters") and output.clusters:
    for i, cl in enumerate(output.clusters):
        if cl is not None:
            instance_pred[cl.cpu().numpy()] = i + 1

# Save output
out_pos = data.pos.cpu().numpy() + offset
dtype = [("x", "f4"), ("y", "f4"), ("z", "f4"), ("semantic_pred", "i4"), ("instance_pred", "i4")]
arr = np.zeros(len(out_pos), dtype=dtype)
arr["x"], arr["y"], arr["z"] = out_pos[:, 0], out_pos[:, 1], out_pos[:, 2]
arr["semantic_pred"], arr["instance_pred"] = sem_pred, instance_pred

PlyData([PlyElement.describe(arr, "vertex")], text=True).write("/output/output.ply")
print(f"Processed {{len(pos)}} -> {{len(out_pos)}} points, {{instance_pred.max()}} instances")
'''

        # Copy input file
        import shutil
        input_dir = tmpdir / "input"
        input_dir.mkdir()
        shutil.copy(input_file, input_dir / "input.ply")

        output_dir = tmpdir / "output"
        output_dir.mkdir()

        # Run Docker
        cmd = [
            "docker", "run", "--rm", "--gpus", "all",
            "--entrypoint", "bash",
            "-v", f"{Path.cwd()}:/workspace",
            "-v", f"{input_dir}:/input",
            "-v", f"{output_dir}:/output",
            "maciekwielgosz/segment-any-tree:latest",
            "-c", f"pip3 install -q scikit-learn==1.0.2 pandas==1.5.3 numpy==1.23.5 2>/dev/null && python3 -c '{script}'"
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Docker inference failed: {result.stderr}")

        logger.info(result.stdout.strip())

        # Load output
        from plyfile import PlyData
        output_ply = PlyData.read(str(output_dir / "output.ply"))
        v = output_ply['vertex']

        return (
            np.column_stack([v['x'], v['y'], v['z']]).astype(np.float32),
            v['semantic_pred'].astype(np.int32),
            v['instance_pred'].astype(np.int32),
        )


def validate_file(
    input_file: Path,
    model_path: Path,
    output_dir: Path,
    use_docker: bool = True,
) -> Dict[str, Any]:
    """
    Validate model on a single file.

    Returns evaluation results dictionary.
    """
    from lidar_panoptic_segmentation.evaluation import (
        evaluate, evaluate_extended, print_confusion_matrix
    )

    logger.info(f"Loading ground truth: {input_file.name}")
    data = load_labeled_ply(input_file)

    if data['semantic_gt'] is None:
        raise ValueError(f"No ground truth labels in {input_file}")

    pos = data['pos']
    gt_semantic = data['semantic_gt']
    gt_instances = data['instance_gt'] if data['instance_gt'] is not None else np.zeros(len(pos), dtype=np.int32)

    logger.info(f"  Points: {len(pos):,}")
    logger.info(f"  GT Trees: {gt_semantic.sum():,} points, {gt_instances.max()} instances")

    # Run inference
    logger.info("Running inference...")
    if use_docker:
        pred_pos, pred_semantic, pred_instances = run_inference_docker(input_file, model_path)
    else:
        # Native inference using MinkowskiEngine
        from lidar_panoptic_segmentation.native_inference import NativeInferencePipeline
        segmenter = NativeInferencePipeline(model_path)
        result = segmenter.segment_points(pos)
        pred_pos = result.points
        pred_semantic = result.semantic_pred
        pred_instances = result.instance_pred

    # Note: Grid sampling reduces points, so we need to handle the mismatch
    # For simplicity, we'll interpolate predictions back to original points
    # or just use the subsampled points for evaluation
    logger.info(f"  Predictions: {len(pred_pos):,} points, {pred_instances.max()} instances")

    # If sizes match, use directly; otherwise, use nearest neighbor interpolation
    if len(pred_pos) != len(pos):
        logger.info("  Interpolating predictions to original point cloud...")
        from scipy.spatial import cKDTree

        tree = cKDTree(pred_pos)
        _, indices = tree.query(pos, k=1)

        pred_semantic_full = pred_semantic[indices]
        pred_instances_full = pred_instances[indices]
    else:
        pred_semantic_full = pred_semantic
        pred_instances_full = pred_instances

    # Run extended evaluation (all metrics)
    logger.info("Computing all metrics...")
    extended_results = evaluate_extended(
        pos=pos,
        pred_semantic=pred_semantic_full,
        gt_semantic=gt_semantic,
        pred_instances=pred_instances_full,
        gt_instances=gt_instances,
        num_classes=2,
        iou_threshold=0.5,
    )

    result = extended_results["basic"]
    detection = extended_results["detection"]
    seg_errors = extended_results["segmentation_errors"]
    tree_geo = extended_results["tree_geometry"]
    forestry = extended_results["forestry"]

    # Print results
    print("\n" + result.summary())
    print("\nConfusion Matrix:")
    print(print_confusion_matrix(result.semantic.confusion_matrix, result.semantic.class_names))

    # Print extended metrics
    print("\n" + "=" * 60)
    print("EXTENDED METRICS")
    print("=" * 60)

    print("\nDetection Metrics (Multi-threshold):")
    if detection:
        print(f"  AP@25:  {detection.ap_25:.4f}")
        print(f"  AP@50:  {detection.ap_50:.4f}")
        print(f"  AP@75:  {detection.ap_75:.4f}")
        print(f"  mAP:    {detection.map:.4f}")
        print(f"  F1@25:  {detection.f1_25:.4f}")
        print(f"  F1@50:  {detection.f1_50:.4f}")
        print(f"  F1@75:  {detection.f1_75:.4f}")

    print("\nSegmentation Error Analysis:")
    print(f"  Over-segmentation Error:  {seg_errors.over_segmentation_error:.4f}")
    print(f"  Under-segmentation Error: {seg_errors.under_segmentation_error:.4f}")
    print(f"  Segmentation Accuracy:    {seg_errors.segmentation_accuracy:.4f}")
    print(f"  Split Instances:          {seg_errors.split_instances}")
    print(f"  Merged Instances:         {seg_errors.merged_instances}")

    print("\nTree Geometric Metrics:")
    print(f"  Height RMSE:        {tree_geo.height_rmse:.2f} m")
    print(f"  Height MAE:         {tree_geo.height_mae:.2f} m")
    print(f"  Height Bias:        {tree_geo.height_bias:.2f} m")
    print(f"  Height R²:          {tree_geo.height_r2:.4f}")
    print(f"  Location RMSE:      {tree_geo.location_rmse:.2f} m")
    print(f"  Location MAE:       {tree_geo.location_mae:.2f} m")
    print(f"  Crown Diameter RMSE:{tree_geo.crown_diameter_rmse:.2f} m")
    print(f"  Count Error:        {tree_geo.count_error} ({tree_geo.count_error_pct:.1f}%)")

    print("\nForestry Metrics:")
    print(f"  Detection Rate:     {forestry.detection_rate:.4f}")
    print(f"  Commission Rate:    {forestry.commission_rate:.4f}")
    print(f"  Omission Rate:      {forestry.omission_rate:.4f}")
    print(f"  Detection (small):  {forestry.detection_rate_small:.4f}")
    print(f"  Detection (medium): {forestry.detection_rate_medium:.4f}")
    print(f"  Detection (large):  {forestry.detection_rate_large:.4f}")
    print(f"  Tree Point Acc:     {forestry.tree_point_accuracy:.4f}")
    print(f"  Ground Point Acc:   {forestry.ground_point_accuracy:.4f}")
    print(f"  Canopy Cover GT:    {forestry.canopy_cover_gt:.1f}%")
    print(f"  Canopy Cover Pred:  {forestry.canopy_cover_pred:.1f}%")
    print(f"  Canopy Cover Error: {forestry.canopy_cover_error:.1f}%")

    # Create visualizations
    logger.info("Creating visualizations...")
    viz_files = create_visualizations(
        result=result,
        pred_semantic=pred_semantic_full,
        gt_semantic=gt_semantic,
        pred_instances=pred_instances_full,
        gt_instances=gt_instances,
        pos=pos,
        output_dir=output_dir,
        filename_prefix=input_file.stem,
    )

    # Build extended metrics dict for saving
    extended_metrics = {
        **result.to_dict(),
        "detection": {
            "ap_25": detection.ap_25 if detection else 0.0,
            "ap_50": detection.ap_50 if detection else 0.0,
            "ap_75": detection.ap_75 if detection else 0.0,
            "map": detection.map if detection else 0.0,
            "f1_25": detection.f1_25 if detection else 0.0,
            "f1_50": detection.f1_50 if detection else 0.0,
            "f1_75": detection.f1_75 if detection else 0.0,
        },
        "segmentation_errors": {
            "over_segmentation_error": seg_errors.over_segmentation_error,
            "under_segmentation_error": seg_errors.under_segmentation_error,
            "segmentation_accuracy": seg_errors.segmentation_accuracy,
            "split_instances": seg_errors.split_instances,
            "merged_instances": seg_errors.merged_instances,
        },
        "tree_geometry": {
            "height_rmse": tree_geo.height_rmse,
            "height_mae": tree_geo.height_mae,
            "height_bias": tree_geo.height_bias,
            "height_r2": tree_geo.height_r2,
            "location_rmse": tree_geo.location_rmse,
            "location_mae": tree_geo.location_mae,
            "crown_diameter_rmse": tree_geo.crown_diameter_rmse,
            "crown_diameter_mae": tree_geo.crown_diameter_mae,
            "count_error": tree_geo.count_error,
            "count_error_pct": tree_geo.count_error_pct,
        },
        "forestry": {
            "detection_rate": forestry.detection_rate,
            "commission_rate": forestry.commission_rate,
            "omission_rate": forestry.omission_rate,
            "detection_rate_small": forestry.detection_rate_small,
            "detection_rate_medium": forestry.detection_rate_medium,
            "detection_rate_large": forestry.detection_rate_large,
            "tree_point_accuracy": forestry.tree_point_accuracy,
            "ground_point_accuracy": forestry.ground_point_accuracy,
            "canopy_cover_gt": forestry.canopy_cover_gt,
            "canopy_cover_pred": forestry.canopy_cover_pred,
            "canopy_cover_error": forestry.canopy_cover_error,
        },
    }

    # Save metrics (sanitize to handle NaN/Inf values)
    metrics_file = output_dir / f"{input_file.stem}_metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(sanitize_for_json(extended_metrics), f, indent=2)
    logger.info(f"Saved: {metrics_file}")

    return extended_metrics


def main():
    parser = argparse.ArgumentParser(description="Validate SegmentAnyTree model")
    parser.add_argument("-i", "--input", required=True, help="Input file or directory with labeled PLY files")
    parser.add_argument("-o", "--output", required=True, help="Output directory for results")
    parser.add_argument("-m", "--model", default="model_file/PointGroup-PAPER.pt", help="Model path")
    parser.add_argument("--no-docker", action="store_true", help="Run without Docker (requires MinkowskiEngine)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    input_path = Path(args.input)
    output_path = Path(args.output)
    model_path = Path(args.model)

    output_path.mkdir(parents=True, exist_ok=True)

    # Validate model exists
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        sys.exit(1)

    # Find input files
    if input_path.is_file():
        files = [input_path]
    else:
        files = list(input_path.glob("*.ply"))

    if not files:
        logger.error(f"No PLY files found in {input_path}")
        sys.exit(1)

    logger.info(f"Found {len(files)} files to validate")

    # Process each file
    all_results = []
    for f in files:
        try:
            result = validate_file(
                input_file=f,
                model_path=model_path,
                output_dir=output_path,
                use_docker=not args.no_docker,
            )
            all_results.append({"file": f.name, **result})
        except Exception as e:
            logger.error(f"Failed to process {f.name}: {e}")
            all_results.append({"file": f.name, "error": str(e)})

    # Save aggregate results (sanitize to handle NaN/Inf values)
    aggregate_file = output_path / "validation_results.json"
    with open(aggregate_file, 'w') as f:
        json.dump(sanitize_for_json(all_results), f, indent=2)
    logger.info(f"Saved aggregate results: {aggregate_file}")

    # Print summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    successful = [r for r in all_results if "error" not in r]
    if successful:
        # Basic metrics
        avg_miou = np.mean([r["semantic"]["miou"] for r in successful])
        avg_pq = np.mean([r["panoptic"]["pq"] for r in successful])
        avg_f1 = np.mean([r["instance"]["f1_score"] for r in successful])

        # Detection metrics
        avg_map = np.mean([r["detection"]["map"] for r in successful])
        avg_ap50 = np.mean([r["detection"]["ap_50"] for r in successful])

        # Segmentation errors
        avg_ose = np.mean([r["segmentation_errors"]["over_segmentation_error"] for r in successful])
        avg_use = np.mean([r["segmentation_errors"]["under_segmentation_error"] for r in successful])

        # Tree geometry
        avg_height_rmse = np.mean([r["tree_geometry"]["height_rmse"] for r in successful])
        avg_loc_rmse = np.mean([r["tree_geometry"]["location_rmse"] for r in successful])

        # Forestry
        avg_det_rate = np.mean([r["forestry"]["detection_rate"] for r in successful])
        avg_commission = np.mean([r["forestry"]["commission_rate"] for r in successful])

        print(f"Files processed: {len(successful)}/{len(all_results)}")
        print()
        print("Core Metrics:")
        print(f"  Average mIoU:           {avg_miou:.4f}")
        print(f"  Average PQ:             {avg_pq:.4f}")
        print(f"  Average Instance F1:    {avg_f1:.4f}")
        print()
        print("Detection Metrics:")
        print(f"  Average mAP:            {avg_map:.4f}")
        print(f"  Average AP@50:          {avg_ap50:.4f}")
        print()
        print("Segmentation Errors:")
        print(f"  Avg Over-seg Error:     {avg_ose:.4f}")
        print(f"  Avg Under-seg Error:    {avg_use:.4f}")
        print()
        print("Tree Geometry:")
        print(f"  Avg Height RMSE:        {avg_height_rmse:.2f} m")
        print(f"  Avg Location RMSE:      {avg_loc_rmse:.2f} m")
        print()
        print("Forestry:")
        print(f"  Avg Detection Rate:     {avg_det_rate:.4f}")
        print(f"  Avg Commission Rate:    {avg_commission:.4f}")
    else:
        print("No files were successfully processed")

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
