"""
LiDAR Panoptic Segmentation

A modular, configuration-driven, enterprise-ready deep learning system
for 3D panoptic segmentation and tree polygon extraction.

Designed for production deployment on Azure Databricks with Unity Catalog,
MLflow, and secure installation of MinkowskiEngine.
"""

__version__ = "1.0.0"
__author__ = "LiDAR Panoptic Segmentation Team"

from lidar_panoptic_segmentation.config import (
    Config,
    load_config,
    validate_config,
)
from lidar_panoptic_segmentation.model import (
    LidarPanopticModel,
    create_model,
)
from lidar_panoptic_segmentation.dataset import (
    LidarPanopticDataset,
    create_dataloader,
)
from lidar_panoptic_segmentation.postprocess import (
    extract_polygons,
    save_geojson,
)
from lidar_panoptic_segmentation.logging_utils import (
    setup_logging,
    MLflowLogger,
)
from lidar_panoptic_segmentation.segment_trees import (
    segment_trees,
    SegmentationResult,
)
from lidar_panoptic_segmentation.evaluation import (
    evaluate,
    EvaluationResult,
    compute_semantic_metrics,
    compute_instance_metrics,
    compute_panoptic_metrics,
    evaluate_polygons,
)

__all__ = [
    # Config
    "Config",
    "load_config",
    "validate_config",
    # Model
    "LidarPanopticModel",
    "create_model",
    # Dataset
    "LidarPanopticDataset",
    "create_dataloader",
    # Postprocessing
    "extract_polygons",
    "save_geojson",
    # Logging
    "setup_logging",
    "MLflowLogger",
    # Tree Segmentation (Pretrained)
    "segment_trees",
    "SegmentationResult",
    # Evaluation
    "evaluate",
    "EvaluationResult",
    "compute_semantic_metrics",
    "compute_instance_metrics",
    "compute_panoptic_metrics",
    "evaluate_polygons",
]
