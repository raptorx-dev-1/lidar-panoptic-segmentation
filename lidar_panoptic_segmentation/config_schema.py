"""
Configuration Schema for LiDAR Panoptic Segmentation

Uses Pydantic for robust validation and type checking.
Supports Unity Catalog abfss:// paths and local development.
"""

from __future__ import annotations

import os
import re
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator,
)


class EnvironmentName(str, Enum):
    """Supported deployment environments."""
    LOCAL = "local"
    DEV = "dev"
    DATABRICKS = "databricks"


class MinkowskiInstallMethod(str, Enum):
    """MinkowskiEngine installation methods."""
    NOTEBOOK = "notebook"
    WHEEL = "wheel"
    CONDA = "conda"
    SKIP = "skip"


class PolygonOutputFormat(str, Enum):
    """Supported polygon output formats."""
    GEOJSON = "geojson"
    SHAPEFILE = "shapefile"
    PARQUET = "parquet"


class LossType(str, Enum):
    """Supported loss functions."""
    PANOPTIC_LOSS = "panoptic_loss"
    CROSS_ENTROPY = "cross_entropy"
    FOCAL_LOSS = "focal_loss"
    DICE_LOSS = "dice_loss"


class SchedulerType(str, Enum):
    """Supported learning rate schedulers."""
    COSINE = "cosine"
    EXPONENTIAL = "exponential"
    STEP = "step"
    MULTI_STEP = "multi_step"
    CYCLIC = "cyclic"
    REDUCE_ON_PLATEAU = "reduce_on_plateau"


class ModelArchitecture(str, Enum):
    """Supported model architectures."""
    MINK_UNET = "MinkUNet"
    POINT_GROUP = "PointGroup"
    POINT_GROUP_3HEADS = "PointGroup3heads"
    SPFORMER_3D = "SPFormer3D"
    KPCONV = "KPConv"
    POINTNET2 = "PointNet2"


# ==============================================================================
# Environment Configuration
# ==============================================================================

class EnvConfig(BaseModel):
    """Environment configuration settings."""

    name: EnvironmentName = Field(
        default=EnvironmentName.LOCAL,
        description="Deployment environment: local, dev, or databricks"
    )
    debug: bool = Field(
        default=False,
        description="Enable debug mode with verbose logging"
    )
    seed: int = Field(
        default=42,
        description="Random seed for reproducibility"
    )
    num_workers: int = Field(
        default=4,
        ge=0,
        description="Number of data loader workers"
    )
    cuda_visible_devices: Optional[str] = Field(
        default=None,
        description="CUDA_VISIBLE_DEVICES setting"
    )


# ==============================================================================
# Path Configuration
# ==============================================================================

class DataPaths(BaseModel):
    """Data path configuration for train/val/test splits."""

    pointclouds: str = Field(
        description="Path to point cloud files (LAS/LAZ/PLY)"
    )
    labels: Optional[str] = Field(
        default=None,
        description="Path to label files (optional for inference)"
    )

    @field_validator("pointclouds", "labels")
    @classmethod
    def validate_path(cls, v: Optional[str]) -> Optional[str]:
        """Validate path format - supports abfss://, dbfs:/, and local paths."""
        if v is None:
            return v
        # Allow abfss:// for Unity Catalog
        if v.startswith("abfss://"):
            return v
        # Allow dbfs:/ for legacy support (but warn)
        if v.startswith("dbfs:/"):
            return v
        # Allow local paths
        return v


class InferencePaths(BaseModel):
    """Inference input/output paths."""

    input: str = Field(
        description="Input path for point clouds to process"
    )
    output: str = Field(
        description="Output path for predictions and polygons"
    )


class PathsConfig(BaseModel):
    """Complete path configuration."""

    data_root: str = Field(
        default="./data",
        description="Root path for all data (supports abfss:// for Unity Catalog)"
    )
    train: Optional[DataPaths] = Field(
        default=None,
        description="Training data paths"
    )
    val: Optional[DataPaths] = Field(
        default=None,
        description="Validation data paths"
    )
    test: Optional[DataPaths] = Field(
        default=None,
        description="Test data paths"
    )
    inference: Optional[InferencePaths] = Field(
        default=None,
        description="Inference input/output paths"
    )
    models: str = Field(
        default="./models",
        description="Path for model checkpoints and artifacts"
    )
    logs: str = Field(
        default="./logs",
        description="Path for logs and metrics"
    )
    cache: str = Field(
        default="./cache",
        description="Path for cached preprocessed data"
    )

    def resolve_path(self, path: str) -> str:
        """Resolve path with variable substitution (e.g., ${paths.data_root})."""
        if "${paths.data_root}" in path:
            path = path.replace("${paths.data_root}", self.data_root)
        return path


# ==============================================================================
# Dependencies Configuration
# ==============================================================================

class MinkowskiEngineConfig(BaseModel):
    """MinkowskiEngine installation configuration."""

    enabled: bool = Field(
        default=True,
        description="Enable MinkowskiEngine for sparse convolutions"
    )
    install_method: MinkowskiInstallMethod = Field(
        default=MinkowskiInstallMethod.NOTEBOOK,
        description="Installation method: notebook, wheel, conda, or skip"
    )
    script_path: str = Field(
        default="./scripts/install_minkowski.sh",
        description="Path to installation script"
    )
    build_wheel: bool = Field(
        default=False,
        description="Build and cache wheel for reuse"
    )
    wheel_output: Optional[str] = Field(
        default=None,
        description="Output path for built wheel"
    )
    env: Dict[str, str] = Field(
        default_factory=lambda: {
            "ME_REPO": "https://github.com/NVIDIA/MinkowskiEngine.git",
            "ME_REF": "master",
        },
        description="Environment variables for installation"
    )


class DependenciesConfig(BaseModel):
    """Dependencies configuration."""

    torch_version: str = Field(
        default="2.1.0+cu121",
        description="PyTorch version string"
    )
    cuda_version: str = Field(
        default="12.1",
        description="CUDA version"
    )
    minkowski_engine: MinkowskiEngineConfig = Field(
        default_factory=MinkowskiEngineConfig,
        description="MinkowskiEngine configuration"
    )


# ==============================================================================
# Augmentation Configuration
# ==============================================================================

class AugmentationConfig(BaseModel):
    """Data augmentation configuration."""

    rotate: bool = Field(
        default=True,
        description="Enable random rotation"
    )
    rotate_range: float = Field(
        default=180.0,
        ge=0,
        le=360,
        description="Rotation range in degrees"
    )
    jitter: bool = Field(
        default=True,
        description="Enable position jittering"
    )
    jitter_std: float = Field(
        default=0.01,
        ge=0,
        description="Standard deviation for jittering"
    )
    scale: bool = Field(
        default=False,
        description="Enable random scaling"
    )
    scale_range: List[float] = Field(
        default=[0.9, 1.1],
        description="Scale range [min, max]"
    )
    flip: bool = Field(
        default=True,
        description="Enable random flipping"
    )
    elastic_distortion: bool = Field(
        default=False,
        description="Enable elastic distortion"
    )
    color_jitter: bool = Field(
        default=False,
        description="Enable color/intensity jittering"
    )
    dropout: float = Field(
        default=0.0,
        ge=0,
        le=1,
        description="Point dropout probability"
    )


# ==============================================================================
# Transformer Configuration
# ==============================================================================

class TransformerConfig(BaseModel):
    """Configuration for SPFormer3D mask transformer architecture."""

    d_model: int = Field(default=256, ge=1, description="Transformer hidden dimension")
    nhead: int = Field(default=8, ge=1, description="Number of attention heads")
    num_decoder_layers: int = Field(default=6, ge=1, description="Number of transformer decoder layers")
    num_queries: int = Field(default=100, ge=1, description="Number of learnable instance queries")
    dim_feedforward: int = Field(default=1024, ge=1, description="FFN intermediate dimension")
    superpoint_voxel_size: float = Field(default=0.1, gt=0, description="Voxel size for superpoint pooling")
    feature_dim: int = Field(default=64, ge=1, description="Backbone output feature dimension")
    backbone_channels: List[int] = Field(
        default=[64, 128, 256, 512],
        description="Channel sizes for each encoder stage"
    )
    confidence_threshold: float = Field(default=0.5, ge=0, le=1, description="Confidence threshold for inference")
    mask_threshold: float = Field(default=0.5, ge=0, le=1, description="Mask binarization threshold")
    weight_ce: float = Field(default=2.0, ge=0, description="Weight for classification CE loss")
    weight_mask: float = Field(default=5.0, ge=0, description="Weight for mask focal loss")
    weight_dice: float = Field(default=5.0, ge=0, description="Weight for mask dice loss")
    weight_aux_semantic: float = Field(default=1.0, ge=0, description="Weight for auxiliary semantic loss")


# ==============================================================================
# Training Configuration
# ==============================================================================

class TrainingConfig(BaseModel):
    """Training configuration."""

    model_name: ModelArchitecture = Field(
        default=ModelArchitecture.POINT_GROUP_3HEADS,
        description="Model architecture to use"
    )
    backbone: str = Field(
        default="MinkUNet34C",
        description="Backbone network configuration"
    )
    num_classes: int = Field(
        default=2,
        ge=1,
        description="Number of semantic classes"
    )
    embed_dim: int = Field(
        default=5,
        ge=1,
        description="Instance embedding dimension"
    )
    learning_rate: float = Field(
        default=0.0001,
        gt=0,
        alias="lr",
        description="Initial learning rate"
    )
    batch_size: int = Field(
        default=8,
        ge=1,
        description="Training batch size"
    )
    epochs: int = Field(
        default=60,
        ge=1,
        description="Number of training epochs"
    )
    scheduler: SchedulerType = Field(
        default=SchedulerType.COSINE,
        description="Learning rate scheduler"
    )
    loss: LossType = Field(
        default=LossType.PANOPTIC_LOSS,
        description="Loss function"
    )
    weight_decay: float = Field(
        default=0.0001,
        ge=0,
        description="Weight decay for regularization"
    )
    gradient_clip: Optional[float] = Field(
        default=1.0,
        ge=0,
        description="Gradient clipping threshold"
    )
    augmentations: AugmentationConfig = Field(
        default_factory=AugmentationConfig,
        description="Data augmentation settings"
    )
    voxel_size: float = Field(
        default=0.02,
        gt=0,
        description="Voxel size for point cloud discretization"
    )
    sample_radius: float = Field(
        default=8.0,
        gt=0,
        description="Sampling radius for training patches"
    )
    checkpoint_frequency: int = Field(
        default=5,
        ge=1,
        description="Save checkpoint every N epochs"
    )
    eval_frequency: int = Field(
        default=1,
        ge=1,
        description="Evaluate every N epochs"
    )
    resume_from: Optional[str] = Field(
        default=None,
        description="Path to checkpoint to resume from"
    )
    mixed_precision: bool = Field(
        default=True,
        description="Enable automatic mixed precision training"
    )
    transformer: TransformerConfig = Field(
        default_factory=TransformerConfig,
        description="SPFormer3D transformer architecture settings"
    )


# ==============================================================================
# Inference Configuration
# ==============================================================================

class InferenceConfig(BaseModel):
    """Inference configuration."""

    model_uri: str = Field(
        default="models:/LidarPanopticSegmentation/latest",
        description="MLflow model URI or local path"
    )
    min_points_per_instance: int = Field(
        default=50,
        ge=1,
        description="Minimum points to consider a valid instance"
    )
    confidence_threshold: float = Field(
        default=0.5,
        ge=0,
        le=1,
        description="Confidence threshold for instance predictions"
    )
    polygon_output_format: PolygonOutputFormat = Field(
        default=PolygonOutputFormat.GEOJSON,
        description="Output format for extracted polygons"
    )
    batch_size: int = Field(
        default=1,
        ge=1,
        description="Inference batch size"
    )
    overlap: float = Field(
        default=0.5,
        ge=0,
        le=1,
        description="Overlap ratio for sliding window inference"
    )
    tile_size: float = Field(
        default=50.0,
        gt=0,
        description="Tile size for large point cloud processing"
    )
    merge_instances: bool = Field(
        default=True,
        description="Merge instances across tile boundaries"
    )
    save_point_clouds: bool = Field(
        default=True,
        description="Save segmented point clouds"
    )
    save_polygons: bool = Field(
        default=True,
        description="Save extracted tree polygons"
    )


# ==============================================================================
# Postprocessing Configuration
# ==============================================================================

class PolygonConfig(BaseModel):
    """Polygon extraction configuration."""

    method: str = Field(
        default="convex_hull",
        description="Polygon method: convex_hull, alphashape, concave_hull"
    )
    alpha: float = Field(
        default=0.5,
        gt=0,
        description="Alpha parameter for alphashape"
    )
    min_area: float = Field(
        default=1.0,
        ge=0,
        description="Minimum polygon area in square meters"
    )
    simplify_tolerance: float = Field(
        default=0.1,
        ge=0,
        description="Polygon simplification tolerance"
    )
    buffer_distance: float = Field(
        default=0.0,
        ge=0,
        description="Buffer distance around polygons"
    )


class PostprocessConfig(BaseModel):
    """Postprocessing configuration."""

    clustering_method: str = Field(
        default="hdbscan",
        description="Clustering method: hdbscan, meanshift, dbscan"
    )
    hdbscan_min_cluster_size: int = Field(
        default=50,
        ge=1,
        description="HDBSCAN minimum cluster size"
    )
    hdbscan_min_samples: int = Field(
        default=10,
        ge=1,
        description="HDBSCAN minimum samples"
    )
    meanshift_bandwidth: Optional[float] = Field(
        default=None,
        description="MeanShift bandwidth (auto if None)"
    )
    polygon: PolygonConfig = Field(
        default_factory=PolygonConfig,
        description="Polygon extraction settings"
    )
    filter_by_height: bool = Field(
        default=True,
        description="Filter instances by height range"
    )
    min_height: float = Field(
        default=2.0,
        description="Minimum tree height in meters"
    )
    max_height: float = Field(
        default=100.0,
        description="Maximum tree height in meters"
    )


# ==============================================================================
# Logging Configuration
# ==============================================================================

class MLflowConfig(BaseModel):
    """MLflow logging configuration."""

    enabled: bool = Field(
        default=True,
        description="Enable MLflow tracking"
    )
    tracking_uri: str = Field(
        default="databricks",
        description="MLflow tracking URI (databricks, or file:// path)"
    )
    experiment_name: str = Field(
        default="LidarPanopticSegmentation_Experiments",
        description="MLflow experiment name"
    )
    run_name: Optional[str] = Field(
        default=None,
        description="MLflow run name (auto-generated if None)"
    )
    registry_uri: Optional[str] = Field(
        default=None,
        description="Model registry URI"
    )
    log_artifacts: bool = Field(
        default=True,
        description="Log artifacts to MLflow"
    )
    log_models: bool = Field(
        default=True,
        description="Log models to MLflow registry"
    )


class WandbConfig(BaseModel):
    """Weights & Biases logging configuration."""

    enabled: bool = Field(
        default=False,
        description="Enable Weights & Biases tracking"
    )
    project: str = Field(
        default="lidar-panoptic-segmentation",
        description="W&B project name"
    )
    entity: Optional[str] = Field(
        default=None,
        description="W&B entity/team name"
    )
    name: Optional[str] = Field(
        default=None,
        description="W&B run name"
    )
    tags: List[str] = Field(
        default_factory=list,
        description="W&B run tags"
    )


class LoggingConfig(BaseModel):
    """Complete logging configuration."""

    level: str = Field(
        default="INFO",
        description="Logging level: DEBUG, INFO, WARNING, ERROR"
    )
    mlflow: MLflowConfig = Field(
        default_factory=MLflowConfig,
        description="MLflow configuration"
    )
    wandb: WandbConfig = Field(
        default_factory=WandbConfig,
        description="Weights & Biases configuration"
    )
    log_to_file: bool = Field(
        default=True,
        description="Also log to file"
    )
    log_frequency: int = Field(
        default=100,
        ge=1,
        description="Log metrics every N iterations"
    )


# ==============================================================================
# Main Configuration
# ==============================================================================

class Config(BaseModel):
    """
    Main configuration for LiDAR Panoptic Segmentation.

    This configuration drives all runtime behavior across environments
    and components, supporting Unity Catalog paths and local development.
    """

    env: EnvConfig = Field(
        default_factory=EnvConfig,
        description="Environment settings"
    )
    paths: PathsConfig = Field(
        default_factory=PathsConfig,
        description="Data and output paths"
    )
    dependencies: DependenciesConfig = Field(
        default_factory=DependenciesConfig,
        description="Dependency versions and installation settings"
    )
    training: TrainingConfig = Field(
        default_factory=TrainingConfig,
        description="Training hyperparameters and settings"
    )
    inference: InferenceConfig = Field(
        default_factory=InferenceConfig,
        description="Inference settings"
    )
    postprocess: PostprocessConfig = Field(
        default_factory=PostprocessConfig,
        description="Postprocessing settings"
    )
    logging: LoggingConfig = Field(
        default_factory=LoggingConfig,
        description="Logging and experiment tracking"
    )

    model_config = {
        "populate_by_name": True,
        "validate_assignment": True,
        "extra": "forbid",
    }

    @model_validator(mode="after")
    def resolve_path_variables(self) -> "Config":
        """Resolve path variable substitutions after model creation."""
        # Resolve paths that use ${paths.data_root}
        if self.paths.train:
            self.paths.train.pointclouds = self.paths.resolve_path(
                self.paths.train.pointclouds
            )
            if self.paths.train.labels:
                self.paths.train.labels = self.paths.resolve_path(
                    self.paths.train.labels
                )
        if self.paths.val:
            self.paths.val.pointclouds = self.paths.resolve_path(
                self.paths.val.pointclouds
            )
            if self.paths.val.labels:
                self.paths.val.labels = self.paths.resolve_path(
                    self.paths.val.labels
                )
        if self.paths.inference:
            self.paths.inference.input = self.paths.resolve_path(
                self.paths.inference.input
            )
            self.paths.inference.output = self.paths.resolve_path(
                self.paths.inference.output
            )
        return self

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return self.model_dump(mode="json")

    def save_yaml(self, path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        import yaml

        with open(path, "w") as f:
            yaml.dump(self.model_dump(mode="json"), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "Config":
        """Load configuration from YAML file."""
        import yaml

        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def get_effective_device(self) -> str:
        """Get the effective compute device based on environment."""
        import torch

        if self.env.cuda_visible_devices:
            os.environ["CUDA_VISIBLE_DEVICES"] = self.env.cuda_visible_devices

        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
