"""
LiDAR Panoptic Segmentation Dataset Module

Provides Unity Catalog-aware dataset handling for LiDAR point cloud data.
Supports abfss://, local paths, and various file formats (LAS, LAZ, PLY).
"""

from __future__ import annotations

import logging
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from lidar_panoptic_segmentation.config_schema import (
    AugmentationConfig,
    Config,
    DataPaths,
)

logger = logging.getLogger(__name__)

# Class mappings for panoptic segmentation
CLASSES = {0: "non-tree", 1: "tree"}
CLASSES_INV = {"non-tree": 0, "tree": 1}
VALID_CLASS_IDS = [0, 1]
SEM_ID_FOR_INSTANCE = [1]  # Classes that have instance segmentation (trees)

# Visualization colors (RGB)
OBJECT_COLORS = {
    0: [179, 116, 81],   # non-tree: brown
    1: [77, 174, 84],    # tree: green
    -1: [128, 128, 128], # unknown: gray
}


@dataclass
class PointCloudData:
    """Container for point cloud data and labels."""

    pos: np.ndarray  # (N, 3) XYZ coordinates
    features: Optional[np.ndarray] = None  # (N, F) additional features
    semantic_labels: Optional[np.ndarray] = None  # (N,) semantic class IDs
    instance_labels: Optional[np.ndarray] = None  # (N,) instance IDs
    intensity: Optional[np.ndarray] = None  # (N,) intensity values
    colors: Optional[np.ndarray] = None  # (N, 3) RGB colors
    normals: Optional[np.ndarray] = None  # (N, 3) normal vectors

    # Metadata
    file_path: Optional[str] = None
    bounds: Optional[np.ndarray] = None  # (2, 3) min/max bounds
    crs: Optional[str] = None  # Coordinate reference system

    def __len__(self) -> int:
        return len(self.pos)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for batching."""
        return {
            "pos": self.pos,
            "features": self.features,
            "semantic_labels": self.semantic_labels,
            "instance_labels": self.instance_labels,
            "intensity": self.intensity,
            "colors": self.colors,
        }


class CloudStorageHandler:
    """
    Handler for cloud storage operations.

    Supports:
    - Unity Catalog: abfss://container@account.dfs.core.windows.net/path
    - Local filesystem paths
    - Legacy DBFS (deprecated): dbfs:/path
    """

    def __init__(self, is_databricks: bool = False):
        self.is_databricks = is_databricks
        self._spark = None
        self._dbutils = None

    @property
    def spark(self):
        """Lazy load Spark session."""
        if self._spark is None and self.is_databricks:
            try:
                from pyspark.sql import SparkSession
                self._spark = SparkSession.builder.getOrCreate()
            except ImportError:
                logger.warning("PySpark not available")
        return self._spark

    @property
    def dbutils(self):
        """Lazy load dbutils."""
        if self._dbutils is None and self.is_databricks:
            try:
                from pyspark.dbutils import DBUtils
                self._dbutils = DBUtils(self.spark)
            except Exception:
                # Try alternative import
                try:
                    import IPython
                    self._dbutils = IPython.get_ipython().user_ns.get("dbutils")
                except Exception:
                    logger.warning("dbutils not available")
        return self._dbutils

    def resolve_path(self, path: str) -> str:
        """
        Resolve cloud storage path to a form usable for file operations.

        Args:
            path: Cloud or local path

        Returns:
            Resolved path
        """
        if path.startswith("abfss://"):
            if self.is_databricks:
                # On Databricks, abfss:// paths work directly with Spark
                return path
            else:
                # For local development, convert to a mounted path or download
                logger.warning(
                    f"abfss:// path not directly accessible locally: {path}"
                )
                return path

        if path.startswith("dbfs:/"):
            # Convert dbfs:/ to local path format
            return path.replace("dbfs:/", "/dbfs/")

        return path

    def list_files(
        self,
        path: str,
        pattern: str = "*.las",
        recursive: bool = True,
    ) -> List[str]:
        """
        List files in a directory.

        Args:
            path: Directory path
            pattern: Glob pattern for filtering
            recursive: Search recursively

        Returns:
            List of file paths
        """
        resolved_path = self.resolve_path(path)

        if path.startswith("abfss://") and self.is_databricks and self.dbutils:
            # Use dbutils for Unity Catalog paths
            try:
                files = []
                for f in self.dbutils.fs.ls(path):
                    if f.isFile() and self._matches_pattern(f.name, pattern):
                        files.append(f.path)
                    elif f.isDir() and recursive:
                        files.extend(self.list_files(f.path, pattern, recursive))
                return files
            except Exception as e:
                logger.error(f"Error listing files: {e}")
                return []

        # Local filesystem
        from glob import glob

        local_path = Path(resolved_path)
        if recursive:
            files = list(local_path.rglob(pattern))
        else:
            files = list(local_path.glob(pattern))

        return [str(f) for f in files]

    def _matches_pattern(self, filename: str, pattern: str) -> bool:
        """Check if filename matches glob pattern."""
        import fnmatch
        return fnmatch.fnmatch(filename.lower(), pattern.lower())

    def read_file(self, path: str) -> bytes:
        """
        Read file contents as bytes.

        Args:
            path: File path

        Returns:
            File contents
        """
        resolved_path = self.resolve_path(path)

        if path.startswith("abfss://") and self.is_databricks and self.dbutils:
            # Read from Unity Catalog via dbutils
            # Note: For large files, consider using Spark or streaming
            local_temp = f"/tmp/{Path(path).name}"
            self.dbutils.fs.cp(path, f"file:{local_temp}")
            with open(local_temp, "rb") as f:
                return f.read()

        with open(resolved_path, "rb") as f:
            return f.read()

    def write_file(self, path: str, data: bytes) -> None:
        """
        Write bytes to file.

        Args:
            path: File path
            data: Bytes to write
        """
        resolved_path = self.resolve_path(path)

        if path.startswith("abfss://") and self.is_databricks and self.dbutils:
            # Write to Unity Catalog via temp file
            local_temp = f"/tmp/{Path(path).name}"
            with open(local_temp, "wb") as f:
                f.write(data)
            self.dbutils.fs.cp(f"file:{local_temp}", path)
            os.remove(local_temp)
            return

        # Ensure parent directory exists
        Path(resolved_path).parent.mkdir(parents=True, exist_ok=True)
        with open(resolved_path, "wb") as f:
            f.write(data)


def read_las_file(
    file_path: str,
    storage_handler: Optional[CloudStorageHandler] = None,
) -> PointCloudData:
    """
    Read a LAS/LAZ file and return point cloud data.

    Args:
        file_path: Path to LAS/LAZ file
        storage_handler: Cloud storage handler for remote files

    Returns:
        PointCloudData object
    """
    try:
        import laspy
    except ImportError:
        raise ImportError("laspy is required for reading LAS files: pip install laspy")

    # Handle cloud storage
    if storage_handler and (
        file_path.startswith("abfss://") or file_path.startswith("dbfs:/")
    ):
        resolved_path = storage_handler.resolve_path(file_path)
    else:
        resolved_path = file_path

    # Read LAS file
    las = laspy.read(resolved_path)

    # Extract coordinates
    pos = np.vstack([las.x, las.y, las.z]).T.astype(np.float32)

    # Extract intensity if available
    intensity = None
    if hasattr(las, "intensity"):
        intensity = np.array(las.intensity, dtype=np.float32)

    # Extract colors if available (RGB)
    colors = None
    if hasattr(las, "red") and hasattr(las, "green") and hasattr(las, "blue"):
        colors = np.vstack([las.red, las.green, las.blue]).T.astype(np.float32)
        colors = colors / 65535.0  # Normalize to [0, 1]

    # Extract classification as semantic labels
    semantic_labels = None
    if hasattr(las, "classification"):
        semantic_labels = np.array(las.classification, dtype=np.int64)

    # Extract instance labels if available (custom field)
    instance_labels = None
    for field_name in ["treeID", "TreeID", "tree_id", "instance_id"]:
        if field_name in las.point_format.dimension_names:
            instance_labels = np.array(las[field_name], dtype=np.int64)
            break

    # Get bounds
    bounds = np.array([[las.x.min(), las.y.min(), las.z.min()],
                       [las.x.max(), las.y.max(), las.z.max()]])

    # Get CRS if available
    crs = None
    if hasattr(las, "header") and hasattr(las.header, "parse_crs"):
        try:
            crs_obj = las.header.parse_crs()
            if crs_obj:
                crs = str(crs_obj)
        except Exception:
            pass

    return PointCloudData(
        pos=pos,
        intensity=intensity,
        colors=colors,
        semantic_labels=semantic_labels,
        instance_labels=instance_labels,
        bounds=bounds,
        crs=crs,
        file_path=file_path,
    )


def read_ply_file(
    file_path: str,
    storage_handler: Optional[CloudStorageHandler] = None,
) -> PointCloudData:
    """
    Read a PLY file and return point cloud data.

    Args:
        file_path: Path to PLY file
        storage_handler: Cloud storage handler for remote files

    Returns:
        PointCloudData object
    """
    try:
        from plyfile import PlyData
    except ImportError:
        raise ImportError("plyfile is required for reading PLY files: pip install plyfile")

    # Handle cloud storage
    if storage_handler and (
        file_path.startswith("abfss://") or file_path.startswith("dbfs:/")
    ):
        import tempfile
        data = storage_handler.read_file(file_path)
        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        plydata = PlyData.read(tmp_path)
        os.remove(tmp_path)
    else:
        plydata = PlyData.read(file_path)

    vertex = plydata["vertex"]

    # Extract coordinates
    pos = np.vstack([vertex["x"], vertex["y"], vertex["z"]]).T.astype(np.float32)

    # Extract colors if available
    colors = None
    if "red" in vertex.data.dtype.names:
        colors = np.vstack([
            vertex["red"], vertex["green"], vertex["blue"]
        ]).T.astype(np.float32)
        if colors.max() > 1:
            colors = colors / 255.0

    # Extract intensity
    intensity = None
    if "intensity" in vertex.data.dtype.names:
        intensity = np.array(vertex["intensity"], dtype=np.float32)

    # Extract semantic labels
    semantic_labels = None
    for name in ["semantic_seg", "label", "class", "classification"]:
        if name in vertex.data.dtype.names:
            semantic_labels = np.array(vertex[name], dtype=np.int64)
            break

    # Extract instance labels
    instance_labels = None
    for name in ["treeID", "instance_id", "instance", "object_id"]:
        if name in vertex.data.dtype.names:
            instance_labels = np.array(vertex[name], dtype=np.int64)
            break

    # Extract normals
    normals = None
    if "nx" in vertex.data.dtype.names:
        normals = np.vstack([vertex["nx"], vertex["ny"], vertex["nz"]]).T.astype(np.float32)

    bounds = np.array([pos.min(axis=0), pos.max(axis=0)])

    return PointCloudData(
        pos=pos,
        intensity=intensity,
        colors=colors,
        normals=normals,
        semantic_labels=semantic_labels,
        instance_labels=instance_labels,
        bounds=bounds,
        file_path=file_path,
    )


def read_point_cloud(
    file_path: str,
    storage_handler: Optional[CloudStorageHandler] = None,
) -> PointCloudData:
    """
    Read a point cloud file (auto-detect format).

    Args:
        file_path: Path to point cloud file
        storage_handler: Cloud storage handler for remote files

    Returns:
        PointCloudData object
    """
    ext = Path(file_path).suffix.lower()

    if ext in [".las", ".laz"]:
        return read_las_file(file_path, storage_handler)
    elif ext == ".ply":
        return read_ply_file(file_path, storage_handler)
    else:
        raise ValueError(f"Unsupported file format: {ext}")


class DataAugmentation:
    """Data augmentation transforms for point clouds."""

    def __init__(self, config: AugmentationConfig):
        self.config = config

    def __call__(self, data: PointCloudData) -> PointCloudData:
        """Apply augmentations to point cloud data."""
        pos = data.pos.copy()

        # Random rotation around Z-axis
        if self.config.rotate:
            angle = np.random.uniform(
                -self.config.rotate_range,
                self.config.rotate_range
            ) * np.pi / 180
            rot_matrix = np.array([
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1]
            ], dtype=np.float32)
            pos = pos @ rot_matrix.T

        # Random scaling
        if self.config.scale:
            scale = np.random.uniform(
                self.config.scale_range[0],
                self.config.scale_range[1]
            )
            pos *= scale

        # Position jittering
        if self.config.jitter:
            jitter = np.random.normal(
                0, self.config.jitter_std, pos.shape
            ).astype(np.float32)
            pos += jitter

        # Random flipping
        if self.config.flip:
            if np.random.random() > 0.5:
                pos[:, 0] *= -1
            if np.random.random() > 0.5:
                pos[:, 1] *= -1

        # Point dropout
        if self.config.dropout > 0:
            mask = np.random.random(len(pos)) > self.config.dropout
            pos = pos[mask]
            if data.semantic_labels is not None:
                data.semantic_labels = data.semantic_labels[mask]
            if data.instance_labels is not None:
                data.instance_labels = data.instance_labels[mask]
            if data.intensity is not None:
                data.intensity = data.intensity[mask]
            if data.colors is not None:
                data.colors = data.colors[mask]

        data.pos = pos
        return data


def grid_sampling(
    pos: np.ndarray,
    voxel_size: float,
    features: Optional[Dict[str, np.ndarray]] = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray]:
    """
    Apply grid/voxel sampling to point cloud.

    Args:
        pos: (N, 3) point coordinates
        voxel_size: Voxel size for sampling
        features: Dictionary of features to sample

    Returns:
        Tuple of (sampled_pos, sampled_features, indices)
    """
    # Quantize coordinates to voxel grid
    quantized = np.floor(pos / voxel_size).astype(np.int32)

    # Find unique voxels and map points to voxels
    _, unique_indices, inverse = np.unique(
        quantized, axis=0, return_index=True, return_inverse=True
    )

    # Sample one point per voxel (first point)
    sampled_pos = pos[unique_indices]

    # Sample features
    sampled_features = {}
    if features:
        for key, feat in features.items():
            if feat is not None:
                sampled_features[key] = feat[unique_indices]

    return sampled_pos, sampled_features, unique_indices


class LidarPanopticDataset(Dataset):
    """
    PyTorch Dataset for LiDAR panoptic segmentation.

    Supports Unity Catalog paths and various file formats.
    """

    def __init__(
        self,
        data_paths: DataPaths,
        config: Config,
        split: str = "train",
        transform: Optional[Callable] = None,
    ):
        """
        Initialize dataset.

        Args:
            data_paths: Paths configuration
            config: Full configuration object
            split: Dataset split (train/val/test)
            transform: Optional transform function
        """
        self.config = config
        self.split = split
        self.transform = transform

        # Initialize storage handler
        is_databricks = config.env.name.value == "databricks"
        self.storage = CloudStorageHandler(is_databricks=is_databricks)

        # Get file list
        self.files = self._get_file_list(data_paths.pointclouds)
        if not self.files:
            logger.warning(f"No files found in {data_paths.pointclouds}")

        # Setup augmentation for training
        if split == "train":
            self.augmentation = DataAugmentation(config.training.augmentations)
        else:
            self.augmentation = None

        logger.info(f"Initialized {split} dataset with {len(self.files)} files")

    def _get_file_list(self, path: str) -> List[str]:
        """Get list of point cloud files."""
        files = []
        for pattern in ["*.las", "*.laz", "*.ply"]:
            files.extend(self.storage.list_files(path, pattern))
        return sorted(files)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample."""
        file_path = self.files[idx]

        # Read point cloud
        data = read_point_cloud(file_path, self.storage)

        # Apply augmentation
        if self.augmentation:
            data = self.augmentation(data)

        # Apply voxelization
        features = {
            "semantic_labels": data.semantic_labels,
            "instance_labels": data.instance_labels,
            "intensity": data.intensity,
        }

        pos, features, _ = grid_sampling(
            data.pos,
            self.config.training.voxel_size,
            features,
        )

        # Prepare features tensor
        feat_list = [pos]  # Use coordinates as features
        if features.get("intensity") is not None:
            feat_list.append(features["intensity"].reshape(-1, 1))

        feature_tensor = np.hstack(feat_list).astype(np.float32)

        # Convert to tensors
        sample = {
            "pos": torch.from_numpy(pos),
            "features": torch.from_numpy(feature_tensor),
            "file_path": file_path,
        }

        if features.get("semantic_labels") is not None:
            sample["semantic_labels"] = torch.from_numpy(
                features["semantic_labels"].astype(np.int64)
            )

        if features.get("instance_labels") is not None:
            sample["instance_labels"] = torch.from_numpy(
                features["instance_labels"].astype(np.int64)
            )

        # Apply additional transforms
        if self.transform:
            sample = self.transform(sample)

        return sample

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Custom collate function for variable-size point clouds."""
        # Create batched coordinates with batch index
        coords_list = []
        features_list = []
        semantic_list = []
        instance_list = []
        file_paths = []

        for batch_idx, sample in enumerate(batch):
            n_points = sample["pos"].shape[0]
            batch_indices = torch.full((n_points, 1), batch_idx, dtype=torch.int32)
            coords = torch.cat([batch_indices, sample["pos"]], dim=1)
            coords_list.append(coords)
            features_list.append(sample["features"])

            if "semantic_labels" in sample:
                semantic_list.append(sample["semantic_labels"])
            if "instance_labels" in sample:
                instance_list.append(sample["instance_labels"])

            file_paths.append(sample["file_path"])

        result = {
            "coords": torch.cat(coords_list, dim=0),
            "features": torch.cat(features_list, dim=0),
            "file_paths": file_paths,
            "batch_size": len(batch),
        }

        if semantic_list:
            result["semantic_labels"] = torch.cat(semantic_list, dim=0)
        if instance_list:
            result["instance_labels"] = torch.cat(instance_list, dim=0)

        return result


def create_dataloader(
    config: Config,
    split: str = "train",
    shuffle: Optional[bool] = None,
) -> DataLoader:
    """
    Create a DataLoader for the specified split.

    Args:
        config: Configuration object
        split: Dataset split (train/val/test/inference)
        shuffle: Override shuffle setting

    Returns:
        DataLoader instance
    """
    # Get appropriate paths
    if split == "train":
        data_paths = config.paths.train
    elif split == "val":
        data_paths = config.paths.val
    elif split == "test":
        data_paths = config.paths.test
    elif split == "inference":
        if config.paths.inference:
            data_paths = DataPaths(
                pointclouds=config.paths.inference.input,
                labels=None,
            )
        else:
            raise ValueError("Inference paths not configured")
    else:
        raise ValueError(f"Unknown split: {split}")

    if data_paths is None:
        raise ValueError(f"Paths not configured for split: {split}")

    dataset = LidarPanopticDataset(
        data_paths=data_paths,
        config=config,
        split=split,
    )

    # Determine shuffle
    if shuffle is None:
        shuffle = split == "train"

    # Determine batch size
    if split == "train":
        batch_size = config.training.batch_size
    else:
        batch_size = config.inference.batch_size

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=config.env.num_workers,
        collate_fn=LidarPanopticDataset.collate_fn,
        pin_memory=True,
        drop_last=(split == "train"),
    )


# Convenience exports
__all__ = [
    "LidarPanopticDataset",
    "PointCloudData",
    "CloudStorageHandler",
    "create_dataloader",
    "read_point_cloud",
    "read_las_file",
    "read_ply_file",
    "grid_sampling",
    "DataAugmentation",
    "CLASSES",
    "CLASSES_INV",
    "VALID_CLASS_IDS",
]
