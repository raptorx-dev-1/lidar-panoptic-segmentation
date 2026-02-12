#!/usr/bin/env python3
"""
LiDAR Panoptic Segmentation Inference CLI

Provides command-line interface for running inference on LiDAR point clouds.
Supports batch processing, tile-based inference for large files, and
multiple output formats.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from tqdm import tqdm

from lidar_panoptic_segmentation.config import (
    Config,
    add_config_args,
    load_config,
    validate_config,
)
from lidar_panoptic_segmentation.dataset import (
    CloudStorageHandler,
    PointCloudData,
    grid_sampling,
    read_point_cloud,
)
from lidar_panoptic_segmentation.logging_utils import setup_logging
from lidar_panoptic_segmentation.model import LidarPanopticModel, load_model
from lidar_panoptic_segmentation.postprocess import (
    PanopticResult,
    postprocess_predictions,
    save_results,
)

logger = logging.getLogger(__name__)


class InferencePipeline:
    """Inference pipeline for LiDAR panoptic segmentation."""

    def __init__(self, config: Config, model: Optional[LidarPanopticModel] = None):
        """
        Initialize inference pipeline.

        Args:
            config: Configuration object
            model: Pre-loaded model (optional)
        """
        self.config = config
        self.device = config.get_effective_device()

        # Setup logging
        setup_logging(config)

        # Validate configuration
        warnings = validate_config(config)
        for warning in warnings:
            logger.warning(warning)

        # Initialize storage handler
        is_databricks = config.env.name.value == "databricks"
        self.storage = CloudStorageHandler(is_databricks=is_databricks)

        # Load or use provided model
        if model is not None:
            self.model = model
        else:
            self.model = self._load_model()

        self.model.eval()

    def _load_model(self) -> LidarPanopticModel:
        """Load model from configuration."""
        model_uri = self.config.inference.model_uri

        logger.info(f"Loading model from: {model_uri}")

        model = load_model(
            model_uri=model_uri,
            config=self.config,
            device=self.device,
        )

        return model

    def process_file(
        self,
        input_path: str,
        output_dir: Optional[str] = None,
    ) -> PanopticResult:
        """
        Process a single point cloud file.

        Args:
            input_path: Path to input point cloud
            output_dir: Output directory (uses config if None)

        Returns:
            PanopticResult with predictions and polygons
        """
        logger.info(f"Processing: {input_path}")
        start_time = time.time()

        # Read point cloud
        data = read_point_cloud(input_path, self.storage)
        logger.info(f"Read {len(data.points)} points")

        # Check if tiling is needed
        if self._needs_tiling(data):
            result = self._process_tiled(data)
        else:
            result = self._process_single(data)

        # Save results
        if output_dir is None:
            output_dir = self.config.paths.inference.output

        base_name = Path(input_path).stem
        output_paths = save_results(
            result=result,
            output_dir=output_dir,
            config=self.config,
            base_name=base_name,
        )

        elapsed = time.time() - start_time
        logger.info(
            f"Processed {len(data.points)} points in {elapsed:.1f}s | "
            f"Found {len(result.instances)} tree instances"
        )

        result.metadata["input_path"] = input_path
        result.metadata["output_paths"] = output_paths
        result.metadata["processing_time"] = elapsed

        return result

    def _needs_tiling(self, data: PointCloudData) -> bool:
        """Check if point cloud needs tile-based processing."""
        if data.bounds is None:
            return False

        extent_x = data.bounds[1, 0] - data.bounds[0, 0]
        extent_y = data.bounds[1, 1] - data.bounds[0, 1]
        tile_size = self.config.inference.tile_size

        return extent_x > tile_size or extent_y > tile_size

    def _process_single(self, data: PointCloudData) -> PanopticResult:
        """Process point cloud in a single pass."""
        # Prepare features
        pos = data.pos

        # Apply voxelization
        features_dict = {"intensity": data.intensity}
        pos, features_dict, indices = grid_sampling(
            pos,
            self.config.training.voxel_size,
            features_dict,
        )

        # Prepare feature tensor
        if features_dict.get("intensity") is not None:
            features = np.hstack([
                pos,
                features_dict["intensity"].reshape(-1, 1)
            ]).astype(np.float32)
        else:
            features = pos.astype(np.float32)

        # Create batch with single sample
        batch_indices = np.zeros((len(pos), 1), dtype=np.int32)
        coords = np.hstack([batch_indices, pos]).astype(np.float32)

        # Convert to tensors
        coords_tensor = torch.from_numpy(coords).to(self.device)
        features_tensor = torch.from_numpy(features).to(self.device)

        # Run inference
        with torch.no_grad():
            output = self.model(coords_tensor, features_tensor)

        # Convert outputs to numpy
        semantic_pred = output.semantic_pred.cpu().numpy()
        embeddings = output.embedding.cpu().numpy() if output.embedding is not None else None
        offset_pred = output.offset_pred.cpu().numpy() if output.offset_pred is not None else None
        instance_pred = output.instance_pred.cpu().numpy() if output.instance_pred is not None else None

        # Postprocess
        result = postprocess_predictions(
            points=pos,
            semantic_pred=semantic_pred,
            embeddings=embeddings,
            offset_pred=offset_pred,
            instance_pred=instance_pred,
            config=self.config,
        )

        return result

    def _process_tiled(self, data: PointCloudData) -> PanopticResult:
        """Process large point cloud using tiling."""
        logger.info("Using tile-based processing for large point cloud")

        tile_size = self.config.inference.tile_size
        overlap = self.config.inference.overlap

        # Generate tiles
        tiles = self._generate_tiles(data.bounds, tile_size, overlap)
        logger.info(f"Processing {len(tiles)} tiles")

        # Process each tile
        all_points = []
        all_semantic = []
        all_instance = []
        instance_offset = 0

        for tile_idx, tile_bounds in enumerate(tqdm(tiles, desc="Processing tiles")):
            # Extract points in tile
            mask = self._points_in_bounds(data.pos, tile_bounds)
            if not mask.any():
                continue

            tile_data = PointCloudData(
                pos=data.pos[mask],
                intensity=data.intensity[mask] if data.intensity is not None else None,
            )

            # Process tile
            tile_result = self._process_single(tile_data)

            # Offset instance IDs to avoid conflicts
            tile_instance = tile_result.instance_pred.copy()
            valid_mask = tile_instance >= 0
            tile_instance[valid_mask] += instance_offset
            instance_offset = tile_instance.max() + 1 if valid_mask.any() else instance_offset

            all_points.append(tile_result.points)
            all_semantic.append(tile_result.semantic_pred)
            all_instance.append(tile_instance)

        # Merge results
        if not all_points:
            return PanopticResult(
                points=data.pos,
                semantic_pred=np.zeros(len(data.pos), dtype=np.int64),
                instance_pred=np.full(len(data.pos), -1, dtype=np.int32),
            )

        merged_points = np.vstack(all_points)
        merged_semantic = np.concatenate(all_semantic)
        merged_instance = np.concatenate(all_instance)

        # Optionally merge instances across tile boundaries
        if self.config.inference.merge_instances:
            merged_instance = self._merge_boundary_instances(
                merged_points,
                merged_instance,
                tile_size * overlap,
            )

        # Final postprocessing
        result = postprocess_predictions(
            points=merged_points,
            semantic_pred=merged_semantic,
            config=self.config,
        )
        result.instance_pred = merged_instance

        return result

    def _generate_tiles(
        self,
        bounds: np.ndarray,
        tile_size: float,
        overlap: float,
    ) -> List[np.ndarray]:
        """Generate tile bounds for processing."""
        min_x, min_y = bounds[0, 0], bounds[0, 1]
        max_x, max_y = bounds[1, 0], bounds[1, 1]

        step = tile_size * (1 - overlap)
        tiles = []

        x = min_x
        while x < max_x:
            y = min_y
            while y < max_y:
                tile_bounds = np.array([
                    [x, y, bounds[0, 2]],
                    [x + tile_size, y + tile_size, bounds[1, 2]],
                ])
                tiles.append(tile_bounds)
                y += step
            x += step

        return tiles

    def _points_in_bounds(
        self,
        points: np.ndarray,
        bounds: np.ndarray,
    ) -> np.ndarray:
        """Get mask for points within bounds."""
        mask = (
            (points[:, 0] >= bounds[0, 0]) &
            (points[:, 0] < bounds[1, 0]) &
            (points[:, 1] >= bounds[0, 1]) &
            (points[:, 1] < bounds[1, 1])
        )
        return mask

    def _merge_boundary_instances(
        self,
        points: np.ndarray,
        instance_pred: np.ndarray,
        merge_distance: float,
    ) -> np.ndarray:
        """Merge instances that span tile boundaries."""
        from scipy.spatial import cKDTree

        unique_instances = np.unique(instance_pred)
        unique_instances = unique_instances[unique_instances >= 0]

        if len(unique_instances) <= 1:
            return instance_pred

        # Compute instance centroids
        centroids = []
        instance_ids = []
        for inst_id in unique_instances:
            mask = instance_pred == inst_id
            centroid = points[mask].mean(axis=0)
            centroids.append(centroid)
            instance_ids.append(inst_id)

        centroids = np.array(centroids)
        instance_ids = np.array(instance_ids)

        # Find nearby centroids
        tree = cKDTree(centroids[:, :2])
        pairs = tree.query_pairs(merge_distance)

        # Build merge mapping
        merge_map = {i: i for i in instance_ids}
        for i, j in pairs:
            id_i, id_j = instance_ids[i], instance_ids[j]
            # Merge to lower ID
            root_i = merge_map[id_i]
            root_j = merge_map[id_j]
            if root_i != root_j:
                new_root = min(root_i, root_j)
                old_root = max(root_i, root_j)
                for k in merge_map:
                    if merge_map[k] == old_root:
                        merge_map[k] = new_root

        # Apply merging
        result = instance_pred.copy()
        for old_id, new_id in merge_map.items():
            if old_id != new_id:
                result[instance_pred == old_id] = new_id

        # Relabel to consecutive IDs
        unique_new = np.unique(result)
        unique_new = unique_new[unique_new >= 0]
        relabel_map = {old: new for new, old in enumerate(unique_new)}
        relabel_map[-1] = -1

        for i in range(len(result)):
            result[i] = relabel_map.get(result[i], result[i])

        n_merged = len(unique_instances) - len(unique_new)
        if n_merged > 0:
            logger.info(f"Merged {n_merged} boundary instances")

        return result

    def process_directory(
        self,
        input_dir: str,
        output_dir: Optional[str] = None,
        recursive: bool = True,
    ) -> List[PanopticResult]:
        """
        Process all point cloud files in a directory.

        Args:
            input_dir: Input directory path
            output_dir: Output directory (uses config if None)
            recursive: Process subdirectories

        Returns:
            List of PanopticResult objects
        """
        # Get file list
        files = []
        for pattern in ["*.las", "*.laz", "*.ply"]:
            files.extend(self.storage.list_files(input_dir, pattern, recursive))

        if not files:
            logger.warning(f"No point cloud files found in {input_dir}")
            return []

        logger.info(f"Found {len(files)} files to process")

        if output_dir is None:
            output_dir = self.config.paths.inference.output

        results = []
        for file_path in tqdm(files, desc="Processing files"):
            try:
                result = self.process_file(file_path, output_dir)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
                if self.config.env.debug:
                    raise

        logger.info(f"Processed {len(results)} / {len(files)} files successfully")
        return results


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for inference CLI."""
    parser = argparse.ArgumentParser(
        description="LiDAR Panoptic Segmentation Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Add standard config args
    add_config_args(parser)

    # Input/output
    parser.add_argument(
        "input",
        type=str,
        nargs="?",
        help="Input file or directory path",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        help="Output directory",
    )

    # Model
    parser.add_argument(
        "-m", "--model",
        type=str,
        help="Model path or MLflow URI (overrides config)",
    )

    # Processing options
    parser.add_argument(
        "--tile-size",
        type=float,
        help="Tile size for large point clouds",
    )
    parser.add_argument(
        "--no-merge",
        action="store_true",
        help="Disable merging instances across tile boundaries",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["geojson", "shapefile", "parquet"],
        help="Output polygon format",
    )

    # Processing mode
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Process directories recursively",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size for inference",
    )

    return parser


def main(args: Optional[argparse.Namespace] = None) -> int:
    """
    Main inference entry point.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code
    """
    if args is None:
        parser = create_parser()
        args = parser.parse_args()

    # Build config overrides
    overrides = {}

    if args.model:
        if "inference" not in overrides:
            overrides["inference"] = {}
        overrides["inference"]["model_uri"] = args.model

    if args.output:
        if "paths" not in overrides:
            overrides["paths"] = {}
        if "inference" not in overrides["paths"]:
            overrides["paths"]["inference"] = {}
        overrides["paths"]["inference"]["output"] = args.output

    if args.tile_size:
        if "inference" not in overrides:
            overrides["inference"] = {}
        overrides["inference"]["tile_size"] = args.tile_size

    if args.no_merge:
        if "inference" not in overrides:
            overrides["inference"] = {}
        overrides["inference"]["merge_instances"] = False

    if args.format:
        if "inference" not in overrides:
            overrides["inference"] = {}
        overrides["inference"]["polygon_output_format"] = args.format

    if args.batch_size:
        if "inference" not in overrides:
            overrides["inference"] = {}
        overrides["inference"]["batch_size"] = args.batch_size

    # Load configuration
    try:
        config = load_config(
            config_path=getattr(args, "config", None),
            overrides=overrides,
            environment=getattr(args, "environment", None),
        )
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return 1

    # Get input path
    input_path = getattr(args, "input", None)
    if input_path is None:
        if config.paths.inference:
            input_path = config.paths.inference.input
        else:
            print("No input path specified")
            return 1

    # Create pipeline
    try:
        pipeline = InferencePipeline(config)

        # Process input
        input_path_obj = Path(input_path)

        if input_path_obj.is_file():
            result = pipeline.process_file(input_path, args.output)
            print(f"Found {len(result.instances)} tree instances")
        elif input_path_obj.is_dir() or input_path.startswith("abfss://"):
            results = pipeline.process_directory(
                input_path,
                args.output,
                recursive=getattr(args, "recursive", False),
            )
            total_instances = sum(len(r.instances) for r in results)
            print(f"Processed {len(results)} files, found {total_instances} tree instances")
        else:
            print(f"Input path not found: {input_path}")
            return 1

        return 0

    except KeyboardInterrupt:
        logger.info("Inference interrupted by user")
        return 130
    except Exception as e:
        logger.exception(f"Inference failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
