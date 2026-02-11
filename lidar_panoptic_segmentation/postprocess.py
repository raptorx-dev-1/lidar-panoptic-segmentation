"""
Postprocessing Module for LiDAR Panoptic Segmentation

Provides instance clustering, tree polygon extraction, and output formatting
in GeoJSON, Shapefile, and Parquet formats.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from lidar_panoptic_segmentation.config_schema import (
    Config,
    PolygonOutputFormat,
    PostprocessConfig,
)

logger = logging.getLogger(__name__)


@dataclass
class TreeInstance:
    """Container for a single tree instance."""

    instance_id: int
    points: np.ndarray  # (N, 3) XYZ coordinates
    semantic_label: int = 1  # Tree class
    confidence: float = 1.0
    center: Optional[np.ndarray] = None  # (3,) centroid
    height: Optional[float] = None  # Tree height in meters
    crown_area: Optional[float] = None  # Crown area in square meters
    polygon: Optional[Any] = None  # Shapely Polygon
    properties: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.center is None:
            self.center = self.points.mean(axis=0)
        if self.height is None:
            self.height = float(self.points[:, 2].max() - self.points[:, 2].min())


@dataclass
class PanopticResult:
    """Container for panoptic segmentation results."""

    points: np.ndarray  # (N, 3) XYZ coordinates
    semantic_pred: np.ndarray  # (N,) semantic predictions
    instance_pred: np.ndarray  # (N,) instance predictions
    instances: List[TreeInstance] = field(default_factory=list)
    polygons: List[Any] = field(default_factory=list)  # Shapely Polygons
    metadata: Dict[str, Any] = field(default_factory=dict)


def cluster_instances(
    points: np.ndarray,
    semantic_pred: np.ndarray,
    embeddings: Optional[np.ndarray] = None,
    offset_pred: Optional[np.ndarray] = None,
    config: Optional[PostprocessConfig] = None,
) -> np.ndarray:
    """
    Cluster points into instances using specified method.

    Args:
        points: (N, 3) point coordinates
        semantic_pred: (N,) semantic class predictions
        embeddings: (N, D) instance embeddings
        offset_pred: (N, 3) offset predictions
        config: Postprocessing configuration

    Returns:
        (N,) instance IDs (-1 for background)
    """
    if config is None:
        config = PostprocessConfig()

    # Mask for tree class (assuming class 1 is tree)
    tree_mask = semantic_pred == 1
    if not tree_mask.any():
        return np.full(len(points), -1, dtype=np.int32)

    instance_pred = np.full(len(points), -1, dtype=np.int32)

    # Get tree points
    tree_indices = np.where(tree_mask)[0]
    tree_points = points[tree_mask]

    # Choose clustering features
    if embeddings is not None:
        cluster_features = embeddings[tree_mask]
    elif offset_pred is not None:
        # Shift points by offset to cluster around instance centers
        shifted_points = tree_points + offset_pred[tree_mask]
        cluster_features = shifted_points
    else:
        # Use spatial coordinates
        cluster_features = tree_points[:, :2]  # XY only

    # Perform clustering
    if config.clustering_method == "hdbscan":
        labels = _cluster_hdbscan(
            cluster_features,
            min_cluster_size=config.hdbscan_min_cluster_size,
            min_samples=config.hdbscan_min_samples,
        )
    elif config.clustering_method == "meanshift":
        labels = _cluster_meanshift(
            cluster_features,
            bandwidth=config.meanshift_bandwidth,
        )
    elif config.clustering_method == "dbscan":
        labels = _cluster_dbscan(cluster_features)
    else:
        raise ValueError(f"Unknown clustering method: {config.clustering_method}")

    # Map cluster labels back to full point cloud
    instance_pred[tree_indices] = labels

    # Relabel to consecutive IDs starting from 0
    unique_labels = np.unique(labels[labels >= 0])
    label_map = {old: new for new, old in enumerate(unique_labels)}
    label_map[-1] = -1

    for i, idx in enumerate(tree_indices):
        instance_pred[idx] = label_map.get(labels[i], -1)

    n_instances = len(unique_labels)
    logger.info(f"Clustered {tree_mask.sum()} tree points into {n_instances} instances")

    return instance_pred


def _cluster_hdbscan(
    features: np.ndarray,
    min_cluster_size: int = 50,
    min_samples: int = 10,
) -> np.ndarray:
    """Cluster using HDBSCAN."""
    try:
        import hdbscan
    except ImportError:
        raise ImportError("hdbscan is required: pip install hdbscan")

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
    )
    labels = clusterer.fit_predict(features)
    return labels


def _cluster_meanshift(
    features: np.ndarray,
    bandwidth: Optional[float] = None,
) -> np.ndarray:
    """Cluster using Mean Shift."""
    from sklearn.cluster import MeanShift, estimate_bandwidth

    if bandwidth is None:
        bandwidth = estimate_bandwidth(features, quantile=0.3)

    if bandwidth <= 0:
        bandwidth = 1.0

    clusterer = MeanShift(bandwidth=bandwidth)
    labels = clusterer.fit_predict(features)
    return labels


def _cluster_dbscan(
    features: np.ndarray,
    eps: float = 0.5,
    min_samples: int = 10,
) -> np.ndarray:
    """Cluster using DBSCAN."""
    from sklearn.cluster import DBSCAN

    clusterer = DBSCAN(eps=eps, min_samples=min_samples)
    labels = clusterer.fit_predict(features)
    return labels


def extract_instances(
    points: np.ndarray,
    instance_pred: np.ndarray,
    semantic_pred: np.ndarray,
    config: Optional[PostprocessConfig] = None,
) -> List[TreeInstance]:
    """
    Extract tree instances from predictions.

    Args:
        points: (N, 3) point coordinates
        instance_pred: (N,) instance IDs
        semantic_pred: (N,) semantic predictions
        config: Postprocessing configuration

    Returns:
        List of TreeInstance objects
    """
    if config is None:
        config = PostprocessConfig()

    instances = []
    unique_ids = np.unique(instance_pred)

    for inst_id in unique_ids:
        if inst_id < 0:
            continue

        mask = instance_pred == inst_id
        inst_points = points[mask]
        inst_semantic = semantic_pred[mask]

        # Skip if too few points
        if len(inst_points) < config.hdbscan_min_cluster_size:
            continue

        # Get dominant semantic label
        semantic_label = int(np.bincount(inst_semantic).argmax())

        # Skip if not tree class
        if semantic_label != 1:
            continue

        # Compute height
        height = float(inst_points[:, 2].max() - inst_points[:, 2].min())

        # Apply height filter
        if config.filter_by_height:
            if height < config.min_height or height > config.max_height:
                continue

        # Create instance
        instance = TreeInstance(
            instance_id=int(inst_id),
            points=inst_points,
            semantic_label=semantic_label,
            height=height,
        )
        instances.append(instance)

    logger.info(f"Extracted {len(instances)} valid tree instances")
    return instances


def extract_polygon(
    points: np.ndarray,
    method: str = "convex_hull",
    alpha: float = 0.5,
    simplify_tolerance: float = 0.1,
) -> Optional[Any]:
    """
    Extract 2D polygon from point cloud XY coordinates.

    Args:
        points: (N, 3) point coordinates
        method: Polygon method (convex_hull, alphashape, concave_hull)
        alpha: Alpha parameter for alphashape
        simplify_tolerance: Simplification tolerance

    Returns:
        Shapely Polygon or None
    """
    try:
        from shapely.geometry import MultiPoint, Polygon
        from shapely.validation import make_valid
    except ImportError:
        raise ImportError("shapely is required: pip install shapely")

    if len(points) < 3:
        return None

    # Use XY coordinates only
    xy_points = points[:, :2]

    try:
        if method == "convex_hull":
            mp = MultiPoint(xy_points)
            polygon = mp.convex_hull
        elif method == "alphashape":
            polygon = _compute_alphashape(xy_points, alpha)
        elif method == "concave_hull":
            polygon = _compute_concave_hull(xy_points, alpha)
        else:
            # Default to convex hull
            mp = MultiPoint(xy_points)
            polygon = mp.convex_hull

        if polygon is None or polygon.is_empty:
            return None

        # Ensure valid geometry
        if not polygon.is_valid:
            polygon = make_valid(polygon)

        # Simplify if requested
        if simplify_tolerance > 0:
            polygon = polygon.simplify(simplify_tolerance)

        # Ensure we have a Polygon (not LineString or Point)
        if polygon.geom_type != "Polygon":
            if polygon.geom_type == "MultiPolygon":
                # Take largest polygon
                polygon = max(polygon.geoms, key=lambda p: p.area)
            else:
                return None

        return polygon

    except Exception as e:
        logger.warning(f"Failed to extract polygon: {e}")
        return None


def _compute_alphashape(points: np.ndarray, alpha: float) -> Optional[Any]:
    """Compute alpha shape (concave hull)."""
    try:
        import alphashape
        return alphashape.alphashape(points, alpha)
    except ImportError:
        # Fall back to convex hull
        from shapely.geometry import MultiPoint
        return MultiPoint(points).convex_hull


def _compute_concave_hull(points: np.ndarray, alpha: float) -> Optional[Any]:
    """Compute concave hull using scipy."""
    try:
        from scipy.spatial import ConvexHull
        from shapely.geometry import Polygon

        # Use ConvexHull as fallback
        hull = ConvexHull(points)
        hull_points = points[hull.vertices]
        return Polygon(hull_points)
    except Exception:
        return None


def extract_polygons(
    instances: List[TreeInstance],
    config: Optional[PostprocessConfig] = None,
) -> List[TreeInstance]:
    """
    Extract polygons for all tree instances.

    Args:
        instances: List of tree instances
        config: Postprocessing configuration

    Returns:
        Updated instances with polygons
    """
    if config is None:
        config = PostprocessConfig()

    poly_config = config.polygon
    valid_instances = []

    for instance in instances:
        polygon = extract_polygon(
            instance.points,
            method=poly_config.method,
            alpha=poly_config.alpha,
            simplify_tolerance=poly_config.simplify_tolerance,
        )

        if polygon is None:
            continue

        # Apply minimum area filter
        if polygon.area < poly_config.min_area:
            continue

        # Apply buffer if requested
        if poly_config.buffer_distance > 0:
            polygon = polygon.buffer(poly_config.buffer_distance)

        instance.polygon = polygon
        instance.crown_area = polygon.area
        valid_instances.append(instance)

    logger.info(
        f"Extracted polygons for {len(valid_instances)} / {len(instances)} instances"
    )
    return valid_instances


def save_geojson(
    instances: List[TreeInstance],
    output_path: Union[str, Path],
    crs: Optional[str] = None,
) -> None:
    """
    Save tree instances as GeoJSON.

    Args:
        instances: List of tree instances with polygons
        output_path: Output file path
        crs: Coordinate reference system (e.g., "EPSG:32633")
    """
    features = []

    for instance in instances:
        if instance.polygon is None:
            continue

        # Convert Shapely polygon to GeoJSON geometry
        from shapely.geometry import mapping

        geometry = mapping(instance.polygon)

        # Build properties
        properties = {
            "tree_id": instance.instance_id,
            "height": round(instance.height, 2) if instance.height else None,
            "crown_area": round(instance.crown_area, 2) if instance.crown_area else None,
            "center_x": round(float(instance.center[0]), 2),
            "center_y": round(float(instance.center[1]), 2),
            "center_z": round(float(instance.center[2]), 2),
            "num_points": len(instance.points),
            "confidence": round(instance.confidence, 3),
        }
        properties.update(instance.properties)

        feature = {
            "type": "Feature",
            "geometry": geometry,
            "properties": properties,
        }
        features.append(feature)

    geojson = {
        "type": "FeatureCollection",
        "features": features,
    }

    # Add CRS if provided
    if crs:
        geojson["crs"] = {
            "type": "name",
            "properties": {"name": crs},
        }

    # Save to file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(geojson, f, indent=2)

    logger.info(f"Saved {len(features)} tree polygons to {output_path}")


def save_shapefile(
    instances: List[TreeInstance],
    output_path: Union[str, Path],
    crs: Optional[str] = None,
) -> None:
    """
    Save tree instances as Shapefile.

    Args:
        instances: List of tree instances with polygons
        output_path: Output file path (without extension)
        crs: Coordinate reference system
    """
    try:
        import geopandas as gpd
        from shapely.geometry import Polygon
    except ImportError:
        raise ImportError(
            "geopandas is required for shapefile export: pip install geopandas"
        )

    # Build GeoDataFrame
    data = []
    geometries = []

    for instance in instances:
        if instance.polygon is None:
            continue

        data.append({
            "tree_id": instance.instance_id,
            "height": instance.height,
            "crown_area": instance.crown_area,
            "center_x": float(instance.center[0]),
            "center_y": float(instance.center[1]),
            "center_z": float(instance.center[2]),
            "num_points": len(instance.points),
            "confidence": instance.confidence,
        })
        geometries.append(instance.polygon)

    if not data:
        logger.warning("No valid instances to save")
        return

    gdf = gpd.GeoDataFrame(data, geometry=geometries, crs=crs)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    gdf.to_file(str(output_path))
    logger.info(f"Saved {len(data)} tree polygons to {output_path}")


def save_parquet(
    instances: List[TreeInstance],
    output_path: Union[str, Path],
    include_points: bool = False,
) -> None:
    """
    Save tree instances as Parquet.

    Args:
        instances: List of tree instances
        output_path: Output file path
        include_points: Whether to include point coordinates
    """
    try:
        import pandas as pd
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        raise ImportError(
            "pyarrow is required for parquet export: pip install pyarrow"
        )

    records = []

    for instance in instances:
        record = {
            "tree_id": instance.instance_id,
            "height": instance.height,
            "crown_area": instance.crown_area,
            "center_x": float(instance.center[0]),
            "center_y": float(instance.center[1]),
            "center_z": float(instance.center[2]),
            "num_points": len(instance.points),
            "confidence": instance.confidence,
        }

        # Add polygon as WKT
        if instance.polygon is not None:
            record["polygon_wkt"] = instance.polygon.wkt

        # Optionally include point cloud
        if include_points:
            record["points_x"] = instance.points[:, 0].tolist()
            record["points_y"] = instance.points[:, 1].tolist()
            record["points_z"] = instance.points[:, 2].tolist()

        records.append(record)

    df = pd.DataFrame(records)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(output_path, engine="pyarrow")
    logger.info(f"Saved {len(records)} tree instances to {output_path}")


def save_point_cloud(
    points: np.ndarray,
    semantic_pred: np.ndarray,
    instance_pred: np.ndarray,
    output_path: Union[str, Path],
    file_format: str = "ply",
) -> None:
    """
    Save segmented point cloud.

    Args:
        points: (N, 3) point coordinates
        semantic_pred: (N,) semantic predictions
        instance_pred: (N,) instance predictions
        output_path: Output file path
        file_format: Output format (ply, las)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if file_format == "ply":
        _save_ply(points, semantic_pred, instance_pred, output_path)
    elif file_format in ["las", "laz"]:
        _save_las(points, semantic_pred, instance_pred, output_path)
    else:
        raise ValueError(f"Unknown file format: {file_format}")


def _save_ply(
    points: np.ndarray,
    semantic_pred: np.ndarray,
    instance_pred: np.ndarray,
    output_path: Path,
) -> None:
    """Save as PLY file."""
    try:
        from plyfile import PlyData, PlyElement
    except ImportError:
        raise ImportError("plyfile is required: pip install plyfile")

    # Create structured array
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("semantic_seg", "i4"),
        ("instance_id", "i4"),
    ]
    vertex = np.empty(len(points), dtype=dtype)
    vertex["x"] = points[:, 0]
    vertex["y"] = points[:, 1]
    vertex["z"] = points[:, 2]
    vertex["semantic_seg"] = semantic_pred
    vertex["instance_id"] = instance_pred

    el = PlyElement.describe(vertex, "vertex")
    PlyData([el]).write(str(output_path))
    logger.info(f"Saved point cloud to {output_path}")


def _save_las(
    points: np.ndarray,
    semantic_pred: np.ndarray,
    instance_pred: np.ndarray,
    output_path: Path,
) -> None:
    """Save as LAS file."""
    try:
        import laspy
    except ImportError:
        raise ImportError("laspy is required: pip install laspy")

    # Create LAS file
    las = laspy.create(point_format=6, file_version="1.4")
    las.x = points[:, 0]
    las.y = points[:, 1]
    las.z = points[:, 2]
    las.classification = semantic_pred.astype(np.uint8)

    # Add instance ID as extra dimension
    las.add_extra_dim(laspy.ExtraBytesParams(name="tree_id", type=np.int32))
    las.tree_id = instance_pred

    las.write(str(output_path))
    logger.info(f"Saved point cloud to {output_path}")


def postprocess_predictions(
    points: np.ndarray,
    semantic_pred: np.ndarray,
    embeddings: Optional[np.ndarray] = None,
    offset_pred: Optional[np.ndarray] = None,
    config: Optional[Config] = None,
) -> PanopticResult:
    """
    Complete postprocessing pipeline.

    Args:
        points: (N, 3) point coordinates
        semantic_pred: (N,) semantic predictions
        embeddings: (N, D) instance embeddings
        offset_pred: (N, 3) offset predictions
        config: Full configuration

    Returns:
        PanopticResult with instances and polygons
    """
    if config is None:
        from lidar_panoptic_segmentation.config_schema import Config
        config = Config()

    postprocess_config = config.postprocess

    # Cluster instances
    instance_pred = cluster_instances(
        points=points,
        semantic_pred=semantic_pred,
        embeddings=embeddings,
        offset_pred=offset_pred,
        config=postprocess_config,
    )

    # Extract instances
    instances = extract_instances(
        points=points,
        instance_pred=instance_pred,
        semantic_pred=semantic_pred,
        config=postprocess_config,
    )

    # Extract polygons
    instances = extract_polygons(instances, config=postprocess_config)

    return PanopticResult(
        points=points,
        semantic_pred=semantic_pred,
        instance_pred=instance_pred,
        instances=instances,
        polygons=[inst.polygon for inst in instances if inst.polygon],
    )


def save_results(
    result: PanopticResult,
    output_dir: Union[str, Path],
    config: Optional[Config] = None,
    base_name: str = "result",
) -> Dict[str, Path]:
    """
    Save all results to output directory.

    Args:
        result: Panoptic result
        output_dir: Output directory
        config: Configuration
        base_name: Base name for output files

    Returns:
        Dictionary of output file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_paths = {}

    # Determine output format
    if config:
        output_format = config.inference.polygon_output_format.value
    else:
        output_format = "geojson"

    # Save polygons
    if result.instances:
        if output_format == "geojson":
            polygon_path = output_dir / f"{base_name}_polygons.geojson"
            save_geojson(result.instances, polygon_path)
            output_paths["polygons"] = polygon_path
        elif output_format == "shapefile":
            polygon_path = output_dir / f"{base_name}_polygons.shp"
            save_shapefile(result.instances, polygon_path)
            output_paths["polygons"] = polygon_path
        elif output_format == "parquet":
            polygon_path = output_dir / f"{base_name}_instances.parquet"
            save_parquet(result.instances, polygon_path)
            output_paths["polygons"] = polygon_path

    # Save point cloud
    if config and config.inference.save_point_clouds:
        pc_path = output_dir / f"{base_name}_segmented.ply"
        save_point_cloud(
            result.points,
            result.semantic_pred,
            result.instance_pred,
            pc_path,
        )
        output_paths["point_cloud"] = pc_path

    logger.info(f"Saved results to {output_dir}")
    return output_paths


# Convenience exports
__all__ = [
    "TreeInstance",
    "PanopticResult",
    "cluster_instances",
    "extract_instances",
    "extract_polygon",
    "extract_polygons",
    "save_geojson",
    "save_shapefile",
    "save_parquet",
    "save_point_cloud",
    "postprocess_predictions",
    "save_results",
]
