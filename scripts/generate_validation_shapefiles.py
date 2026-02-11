#!/usr/bin/env python3
"""
Generate Shapefiles for Validation Results

Creates shapefiles with both predicted and ground truth tree polygons
for visual comparison and GIS analysis.

Usage:
    python scripts/generate_validation_shapefiles.py \
        --input data/validation \
        --output validation_results_extended \
        --model model_file/PointGroup-PAPER.pt
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

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

    if 'semantic_seg' in v.data.dtype.names:
        data['semantic_gt'] = v['semantic_seg'].astype(np.int32)
    else:
        data['semantic_gt'] = None

    if 'treeID' in v.data.dtype.names:
        data['instance_gt'] = v['treeID'].astype(np.int32)
    elif 'instance_gt' in v.data.dtype.names:
        data['instance_gt'] = v['instance_gt'].astype(np.int32)
    else:
        data['instance_gt'] = None

    return data


def instances_to_tree_instances(
    pos: np.ndarray,
    instance_labels: np.ndarray,
    semantic_labels: np.ndarray,
    prefix: str = "",
) -> List:
    """Convert instance arrays to TreeInstance objects."""
    from lidar_panoptic_segmentation.postprocess import TreeInstance

    instances = []
    unique_ids = np.unique(instance_labels)
    unique_ids = unique_ids[unique_ids > 0]  # Skip background (0)

    for inst_id in unique_ids:
        mask = instance_labels == inst_id
        points = pos[mask]

        if len(points) < 3:
            continue

        # Calculate properties
        height = points[:, 2].max() - points[:, 2].min()
        centroid = points.mean(axis=0)

        instance = TreeInstance(
            instance_id=int(inst_id),
            points=points,
            semantic_label=1,  # tree
            confidence=1.0,
            height=height,
            center=centroid,
        )
        instances.append(instance)

    return instances


def generate_validation_shapefile(
    input_file: Path,
    model_path: Path,
    output_dir: Path,
) -> Optional[Path]:
    """
    Generate shapefile with both predicted and ground truth polygons.

    Returns path to the generated shapefile.
    """
    try:
        import geopandas as gpd
        from shapely.geometry import Polygon, MultiPoint
        from scipy.spatial import ConvexHull
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        logger.error("Install with: pip install geopandas shapely")
        return None

    from lidar_panoptic_segmentation.postprocess import (
        TreeInstance, extract_polygon, PostprocessConfig
    )
    from lidar_panoptic_segmentation.native_inference import NativeInferencePipeline

    logger.info(f"Processing: {input_file.name}")

    # Load ground truth
    data = load_labeled_ply(input_file)
    pos = data['pos']
    gt_semantic = data['semantic_gt']
    gt_instances = data['instance_gt']

    if gt_instances is None:
        logger.warning(f"No instance labels in {input_file}")
        return None

    # Run inference
    logger.info("Running model inference...")
    segmenter = NativeInferencePipeline(model_path)
    result = segmenter.segment_points(pos)

    pred_pos = result.points
    pred_semantic = result.semantic_pred
    pred_instances = result.instance_pred

    # Interpolate predictions to original points if needed
    if len(pred_pos) != len(pos):
        from scipy.spatial import cKDTree
        tree = cKDTree(pred_pos)
        _, indices = tree.query(pos, k=1)
        pred_semantic_full = pred_semantic[indices]
        pred_instances_full = pred_instances[indices]
    else:
        pred_semantic_full = pred_semantic
        pred_instances_full = pred_instances

    # Generate polygons for ground truth instances
    logger.info("Generating ground truth polygons...")
    gt_tree_instances = instances_to_tree_instances(
        pos, gt_instances, gt_semantic, prefix="gt_"
    )

    # Generate polygons for predicted instances
    logger.info("Generating predicted polygons...")
    pred_tree_instances = instances_to_tree_instances(
        pos, pred_instances_full, pred_semantic_full, prefix="pred_"
    )

    # Extract polygons
    config = PostprocessConfig()

    gt_polygons = []
    gt_attrs = []
    for inst in gt_tree_instances:
        poly = extract_polygon(inst.points, method="convex_hull")
        if poly is not None and poly.area > 0.1:
            gt_polygons.append(poly)
            gt_attrs.append({
                'id': inst.instance_id,
                'type': 'ground_truth',
                'height': float(inst.height),
                'area': float(poly.area),
                'n_points': len(inst.points),
            })

    pred_polygons = []
    pred_attrs = []
    for inst in pred_tree_instances:
        poly = extract_polygon(inst.points, method="convex_hull")
        if poly is not None and poly.area > 0.1:
            pred_polygons.append(poly)
            pred_attrs.append({
                'id': inst.instance_id,
                'type': 'prediction',
                'height': float(inst.height),
                'area': float(poly.area),
                'n_points': len(inst.points),
            })

    logger.info(f"  GT polygons: {len(gt_polygons)}")
    logger.info(f"  Predicted polygons: {len(pred_polygons)}")

    # Combine into single GeoDataFrame
    all_polygons = gt_polygons + pred_polygons
    all_attrs = gt_attrs + pred_attrs

    if not all_polygons:
        logger.warning("No polygons generated")
        return None

    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(all_attrs, geometry=all_polygons)

    # Save shapefile
    output_path = output_dir / f"{input_file.stem}_polygons.shp"
    gdf.to_file(output_path)
    logger.info(f"Saved: {output_path}")

    # Also save separate shapefiles for GT and predictions
    if gt_polygons:
        gt_gdf = gpd.GeoDataFrame(gt_attrs, geometry=gt_polygons)
        gt_path = output_dir / f"{input_file.stem}_gt_polygons.shp"
        gt_gdf.to_file(gt_path)
        logger.info(f"Saved: {gt_path}")

    if pred_polygons:
        pred_gdf = gpd.GeoDataFrame(pred_attrs, geometry=pred_polygons)
        pred_path = output_dir / f"{input_file.stem}_pred_polygons.shp"
        pred_gdf.to_file(pred_path)
        logger.info(f"Saved: {pred_path}")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate validation shapefiles with GT and predicted polygons"
    )
    parser.add_argument(
        "-i", "--input", required=True,
        help="Input file or directory with labeled PLY files"
    )
    parser.add_argument(
        "-o", "--output", required=True,
        help="Output directory for shapefiles"
    )
    parser.add_argument(
        "-m", "--model", default="model_file/PointGroup-PAPER.pt",
        help="Model path"
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    model_path = Path(args.model)

    output_path.mkdir(parents=True, exist_ok=True)

    # Find input files
    if input_path.is_file():
        files = [input_path]
    else:
        files = sorted(input_path.glob("*.ply"))

    if not files:
        logger.error(f"No PLY files found in {input_path}")
        return

    logger.info(f"Processing {len(files)} files...")

    for f in files:
        try:
            generate_validation_shapefile(f, model_path, output_path)
        except Exception as e:
            logger.error(f"Failed to process {f.name}: {e}")
            import traceback
            traceback.print_exc()

    logger.info("Done!")


if __name__ == "__main__":
    main()
