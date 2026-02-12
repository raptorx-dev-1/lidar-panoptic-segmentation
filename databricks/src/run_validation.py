"""Databricks entry point for polygon validation against field data."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from databricks.src.common import (
    load_databricks_config,
    parse_common_args,
    setup_environment,
)
from lidar_panoptic_segmentation.evaluation import evaluate_polygons, sanitize_for_json
from lidar_panoptic_segmentation.logging_utils import experiment_context

logger = logging.getLogger(__name__)


def _load_geojson_polygons(path: str):
    """Load Shapely polygons from a GeoJSON file."""
    from shapely.geometry import shape

    with open(path) as f:
        data = json.load(f)

    polygons = []
    features = data.get("features", [data] if data.get("geometry") else [])
    for feat in features:
        geom = feat.get("geometry") or feat
        polygons.append(shape(geom))
    return polygons


def main() -> int:
    args = parse_common_args()
    config = load_databricks_config(args)
    setup_environment(config)

    pred_dir = Path(config.paths.inference.output)
    gt_dir = Path(f"/Volumes/{args.catalog}/{args.schema}/field_validation/polygons")

    pred_files = {p.stem: p for p in sorted(pred_dir.glob("*.geojson"))}
    gt_files = {p.stem: p for p in sorted(gt_dir.glob("*.geojson"))}

    matched_stems = sorted(set(pred_files) & set(gt_files))
    if not matched_stems:
        logger.warning("No matching prediction/ground-truth file pairs found.")
        return 0

    logger.info("Matched %d file pairs for validation.", len(matched_stems))

    all_metrics = []
    for stem in matched_stems:
        pred_polygons = _load_geojson_polygons(str(pred_files[stem]))
        gt_polygons = _load_geojson_polygons(str(gt_files[stem]))
        metrics = evaluate_polygons(pred_polygons, gt_polygons)
        metrics["file"] = stem
        all_metrics.append(metrics)
        logger.info("%s — P=%.3f R=%.3f F1=%.3f",
                     stem, metrics["precision"], metrics["recall"], metrics["f1_score"])

    n = len(all_metrics)
    aggregate = {
        "precision": sum(m["precision"] for m in all_metrics) / n,
        "recall": sum(m["recall"] for m in all_metrics) / n,
        "f1_score": sum(m["f1_score"] for m in all_metrics) / n,
        "num_files": n,
        "per_file": all_metrics,
    }
    logger.info(
        "Aggregate — P=%.3f R=%.3f F1=%.3f",
        aggregate["precision"],
        aggregate["recall"],
        aggregate["f1_score"],
    )

    output_path = pred_dir / "validation_metrics.json"
    sanitized = sanitize_for_json(aggregate)
    with open(output_path, "w") as f:
        json.dump(sanitized, f, indent=2)
    logger.info("Saved validation metrics to %s", output_path)

    with experiment_context(config, run_name="polygon_validation") as exp_logger:
        exp_logger.log_metrics({
            "val/polygon_precision": aggregate["precision"],
            "val/polygon_recall": aggregate["recall"],
            "val/polygon_f1": aggregate["f1_score"],
        })
        exp_logger.log_artifact(str(output_path))

    return 0


if __name__ == "__main__":
    sys.exit(main())
