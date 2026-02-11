"""
Tests for Postprocessing Module

Tests polygon extraction, instance clustering, and output formatting.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from lidar_panoptic_segmentation.config_schema import (
    Config,
    PolygonConfig,
    PostprocessConfig,
)
from lidar_panoptic_segmentation.postprocess import (
    PanopticResult,
    TreeInstance,
    cluster_instances,
    extract_instances,
    extract_polygon,
    extract_polygons,
    postprocess_predictions,
    save_geojson,
    save_parquet,
    save_results,
)


class TestTreeInstance:
    """Tests for TreeInstance dataclass."""

    def test_create_instance(self):
        """Test creating a tree instance."""
        points = np.random.rand(100, 3).astype(np.float32)
        instance = TreeInstance(instance_id=1, points=points)

        assert instance.instance_id == 1
        assert len(instance.points) == 100
        assert instance.center is not None
        assert instance.height is not None

    def test_compute_center(self):
        """Test automatic center computation."""
        points = np.array([
            [0, 0, 0],
            [2, 0, 0],
            [0, 2, 0],
            [2, 2, 0],
        ], dtype=np.float32)
        instance = TreeInstance(instance_id=1, points=points)

        assert np.allclose(instance.center, [1, 1, 0], atol=0.01)

    def test_compute_height(self):
        """Test automatic height computation."""
        points = np.array([
            [0, 0, 0],
            [0, 0, 10],
            [0, 0, 5],
        ], dtype=np.float32)
        instance = TreeInstance(instance_id=1, points=points)

        assert instance.height == 10.0


class TestPolygonExtraction:
    """Tests for polygon extraction."""

    def test_convex_hull_extraction(self):
        """Test convex hull polygon extraction."""
        # Create points in a square pattern
        points = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0.5, 0.5, 0],  # Interior point
        ], dtype=np.float32)

        polygon = extract_polygon(points, method="convex_hull")

        assert polygon is not None
        assert polygon.is_valid
        assert polygon.area > 0
        # Convex hull of unit square should have area ~1
        assert 0.9 < polygon.area < 1.1

    def test_polygon_too_few_points(self):
        """Test that polygon extraction fails with too few points."""
        points = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
        polygon = extract_polygon(points)
        assert polygon is None

    def test_polygon_simplification(self):
        """Test polygon simplification."""
        # Create circular points
        theta = np.linspace(0, 2 * np.pi, 100)
        x = np.cos(theta)
        y = np.sin(theta)
        z = np.zeros_like(x)
        points = np.stack([x, y, z], axis=1).astype(np.float32)

        polygon = extract_polygon(points, simplify_tolerance=0.1)

        assert polygon is not None
        # Simplified polygon should have fewer vertices
        assert len(polygon.exterior.coords) < 100

    def test_extract_polygons_filtering(self):
        """Test polygon extraction with area filtering."""
        # Create small and large instances
        small_points = np.array([
            [0, 0, 0], [0.1, 0, 0], [0, 0.1, 0], [0.1, 0.1, 0]
        ], dtype=np.float32)

        large_points = np.array([
            [0, 0, 0], [5, 0, 0], [5, 5, 0], [0, 5, 0]
        ], dtype=np.float32)

        instances = [
            TreeInstance(instance_id=1, points=small_points),
            TreeInstance(instance_id=2, points=large_points),
        ]

        config = PostprocessConfig(
            polygon=PolygonConfig(min_area=1.0)
        )

        result = extract_polygons(instances, config)

        # Only large instance should pass
        assert len(result) == 1
        assert result[0].instance_id == 2


class TestInstanceClustering:
    """Tests for instance clustering."""

    def test_cluster_instances_hdbscan(self):
        """Test HDBSCAN clustering."""
        pytest.importorskip("hdbscan")

        # Create two distinct clusters
        cluster1 = np.random.randn(50, 3).astype(np.float32)
        cluster2 = np.random.randn(50, 3).astype(np.float32) + 10

        points = np.vstack([cluster1, cluster2])
        semantic_pred = np.ones(100, dtype=np.int64)  # All tree class

        config = PostprocessConfig(
            clustering_method="hdbscan",
            hdbscan_min_cluster_size=10,
            hdbscan_min_samples=5,
        )

        instance_pred = cluster_instances(
            points=points,
            semantic_pred=semantic_pred,
            config=config,
        )

        # Should find at least 2 clusters
        unique_ids = np.unique(instance_pred[instance_pred >= 0])
        assert len(unique_ids) >= 2

    def test_cluster_instances_non_tree_ignored(self):
        """Test that non-tree points are not clustered."""
        points = np.random.randn(100, 3).astype(np.float32)
        semantic_pred = np.zeros(100, dtype=np.int64)  # All non-tree

        config = PostprocessConfig(clustering_method="meanshift")

        instance_pred = cluster_instances(
            points=points,
            semantic_pred=semantic_pred,
            config=config,
        )

        # All should be -1 (no instance)
        assert np.all(instance_pred == -1)


class TestExtractInstances:
    """Tests for instance extraction."""

    def test_extract_instances_basic(self):
        """Test basic instance extraction."""
        points = np.random.randn(200, 3).astype(np.float32)
        points[:, 2] = np.abs(points[:, 2]) * 10  # Positive Z for height

        instance_pred = np.array([0] * 100 + [1] * 100, dtype=np.int32)
        semantic_pred = np.ones(200, dtype=np.int64)

        config = PostprocessConfig(
            hdbscan_min_cluster_size=10,
            filter_by_height=False,
        )

        instances = extract_instances(
            points=points,
            instance_pred=instance_pred,
            semantic_pred=semantic_pred,
            config=config,
        )

        assert len(instances) == 2

    def test_extract_instances_height_filter(self):
        """Test height filtering during extraction."""
        # Create short and tall trees
        short_points = np.random.randn(50, 3).astype(np.float32)
        short_points[:, 2] = np.random.uniform(0, 1.5, 50)  # Height < 2m

        tall_points = np.random.randn(50, 3).astype(np.float32)
        tall_points[:, 2] = np.random.uniform(0, 15, 50)  # Height > 2m

        points = np.vstack([short_points, tall_points])
        instance_pred = np.array([0] * 50 + [1] * 50, dtype=np.int32)
        semantic_pred = np.ones(100, dtype=np.int64)

        config = PostprocessConfig(
            hdbscan_min_cluster_size=10,
            filter_by_height=True,
            min_height=2.0,
        )

        instances = extract_instances(
            points=points,
            instance_pred=instance_pred,
            semantic_pred=semantic_pred,
            config=config,
        )

        # Only tall tree should pass
        assert len(instances) == 1


class TestOutputFormats:
    """Tests for output format generation."""

    def test_save_geojson(self):
        """Test GeoJSON output."""
        points = np.array([
            [0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0]
        ], dtype=np.float32)
        points[:, 2] = np.array([0, 0, 5, 5])

        instance = TreeInstance(instance_id=1, points=points)
        instance.polygon = extract_polygon(points)
        instance.crown_area = instance.polygon.area if instance.polygon else 0

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "trees.geojson"
            save_geojson([instance], output_path)

            assert output_path.exists()

            with open(output_path) as f:
                geojson = json.load(f)

            assert geojson["type"] == "FeatureCollection"
            assert len(geojson["features"]) == 1
            assert geojson["features"][0]["properties"]["tree_id"] == 1

    def test_save_parquet(self):
        """Test Parquet output."""
        pytest.importorskip("pyarrow")
        pytest.importorskip("pandas")

        points = np.random.randn(100, 3).astype(np.float32)
        instance = TreeInstance(instance_id=1, points=points)
        instance.polygon = extract_polygon(points)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "trees.parquet"
            save_parquet([instance], output_path)

            assert output_path.exists()


class TestPostprocessPipeline:
    """Tests for complete postprocessing pipeline."""

    def test_postprocess_predictions(self):
        """Test complete postprocessing pipeline."""
        pytest.importorskip("hdbscan")

        # Create synthetic predictions
        n_points = 500
        points = np.random.randn(n_points, 3).astype(np.float32) * 5
        points[:, 2] = np.abs(points[:, 2]) * 10

        # Create two clusters
        points[:250, :2] += 20

        semantic_pred = np.ones(n_points, dtype=np.int64)

        config = Config()
        config.postprocess.hdbscan_min_cluster_size = 20
        config.postprocess.filter_by_height = False

        result = postprocess_predictions(
            points=points,
            semantic_pred=semantic_pred,
            config=config,
        )

        assert isinstance(result, PanopticResult)
        assert len(result.points) == n_points
        assert len(result.semantic_pred) == n_points
        assert len(result.instance_pred) == n_points

    def test_save_results(self):
        """Test saving complete results."""
        pytest.importorskip("hdbscan")

        points = np.random.randn(100, 3).astype(np.float32) * 5
        points[:, 2] = np.abs(points[:, 2]) * 10

        semantic_pred = np.ones(100, dtype=np.int64)
        instance_pred = np.zeros(100, dtype=np.int32)

        instance = TreeInstance(instance_id=0, points=points)
        instance.polygon = extract_polygon(points)

        result = PanopticResult(
            points=points,
            semantic_pred=semantic_pred,
            instance_pred=instance_pred,
            instances=[instance],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_paths = save_results(
                result=result,
                output_dir=tmpdir,
                base_name="test",
            )

            assert "polygons" in output_paths
            assert output_paths["polygons"].exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
