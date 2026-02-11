"""
Tests for Inference Pipeline

Tests the inference pipeline including model loading and prediction.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from lidar_panoptic_segmentation.config_schema import (
    Config,
    DataPaths,
    InferencePaths,
)
from lidar_panoptic_segmentation.dataset import PointCloudData
from lidar_panoptic_segmentation.model import (
    LidarPanopticModel,
    PanopticOutput,
    check_minkowski_available,
    create_model,
    load_model,
    save_model,
)


class TestModelCreation:
    """Tests for model creation."""

    def test_create_model_default(self):
        """Test creating model with default configuration."""
        config = Config()
        model = create_model(config, device="cpu")

        assert isinstance(model, LidarPanopticModel)
        assert model.num_classes == 2
        assert model.embed_dim == 5

    def test_create_model_custom_config(self):
        """Test creating model with custom configuration."""
        config = Config()
        config.training.num_classes = 3
        config.training.embed_dim = 8

        model = create_model(config, device="cpu")

        assert model.num_classes == 3
        assert model.embed_dim == 8

    def test_model_forward_fallback(self):
        """Test model forward pass with fallback (no MinkowskiEngine)."""
        config = Config()
        model = LidarPanopticModel(
            num_classes=2,
            embed_dim=5,
            in_channels=4,
        )
        model.eval()

        # Create dummy input
        batch_size = 2
        n_points = 100
        coords = torch.zeros(n_points, 4)
        coords[:, 0] = torch.randint(0, batch_size, (n_points,))
        coords[:, 1:] = torch.randn(n_points, 3)
        features = torch.randn(n_points, 4)

        with torch.no_grad():
            output = model(coords, features)

        assert isinstance(output, PanopticOutput)
        assert output.semantic_logits.shape == (n_points, 2)
        assert output.semantic_pred.shape == (n_points,)


class TestModelSaveLoad:
    """Tests for model saving and loading."""

    def test_save_and_load_model(self):
        """Test saving and loading model checkpoint."""
        config = Config()
        config.training.num_classes = 2
        config.training.embed_dim = 5

        model = create_model(config, device="cpu")

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "model.pt"
            save_model(model, checkpoint_path, config=config)

            assert checkpoint_path.exists()

            loaded_model = load_model(str(checkpoint_path), config, device="cpu")

            assert loaded_model.num_classes == model.num_classes
            assert loaded_model.embed_dim == model.embed_dim

    def test_save_model_with_optimizer(self):
        """Test saving model with optimizer state."""
        config = Config()
        model = create_model(config, device="cpu")
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "model.pt"
            save_model(
                model,
                checkpoint_path,
                config=config,
                optimizer=optimizer,
                epoch=10,
            )

            checkpoint = torch.load(checkpoint_path, weights_only=False)
            assert "optimizer_state_dict" in checkpoint
            assert checkpoint["epoch"] == 10


class TestPanopticOutput:
    """Tests for PanopticOutput dataclass."""

    def test_panoptic_output_to_numpy(self):
        """Test converting PanopticOutput to numpy."""
        n_points = 100
        output = PanopticOutput(
            semantic_logits=torch.randn(n_points, 2),
            semantic_pred=torch.randint(0, 2, (n_points,)),
            offset_pred=torch.randn(n_points, 3),
            embedding=torch.randn(n_points, 5),
        )

        numpy_out = output.to_numpy()

        assert isinstance(numpy_out["semantic_logits"], np.ndarray)
        assert numpy_out["semantic_logits"].shape == (n_points, 2)
        assert numpy_out["offset_pred"].shape == (n_points, 3)
        assert numpy_out["embedding"].shape == (n_points, 5)


class TestModelLoss:
    """Tests for model loss computation."""

    def test_compute_loss_semantic_only(self):
        """Test loss computation with semantic labels only."""
        model = LidarPanopticModel(num_classes=2, embed_dim=5)
        model.eval()

        n_points = 100
        coords = torch.zeros(n_points, 4)
        coords[:, 1:] = torch.randn(n_points, 3)
        features = torch.randn(n_points, 4)
        semantic_labels = torch.randint(0, 2, (n_points,))

        with torch.no_grad():
            output = model(coords, features)
            loss = model.compute_loss(output, semantic_labels)

        assert loss.total.item() >= 0
        assert loss.semantic.item() >= 0

    def test_compute_loss_with_instances(self):
        """Test loss computation with instance labels."""
        model = LidarPanopticModel(num_classes=2, embed_dim=5)
        model.eval()

        n_points = 100
        coords = torch.zeros(n_points, 4)
        coords[:, 1:] = torch.randn(n_points, 3)
        features = torch.randn(n_points, 4)
        semantic_labels = torch.randint(0, 2, (n_points,))
        instance_labels = torch.randint(0, 5, (n_points,))

        with torch.no_grad():
            output = model(coords, features)
            loss = model.compute_loss(output, semantic_labels, instance_labels)

        assert loss.total.item() >= 0
        assert loss.semantic.item() >= 0

    def test_loss_to_dict(self):
        """Test converting loss to dictionary."""
        from lidar_panoptic_segmentation.model import PanopticLoss

        loss = PanopticLoss(
            total=torch.tensor(1.5),
            semantic=torch.tensor(0.8),
            offset=torch.tensor(0.5),
            embedding=torch.tensor(0.2),
        )

        loss_dict = loss.to_dict()

        assert "loss/total" in loss_dict
        assert loss_dict["loss/total"] == pytest.approx(1.5, rel=1e-5)
        assert loss_dict["loss/semantic"] == pytest.approx(0.8, rel=1e-5)
        assert loss_dict["loss/offset"] == pytest.approx(0.5, rel=1e-5)
        assert loss_dict["loss/embedding"] == pytest.approx(0.2, rel=1e-5)


class TestMinkowskiEngineCheck:
    """Tests for MinkowskiEngine availability check."""

    def test_check_minkowski_available(self):
        """Test MinkowskiEngine availability check."""
        result = check_minkowski_available()
        assert isinstance(result, bool)


class TestInferencePipeline:
    """Tests for inference pipeline."""

    def test_inference_pipeline_initialization(self):
        """Test inference pipeline initialization."""
        from lidar_panoptic_segmentation.infer import InferencePipeline

        config = Config()
        config.paths.inference = InferencePaths(
            input="./test_input",
            output="./test_output",
        )

        # Mock model loading
        with patch(
            "lidar_panoptic_segmentation.infer.load_model"
        ) as mock_load:
            mock_model = MagicMock(spec=LidarPanopticModel)
            mock_model.eval = MagicMock()
            mock_load.return_value = mock_model

            pipeline = InferencePipeline(config)

            assert pipeline.model is not None

    def test_needs_tiling_small_cloud(self):
        """Test tiling check for small point cloud."""
        from lidar_panoptic_segmentation.infer import InferencePipeline

        config = Config()
        config.inference.tile_size = 50.0
        config.paths.inference = InferencePaths(
            input="./test_input",
            output="./test_output",
        )

        with patch(
            "lidar_panoptic_segmentation.infer.load_model"
        ) as mock_load:
            mock_model = MagicMock(spec=LidarPanopticModel)
            mock_model.eval = MagicMock()
            mock_load.return_value = mock_model

            pipeline = InferencePipeline(config)

            # Small point cloud (20x20m)
            data = PointCloudData(
                pos=np.random.rand(100, 3) * 20,
                bounds=np.array([[0, 0, 0], [20, 20, 10]]),
            )

            needs_tiling = pipeline._needs_tiling(data)
            assert needs_tiling == False

    def test_needs_tiling_large_cloud(self):
        """Test tiling check for large point cloud."""
        from lidar_panoptic_segmentation.infer import InferencePipeline

        config = Config()
        config.inference.tile_size = 50.0
        config.paths.inference = InferencePaths(
            input="./test_input",
            output="./test_output",
        )

        with patch(
            "lidar_panoptic_segmentation.infer.load_model"
        ) as mock_load:
            mock_model = MagicMock(spec=LidarPanopticModel)
            mock_model.eval = MagicMock()
            mock_load.return_value = mock_model

            pipeline = InferencePipeline(config)

            # Large point cloud (100x100m)
            data = PointCloudData(
                pos=np.random.rand(1000, 3) * 100,
                bounds=np.array([[0, 0, 0], [100, 100, 50]]),
            )

            needs_tiling = pipeline._needs_tiling(data)
            assert needs_tiling == True


class TestModelGradients:
    """Tests for model gradient computation."""

    def test_model_gradients(self):
        """Test that model computes gradients correctly."""
        model = LidarPanopticModel(num_classes=2, embed_dim=5)
        model.train()

        n_points = 50
        coords = torch.zeros(n_points, 4)
        coords[:, 1:] = torch.randn(n_points, 3)
        features = torch.randn(n_points, 4, requires_grad=True)
        semantic_labels = torch.randint(0, 2, (n_points,))

        output = model(coords, features)
        loss = model.compute_loss(output, semantic_labels)
        loss.total.backward()

        # Check gradients exist
        has_grad = False
        for param in model.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break

        assert has_grad, "Model should have non-zero gradients"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
