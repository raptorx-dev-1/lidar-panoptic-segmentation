"""
Tests for Configuration Loading and Validation

Tests the config loading, validation, and schema enforcement.
"""

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from lidar_panoptic_segmentation.config import (
    Config,
    create_default_config,
    load_config,
    save_config,
    validate_config,
)
from lidar_panoptic_segmentation.config_schema import (
    AugmentationConfig,
    DataPaths,
    EnvConfig,
    EnvironmentName,
    InferenceConfig,
    LoggingConfig,
    ModelArchitecture,
    PathsConfig,
    PolygonOutputFormat,
    PostprocessConfig,
    SchedulerType,
    TrainingConfig,
)


class TestConfigSchema:
    """Tests for configuration schema validation."""

    def test_default_config_creation(self):
        """Test that default config can be created."""
        config = Config()
        assert config.env.name == EnvironmentName.LOCAL
        assert config.env.debug is False
        assert config.training.epochs == 60

    def test_env_config_validation(self):
        """Test environment configuration validation."""
        env = EnvConfig(name="databricks", debug=True, seed=123)
        assert env.name == EnvironmentName.DATABRICKS
        assert env.debug is True
        assert env.seed == 123

    def test_invalid_env_name_rejected(self):
        """Test that invalid environment names are rejected."""
        with pytest.raises(ValueError):
            EnvConfig(name="invalid_env")

    def test_training_config_defaults(self):
        """Test training configuration defaults."""
        training = TrainingConfig()
        assert training.model_name == ModelArchitecture.POINT_GROUP_3HEADS
        assert training.batch_size == 8
        assert training.epochs == 60
        assert training.learning_rate == 0.0001
        assert training.scheduler == SchedulerType.COSINE

    def test_training_config_validation(self):
        """Test training configuration validation."""
        # Valid config
        training = TrainingConfig(
            batch_size=16,
            epochs=100,
            learning_rate=0.001,
        )
        assert training.batch_size == 16
        assert training.epochs == 100

        # Invalid batch size
        with pytest.raises(ValueError):
            TrainingConfig(batch_size=0)

        # Invalid epochs
        with pytest.raises(ValueError):
            TrainingConfig(epochs=0)

    def test_augmentation_config(self):
        """Test augmentation configuration."""
        aug = AugmentationConfig(
            rotate=True,
            rotate_range=90.0,
            jitter=False,
            scale=True,
            scale_range=(0.8, 1.2),
        )
        assert aug.rotate is True
        assert aug.rotate_range == 90.0
        assert aug.jitter is False
        assert aug.scale is True

    def test_inference_config(self):
        """Test inference configuration."""
        inference = InferenceConfig(
            min_points_per_instance=100,
            confidence_threshold=0.7,
            polygon_output_format=PolygonOutputFormat.GEOJSON,
        )
        assert inference.min_points_per_instance == 100
        assert inference.confidence_threshold == 0.7
        assert inference.polygon_output_format == PolygonOutputFormat.GEOJSON

    def test_paths_config(self):
        """Test paths configuration."""
        paths = PathsConfig(
            data_root="./data",
            train=DataPaths(pointclouds="./data/train/las"),
            models="./models",
        )
        assert paths.data_root == "./data"
        assert paths.train.pointclouds == "./data/train/las"

    def test_abfss_paths_accepted(self):
        """Test that Unity Catalog abfss:// paths are accepted."""
        paths = DataPaths(
            pointclouds="abfss://container@account.dfs.core.windows.net/data"
        )
        assert paths.pointclouds.startswith("abfss://")


class TestConfigLoading:
    """Tests for configuration loading and saving."""

    def test_load_from_yaml(self):
        """Test loading configuration from YAML file."""
        config_dict = {
            "env": {"name": "local", "debug": True},
            "training": {"epochs": 30, "batch_size": 4},
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(config_dict, f)
            temp_path = f.name

        try:
            config = load_config(temp_path)
            assert config.env.debug is True
            assert config.training.epochs == 30
            assert config.training.batch_size == 4
        finally:
            os.unlink(temp_path)

    def test_save_and_reload(self):
        """Test saving and reloading configuration."""
        original = Config(
            env=EnvConfig(name="databricks", debug=True),
            training=TrainingConfig(epochs=50, batch_size=16),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.yaml"
            save_config(original, path)

            loaded = load_config(path)
            assert loaded.env.name == original.env.name
            assert loaded.env.debug == original.env.debug
            assert loaded.training.epochs == original.training.epochs
            assert loaded.training.batch_size == original.training.batch_size

    def test_load_with_overrides(self):
        """Test loading configuration with overrides."""
        config_dict = {"training": {"epochs": 30}}

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(config_dict, f)
            temp_path = f.name

        try:
            overrides = {"training": {"batch_size": 32}}
            config = load_config(temp_path, overrides=overrides)
            assert config.training.epochs == 30
            assert config.training.batch_size == 32
        finally:
            os.unlink(temp_path)

    def test_load_with_environment_override(self):
        """Test loading configuration with environment override."""
        config_dict = {"env": {"name": "local"}}

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(config_dict, f)
            temp_path = f.name

        try:
            config = load_config(temp_path, environment="databricks")
            assert config.env.name == EnvironmentName.DATABRICKS
        finally:
            os.unlink(temp_path)

    def test_create_default_config(self):
        """Test creating default configuration dictionary."""
        config_dict = create_default_config()
        assert isinstance(config_dict, dict)
        assert "env" in config_dict
        assert "training" in config_dict
        assert "paths" in config_dict


class TestConfigValidation:
    """Tests for configuration validation."""

    def test_validate_missing_training_paths(self):
        """Test validation warns about missing training paths."""
        config = Config(
            training=TrainingConfig(epochs=10),
            paths=PathsConfig(data_root="./data"),
        )
        warnings = validate_config(config)
        assert len(warnings) > 0
        assert any("training paths" in w.lower() for w in warnings)

    def test_validate_databricks_dbfs_warning(self):
        """Test validation warns about dbfs:/ paths in Databricks."""
        config = Config(
            env=EnvConfig(name="databricks"),
            paths=PathsConfig(data_root="dbfs:/mnt/data"),
        )
        warnings = validate_config(config)
        assert any("dbfs:/" in w for w in warnings)

    def test_config_to_dict(self):
        """Test converting configuration to dictionary."""
        config = Config()
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert "env" in config_dict
        assert config_dict["env"]["name"] == "local"


class TestConfigPaths:
    """Tests for path resolution."""

    def test_path_variable_substitution(self):
        """Test path variable substitution."""
        config_dict = {
            "paths": {
                "data_root": "/data",
                "train": {
                    "pointclouds": "${paths.data_root}/train/las",
                },
            },
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(config_dict, f)
            temp_path = f.name

        try:
            config = load_config(temp_path)
            assert config.paths.train.pointclouds == "/data/train/las"
        finally:
            os.unlink(temp_path)

    def test_resolve_path_method(self):
        """Test PathsConfig.resolve_path method."""
        paths = PathsConfig(data_root="/my/data")
        resolved = paths.resolve_path("${paths.data_root}/subdir")
        assert resolved == "/my/data/subdir"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
