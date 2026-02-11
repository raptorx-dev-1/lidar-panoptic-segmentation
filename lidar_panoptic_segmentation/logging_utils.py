"""
Logging and Experiment Tracking Module

Provides unified logging interface for MLflow, Weights & Biases,
and standard Python logging.
"""

from __future__ import annotations

import logging
import os
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from lidar_panoptic_segmentation.config_schema import Config, LoggingConfig

# Configure root logger
logger = logging.getLogger("lidar_panoptic_segmentation")


def setup_logging(
    config: Config,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """
    Setup logging based on configuration.

    Args:
        config: Configuration object
        log_file: Optional path to log file

    Returns:
        Configured logger
    """
    log_level = getattr(logging, config.logging.level.upper(), logging.INFO)

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Configure root logger
    logger.setLevel(log_level)
    logger.handlers = []  # Clear existing handlers

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if config.logging.log_to_file or log_file:
        if log_file is None:
            log_dir = Path(config.paths.logs)
            log_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = log_dir / f"lidar_panoptic_{timestamp}.log"

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info(f"Logging to file: {log_file}")

    return logger


class BaseExperimentLogger:
    """Base class for experiment loggers."""

    def __init__(self, config: Config):
        self.config = config
        self.step = 0

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log hyperparameters."""
        pass

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
    ) -> None:
        """Log metrics."""
        pass

    def log_artifact(self, path: str, artifact_type: str = "file") -> None:
        """Log an artifact file."""
        pass

    def log_model(self, model: Any, name: str = "model") -> None:
        """Log a model."""
        pass

    def set_tags(self, tags: Dict[str, str]) -> None:
        """Set experiment tags."""
        pass

    def start_run(self, run_name: Optional[str] = None) -> None:
        """Start a new run."""
        pass

    def end_run(self) -> None:
        """End the current run."""
        pass

    def __enter__(self):
        self.start_run()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_run()


class MLflowLogger(BaseExperimentLogger):
    """MLflow experiment logger."""

    def __init__(self, config: Config):
        super().__init__(config)
        self.mlflow = None
        self.run = None
        self._setup()

    def _setup(self) -> None:
        """Initialize MLflow."""
        if not self.config.logging.mlflow.enabled:
            return

        try:
            import mlflow

            self.mlflow = mlflow

            # Set tracking URI
            tracking_uri = self.config.logging.mlflow.tracking_uri
            if tracking_uri == "databricks":
                # Databricks-managed MLflow
                logger.info("Using Databricks-managed MLflow")
            else:
                mlflow.set_tracking_uri(tracking_uri)
                logger.info(f"MLflow tracking URI: {tracking_uri}")

            # Set experiment
            experiment_name = self.config.logging.mlflow.experiment_name
            mlflow.set_experiment(experiment_name)
            logger.info(f"MLflow experiment: {experiment_name}")

        except ImportError:
            logger.warning("MLflow not available. Install with: pip install mlflow")
            self.config.logging.mlflow.enabled = False

    def start_run(self, run_name: Optional[str] = None) -> None:
        """Start a new MLflow run."""
        if not self.config.logging.mlflow.enabled or self.mlflow is None:
            return

        name = run_name or self.config.logging.mlflow.run_name
        if name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            name = f"run_{timestamp}"

        self.run = self.mlflow.start_run(run_name=name)
        logger.info(f"Started MLflow run: {name}")

    def end_run(self) -> None:
        """End the current MLflow run."""
        if self.run is not None and self.mlflow is not None:
            self.mlflow.end_run()
            self.run = None

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log hyperparameters to MLflow."""
        if not self.config.logging.mlflow.enabled or self.mlflow is None:
            return

        # Flatten nested dicts
        flat_params = self._flatten_dict(params)

        # MLflow has a 500-char limit for param values
        for key, value in flat_params.items():
            try:
                str_value = str(value)
                if len(str_value) > 500:
                    str_value = str_value[:497] + "..."
                self.mlflow.log_param(key, str_value)
            except Exception as e:
                logger.warning(f"Failed to log param {key}: {e}")

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
    ) -> None:
        """Log metrics to MLflow."""
        if not self.config.logging.mlflow.enabled or self.mlflow is None:
            return

        if step is None:
            step = self.step
            self.step += 1

        for key, value in metrics.items():
            try:
                self.mlflow.log_metric(key, float(value), step=step)
            except Exception as e:
                logger.warning(f"Failed to log metric {key}: {e}")

    def log_artifact(self, path: str, artifact_type: str = "file") -> None:
        """Log an artifact to MLflow."""
        if not self.config.logging.mlflow.enabled or self.mlflow is None:
            return

        if not self.config.logging.mlflow.log_artifacts:
            return

        try:
            if os.path.isdir(path):
                self.mlflow.log_artifacts(path)
            else:
                self.mlflow.log_artifact(path)
            logger.debug(f"Logged artifact: {path}")
        except Exception as e:
            logger.warning(f"Failed to log artifact {path}: {e}")

    def log_model(
        self,
        model: Any,
        name: str = "model",
        registered_model_name: Optional[str] = None,
    ) -> None:
        """Log a PyTorch model to MLflow."""
        if not self.config.logging.mlflow.enabled or self.mlflow is None:
            return

        if not self.config.logging.mlflow.log_models:
            return

        try:
            import mlflow.pytorch

            if registered_model_name is None:
                registered_model_name = "LidarPanopticSegmentation"

            mlflow.pytorch.log_model(
                model,
                name,
                registered_model_name=registered_model_name,
            )
            logger.info(f"Logged model to MLflow: {registered_model_name}")
        except Exception as e:
            logger.warning(f"Failed to log model: {e}")

    def set_tags(self, tags: Dict[str, str]) -> None:
        """Set MLflow run tags."""
        if not self.config.logging.mlflow.enabled or self.mlflow is None:
            return

        for key, value in tags.items():
            try:
                self.mlflow.set_tag(key, value)
            except Exception as e:
                logger.warning(f"Failed to set tag {key}: {e}")

    @staticmethod
    def _flatten_dict(
        d: Dict[str, Any],
        parent_key: str = "",
        sep: str = ".",
    ) -> Dict[str, Any]:
        """Flatten nested dictionary."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(
                    MLflowLogger._flatten_dict(v, new_key, sep=sep).items()
                )
            else:
                items.append((new_key, v))
        return dict(items)


class WandbLogger(BaseExperimentLogger):
    """Weights & Biases experiment logger."""

    def __init__(self, config: Config):
        super().__init__(config)
        self.wandb = None
        self.run = None
        self._setup()

    def _setup(self) -> None:
        """Initialize Weights & Biases."""
        if not self.config.logging.wandb.enabled:
            return

        try:
            import wandb

            self.wandb = wandb
            logger.info("Weights & Biases initialized")
        except ImportError:
            logger.warning("wandb not available. Install with: pip install wandb")
            self.config.logging.wandb.enabled = False

    def start_run(self, run_name: Optional[str] = None) -> None:
        """Start a new W&B run."""
        if not self.config.logging.wandb.enabled or self.wandb is None:
            return

        name = run_name or self.config.logging.wandb.name
        if name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            name = f"run_{timestamp}"

        self.run = self.wandb.init(
            project=self.config.logging.wandb.project,
            entity=self.config.logging.wandb.entity,
            name=name,
            tags=self.config.logging.wandb.tags,
            config=self.config.model_dump(),
        )
        logger.info(f"Started W&B run: {name}")

    def end_run(self) -> None:
        """End the current W&B run."""
        if self.run is not None and self.wandb is not None:
            self.wandb.finish()
            self.run = None

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log hyperparameters to W&B."""
        if not self.config.logging.wandb.enabled or self.wandb is None:
            return

        self.wandb.config.update(params)

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
    ) -> None:
        """Log metrics to W&B."""
        if not self.config.logging.wandb.enabled or self.wandb is None:
            return

        if step is None:
            step = self.step
            self.step += 1

        self.wandb.log(metrics, step=step)

    def log_artifact(self, path: str, artifact_type: str = "file") -> None:
        """Log an artifact to W&B."""
        if not self.config.logging.wandb.enabled or self.wandb is None:
            return

        artifact = self.wandb.Artifact(
            name=Path(path).stem,
            type=artifact_type,
        )
        if os.path.isdir(path):
            artifact.add_dir(path)
        else:
            artifact.add_file(path)
        self.wandb.log_artifact(artifact)

    def log_model(self, model: Any, name: str = "model") -> None:
        """Log a model to W&B."""
        if not self.config.logging.wandb.enabled or self.wandb is None:
            return

        # Save model locally first
        model_path = f"/tmp/{name}.pt"
        import torch
        torch.save(model.state_dict(), model_path)

        self.log_artifact(model_path, artifact_type="model")


class CombinedLogger(BaseExperimentLogger):
    """Combined logger that dispatches to multiple backends."""

    def __init__(self, config: Config):
        super().__init__(config)
        self.loggers: List[BaseExperimentLogger] = []

        # Add MLflow logger
        if config.logging.mlflow.enabled:
            self.loggers.append(MLflowLogger(config))

        # Add W&B logger
        if config.logging.wandb.enabled:
            self.loggers.append(WandbLogger(config))

    def start_run(self, run_name: Optional[str] = None) -> None:
        """Start runs on all backends."""
        for logger_instance in self.loggers:
            logger_instance.start_run(run_name)

    def end_run(self) -> None:
        """End runs on all backends."""
        for logger_instance in self.loggers:
            logger_instance.end_run()

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log params to all backends."""
        for logger_instance in self.loggers:
            logger_instance.log_params(params)

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
    ) -> None:
        """Log metrics to all backends."""
        for logger_instance in self.loggers:
            logger_instance.log_metrics(metrics, step)

    def log_artifact(self, path: str, artifact_type: str = "file") -> None:
        """Log artifact to all backends."""
        for logger_instance in self.loggers:
            logger_instance.log_artifact(path, artifact_type)

    def log_model(self, model: Any, name: str = "model") -> None:
        """Log model to all backends."""
        for logger_instance in self.loggers:
            logger_instance.log_model(model, name)

    def set_tags(self, tags: Dict[str, str]) -> None:
        """Set tags on all backends."""
        for logger_instance in self.loggers:
            logger_instance.set_tags(tags)


def create_experiment_logger(config: Config) -> BaseExperimentLogger:
    """
    Create an experiment logger based on configuration.

    Args:
        config: Configuration object

    Returns:
        Configured experiment logger
    """
    return CombinedLogger(config)


@contextmanager
def experiment_context(config: Config, run_name: Optional[str] = None):
    """
    Context manager for experiment tracking.

    Args:
        config: Configuration object
        run_name: Optional run name

    Yields:
        Experiment logger
    """
    exp_logger = create_experiment_logger(config)
    exp_logger.start_run(run_name)
    try:
        yield exp_logger
    finally:
        exp_logger.end_run()


# Convenience exports
__all__ = [
    "setup_logging",
    "MLflowLogger",
    "WandbLogger",
    "CombinedLogger",
    "create_experiment_logger",
    "experiment_context",
    "logger",
]
