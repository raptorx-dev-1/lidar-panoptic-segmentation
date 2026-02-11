"""
Configuration Loader for LiDAR Panoptic Segmentation

Provides utilities to load, validate, and merge configurations from
YAML files, environment variables, and CLI arguments.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

from lidar_panoptic_segmentation.config_schema import (
    Config,
    EnvironmentName,
)

logger = logging.getLogger(__name__)

# Default config locations to search
DEFAULT_CONFIG_PATHS = [
    Path("config.yaml"),
    Path("lidar_panoptic_segmentation/config.yaml"),
    Path.home() / ".config" / "lidar_panoptic_segmentation" / "config.yaml",
]


def _resolve_env_vars(value: Any) -> Any:
    """
    Recursively resolve environment variables in config values.

    Supports ${ENV_VAR} and ${ENV_VAR:-default} syntax.
    Does NOT resolve ${paths.xxx} style config references (handled by Pydantic).
    """
    if isinstance(value, str):
        # Match ${VAR} or ${VAR:-default}, but NOT ${paths.xxx} config references
        pattern = r"\$\{([^}:.]+)(?::-([^}]*))?\}"

        def replacer(match):
            var_name = match.group(1)
            # Skip config path references like ${paths.data_root}
            if var_name.startswith("paths"):
                return match.group(0)  # Return original unchanged
            default = match.group(2) or ""
            return os.environ.get(var_name, default)

        return re.sub(pattern, replacer, value)
    elif isinstance(value, dict):
        return {k: _resolve_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_resolve_env_vars(item) for item in value]
    return value


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dictionaries, with override taking precedence.

    Args:
        base: Base dictionary
        override: Override dictionary (takes precedence)

    Returns:
        Merged dictionary
    """
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value

    return result


def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load a YAML configuration file.

    Args:
        path: Path to YAML file

    Returns:
        Dictionary of configuration values

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML is invalid
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}

    # Resolve environment variables
    data = _resolve_env_vars(data)

    logger.debug(f"Loaded configuration from {path}")
    return data


def find_config_file(
    explicit_path: Optional[Union[str, Path]] = None,
    search_paths: Optional[List[Path]] = None,
) -> Optional[Path]:
    """
    Find a configuration file by searching default locations.

    Args:
        explicit_path: Explicitly specified config path
        search_paths: Additional paths to search

    Returns:
        Path to config file, or None if not found
    """
    if explicit_path:
        path = Path(explicit_path)
        if path.exists():
            return path
        raise FileNotFoundError(f"Specified config file not found: {path}")

    paths_to_check = list(search_paths or []) + DEFAULT_CONFIG_PATHS

    for path in paths_to_check:
        if path.exists():
            logger.info(f"Found configuration at {path}")
            return path

    return None


def load_config(
    config_path: Optional[Union[str, Path]] = None,
    overrides: Optional[Dict[str, Any]] = None,
    environment: Optional[str] = None,
) -> Config:
    """
    Load and validate configuration from a YAML file.

    Args:
        config_path: Path to config file (searches default locations if None)
        overrides: Dictionary of values to override
        environment: Force a specific environment (local, dev, databricks)

    Returns:
        Validated Config object

    Raises:
        FileNotFoundError: If no config file found
        pydantic.ValidationError: If config is invalid
    """
    # Find config file
    path = find_config_file(config_path)

    if path:
        config_data = load_yaml(path)
    else:
        logger.warning("No config file found, using defaults")
        config_data = {}

    # Apply overrides
    if overrides:
        config_data = _deep_merge(config_data, overrides)

    # Override environment if specified
    if environment:
        if "env" not in config_data:
            config_data["env"] = {}
        config_data["env"]["name"] = environment

    # Create and validate config
    config = Config(**config_data)

    logger.info(
        f"Configuration loaded: env={config.env.name}, "
        f"debug={config.env.debug}"
    )

    return config


def validate_config(config: Config) -> List[str]:
    """
    Perform additional validation beyond schema validation.

    Args:
        config: Configuration to validate

    Returns:
        List of warning messages (empty if all good)
    """
    warnings = []

    # Check for Unity Catalog paths in Databricks environment
    if config.env.name == EnvironmentName.DATABRICKS:
        if config.paths.data_root.startswith("dbfs:/"):
            warnings.append(
                "Using dbfs:/ paths is deprecated. "
                "Consider migrating to Unity Catalog abfss:// paths."
            )

    # Check for required training paths if training config is defined
    if config.training and config.paths.train is None:
        warnings.append(
            "Training configuration defined but no training paths specified."
        )

    # Check for required inference paths
    if config.paths.inference is None and not config.paths.train:
        warnings.append(
            "Neither training nor inference paths specified."
        )

    # Check MinkowskiEngine configuration
    if config.dependencies.minkowski_engine.enabled:
        me_config = config.dependencies.minkowski_engine
        if me_config.install_method.value == "notebook":
            script_path = Path(me_config.script_path)
            if not script_path.exists() and config.env.name != EnvironmentName.DATABRICKS:
                warnings.append(
                    f"MinkowskiEngine script not found: {script_path}. "
                    "Run install script before training."
                )

    # Log warnings
    for warning in warnings:
        logger.warning(warning)

    return warnings


def create_default_config() -> Dict[str, Any]:
    """
    Create a default configuration dictionary.

    Returns:
        Default configuration as dictionary
    """
    return Config().model_dump(mode="json")


def save_config(config: Config, path: Union[str, Path]) -> None:
    """
    Save configuration to a YAML file.

    Args:
        config: Configuration to save
        path: Output file path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        yaml.dump(
            config.model_dump(mode="json"),
            f,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )

    logger.info(f"Configuration saved to {path}")


def config_from_args(args: argparse.Namespace) -> Config:
    """
    Create configuration from parsed command-line arguments.

    Args:
        args: Parsed argparse namespace

    Returns:
        Config object
    """
    overrides = {}

    # Map common CLI args to config structure
    if hasattr(args, "batch_size") and args.batch_size:
        if "training" not in overrides:
            overrides["training"] = {}
        overrides["training"]["batch_size"] = args.batch_size

    if hasattr(args, "epochs") and args.epochs:
        if "training" not in overrides:
            overrides["training"] = {}
        overrides["training"]["epochs"] = args.epochs

    if hasattr(args, "lr") and args.lr:
        if "training" not in overrides:
            overrides["training"] = {}
        overrides["training"]["learning_rate"] = args.lr

    if hasattr(args, "debug") and args.debug:
        if "env" not in overrides:
            overrides["env"] = {}
        overrides["env"]["debug"] = True

    config_path = getattr(args, "config", None)
    environment = getattr(args, "environment", None)

    return load_config(
        config_path=config_path,
        overrides=overrides,
        environment=environment,
    )


def add_config_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Add standard configuration arguments to an argument parser.

    Args:
        parser: ArgumentParser to add arguments to

    Returns:
        Modified parser
    """
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--environment", "-e",
        type=str,
        choices=["local", "dev", "databricks"],
        help="Override environment setting"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Override batch size"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Override number of epochs"
    )
    parser.add_argument(
        "--lr",
        type=float,
        help="Override learning rate"
    )

    return parser


# Convenience exports
__all__ = [
    "Config",
    "load_config",
    "validate_config",
    "save_config",
    "config_from_args",
    "add_config_args",
    "create_default_config",
]
