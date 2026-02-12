"""Shared utilities for Databricks entry points."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

import yaml

from lidar_panoptic_segmentation.config_schema import Config
from lidar_panoptic_segmentation.logging_utils import setup_logging
from lidar_panoptic_segmentation.utils import set_seed


def parse_common_args() -> argparse.Namespace:
    """Parse common command-line arguments for Databricks jobs."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", required=True, help="Unity Catalog name")
    parser.add_argument("--schema", required=True, help="UC schema name")
    parser.add_argument(
        "--model-name",
        default="LidarPanopticSegmentation",
        help="Registered model name",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to config YAML (defaults to databricks/config.yaml)",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    return parser.parse_args()


def _resolve_placeholders(
    text: str,
    catalog: str,
    schema: str,
    model_name: str,
) -> str:
    """Replace placeholder tokens in raw YAML text."""
    text = text.replace("__CATALOG__", catalog)
    text = text.replace("__SCHEMA__", schema)
    text = text.replace("__MODEL_NAME__", model_name)
    return text


def load_databricks_config(args: argparse.Namespace) -> Config:
    """Load config.yaml, resolve UC placeholders, and return a validated Config."""
    config_path = args.config
    if config_path is None:
        config_path = str(Path(__file__).resolve().parent.parent / "config.yaml")

    with open(config_path) as f:
        raw = f.read()

    resolved = _resolve_placeholders(raw, args.catalog, args.schema, args.model_name)
    data = yaml.safe_load(resolved)

    if args.debug:
        data.setdefault("env", {})["debug"] = True

    return Config(**data)


def setup_environment(config: Config) -> None:
    """Configure logging and set random seeds."""
    setup_logging(config)
    set_seed(config.env.seed)
