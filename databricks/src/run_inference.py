"""Databricks entry point for inference."""

from __future__ import annotations

import sys

from databricks.src.common import (
    load_databricks_config,
    parse_common_args,
    setup_environment,
)
from lidar_panoptic_segmentation.infer import InferencePipeline


def main() -> int:
    args = parse_common_args()
    config = load_databricks_config(args)
    setup_environment(config)

    pipeline = InferencePipeline(config)
    pipeline.process_directory(
        input_dir=config.paths.inference.input,
        output_dir=config.paths.inference.output,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
