"""Databricks entry point for model training."""

from __future__ import annotations

import sys

from databricks.src.common import (
    load_databricks_config,
    parse_common_args,
    setup_environment,
)
from lidar_panoptic_segmentation.train import Trainer


def main() -> int:
    args = parse_common_args()
    config = load_databricks_config(args)
    setup_environment(config)

    trainer = Trainer(config)
    trainer.setup()
    trainer.train()
    return 0


if __name__ == "__main__":
    sys.exit(main())
