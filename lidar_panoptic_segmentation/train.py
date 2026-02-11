#!/usr/bin/env python3
"""
LiDAR Panoptic Segmentation Training CLI

Provides command-line interface for training panoptic segmentation models.
Supports distributed training, mixed precision, and experiment tracking.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    ExponentialLR,
    MultiStepLR,
    ReduceLROnPlateau,
    StepLR,
)

from lidar_panoptic_segmentation.config import (
    Config,
    add_config_args,
    config_from_args,
    load_config,
    save_config,
    validate_config,
)
from lidar_panoptic_segmentation.dataset import create_dataloader
from lidar_panoptic_segmentation.logging_utils import (
    create_experiment_logger,
    setup_logging,
)
from lidar_panoptic_segmentation.model import (
    LidarPanopticModel,
    create_model,
    save_model,
)

logger = logging.getLogger(__name__)


class Trainer:
    """Training orchestrator for LiDAR panoptic segmentation."""

    def __init__(self, config: Config):
        """
        Initialize trainer.

        Args:
            config: Configuration object
        """
        self.config = config
        self.device = config.get_effective_device()

        # Setup logging
        setup_logging(config)

        # Validate configuration
        warnings = validate_config(config)
        for warning in warnings:
            logger.warning(warning)

        # Initialize components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.train_loader = None
        self.val_loader = None
        self.exp_logger = None

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = float("inf")

    def setup(self) -> None:
        """Setup training components."""
        logger.info("Setting up training...")
        logger.info(f"Device: {self.device}")

        # Create model
        self.model = create_model(self.config, self.device)

        # Create optimizer
        self.optimizer = self._create_optimizer()

        # Create scheduler
        self.scheduler = self._create_scheduler()

        # Create mixed precision scaler
        if self.config.training.mixed_precision and self.device == "cuda":
            self.scaler = GradScaler()
            logger.info("Using mixed precision training")

        # Create data loaders
        self.train_loader = create_dataloader(self.config, split="train")
        logger.info(f"Train loader: {len(self.train_loader)} batches")

        if self.config.paths.val:
            self.val_loader = create_dataloader(self.config, split="val")
            logger.info(f"Val loader: {len(self.val_loader)} batches")

        # Create experiment logger
        self.exp_logger = create_experiment_logger(self.config)

        # Resume from checkpoint if specified
        if self.config.training.resume_from:
            self._resume_checkpoint(self.config.training.resume_from)

        logger.info("Training setup complete")

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer based on configuration."""
        params = self.model.parameters()
        lr = self.config.training.learning_rate
        weight_decay = self.config.training.weight_decay

        optimizer = torch.optim.AdamW(
            params,
            lr=lr,
            weight_decay=weight_decay,
        )

        logger.info(f"Created AdamW optimizer: lr={lr}, weight_decay={weight_decay}")
        return optimizer

    def _create_scheduler(self) -> Optional[Any]:
        """Create learning rate scheduler based on configuration."""
        scheduler_type = self.config.training.scheduler.value
        epochs = self.config.training.epochs

        if scheduler_type == "cosine":
            scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=epochs,
                eta_min=1e-7,
            )
        elif scheduler_type == "exponential":
            scheduler = ExponentialLR(
                self.optimizer,
                gamma=0.95,
            )
        elif scheduler_type == "step":
            scheduler = StepLR(
                self.optimizer,
                step_size=10,
                gamma=0.5,
            )
        elif scheduler_type == "multi_step":
            scheduler = MultiStepLR(
                self.optimizer,
                milestones=[20, 40, 50],
                gamma=0.1,
            )
        elif scheduler_type == "reduce_on_plateau":
            scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                patience=5,
                factor=0.5,
            )
        else:
            logger.warning(f"Unknown scheduler type: {scheduler_type}")
            return None

        logger.info(f"Created {scheduler_type} scheduler")
        return scheduler

    def _resume_checkpoint(self, checkpoint_path: str) -> None:
        """Resume training from checkpoint."""
        logger.info(f"Resuming from checkpoint: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])

        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if "epoch" in checkpoint:
            self.current_epoch = checkpoint["epoch"]

        if "global_step" in checkpoint:
            self.global_step = checkpoint["global_step"]

        if "best_metric" in checkpoint:
            self.best_metric = checkpoint["best_metric"]

        logger.info(f"Resumed from epoch {self.current_epoch}")

    def train(self) -> Dict[str, float]:
        """
        Run training loop.

        Returns:
            Dictionary of final metrics
        """
        logger.info("Starting training...")

        # Start experiment tracking
        self.exp_logger.start_run()

        # Log configuration
        self.exp_logger.log_params(self.config.model_dump())

        # Log tags
        self.exp_logger.set_tags({
            "model": self.config.training.model_name.value,
            "backbone": self.config.training.backbone,
            "environment": self.config.env.name.value,
        })

        try:
            epochs = self.config.training.epochs
            start_epoch = self.current_epoch

            for epoch in range(start_epoch, epochs):
                self.current_epoch = epoch

                # Training epoch
                train_metrics = self._train_epoch()

                # Log training metrics
                self.exp_logger.log_metrics(train_metrics, step=epoch)

                # Validation
                if (
                    self.val_loader is not None
                    and (epoch + 1) % self.config.training.eval_frequency == 0
                ):
                    val_metrics = self._validate()
                    self.exp_logger.log_metrics(val_metrics, step=epoch)

                    # Check for improvement
                    val_loss = val_metrics.get("val/loss", float("inf"))
                    if val_loss < self.best_metric:
                        self.best_metric = val_loss
                        self._save_best_model()

                # Step scheduler
                if self.scheduler is not None:
                    if isinstance(self.scheduler, ReduceLROnPlateau):
                        self.scheduler.step(train_metrics["train/loss"])
                    else:
                        self.scheduler.step()

                # Save checkpoint
                if (epoch + 1) % self.config.training.checkpoint_frequency == 0:
                    self._save_checkpoint()

                # Log learning rate
                current_lr = self.optimizer.param_groups[0]["lr"]
                self.exp_logger.log_metrics({"lr": current_lr}, step=epoch)

            # Final save
            self._save_checkpoint()

            # Log final model
            self.exp_logger.log_model(self.model, "final_model")

            logger.info("Training complete!")

            return {"best_val_loss": self.best_metric}

        finally:
            self.exp_logger.end_run()

    def _train_epoch(self) -> Dict[str, float]:
        """Run one training epoch."""
        self.model.train()
        epoch_losses = []
        epoch_start = time.time()

        for batch_idx, batch in enumerate(self.train_loader):
            loss = self._train_step(batch)
            epoch_losses.append(loss)

            self.global_step += 1

            # Log periodically
            if (batch_idx + 1) % self.config.logging.log_frequency == 0:
                avg_loss = np.mean(epoch_losses[-self.config.logging.log_frequency:])
                logger.info(
                    f"Epoch {self.current_epoch} | "
                    f"Batch {batch_idx + 1}/{len(self.train_loader)} | "
                    f"Loss: {avg_loss:.4f}"
                )

        epoch_time = time.time() - epoch_start
        avg_loss = np.mean(epoch_losses)

        logger.info(
            f"Epoch {self.current_epoch} complete | "
            f"Loss: {avg_loss:.4f} | "
            f"Time: {epoch_time:.1f}s"
        )

        return {
            "train/loss": avg_loss,
            "train/epoch_time": epoch_time,
        }

    def _train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Run one training step."""
        # Move data to device
        coords = batch["coords"].to(self.device)
        features = batch["features"].to(self.device)
        semantic_labels = batch.get("semantic_labels")
        instance_labels = batch.get("instance_labels")

        if semantic_labels is not None:
            semantic_labels = semantic_labels.to(self.device)
        if instance_labels is not None:
            instance_labels = instance_labels.to(self.device)

        # Zero gradients
        self.optimizer.zero_grad()

        # Forward pass with optional mixed precision
        if self.scaler is not None:
            with autocast():
                output = self.model(coords, features)
                loss_dict = self.model.compute_loss(
                    output, semantic_labels, instance_labels
                )
                loss = loss_dict.total

            # Backward pass
            self.scaler.scale(loss).backward()

            # Gradient clipping
            if self.config.training.gradient_clip:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.gradient_clip,
                )

            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # Standard forward/backward
            output = self.model(coords, features)
            loss_dict = self.model.compute_loss(
                output, semantic_labels, instance_labels
            )
            loss = loss_dict.total

            loss.backward()

            # Gradient clipping
            if self.config.training.gradient_clip:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.gradient_clip,
                )

            self.optimizer.step()

        return loss.item()

    def _validate(self) -> Dict[str, float]:
        """Run validation."""
        self.model.eval()
        val_losses = []

        with torch.no_grad():
            for batch in self.val_loader:
                coords = batch["coords"].to(self.device)
                features = batch["features"].to(self.device)
                semantic_labels = batch.get("semantic_labels")
                instance_labels = batch.get("instance_labels")

                if semantic_labels is not None:
                    semantic_labels = semantic_labels.to(self.device)
                if instance_labels is not None:
                    instance_labels = instance_labels.to(self.device)

                output = self.model(coords, features)
                loss_dict = self.model.compute_loss(
                    output, semantic_labels, instance_labels
                )
                val_losses.append(loss_dict.total.item())

        avg_loss = np.mean(val_losses)
        logger.info(f"Validation loss: {avg_loss:.4f}")

        return {"val/loss": avg_loss}

    def _save_checkpoint(self) -> None:
        """Save training checkpoint."""
        checkpoint_dir = Path(self.config.paths.models)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = checkpoint_dir / f"checkpoint_epoch{self.current_epoch}.pt"

        save_model(
            model=self.model,
            path=checkpoint_path,
            config=self.config,
            optimizer=self.optimizer,
            epoch=self.current_epoch,
            metrics={"best_metric": self.best_metric},
        )

        # Also save as latest
        latest_path = checkpoint_dir / "checkpoint_latest.pt"
        save_model(
            model=self.model,
            path=latest_path,
            config=self.config,
            optimizer=self.optimizer,
            epoch=self.current_epoch,
            metrics={"best_metric": self.best_metric},
        )

    def _save_best_model(self) -> None:
        """Save best model checkpoint."""
        checkpoint_dir = Path(self.config.paths.models)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        best_path = checkpoint_dir / "checkpoint_best.pt"
        save_model(
            model=self.model,
            path=best_path,
            config=self.config,
            epoch=self.current_epoch,
            metrics={"best_metric": self.best_metric},
        )

        logger.info(f"Saved new best model (loss: {self.best_metric:.4f})")


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for training CLI."""
    parser = argparse.ArgumentParser(
        description="LiDAR Panoptic Segmentation Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Add standard config args
    add_config_args(parser)

    # Training-specific args
    parser.add_argument(
        "--resume",
        type=str,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only run validation (no training)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Override output directory",
    )

    return parser


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main(args: Optional[argparse.Namespace] = None) -> int:
    """
    Main training entry point.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code
    """
    if args is None:
        parser = create_parser()
        args = parser.parse_args()

    # Build config overrides
    overrides = {}
    if args.resume:
        overrides["training"] = {"resume_from": args.resume}
    if args.output_dir:
        overrides["paths"] = {"models": args.output_dir}

    # Load configuration
    try:
        config = load_config(
            config_path=getattr(args, "config", None),
            overrides=overrides,
            environment=getattr(args, "environment", None),
        )
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return 1

    # Apply seed
    seed = getattr(args, "seed", 42) or config.env.seed
    set_seed(seed)
    config.env.seed = seed

    # Apply debug mode
    if getattr(args, "debug", False):
        config.env.debug = True

    # Create and run trainer
    try:
        trainer = Trainer(config)
        trainer.setup()

        if getattr(args, "validate_only", False):
            if trainer.val_loader is None:
                print("No validation data configured")
                return 1
            trainer._validate()
        else:
            trainer.train()

        return 0

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return 130
    except Exception as e:
        logger.exception(f"Training failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
