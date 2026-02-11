"""
LiDAR Panoptic Segmentation Model Module

Provides architecture abstraction for panoptic segmentation models,
with support for MinkowskiEngine sparse convolutions and various backbones.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lidar_panoptic_segmentation.config_schema import Config, ModelArchitecture

logger = logging.getLogger(__name__)


@dataclass
class PanopticOutput:
    """Output container for panoptic segmentation predictions."""

    semantic_logits: torch.Tensor  # (N, num_classes) semantic class logits
    semantic_pred: torch.Tensor  # (N,) predicted semantic class
    instance_pred: Optional[torch.Tensor] = None  # (N,) predicted instance IDs
    offset_pred: Optional[torch.Tensor] = None  # (N, 3) offset predictions
    embedding: Optional[torch.Tensor] = None  # (N, embed_dim) instance embeddings
    confidence: Optional[torch.Tensor] = None  # (N,) prediction confidence
    mask_scores: Optional[torch.Tensor] = None  # Instance mask scores

    def to_numpy(self) -> Dict[str, np.ndarray]:
        """Convert all tensors to numpy arrays."""
        result = {
            "semantic_logits": self.semantic_logits.cpu().numpy(),
            "semantic_pred": self.semantic_pred.cpu().numpy(),
        }
        if self.instance_pred is not None:
            result["instance_pred"] = self.instance_pred.cpu().numpy()
        if self.offset_pred is not None:
            result["offset_pred"] = self.offset_pred.cpu().numpy()
        if self.embedding is not None:
            result["embedding"] = self.embedding.cpu().numpy()
        if self.confidence is not None:
            result["confidence"] = self.confidence.cpu().numpy()
        return result


@dataclass
class PanopticLoss:
    """Container for loss components."""

    total: torch.Tensor
    semantic: torch.Tensor
    offset: Optional[torch.Tensor] = None
    embedding: Optional[torch.Tensor] = None
    mask: Optional[torch.Tensor] = None

    def to_dict(self) -> Dict[str, float]:
        """Convert losses to dictionary for logging."""
        result = {
            "loss/total": self.total.item(),
            "loss/semantic": self.semantic.item(),
        }
        if self.offset is not None:
            result["loss/offset"] = self.offset.item()
        if self.embedding is not None:
            result["loss/embedding"] = self.embedding.item()
        if self.mask is not None:
            result["loss/mask"] = self.mask.item()
        return result


class BasePanopticModel(ABC, nn.Module):
    """
    Abstract base class for panoptic segmentation models.

    Provides common interface for training and inference.
    """

    def __init__(
        self,
        num_classes: int = 2,
        embed_dim: int = 5,
        **kwargs,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim

    @abstractmethod
    def forward(
        self,
        coords: torch.Tensor,
        features: torch.Tensor,
    ) -> PanopticOutput:
        """
        Forward pass.

        Args:
            coords: (N, 4) batched coordinates [batch_idx, x, y, z]
            features: (N, F) input features

        Returns:
            PanopticOutput with predictions
        """
        pass

    @abstractmethod
    def compute_loss(
        self,
        output: PanopticOutput,
        semantic_labels: torch.Tensor,
        instance_labels: Optional[torch.Tensor] = None,
    ) -> PanopticLoss:
        """
        Compute training loss.

        Args:
            output: Model output
            semantic_labels: (N,) ground truth semantic labels
            instance_labels: (N,) ground truth instance labels

        Returns:
            PanopticLoss with loss components
        """
        pass

    def get_parameters_by_group(self) -> Dict[str, List[nn.Parameter]]:
        """Get parameters grouped for different learning rates."""
        return {"default": list(self.parameters())}


def check_minkowski_available() -> bool:
    """Check if MinkowskiEngine is available."""
    try:
        import MinkowskiEngine as ME
        return True
    except ImportError:
        return False


class MinkowskiBackbone(nn.Module):
    """
    MinkowskiEngine-based sparse convolution backbone.

    Implements UNet-like architecture for point cloud processing.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        backbone_config: str = "MinkUNet34C",
        dimension: int = 3,
    ):
        super().__init__()

        if not check_minkowski_available():
            raise ImportError(
                "MinkowskiEngine is required for this backbone. "
                "Run the installation script: scripts/install_minkowski.sh"
            )

        import MinkowskiEngine as ME
        self.ME = ME

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dimension = dimension

        # Build UNet architecture
        # Encoder
        self.conv1 = ME.MinkowskiConvolution(
            in_channels, 32, kernel_size=3, stride=1, dimension=dimension
        )
        self.bn1 = ME.MinkowskiBatchNorm(32)

        self.conv2 = ME.MinkowskiConvolution(
            32, 64, kernel_size=3, stride=2, dimension=dimension
        )
        self.bn2 = ME.MinkowskiBatchNorm(64)

        self.conv3 = ME.MinkowskiConvolution(
            64, 128, kernel_size=3, stride=2, dimension=dimension
        )
        self.bn3 = ME.MinkowskiBatchNorm(128)

        self.conv4 = ME.MinkowskiConvolution(
            128, 256, kernel_size=3, stride=2, dimension=dimension
        )
        self.bn4 = ME.MinkowskiBatchNorm(256)

        # Decoder
        self.deconv4 = ME.MinkowskiConvolutionTranspose(
            256, 128, kernel_size=3, stride=2, dimension=dimension
        )
        self.debn4 = ME.MinkowskiBatchNorm(128)

        self.deconv3 = ME.MinkowskiConvolutionTranspose(
            256, 64, kernel_size=3, stride=2, dimension=dimension
        )
        self.debn3 = ME.MinkowskiBatchNorm(64)

        self.deconv2 = ME.MinkowskiConvolutionTranspose(
            128, 32, kernel_size=3, stride=2, dimension=dimension
        )
        self.debn2 = ME.MinkowskiBatchNorm(32)

        # Final convolution
        self.final_conv = ME.MinkowskiConvolution(
            64, out_channels, kernel_size=1, stride=1, dimension=dimension
        )

        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x):
        """Forward pass through UNet backbone."""
        # Encoder
        e1 = self.relu(self.bn1(self.conv1(x)))
        e2 = self.relu(self.bn2(self.conv2(e1)))
        e3 = self.relu(self.bn3(self.conv3(e2)))
        e4 = self.relu(self.bn4(self.conv4(e3)))

        # Decoder with skip connections
        d4 = self.relu(self.debn4(self.deconv4(e4)))
        # Concatenate with encoder features
        d4 = self.ME.cat(d4, e3)

        d3 = self.relu(self.debn3(self.deconv3(d4)))
        d3 = self.ME.cat(d3, e2)

        d2 = self.relu(self.debn2(self.deconv2(d3)))
        d2 = self.ME.cat(d2, e1)

        # Final output
        out = self.final_conv(d2)

        return out


class SemanticHead(nn.Module):
    """Semantic segmentation head."""

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()

        if check_minkowski_available():
            import MinkowskiEngine as ME
            self.classifier = ME.MinkowskiConvolution(
                in_channels, num_classes, kernel_size=1, stride=1, dimension=3
            )
            self.use_minkowski = True
        else:
            self.classifier = nn.Linear(in_channels, num_classes)
            self.use_minkowski = False

    def forward(self, x):
        if self.use_minkowski:
            return self.classifier(x)
        return self.classifier(x)


class OffsetHead(nn.Module):
    """Offset prediction head for instance center regression."""

    def __init__(self, in_channels: int, hidden_dim: int = 64):
        super().__init__()

        if check_minkowski_available():
            import MinkowskiEngine as ME
            self.mlp = nn.Sequential(
                ME.MinkowskiConvolution(in_channels, hidden_dim, kernel_size=1, dimension=3),
                ME.MinkowskiBatchNorm(hidden_dim),
                ME.MinkowskiReLU(),
                ME.MinkowskiConvolution(hidden_dim, 3, kernel_size=1, dimension=3),
            )
            self.use_minkowski = True
        else:
            self.mlp = nn.Sequential(
                nn.Linear(in_channels, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 3),
            )
            self.use_minkowski = False

    def forward(self, x):
        return self.mlp(x)


class EmbeddingHead(nn.Module):
    """Instance embedding head for discriminative clustering."""

    def __init__(self, in_channels: int, embed_dim: int = 5, hidden_dim: int = 64):
        super().__init__()

        if check_minkowski_available():
            import MinkowskiEngine as ME
            self.mlp = nn.Sequential(
                ME.MinkowskiConvolution(in_channels, hidden_dim, kernel_size=1, dimension=3),
                ME.MinkowskiBatchNorm(hidden_dim),
                ME.MinkowskiReLU(),
                ME.MinkowskiConvolution(hidden_dim, embed_dim, kernel_size=1, dimension=3),
            )
            self.use_minkowski = True
        else:
            self.mlp = nn.Sequential(
                nn.Linear(in_channels, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, embed_dim),
            )
            self.use_minkowski = False

    def forward(self, x):
        return self.mlp(x)


class LidarPanopticModel(BasePanopticModel):
    """
    Main LiDAR Panoptic Segmentation Model.

    Implements PointGroup-style 3-head architecture with:
    - Semantic segmentation head
    - Offset prediction head
    - Instance embedding head
    """

    def __init__(
        self,
        num_classes: int = 2,
        embed_dim: int = 5,
        in_channels: int = 4,  # xyz + intensity
        backbone_channels: int = 128,
        backbone_config: str = "MinkUNet34C",
        use_offset_head: bool = True,
        use_embedding_head: bool = True,
        semantic_weight: float = 1.0,
        offset_weight: float = 1.0,
        embedding_weight: float = 1.0,
    ):
        super().__init__(num_classes=num_classes, embed_dim=embed_dim)

        self.in_channels = in_channels
        self.backbone_channels = backbone_channels
        self.use_offset_head = use_offset_head
        self.use_embedding_head = use_embedding_head

        # Loss weights
        self.semantic_weight = semantic_weight
        self.offset_weight = offset_weight
        self.embedding_weight = embedding_weight

        # Check for MinkowskiEngine
        self.use_minkowski = check_minkowski_available()

        if self.use_minkowski:
            self._build_minkowski_model(in_channels, backbone_channels, backbone_config)
        else:
            self._build_fallback_model(in_channels, backbone_channels)

        logger.info(
            f"Initialized LidarPanopticModel: "
            f"num_classes={num_classes}, embed_dim={embed_dim}, "
            f"use_minkowski={self.use_minkowski}"
        )

    def _build_minkowski_model(
        self,
        in_channels: int,
        backbone_channels: int,
        backbone_config: str,
    ):
        """Build model with MinkowskiEngine sparse convolutions."""
        import MinkowskiEngine as ME

        self.backbone = MinkowskiBackbone(
            in_channels=in_channels,
            out_channels=backbone_channels,
            backbone_config=backbone_config,
        )

        self.semantic_head = SemanticHead(backbone_channels, self.num_classes)

        if self.use_offset_head:
            self.offset_head = OffsetHead(backbone_channels)

        if self.use_embedding_head:
            self.embedding_head = EmbeddingHead(backbone_channels, self.embed_dim)

    def _build_fallback_model(self, in_channels: int, backbone_channels: int):
        """Build fallback model without MinkowskiEngine (for CPU testing)."""
        logger.warning(
            "MinkowskiEngine not available. Using fallback MLP model. "
            "This is not recommended for production use."
        )

        self.backbone = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, backbone_channels),
            nn.BatchNorm1d(backbone_channels),
            nn.ReLU(),
        )

        self.semantic_head = nn.Linear(backbone_channels, self.num_classes)

        if self.use_offset_head:
            self.offset_head = nn.Sequential(
                nn.Linear(backbone_channels, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Linear(64, 3),
            )

        if self.use_embedding_head:
            self.embedding_head = nn.Sequential(
                nn.Linear(backbone_channels, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Linear(64, self.embed_dim),
            )

    def forward(
        self,
        coords: torch.Tensor,
        features: torch.Tensor,
    ) -> PanopticOutput:
        """
        Forward pass.

        Args:
            coords: (N, 4) batched coordinates [batch_idx, x, y, z]
            features: (N, F) input features

        Returns:
            PanopticOutput with predictions
        """
        if self.use_minkowski:
            return self._forward_minkowski(coords, features)
        else:
            return self._forward_fallback(coords, features)

    def _forward_minkowski(
        self,
        coords: torch.Tensor,
        features: torch.Tensor,
    ) -> PanopticOutput:
        """Forward pass using MinkowskiEngine."""
        import MinkowskiEngine as ME

        # Create sparse tensor
        x = ME.SparseTensor(
            features=features,
            coordinates=coords.int(),
            device=features.device,
        )

        # Backbone
        backbone_out = self.backbone(x)

        # Semantic prediction
        semantic_out = self.semantic_head(backbone_out)
        semantic_logits = semantic_out.F
        semantic_pred = semantic_logits.argmax(dim=1)

        # Offset prediction
        offset_pred = None
        if self.use_offset_head:
            offset_out = self.offset_head(backbone_out)
            offset_pred = offset_out.F

        # Embedding prediction
        embedding = None
        if self.use_embedding_head:
            embedding_out = self.embedding_head(backbone_out)
            embedding = embedding_out.F

        return PanopticOutput(
            semantic_logits=semantic_logits,
            semantic_pred=semantic_pred,
            offset_pred=offset_pred,
            embedding=embedding,
        )

    def _forward_fallback(
        self,
        coords: torch.Tensor,
        features: torch.Tensor,
    ) -> PanopticOutput:
        """Forward pass using fallback MLP model."""
        # Use only features (coords are implicit)
        backbone_out = self.backbone(features)

        # Semantic prediction
        semantic_logits = self.semantic_head(backbone_out)
        semantic_pred = semantic_logits.argmax(dim=1)

        # Offset prediction
        offset_pred = None
        if self.use_offset_head:
            offset_pred = self.offset_head(backbone_out)

        # Embedding prediction
        embedding = None
        if self.use_embedding_head:
            embedding = self.embedding_head(backbone_out)

        return PanopticOutput(
            semantic_logits=semantic_logits,
            semantic_pred=semantic_pred,
            offset_pred=offset_pred,
            embedding=embedding,
        )

    def compute_loss(
        self,
        output: PanopticOutput,
        semantic_labels: torch.Tensor,
        instance_labels: Optional[torch.Tensor] = None,
    ) -> PanopticLoss:
        """
        Compute training loss.

        Args:
            output: Model output
            semantic_labels: (N,) ground truth semantic labels
            instance_labels: (N,) ground truth instance labels

        Returns:
            PanopticLoss with loss components
        """
        # Semantic cross-entropy loss
        semantic_loss = F.cross_entropy(
            output.semantic_logits,
            semantic_labels,
            ignore_index=-1,
        )

        total_loss = self.semantic_weight * semantic_loss

        # Offset loss (L1 regression to instance centers)
        offset_loss = None
        if output.offset_pred is not None and instance_labels is not None:
            offset_loss = self._compute_offset_loss(
                output.offset_pred, instance_labels
            )
            total_loss = total_loss + self.offset_weight * offset_loss

        # Embedding loss (discriminative loss)
        embedding_loss = None
        if output.embedding is not None and instance_labels is not None:
            embedding_loss = self._compute_embedding_loss(
                output.embedding, instance_labels
            )
            total_loss = total_loss + self.embedding_weight * embedding_loss

        return PanopticLoss(
            total=total_loss,
            semantic=semantic_loss,
            offset=offset_loss,
            embedding=embedding_loss,
        )

    def _compute_offset_loss(
        self,
        offset_pred: torch.Tensor,
        instance_labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute offset regression loss."""
        # Only compute for valid instances (instance_id > 0)
        valid_mask = instance_labels > 0
        if not valid_mask.any():
            return torch.tensor(0.0, device=offset_pred.device)

        # Placeholder: compute L1 loss to instance centers
        # In full implementation, compute actual instance centers
        return torch.abs(offset_pred[valid_mask]).mean()

    def _compute_embedding_loss(
        self,
        embedding: torch.Tensor,
        instance_labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute discriminative embedding loss."""
        # Only compute for valid instances
        valid_mask = instance_labels > 0
        if not valid_mask.any():
            return torch.tensor(0.0, device=embedding.device)

        valid_embedding = embedding[valid_mask]
        valid_labels = instance_labels[valid_mask]

        # Compute variance (intra-cluster) loss
        unique_labels = valid_labels.unique()
        var_loss = torch.tensor(0.0, device=embedding.device)

        for label in unique_labels:
            mask = valid_labels == label
            if mask.sum() > 1:
                cluster_embedding = valid_embedding[mask]
                center = cluster_embedding.mean(dim=0, keepdim=True)
                var_loss = var_loss + torch.mean((cluster_embedding - center) ** 2)

        var_loss = var_loss / max(len(unique_labels), 1)

        # Compute distance (inter-cluster) loss
        # Encourage cluster centers to be far apart
        dist_loss = torch.tensor(0.0, device=embedding.device)
        if len(unique_labels) > 1:
            centers = []
            for label in unique_labels:
                mask = valid_labels == label
                centers.append(valid_embedding[mask].mean(dim=0))
            centers = torch.stack(centers)

            for i in range(len(centers)):
                for j in range(i + 1, len(centers)):
                    dist = torch.norm(centers[i] - centers[j])
                    margin = 1.5
                    dist_loss = dist_loss + F.relu(margin - dist) ** 2

            dist_loss = dist_loss / (len(unique_labels) * (len(unique_labels) - 1) / 2)

        return var_loss + dist_loss


def create_model(config: Config, device: Optional[str] = None) -> LidarPanopticModel:
    """
    Create a panoptic segmentation model from configuration.

    Args:
        config: Configuration object
        device: Target device (cuda/cpu)

    Returns:
        Initialized model
    """
    if device is None:
        device = config.get_effective_device()

    model = LidarPanopticModel(
        num_classes=config.training.num_classes,
        embed_dim=config.training.embed_dim,
        in_channels=4,  # xyz + intensity
        backbone_channels=128,
        backbone_config=config.training.backbone,
        use_offset_head=True,
        use_embedding_head=True,
    )

    model = model.to(device)

    # Log model info
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"Created model with {num_params:,} parameters ({num_trainable:,} trainable)"
    )

    return model


def load_model(
    model_uri: str,
    config: Optional[Config] = None,
    device: Optional[str] = None,
) -> LidarPanopticModel:
    """
    Load a trained model from checkpoint or MLflow.

    Args:
        model_uri: Path to checkpoint or MLflow model URI
        config: Optional configuration for model initialization
        device: Target device

    Returns:
        Loaded model
    """
    # Determine device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Check if MLflow URI
    if model_uri.startswith("models:/"):
        return _load_from_mlflow(model_uri, device)

    # Load from local checkpoint
    return _load_from_checkpoint(model_uri, config, device)


def _load_from_mlflow(model_uri: str, device: str) -> LidarPanopticModel:
    """Load model from MLflow model registry."""
    try:
        import mlflow.pytorch

        logger.info(f"Loading model from MLflow: {model_uri}")
        model = mlflow.pytorch.load_model(model_uri)
        model = model.to(device)
        model.eval()
        return model
    except ImportError:
        raise ImportError("MLflow is required for loading from model registry")


def _load_from_checkpoint(
    checkpoint_path: str,
    config: Optional[Config],
    device: str,
) -> LidarPanopticModel:
    """Load model from local checkpoint file."""
    logger.info(f"Loading model from checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Get model config from checkpoint or use provided config
    if "config" in checkpoint and config is None:
        model_config = checkpoint["config"]
        model = LidarPanopticModel(**model_config)
    elif config is not None:
        model = create_model(config, device)
    else:
        raise ValueError(
            "No config found in checkpoint and no config provided"
        )

    # Load state dict
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    return model


def save_model(
    model: LidarPanopticModel,
    path: Union[str, Path],
    config: Optional[Config] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: Optional[int] = None,
    metrics: Optional[Dict[str, float]] = None,
) -> None:
    """
    Save model checkpoint.

    Args:
        model: Model to save
        path: Output path
        config: Configuration to save with model
        optimizer: Optimizer state to save
        epoch: Current epoch
        metrics: Training metrics to save
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": {
            "num_classes": model.num_classes,
            "embed_dim": model.embed_dim,
            "in_channels": model.in_channels,
            "backbone_channels": model.backbone_channels,
            "use_offset_head": model.use_offset_head,
            "use_embedding_head": model.use_embedding_head,
        },
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    if epoch is not None:
        checkpoint["epoch"] = epoch

    if metrics is not None:
        checkpoint["metrics"] = metrics

    if config is not None:
        checkpoint["full_config"] = config.model_dump()

    torch.save(checkpoint, path)
    logger.info(f"Model saved to {path}")


# Convenience exports
__all__ = [
    "LidarPanopticModel",
    "BasePanopticModel",
    "PanopticOutput",
    "PanopticLoss",
    "create_model",
    "load_model",
    "save_model",
    "check_minkowski_available",
]
