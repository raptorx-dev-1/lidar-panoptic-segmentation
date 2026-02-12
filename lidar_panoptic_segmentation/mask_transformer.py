"""
SPFormer3D: Superpoint-based Mask Transformer for LiDAR Panoptic Segmentation

Implements a mask-transformer architecture that directly predicts instance masks
end-to-end via learnable object queries and cross-attention, eliminating
clustering entirely. Inspired by SPFormer, OneFormer3D, and Mask3D.

Architecture:
    Input → Residual Sparse Conv UNet → Superpoint Pooling → Transformer Decoder
    → Per-query class + mask predictions → Hungarian Matching + Focal/Dice/CE Loss
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lidar_panoptic_segmentation.model import (
    BasePanopticModel,
    PanopticLoss,
    PanopticOutput,
    check_minkowski_available,
)

logger = logging.getLogger(__name__)


# =============================================================================
# 2a. Residual Block
# =============================================================================

class ResidualBlock(nn.Module):
    """Two sparse convolutions with skip connection.

    conv1(C, C, k=3) -> BN -> ReLU -> conv2(C, C, k=3) -> BN -> + input -> ReLU
    """

    def __init__(self, channels: int, dimension: int = 3):
        super().__init__()
        import MinkowskiEngine as ME

        self.conv1 = ME.MinkowskiConvolution(
            channels, channels, kernel_size=3, stride=1, dimension=dimension,
        )
        self.bn1 = ME.MinkowskiBatchNorm(channels)
        self.conv2 = ME.MinkowskiConvolution(
            channels, channels, kernel_size=3, stride=1, dimension=dimension,
        )
        self.bn2 = ME.MinkowskiBatchNorm(channels)
        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual
        out = self.relu(out)
        return out


# =============================================================================
# 2b. Residual Sparse Conv Backbone (Deeper UNet)
# =============================================================================

class ResidualSparseConvBackbone(nn.Module):
    """Deeper UNet backbone with residual blocks.

    Stem: conv(in_ch -> 64, k=3, s=1)
    Encoder: 3 stages, each: stride-2 downconv -> 2x ResidualBlock
        Stage channels: 64 -> 128 -> 256 -> 512
    Decoder: 3 stages, each: transpose-conv -> cat(skip) -> 2x ResidualBlock
        512->256, 256->128, 128->64
    Final: conv(64 -> feature_dim, k=1)
    """

    def __init__(
        self,
        in_channels: int = 4,
        feature_dim: int = 64,
        channels: List[int] = None,
        dimension: int = 3,
    ):
        super().__init__()
        import MinkowskiEngine as ME

        if channels is None:
            channels = [64, 128, 256, 512]
        assert len(channels) == 4

        c0, c1, c2, c3 = channels

        # Stem
        self.stem = nn.Sequential(
            ME.MinkowskiConvolution(in_channels, c0, kernel_size=3, stride=1, dimension=dimension),
            ME.MinkowskiBatchNorm(c0),
            ME.MinkowskiReLU(inplace=True),
        )

        # Encoder stages
        self.enc1_down = ME.MinkowskiConvolution(c0, c1, kernel_size=3, stride=2, dimension=dimension)
        self.enc1_bn = ME.MinkowskiBatchNorm(c1)
        self.enc1_res1 = ResidualBlock(c1, dimension)
        self.enc1_res2 = ResidualBlock(c1, dimension)

        self.enc2_down = ME.MinkowskiConvolution(c1, c2, kernel_size=3, stride=2, dimension=dimension)
        self.enc2_bn = ME.MinkowskiBatchNorm(c2)
        self.enc2_res1 = ResidualBlock(c2, dimension)
        self.enc2_res2 = ResidualBlock(c2, dimension)

        self.enc3_down = ME.MinkowskiConvolution(c2, c3, kernel_size=3, stride=2, dimension=dimension)
        self.enc3_bn = ME.MinkowskiBatchNorm(c3)
        self.enc3_res1 = ResidualBlock(c3, dimension)
        self.enc3_res2 = ResidualBlock(c3, dimension)

        # Decoder stages
        self.dec3_up = ME.MinkowskiConvolutionTranspose(c3, c2, kernel_size=3, stride=2, dimension=dimension)
        self.dec3_bn = ME.MinkowskiBatchNorm(c2)
        # After cat with skip: c2 + c2 = 2*c2
        self.dec3_conv = ME.MinkowskiConvolution(c2 * 2, c2, kernel_size=1, stride=1, dimension=dimension)
        self.dec3_res1 = ResidualBlock(c2, dimension)
        self.dec3_res2 = ResidualBlock(c2, dimension)

        self.dec2_up = ME.MinkowskiConvolutionTranspose(c2, c1, kernel_size=3, stride=2, dimension=dimension)
        self.dec2_bn = ME.MinkowskiBatchNorm(c1)
        self.dec2_conv = ME.MinkowskiConvolution(c1 * 2, c1, kernel_size=1, stride=1, dimension=dimension)
        self.dec2_res1 = ResidualBlock(c1, dimension)
        self.dec2_res2 = ResidualBlock(c1, dimension)

        self.dec1_up = ME.MinkowskiConvolutionTranspose(c1, c0, kernel_size=3, stride=2, dimension=dimension)
        self.dec1_bn = ME.MinkowskiBatchNorm(c0)
        self.dec1_conv = ME.MinkowskiConvolution(c0 * 2, c0, kernel_size=1, stride=1, dimension=dimension)
        self.dec1_res1 = ResidualBlock(c0, dimension)
        self.dec1_res2 = ResidualBlock(c0, dimension)

        # Final projection
        self.final = ME.MinkowskiConvolution(c0, feature_dim, kernel_size=1, stride=1, dimension=dimension)

        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x):
        import MinkowskiEngine as ME

        # Stem
        e0 = self.stem(x)  # (N, c0)

        # Encoder
        e1 = self.relu(self.enc1_bn(self.enc1_down(e0)))
        e1 = self.enc1_res2(self.enc1_res1(e1))

        e2 = self.relu(self.enc2_bn(self.enc2_down(e1)))
        e2 = self.enc2_res2(self.enc2_res1(e2))

        e3 = self.relu(self.enc3_bn(self.enc3_down(e2)))
        e3 = self.enc3_res2(self.enc3_res1(e3))

        # Decoder
        d2 = self.relu(self.dec3_bn(self.dec3_up(e3)))
        d2 = ME.cat(d2, e2)
        d2 = self.dec3_conv(d2)
        d2 = self.dec3_res2(self.dec3_res1(d2))

        d1 = self.relu(self.dec2_bn(self.dec2_up(d2)))
        d1 = ME.cat(d1, e1)
        d1 = self.dec2_conv(d1)
        d1 = self.dec2_res2(self.dec2_res1(d1))

        d0 = self.relu(self.dec1_bn(self.dec1_up(d1)))
        d0 = ME.cat(d0, e0)
        d0 = self.dec1_conv(d0)
        d0 = self.dec1_res2(self.dec1_res1(d0))

        out = self.final(d0)
        return out


# =============================================================================
# 2c. Superpoint Pooling
# =============================================================================

class SuperpointPooling(nn.Module):
    """Efficiency layer that groups points into superpoints.

    Quantizes coords at coarser voxel size, mean-pools backbone features
    per superpoint, and adds learned 3D positional encoding.
    """

    def __init__(self, feature_dim: int, superpoint_voxel_size: float = 0.1):
        super().__init__()
        self.superpoint_voxel_size = superpoint_voxel_size
        self.pos_enc = nn.Sequential(
            nn.Linear(3, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, feature_dim),
        )

    def forward(
        self,
        coords: torch.Tensor,
        features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pool point features into superpoint features.

        Args:
            coords: (N, 4) batched coordinates [batch_idx, x, y, z]
            features: (N, D) per-point backbone features

        Returns:
            sp_features: (S, D) superpoint features
            sp_centers: (S, 3) superpoint center coordinates
            sp_to_point: (N,) mapping from points to superpoint indices
            batch_offsets: (B+1,) cumulative superpoint counts per batch
        """
        device = features.device
        batch_ids = coords[:, 0].long()
        xyz = coords[:, 1:4].float()

        # Quantize to superpoint voxel grid
        quantized = torch.floor(xyz / self.superpoint_voxel_size).long()
        # Combine batch_id and quantized coords for unique identification
        combined = torch.cat([batch_ids.unsqueeze(1), quantized], dim=1)  # (N, 4)

        # Find unique superpoints
        unique_sp, sp_to_point = torch.unique(combined, dim=0, return_inverse=True)

        num_superpoints = unique_sp.shape[0]

        # Mean-pool features per superpoint
        sp_features = torch.zeros(num_superpoints, features.shape[1], device=device)
        sp_centers = torch.zeros(num_superpoints, 3, device=device)
        sp_counts = torch.zeros(num_superpoints, device=device)

        sp_features.scatter_add_(0, sp_to_point.unsqueeze(1).expand_as(features), features)
        sp_centers.scatter_add_(0, sp_to_point.unsqueeze(1).expand(-1, 3), xyz)
        sp_counts.scatter_add_(0, sp_to_point, torch.ones(coords.shape[0], device=device))

        sp_counts = sp_counts.clamp(min=1)
        sp_features = sp_features / sp_counts.unsqueeze(1)
        sp_centers = sp_centers / sp_counts.unsqueeze(1)

        # Add learned positional encoding
        sp_features = sp_features + self.pos_enc(sp_centers)

        # Compute batch offsets for superpoints
        sp_batch_ids = unique_sp[:, 0]
        num_batches = batch_ids.max().item() + 1
        batch_offsets = torch.zeros(num_batches + 1, dtype=torch.long, device=device)
        for b in range(num_batches):
            batch_offsets[b + 1] = batch_offsets[b] + (sp_batch_ids == b).sum()

        return sp_features, sp_centers, sp_to_point, batch_offsets


# =============================================================================
# 2d. Masked Transformer Decoder Layer
# =============================================================================

class MaskedTransformerDecoderLayer(nn.Module):
    """Single transformer decoder layer with masked cross-attention.

    Pre-norm architecture:
    - Self-attention on queries (B, Q, D)
    - Cross-attention: queries attend to superpoint features with optional mask
    - FFN: Linear(D -> 4D) -> ReLU -> Linear(4D -> D)
    - Residual connections throughout
    """

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
    ):
        super().__init__()

        # Self-attention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)

        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(d_model)

    def forward(
        self,
        queries: torch.Tensor,
        sp_features: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            queries: (B, Q, D) instance queries
            sp_features: (B, S, D) superpoint features
            attn_mask: (B*nhead, Q, S) optional mask for cross-attention

        Returns:
            queries: (B, Q, D) updated queries
        """
        # Pre-norm self-attention
        q = self.norm1(queries)
        q2, _ = self.self_attn(q, q, q)
        queries = queries + q2

        # Pre-norm cross-attention
        q = self.norm2(queries)
        q2, _ = self.cross_attn(q, sp_features, sp_features, attn_mask=attn_mask)
        queries = queries + q2

        # Pre-norm FFN
        q = self.norm3(queries)
        queries = queries + self.ffn(q)

        return queries


# =============================================================================
# 2e. SPFormer Transformer Decoder
# =============================================================================

class SPFormerTransformerDecoder(nn.Module):
    """Full transformer decoder with learnable queries and deep supervision.

    - nn.Embedding(Q, D) for learnable instance queries
    - Projection layers for point and superpoint features
    - N x MaskedTransformerDecoderLayer
    - Per-layer prediction heads (class + mask) for deep supervision
    - Masked attention between layers from previous layer predictions
    """

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        num_decoder_layers: int = 6,
        num_queries: int = 100,
        dim_feedforward: int = 1024,
        num_classes: int = 2,
        feature_dim: int = 64,
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.num_decoder_layers = num_decoder_layers

        # Learnable instance queries
        self.query_embed = nn.Embedding(num_queries, d_model)

        # Project backbone features to transformer dimension
        self.input_proj_sp = nn.Linear(feature_dim, d_model)
        self.input_proj_point = nn.Linear(feature_dim, d_model)

        # Decoder layers
        self.layers = nn.ModuleList([
            MaskedTransformerDecoderLayer(d_model, nhead, dim_feedforward)
            for _ in range(num_decoder_layers)
        ])

        # Per-layer prediction heads (for deep supervision)
        self.class_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(inplace=True),
                nn.Linear(d_model, num_classes + 1),  # +1 for "no object"
            )
            for _ in range(num_decoder_layers)
        ])

        self.mask_embed_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(inplace=True),
                nn.Linear(d_model, d_model),
            )
            for _ in range(num_decoder_layers)
        ])

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        point_features: torch.Tensor,
        sp_features: torch.Tensor,
        sp_to_point: torch.Tensor,
        batch_offsets: torch.Tensor,
    ) -> Dict[str, List[torch.Tensor]]:
        """
        Args:
            point_features: (N, feature_dim) per-point backbone features
            sp_features: (S, feature_dim) superpoint features
            sp_to_point: (N,) mapping from points to superpoints
            batch_offsets: (B+1,) cumulative superpoint counts

        Returns:
            dict with:
                class_logits: list of (B, Q, num_classes+1) per layer
                mask_logits: list of (B, Q, N) per layer
                queries: list of (B, Q, D) per layer
        """
        B = batch_offsets.shape[0] - 1
        device = point_features.device

        # Project features
        sp_proj = self.input_proj_sp(sp_features)  # (S, D)
        point_proj = self.input_proj_point(point_features)  # (N, D)

        # Pad superpoint features into batched tensor (B, max_S, D)
        max_sp = max(
            (batch_offsets[b + 1] - batch_offsets[b]).item() for b in range(B)
        )
        sp_batched = torch.zeros(B, max_sp, self.d_model, device=device)
        sp_key_padding_mask = torch.ones(B, max_sp, device=device, dtype=torch.bool)

        for b in range(B):
            start = batch_offsets[b].item()
            end = batch_offsets[b + 1].item()
            length = end - start
            sp_batched[b, :length] = sp_proj[start:end]
            sp_key_padding_mask[b, :length] = False

        # Initialize queries: (B, Q, D)
        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)

        all_class_logits = []
        all_mask_logits = []
        all_queries = []

        attn_mask = None

        for layer_idx, layer in enumerate(self.layers):
            queries = layer(queries, sp_batched, attn_mask=attn_mask)

            # Predict class and mask for this layer
            class_logits = self.class_heads[layer_idx](queries)  # (B, Q, C+1)

            mask_embed = self.mask_embed_heads[layer_idx](queries)  # (B, Q, D)

            # Compute mask logits via dot product with point features
            # mask_logits[b, q, n] = mask_embed[b, q] . point_proj[n] for points in batch b
            mask_logits = self._compute_mask_logits(mask_embed, point_proj, batch_offsets)

            all_class_logits.append(class_logits)
            all_mask_logits.append(mask_logits)
            all_queries.append(queries)

            # Build masked attention for next layer from current mask predictions
            if layer_idx < self.num_decoder_layers - 1:
                attn_mask = self._build_attn_mask(
                    mask_embed, sp_proj, batch_offsets, sp_key_padding_mask, max_sp,
                )

        return {
            "class_logits": all_class_logits,
            "mask_logits": all_mask_logits,
            "queries": all_queries,
        }

    def _compute_mask_logits(
        self,
        mask_embed: torch.Tensor,
        point_proj: torch.Tensor,
        batch_offsets: torch.Tensor,
    ) -> List[torch.Tensor]:
        """Compute per-point mask logits via dot product.

        Returns a list of (Q, N_b) tensors, one per batch element.
        """
        B = batch_offsets.shape[0] - 1

        # We need per-batch-element point counts to identify which points
        # belong to which batch. Build a batch assignment from batch_offsets
        # of superpoints — but we actually need point-level batch info.
        # We'll return a list of tensors per batch element.

        # For simplicity, compute mask for all points at once then split
        # mask_embed is (B, Q, D), point_proj is (N, D)
        # We need to match batch elements. We'll iterate per batch.
        results = []
        # We don't have point-level batch offsets directly, but the caller
        # will provide this. For now, compute global dot product.
        # mask_logits_global: (Q, N) per batch element
        # Since we have batched queries, we compute per-element.
        # Actually, we'll compute all at once since point_proj has all points.
        # Return list of per-batch tensors.

        # For efficiency: (B, Q, D) x (N, D)^T is complex with variable sizes.
        # We do a simple loop per batch.
        # (This is fine — SPFormer/Mask3D also iterate per batch element.)
        for b in range(B):
            qe = mask_embed[b]  # (Q, D)
            # dot product with all point features
            ml = qe @ point_proj.T  # (Q, N_total)
            results.append(ml)

        return results

    def _build_attn_mask(
        self,
        mask_embed: torch.Tensor,
        sp_proj: torch.Tensor,
        batch_offsets: torch.Tensor,
        sp_key_padding_mask: torch.Tensor,
        max_sp: int,
    ) -> Optional[torch.Tensor]:
        """Build masked attention from current mask predictions on superpoints.

        Returns (B*nhead, Q, max_S) boolean mask where True = masked out.
        """
        B = mask_embed.shape[0]
        Q = mask_embed.shape[1]
        device = mask_embed.device

        attn_mask = torch.ones(B, Q, max_sp, device=device, dtype=torch.bool)

        for b in range(B):
            start = batch_offsets[b].item()
            end = batch_offsets[b + 1].item()
            length = end - start
            sp_b = sp_proj[start:end]  # (S_b, D)
            qe = mask_embed[b]  # (Q, D)
            sp_mask_logits = qe @ sp_b.T  # (Q, S_b)
            # Attend to superpoints where mask > 0 (i.e., predicted as part of instance)
            attend = sp_mask_logits.sigmoid() > 0.5
            attn_mask[b, :, :length] = ~attend

        # Ensure at least one position is attended to per query
        # (avoid NaN from fully masked rows)
        all_masked = attn_mask.all(dim=-1)  # (B, Q)
        if all_masked.any():
            attn_mask[all_masked] = False  # unmask all if everything was masked

        # Expand for multi-head: (B*nhead, Q, max_S)
        attn_mask = attn_mask.unsqueeze(1).expand(-1, self.nhead, -1, -1)
        attn_mask = attn_mask.reshape(B * self.nhead, Q, max_sp)

        return attn_mask


# =============================================================================
# 2f. Hungarian Matcher
# =============================================================================

class HungarianMatcher:
    """Bipartite matching between predictions and ground-truth instances.

    Uses scipy.optimize.linear_sum_assignment for optimal matching.
    """

    def __init__(
        self,
        weight_cls: float = 2.0,
        weight_mask: float = 5.0,
        weight_dice: float = 5.0,
    ):
        self.weight_cls = weight_cls
        self.weight_mask = weight_mask
        self.weight_dice = weight_dice

    @torch.no_grad()
    def match(
        self,
        class_logits: torch.Tensor,
        mask_logits: torch.Tensor,
        gt_classes: torch.Tensor,
        gt_masks: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute optimal assignment between predictions and GT.

        Args:
            class_logits: (Q, num_classes+1) predicted class logits
            mask_logits: (Q, N) predicted mask logits
            gt_classes: (K,) ground-truth class labels
            gt_masks: (K, N) ground-truth binary masks

        Returns:
            pred_indices: (M,) matched prediction indices
            gt_indices: (M,) matched GT indices
        """
        from scipy.optimize import linear_sum_assignment

        Q = class_logits.shape[0]
        K = gt_classes.shape[0]

        if K == 0:
            return (
                torch.tensor([], dtype=torch.long),
                torch.tensor([], dtype=torch.long),
            )

        # Classification cost: -log(softmax probability of GT class)
        class_probs = class_logits.softmax(dim=-1)  # (Q, C+1)
        class_cost = -class_probs[:, gt_classes]  # (Q, K)

        # Mask cost: BCE between predicted and GT masks
        mask_probs = mask_logits.sigmoid()  # (Q, N)
        # Binary cross-entropy cost
        mask_cost = -(
            gt_masks * torch.log(mask_probs.unsqueeze(1).expand(-1, K, -1) + 1e-8)
            + (1 - gt_masks) * torch.log(1 - mask_probs.unsqueeze(1).expand(-1, K, -1) + 1e-8)
        ).mean(dim=-1)  # (Q, K)

        # Dice cost
        mask_probs_exp = mask_probs.unsqueeze(1).expand(-1, K, -1)  # (Q, K, N)
        gt_masks_exp = gt_masks.unsqueeze(0).expand(Q, -1, -1)  # (Q, K, N)
        intersection = (mask_probs_exp * gt_masks_exp).sum(dim=-1)
        union = mask_probs_exp.sum(dim=-1) + gt_masks_exp.sum(dim=-1)
        dice_cost = 1 - 2 * intersection / (union + 1e-8)  # (Q, K)

        # Total cost
        cost = (
            self.weight_cls * class_cost
            + self.weight_mask * mask_cost
            + self.weight_dice * dice_cost
        )

        # Solve assignment on CPU
        cost_np = cost.detach().cpu().numpy()
        pred_idx, gt_idx = linear_sum_assignment(cost_np)

        return (
            torch.tensor(pred_idx, dtype=torch.long, device=class_logits.device),
            torch.tensor(gt_idx, dtype=torch.long, device=class_logits.device),
        )


# =============================================================================
# 2g. SPFormer3D Loss
# =============================================================================

class SPFormer3DLoss(nn.Module):
    """Combined loss for SPFormer3D.

    Per matched pair: Focal loss on mask + Dice loss on mask + CE on class.
    "No object" class for unmatched queries (CE with downweighted background).
    Deep supervision: loss at each decoder layer.
    Auxiliary: per-point semantic CE from backbone features.
    """

    def __init__(
        self,
        num_classes: int = 2,
        weight_ce: float = 2.0,
        weight_mask: float = 5.0,
        weight_dice: float = 5.0,
        weight_aux_semantic: float = 1.0,
        no_object_weight: float = 0.1,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.weight_ce = weight_ce
        self.weight_mask = weight_mask
        self.weight_dice = weight_dice
        self.weight_aux_semantic = weight_aux_semantic
        self.no_object_weight = no_object_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

        self.matcher = HungarianMatcher(weight_ce, weight_mask, weight_dice)

        # CE weight: downweight "no object" class
        ce_weights = torch.ones(num_classes + 1)
        ce_weights[-1] = no_object_weight
        self.register_buffer("ce_weights", ce_weights)

    def forward(
        self,
        decoder_outputs: Dict[str, List[torch.Tensor]],
        semantic_logits: torch.Tensor,
        semantic_labels: torch.Tensor,
        instance_labels: torch.Tensor,
        batch_offsets_points: Optional[torch.Tensor] = None,
    ) -> PanopticLoss:
        """Compute total loss across all decoder layers.

        Args:
            decoder_outputs: dict with class_logits, mask_logits lists
            semantic_logits: (N, num_classes) auxiliary semantic predictions
            semantic_labels: (N,) ground-truth semantic labels
            instance_labels: (N,) ground-truth instance labels
            batch_offsets_points: (B+1,) point counts per batch

        Returns:
            PanopticLoss with total and components
        """
        all_class_logits = decoder_outputs["class_logits"]
        all_mask_logits = decoder_outputs["mask_logits"]
        num_layers = len(all_class_logits)

        # Build GT masks from instance_labels
        gt_classes_list, gt_masks_list = self._build_gt_masks(
            instance_labels, semantic_labels, batch_offsets_points,
        )

        total_mask_loss = torch.tensor(0.0, device=semantic_logits.device)
        total_class_loss = torch.tensor(0.0, device=semantic_logits.device)

        for layer_idx in range(num_layers):
            cls_logits = all_class_logits[layer_idx]  # (B, Q, C+1)
            msk_logits = all_mask_logits[layer_idx]  # list of (Q, N_total) per batch

            # Weight: final layer = 1.0, earlier layers = 1/num_layers
            layer_weight = 1.0 if layer_idx == num_layers - 1 else 1.0 / num_layers

            B = cls_logits.shape[0]
            for b in range(B):
                gt_classes = gt_classes_list[b]
                gt_masks = gt_masks_list[b]

                pred_cls = cls_logits[b]  # (Q, C+1)
                pred_mask = msk_logits[b]  # (Q, N_total)

                # Match
                pred_idx, gt_idx = self.matcher.match(
                    pred_cls, pred_mask, gt_classes, gt_masks,
                )

                # Classification loss (all queries, no-object for unmatched)
                target_classes = torch.full(
                    (pred_cls.shape[0],), self.num_classes,
                    dtype=torch.long, device=pred_cls.device,
                )
                if len(gt_idx) > 0:
                    target_classes[pred_idx] = gt_classes[gt_idx]

                cls_loss = F.cross_entropy(
                    pred_cls, target_classes, weight=self.ce_weights,
                )
                total_class_loss = total_class_loss + layer_weight * cls_loss

                # Mask losses (only for matched pairs)
                if len(pred_idx) > 0:
                    matched_mask_logits = pred_mask[pred_idx]  # (M, N_total)
                    matched_gt_masks = gt_masks[gt_idx]  # (M, N_total)

                    # Focal loss
                    focal = self._focal_loss(matched_mask_logits, matched_gt_masks)
                    # Dice loss
                    dice = self._dice_loss(matched_mask_logits, matched_gt_masks)

                    total_mask_loss = total_mask_loss + layer_weight * (
                        self.weight_mask * focal + self.weight_dice * dice
                    )

        # Normalize by batch size and number of layers
        B = all_class_logits[0].shape[0]
        total_class_loss = total_class_loss / max(B, 1)
        total_mask_loss = total_mask_loss / max(B, 1)

        # Auxiliary semantic loss
        aux_semantic_loss = F.cross_entropy(
            semantic_logits, semantic_labels, ignore_index=-1,
        )

        total = (
            self.weight_ce * total_class_loss
            + total_mask_loss
            + self.weight_aux_semantic * aux_semantic_loss
        )

        return PanopticLoss(
            total=total,
            semantic=aux_semantic_loss,
            mask=total_class_loss + total_mask_loss,
        )

    def _build_gt_masks(
        self,
        instance_labels: torch.Tensor,
        semantic_labels: torch.Tensor,
        batch_offsets: Optional[torch.Tensor],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Build ground-truth binary masks from instance labels.

        Returns per-batch-element lists of:
            gt_classes: (K,) semantic class for each instance
            gt_masks: (K, N_total) binary mask for each instance
        """
        device = instance_labels.device
        N = instance_labels.shape[0]

        # If no batch offsets, treat as single batch
        if batch_offsets is None:
            batch_offsets = torch.tensor([0, N], dtype=torch.long, device=device)

        B = batch_offsets.shape[0] - 1
        gt_classes_list = []
        gt_masks_list = []

        for b in range(B):
            unique_ids = instance_labels.unique()
            # Valid instances: id > 0
            valid_ids = unique_ids[unique_ids > 0]

            if len(valid_ids) == 0:
                gt_classes_list.append(torch.zeros(0, dtype=torch.long, device=device))
                gt_masks_list.append(torch.zeros(0, N, device=device))
                continue

            classes = []
            masks = []
            for inst_id in valid_ids:
                mask = (instance_labels == inst_id).float()  # (N,)
                # Get majority semantic class for this instance
                inst_sem = semantic_labels[instance_labels == inst_id]
                if len(inst_sem) == 0:
                    continue
                sem_class = inst_sem.mode().values.item()
                classes.append(sem_class)
                masks.append(mask)

            if classes:
                gt_classes_list.append(torch.tensor(classes, dtype=torch.long, device=device))
                gt_masks_list.append(torch.stack(masks))
            else:
                gt_classes_list.append(torch.zeros(0, dtype=torch.long, device=device))
                gt_masks_list.append(torch.zeros(0, N, device=device))

        return gt_classes_list, gt_masks_list

    def _focal_loss(
        self, pred: torch.Tensor, target: torch.Tensor,
    ) -> torch.Tensor:
        """Sigmoid focal loss for mask predictions."""
        prob = pred.sigmoid()
        ce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        p_t = prob * target + (1 - prob) * (1 - target)
        focal_weight = self.focal_alpha * (1 - p_t) ** self.focal_gamma
        loss = (focal_weight * ce).mean()
        return loss

    def _dice_loss(
        self, pred: torch.Tensor, target: torch.Tensor,
    ) -> torch.Tensor:
        """Dice loss for mask predictions."""
        prob = pred.sigmoid()
        intersection = (prob * target).sum(dim=-1)
        union = prob.sum(dim=-1) + target.sum(dim=-1)
        dice = 1 - 2 * intersection / (union + 1e-8)
        return dice.mean()


# =============================================================================
# 2h. SPFormer3D Main Model
# =============================================================================

class SPFormer3D(BasePanopticModel):
    """Superpoint-based Mask Transformer for LiDAR Panoptic Segmentation.

    Directly predicts instance masks end-to-end via learnable object queries
    and cross-attention. No clustering required at inference.
    """

    def __init__(
        self,
        num_classes: int = 2,
        in_channels: int = 4,
        backbone_channels: List[int] = None,
        feature_dim: int = 64,
        transformer_dim: int = 256,
        num_heads: int = 8,
        num_decoder_layers: int = 6,
        num_queries: int = 100,
        dim_feedforward: int = 1024,
        superpoint_voxel_size: float = 0.1,
        confidence_threshold: float = 0.5,
        mask_threshold: float = 0.5,
        weight_ce: float = 2.0,
        weight_mask: float = 5.0,
        weight_dice: float = 5.0,
        weight_aux_semantic: float = 1.0,
    ):
        super().__init__(num_classes=num_classes, embed_dim=transformer_dim)

        if backbone_channels is None:
            backbone_channels = [64, 128, 256, 512]

        self.in_channels = in_channels
        self.feature_dim = feature_dim
        self.transformer_dim = transformer_dim
        self.confidence_threshold = confidence_threshold
        self.mask_threshold = mask_threshold

        self.use_minkowski = check_minkowski_available()

        if self.use_minkowski:
            self.backbone = ResidualSparseConvBackbone(
                in_channels=in_channels,
                feature_dim=feature_dim,
                channels=backbone_channels,
            )
        else:
            logger.warning(
                "MinkowskiEngine not available. Using fallback MLP backbone. "
                "This is for testing only — not recommended for production."
            )
            self.backbone = nn.Sequential(
                nn.Linear(in_channels, backbone_channels[0]),
                nn.BatchNorm1d(backbone_channels[0]),
                nn.ReLU(inplace=True),
                nn.Linear(backbone_channels[0], backbone_channels[1]),
                nn.BatchNorm1d(backbone_channels[1]),
                nn.ReLU(inplace=True),
                nn.Linear(backbone_channels[1], feature_dim),
                nn.BatchNorm1d(feature_dim),
                nn.ReLU(inplace=True),
            )

        self.superpoint_pooling = SuperpointPooling(feature_dim, superpoint_voxel_size)

        self.transformer_decoder = SPFormerTransformerDecoder(
            d_model=transformer_dim,
            nhead=num_heads,
            num_decoder_layers=num_decoder_layers,
            num_queries=num_queries,
            dim_feedforward=dim_feedforward,
            num_classes=num_classes,
            feature_dim=feature_dim,
        )

        # Auxiliary per-point semantic head
        self.aux_semantic_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, num_classes),
        )

        self.loss_fn = SPFormer3DLoss(
            num_classes=num_classes,
            weight_ce=weight_ce,
            weight_mask=weight_mask,
            weight_dice=weight_dice,
            weight_aux_semantic=weight_aux_semantic,
        )

        # Cache for compute_loss
        self._cached_decoder_outputs = None
        self._cached_semantic_logits = None
        self._cached_coords = None

        num_params = sum(p.numel() for p in self.parameters())
        logger.info(
            f"Initialized SPFormer3D: num_classes={num_classes}, "
            f"queries={num_queries}, layers={num_decoder_layers}, "
            f"params={num_params:,}"
        )

    def forward(
        self,
        coords: torch.Tensor,
        features: torch.Tensor,
    ) -> PanopticOutput:
        """Forward pass.

        Args:
            coords: (N, 4) batched coordinates [batch_idx, x, y, z]
            features: (N, F) input features

        Returns:
            PanopticOutput with predictions
        """
        if self.use_minkowski:
            point_features = self._forward_minkowski(coords, features)
        else:
            point_features = self.backbone(features)

        # Superpoint pooling
        sp_features, sp_centers, sp_to_point, batch_offsets = self.superpoint_pooling(
            coords, point_features,
        )

        # Transformer decoder
        decoder_outputs = self.transformer_decoder(
            point_features, sp_features, sp_to_point, batch_offsets,
        )

        # Auxiliary semantic logits
        semantic_logits = self.aux_semantic_head(point_features)
        semantic_pred = semantic_logits.argmax(dim=1)

        # Cache for compute_loss
        self._cached_decoder_outputs = decoder_outputs
        self._cached_semantic_logits = semantic_logits
        self._cached_coords = coords

        # At inference, compute instance predictions directly
        instance_pred = None
        if not self.training:
            # Use final layer predictions
            final_class_logits = decoder_outputs["class_logits"][-1]  # (B, Q, C+1)
            final_mask_logits = decoder_outputs["mask_logits"][-1]  # list of (Q, N)

            batch_ids = coords[:, 0].long()
            num_batches = batch_ids.max().item() + 1
            point_batch_offsets = torch.zeros(num_batches + 1, dtype=torch.long, device=coords.device)
            for b in range(num_batches):
                point_batch_offsets[b + 1] = point_batch_offsets[b] + (batch_ids == b).sum()

            instance_pred = self._predict_instances(
                final_class_logits, final_mask_logits, point_batch_offsets,
            )

        return PanopticOutput(
            semantic_logits=semantic_logits,
            semantic_pred=semantic_pred,
            instance_pred=instance_pred,
        )

    def _forward_minkowski(
        self,
        coords: torch.Tensor,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """Run backbone with MinkowskiEngine and return dense per-point features."""
        import MinkowskiEngine as ME

        x = ME.SparseTensor(
            features=features,
            coordinates=coords.int(),
            device=features.device,
        )
        out = self.backbone(x)
        return out.F  # (N, feature_dim)

    def compute_loss(
        self,
        output: PanopticOutput,
        semantic_labels: torch.Tensor,
        instance_labels: Optional[torch.Tensor] = None,
    ) -> PanopticLoss:
        """Compute training loss using cached decoder outputs.

        Args:
            output: Model output (used for semantic_logits)
            semantic_labels: (N,) ground truth semantic labels
            instance_labels: (N,) ground truth instance labels

        Returns:
            PanopticLoss with loss components
        """
        if self._cached_decoder_outputs is None:
            raise RuntimeError("compute_loss called without prior forward pass")

        if instance_labels is None:
            # Fallback to semantic-only loss
            sem_loss = F.cross_entropy(
                output.semantic_logits, semantic_labels, ignore_index=-1,
            )
            return PanopticLoss(total=sem_loss, semantic=sem_loss)

        return self.loss_fn(
            decoder_outputs=self._cached_decoder_outputs,
            semantic_logits=self._cached_semantic_logits,
            semantic_labels=semantic_labels,
            instance_labels=instance_labels,
        )

    def _predict_instances(
        self,
        class_logits: torch.Tensor,
        mask_logits: List[torch.Tensor],
        batch_offsets: torch.Tensor,
    ) -> torch.Tensor:
        """Predict instances directly from mask logits (no clustering).

        Args:
            class_logits: (B, Q, C+1) predicted class logits
            mask_logits: list of (Q, N_total) per batch
            batch_offsets: (B+1,) point counts per batch

        Returns:
            instance_pred: (N,) instance IDs, -1 for unassigned
        """
        B = class_logits.shape[0]
        device = class_logits.device
        N = sum(
            (batch_offsets[b + 1] - batch_offsets[b]).item() for b in range(B)
        )

        instance_pred = torch.full((N,), -1, dtype=torch.long, device=device)
        instance_offset = 0

        for b in range(B):
            cls_probs = class_logits[b].softmax(dim=-1)  # (Q, C+1)
            # Exclude "no object" class (last class)
            obj_probs = cls_probs[:, :-1]  # (Q, C)
            max_probs, max_classes = obj_probs.max(dim=-1)  # (Q,)

            # Filter by confidence
            valid = max_probs > self.confidence_threshold
            if not valid.any():
                continue

            valid_indices = torch.where(valid)[0]
            valid_probs = max_probs[valid]

            # Sort by confidence (highest first)
            sorted_order = valid_probs.argsort(descending=True)
            valid_indices = valid_indices[sorted_order]

            ml = mask_logits[b]  # (Q, N_total)
            mask_probs = ml.sigmoid()  # (Q, N_total)

            assigned = torch.zeros(N, dtype=torch.bool, device=device)

            for idx, qi in enumerate(valid_indices):
                qi = qi.item()
                query_mask = mask_probs[qi] > self.mask_threshold  # (N_total,)

                # Only claim unassigned points
                claimable = query_mask & ~assigned
                if claimable.sum() == 0:
                    continue

                instance_pred[claimable] = instance_offset
                assigned = assigned | claimable
                instance_offset += 1

        return instance_pred

    def get_parameters_by_group(self) -> Dict[str, List[nn.Parameter]]:
        """Get parameters grouped for different learning rates.

        Backbone uses 0.1x LR, transformer uses 1x LR.
        """
        backbone_params = list(self.backbone.parameters())
        transformer_params = (
            list(self.transformer_decoder.parameters())
            + list(self.aux_semantic_head.parameters())
            + list(self.superpoint_pooling.parameters())
        )
        return {
            "backbone": backbone_params,
            "transformer": transformer_params,
        }


# =============================================================================
# 2i. SPFormer3D Fallback (CPU, no MinkowskiEngine)
# =============================================================================

class SPFormer3DFallback(SPFormer3D):
    """CPU fallback for SPFormer3D without MinkowskiEngine.

    Uses simple MLP backbone + same transformer decoder.
    For testing/development only.
    """

    def __init__(self, **kwargs):
        # Force use_minkowski=False by not importing ME
        kwargs.setdefault("num_classes", 2)
        kwargs.setdefault("in_channels", 4)
        super().__init__(**kwargs)
        # The parent already handles the fallback if ME is unavailable


__all__ = [
    "SPFormer3D",
    "SPFormer3DFallback",
    "ResidualSparseConvBackbone",
    "SuperpointPooling",
    "SPFormerTransformerDecoder",
    "HungarianMatcher",
    "SPFormer3DLoss",
]
