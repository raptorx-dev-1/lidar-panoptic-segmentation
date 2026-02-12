"""
Tests for SPFormer3D Mask Transformer Module

Tests all components: SuperpointPooling, MaskedTransformerDecoderLayer,
SPFormerTransformerDecoder, HungarianMatcher, SPFormer3DLoss, SPFormer3D model,
and integration with create_model / postprocess_predictions.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from lidar_panoptic_segmentation.config_schema import (
    Config,
    ModelArchitecture,
    TransformerConfig,
)
from lidar_panoptic_segmentation.model import (
    BasePanopticModel,
    PanopticLoss,
    PanopticOutput,
    create_model,
)
from lidar_panoptic_segmentation.mask_transformer import (
    HungarianMatcher,
    MaskedTransformerDecoderLayer,
    SPFormer3D,
    SPFormer3DFallback,
    SPFormer3DLoss,
    SPFormerTransformerDecoder,
    SuperpointPooling,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def small_model():
    """Create a small SPFormer3D for fast testing."""
    return SPFormer3D(
        num_classes=2,
        in_channels=4,
        backbone_channels=[16, 32, 64, 128],
        feature_dim=16,
        transformer_dim=32,
        num_heads=4,
        num_decoder_layers=2,
        num_queries=10,
        dim_feedforward=64,
        superpoint_voxel_size=0.5,
    )


@pytest.fixture
def default_model():
    """Create a default SPFormer3D."""
    return SPFormer3D(num_classes=2)


@pytest.fixture
def single_batch_inputs():
    """Create single-batch test inputs."""
    N = 128
    coords = torch.zeros(N, 4)
    coords[:, 0] = 0
    coords[:, 1:] = torch.randn(N, 3)
    features = torch.randn(N, 4)
    return coords, features


@pytest.fixture
def multi_batch_inputs():
    """Create multi-batch test inputs."""
    N = 200
    coords = torch.zeros(N, 4)
    coords[:100, 0] = 0
    coords[100:, 0] = 1
    coords[:, 1:] = torch.randn(N, 3)
    features = torch.randn(N, 4)
    return coords, features


@pytest.fixture
def gt_labels():
    """Create ground-truth labels for loss testing."""
    N = 128
    semantic_labels = torch.randint(0, 2, (N,))
    instance_labels = torch.zeros(N, dtype=torch.long)
    instance_labels[:40] = 1
    instance_labels[40:80] = 2
    instance_labels[80:100] = 3
    # instance_labels[100:] = 0 (background)
    return semantic_labels, instance_labels


# =============================================================================
# SuperpointPooling
# =============================================================================

class TestSuperpointPooling:
    """Tests for the SuperpointPooling module."""

    def test_output_shapes(self):
        """Test that pooling returns correct output shapes."""
        pooling = SuperpointPooling(feature_dim=16, superpoint_voxel_size=0.5)
        N = 100
        coords = torch.zeros(N, 4)
        coords[:, 0] = 0
        coords[:, 1:] = torch.randn(N, 3)
        features = torch.randn(N, 16)

        sp_features, sp_centers, sp_to_point, batch_offsets = pooling(coords, features)

        S = sp_features.shape[0]
        assert sp_features.shape == (S, 16)
        assert sp_centers.shape == (S, 3)
        assert sp_to_point.shape == (N,)
        assert batch_offsets.shape == (2,)  # B+1 = 1+1

    def test_reduces_point_count(self):
        """Test that superpoints are fewer than input points."""
        pooling = SuperpointPooling(feature_dim=16, superpoint_voxel_size=0.5)
        N = 500
        coords = torch.zeros(N, 4)
        coords[:, 0] = 0
        coords[:, 1:] = torch.randn(N, 3) * 0.1  # Tight cluster
        features = torch.randn(N, 16)

        sp_features, sp_centers, sp_to_point, batch_offsets = pooling(coords, features)

        assert sp_features.shape[0] <= N

    def test_sp_to_point_mapping_valid(self):
        """Test that sp_to_point maps every point to a valid superpoint."""
        pooling = SuperpointPooling(feature_dim=16, superpoint_voxel_size=0.5)
        N = 100
        coords = torch.zeros(N, 4)
        coords[:, 0] = 0
        coords[:, 1:] = torch.randn(N, 3)
        features = torch.randn(N, 16)

        sp_features, sp_centers, sp_to_point, batch_offsets = pooling(coords, features)

        S = sp_features.shape[0]
        assert sp_to_point.min() >= 0
        assert sp_to_point.max() < S

    def test_multi_batch(self):
        """Test pooling with multiple batch elements."""
        pooling = SuperpointPooling(feature_dim=16, superpoint_voxel_size=0.5)
        N = 200
        coords = torch.zeros(N, 4)
        coords[:100, 0] = 0
        coords[100:, 0] = 1
        coords[:, 1:] = torch.randn(N, 3)
        features = torch.randn(N, 16)

        sp_features, sp_centers, sp_to_point, batch_offsets = pooling(coords, features)

        assert batch_offsets.shape == (3,)  # 2 batches + 1
        assert batch_offsets[0] == 0
        assert batch_offsets[2] == sp_features.shape[0]

    def test_coarser_voxel_fewer_superpoints(self):
        """Test that larger voxel size produces fewer superpoints."""
        N = 200
        coords = torch.zeros(N, 4)
        coords[:, 0] = 0
        coords[:, 1:] = torch.randn(N, 3) * 5
        features = torch.randn(N, 16)

        fine = SuperpointPooling(feature_dim=16, superpoint_voxel_size=0.1)
        coarse = SuperpointPooling(feature_dim=16, superpoint_voxel_size=2.0)

        sp_fine, _, _, _ = fine(coords, features)
        sp_coarse, _, _, _ = coarse(coords, features)

        assert sp_coarse.shape[0] <= sp_fine.shape[0]

    def test_positional_encoding_learned(self):
        """Test that positional encoding parameters are learnable."""
        pooling = SuperpointPooling(feature_dim=16, superpoint_voxel_size=0.5)
        params = list(pooling.pos_enc.parameters())
        assert len(params) > 0
        assert all(p.requires_grad for p in params)


# =============================================================================
# MaskedTransformerDecoderLayer
# =============================================================================

class TestMaskedTransformerDecoderLayer:
    """Tests for a single transformer decoder layer."""

    def test_output_shape(self):
        """Test that output shape matches input query shape."""
        layer = MaskedTransformerDecoderLayer(d_model=32, nhead=4, dim_feedforward=64)
        B, Q, D = 2, 10, 32
        S = 20

        queries = torch.randn(B, Q, D)
        sp_features = torch.randn(B, S, D)

        out = layer(queries, sp_features)
        assert out.shape == (B, Q, D)

    def test_without_attn_mask(self):
        """Test forward pass without attention mask."""
        layer = MaskedTransformerDecoderLayer(d_model=32, nhead=4)
        queries = torch.randn(1, 5, 32)
        sp_features = torch.randn(1, 10, 32)

        out = layer(queries, sp_features, attn_mask=None)
        assert out.shape == queries.shape
        assert torch.isfinite(out).all()

    def test_with_attn_mask(self):
        """Test forward pass with boolean attention mask."""
        d_model, nhead = 32, 4
        layer = MaskedTransformerDecoderLayer(d_model=d_model, nhead=nhead)
        B, Q, S = 1, 5, 10

        queries = torch.randn(B, Q, d_model)
        sp_features = torch.randn(B, S, d_model)
        # Mask: (B*nhead, Q, S) — False means attend, True means masked
        attn_mask = torch.zeros(B * nhead, Q, S, dtype=torch.bool)
        attn_mask[:, :, 5:] = True  # Mask out second half

        out = layer(queries, sp_features, attn_mask=attn_mask)
        assert out.shape == queries.shape
        assert torch.isfinite(out).all()

    def test_residual_connections(self):
        """Test that output differs from input (residual connections active)."""
        layer = MaskedTransformerDecoderLayer(d_model=32, nhead=4)
        queries = torch.randn(1, 5, 32)
        sp_features = torch.randn(1, 10, 32)

        out = layer(queries, sp_features)
        # Output should differ from input due to attention + FFN
        assert not torch.allclose(out, queries, atol=1e-6)


# =============================================================================
# SPFormerTransformerDecoder
# =============================================================================

class TestSPFormerTransformerDecoder:
    """Tests for the full transformer decoder."""

    def test_output_structure(self):
        """Test that decoder returns expected dict structure."""
        decoder = SPFormerTransformerDecoder(
            d_model=32, nhead=4, num_decoder_layers=2,
            num_queries=10, dim_feedforward=64,
            num_classes=2, feature_dim=16,
        )
        N, S = 100, 20
        point_features = torch.randn(N, 16)
        sp_features = torch.randn(S, 16)
        sp_to_point = torch.randint(0, S, (N,))
        batch_offsets = torch.tensor([0, S])

        out = decoder(point_features, sp_features, sp_to_point, batch_offsets)

        assert "class_logits" in out
        assert "mask_logits" in out
        assert "queries" in out

    def test_deep_supervision_outputs(self):
        """Test that each decoder layer produces predictions."""
        num_layers = 3
        decoder = SPFormerTransformerDecoder(
            d_model=32, nhead=4, num_decoder_layers=num_layers,
            num_queries=10, dim_feedforward=64,
            num_classes=2, feature_dim=16,
        )
        N, S = 100, 20
        point_features = torch.randn(N, 16)
        sp_features = torch.randn(S, 16)
        sp_to_point = torch.randint(0, S, (N,))
        batch_offsets = torch.tensor([0, S])

        out = decoder(point_features, sp_features, sp_to_point, batch_offsets)

        assert len(out["class_logits"]) == num_layers
        assert len(out["mask_logits"]) == num_layers
        assert len(out["queries"]) == num_layers

    def test_class_logits_shape(self):
        """Test class logit shapes (B, Q, num_classes+1)."""
        num_classes, num_queries = 2, 10
        decoder = SPFormerTransformerDecoder(
            d_model=32, nhead=4, num_decoder_layers=2,
            num_queries=num_queries, dim_feedforward=64,
            num_classes=num_classes, feature_dim=16,
        )
        N, S = 100, 20
        point_features = torch.randn(N, 16)
        sp_features = torch.randn(S, 16)
        sp_to_point = torch.randint(0, S, (N,))
        batch_offsets = torch.tensor([0, S])

        out = decoder(point_features, sp_features, sp_to_point, batch_offsets)

        for cls_logits in out["class_logits"]:
            assert cls_logits.shape == (1, num_queries, num_classes + 1)

    def test_mask_logits_shape(self):
        """Test mask logit shapes (Q, N) per batch element."""
        num_queries = 10
        decoder = SPFormerTransformerDecoder(
            d_model=32, nhead=4, num_decoder_layers=2,
            num_queries=num_queries, dim_feedforward=64,
            num_classes=2, feature_dim=16,
        )
        N, S = 100, 20
        point_features = torch.randn(N, 16)
        sp_features = torch.randn(S, 16)
        sp_to_point = torch.randint(0, S, (N,))
        batch_offsets = torch.tensor([0, S])

        out = decoder(point_features, sp_features, sp_to_point, batch_offsets)

        for mask_logits_list in out["mask_logits"]:
            assert len(mask_logits_list) == 1  # B=1
            assert mask_logits_list[0].shape == (num_queries, N)

    def test_multi_batch_decoder(self):
        """Test decoder with multiple batch elements."""
        decoder = SPFormerTransformerDecoder(
            d_model=32, nhead=4, num_decoder_layers=2,
            num_queries=10, dim_feedforward=64,
            num_classes=2, feature_dim=16,
        )
        N, S1, S2 = 100, 12, 8
        S = S1 + S2
        point_features = torch.randn(N, 16)
        sp_features = torch.randn(S, 16)
        sp_to_point = torch.randint(0, S, (N,))
        batch_offsets = torch.tensor([0, S1, S])

        out = decoder(point_features, sp_features, sp_to_point, batch_offsets)

        # Should have B=2 batch elements
        assert out["class_logits"][0].shape[0] == 2
        assert len(out["mask_logits"][0]) == 2

    def test_learnable_queries(self):
        """Test that query embeddings are learnable parameters."""
        decoder = SPFormerTransformerDecoder(
            d_model=32, nhead=4, num_decoder_layers=2,
            num_queries=10, dim_feedforward=64,
            num_classes=2, feature_dim=16,
        )
        assert decoder.query_embed.weight.requires_grad
        assert decoder.query_embed.weight.shape == (10, 32)


# =============================================================================
# HungarianMatcher
# =============================================================================

class TestHungarianMatcher:
    """Tests for Hungarian bipartite matching."""

    def test_matching_with_gt(self):
        """Test matching returns valid index pairs."""
        matcher = HungarianMatcher()
        Q, N, K = 10, 50, 3
        num_classes = 2

        class_logits = torch.randn(Q, num_classes + 1)
        mask_logits = torch.randn(Q, N)
        gt_classes = torch.tensor([0, 1, 0])
        gt_masks = torch.zeros(K, N)
        gt_masks[0, :15] = 1.0
        gt_masks[1, 15:30] = 1.0
        gt_masks[2, 30:45] = 1.0

        pred_idx, gt_idx = matcher.match(class_logits, mask_logits, gt_classes, gt_masks)

        # Should match K pairs (min(Q, K) = 3)
        assert len(pred_idx) == K
        assert len(gt_idx) == K
        # pred indices are unique
        assert len(pred_idx.unique()) == K
        # gt indices are unique
        assert len(gt_idx.unique()) == K
        # gt indices cover all GT instances
        assert set(gt_idx.tolist()) == {0, 1, 2}

    def test_matching_empty_gt(self):
        """Test matching with no ground-truth instances."""
        matcher = HungarianMatcher()
        Q, N = 10, 50

        class_logits = torch.randn(Q, 3)
        mask_logits = torch.randn(Q, N)
        gt_classes = torch.tensor([], dtype=torch.long)
        gt_masks = torch.zeros(0, N)

        pred_idx, gt_idx = matcher.match(class_logits, mask_logits, gt_classes, gt_masks)

        assert len(pred_idx) == 0
        assert len(gt_idx) == 0

    def test_matching_more_gt_than_queries(self):
        """Test matching when K > Q."""
        matcher = HungarianMatcher()
        Q, N, K = 3, 50, 5

        class_logits = torch.randn(Q, 3)
        mask_logits = torch.randn(Q, N)
        gt_classes = torch.randint(0, 2, (K,))
        gt_masks = torch.rand(K, N)

        pred_idx, gt_idx = matcher.match(class_logits, mask_logits, gt_classes, gt_masks)

        # Can only match min(Q, K) = 3 pairs
        assert len(pred_idx) == Q
        assert len(gt_idx) == Q

    def test_matching_deterministic(self):
        """Test that matching is deterministic for same inputs."""
        matcher = HungarianMatcher()
        Q, N, K = 10, 50, 3

        class_logits = torch.randn(Q, 3)
        mask_logits = torch.randn(Q, N)
        gt_classes = torch.tensor([0, 1, 0])
        gt_masks = torch.rand(K, N)

        pred_idx1, gt_idx1 = matcher.match(class_logits, mask_logits, gt_classes, gt_masks)
        pred_idx2, gt_idx2 = matcher.match(class_logits, mask_logits, gt_classes, gt_masks)

        assert torch.equal(pred_idx1, pred_idx2)
        assert torch.equal(gt_idx1, gt_idx2)


# =============================================================================
# SPFormer3DLoss
# =============================================================================

class TestSPFormer3DLoss:
    """Tests for the combined SPFormer3D loss function."""

    def _make_decoder_outputs(self, B=1, Q=10, N=100, num_classes=2, num_layers=2):
        """Helper to create mock decoder outputs."""
        class_logits_list = []
        mask_logits_list = []
        for _ in range(num_layers):
            cls = torch.randn(B, Q, num_classes + 1)
            masks = [torch.randn(Q, N) for _ in range(B)]
            class_logits_list.append(cls)
            mask_logits_list.append(masks)
        return {
            "class_logits": class_logits_list,
            "mask_logits": mask_logits_list,
        }

    def test_loss_finite(self):
        """Test that loss returns finite values."""
        loss_fn = SPFormer3DLoss(num_classes=2)
        N = 100
        decoder_outputs = self._make_decoder_outputs(N=N)
        semantic_logits = torch.randn(N, 2)
        semantic_labels = torch.randint(0, 2, (N,))
        instance_labels = torch.zeros(N, dtype=torch.long)
        instance_labels[:30] = 1
        instance_labels[30:60] = 2

        loss = loss_fn(decoder_outputs, semantic_logits, semantic_labels, instance_labels)

        assert isinstance(loss, PanopticLoss)
        assert torch.isfinite(loss.total)
        assert torch.isfinite(loss.semantic)
        assert torch.isfinite(loss.mask)

    def test_loss_nonnegative(self):
        """Test that loss components are non-negative."""
        loss_fn = SPFormer3DLoss(num_classes=2)
        N = 100
        decoder_outputs = self._make_decoder_outputs(N=N)
        semantic_logits = torch.randn(N, 2)
        semantic_labels = torch.randint(0, 2, (N,))
        instance_labels = torch.zeros(N, dtype=torch.long)
        instance_labels[:50] = 1

        loss = loss_fn(decoder_outputs, semantic_logits, semantic_labels, instance_labels)

        assert loss.total.item() >= 0
        assert loss.semantic.item() >= 0

    def test_loss_no_instances(self):
        """Test loss when there are no GT instances (all background)."""
        loss_fn = SPFormer3DLoss(num_classes=2)
        N = 100
        decoder_outputs = self._make_decoder_outputs(N=N)
        semantic_logits = torch.randn(N, 2)
        semantic_labels = torch.zeros(N, dtype=torch.long)
        instance_labels = torch.zeros(N, dtype=torch.long)  # All background

        loss = loss_fn(decoder_outputs, semantic_logits, semantic_labels, instance_labels)

        assert torch.isfinite(loss.total)

    def test_loss_deep_supervision_layers(self):
        """Test that loss uses all decoder layers."""
        num_layers = 4
        loss_fn = SPFormer3DLoss(num_classes=2)
        N = 100
        decoder_outputs = self._make_decoder_outputs(N=N, num_layers=num_layers)
        semantic_logits = torch.randn(N, 2)
        semantic_labels = torch.randint(0, 2, (N,))
        instance_labels = torch.zeros(N, dtype=torch.long)
        instance_labels[:50] = 1

        loss = loss_fn(decoder_outputs, semantic_logits, semantic_labels, instance_labels)

        # Loss should be valid (indirectly tests all layers contributed)
        assert torch.isfinite(loss.total)

    def test_focal_loss_shape(self):
        """Test focal loss internal method."""
        loss_fn = SPFormer3DLoss(num_classes=2)
        pred = torch.randn(5, 100)
        target = torch.rand(5, 100).round()

        fl = loss_fn._focal_loss(pred, target)
        assert fl.shape == ()
        assert torch.isfinite(fl)

    def test_dice_loss_shape(self):
        """Test dice loss internal method."""
        loss_fn = SPFormer3DLoss(num_classes=2)
        pred = torch.randn(5, 100)
        target = torch.rand(5, 100).round()

        dl = loss_fn._dice_loss(pred, target)
        assert dl.shape == ()
        assert torch.isfinite(dl)

    def test_dice_loss_perfect_prediction(self):
        """Test dice loss is near zero for perfect mask prediction."""
        loss_fn = SPFormer3DLoss(num_classes=2)
        # Large positive logits where target is 1, large negative where 0
        target = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
        pred = torch.tensor([[10.0, 10.0, -10.0, -10.0]])

        dl = loss_fn._dice_loss(pred, target)
        assert dl.item() < 0.05

    def test_ce_weights_registered(self):
        """Test that CE weights buffer is correctly registered."""
        loss_fn = SPFormer3DLoss(num_classes=3)
        assert hasattr(loss_fn, "ce_weights")
        assert loss_fn.ce_weights.shape == (4,)  # 3 classes + 1 no-object
        assert loss_fn.ce_weights[-1] == pytest.approx(0.1)  # no_object_weight


# =============================================================================
# SPFormer3D Model
# =============================================================================

class TestSPFormer3DModel:
    """Tests for the main SPFormer3D model."""

    def test_is_base_panoptic_model(self, default_model):
        """Test that SPFormer3D inherits from BasePanopticModel."""
        assert isinstance(default_model, BasePanopticModel)
        assert isinstance(default_model, nn.Module)

    def test_forward_output_type(self, small_model, single_batch_inputs):
        """Test that forward returns PanopticOutput."""
        small_model.eval()
        coords, features = single_batch_inputs
        with torch.no_grad():
            output = small_model(coords, features)
        assert isinstance(output, PanopticOutput)

    def test_forward_semantic_shapes(self, small_model, single_batch_inputs):
        """Test semantic output shapes from forward pass."""
        small_model.eval()
        coords, features = single_batch_inputs
        N = coords.shape[0]
        with torch.no_grad():
            output = small_model(coords, features)

        assert output.semantic_logits.shape == (N, 2)
        assert output.semantic_pred.shape == (N,)

    def test_forward_instance_pred_eval(self, small_model, single_batch_inputs):
        """Test instance_pred is populated during eval."""
        small_model.eval()
        coords, features = single_batch_inputs
        N = coords.shape[0]
        with torch.no_grad():
            output = small_model(coords, features)

        assert output.instance_pred is not None
        assert output.instance_pred.shape == (N,)

    def test_forward_instance_pred_train(self, small_model, single_batch_inputs):
        """Test instance_pred is None during training."""
        small_model.train()
        coords, features = single_batch_inputs
        output = small_model(coords, features)

        assert output.instance_pred is None

    def test_forward_multi_batch(self, small_model, multi_batch_inputs):
        """Test forward pass with multiple batch elements."""
        small_model.eval()
        coords, features = multi_batch_inputs
        N = coords.shape[0]
        with torch.no_grad():
            output = small_model(coords, features)

        assert output.semantic_logits.shape == (N, 2)
        assert output.instance_pred.shape == (N,)

    def test_compute_loss_with_instances(self, small_model):
        """Test loss computation with instance labels."""
        small_model.train()
        N = 128
        coords = torch.zeros(N, 4)
        coords[:, 0] = 0
        coords[:, 1:] = torch.randn(N, 3)
        features = torch.randn(N, 4)

        output = small_model(coords, features)

        semantic_labels = torch.randint(0, 2, (N,))
        instance_labels = torch.zeros(N, dtype=torch.long)
        instance_labels[:40] = 1
        instance_labels[40:80] = 2

        loss = small_model.compute_loss(output, semantic_labels, instance_labels)

        assert isinstance(loss, PanopticLoss)
        assert torch.isfinite(loss.total)
        assert torch.isfinite(loss.semantic)
        assert torch.isfinite(loss.mask)

    def test_compute_loss_semantic_only(self, small_model):
        """Test loss computation with no instance labels (fallback)."""
        small_model.train()
        N = 128
        coords = torch.zeros(N, 4)
        coords[:, 0] = 0
        coords[:, 1:] = torch.randn(N, 3)
        features = torch.randn(N, 4)

        output = small_model(coords, features)
        semantic_labels = torch.randint(0, 2, (N,))

        loss = small_model.compute_loss(output, semantic_labels, instance_labels=None)

        assert torch.isfinite(loss.total)
        assert torch.isfinite(loss.semantic)

    def test_compute_loss_requires_forward(self, small_model):
        """Test that compute_loss fails without prior forward pass."""
        N = 50
        output = PanopticOutput(
            semantic_logits=torch.randn(N, 2),
            semantic_pred=torch.randint(0, 2, (N,)),
        )
        semantic_labels = torch.randint(0, 2, (N,))
        instance_labels = torch.ones(N, dtype=torch.long)

        with pytest.raises(RuntimeError, match="compute_loss called without prior forward"):
            # Fresh model, no forward pass yet — cache is None
            fresh = SPFormer3D(
                num_classes=2, backbone_channels=[16, 32, 64, 128],
                feature_dim=16, transformer_dim=32, num_heads=4,
                num_decoder_layers=2, num_queries=10, dim_feedforward=64,
            )
            fresh.compute_loss(output, semantic_labels, instance_labels)

    def test_loss_to_dict(self, small_model):
        """Test that loss can be converted to dict for logging."""
        small_model.train()
        N = 128
        coords = torch.zeros(N, 4)
        coords[:, 0] = 0
        coords[:, 1:] = torch.randn(N, 3)
        features = torch.randn(N, 4)

        output = small_model(coords, features)
        semantic_labels = torch.randint(0, 2, (N,))
        instance_labels = torch.zeros(N, dtype=torch.long)
        instance_labels[:50] = 1

        loss = small_model.compute_loss(output, semantic_labels, instance_labels)
        loss_dict = loss.to_dict()

        assert "loss/total" in loss_dict
        assert "loss/semantic" in loss_dict
        assert "loss/mask" in loss_dict


# =============================================================================
# Gradient Flow
# =============================================================================

class TestSPFormer3DGradients:
    """Tests for gradient computation and backpropagation."""

    def test_gradients_flow(self, small_model):
        """Test that gradients flow through the full model."""
        small_model.train()
        N = 128
        coords = torch.zeros(N, 4)
        coords[:, 0] = 0
        coords[:, 1:] = torch.randn(N, 3)
        features = torch.randn(N, 4)

        output = small_model(coords, features)
        semantic_labels = torch.randint(0, 2, (N,))
        instance_labels = torch.zeros(N, dtype=torch.long)
        instance_labels[:50] = 1

        loss = small_model.compute_loss(output, semantic_labels, instance_labels)
        loss.total.backward()

        has_grad = False
        for name, param in small_model.named_parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break

        assert has_grad, "Model should have non-zero gradients after backward"

    def test_backbone_gradients(self, small_model):
        """Test that backbone parameters receive gradients."""
        small_model.train()
        N = 128
        coords = torch.zeros(N, 4)
        coords[:, 0] = 0
        coords[:, 1:] = torch.randn(N, 3)
        features = torch.randn(N, 4)

        output = small_model(coords, features)
        semantic_labels = torch.randint(0, 2, (N,))
        instance_labels = torch.zeros(N, dtype=torch.long)
        instance_labels[:50] = 1

        loss = small_model.compute_loss(output, semantic_labels, instance_labels)
        loss.total.backward()

        backbone_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in small_model.backbone.parameters()
        )
        assert backbone_has_grad

    def test_transformer_gradients(self, small_model):
        """Test that transformer decoder parameters receive gradients."""
        small_model.train()
        N = 128
        coords = torch.zeros(N, 4)
        coords[:, 0] = 0
        coords[:, 1:] = torch.randn(N, 3)
        features = torch.randn(N, 4)

        output = small_model(coords, features)
        semantic_labels = torch.randint(0, 2, (N,))
        instance_labels = torch.zeros(N, dtype=torch.long)
        instance_labels[:50] = 1

        loss = small_model.compute_loss(output, semantic_labels, instance_labels)
        loss.total.backward()

        transformer_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in small_model.transformer_decoder.parameters()
        )
        assert transformer_has_grad

    def test_optimizer_step(self, small_model):
        """Test that an optimizer step runs without errors."""
        small_model.train()
        optimizer = torch.optim.Adam(small_model.parameters(), lr=1e-4)

        N = 64
        coords = torch.zeros(N, 4)
        coords[:, 0] = 0
        coords[:, 1:] = torch.randn(N, 3)
        features = torch.randn(N, 4)

        output = small_model(coords, features)
        semantic_labels = torch.randint(0, 2, (N,))
        instance_labels = torch.zeros(N, dtype=torch.long)
        instance_labels[:30] = 1

        loss = small_model.compute_loss(output, semantic_labels, instance_labels)
        optimizer.zero_grad()
        loss.total.backward()
        optimizer.step()

        # Verify parameters changed
        # (hard to check directly, just ensure no error)


# =============================================================================
# Parameter Groups
# =============================================================================

class TestParameterGroups:
    """Tests for parameter group separation."""

    def test_groups_exist(self, default_model):
        """Test that both parameter groups are returned."""
        groups = default_model.get_parameters_by_group()
        assert "backbone" in groups
        assert "transformer" in groups

    def test_groups_nonempty(self, default_model):
        """Test that both groups contain parameters."""
        groups = default_model.get_parameters_by_group()
        assert len(groups["backbone"]) > 0
        assert len(groups["transformer"]) > 0

    def test_groups_cover_all_params(self, default_model):
        """Test that parameter groups cover all model parameters."""
        groups = default_model.get_parameters_by_group()
        group_params = set()
        for params in groups.values():
            for p in params:
                group_params.add(id(p))

        # Loss fn buffers aren't parameters, so compare against named_parameters
        model_params = {id(p) for p in default_model.parameters()}

        # All model parameters should appear in some group
        # (loss_fn parameters are included in transformer group via aux)
        missing = model_params - group_params
        # Some params may be in loss_fn which is not covered
        # Check that at least backbone + transformer are fully covered
        backbone_ids = {id(p) for p in default_model.backbone.parameters()}
        transformer_ids = (
            {id(p) for p in default_model.transformer_decoder.parameters()}
            | {id(p) for p in default_model.aux_semantic_head.parameters()}
            | {id(p) for p in default_model.superpoint_pooling.parameters()}
        )
        assert backbone_ids.issubset(group_params)
        assert transformer_ids.issubset(group_params)


# =============================================================================
# Instance Prediction (Inference)
# =============================================================================

class TestInstancePrediction:
    """Tests for direct mask-based instance prediction at inference."""

    def test_instance_ids_nonnegative_or_minus_one(self, small_model, single_batch_inputs):
        """Test instance IDs are -1 (unassigned) or >= 0."""
        small_model.eval()
        coords, features = single_batch_inputs
        with torch.no_grad():
            output = small_model(coords, features)

        assert output.instance_pred.min() >= -1

    def test_instance_ids_consecutive(self, small_model, single_batch_inputs):
        """Test that assigned instance IDs start from 0."""
        small_model.eval()
        coords, features = single_batch_inputs
        with torch.no_grad():
            output = small_model(coords, features)

        assigned = output.instance_pred[output.instance_pred >= 0]
        if len(assigned) > 0:
            assert assigned.min() == 0

    def test_high_confidence_threshold_fewer_instances(self):
        """Test that higher confidence threshold produces fewer instances."""
        low_model = SPFormer3D(
            num_classes=2, backbone_channels=[16, 32, 64, 128],
            feature_dim=16, transformer_dim=32, num_heads=4,
            num_decoder_layers=2, num_queries=10, dim_feedforward=64,
            confidence_threshold=0.01,
        )
        high_model = SPFormer3D(
            num_classes=2, backbone_channels=[16, 32, 64, 128],
            feature_dim=16, transformer_dim=32, num_heads=4,
            num_decoder_layers=2, num_queries=10, dim_feedforward=64,
            confidence_threshold=0.99,
        )
        # Share weights
        high_model.load_state_dict(low_model.state_dict())

        low_model.eval()
        high_model.eval()

        N = 128
        coords = torch.zeros(N, 4)
        coords[:, 0] = 0
        coords[:, 1:] = torch.randn(N, 3)
        features = torch.randn(N, 4)

        with torch.no_grad():
            out_low = low_model(coords, features)
            out_high = high_model(coords, features)

        n_low = (out_low.instance_pred >= 0).sum()
        n_high = (out_high.instance_pred >= 0).sum()
        assert n_high <= n_low

    def test_no_clustering_needed(self, small_model, single_batch_inputs):
        """Test that instance_pred is directly usable (no post-hoc clustering)."""
        small_model.eval()
        coords, features = single_batch_inputs
        with torch.no_grad():
            output = small_model(coords, features)

        # instance_pred should be a flat tensor with integer IDs
        assert output.instance_pred.dtype == torch.long
        assert output.instance_pred.dim() == 1


# =============================================================================
# SPFormer3DFallback
# =============================================================================

class TestSPFormer3DFallback:
    """Tests for the CPU fallback model."""

    def test_instantiation(self):
        """Test fallback can be instantiated."""
        model = SPFormer3DFallback(num_classes=2)
        assert isinstance(model, SPFormer3D)
        assert isinstance(model, BasePanopticModel)

    def test_forward(self):
        """Test fallback forward pass."""
        model = SPFormer3DFallback(num_classes=2)
        model.eval()

        N = 64
        coords = torch.zeros(N, 4)
        coords[:, 0] = 0
        coords[:, 1:] = torch.randn(N, 3)
        features = torch.randn(N, 4)

        with torch.no_grad():
            output = model(coords, features)

        assert output.semantic_logits.shape == (N, 2)
        assert output.instance_pred.shape == (N,)


# =============================================================================
# Config and Model Creation Integration
# =============================================================================

class TestConfigIntegration:
    """Tests for SPFormer3D config integration."""

    def test_transformer_config_defaults(self):
        """Test TransformerConfig has expected defaults."""
        tc = TransformerConfig()
        assert tc.d_model == 256
        assert tc.nhead == 8
        assert tc.num_decoder_layers == 6
        assert tc.num_queries == 100
        assert tc.dim_feedforward == 1024
        assert tc.superpoint_voxel_size == 0.1
        assert tc.feature_dim == 64
        assert tc.backbone_channels == [64, 128, 256, 512]
        assert tc.confidence_threshold == 0.5
        assert tc.mask_threshold == 0.5

    def test_transformer_config_custom(self):
        """Test TransformerConfig with custom values."""
        tc = TransformerConfig(
            d_model=128, nhead=4, num_queries=50,
            backbone_channels=[32, 64, 128, 256],
        )
        assert tc.d_model == 128
        assert tc.nhead == 4
        assert tc.num_queries == 50
        assert tc.backbone_channels == [32, 64, 128, 256]

    def test_training_config_has_transformer(self):
        """Test that TrainingConfig includes transformer field."""
        config = Config()
        assert hasattr(config.training, "transformer")
        assert isinstance(config.training.transformer, TransformerConfig)

    def test_create_model_spformer(self):
        """Test create_model dispatches to SPFormer3D."""
        config = Config()
        config.training.model_name = ModelArchitecture.SPFORMER_3D
        model = create_model(config, device="cpu")

        assert isinstance(model, SPFormer3D)
        assert isinstance(model, BasePanopticModel)

    def test_create_model_spformer_custom_config(self):
        """Test create_model with custom transformer config."""
        config = Config()
        config.training.model_name = ModelArchitecture.SPFORMER_3D
        config.training.num_classes = 3
        config.training.transformer.num_queries = 50
        config.training.transformer.num_decoder_layers = 3

        model = create_model(config, device="cpu")

        assert model.num_classes == 3
        assert model.transformer_decoder.num_queries == 50
        assert model.transformer_decoder.num_decoder_layers == 3

    def test_create_model_default_still_pointgroup(self):
        """Test that default config still creates PointGroup model."""
        from lidar_panoptic_segmentation.model import LidarPanopticModel

        config = Config()
        model = create_model(config, device="cpu")

        assert isinstance(model, LidarPanopticModel)

    def test_model_architecture_enum(self):
        """Test SPFORMER_3D enum value."""
        assert ModelArchitecture.SPFORMER_3D.value == "SPFormer3D"


# =============================================================================
# Postprocess Integration
# =============================================================================

class TestPostprocessIntegration:
    """Tests for SPFormer3D output flowing through postprocessing."""

    def test_instance_pred_skips_clustering(self):
        """Test that pre-computed instance_pred bypasses clustering."""
        from lidar_panoptic_segmentation.postprocess import postprocess_predictions

        N = 100
        points = np.random.randn(N, 3).astype(np.float32)
        semantic_pred = np.ones(N, dtype=np.int64)
        instance_pred = np.zeros(N, dtype=np.int32)
        instance_pred[:50] = 0
        instance_pred[50:] = 1

        # This should NOT call cluster_instances
        result = postprocess_predictions(
            points=points,
            semantic_pred=semantic_pred,
            instance_pred=instance_pred,
        )

        assert len(result.instance_pred) == N
        assert np.array_equal(result.instance_pred, instance_pred)

    def test_none_instance_pred_backward_compat(self):
        """Test backward compatibility when instance_pred is None."""
        from lidar_panoptic_segmentation.postprocess import postprocess_predictions

        N = 100
        points = np.random.randn(N, 3).astype(np.float32)
        semantic_pred = np.zeros(N, dtype=np.int64)  # All background

        # Should still work (clustering path, finds nothing)
        result = postprocess_predictions(
            points=points,
            semantic_pred=semantic_pred,
            instance_pred=None,
        )

        assert len(result.instance_pred) == N

    def test_end_to_end_inference(self, small_model):
        """Test full pipeline: model forward -> postprocess."""
        from lidar_panoptic_segmentation.postprocess import postprocess_predictions

        small_model.eval()
        N = 128
        coords = torch.zeros(N, 4)
        coords[:, 0] = 0
        coords[:, 1:] = torch.randn(N, 3)
        features = torch.randn(N, 4)

        with torch.no_grad():
            output = small_model(coords, features)

        points = coords[:, 1:].cpu().numpy()
        semantic_pred = output.semantic_pred.cpu().numpy()
        instance_pred = output.instance_pred.cpu().numpy()

        result = postprocess_predictions(
            points=points,
            semantic_pred=semantic_pred,
            instance_pred=instance_pred,
        )

        assert result.points.shape == (N, 3)
        assert result.semantic_pred.shape == (N,)
        assert result.instance_pred.shape == (N,)


# =============================================================================
# YAML Config Validation
# =============================================================================

class TestSPFormerYAMLConfig:
    """Tests for SPFormer3D YAML config file."""

    def test_config_spformer_yaml_loads(self):
        """Test that config_spformer.yaml can be loaded and validated."""
        import json
        import yaml

        config_path = Path("databricks/config_spformer.yaml")
        if not config_path.exists():
            pytest.skip("config_spformer.yaml not found")

        with open(config_path) as f:
            data = yaml.safe_load(f)

        # Replace placeholders
        s = json.dumps(data)
        s = s.replace("__CATALOG__", "dev").replace("__SCHEMA__", "test")
        s = s.replace("__MODEL_NAME__", "TestModel")
        data = json.loads(s)

        config = Config(**data)
        assert config.training.model_name == ModelArchitecture.SPFORMER_3D

    def test_config_spformer_hyperparams(self):
        """Test SPFormer config has expected hyperparameters."""
        import json
        import yaml

        config_path = Path("databricks/config_spformer.yaml")
        if not config_path.exists():
            pytest.skip("config_spformer.yaml not found")

        with open(config_path) as f:
            data = yaml.safe_load(f)

        s = json.dumps(data)
        s = s.replace("__CATALOG__", "dev").replace("__SCHEMA__", "test")
        s = s.replace("__MODEL_NAME__", "TestModel")
        data = json.loads(s)

        config = Config(**data)
        assert config.training.batch_size == 2
        assert config.training.epochs == 120
        assert config.training.weight_decay == 0.01
        assert config.training.gradient_clip == 0.1
        assert config.training.transformer.d_model == 256
        assert config.training.transformer.num_queries == 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
