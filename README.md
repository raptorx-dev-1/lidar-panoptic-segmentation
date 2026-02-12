# LiDAR Panoptic Segmentation

> A modular, configuration-driven, enterprise-ready deep learning system for 3D panoptic segmentation of airborne LiDAR point clouds and automatic tree crown polygon extraction.

Designed for production deployment on **Azure Databricks 15.4 LTS GPU** (single-node) with Unity Catalog, MLflow experiment tracking, and secure MinkowskiEngine installation. Supports two model architectures: the proven **PointGroup-3heads** clustering approach and the state-of-the-art **SPFormer3D** mask transformer.

---

## Table of Contents

- [Overview](#overview)
- [Model Architectures](#model-architectures)
  - [PointGroup-3heads](#1-pointgroup-3heads-clustering-based)
  - [SPFormer3D](#2-spformer3d-mask-transformer)
  - [Architecture Comparison](#architecture-comparison)
- [Quick Start](#quick-start)
  - [VS Code Devcontainer (Recommended)](#vs-code-devcontainer-recommended)
  - [Local Development (Conda)](#local-development-conda)
- [Databricks Deployment Guide](#databricks-deployment-guide)
  - [Prerequisites](#prerequisites)
  - [Unity Catalog Data Layout](#unity-catalog-data-layout)
  - [Bundle Configuration](#bundle-configuration)
  - [Deploying Jobs](#deploying-jobs)
  - [Running Training](#running-training)
  - [Running Inference](#running-inference)
  - [Running Validation](#running-validation)
- [Configuration Reference](#configuration-reference)
  - [PointGroup-3heads Config](#pointgroup-3heads-config)
  - [SPFormer3D Config](#spformer3d-config)
  - [Configuration Placeholders](#configuration-placeholders)
- [Data Formats](#data-formats)
- [Training Pipeline](#training-pipeline)
  - [Training Loop Features](#training-loop-features)
  - [Learning Rate Schedulers](#learning-rate-schedulers)
  - [Data Augmentation](#data-augmentation)
- [Inference Pipeline](#inference-pipeline)
- [Evaluation Metrics](#evaluation-metrics)
  - [Semantic Segmentation Metrics](#semantic-segmentation-metrics)
  - [Instance Segmentation Metrics](#instance-segmentation-metrics)
  - [Panoptic Quality](#panoptic-quality)
  - [Detection Metrics](#detection-metrics)
  - [Forestry Metrics](#forestry-metrics)
  - [Validation Results Visualization](#validation-results-visualization)
- [MLflow Integration](#mlflow-integration)
- [Project Structure](#project-structure)
- [Testing](#testing)
- [API Reference](#api-reference)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)
- [License](#license)

---

## Overview

This system takes raw airborne LiDAR point clouds (LAS/LAZ/PLY files) and produces:

1. **Semantic segmentation** — per-point classification (tree vs. non-tree)
2. **Instance segmentation** — per-point instance IDs (which tree each point belongs to)
3. **Panoptic segmentation** — the combination of both
4. **Tree crown polygons** — 2D GeoJSON/Shapefile polygons of individual tree canopies

The pipeline handles everything from data loading through training, inference, postprocessing (polygon extraction), and validation against field-measured ground truth polygons.

```
                    ┌─────────────────────┐
                    │  LAS/LAZ/PLY Input  │
                    └─────────┬───────────┘
                              │
                    ┌─────────▼───────────┐
                    │   Voxelization &     │
                    │   Feature Extraction │
                    └─────────┬───────────┘
                              │
              ┌───────────────┼───────────────┐
              │                               │
    ┌─────────▼─────────┐          ┌─────────▼──────────┐
    │  PointGroup-3heads │          │    SPFormer3D       │
    │                    │          │                     │
    │  MinkowskiUNet     │          │  Residual UNet      │
    │  ─────────────     │          │  ──────────────     │
    │  Semantic head     │          │  Superpoint Pooling │
    │  Offset head       │          │  Transformer Decoder│
    │  Embedding head    │          │  Mask Prediction    │
    │                    │          │                     │
    │  HDBSCAN Clustering│          │  Hungarian Matching │
    └─────────┬──────────┘          └─────────┬──────────┘
              │                               │
              └───────────────┬───────────────┘
                              │
                    ┌─────────▼───────────┐
                    │   Postprocessing     │
                    │   Polygon Extraction │
                    └─────────┬───────────┘
                              │
                    ┌─────────▼───────────┐
                    │  GeoJSON / Shapefile │
                    │  / Parquet Output    │
                    └─────────────────────┘
```

---

## Model Architectures

### 1. PointGroup-3heads (Clustering-Based)

The original architecture, adapted from SegmentAnyTree. Uses per-point predictions followed by post-hoc clustering to group points into tree instances.

```
Input: coords (N,4), features (N,4)
    │
    ▼
┌───────────────────────────────────┐
│      MinkowskiEngine UNet         │
│  Sparse Conv Backbone             │
│  Channels: [32, 64, 128, 256]    │
│  Encoder → Bottleneck → Decoder   │
└─────────────┬─────────────────────┘
              │
              ├──► features (N, D)
              │
    ┌─────────┼─────────────┐
    │         │             │
    ▼         ▼             ▼
┌────────┐ ┌────────┐ ┌──────────┐
│Semantic│ │ Offset │ │Embedding │
│  Head  │ │  Head  │ │   Head   │
│(N, 2)  │ │(N, 3)  │ │(N, 5)   │
└───┬────┘ └───┬────┘ └────┬─────┘
    │          │            │
    │          └────┬───────┘
    │               │
    │         ┌─────▼──────┐
    │         │  HDBSCAN   │
    │         │ Clustering │
    │         └─────┬──────┘
    │               │
    ▼               ▼
semantic_pred   instance_pred
  (N,)            (N,)
```

**How it works:**

| Component | Description |
|-----------|-------------|
| **Backbone** | MinkowskiEngine sparse convolutional UNet with channels [32, 64, 128, 256]. Processes 3D voxelized point clouds efficiently using sparse tensor operations. |
| **Semantic Head** | Linear layer mapping backbone features to 2 classes (non-tree, tree). Trained with cross-entropy loss. |
| **Offset Head** | Linear layer predicting 3D offset vectors pointing toward each tree's center. Trained with L1 loss on tree points. |
| **Embedding Head** | Linear layer producing 5D embeddings for discriminative clustering. Trained with discriminative loss (pull same-instance together, push different-instance apart). |
| **HDBSCAN Clustering** | At inference, shifted points (pos + offset) are clustered using HDBSCAN on the embedding space to produce instance IDs. Requires tuning `min_cluster_size` and `min_samples`. |

**Strengths:** Simple, well-understood, fast inference for small scenes.

**Limitations:** Instance quality depends on clustering hyperparameters; no global context beyond the convolutional receptive field; touching/overlapping tree crowns confuse the embedding space.

### 2. SPFormer3D (Mask Transformer)

A superpoint-based mask transformer that directly predicts instance masks end-to-end via learnable object queries and cross-attention, eliminating clustering entirely. Inspired by SPFormer (AAAI 2023), OneFormer3D (CVPR 2024), and Mask3D.

```
Input: coords (N,4), features (N,4)
    │
    ▼
┌──────────────────────────────────────┐
│   Residual Sparse Conv UNet          │
│   Deeper: [64, 128, 256, 512]       │
│   2 Residual Blocks per stage        │
│   Encoder → Bottleneck → Decoder     │
└─────────────┬────────────────────────┘
              │
              ├──► per-point features (N, D=64)
              │
              ▼
┌──────────────────────────────────────┐
│        Superpoint Pooling            │
│                                      │
│  Coarser voxel grouping (0.1m)      │
│  Mean pool features per superpoint   │
│  Learned 3D positional encoding      │
│  MLP(3 → D) on superpoint centers   │
│                                      │
│  Output: (S, D) where S ≈ N/16..N/64│
└─────────────┬────────────────────────┘
              │
              ▼
┌──────────────────────────────────────┐
│     Transformer Decoder × 6         │
│                                      │
│  100 Learnable Instance Queries      │
│  ┌────────────────────────────┐     │
│  │ Self-Attention on Queries  │     │
│  │ Cross-Attention → SP Feats │     │
│  │ FFN (D → 4D → D)          │     │
│  │ Masked Attention           │     │
│  └────────────────────────────┘     │
│                                      │
│  Deep Supervision at Each Layer      │
└─────────────┬────────────────────────┘
              │
    ┌─────────┼──────────────┐
    │         │              │
    ▼         ▼              ▼
┌────────┐ ┌──────────┐ ┌──────────────┐
│ Class  │ │  Mask    │ │  Auxiliary    │
│ Head   │ │  Head    │ │ Semantic Head│
│(Q,C+1) │ │  (Q,N)   │ │   (N, 2)     │
└───┬────┘ └────┬─────┘ └──────┬───────┘
    │           │              │
    │     ┌─────▼──────┐       │
    │     │ Confidence │       │
    │     │ Filter +   │       │
    │     │ Threshold  │       │
    │     └─────┬──────┘       │
    │           │              │
    ▼           ▼              ▼
class_pred  instance_pred  semantic_pred
 (Q,)        (N,)           (N,)
```

**How it works:**

| Component | Description |
|-----------|-------------|
| **Backbone** | Deeper residual sparse convolutional UNet with channels [64, 128, 256, 512]. Each stage has 2 residual blocks (conv-BN-ReLU-conv-BN + skip) for richer feature extraction. |
| **Superpoint Pooling** | Quantizes points at a coarser voxel size (0.1m vs 0.02m) to create "superpoints". Mean-pools backbone features per superpoint and adds learned 3D positional encoding. Reduces sequence length for efficient transformer processing. |
| **Transformer Decoder** | 6 layers of masked cross-attention. 100 learnable instance queries attend to superpoint features. Each layer predicts intermediate class and mask logits (deep supervision). Masked attention uses previous layer's predictions to focus attention. |
| **Class Head** | Per-query MLP predicting (num_classes + 1) probabilities, where +1 is "no object". |
| **Mask Head** | Per-query MLP producing mask embeddings, then dot-product with per-point features to get (Q, N) binary mask logits. |
| **Hungarian Matching** | During training, optimal bipartite matching between predicted and ground-truth instances using combined class + mask + dice cost. Implemented with `scipy.optimize.linear_sum_assignment`. |
| **Loss** | Focal loss (mask) + Dice loss (mask) + Cross-entropy (class) + Auxiliary semantic CE. Deep supervision at all 6 decoder layers. |
| **Inference** | No clustering needed! Softmax class scores → confidence filter → sigmoid mask threshold → greedy assignment of highest-confidence queries to unassigned points. |

**Strengths:** Global context via cross-attention resolves overlapping crowns; end-to-end mask prediction with no clustering hyperparameters; Hungarian matching ensures optimal training signal; state-of-the-art on 3D panoptic benchmarks.

**Limitations:** Higher GPU memory requirements; needs more training epochs to converge; requires A100 GPU for training.

### Architecture Comparison

| Feature | PointGroup-3heads | SPFormer3D |
|---------|:-----------------:|:----------:|
| **Backbone** | MinkUNet [32,64,128,256] | Residual UNet [64,128,256,512] |
| **Instance Method** | HDBSCAN clustering | Direct mask prediction |
| **Global Context** | Local conv receptive field | Transformer cross-attention |
| **Training Loss** | CE + L1 + Discriminative | Focal + Dice + CE + Hungarian |
| **Inference** | Clustering (hyperparameter-dependent) | Threshold + greedy (hyperparameter-free) |
| **Overlapping Crowns** | Struggles with ambiguity | Resolves via masked attention |
| **Batch Size** | 4 (default) | 2 (higher memory) |
| **Training Epochs** | 60 | 120 |
| **GPU Requirement** | T4 / A10 sufficient | A100 recommended |
| **Weight Decay** | 0.0001 | 0.01 |
| **Gradient Clip** | 1.0 | 0.1 |
| **Config File** | `databricks/config.yaml` | `databricks/config_spformer.yaml` |

---

## Quick Start

### VS Code Devcontainer (Recommended)

The devcontainer matches Databricks 15.4 LTS GPU runtime (CUDA 12.1, Python 3.10, PyTorch 2.1.0):

1. **Install Prerequisites:**
   - [VS Code](https://code.visualstudio.com/)
   - [Docker](https://www.docker.com/products/docker-desktop/)
   - [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
   - [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)

2. **Open in Container:**
   - Open this folder in VS Code
   - Click "Reopen in Container" when prompted (or `Ctrl+Shift+P` → `Dev Containers: Reopen in Container`)
   - Wait for the container to build (~10 minutes first time)

3. **Verify Installation:**
   ```bash
   # Run all tests (111 tests)
   pytest tests/ -v

   # Quick smoke test for SPFormer3D
   python -c "from lidar_panoptic_segmentation.mask_transformer import SPFormer3D; print('OK')"
   ```

### Local Development (Conda)

```bash
# Create conda environment
conda env create -f environment.yml
conda activate lidar-panoptic-segmentation

# Install the package in editable mode
pip install -e .

# Run tests
pytest tests/ -v

# Train PointGroup-3heads
python -m lidar_panoptic_segmentation.train --config config.yaml

# Train SPFormer3D
python -m lidar_panoptic_segmentation.train --config config_spformer.yaml
```

---

## Databricks Deployment Guide

### Prerequisites

Before deploying, ensure you have:

1. **Databricks CLI v0.200+** installed and authenticated:
   ```bash
   # Install
   curl -fsSL https://raw.githubusercontent.com/databricks/setup-cli/main/install.sh | sh

   # Authenticate (configure your workspace URL and token)
   databricks configure
   ```

2. **Unity Catalog** access with:
   - A catalog (e.g., `dev_lidar` for development, `prod_lidar` for production)
   - A schema (e.g., `panoptic_segmentation`)
   - Permissions to create Volumes and register models

3. **GPU Compute** quota for:
   - `Standard_NC24ads_A100_v4` (training/inference for SPFormer3D)
   - `Standard_NC4as_T4_v3` (validation, PointGroup training)

### Unity Catalog Data Layout

The system expects data organized in Unity Catalog Volumes:

```
/Volumes/{catalog}/{schema}/
├── raw_data/
│   ├── train/
│   │   ├── las/           # Training LAS/LAZ point clouds
│   │   │   ├── plot_001.las
│   │   │   ├── plot_002.las
│   │   │   └── ...
│   │   └── labels/        # Corresponding label files (PLY with semantic + instance)
│   │       ├── plot_001.ply
│   │       ├── plot_002.ply
│   │       └── ...
│   └── val/
│       ├── las/           # Validation point clouds
│       └── labels/        # Validation labels
├── vizionair_lidar/
│   ├── input/             # Inference input LAS files
│   ├── output_polygons/   # PointGroup-3heads output GeoJSON
│   └── output_polygons_spformer/  # SPFormer3D output GeoJSON
├── field_validation/
│   └── polygons/          # Ground truth GeoJSON polygons for validation
│       ├── plot_001.geojson
│       └── ...
├── models/                # Saved model checkpoints
└── logs/                  # Training logs
```

**Upload your data:**
```bash
# Upload training data to Unity Catalog Volume
databricks fs cp -r ./data/train /Volumes/dev_lidar/panoptic_segmentation/raw_data/train

# Upload validation data
databricks fs cp -r ./data/val /Volumes/dev_lidar/panoptic_segmentation/raw_data/val

# Upload inference input
databricks fs cp -r ./data/inference /Volumes/dev_lidar/panoptic_segmentation/vizionair_lidar/input

# Upload ground truth for validation
databricks fs cp -r ./data/gt_polygons /Volumes/dev_lidar/panoptic_segmentation/field_validation/polygons
```

### Bundle Configuration

The project uses [Databricks Asset Bundles](https://docs.databricks.com/en/dev-tools/bundles/index.html) for deployment. The bundle is defined in `databricks/databricks.yml`:

```yaml
bundle:
  name: lidar-panoptic-segmentation

variables:
  catalog:
    description: Unity Catalog name
  schema:
    description: UC schema name
  model_name:
    description: Registered model name in UC
    default: LidarPanopticSegmentation

artifacts:
  lidar_wheel:
    type: whl
    path: ..
    build: pip wheel --no-deps -w dist .

include:
  - resources/*.yml          # Auto-discovers all job definitions

targets:
  dev:
    mode: development
    default: true
    variables:
      catalog: dev_lidar
      schema: panoptic_segmentation

  prod:
    mode: production
    variables:
      catalog: prod_lidar
      schema: panoptic_segmentation
    run_as:
      service_principal_name: lidar-panoptic-sp
```

**Key points:**
- The `artifacts` section builds a wheel from the project root, packaging `lidar_panoptic_segmentation` as a pip-installable library
- `include: resources/*.yml` auto-discovers all job definitions — no need to register new jobs manually
- `targets` define environment-specific variables (`dev` vs `prod`)
- The wheel is installed on each job cluster via the `libraries` section

### Deploying Jobs

```bash
cd databricks

# Validate the bundle configuration
databricks bundle validate -t dev

# Deploy all jobs to dev
databricks bundle deploy -t dev

# Deploy to production
databricks bundle deploy -t prod
```

After deploying, you will see these jobs in your Databricks workspace:

| Job Name | Config | GPU | Purpose |
|----------|--------|-----|---------|
| `[dev] LiDAR Panoptic - Training` | `config.yaml` | A100 | PointGroup-3heads training |
| `[dev] LiDAR Panoptic - Inference` | `config.yaml` | A100 | PointGroup-3heads inference |
| `[dev] LiDAR Panoptic - Validation` | `config.yaml` | T4 | PointGroup polygon validation |
| `[dev] LiDAR Panoptic - SPFormer3D Training` | `config_spformer.yaml` | A100 | SPFormer3D training |
| `[dev] LiDAR Panoptic - SPFormer3D Inference` | `config_spformer.yaml` | A100 | SPFormer3D inference |
| `[dev] LiDAR Panoptic - SPFormer3D Validation` | `config_spformer.yaml` | T4 | SPFormer3D polygon validation |

### Running Training

**PointGroup-3heads:**
```bash
# Run training job (dev target)
databricks bundle run training_job -t dev

# With overrides
databricks bundle run training_job -t dev \
  --params catalog=dev_lidar,schema=panoptic_segmentation
```

**SPFormer3D:**
```bash
# Run SPFormer3D training job
databricks bundle run training_spformer_job -t dev
```

**What happens during training:**

1. The cluster spins up with the specified GPU node type
2. The `lidar_panoptic_segmentation` wheel is installed
3. `run_training.py` executes:
   - Parses `--catalog`, `--schema`, `--model-name`, `--config` arguments
   - Loads the YAML config and resolves `__CATALOG__`/`__SCHEMA__`/`__MODEL_NAME__` placeholders
   - Creates the model, optimizer, scheduler, and data loaders
   - Trains for the configured number of epochs with:
     - Mixed precision (FP16) training
     - Gradient clipping
     - Learning rate scheduling (cosine annealing by default)
     - Checkpoint saving (best + latest)
     - MLflow metric logging (loss, learning rate, epoch timing)
   - Registers the best model in Unity Catalog via MLflow

**Monitor training:**
- Open the Databricks workspace → Jobs → click the running job
- View Spark driver logs for training progress
- Open MLflow Experiments UI to see real-time metrics

### Running Inference

**PointGroup-3heads:**
```bash
databricks bundle run inference_job -t dev
```

**SPFormer3D:**
```bash
databricks bundle run inference_spformer_job -t dev
```

**What happens during inference:**

1. `run_inference.py` loads the model from Unity Catalog Model Registry:
   - PointGroup: `models:/LidarPanopticSegmentation/Production`
   - SPFormer3D: `models:/LidarPanopticSegmentation/Production` (same registry, different model artifact)
2. Processes all LAS/LAZ files in the `input` directory
3. For each file:
   - Reads the point cloud
   - Applies voxelization and feature extraction
   - Runs the model forward pass
   - PointGroup: runs HDBSCAN clustering on embeddings + offsets
   - SPFormer3D: directly uses predicted instance masks (no clustering)
   - Extracts tree polygons from instance segments via alpha-shape / convex hull
   - Saves GeoJSON polygons to the `output` directory
4. Output structure:
   ```
   output_polygons_spformer/
   ├── plot_001.geojson      # Tree crown polygons
   ├── plot_002.geojson
   └── ...
   ```

### Running Validation

Validation compares predicted polygons against field-measured ground truth:

**PointGroup-3heads:**
```bash
databricks bundle run validation_job -t dev
```

**SPFormer3D:**
```bash
databricks bundle run validation_spformer_job -t dev
```

**What happens during validation:**

1. `run_validation.py` finds matching prediction/ground-truth GeoJSON pairs by filename
2. For each matched pair, computes polygon-level IoU and matches using greedy assignment
3. Calculates per-file and aggregate metrics:
   - **Precision**: fraction of predicted polygons that match a ground truth (IoU >= 0.5)
   - **Recall**: fraction of ground truth polygons detected
   - **F1 Score**: harmonic mean of precision and recall
4. Saves `validation_metrics.json` to the output directory
5. Logs aggregate metrics to MLflow:
   - `val/polygon_precision`
   - `val/polygon_recall`
   - `val/polygon_f1`

---

## Configuration Reference

### PointGroup-3heads Config

File: `databricks/config.yaml`

```yaml
env:
  name: databricks
  debug: false
  seed: 42
  num_workers: 0

paths:
  data_root: "/Volumes/__CATALOG__/__SCHEMA__/raw_data"
  train:
    pointclouds: "/Volumes/__CATALOG__/__SCHEMA__/raw_data/train/las"
    labels: "/Volumes/__CATALOG__/__SCHEMA__/raw_data/train/labels"
  val:
    pointclouds: "/Volumes/__CATALOG__/__SCHEMA__/raw_data/val/las"
    labels: "/Volumes/__CATALOG__/__SCHEMA__/raw_data/val/labels"
  inference:
    input: "/Volumes/__CATALOG__/__SCHEMA__/vizionair_lidar/input"
    output: "/Volumes/__CATALOG__/__SCHEMA__/vizionair_lidar/output_polygons"
  models: "/Volumes/__CATALOG__/__SCHEMA__/models"
  logs: "/Volumes/__CATALOG__/__SCHEMA__/logs"
  cache: "/local_disk0/cache"

training:
  batch_size: 4
  mixed_precision: true

inference:
  model_uri: "models:/__MODEL_NAME__/Production"

logging:
  level: INFO
  mlflow:
    enabled: true
    tracking_uri: databricks
    experiment_name: LidarPanopticSegmentation_Experiments
    registry_uri: databricks-uc
```

### SPFormer3D Config

File: `databricks/config_spformer.yaml`

```yaml
env:
  name: databricks
  debug: false
  seed: 42
  num_workers: 0

paths:
  data_root: "/Volumes/__CATALOG__/__SCHEMA__/raw_data"
  train:
    pointclouds: "/Volumes/__CATALOG__/__SCHEMA__/raw_data/train/las"
    labels: "/Volumes/__CATALOG__/__SCHEMA__/raw_data/train/labels"
  val:
    pointclouds: "/Volumes/__CATALOG__/__SCHEMA__/raw_data/val/las"
    labels: "/Volumes/__CATALOG__/__SCHEMA__/raw_data/val/labels"
  inference:
    input: "/Volumes/__CATALOG__/__SCHEMA__/vizionair_lidar/input"
    output: "/Volumes/__CATALOG__/__SCHEMA__/vizionair_lidar/output_polygons_spformer"
  models: "/Volumes/__CATALOG__/__SCHEMA__/models"
  logs: "/Volumes/__CATALOG__/__SCHEMA__/logs"
  cache: "/local_disk0/cache"

training:
  model_name: SPFormer3D       # Selects mask transformer architecture
  batch_size: 2                # Smaller batch for higher memory usage
  epochs: 120                  # Transformers need more epochs
  lr: 0.0001
  weight_decay: 0.01           # Higher regularization for transformers
  gradient_clip: 0.1           # Tighter clipping for training stability
  mixed_precision: true
  scheduler: cosine
  augmentations:
    rotate: true
    rotate_range: 180.0
    jitter: true
    jitter_std: 0.01
    scale: true
    scale_range: [0.9, 1.1]
    flip: true
    elastic_distortion: true   # Stronger augmentation for transformers
    color_jitter: false
    dropout: 0.0
  transformer:                 # SPFormer3D-specific hyperparameters
    d_model: 256               # Transformer hidden dimension
    nhead: 8                   # Number of attention heads
    num_decoder_layers: 6      # Transformer decoder depth
    num_queries: 100           # Learnable instance queries
    dim_feedforward: 1024      # FFN intermediate dimension
    superpoint_voxel_size: 0.1 # Superpoint grouping resolution (meters)
    feature_dim: 64            # Backbone output feature dimension
    backbone_channels: [64, 128, 256, 512]  # UNet channel progression
    confidence_threshold: 0.5  # Minimum query confidence at inference
    mask_threshold: 0.5        # Binary mask threshold at inference
    weight_ce: 2.0             # Cross-entropy loss weight
    weight_mask: 5.0           # Focal (mask) loss weight
    weight_dice: 5.0           # Dice loss weight
    weight_aux_semantic: 1.0   # Auxiliary semantic loss weight

inference:
  model_uri: "models:/__MODEL_NAME__/Production"

logging:
  level: INFO
  mlflow:
    enabled: true
    tracking_uri: databricks
    experiment_name: LidarPanopticSegmentation_SPFormer3D  # Separate experiment
    registry_uri: databricks-uc
```

### Configuration Placeholders

The Databricks configs use placeholder tokens that are resolved at runtime:

| Placeholder | Resolved From | Example |
|-------------|---------------|---------|
| `__CATALOG__` | `--catalog` CLI arg | `dev_lidar` |
| `__SCHEMA__` | `--schema` CLI arg | `panoptic_segmentation` |
| `__MODEL_NAME__` | `--model-name` CLI arg | `LidarPanopticSegmentation` |

These are resolved by `databricks/src/common.py:_resolve_placeholders()` before the config is parsed by Pydantic.

---

## Data Formats

### Input

| Format | Extension | Description |
|--------|-----------|-------------|
| LAS | `.las` | Standard LiDAR format (ASPRS) |
| LAZ | `.laz` | Compressed LAS |
| PLY | `.ply` | Stanford polygon format with labels |

**Required point attributes:**
- XYZ coordinates (mandatory)
- Intensity (optional, used as feature)
- RGB colors (optional, used as features)

**Label files** (for training/validation):
- PLY format with `semantic_label` and `instance_label` scalar fields
- `semantic_label`: 0 = non-tree, 1 = tree
- `instance_label`: unique integer per tree instance (0 or -1 = background)

### Output

| Format | Description | Use Case |
|--------|-------------|----------|
| GeoJSON | RFC 7946 compliant | Web mapping, GIS tools |
| Shapefile | ESRI format | Desktop GIS (ArcGIS, QGIS) |
| Parquet | Columnar format | Big data analytics (Spark, DuckDB) |

Each polygon feature includes properties:
```json
{
  "type": "Feature",
  "geometry": { "type": "Polygon", "coordinates": [...] },
  "properties": {
    "instance_id": 42,
    "height": 18.3,
    "crown_area": 25.7,
    "num_points": 1847,
    "confidence": 0.92
  }
}
```

---

## Training Pipeline

### Training Loop Features

The `Trainer` class in `train.py` provides:

| Feature | Description |
|---------|-------------|
| **Mixed Precision** | FP16 training via PyTorch `GradScaler` + `autocast` for 2x memory savings |
| **Gradient Clipping** | Configurable max gradient norm (default: 1.0 for PointGroup, 0.1 for SPFormer3D) |
| **Checkpointing** | Saves `checkpoint_best.pt` and `checkpoint_latest.pt` to the models directory |
| **Resume** | Resume from any checkpoint with `--resume path/to/checkpoint.pt` |
| **Validation** | Automatic validation after each epoch if validation data is configured |
| **Early Stopping** | Monitors validation loss; stops if no improvement after patience epochs |
| **MLflow Logging** | Logs all metrics, hyperparameters, and artifacts to MLflow |
| **Differential LR** | SPFormer3D uses 0.1x learning rate for the backbone vs 1x for the transformer |

### Learning Rate Schedulers

Configured via `training.scheduler`:

| Scheduler | Value | Description |
|-----------|-------|-------------|
| Cosine Annealing | `cosine` | Smooth decay following cosine curve (recommended for SPFormer3D) |
| Step | `step` | Reduce by factor at fixed intervals |
| Multi-Step | `multi_step` | Reduce at specific milestones |
| Exponential | `exponential` | Multiply by constant factor each epoch |
| Reduce on Plateau | `reduce_on_plateau` | Reduce when validation loss plateaus |

### Data Augmentation

Configured via `training.augmentations`:

| Augmentation | Effect | Recommended For |
|--------------|--------|-----------------|
| `rotate` | Random rotation around Z-axis (up to `rotate_range` degrees) | Both |
| `jitter` | Gaussian noise on XYZ coordinates (`jitter_std` meters) | Both |
| `scale` | Random scaling within `scale_range` | SPFormer3D |
| `flip` | Random horizontal flip (X and/or Y axes) | Both |
| `elastic_distortion` | Non-linear deformation of point positions | SPFormer3D |
| `color_jitter` | Random perturbation of RGB features | If RGB available |
| `dropout` | Random point removal (fraction) | Light use only |

---

## Inference Pipeline

The `InferencePipeline` class handles:

1. **Model Loading**: From MLflow Model Registry or local checkpoint
2. **Tile-Based Processing**: Large point clouds are split into overlapping tiles for memory efficiency
3. **Batch Processing**: Multiple tiles processed in parallel
4. **Model-Aware Postprocessing**:
   - PointGroup-3heads: Embeddings + offsets → HDBSCAN clustering → instance IDs
   - SPFormer3D: Direct instance mask prediction → threshold → instance IDs (no clustering)
5. **Polygon Extraction**: Instance point clouds → alpha-shape / convex hull → 2D polygons
6. **Output Saving**: GeoJSON, Shapefile, or Parquet format

```bash
# Local inference (single file)
python -m lidar_panoptic_segmentation.infer \
    --config config.yaml \
    --input ./data/forest.las \
    --output ./predictions/

# Batch inference (directory)
python -m lidar_panoptic_segmentation.infer \
    --config config.yaml \
    --input ./data/lidar/ \
    --output ./predictions/ \
    --recursive

# With specific model checkpoint
python -m lidar_panoptic_segmentation.infer \
    --config config.yaml \
    --model models:/LidarPanopticSegmentation/Production \
    --input ./data/forest.las
```

---

## Evaluation Metrics

The evaluation module (`evaluation.py`) provides a comprehensive suite of metrics organized into six categories.

### Semantic Segmentation Metrics

| Metric | Description | Range |
|--------|-------------|-------|
| **mIoU** | Mean Intersection over Union across classes | 0–1 (higher is better) |
| **Overall Accuracy** | Fraction of correctly classified points | 0–1 |
| **Per-class IoU** | IoU for each class (non-tree, tree) | 0–1 |
| **Confusion Matrix** | Full C×C matrix of predictions vs ground truth | Counts |

### Instance Segmentation Metrics

| Metric | Description | Range |
|--------|-------------|-------|
| **Precision** | TP / (TP + FP) — what fraction of predictions are correct | 0–1 |
| **Recall** | TP / (TP + FN) — what fraction of ground truth was detected | 0–1 |
| **F1 Score** | Harmonic mean of Precision and Recall | 0–1 |
| **MUCov** | Mean Unweighted Coverage — average best IoU per GT instance | 0–1 |
| **MWCov** | Mean Weighted Coverage — size-weighted average best IoU | 0–1 |

### Panoptic Quality

Following Kirillov et al. (CVPR 2019):

| Metric | Formula | Description |
|--------|---------|-------------|
| **PQ** | SQ × RQ | Overall panoptic quality |
| **SQ** | Mean IoU of matched segments | How well matched segments align |
| **RQ** | TP / (TP + FP/2 + FN/2) | How well segments are detected (F1-like) |
| **PQ_things** | PQ for instance classes (trees) | Instance-aware quality |
| **PQ_stuff** | PQ for stuff classes (non-tree) | Semantic-only quality |

### Detection Metrics

Multi-threshold detection evaluation:

| Metric | Description |
|--------|-------------|
| **AP@25** | Average Precision at IoU ≥ 0.25 |
| **AP@50** | Average Precision at IoU ≥ 0.50 |
| **AP@75** | Average Precision at IoU ≥ 0.75 |
| **mAP** | Mean AP across thresholds |
| **F1@50** | F1 score at IoU ≥ 0.50 |

### Forestry Metrics

Domain-specific metrics for forestry applications:

| Metric | Description |
|--------|-------------|
| **Detection Rate** | Overall tree detection rate |
| **Commission Rate** | False positive rate (phantom trees) |
| **Omission Rate** | False negative rate (missed trees) |
| **Detection by Size** | Separate rates for small (<5m), medium (5–15m), large (>15m) trees |
| **Height RMSE/MAE** | Tree height estimation error |
| **Location RMSE** | Centroid location error |
| **Crown Diameter RMSE** | Crown diameter estimation error |
| **Canopy Cover Error** | Difference in predicted vs actual canopy coverage |

### Validation Results Visualization

After running validation, the `validation_metrics.json` file contains structured results that can be visualized. Here's what the output looks like:

```
============================================================
VALIDATION RESULTS — Polygon Comparison
============================================================

Per-file Results:
┌─────────────┬───────────┬────────┬────────┐
│ File        │ Precision │ Recall │   F1   │
├─────────────┼───────────┼────────┼────────┤
│ plot_001    │   0.847   │ 0.912  │ 0.878  │
│ plot_002    │   0.923   │ 0.856  │ 0.888  │
│ plot_003    │   0.891   │ 0.934  │ 0.912  │
│ ...         │   ...     │  ...   │  ...   │
├─────────────┼───────────┼────────┼────────┤
│ AGGREGATE   │   0.887   │ 0.901  │ 0.893  │
└─────────────┴───────────┴────────┴────────┘

Instance Matching (IoU ≥ 0.5):
  True Positives:   487
  False Positives:   62
  False Negatives:   53
  Mean Matched IoU: 0.723
============================================================
```

**Interpreting results in MLflow:**

The validation pipeline logs these metrics to MLflow, accessible via the Databricks MLflow UI:

- Navigate to **Machine Learning → Experiments** in Databricks
- Select the experiment:
  - `LidarPanopticSegmentation_Experiments` for PointGroup-3heads
  - `LidarPanopticSegmentation_SPFormer3D` for SPFormer3D
- Click on the `polygon_validation` run
- View metrics: `val/polygon_precision`, `val/polygon_recall`, `val/polygon_f1`
- Download the `validation_metrics.json` artifact for detailed per-file results

**Comparing models in MLflow:**

To compare PointGroup-3heads vs SPFormer3D:

1. Open the MLflow **Compare Runs** view
2. Select validation runs from both experiments
3. Compare `val/polygon_f1` to see which architecture performs better on your data
4. Look at per-file results in the artifacts to identify scenes where one model excels

```
Typical Comparison (example results):
┌──────────────────┬───────────┬────────┬────────┐
│ Model            │ Precision │ Recall │   F1   │
├──────────────────┼───────────┼────────┼────────┤
│ PointGroup-3heads│   0.82    │ 0.78   │ 0.80   │
│ SPFormer3D       │   0.89    │ 0.90   │ 0.89   │
└──────────────────┴───────────┴────────┴────────┘

SPFormer3D typically shows improvement in:
  • Dense canopy with overlapping crowns (+15% recall)
  • Small/suppressed trees (+10% detection rate)
  • Boundary precision between adjacent trees
```

---

## MLflow Integration

All training, inference, and validation runs are tracked in MLflow:

### Logged During Training

| Category | Items |
|----------|-------|
| **Parameters** | All config values (model name, batch size, lr, epochs, etc.) |
| **Metrics** | `train/loss`, `train/semantic_loss`, `train/lr`, `val/loss`, `val/miou` |
| **Artifacts** | Best checkpoint, training config, loss curves |
| **Model** | Registered in Unity Catalog Model Registry |

### Logged During Validation

| Category | Items |
|----------|-------|
| **Metrics** | `val/polygon_precision`, `val/polygon_recall`, `val/polygon_f1` |
| **Artifacts** | `validation_metrics.json` with per-file breakdown |

### Using Models from Registry

```python
import mlflow

# Load the production model
model = mlflow.pytorch.load_model("models:/LidarPanopticSegmentation/Production")

# Load a specific version
model = mlflow.pytorch.load_model("models:/LidarPanopticSegmentation/3")

# List all model versions
from mlflow.tracking import MlflowClient
client = MlflowClient()
versions = client.search_model_versions("name='LidarPanopticSegmentation'")
```

---

## Project Structure

```
lidar-panoptic-segmentation/
├── .devcontainer/                       # VS Code devcontainer (Databricks 15.4 LTS GPU)
│   ├── devcontainer.json                # Container configuration
│   ├── Dockerfile                       # CUDA 12.1, Python 3.10, PyTorch 2.1.0
│   └── post-create.sh                   # Setup: MinkowskiEngine, pip install, pytest
│
├── lidar_panoptic_segmentation/         # Main Python package
│   ├── __init__.py
│   ├── config.py                        # Config loader (load_config, save_config, validate)
│   ├── config_schema.py                 # Pydantic schemas (Config, TransformerConfig, etc.)
│   ├── dataset.py                       # UC-aware data loading (LAS/LAZ/PLY, CloudStorage)
│   ├── model.py                         # BasePanopticModel, LidarPanopticModel, create_model
│   ├── mask_transformer.py              # SPFormer3D (backbone, superpoints, decoder, loss)
│   ├── train.py                         # Trainer class (mixed precision, scheduling, MLflow)
│   ├── infer.py                         # InferencePipeline (tiling, batch, postprocess)
│   ├── postprocess.py                   # HDBSCAN clustering, polygon extraction, save
│   ├── evaluation.py                    # All metrics (semantic, instance, panoptic, forestry)
│   ├── logging_utils.py                 # MLflow + WandB experiment loggers
│   └── utils.py                         # Utilities (set_seed, etc.)
│
├── databricks/                          # Databricks Asset Bundle
│   ├── databricks.yml                   # Bundle config (targets, variables, artifacts)
│   ├── config.yaml                      # PointGroup-3heads config (UC placeholders)
│   ├── config_spformer.yaml             # SPFormer3D config (UC placeholders)
│   ├── src/
│   │   ├── common.py                    # Shared: parse args, resolve placeholders, load config
│   │   ├── run_training.py              # Training entry point
│   │   ├── run_inference.py             # Inference entry point
│   │   └── run_validation.py            # Polygon validation entry point
│   └── resources/                       # Job definitions (auto-discovered)
│       ├── training_job.yml             # PointGroup training (A100)
│       ├── inference_job.yml            # PointGroup inference (A100)
│       ├── validation_job.yml           # PointGroup validation (T4)
│       ├── training_spformer_job.yml    # SPFormer3D training (A100)
│       ├── inference_spformer_job.yml   # SPFormer3D inference (A100)
│       └── validation_spformer_job.yml  # SPFormer3D validation (T4)
│
├── tests/                               # Test suite (111 tests)
│   ├── test_config.py                   # Configuration validation tests
│   ├── test_infer.py                    # Inference pipeline tests
│   ├── test_postprocess.py              # Postprocessing tests
│   └── test_mask_transformer.py         # SPFormer3D tests (63 tests, 13 test classes)
│
├── scripts/
│   └── install_minkowski.sh             # MinkowskiEngine GPU installer
│
├── notebooks/
│   └── LidarPanopticSegmentation_Demo.ipynb  # Interactive demo
│
├── config.yaml                          # Local development config
├── environment.yml                      # Conda environment
├── setup.py / pyproject.toml            # Package definition
├── cluster_config_guidance.md           # Databricks cluster setup notes
└── README.md                            # This file
```

---

## Testing

The project has 111 tests covering all components:

```bash
# Run all tests
pytest tests/ -v

# Run specific test files
pytest tests/test_mask_transformer.py -v   # 63 SPFormer3D tests
pytest tests/test_config.py -v             # Configuration tests
pytest tests/test_infer.py -v              # Inference tests
pytest tests/test_postprocess.py -v        # Postprocessing tests

# Run with coverage
pytest tests/ --cov=lidar_panoptic_segmentation --cov-report=html

# Run in the dev container (recommended for full GPU testing)
# The devcontainer auto-runs pytest on creation
```

### SPFormer3D Test Coverage

The `test_mask_transformer.py` file contains 63 tests across 13 test classes:

| Test Class | Tests | What It Validates |
|------------|:-----:|-------------------|
| `TestSuperpointPooling` | 6 | Superpoint creation, pooling, positional encoding |
| `TestMaskedTransformerDecoderLayer` | 4 | Self/cross attention, masking, output shapes |
| `TestSPFormerTransformerDecoder` | 6 | Full decoder, deep supervision, mask predictions |
| `TestHungarianMatcher` | 4 | Bipartite matching, cost matrix, empty cases |
| `TestSPFormer3DLoss` | 8 | Focal, dice, CE loss, deep supervision, aux semantic |
| `TestSPFormer3DModel` | 10 | Forward pass, output shapes, batch handling, eval mode |
| `TestSPFormer3DGradients` | 4 | Gradient flow through all components |
| `TestParameterGroups` | 3 | Differential LR, backbone vs transformer params |
| `TestInstancePrediction` | 4 | Confidence filtering, mask thresholding, greedy assign |
| `TestSPFormer3DFallback` | 2 | CPU fallback model (no MinkowskiEngine) |
| `TestConfigIntegration` | 7 | TransformerConfig, enum values, create_model dispatch |
| `TestPostprocessIntegration` | 3 | instance_pred passthrough, backward compatibility |
| `TestSPFormerYAMLConfig` | 2 | YAML config loading and validation |

---

## API Reference

### Configuration

```python
from lidar_panoptic_segmentation.config import load_config, validate_config

config = load_config("config.yaml")
warnings = validate_config(config)
```

### Model Creation

```python
from lidar_panoptic_segmentation.model import create_model, load_model

# Create from config (dispatches to PointGroup or SPFormer3D based on model_name)
model = create_model(config)

# Load from MLflow
model = load_model("models:/LidarPanopticSegmentation/latest")
```

### SPFormer3D Direct Usage

```python
from lidar_panoptic_segmentation.mask_transformer import SPFormer3D

model = SPFormer3D(
    num_classes=2,
    in_channels=4,
    backbone_channels=[64, 128, 256, 512],
    feature_dim=64,
    transformer_dim=256,
    num_heads=8,
    num_decoder_layers=6,
    num_queries=100,
)

# Forward pass
output = model(coords, features)  # coords: (N,4), features: (N,4)
# output.semantic_pred: (N,)
# output.instance_pred: (N,)

# Compute loss (training)
loss = model.compute_loss(output, semantic_labels, instance_labels)
# loss.total, loss.semantic, loss.mask
```

### Dataset

```python
from lidar_panoptic_segmentation.dataset import create_dataloader, read_point_cloud

train_loader = create_dataloader(config, split="train")
val_loader = create_dataloader(config, split="val")
data = read_point_cloud("forest.las")
```

### Postprocessing

```python
from lidar_panoptic_segmentation.postprocess import postprocess_predictions, save_results

# With PointGroup (clustering)
result = postprocess_predictions(points, semantic_pred, embeddings=emb, offset_pred=off, config=config)

# With SPFormer3D (direct instance_pred, no clustering)
result = postprocess_predictions(points, semantic_pred, instance_pred=inst_pred, config=config)

save_results(result, "output/trees.geojson")
```

### Evaluation

```python
from lidar_panoptic_segmentation.evaluation import evaluate, evaluate_extended

# Basic evaluation
result = evaluate(pred_semantic, gt_semantic, pred_instances, gt_instances)
print(result.summary())

# Extended evaluation (all metrics)
extended = evaluate_extended(pos, pred_semantic, gt_semantic, pred_instances, gt_instances)
```

---

## Troubleshooting

### MinkowskiEngine Installation

MinkowskiEngine requires CUDA toolkit and compilation from source:

```bash
# In devcontainer (automatic)
# MinkowskiEngine is built during post-create.sh

# On Databricks (via notebook)
%sh
export ME_REPO="https://github.com/NVIDIA/MinkowskiEngine.git"
export ME_REF="master"
bash ./scripts/install_minkowski.sh

# Verify installation
python -c "import MinkowskiEngine as ME; print(ME.__version__)"
```

### Common Issues

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: MinkowskiEngine` | Install MinkowskiEngine from source with CUDA support |
| `CUDA out of memory` (SPFormer3D) | Reduce `training.batch_size` to 1, or use A100 GPU |
| `CUDA out of memory` (PointGroup) | Reduce `training.batch_size` to 2 |
| `scipy not found` | `pip install scipy` (required for Hungarian matching in SPFormer3D) |
| `hdbscan not found` | `pip install hdbscan` (required for PointGroup-3heads clustering) |
| `shapely not found` | `pip install shapely` (required for polygon extraction) |
| Databricks bundle validate fails | Ensure `databricks.yml` target variables match your UC catalog/schema |
| No matching validation files | Ensure prediction and ground truth GeoJSON files share the same stem names |
| Training loss NaN (SPFormer3D) | Reduce `training.lr` or tighten `training.gradient_clip` |
| Slow training convergence | Check that augmentations are enabled; SPFormer3D needs 120+ epochs |

### GPU Recommendations

| Task | Minimum GPU | Recommended GPU |
|------|-------------|-----------------|
| PointGroup training | T4 (16GB) | A10 (24GB) |
| SPFormer3D training | A10 (24GB) | A100 (40/80GB) |
| Inference (both) | T4 (16GB) | T4 (16GB) |
| Validation | CPU sufficient | T4 (16GB) |

---

## Citation

If you use this code in your research, please cite the original SegmentAnyTree paper:

```bibtex
@article{WIELGOSZ2024114367,
title = {SegmentAnyTree: A sensor and platform agnostic deep learning model for tree segmentation using laser scanning data},
journal = {Remote Sensing of Environment},
volume = {313},
pages = {114367},
year = {2024},
issn = {0034-4257},
doi = {https://doi.org/10.1016/j.rse.2024.114367},
url = {https://www.sciencedirect.com/science/article/pii/S0034425724003936},
author = {Maciej Wielgosz and Stefano Puliti and Binbin Xiang and Konrad Schindler and Rasmus Astrup},
keywords = {3D deep learning, Instance segmentation, ITC, ALS, TLS, Drones}
}
```

For the SPFormer architecture:

```bibtex
@inproceedings{sun2023spformer,
  title={Superpoint Transformer for 3D Scene Instance Segmentation},
  author={Sun, Jiahao and Qing, Chunhou and Tan, Tieniu and Xu, Qiang},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2023}
}
```

## License

See [LICENSE](LICENSE) for details.

## Acknowledgments

This project builds on:
- [SegmentAnyTree](https://github.com/maciekwielgosz/SegmentAnyTree) — Original implementation
- [SPFormer](https://github.com/sunjiahao1999/SPFormer) — Superpoint mask transformer
- [OneFormer3D](https://github.com/oneformer3d/oneformer3d) — Unified 3D panoptic
- [torch-points3d](https://github.com/torch-points3d/torch-points3d) — Deep learning framework
- [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine) — Sparse tensor operations
- [PointGroup](https://github.com/dvlab-research/PointGroup) — Panoptic architecture

## Issues

If you encounter any issues with the code please provide your feedback by raising an issue in this repo!
