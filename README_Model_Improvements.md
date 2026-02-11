# Model Accuracy Optimization Guide

This guide provides recommendations for improving the accuracy of the LiDAR Panoptic Segmentation model for tree detection and instance segmentation.

## Table of Contents

1. [Architecture Upgrades](#architecture-upgrades)
2. [Data Augmentation](#data-augmentation)
3. [Hyperparameter Optimization](#hyperparameter-optimization)
4. [Loss Function Improvements](#loss-function-improvements)
5. [Clustering Refinement](#clustering-refinement)
6. [Post-Processing Enhancements](#post-processing-enhancements)
7. [Training Strategies](#training-strategies)

---

## Architecture Upgrades

### 1. ForestFormer (Transformer-based)

Replace the CNN backbone with a transformer architecture for better long-range dependencies:

```yaml
# config.yaml
training:
  model_name: ForestFormer
  backbone: ViT-Base
  attention_heads: 8
  transformer_layers: 6
```

**Benefits:**
- Better global context understanding
- Improved handling of varying tree densities
- More robust to point cloud noise

**Implementation Notes:**
- Requires more GPU memory
- Consider gradient checkpointing for large models

### 2. KPConv++ (Improved Kernel Point Convolution)

Upgrade to KPConv++ for better point cloud processing:

```python
# Enhanced KPConv configuration
backbone_config = {
    "architecture": "kpconv++",
    "num_kernel_points": 15,
    "kernel_influence_radius": 0.1,
    "deformable": True,
    "aggregation": "attention",
}
```

**Benefits:**
- Deformable kernels adapt to tree shapes
- Attention-based aggregation for better feature fusion

### 3. Swin3D (3D Swin Transformer)

For state-of-the-art performance on 3D point clouds:

```yaml
training:
  model_name: Swin3D
  backbone: swin3d-base
  window_size: 8
  depth: [2, 2, 18, 2]
```

**Benefits:**
- Hierarchical feature extraction
- Efficient self-attention with window partitioning
- Pre-training on large datasets possible

### 4. Hybrid Architecture

Combine local and global processing:

```python
class HybridPanopticModel(nn.Module):
    def __init__(self):
        self.local_encoder = KPConvEncoder()  # Local features
        self.global_encoder = TransformerEncoder()  # Global context
        self.fusion = CrossAttentionFusion()
        self.heads = PanopticHeads()
```

---

## Data Augmentation

### Standard Augmentations

Enable these in `config.yaml`:

```yaml
training:
  augmentations:
    rotate: true
    rotate_range: 180.0
    jitter: true
    jitter_std: 0.02
    scale: true
    scale_range: [0.9, 1.1]
    flip: true
    dropout: 0.05
```

### Advanced Augmentations

#### Elastic Distortion

```python
def elastic_distortion(points, granularity=0.2, magnitude=0.4):
    """Apply elastic distortion to point cloud."""
    noise = np.random.randn(*points.shape) * magnitude
    # Smooth noise with Gaussian filter
    from scipy.ndimage import gaussian_filter
    noise = gaussian_filter(noise, sigma=granularity * points.shape[0])
    return points + noise
```

#### Mix3D (3D MixUp)

```python
def mix3d(points1, points2, labels1, labels2, alpha=0.5):
    """Mix two point clouds for augmentation."""
    # Random mixing ratio
    lam = np.random.beta(alpha, alpha)

    # Sample points from each cloud
    n1 = int(len(points1) * lam)
    n2 = len(points1) - n1

    idx1 = np.random.choice(len(points1), n1, replace=False)
    idx2 = np.random.choice(len(points2), n2, replace=False)

    mixed_points = np.vstack([points1[idx1], points2[idx2]])
    mixed_labels = np.concatenate([labels1[idx1], labels2[idx2]])

    return mixed_points, mixed_labels
```

#### CutMix3D

```python
def cutmix3d(points1, points2, labels1, labels2):
    """Cut and paste regions between point clouds."""
    # Random bounding box
    min_bound = points1.min(axis=0)
    max_bound = points1.max(axis=0)

    cut_ratio = np.random.uniform(0.2, 0.5)
    cut_size = (max_bound - min_bound) * cut_ratio
    cut_center = min_bound + np.random.rand(3) * (max_bound - min_bound)

    # Mask for points in cut region
    mask1 = np.all(np.abs(points1 - cut_center) < cut_size/2, axis=1)
    mask2 = np.all(np.abs(points2 - cut_center) < cut_size/2, axis=1)

    # Replace region
    mixed_points = np.vstack([points1[~mask1], points2[mask2] + (cut_center - points2[mask2].mean(axis=0))])
    mixed_labels = np.concatenate([labels1[~mask1], labels2[mask2]])

    return mixed_points, mixed_labels
```

---

## Hyperparameter Optimization

### Using Optuna

```python
import optuna

def objective(trial):
    # Suggest hyperparameters
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [4, 8, 16, 32])
    embed_dim = trial.suggest_categorical("embed_dim", [4, 5, 8, 16])
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)

    # Update config
    config.training.learning_rate = lr
    config.training.batch_size = batch_size
    config.training.embed_dim = embed_dim
    config.training.weight_decay = weight_decay

    # Train and evaluate
    trainer = Trainer(config)
    trainer.setup()
    metrics = trainer.train()

    return metrics["best_val_loss"]

# Run optimization
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100)

print("Best hyperparameters:", study.best_params)
```

### Using Ray Tune

```python
from ray import tune
from ray.tune.schedulers import ASHAScheduler

config_space = {
    "lr": tune.loguniform(1e-5, 1e-2),
    "batch_size": tune.choice([4, 8, 16, 32]),
    "embed_dim": tune.choice([4, 5, 8, 16]),
    "scheduler": tune.choice(["cosine", "exponential", "step"]),
}

scheduler = ASHAScheduler(
    max_t=100,
    grace_period=10,
    reduction_factor=2,
)

analysis = tune.run(
    train_model,
    config=config_space,
    scheduler=scheduler,
    num_samples=50,
    resources_per_trial={"gpu": 1},
)

print("Best config:", analysis.best_config)
```

### Key Hyperparameters to Tune

| Parameter | Search Range | Impact |
|-----------|--------------|--------|
| Learning Rate | 1e-5 to 1e-2 | High |
| Batch Size | 4 to 32 | Medium |
| Embed Dim | 4 to 16 | Medium |
| Weight Decay | 1e-6 to 1e-2 | Medium |
| Voxel Size | 0.01 to 0.05 | High |
| Sample Radius | 5 to 15 | Medium |

---

## Loss Function Improvements

### Focal Loss for Class Imbalance

```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()
```

### Dice Loss for Better Segmentation

```python
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=logits.shape[1])
        targets_one_hot = targets_one_hot.permute(0, 2, 1).float()

        intersection = (probs * targets_one_hot).sum()
        union = probs.sum() + targets_one_hot.sum()

        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice
```

### Combined Panoptic Loss

```python
class ImprovedPanopticLoss(nn.Module):
    def __init__(self, semantic_weight=1.0, offset_weight=1.0,
                 embedding_weight=1.0, dice_weight=0.5):
        super().__init__()
        self.semantic_weight = semantic_weight
        self.offset_weight = offset_weight
        self.embedding_weight = embedding_weight
        self.dice_weight = dice_weight

        self.focal_loss = FocalLoss()
        self.dice_loss = DiceLoss()

    def forward(self, output, semantic_labels, instance_labels):
        # Semantic losses
        focal = self.focal_loss(output.semantic_logits, semantic_labels)
        dice = self.dice_loss(output.semantic_logits, semantic_labels)
        semantic_loss = focal + self.dice_weight * dice

        # Offset loss with smooth L1
        offset_loss = F.smooth_l1_loss(
            output.offset_pred,
            self._compute_offsets(output.coords, instance_labels),
        )

        # Embedding loss with margin-based discrimination
        embedding_loss = self._lovasz_softmax(
            output.embedding,
            instance_labels,
        )

        total = (
            self.semantic_weight * semantic_loss +
            self.offset_weight * offset_loss +
            self.embedding_weight * embedding_loss
        )

        return total
```

---

## Clustering Refinement

### Improved HDBSCAN Parameters

```yaml
postprocess:
  clustering_method: hdbscan
  hdbscan_min_cluster_size: 30  # Lower for detecting small trees
  hdbscan_min_samples: 5
  hdbscan_cluster_selection_epsilon: 0.5  # Merge nearby clusters
```

### Two-Stage Clustering

```python
def two_stage_clustering(points, embeddings):
    """Two-stage clustering for better instance separation."""
    # Stage 1: Coarse clustering with spatial coordinates
    from sklearn.cluster import DBSCAN
    coarse_labels = DBSCAN(eps=2.0, min_samples=20).fit_predict(points[:, :2])

    # Stage 2: Fine clustering with embeddings within each coarse cluster
    final_labels = np.zeros_like(coarse_labels)
    offset = 0

    for cluster_id in np.unique(coarse_labels):
        if cluster_id == -1:
            continue

        mask = coarse_labels == cluster_id
        cluster_embeddings = embeddings[mask]

        # Fine clustering on embeddings
        fine_labels = hdbscan.HDBSCAN(
            min_cluster_size=10,
            metric='euclidean',
        ).fit_predict(cluster_embeddings)

        # Offset labels
        fine_labels[fine_labels >= 0] += offset
        final_labels[mask] = fine_labels
        offset = final_labels.max() + 1

    return final_labels
```

### Graph-Based Refinement

```python
def graph_refine_clusters(points, instance_pred, k_neighbors=15):
    """Refine clusters using graph connectivity."""
    from scipy.sparse.csgraph import connected_components
    from scipy.sparse import csr_matrix

    # Build k-NN graph
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=k_neighbors)
    nn.fit(points)
    distances, indices = nn.kneighbors(points)

    # Create adjacency matrix (same instance = connected)
    n_points = len(points)
    rows, cols = [], []

    for i in range(n_points):
        for j in indices[i]:
            if instance_pred[i] == instance_pred[j] and instance_pred[i] >= 0:
                rows.append(i)
                cols.append(j)

    adj = csr_matrix(
        (np.ones(len(rows)), (rows, cols)),
        shape=(n_points, n_points)
    )

    # Find connected components
    n_components, labels = connected_components(adj, directed=False)

    return labels
```

---

## Post-Processing Enhancements

### Instance Merging

```python
def merge_split_instances(instances, distance_threshold=2.0, height_threshold=3.0):
    """Merge instances that were incorrectly split."""
    merged = []
    used = set()

    for i, inst1 in enumerate(instances):
        if i in used:
            continue

        current = inst1
        used.add(i)

        for j, inst2 in enumerate(instances):
            if j in used or j <= i:
                continue

            # Check if should merge
            center_dist = np.linalg.norm(inst1.center[:2] - inst2.center[:2])
            height_diff = abs(inst1.height - inst2.height)

            if center_dist < distance_threshold and height_diff < height_threshold:
                # Merge points
                current = TreeInstance(
                    instance_id=current.instance_id,
                    points=np.vstack([current.points, inst2.points]),
                )
                used.add(j)

        merged.append(current)

    return merged
```

### Noise Removal

```python
def remove_noisy_instances(instances, min_points=50, min_height=2.0,
                           max_aspect_ratio=10.0):
    """Remove noisy or invalid instances."""
    valid = []

    for inst in instances:
        # Check point count
        if len(inst.points) < min_points:
            continue

        # Check height
        if inst.height < min_height:
            continue

        # Check aspect ratio (crown vs height)
        crown_diameter = np.max([
            inst.points[:, 0].max() - inst.points[:, 0].min(),
            inst.points[:, 1].max() - inst.points[:, 1].min(),
        ])
        aspect_ratio = inst.height / (crown_diameter + 1e-6)

        if aspect_ratio > max_aspect_ratio:
            continue

        valid.append(inst)

    return valid
```

### Polygon Smoothing

```python
def smooth_polygon(polygon, tolerance=0.5):
    """Smooth polygon boundary."""
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    # Buffer out and in to smooth
    smoothed = polygon.buffer(tolerance).buffer(-tolerance)

    # Simplify
    smoothed = smoothed.simplify(tolerance * 0.5)

    return smoothed
```

---

## Training Strategies

### Curriculum Learning

Start with easy samples, gradually increase difficulty:

```python
class CurriculumSampler:
    def __init__(self, dataset, epochs):
        self.dataset = dataset
        self.epochs = epochs
        self.difficulties = self._compute_difficulties()

    def _compute_difficulties(self):
        """Compute sample difficulty based on tree density."""
        difficulties = []
        for i in range(len(self.dataset)):
            sample = self.dataset[i]
            n_instances = len(np.unique(sample["instance_labels"]))
            difficulties.append(n_instances)
        return np.array(difficulties)

    def get_weights(self, epoch):
        """Get sampling weights for current epoch."""
        progress = epoch / self.epochs

        # Start with easy (low tree count), end with all samples
        threshold = np.percentile(self.difficulties, (1 - progress) * 50)
        weights = (self.difficulties <= threshold).astype(float)
        weights = weights / weights.sum()

        return weights
```

### Self-Training with Pseudo Labels

```python
def self_training_iteration(model, labeled_loader, unlabeled_loader,
                           confidence_threshold=0.9):
    """One iteration of self-training."""
    model.eval()
    pseudo_labels = []

    # Generate pseudo labels for unlabeled data
    with torch.no_grad():
        for batch in unlabeled_loader:
            output = model(batch["coords"], batch["features"])
            confidence = F.softmax(output.semantic_logits, dim=1).max(dim=1)[0]

            # Keep high-confidence predictions
            mask = confidence > confidence_threshold
            if mask.sum() > 0:
                pseudo_labels.append({
                    "points": batch["points"][mask],
                    "labels": output.semantic_pred[mask],
                })

    # Combine labeled and pseudo-labeled data
    combined_dataset = ConcatDataset([
        labeled_dataset,
        PseudoLabeledDataset(pseudo_labels),
    ])

    return combined_dataset
```

### Knowledge Distillation

```python
def distillation_loss(student_output, teacher_output, labels,
                     temperature=4.0, alpha=0.5):
    """Knowledge distillation loss."""
    # Hard label loss
    hard_loss = F.cross_entropy(student_output.semantic_logits, labels)

    # Soft label loss (distillation)
    soft_student = F.log_softmax(student_output.semantic_logits / temperature, dim=1)
    soft_teacher = F.softmax(teacher_output.semantic_logits / temperature, dim=1)
    soft_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean')
    soft_loss *= temperature ** 2

    return alpha * hard_loss + (1 - alpha) * soft_loss
```

---

## Recommended Optimization Pipeline

1. **Start with baseline**: Train with default config
2. **Data augmentation**: Enable all augmentations
3. **Hyperparameter tuning**: Use Optuna for 50-100 trials
4. **Loss function**: Add focal + dice loss
5. **Architecture**: Try KPConv++ or Swin3D
6. **Post-processing**: Implement merging and noise removal
7. **Self-training**: Use unlabeled data if available
8. **Ensemble**: Combine 3-5 models for final predictions

## Expected Improvements

| Optimization | mIoU Improvement |
|--------------|------------------|
| Data Augmentation | +2-5% |
| Loss Function | +1-3% |
| Architecture Upgrade | +3-8% |
| Hyperparameter Tuning | +2-4% |
| Post-processing | +1-3% |
| Ensemble | +2-4% |

**Total potential improvement: 10-25% mIoU over baseline**
