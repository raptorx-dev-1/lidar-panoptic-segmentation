#!/usr/bin/env python3
"""
Create synthetic sample data for testing the training and inference pipeline.

Generates simple tree-like point clouds with semantic and instance labels.
"""

import numpy as np
from pathlib import Path


def create_tree_pointcloud(
    center: tuple,
    n_points: int = 500,
    tree_id: int = 1,
    height: float = 10.0,
    crown_radius: float = 3.0,
) -> dict:
    """Create a simple tree-shaped point cloud."""
    cx, cy, cz = center

    # Trunk points (cylinder)
    trunk_height = height * 0.4
    n_trunk = n_points // 4
    trunk_r = crown_radius * 0.15
    theta = np.random.uniform(0, 2 * np.pi, n_trunk)
    trunk_z = np.random.uniform(0, trunk_height, n_trunk)
    trunk_x = cx + trunk_r * np.cos(theta) + np.random.normal(0, 0.05, n_trunk)
    trunk_y = cy + trunk_r * np.sin(theta) + np.random.normal(0, 0.05, n_trunk)

    # Crown points (ellipsoid)
    n_crown = n_points - n_trunk
    crown_center_z = cz + trunk_height + (height - trunk_height) * 0.5

    # Sample points in ellipsoid
    phi = np.random.uniform(0, 2 * np.pi, n_crown)
    cos_theta = np.random.uniform(-1, 1, n_crown)
    sin_theta = np.sqrt(1 - cos_theta**2)
    r = crown_radius * np.cbrt(np.random.uniform(0, 1, n_crown))

    crown_x = cx + r * sin_theta * np.cos(phi)
    crown_y = cy + r * sin_theta * np.sin(phi)
    crown_z = crown_center_z + r * cos_theta * (height - trunk_height) / (2 * crown_radius)

    # Combine trunk and crown
    x = np.concatenate([trunk_x, crown_x])
    y = np.concatenate([trunk_y, crown_y])
    z = np.concatenate([cz + trunk_z, crown_z])

    points = np.column_stack([x, y, z]).astype(np.float32)

    # All points are tree class (1)
    semantic = np.ones(n_points, dtype=np.int32)

    # All points belong to same instance
    instance = np.full(n_points, tree_id, dtype=np.int32)

    # Random intensity
    intensity = np.random.uniform(0.3, 0.9, n_points).astype(np.float32)

    return {
        "points": points,
        "semantic": semantic,
        "instance": instance,
        "intensity": intensity,
    }


def create_ground_points(
    bounds: tuple,
    n_points: int = 1000,
    z_base: float = 0.0,
) -> dict:
    """Create ground plane points."""
    x_min, x_max, y_min, y_max = bounds

    x = np.random.uniform(x_min, x_max, n_points).astype(np.float32)
    y = np.random.uniform(y_min, y_max, n_points).astype(np.float32)
    z = z_base + np.random.normal(0, 0.1, n_points).astype(np.float32)

    points = np.column_stack([x, y, z])

    # Ground is non-tree (0)
    semantic = np.zeros(n_points, dtype=np.int32)
    instance = np.zeros(n_points, dtype=np.int32)  # No instance for ground
    intensity = np.random.uniform(0.1, 0.4, n_points).astype(np.float32)

    return {
        "points": points,
        "semantic": semantic,
        "instance": instance,
        "intensity": intensity,
    }


def create_sample_scene(
    n_trees: int = 5,
    area_size: float = 30.0,
    points_per_tree: int = 500,
    ground_points: int = 2000,
) -> dict:
    """Create a sample scene with multiple trees and ground."""
    all_points = []
    all_semantic = []
    all_instance = []
    all_intensity = []

    # Add ground
    ground = create_ground_points(
        bounds=(0, area_size, 0, area_size),
        n_points=ground_points,
    )
    all_points.append(ground["points"])
    all_semantic.append(ground["semantic"])
    all_instance.append(ground["instance"])
    all_intensity.append(ground["intensity"])

    # Add trees at random positions
    for i in range(n_trees):
        cx = np.random.uniform(3, area_size - 3)
        cy = np.random.uniform(3, area_size - 3)
        height = np.random.uniform(8, 15)
        radius = np.random.uniform(2, 4)

        tree = create_tree_pointcloud(
            center=(cx, cy, 0),
            n_points=points_per_tree,
            tree_id=i + 1,
            height=height,
            crown_radius=radius,
        )
        all_points.append(tree["points"])
        all_semantic.append(tree["semantic"])
        all_instance.append(tree["instance"])
        all_intensity.append(tree["intensity"])

    return {
        "points": np.vstack(all_points),
        "semantic": np.concatenate(all_semantic),
        "instance": np.concatenate(all_instance),
        "intensity": np.concatenate(all_intensity),
    }


def save_as_ply(data: dict, output_path: Path):
    """Save point cloud data as PLY file with labels."""
    points = data["points"]
    semantic = data["semantic"]
    instance = data["instance"]
    intensity = data["intensity"]

    n_points = len(points)

    # Create PLY header
    header = f"""ply
format ascii 1.0
element vertex {n_points}
property float x
property float y
property float z
property float intensity
property int semantic_seg
property int treeID
end_header
"""

    with open(output_path, "w") as f:
        f.write(header)
        for i in range(n_points):
            f.write(f"{points[i, 0]:.6f} {points[i, 1]:.6f} {points[i, 2]:.6f} "
                    f"{intensity[i]:.6f} {semantic[i]} {instance[i]}\n")

    print(f"Saved {n_points} points to {output_path}")


def main():
    """Create sample dataset."""
    # Create output directories
    base_dir = Path(__file__).parent.parent / "data" / "sample"
    train_dir = base_dir / "train" / "las"
    val_dir = base_dir / "val" / "las"
    test_dir = base_dir / "test" / "las"

    for d in [train_dir, val_dir, test_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print("Creating sample dataset...")

    # Create training samples (3 scenes)
    for i in range(3):
        np.random.seed(42 + i)
        scene = create_sample_scene(n_trees=5, area_size=30.0)
        save_as_ply(scene, train_dir / f"train_scene_{i:02d}.ply")

    # Create validation sample (1 scene)
    np.random.seed(100)
    scene = create_sample_scene(n_trees=4, area_size=25.0)
    save_as_ply(scene, val_dir / "val_scene_00.ply")

    # Create test sample (1 scene)
    np.random.seed(200)
    scene = create_sample_scene(n_trees=6, area_size=35.0)
    save_as_ply(scene, test_dir / "test_scene_00.ply")

    print(f"\nSample dataset created at: {base_dir}")
    print(f"  Training:   {train_dir} (3 files)")
    print(f"  Validation: {val_dir} (1 file)")
    print(f"  Test:       {test_dir} (1 file)")

    # Create a sample config for this data
    config_content = f"""# Sample configuration for testing with synthetic data
env:
  name: local
  debug: true
  seed: 42

paths:
  data_root: {base_dir}
  train:
    pointclouds: {train_dir}
  val:
    pointclouds: {val_dir}
  test:
    pointclouds: {test_dir}
  models: {base_dir / 'models'}
  logs: {base_dir / 'logs'}

training:
  model_name: PointGroup3heads
  num_classes: 2
  embed_dim: 5
  epochs: 5
  batch_size: 2
  learning_rate: 0.001
  voxel_size: 0.1
  mixed_precision: false
  checkpoint_frequency: 1

inference:
  min_points_per_instance: 20
  confidence_threshold: 0.5

postprocess:
  clustering_method: hdbscan
  hdbscan_min_cluster_size: 20
  min_height: 1.0

logging:
  level: DEBUG
  mlflow:
    enabled: false
  wandb:
    enabled: false
"""

    config_path = base_dir / "config_sample.yaml"
    with open(config_path, "w") as f:
        f.write(config_content)

    print(f"  Config:     {config_path}")


if __name__ == "__main__":
    main()
