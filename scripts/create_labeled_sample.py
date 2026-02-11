#!/usr/bin/env python3
"""
Create sample labeled point cloud for validation testing.

Generates synthetic LiDAR data with:
- Ground points (class 0)
- Tree points (class 1) with instance IDs
"""

import numpy as np
from pathlib import Path
from plyfile import PlyData, PlyElement


def create_synthetic_forest(
    num_trees: int = 15,
    points_per_tree: int = 300,
    ground_points: int = 2000,
    area_size: float = 50.0,
    seed: int = 42,
) -> dict:
    """
    Create synthetic forest point cloud with labels.

    Returns dict with: pos, semantic_seg, treeID
    """
    np.random.seed(seed)

    all_points = []
    all_semantic = []
    all_instance = []

    # Generate ground points
    ground_xy = np.random.uniform(0, area_size, (ground_points, 2))
    ground_z = np.random.normal(0, 0.1, ground_points)  # Small variation
    ground = np.column_stack([ground_xy, ground_z])

    all_points.append(ground)
    all_semantic.append(np.zeros(ground_points, dtype=np.int32))
    all_instance.append(np.zeros(ground_points, dtype=np.int32))

    # Generate trees
    tree_centers = np.random.uniform(5, area_size - 5, (num_trees, 2))

    for tree_id, (tx, ty) in enumerate(tree_centers, start=1):
        # Tree parameters
        height = np.random.uniform(8, 20)
        crown_radius = np.random.uniform(2, 5)
        trunk_radius = np.random.uniform(0.2, 0.5)

        # Trunk points (cylinder)
        n_trunk = points_per_tree // 4
        trunk_z = np.random.uniform(0, height * 0.4, n_trunk)
        trunk_r = np.random.uniform(0, trunk_radius, n_trunk)
        trunk_theta = np.random.uniform(0, 2 * np.pi, n_trunk)
        trunk_x = tx + trunk_r * np.cos(trunk_theta)
        trunk_y = ty + trunk_r * np.sin(trunk_theta)
        trunk = np.column_stack([trunk_x, trunk_y, trunk_z])

        # Crown points (cone/ellipsoid)
        n_crown = points_per_tree - n_trunk
        crown_z = np.random.uniform(height * 0.3, height, n_crown)
        crown_height_ratio = (crown_z - height * 0.3) / (height * 0.7)
        crown_r_max = crown_radius * (1 - crown_height_ratio * 0.5)  # Taper
        crown_r = np.random.uniform(0, 1, n_crown) * crown_r_max
        crown_theta = np.random.uniform(0, 2 * np.pi, n_crown)
        crown_x = tx + crown_r * np.cos(crown_theta)
        crown_y = ty + crown_r * np.sin(crown_theta)
        crown = np.column_stack([crown_x, crown_y, crown_z])

        tree_points = np.vstack([trunk, crown])

        # Add noise
        tree_points += np.random.normal(0, 0.1, tree_points.shape)

        all_points.append(tree_points)
        all_semantic.append(np.ones(len(tree_points), dtype=np.int32))
        all_instance.append(np.full(len(tree_points), tree_id, dtype=np.int32))

    return {
        'pos': np.vstack(all_points).astype(np.float32),
        'semantic_seg': np.concatenate(all_semantic).astype(np.int32),
        'treeID': np.concatenate(all_instance).astype(np.int32),
    }


def save_labeled_ply(data: dict, path: Path):
    """Save labeled point cloud as PLY."""
    n = len(data['pos'])

    dtype = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('semantic_seg', 'i4'),
        ('treeID', 'i4'),
    ]

    arr = np.zeros(n, dtype=dtype)
    arr['x'] = data['pos'][:, 0]
    arr['y'] = data['pos'][:, 1]
    arr['z'] = data['pos'][:, 2]
    arr['semantic_seg'] = data['semantic_seg']
    arr['treeID'] = data['treeID']

    el = PlyElement.describe(arr, 'vertex')
    PlyData([el], text=True).write(str(path))


def main():
    output_dir = Path("data/validation")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Creating labeled validation samples...")

    # Create multiple samples with different characteristics
    samples = [
        {"num_trees": 10, "points_per_tree": 400, "ground_points": 2000, "seed": 42, "name": "sparse_forest"},
        {"num_trees": 25, "points_per_tree": 300, "ground_points": 1500, "seed": 123, "name": "dense_forest"},
        {"num_trees": 15, "points_per_tree": 500, "ground_points": 3000, "seed": 456, "name": "mixed_forest"},
    ]

    for sample in samples:
        name = sample.pop("name")
        data = create_synthetic_forest(**sample)

        out_path = output_dir / f"{name}_labeled.ply"
        save_labeled_ply(data, out_path)

        n_points = len(data['pos'])
        n_trees = data['treeID'].max()
        n_tree_points = (data['semantic_seg'] == 1).sum()

        print(f"  {name}: {n_points:,} points, {n_trees} trees, {n_tree_points:,} tree points")
        print(f"    Saved: {out_path}")

    print(f"\nValidation data created in: {output_dir}")
    print("\nTo run validation:")
    print("  python scripts/validate_model.py -i data/validation -o validation_results")


if __name__ == "__main__":
    main()
