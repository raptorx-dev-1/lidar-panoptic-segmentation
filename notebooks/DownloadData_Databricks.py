# Databricks notebook source
# MAGIC %md
# MAGIC # Download Training Data to Unity Catalog
# MAGIC
# MAGIC This notebook downloads training datasets for LiDAR Panoptic Segmentation
# MAGIC directly to Unity Catalog Volumes.
# MAGIC
# MAGIC ## Available Datasets
# MAGIC
# MAGIC | Dataset | Size | Description |
# MAGIC |---------|------|-------------|
# MAGIC | FOR-instance | ~15 GB | Forest instance segmentation benchmark (ALS, TLS, MLS) |
# MAGIC | sample | ~10 MB | Generated sample data for testing |
# MAGIC
# MAGIC ## Prerequisites
# MAGIC - Unity Catalog Volume with write access
# MAGIC - Internet access from cluster

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

# Configure your Unity Catalog paths
CATALOG = "your_catalog"           # Your Unity Catalog name
SCHEMA = "your_schema"             # Your schema name
VOLUME = "training_data"           # Volume for training data

# Derived path
DATA_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"

print(f"Data will be downloaded to: {DATA_PATH}")

# COMMAND ----------

# Verify Volume exists and is writable
import os

if os.path.exists(DATA_PATH):
    print(f"Volume exists: {DATA_PATH}")

    # Check if writable
    test_file = os.path.join(DATA_PATH, ".write_test")
    try:
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
        print("Volume is writable")
    except Exception as e:
        print(f"Warning: Volume may not be writable: {e}")
else:
    print(f"Creating Volume directory: {DATA_PATH}")
    os.makedirs(DATA_PATH, exist_ok=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Option 1: Generate Sample Data
# MAGIC
# MAGIC Generate small sample data for testing the pipeline.

# COMMAND ----------

import numpy as np
import sys
sys.path.insert(0, "/Workspace/Repos/your_user/SegmentAnyTree")

from pathlib import Path

def generate_sample_ply(filepath: Path, n_points: int = 5000, n_trees: int = 5):
    """Generate a sample PLY file with synthetic forest data."""
    from plyfile import PlyData, PlyElement

    # Ground points (class 0 - non-tree)
    n_ground = n_points // 3
    ground_x = np.random.uniform(0, 50, n_ground)
    ground_y = np.random.uniform(0, 50, n_ground)
    ground_z = np.random.uniform(0, 0.5, n_ground)
    ground_sem = np.ones(n_ground, dtype=np.int32)  # semantic: 1 = non-tree
    ground_inst = np.zeros(n_ground, dtype=np.int32)

    # Tree points (class 1 - tree)
    tree_points = []
    tree_sems = []
    tree_insts = []

    for tree_id in range(1, n_trees + 1):
        n_tree_points = np.random.randint(200, 500)
        center_x = np.random.uniform(5, 45)
        center_y = np.random.uniform(5, 45)
        tree_height = np.random.uniform(8, 25)

        # Simple cone/cylinder tree shape
        t_z = np.random.uniform(0.5, tree_height, n_tree_points)
        radius = (tree_height - t_z) / tree_height * np.random.uniform(1, 3)
        angle = np.random.uniform(0, 2 * np.pi, n_tree_points)
        t_x = center_x + radius * np.cos(angle) * np.random.uniform(0.5, 1.5, n_tree_points)
        t_y = center_y + radius * np.sin(angle) * np.random.uniform(0.5, 1.5, n_tree_points)

        tree_points.append(np.column_stack([t_x, t_y, t_z]))
        tree_sems.append(np.full(n_tree_points, 2, dtype=np.int32))  # semantic: 2 = tree
        tree_insts.append(np.full(n_tree_points, tree_id, dtype=np.int32))

    # Combine all points
    tree_xyz = np.vstack(tree_points)
    tree_sem = np.concatenate(tree_sems)
    tree_inst = np.concatenate(tree_insts)

    all_x = np.concatenate([ground_x, tree_xyz[:, 0]]).astype(np.float32)
    all_y = np.concatenate([ground_y, tree_xyz[:, 1]]).astype(np.float32)
    all_z = np.concatenate([ground_z, tree_xyz[:, 2]]).astype(np.float32)
    all_sem = np.concatenate([ground_sem, tree_sem]).astype(np.int32)
    all_inst = np.concatenate([ground_inst, tree_inst]).astype(np.int32)

    # Create PLY structure
    vertex_data = np.zeros(
        len(all_x),
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("semantic_seg", "i4"),
            ("treeID", "i4"),
        ]
    )
    vertex_data["x"] = all_x
    vertex_data["y"] = all_y
    vertex_data["z"] = all_z
    vertex_data["semantic_seg"] = all_sem
    vertex_data["treeID"] = all_inst

    el = PlyElement.describe(vertex_data, "vertex")
    PlyData([el], text=True).write(str(filepath))

    return len(all_x), n_trees

# COMMAND ----------

# Generate sample data
sample_dir = Path(DATA_PATH) / "sample" / "raw"

# Create directories
for split in ["train", "val", "test"]:
    split_dir = sample_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)

    n_files = 5 if split == "train" else 2

    for i in range(n_files):
        filename = f"sample_{split}_{i+1}"
        if split == "val":
            filename += "_val"
        elif split == "test":
            filename += "_test"
        filename += ".ply"

        filepath = split_dir / filename
        n_points, n_trees = generate_sample_ply(filepath)
        print(f"Created: {filepath.relative_to(DATA_PATH)} ({n_points} points, {n_trees} trees)")

print("\nSample data generation complete!")
print(f"Data location: {sample_dir}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Option 2: Download FOR-instance Dataset from Zenodo
# MAGIC
# MAGIC Downloads the full FOR-instance benchmark dataset (~15 GB).
# MAGIC This may take a while depending on network speed.

# COMMAND ----------

import json
from urllib.request import urlopen, urlretrieve
from urllib.error import HTTPError, URLError

ZENODO_RECORD_ID = "8287792"
ZENODO_API_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"

def get_zenodo_files():
    """Get list of files from Zenodo record."""
    try:
        with urlopen(ZENODO_API_URL) as response:
            data = json.loads(response.read().decode())
        return data.get("files", [])
    except Exception as e:
        print(f"Error accessing Zenodo API: {e}")
        return []

# Query Zenodo for available files
print("Querying Zenodo for FOR-instance files...")
files = get_zenodo_files()

if files:
    print(f"\nFound {len(files)} files:")
    total_size = 0
    for f in files:
        size_mb = f.get("size", 0) / (1024 * 1024)
        total_size += size_mb
        print(f"  - {f['key']} ({size_mb:.1f} MB)")
    print(f"\nTotal size: {total_size/1024:.1f} GB")
else:
    print("No files found. Please download manually from:")
    print(f"  https://zenodo.org/records/{ZENODO_RECORD_ID}")

# COMMAND ----------

# Download files from Zenodo
# WARNING: This will download ~15 GB of data

DOWNLOAD_FOR_INSTANCE = False  # Set to True to start download

if DOWNLOAD_FOR_INSTANCE and files:
    download_dir = Path(DATA_PATH) / "for-instance" / "downloads"
    download_dir.mkdir(parents=True, exist_ok=True)

    for file_info in files:
        filename = file_info["key"]
        download_url = file_info["links"]["self"]
        output_path = download_dir / filename

        if output_path.exists():
            print(f"Skipping (exists): {filename}")
            continue

        print(f"Downloading: {filename}")
        try:
            urlretrieve(download_url, output_path)
            print(f"  Saved to: {output_path}")
        except Exception as e:
            print(f"  Error: {e}")

    print("\nDownload complete!")
    print(f"Files saved to: {download_dir}")
else:
    print("Set DOWNLOAD_FOR_INSTANCE = True to start download")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Option 3: Upload Your Own LiDAR Data
# MAGIC
# MAGIC If you have your own LiDAR data, upload it to the Volume using the Databricks UI
# MAGIC or by copying files programmatically.

# COMMAND ----------

# Check for existing LiDAR files in the Volume
import glob

lidar_extensions = [".las", ".laz", ".ply", ".LAS", ".LAZ", ".PLY"]
lidar_files = []

for ext in lidar_extensions:
    lidar_files.extend(glob.glob(f"{DATA_PATH}/**/*{ext}", recursive=True))

if lidar_files:
    print(f"Found {len(lidar_files)} LiDAR files in {DATA_PATH}:")
    for f in sorted(lidar_files)[:20]:  # Show first 20
        size_mb = os.path.getsize(f) / (1024 * 1024)
        print(f"  - {os.path.relpath(f, DATA_PATH)} ({size_mb:.1f} MB)")
    if len(lidar_files) > 20:
        print(f"  ... and {len(lidar_files) - 20} more files")
else:
    print(f"No LiDAR files found in {DATA_PATH}")
    print("\nTo upload your data:")
    print("1. Use the Databricks UI to upload files to the Volume")
    print("2. Or use dbutils.fs.cp() to copy from another location")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC After running this notebook, your data should be organized as:
# MAGIC
# MAGIC ```
# MAGIC /Volumes/{catalog}/{schema}/{volume}/
# MAGIC     sample/           # Sample data for testing
# MAGIC         raw/
# MAGIC             train/
# MAGIC             val/
# MAGIC             test/
# MAGIC     for-instance/     # FOR-instance dataset (if downloaded)
# MAGIC         raw/
# MAGIC             ...
# MAGIC     your_data/        # Your own data (if uploaded)
# MAGIC         input/
# MAGIC         output/
# MAGIC ```
# MAGIC
# MAGIC Update your `config.databricks.yaml` to point to these paths.
