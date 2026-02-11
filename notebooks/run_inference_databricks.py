# Databricks notebook source
# MAGIC %md
# MAGIC # SegmentAnyTree - Native Inference on Databricks
# MAGIC
# MAGIC Run tree segmentation on LiDAR data using the pretrained PointGroup3heads model.
# MAGIC
# MAGIC **Requirements:**
# MAGIC - DBR 15.4 LTS GPU Runtime
# MAGIC - MinkowskiEngine installed (see init script)
# MAGIC - Model file uploaded to Unity Catalog Volume

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

# Paths - UPDATE THESE FOR YOUR ENVIRONMENT
MODEL_PATH = "/Volumes/your_catalog/your_schema/models/PointGroup-PAPER.pt"
INPUT_PATH = "/Volumes/your_catalog/your_schema/lidar_data/input"
OUTPUT_PATH = "/Volumes/your_catalog/your_schema/lidar_data/output"

# Processing parameters
GRID_SIZE = 0.2  # Voxel size in meters

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install Dependencies (if needed)

# COMMAND ----------

# Uncomment if MinkowskiEngine not installed
# %pip install ninja
# import subprocess
# subprocess.run(["pip", "install", "git+https://github.com/NVIDIA/MinkowskiEngine.git", "--no-build-isolation"], check=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Model

# COMMAND ----------

import torch
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf

print("Checking GPU...")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

print(f"\nLoading model from {MODEL_PATH}...")
checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
print(f"Checkpoint keys: {list(checkpoint.keys())}")

dataset_props = checkpoint.get("dataset_properties", {})
print(f"Dataset properties: {dataset_props}")

# COMMAND ----------

# Import MinkowskiEngine and model factory
import MinkowskiEngine
print(f"MinkowskiEngine version: {MinkowskiEngine.__version__}")

from torch_points3d.models.model_factory import instantiate_model

# Create dataset mock for model instantiation
class MockDataset:
    def __init__(self, props):
        self.feature_dimension = props.get("feature_dimension", 4)
        self.num_classes = props.get("num_classes", 2)
        self.stuff_classes = torch.tensor([0])  # non-tree
        self.thing_classes = torch.tensor([1])  # tree

dataset = MockDataset(dataset_props)

# Instantiate model
run_config = OmegaConf.create(checkpoint["run_config"])
model = instantiate_model(run_config, dataset)

# Load weights (prefer best_miou)
weights = checkpoint["models"]
if "best_miou" in weights:
    state_dict = weights["best_miou"]
    print("Loading best_miou weights")
elif "latest" in weights:
    state_dict = weights["latest"]
    print("Loading latest weights")
else:
    state_dict = weights[list(weights.keys())[0]]

model.load_state_dict(state_dict, strict=False)
model = model.cuda().eval()

print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define Inference Function

# COMMAND ----------

from torch_geometric.data import Data
from torch_points3d.core.data_transform import GridSampling3D
from plyfile import PlyData, PlyElement

def segment_point_cloud(points: np.ndarray, grid_size: float = 0.2):
    """
    Segment trees in a point cloud.

    Args:
        points: (N, 3) array of XYZ coordinates
        grid_size: Voxel size for grid sampling

    Returns:
        dict with keys: points, semantic_pred, instance_pred, num_trees
    """
    # Normalize coordinates
    offset = points.min(axis=0)
    pos = (points - offset).astype(np.float32)
    center = pos.mean(axis=0)

    # Create features
    features = np.column_stack([
        pos[:, 0] - center[0],
        pos[:, 1] - center[1],
        pos[:, 2] - center[2],
        pos[:, 2]
    ]).astype(np.float32)

    n_points = len(pos)

    # Create Data object with required fields
    data = Data(
        pos=torch.from_numpy(pos),
        x=torch.from_numpy(features),
        # Dummy labels required by model
        center_label=torch.zeros(n_points, 3, dtype=torch.float32),
        y=torch.zeros(n_points, dtype=torch.long),
        num_instances=torch.tensor([0]),
        instance_labels=torch.zeros(n_points, dtype=torch.long),
        instance_mask=torch.zeros(n_points, dtype=torch.bool),
        vote_label=torch.zeros(n_points, 3, dtype=torch.float32),
    )

    # Grid sampling
    sampler = GridSampling3D(size=grid_size, quantize_coords=True, mode="last")
    data = sampler(data)
    data.batch = torch.zeros(data.pos.shape[0], dtype=torch.long)

    # Run inference
    data = data.cuda()
    with torch.no_grad():
        model.set_input(data, torch.device("cuda"))
        model.forward(epoch=999)
        output = model.get_output()

    # Extract predictions
    sem_pred = torch.argmax(output.semantic_logits, dim=1).cpu().numpy()
    instance_pred = np.zeros(len(sem_pred), dtype=np.int32)

    num_trees = 0
    if hasattr(output, "clusters") and output.clusters:
        num_trees = len(output.clusters)
        for i, cl in enumerate(output.clusters):
            if cl is not None:
                instance_pred[cl.cpu().numpy()] = i + 1

    # Restore original coordinates
    out_pos = data.pos.cpu().numpy() + offset

    return {
        "points": out_pos,
        "semantic_pred": sem_pred,
        "instance_pred": instance_pred,
        "num_trees": num_trees
    }


def read_point_cloud(path: str) -> np.ndarray:
    """Read PLY or LAS file."""
    path = Path(path)

    if path.suffix.lower() == ".ply":
        ply = PlyData.read(str(path))
        v = ply["vertex"]
        return np.column_stack([v["x"], v["y"], v["z"]]).astype(np.float32)

    elif path.suffix.lower() in [".las", ".laz"]:
        import laspy
        las = laspy.read(str(path))
        return np.column_stack([las.x, las.y, las.z]).astype(np.float32)

    else:
        raise ValueError(f"Unsupported format: {path.suffix}")


def save_result(result: dict, output_path: str):
    """Save segmentation result as PLY."""
    dtype = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("semantic_seg", "i4"), ("treeID", "i4")
    ]
    arr = np.zeros(len(result["points"]), dtype=dtype)
    arr["x"] = result["points"][:, 0]
    arr["y"] = result["points"][:, 1]
    arr["z"] = result["points"][:, 2]
    arr["semantic_seg"] = result["semantic_pred"]
    arr["treeID"] = result["instance_pred"]

    el = PlyElement.describe(arr, "vertex")
    PlyData([el], text=True).write(str(output_path))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Process Files

# COMMAND ----------

import os
from pathlib import Path

# Find input files
input_path = Path(INPUT_PATH)
output_path = Path(OUTPUT_PATH)
output_path.mkdir(parents=True, exist_ok=True)

# Get all LiDAR files
files = list(input_path.glob("*.ply")) + list(input_path.glob("*.las")) + list(input_path.glob("*.laz"))
print(f"Found {len(files)} files to process")

# Process each file
results_summary = []

for i, file_path in enumerate(files):
    print(f"\n[{i+1}/{len(files)}] Processing: {file_path.name}")

    try:
        # Read points
        points = read_point_cloud(str(file_path))
        print(f"  Loaded {len(points):,} points")

        # Segment
        result = segment_point_cloud(points, grid_size=GRID_SIZE)

        tree_points = (result["semantic_pred"] == 1).sum()
        print(f"  Semantic: {tree_points:,} tree points")
        print(f"  Instances: {result['num_trees']} trees detected")

        # Save
        out_file = output_path / f"{file_path.stem}_segmented.ply"
        save_result(result, str(out_file))
        print(f"  Saved: {out_file.name}")

        results_summary.append({
            "file": file_path.name,
            "input_points": len(points),
            "output_points": len(result["points"]),
            "tree_points": int(tree_points),
            "num_trees": result["num_trees"],
            "status": "success"
        })

    except Exception as e:
        print(f"  ERROR: {e}")
        results_summary.append({
            "file": file_path.name,
            "status": "failed",
            "error": str(e)
        })

# COMMAND ----------

# MAGIC %md
# MAGIC ## Results Summary

# COMMAND ----------

import pandas as pd

df = pd.DataFrame(results_summary)
display(df)

# COMMAND ----------

# Summary statistics
total_trees = df[df["status"] == "success"]["num_trees"].sum()
total_points = df[df["status"] == "success"]["output_points"].sum()
successful = (df["status"] == "success").sum()

print(f"Processed: {successful}/{len(df)} files")
print(f"Total trees detected: {total_trees:,}")
print(f"Total points processed: {total_points:,}")
print(f"Output location: {OUTPUT_PATH}")
