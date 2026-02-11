#!/bin/bash
# Test native inference using Docker
# Usage: ./scripts/test_inference_docker.sh [input_file_or_dir] [output_dir]

set -e

INPUT="${1:-data/sample/test/las}"
OUTPUT="${2:-test_output}"

echo "=== SegmentAnyTree Inference Test ==="
echo "Input: $INPUT"
echo "Output: $OUTPUT"
echo ""

# Create output directory
mkdir -p "$OUTPUT"

# Run inference in Docker
docker run --rm --gpus all \
  --entrypoint bash \
  -v "$(pwd)":/workspace \
  -v "$(realpath $INPUT)":/input \
  -v "$(realpath $OUTPUT)":/output \
  maciekwielgosz/segment-any-tree:latest \
  -c '
# Fix package versions
pip3 install -q scikit-learn==1.0.2 pandas==1.5.3 numpy==1.23.5 2>/dev/null

cd /workspace
export PYTHONPATH=/workspace:$PYTHONPATH

python3 << "PYEOF"
import sys
sys.path.insert(0, "/workspace")

import torch
import numpy as np
from pathlib import Path
from glob import glob
from omegaconf import OmegaConf

print("Loading model...")
checkpoint = torch.load("/workspace/model_file/PointGroup-PAPER.pt", map_location="cpu")
dataset_props = checkpoint.get("dataset_properties", {})

from torch_points3d.models.model_factory import instantiate_model

class MockDataset:
    def __init__(self):
        self.feature_dimension = dataset_props.get("feature_dimension", 4)
        self.num_classes = dataset_props.get("num_classes", 2)
        self.stuff_classes = torch.tensor([0])
        self.thing_classes = torch.tensor([1])

model = instantiate_model(OmegaConf.create(checkpoint["run_config"]), MockDataset())
model.load_state_dict(checkpoint["models"].get("best_miou", checkpoint["models"]["latest"]), strict=False)
model = model.cuda().eval()
print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")

# Find input files
from plyfile import PlyData, PlyElement
from torch_geometric.data import Data
from torch_points3d.core.data_transform import GridSampling3D

input_files = list(Path("/input").glob("*.ply")) + list(Path("/input").glob("*.las"))
if not input_files:
    # Check if /input is a file
    if Path("/input").is_file():
        input_files = [Path("/input")]
    else:
        print("No .ply or .las files found in input directory")
        sys.exit(1)

print(f"Found {len(input_files)} files to process")

for input_file in input_files:
    print(f"\nProcessing: {input_file.name}")

    # Read file
    if input_file.suffix.lower() == ".ply":
        ply = PlyData.read(str(input_file))
        v = ply["vertex"]
        pos = np.column_stack([v["x"], v["y"], v["z"]]).astype(np.float32)
    else:
        import laspy
        las = laspy.read(str(input_file))
        pos = np.column_stack([las.x, las.y, las.z]).astype(np.float32)

    print(f"  Loaded {len(pos):,} points")

    # Preprocess
    offset = pos.min(axis=0)
    pos_norm = pos - offset
    center = pos_norm.mean(axis=0)
    features = np.column_stack([
        pos_norm[:, 0] - center[0],
        pos_norm[:, 1] - center[1],
        pos_norm[:, 2] - center[2],
        pos_norm[:, 2]
    ]).astype(np.float32)

    n_points = len(pos_norm)
    data = Data(
        pos=torch.from_numpy(pos_norm),
        x=torch.from_numpy(features),
        center_label=torch.zeros(n_points, 3, dtype=torch.float32),
        y=torch.zeros(n_points, dtype=torch.long),
        num_instances=torch.tensor([0]),
        instance_labels=torch.zeros(n_points, dtype=torch.long),
        instance_mask=torch.zeros(n_points, dtype=torch.bool),
        vote_label=torch.zeros(n_points, 3, dtype=torch.float32),
    )

    sampler = GridSampling3D(size=0.2, quantize_coords=True, mode="last")
    data = sampler(data)
    data.batch = torch.zeros(data.pos.shape[0], dtype=torch.long)
    print(f"  After grid sampling: {data.pos.shape[0]:,} points")

    # Inference
    data = data.cuda()
    with torch.no_grad():
        model.set_input(data, torch.device("cuda"))
        model.forward(epoch=999)
        output = model.get_output()

    # Get predictions
    sem_pred = torch.argmax(output.semantic_logits, dim=1).cpu().numpy()
    instance_pred = np.zeros(len(sem_pred), dtype=np.int32)

    n_trees = 0
    if hasattr(output, "clusters") and output.clusters:
        n_trees = len(output.clusters)
        for i, cl in enumerate(output.clusters):
            if cl is not None:
                instance_pred[cl.cpu().numpy()] = i + 1

    tree_points = (sem_pred == 1).sum()
    print(f"  Semantic: {tree_points:,} tree points, {len(sem_pred) - tree_points:,} non-tree")
    print(f"  Instances: {n_trees} trees detected")

    # Save output
    out_pos = data.pos.cpu().numpy() + offset
    dtype = [("x", "f4"), ("y", "f4"), ("z", "f4"), ("semantic_seg", "i4"), ("treeID", "i4")]
    arr = np.zeros(len(out_pos), dtype=dtype)
    arr["x"], arr["y"], arr["z"] = out_pos[:, 0], out_pos[:, 1], out_pos[:, 2]
    arr["semantic_seg"], arr["treeID"] = sem_pred, instance_pred

    out_file = Path("/output") / f"{input_file.stem}_segmented.ply"
    PlyData([PlyElement.describe(arr, "vertex")], text=True).write(str(out_file))
    print(f"  Saved: {out_file.name}")

print("\n=== Done ===")
PYEOF
'

echo ""
echo "Output files:"
ls -la "$OUTPUT"/*.ply 2>/dev/null || echo "No output files found"
