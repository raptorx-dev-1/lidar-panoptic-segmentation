# LiDAR Panoptic Segmentation

> **Refactored from SegmentAnyTree** - A modular, configuration-driven, enterprise-ready deep learning system for 3D panoptic segmentation and tree polygon extraction.

Designed for production deployment on Azure Databricks 15.4 LTS GPU (single-node) with Unity Catalog, MLflow, and secure installation of MinkowskiEngine without init scripts or Docker.

## Features

- **Unified YAML Configuration**: Single `config.yaml` drives all runtime behavior
- **Unity Catalog Support**: Works with `abfss://` paths for secure data access
- **MLflow Integration**: Full experiment tracking and model registry
- **Modular Architecture**: Clean separation of concerns for maintainability
- **No Docker Required**: Conda-based local development, notebook-based Databricks deployment
- **Production Ready**: Robust error handling, logging, and monitoring

## Quick Start

### VS Code Devcontainer (Recommended)

The easiest way to get started with GPU support matching Databricks 15.4 LTS:

1. **Install Prerequisites**:
   - [VS Code](https://code.visualstudio.com/)
   - [Docker](https://www.docker.com/products/docker-desktop/)
   - [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
   - [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)

2. **Open in Container**:
   - Open this folder in VS Code
   - Click "Reopen in Container" when prompted (or use Command Palette: `Dev Containers: Reopen in Container`)
   - Wait for the container to build and setup

3. **Start Development**:
   ```bash
   # Tests run automatically, but you can re-run
   pytest tests/ -v

   # Train a model
   python -m lidar_panoptic_segmentation.train --config config.yaml
   ```

### Local Development (Conda)

```bash
# Create conda environment
conda env create -f environment.yml
conda activate lidar-panoptic-segmentation

# Install the package
pip install -e .

# Run tests
pytest tests/ -v

# Train a model
python -m lidar_panoptic_segmentation.train --config config.yaml

# Run inference
python -m lidar_panoptic_segmentation.infer \
    --config config.yaml \
    --input ./data/test.las \
    --output ./output/
```

### Azure Databricks

1. **Create Cluster**: Use Databricks Runtime 15.4 LTS GPU (single-node)

2. **Install MinkowskiEngine** (run in notebook):
   ```python
   # Cell 1: Get PAT from secrets
   ADO_PAT = dbutils.secrets.get("azure", "ado_pat")
   with open("/tmp/ado_pat", "w") as f:
       f.write(ADO_PAT)
   ```
   ```bash
   # Cell 2: Run installer
   %sh
   export ADO_PAT=$(cat /tmp/ado_pat)
   export ME_REPO="https://github.com/NVIDIA/MinkowskiEngine.git"
   export ME_REF="master"
   bash ./scripts/install_minkowski.sh
   ```

3. **Configure Paths** (update `config.yaml`):
   ```yaml
   paths:
     data_root: abfss://forest-data@youraccount.dfs.core.windows.net/
     models: /Volumes/catalog/schema/models
   ```

4. **Run Training or Inference**: Use the demo notebook or CLI

## Project Structure

```
lidar-panoptic-segmentation/
├── .devcontainer/                   # VS Code devcontainer config
│   ├── devcontainer.json            # Container configuration
│   ├── Dockerfile                   # Databricks 15.4 LTS GPU compatible
│   └── post-create.sh               # Setup script
├── lidar_panoptic_segmentation/     # Main package
│   ├── __init__.py
│   ├── config.py                    # Configuration loader
│   ├── config_schema.py             # Pydantic schemas
│   ├── dataset.py                   # UC-aware data handling
│   ├── model.py                     # Model architectures
│   ├── train.py                     # Training CLI
│   ├── infer.py                     # Inference CLI
│   ├── postprocess.py               # Polygon extraction
│   ├── logging_utils.py             # MLflow + WandB
│   └── utils.py                     # Utilities
├── scripts/
│   └── install_minkowski.sh         # MinkowskiEngine installer
├── tests/
│   ├── test_config.py
│   ├── test_infer.py
│   └── test_postprocess.py
├── notebooks/
│   └── LidarPanopticSegmentation_Demo.ipynb
├── config.yaml                      # Main configuration
├── environment.yml                  # Conda environment
├── cluster_config_guidance.md       # Databricks cluster setup
├── README_Model_Improvements.md     # Accuracy optimization
└── README.md                        # This file
```

## Configuration

The system is driven by a single `config.yaml` file:

```yaml
env:
  name: databricks  # local | dev | databricks
  debug: false

paths:
  data_root: abfss://container@account.dfs.core.windows.net/
  train:
    pointclouds: ${paths.data_root}/train/las
    labels: ${paths.data_root}/train/labels
  inference:
    input: ${paths.data_root}/inference/input
    output: ${paths.data_root}/inference/output

training:
  model_name: PointGroup3heads
  epochs: 60
  batch_size: 8
  lr: 0.0001

inference:
  model_uri: models:/LidarPanopticSegmentation/latest
  polygon_output_format: geojson

logging:
  mlflow:
    enabled: true
    experiment_name: LidarPanopticSegmentation_Experiments
```

See `config.yaml` for the complete configuration reference.

## Data Formats

### Input
- LAS/LAZ files (recommended)
- PLY files with semantic and instance labels

### Output
- GeoJSON polygons (tree crowns)
- Shapefile export
- Parquet format
- Segmented point clouds (PLY/LAS)

## Training

```bash
# Full training
python -m lidar_panoptic_segmentation.train \
    --config config.yaml \
    --epochs 60 \
    --batch-size 8

# Resume training
python -m lidar_panoptic_segmentation.train \
    --config config.yaml \
    --resume ./models/checkpoint_latest.pt

# Validation only
python -m lidar_panoptic_segmentation.train \
    --config config.yaml \
    --validate-only
```

## Inference

```bash
# Single file
python -m lidar_panoptic_segmentation.infer \
    --input ./data/forest.las \
    --output ./predictions/

# Directory (batch processing)
python -m lidar_panoptic_segmentation.infer \
    --input ./data/lidar/ \
    --output ./predictions/ \
    --recursive

# With specific model
python -m lidar_panoptic_segmentation.infer \
    --model models:/LidarPanopticSegmentation/production \
    --input ./data/forest.las
```

## MLflow Integration

All training runs are tracked in MLflow:

```python
import mlflow

# View experiments
mlflow.search_experiments()

# Load registered model
model = mlflow.pytorch.load_model("models:/LidarPanopticSegmentation/latest")

# Load specific run artifacts
run = mlflow.get_run("run_id")
config = run.data.params
metrics = run.data.metrics
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_config.py -v

# Run with coverage
pytest tests/ --cov=lidar_panoptic_segmentation --cov-report=html
```

## API Reference

### Configuration

```python
from lidar_panoptic_segmentation.config import load_config, validate_config

config = load_config("config.yaml")
warnings = validate_config(config)
```

### Model

```python
from lidar_panoptic_segmentation.model import create_model, load_model

model = create_model(config)
model = load_model("models:/LidarPanopticSegmentation/latest")
```

### Dataset

```python
from lidar_panoptic_segmentation.dataset import create_dataloader, read_point_cloud

train_loader = create_dataloader(config, split="train")
data = read_point_cloud("forest.las")
```

### Postprocessing

```python
from lidar_panoptic_segmentation.postprocess import (
    postprocess_predictions,
    save_geojson,
)

result = postprocess_predictions(points, semantic_pred, config=config)
save_geojson(result.instances, "trees.geojson")
```

## Requirements

- Python 3.10+
- PyTorch 2.1.0+ with CUDA 12.1
- MinkowskiEngine (GPU builds from source)
- See `environment.yml` for complete list

## Documentation

- [Cluster Configuration Guide](cluster_config_guidance.md) - Databricks setup
- [Model Improvements Guide](README_Model_Improvements.md) - Accuracy optimization
- [Demo Notebook](notebooks/LidarPanopticSegmentation_Demo.ipynb) - Interactive examples

## Legacy Usage (Docker)

For the original Docker-based workflow, see the legacy instructions below:

<details>
<summary>Original SegmentAnyTree Docker Usage</summary>

```bash
mkdir -p $HOME/segmentanytree/input
mkdir -p $HOME/segmentanytree/output

docker pull maciekwielgosz/segment-any-tree:latest

docker run -it --rm --gpus all \
  --mount type=bind,source=$HOME/segmentanytree/input,target=/home/nibio/mutable-outside-world/bucket_in_folder \
  --mount type=bind,source=$HOME/segmentanytree/output,target=/home/nibio/mutable-outside-world/bucket_out_folder \
  maciekwielgosz/segment-any-tree:latest
```

</details>

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

## License

See [LICENSE](LICENSE) for details.

## Acknowledgments

This project builds on:
- [SegmentAnyTree](https://github.com/maciekwielgosz/SegmentAnyTree) - Original implementation
- [torch-points3d](https://github.com/torch-points3d/torch-points3d) - Deep learning framework
- [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine) - Sparse tensor operations
- [PointGroup](https://github.com/dvlab-research/PointGroup) - Panoptic architecture

## Issues

If you encounter any issues with the code please provide your feedback by raising an issue in this repo!
