#!/bin/bash
# Databricks Cluster Init Script for MinkowskiEngine
#
# Usage: Add this script to your cluster init scripts in Databricks
# Cluster Settings > Advanced Options > Init Scripts
#
# Requirements:
# - DBR 15.4 LTS GPU Runtime (CUDA 12.1, Python 3.10)
# - GPU instance (e.g., Standard_NC6s_v3)

set -e

echo "=== Installing MinkowskiEngine for SegmentAnyTree ==="

# Check if GPU is available
if ! nvidia-smi &> /dev/null; then
    echo "WARNING: No GPU detected. MinkowskiEngine requires CUDA."
    exit 0
fi

echo "GPU detected:"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

# Install build dependencies
echo "Installing build dependencies..."
apt-get update -qq
apt-get install -y -qq libopenblas-dev ninja-build

# Install Python dependencies
echo "Installing Python dependencies..."
/databricks/python/bin/pip install --quiet ninja wheel setuptools

# Install MinkowskiEngine
echo "Installing MinkowskiEngine (this may take 5-10 minutes)..."
export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0"
export MAX_JOBS=4

/databricks/python/bin/pip install \
    git+https://github.com/NVIDIA/MinkowskiEngine.git \
    --no-build-isolation \
    --quiet

# Verify installation
echo "Verifying installation..."
/databricks/python/bin/python -c "
import MinkowskiEngine
import torch
print(f'MinkowskiEngine {MinkowskiEngine.__version__} installed successfully')
print(f'PyTorch {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
"

echo "=== MinkowskiEngine installation complete ==="
