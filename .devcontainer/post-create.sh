#!/bin/bash
# Post-create setup script for devcontainer
# Installs the package and optionally MinkowskiEngine

set -e

echo "=============================================="
echo "LiDAR Panoptic Segmentation - Devcontainer Setup"
echo "=============================================="

# Ensure we're in the workspace
cd /workspace

# Install the package in editable mode
echo ""
echo "[1/4] Installing lidar-panoptic-segmentation package..."
pip install -e .

# Verify PyTorch and CUDA
echo ""
echo "[2/4] Verifying PyTorch and CUDA..."
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}' if torch.cuda.is_available() else '')"

# Install MinkowskiEngine (optional - can be slow)
echo ""
echo "[3/4] MinkowskiEngine installation..."
if [ "${INSTALL_MINKOWSKI:-false}" = "true" ]; then
    echo "Installing MinkowskiEngine from source (this may take 10-20 minutes)..."

    # Set build environment
    export MAX_JOBS=4
    export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0"

    # Clone and install
    git clone --depth 1 https://github.com/NVIDIA/MinkowskiEngine.git /tmp/MinkowskiEngine
    cd /tmp/MinkowskiEngine
    pip install -v --no-deps .
    cd /workspace
    rm -rf /tmp/MinkowskiEngine

    echo "MinkowskiEngine installed successfully!"
else
    echo "Skipping MinkowskiEngine installation."
    echo "To install, run: INSTALL_MINKOWSKI=true bash .devcontainer/post-create.sh"
    echo "Or run: bash scripts/install_minkowski.sh"
fi

# Run tests to verify setup
echo ""
echo "[4/4] Running tests to verify setup..."
pytest tests/ -v --tb=short || echo "Some tests may have failed - check output above"

echo ""
echo "=============================================="
echo "Setup complete!"
echo ""
echo "Quick start:"
echo "  python -m lidar_panoptic_segmentation.train --config config.yaml"
echo "  python -m lidar_panoptic_segmentation.infer --input data/ --output output/"
echo ""
echo "For Databricks Unity Catalog paths, update config.yaml:"
echo "  paths.data_root: abfss://container@account.dfs.core.windows.net/"
echo "=============================================="
