# Devcontainer for LiDAR Panoptic Segmentation
# Matches Databricks Runtime 15.4 LTS GPU environment
# - CUDA 12.1
# - Python 3.10
# - PyTorch 2.1.0

ARG CUDA_VERSION=12.1.0
ARG PYTHON_VERSION=3.10

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04

ARG PYTHON_VERSION

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    curl \
    git \
    libboost-all-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ninja-build \
    libopenblas-dev \
    pkg-config \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python${PYTHON_VERSION}-venv \
    software-properties-common \
    sudo \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Miniforge (no ToS requirement, uses conda-forge)
ENV CONDA_DIR=/opt/conda
RUN wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh \
    && bash /tmp/miniforge.sh -b -p ${CONDA_DIR} \
    && rm /tmp/miniforge.sh \
    && ${CONDA_DIR}/bin/conda clean -afy

ENV PATH="${CONDA_DIR}/bin:${PATH}"

# Create conda environment matching Databricks 15.4 LTS
RUN conda create -n lidar-panoptic python=${PYTHON_VERSION} -y \
    && conda clean -afy

# Activate environment by default
ENV CONDA_DEFAULT_ENV=lidar-panoptic
ENV PATH="${CONDA_DIR}/envs/lidar-panoptic/bin:${PATH}"

# Install PyTorch with CUDA 12.1 support (matches Databricks 15.4 LTS)
RUN pip install --no-cache-dir \
    torch==2.1.0+cu121 \
    torchvision==0.16.0+cu121 \
    torchaudio==2.1.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# Install core dependencies (NumPy <2.0 for PyTorch 2.1 compatibility)
RUN pip install --no-cache-dir \
    "numpy>=1.24.0,<2.0.0" \
    scipy>=1.10.0 \
    scikit-learn>=1.2.0 \
    pandas>=2.0.0 \
    pyyaml>=6.0 \
    pydantic>=2.0.0 \
    tqdm>=4.65.0 \
    shapely>=2.0.0 \
    pyarrow>=12.0.0 \
    hdbscan>=0.8.29 \
    laspy>=2.5.0 \
    plyfile>=0.9.0 \
    mlflow>=2.9.0 \
    azure-storage-blob>=12.0.0 \
    azure-identity>=1.14.0 \
    databricks-sdk>=0.12.0

# Install development dependencies
RUN pip install --no-cache-dir \
    pytest>=7.0.0 \
    pytest-cov>=4.0.0 \
    ruff>=0.1.0 \
    mypy>=1.0.0 \
    ipykernel>=6.0.0 \
    jupyterlab>=4.0.0

# Create non-root user for VS Code
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Set CUDA environment variables
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
ENV PATH="${CUDA_HOME}/bin:${PATH}"

# MinkowskiEngine build environment
ENV MAX_JOBS=4
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0"

WORKDIR /workspace

USER $USERNAME
