#!/usr/bin/env bash
#
# install_minkowski.sh
#
# End-to-end installer for MinkowskiEngine (CUDA 12.1) on Azure Databricks GPU clusters.
# Designed for notebook-based installation without init scripts or Docker.
#
# Key features:
# - Uses Azure DevOps PAT (ADO_PAT) via HTTP Authorization header (no interactive prompts)
# - Validates pinned commit; falls back to branch/tag if commit is missing
# - Installs NVCC (CUDA Toolkit 12.1) when needed
# - Vendors CPython-3.11 stdlib `distutils` for numpy.distutils compatibility
# - Pins setuptools (<60.0) for numpy.distutils
# - Applies CUDA 12.x Thrust header patches
# - Configures TORCH_CUDA_ARCH_LIST for V100/A10/A100 and suppresses nvcc warnings
# - Builds MinkowskiEngine with pip (no `setup.py install`)
# - Runs a CUDA smoke test
#
# Usage (Databricks notebook):
#   # Cell 1: Get PAT from secrets
#   ADO_PAT = dbutils.secrets.get("azure", "ado_pat")
#   with open("/tmp/ado_pat", "w") as f:
#       f.write(ADO_PAT)
#
#   # Cell 2: Run installer
#   %sh
#   export ADO_PAT=$(cat /tmp/ado_pat)
#   export ME_REPO="https://dev.azure.com/org/project/_git/MinkowskiEngine"
#   export ME_REF="main"
#   bash ./scripts/install_minkowski.sh
#
# Environment Variables:
#   CPU_ONLY=1                       # Build ME without CUDA (fallback)
#   DBFS_WHEEL_PATH=/local_disk0/... # If set, build & copy a wheel there
#   TORCH_CUDA_ARCH_LIST="7.0"       # Override auto-detected arch list
#   ADO_PAT=<PAT>                    # Azure DevOps PAT (use secrets; do NOT hard-code)
#   ME_REPO=...                      # Azure DevOps or GitHub repo URL
#   ME_REF=main                      # Branch/tag that exists in your repo
#   ME_COMMIT=<sha>                  # Pin SHA only if present in your repo
#   INSTALL_PREFIX=/databricks/driver# Install root (falls back to /tmp)
#   WORKDIR=/databricks/driver/MinkowskiEngine # Clone/build directory
#   MAX_JOBS=$(nproc)                # Parallel build jobs
#
set -euo pipefail

# ==============================================================================
# Logging Functions
# ==============================================================================
log(){ echo -e "\033[1;32m[ME-Install]\033[0m $*"; }
warn(){ echo -e "\033[1;33m[ME-Install][WARN]\033[0m $*"; }
err(){ echo -e "\033[1;31m[ME-Install][ERR]\033[0m $*" >&2; }

# ==============================================================================
# Configuration (Environment Defaults)
# ==============================================================================
ME_REPO="${ME_REPO:-https://github.com/NVIDIA/MinkowskiEngine.git}"
ME_REF="${ME_REF:-master}"
ME_COMMIT="${ME_COMMIT:-}"
INSTALL_PREFIX="${INSTALL_PREFIX:-/databricks/driver}"
[[ -d "$INSTALL_PREFIX" ]] || INSTALL_PREFIX="/tmp"
WORKDIR="${WORKDIR:-$INSTALL_PREFIX/MinkowskiEngine}"
MAX_JOBS="${MAX_JOBS:-$(nproc)}"
CPU_ONLY="${CPU_ONLY:-0}"

log "MinkowskiEngine Installer"
log "========================="
log "Repository: $ME_REPO"
log "Reference: $ME_REF"
log "Install Prefix: $INSTALL_PREFIX"
log "Work Directory: $WORKDIR"

# ==============================================================================
# Stage 0: Python / PyTorch Diagnostics
# ==============================================================================
log "Stage 0: Python / PyTorch diagnostics"
python - <<'PY' || { echo "Python unavailable"; exit 1; }
import sys, torch
print("  Python:", sys.version)
print("  PyTorch:", torch.__version__)
print("  PyTorch CUDA:", torch.version.cuda)
print("  CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("  Device[0]:", torch.cuda.get_device_name(0))
    print("  Capability:", torch.cuda.get_device_capability(0))
PY

# ==============================================================================
# Stage 1: Ensure CUDA Toolkit 12.1 (NVCC) is Installed
# ==============================================================================
log "Stage 1: Ensure CUDA Toolkit 12.1 (NVCC) is installed"
if ! command -v nvcc >/dev/null 2>&1 && [[ "$CPU_ONLY" != "1" ]]; then
  log "  Installing CUDA Toolkit 12.1..."
  wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
  sudo dpkg -i cuda-keyring_1.1-1_all.deb
  sudo apt-get update -y
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y cuda-toolkit-12-1
fi

# Remove duplicate CUDA repo entry to silence apt warnings
DUP_REPO="/etc/apt/sources.list.d/archive_uri-https_developer_download_nvidia_com_compute_cuda_repos_ubuntu2204_x86_64_-jammy.list"
if [[ -f "$DUP_REPO" ]]; then
  sudo rm -f "$DUP_REPO"
  sudo apt-get update -y || true
fi

# Export CUDA paths
if [[ -d /usr/local/cuda-12.1 ]]; then
  export CUDA_HOME="/usr/local/cuda-12.1"
elif [[ -d /usr/local/cuda ]]; then
  export CUDA_HOME="/usr/local/cuda"
fi
export PATH="${CUDA_HOME:+$CUDA_HOME/bin:}$PATH"
export LD_LIBRARY_PATH="${CUDA_HOME:+$CUDA_HOME/lib64:}${LD_LIBRARY_PATH:-}"
log "  CUDA_HOME=${CUDA_HOME:-<unset>} (nvcc: $(command -v nvcc || echo 'not found'))"

# ==============================================================================
# Stage 2: Build Dependencies + Pip Toolchain
# ==============================================================================
log "Stage 2: Build dependencies (OpenBLAS, g++, ninja, git) + pip toolchain"
sudo apt-get update -y
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
    libopenblas-dev build-essential g++ ninja-build git python3-dev
python -m pip install --upgrade pip
# Pin setuptools <60.0 for numpy.distutils compatibility
python -m pip install --upgrade "setuptools<60.0" wheel "ninja>=1.10"

# Ensure Python headers match the interpreter
PYVER="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [[ "$PYVER" == "3.11" ]]; then
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y python3.11-dev
elif [[ "$PYVER" == "3.10" ]]; then
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y python3.10-dev
else
  warn "Python $PYVER detected; ensure matching dev headers are installed."
fi

# ==============================================================================
# Stage 3: Vendor CPython stdlib 'distutils' (if missing)
# ==============================================================================
log "Stage 3: Vendor CPython stdlib 'distutils' (if missing)"
python - <<'PY' || true
import sys, sysconfig, os, urllib.request, tarfile, shutil, importlib

def vendor_distutils(target_site):
    ver = ".".join(map(str, sys.version_info[:3]))
    base_url = f"https://www.python.org/ftp/python/{ver}/Python-{ver}.tar.xz"
    tmp = "/tmp/cpython-src.tar.xz"
    print(f"  Vendoring distutils for Python {ver} → {target_site}")
    urllib.request.urlretrieve(base_url, tmp)
    with tarfile.open(tmp, "r:xz") as tf:
        members = [m for m in tf.getmembers() if m.name.startswith(f"Python-{ver}/Lib/distutils/")]
        tf.extractall(path="/tmp", members=members)
    src = f"/tmp/Python-{ver}/Lib/distutils"
    dst = os.path.join(target_site, "distutils")
    if os.path.exists(dst): shutil.rmtree(dst)
    shutil.copytree(src, dst)
    return dst

try:
    importlib.import_module("distutils.msvccompiler")
    print("  distutils.msvccompiler is available")
except Exception:
    dstdir = vendor_distutils(sysconfig.get_paths()["platlib"])
    importlib.import_module("distutils.msvccompiler")
    print("  OK: distutils vendored at", dstdir)

# Mirror vendored distutils to Databricks global site-packages (defensive)
platlib = sysconfig.get_paths()["platlib"]
global_site = "/databricks/python/lib/python3.11/site-packages"
if os.path.exists(global_site) and os.path.realpath(platlib) != os.path.realpath(global_site):
    src = os.path.join(platlib, "distutils")
    dst = os.path.join(global_site, "distutils")
    if os.path.isdir(src):
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        print("  Mirrored distutils to", dst)
PY

# Prioritize vendored site-packages on sys.path
export PYTHONPATH="/local_disk0/.ephemeral_nfs/cluster_libraries/python/lib/python3.11/site-packages:${PYTHONPATH:-}"

# ==============================================================================
# Stage 4: Clone or Update MinkowskiEngine & Apply CUDA 12.x Thrust Patches
# ==============================================================================
log "Stage 4: Clone or update MinkowskiEngine and apply CUDA 12.x Thrust patches"

# Auth: use Azure DevOps PAT via Basic header (username blank, password = PAT)
if [[ -z "${ADO_PAT:-}" ]]; then
  log "  ADO_PAT not set; using public repo or cached credentials."
fi
AUTH_HEADER=""
if [[ -n "${ADO_PAT:-}" ]]; then
  AUTH_HEADER=$(printf ":%s" "$ADO_PAT" | base64 -w0)
fi

if [[ -d "$WORKDIR" ]]; then
  log "  Using existing $WORKDIR"
  cd "$WORKDIR"
  git remote set-url origin "$ME_REPO" || true
  if [[ -n "$AUTH_HEADER" ]]; then
    git -c "http.extraheader=Authorization: Basic $AUTH_HEADER" fetch --all --tags --prune || true
  else
    git fetch --all --tags --prune || true
  fi
else
  log "  Cloning $ME_REPO → $WORKDIR"
  if [[ -n "$AUTH_HEADER" ]]; then
    git -c "http.extraheader=Authorization: Basic $AUTH_HEADER" clone "$ME_REPO" "$WORKDIR"
  else
    git clone "$ME_REPO" "$WORKDIR"
  fi
  cd "$WORKDIR"
  if [[ -n "$AUTH_HEADER" ]]; then
    git -c "http.extraheader=Authorization: Basic $AUTH_HEADER" fetch --all --tags --prune || true
  else
    git fetch --all --tags --prune || true
  fi
fi

# Choose ref to checkout: prefer valid commit; else ME_REF; else stay on HEAD
if [[ -n "${ME_COMMIT:-}" ]] && git cat-file -t "$ME_COMMIT" 2>/dev/null | grep -q '^commit'; then
  log "  Checking out commit $ME_COMMIT"
  git checkout --detach "$ME_COMMIT"
elif [[ -n "$ME_REF" ]] && git rev-parse --verify "origin/$ME_REF" >/dev/null 2>&1; then
  log "  Checking out branch/tag: $ME_REF"
  git checkout "$ME_REF"
else
  HEAD_REF=$(git symbolic-ref -q --short HEAD || git rev-parse --short HEAD)
  warn "  Commit/Ref not found; staying on current HEAD ($HEAD_REF)"
fi

# Apply CUDA 12.x Thrust patches if needed
py_cuda_ver=$(python - <<'PY'
import torch; print(torch.version.cuda or "")
PY
)
CUDA_MAJOR="${py_cuda_ver%%.*}"
if [[ -n "$py_cuda_ver" && "$CUDA_MAJOR" -ge 12 && "$CPU_ONLY" != "1" ]]; then
  log "  Applying CUDA 12.x Thrust header patches"
  add_if_missing() {
    local f="$1" h="$2"
    if [[ -f "$f" ]]; then
      grep -qF "$h" "$f" || sed -i "1i $h" "$f"
    fi
  }
  add_if_missing "src/3rdparty/concurrent_unordered_map.cuh" "#include <thrust/execution_policy.h>"
  add_if_missing "src/convolution_kernel.cuh"                "#include <thrust/execution_policy.h>"
  add_if_missing "src/coordinate_map_gpu.cu"                 "#include <thrust/unique.h>"
  add_if_missing "src/coordinate_map_gpu.cu"                 "#include <thrust/remove.h>"
  add_if_missing "src/spmm.cu"                               "#include <thrust/execution_policy.h>"
  add_if_missing "src/spmm.cu"                               "#include <thrust/reduce.h>"
  add_if_missing "src/spmm.cu"                               "#include <thrust/sort.h>"
else
  log "  CUDA < 12 or CPU-only; no Thrust patches needed."
fi

# ==============================================================================
# Stage 5: Configure TORCH_CUDA_ARCH_LIST and nvcc Warnings
# ==============================================================================
log "Stage 5: Configure TORCH_CUDA_ARCH_LIST (major.minor) and nvcc warnings"
if [[ -z "${TORCH_CUDA_ARCH_LIST:-}" && "$CPU_ONLY" != "1" ]]; then
  CAP=$(python - <<'PY'
import torch
maj,min = (7,0) if not torch.cuda.is_available() else torch.cuda.get_device_capability(0)
print(f"{maj}.{min}")
PY
  )
  export TORCH_CUDA_ARCH_LIST="$CAP"
fi
export TORCH_NVCC_FLAGS='-Xcudafe --diag_suppress=177'
log "  TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-<unset>}  TORCH_NVCC_FLAGS=${TORCH_NVCC_FLAGS}"

# ==============================================================================
# Stage 6: Clean & Build MinkowskiEngine with pip
# ==============================================================================
log "Stage 6: Clean & build MinkowskiEngine with pip (no build isolation)"
cd "$WORKDIR"
python setup.py clean || true
rm -rf build/ MinkowskiEngine.egg-info/ dist/
python -m pip install --no-build-isolation .

# ==============================================================================
# Stage 7: Verify MinkowskiEngine Diagnostics & CUDA Smoke Test
# ==============================================================================
log "Stage 7: Verify MinkowskiEngine diagnostics & CUDA smoke test"
python - <<'PY'
import torch, MinkowskiEngine as ME

# Print compile/runtime diagnostics
ME.print_diagnostics()

# Target device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Batched coordinates (D+1 format; batch index + 3D spatial dims)
coords = ME.utils.batched_coordinates(
    [torch.randint(0, 32, (1024, 3), dtype=torch.int32)]
)

# IMPORTANT: move coords/features to GPU BEFORE SparseTensor init
if torch.cuda.is_available():
    coords = coords.to(device)

feats = torch.randn((coords.shape[0], 4), dtype=torch.float32, device=device)

# Construct SparseTensor directly on the target device
x = ME.SparseTensor(features=feats, coordinates=coords, device=device)

# Minkowski modules are nn.Modules → .to(device) is fine
conv = ME.MinkowskiConvolution(
    in_channels=4, out_channels=8,
    kernel_size=3, stride=1, dimension=3, bias=False
).to(device)

y = conv(x)
print("Smoke test OK — output:", y.F.shape, "device:", y.F.device)
PY

# ==============================================================================
# Stage 8 (Optional): Package a Wheel for Reuse
# ==============================================================================
log "Stage 8 (optional): Package a wheel for reuse"
if [[ -n "${DBFS_WHEEL_PATH:-}" ]]; then
  python -m pip install --upgrade build
  python -m build
  mkdir -p "$(dirname "$DBFS_WHEEL_PATH")"
  cp dist/*.whl "$DBFS_WHEEL_PATH"
  log "  Wheel copied to $DBFS_WHEEL_PATH"
fi

log "All done — MinkowskiEngine installed successfully."
log ""
log "To verify installation in your notebook, run:"
log "  import MinkowskiEngine as ME"
log "  ME.print_diagnostics()"
