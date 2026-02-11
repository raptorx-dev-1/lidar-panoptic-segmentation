"""
Pretrained Inference Pipeline

Runs inference using the original SegmentAnyTree pretrained model.
Integrates with the new config system for Unity Catalog path support.
"""

import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch

from .config import load_config
from .config_schema import Config

logger = logging.getLogger(__name__)


class PretrainedModelError(Exception):
    """Raised when there's an issue with the pretrained model."""
    pass


def check_model_file(model_path: Path) -> bool:
    """Check if the model file is a real checkpoint or a Git LFS pointer."""
    if not model_path.exists():
        return False

    # Check file size - LFS pointers are small text files
    if model_path.stat().st_size < 1000:
        with open(model_path, 'r') as f:
            try:
                content = f.read(100)
                if 'version https://git-lfs.github.com' in content:
                    return False  # It's an LFS pointer
            except UnicodeDecodeError:
                pass  # Binary file, probably real model

    return True


def download_pretrained_model(
    model_dir: Path,
    model_name: str = "PointGroup-PAPER",
) -> Path:
    """
    Download or verify the pretrained model.

    Args:
        model_dir: Directory containing/to contain the model
        model_name: Name of the model file (without .pt extension)

    Returns:
        Path to the model file
    """
    model_path = model_dir / f"{model_name}.pt"

    if check_model_file(model_path):
        logger.info(f"Pretrained model found at {model_path}")
        return model_path

    # Check if we need to pull from Git LFS
    if model_path.exists():
        logger.warning(
            f"Model file at {model_path} appears to be a Git LFS pointer. "
            "Attempting to pull the actual file..."
        )
        try:
            # Try git lfs pull
            result = subprocess.run(
                ["git", "lfs", "pull", "--include", str(model_path)],
                capture_output=True,
                text=True,
                cwd=model_dir.parent,
            )
            if result.returncode == 0 and check_model_file(model_path):
                logger.info("Successfully pulled model from Git LFS")
                return model_path
        except FileNotFoundError:
            pass  # git not available

    # Try to download from GitHub releases or other sources
    download_url = (
        "https://github.com/maciekwielgosz/SegmentAnyTree/releases/download/"
        f"v1.0/{model_name}.pt"
    )

    logger.info(f"Attempting to download pretrained model from {download_url}")

    try:
        import urllib.request
        urllib.request.urlretrieve(download_url, model_path)
        if check_model_file(model_path):
            logger.info(f"Successfully downloaded model to {model_path}")
            return model_path
    except Exception as e:
        logger.warning(f"Failed to download model: {e}")

    raise PretrainedModelError(
        f"Could not obtain pretrained model at {model_path}. "
        "Please ensure Git LFS is installed and run 'git lfs pull', "
        "or download the model manually from: "
        "https://github.com/maciekwielgosz/SegmentAnyTree"
    )


def setup_inference_environment(
    config: Config,
    input_files: List[Path],
    output_dir: Path,
) -> Path:
    """
    Set up the environment for running inference with the original pipeline.

    Creates a temporary eval.yaml config file pointing to the input files.

    Args:
        config: Configuration object
        input_files: List of input LAS/PLY files
        output_dir: Output directory for results

    Returns:
        Path to the temporary eval.yaml config
    """
    # Get the project root (where torch_points3d lives)
    project_root = Path(__file__).parent.parent

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get model directory
    model_dir = project_root / "model_file"

    # Build the eval config
    eval_config = f"""defaults:
  - visualization: eval

num_workers: 0
batch_size: 1
cuda: 0
weight_name: "latest"
enable_cudnn: True
checkpoint_dir: "{model_dir}"
model_name: PointGroup-PAPER
precompute_multi_scale: True
enable_dropout: False
voting_runs: 1

data:
  fold: {[str(f) for f in input_files]}

tracker_options:
  full_res: True
  make_submission: True
  ply_output: "segmented.ply"

hydra:
  run:
    dir: {output_dir}
"""

    # Write the eval config to output directory
    eval_yaml_path = output_dir / "eval.yaml"
    with open(eval_yaml_path, 'w') as f:
        f.write(eval_config)

    return eval_yaml_path


def preprocess_input_file(
    input_path: Path,
    output_dir: Path,
) -> Path:
    """
    Preprocess an input LAS/LAZ/PLY file for inference.

    - Converts LAS/LAZ to PLY format expected by the model
    - Handles coordinate normalization

    Args:
        input_path: Path to input file
        output_dir: Directory for preprocessed files

    Returns:
        Path to preprocessed PLY file
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    suffix = input_path.suffix.lower()
    stem = input_path.stem

    if suffix in ['.las', '.laz']:
        # Convert LAS/LAZ to PLY
        try:
            import laspy
            from plyfile import PlyData, PlyElement
        except ImportError:
            raise ImportError(
                "laspy and plyfile are required for LAS file conversion. "
                "Install with: pip install laspy plyfile"
            )

        logger.info(f"Converting {input_path} to PLY format...")

        # Read LAS file
        las = laspy.read(str(input_path))

        # Extract coordinates
        x = las.x.astype(np.float32)
        y = las.y.astype(np.float32)
        z = las.z.astype(np.float32)

        # Normalize to local coordinates (important for model)
        x = x - x.min()
        y = y - y.min()
        z = z - z.min()

        n_points = len(x)

        # Create PLY array with semantic_seg field (all 1 = tree for inference)
        # The model expects this field but will predict the actual labels
        dtype = [
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('semantic_seg', 'i4'), ('treeID', 'i4')
        ]
        ply_data = np.zeros(n_points, dtype=dtype)
        ply_data['x'] = x
        ply_data['y'] = y
        ply_data['z'] = z
        ply_data['semantic_seg'] = 1  # Placeholder
        ply_data['treeID'] = 0  # Placeholder

        # Write PLY
        output_path = output_dir / f"{stem}_preprocessed.ply"
        el = PlyElement.describe(ply_data, 'vertex')
        PlyData([el], text=True).write(str(output_path))

        logger.info(f"Converted {n_points} points to {output_path}")
        return output_path

    elif suffix == '.ply':
        # Copy PLY file to output directory
        output_path = output_dir / f"{stem}_preprocessed.ply"
        shutil.copy(input_path, output_path)
        return output_path

    else:
        raise ValueError(f"Unsupported file format: {suffix}")


def run_pretrained_inference(
    config: Config,
    input_path: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    device: str = "cuda",
) -> Path:
    """
    Run inference using the pretrained SegmentAnyTree model.

    Args:
        config: Configuration object with paths
        input_path: Path to input LAS/LAZ/PLY file or directory
        output_dir: Output directory (defaults to config.paths.inference.output)
        device: Device to run on ('cuda' or 'cpu')

    Returns:
        Path to output directory with segmented files
    """
    input_path = Path(input_path)

    if output_dir is None:
        if config.paths.inference:
            output_dir = Path(config.paths.inference.output)
        else:
            output_dir = input_path.parent / "inference_output"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Find input files
    if input_path.is_dir():
        input_files = list(input_path.glob("*.las")) + \
                      list(input_path.glob("*.laz")) + \
                      list(input_path.glob("*.ply"))
    else:
        input_files = [input_path]

    if not input_files:
        raise ValueError(f"No LAS/LAZ/PLY files found in {input_path}")

    logger.info(f"Found {len(input_files)} input files")

    # Check for pretrained model
    project_root = Path(__file__).parent.parent
    model_dir = project_root / "model_file"
    model_path = download_pretrained_model(model_dir)

    # Preprocess input files
    preprocessed_dir = output_dir / "preprocessed"
    preprocessed_files = []
    for f in input_files:
        preprocessed = preprocess_input_file(f, preprocessed_dir)
        preprocessed_files.append(preprocessed)

    # Set up inference environment
    eval_yaml = setup_inference_environment(config, preprocessed_files, output_dir)

    # Run inference using the original eval.py
    logger.info("Running pretrained model inference...")

    # We need to run eval.py with the proper Python path
    eval_script = project_root / "eval.py"

    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root) + ":" + env.get("PYTHONPATH", "")

    if device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""

    cmd = [
        sys.executable,
        str(eval_script),
        f"--config-name={eval_yaml}",
    ]

    logger.info(f"Running: {' '.join(cmd)}")

    result = subprocess.run(
        cmd,
        cwd=str(project_root),
        env=env,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        logger.error(f"Inference failed:\n{result.stderr}")
        raise RuntimeError(f"Inference failed: {result.stderr}")

    logger.info(f"Inference completed. Results saved to {output_dir}")
    return output_dir


def run_inference_cli():
    """Command-line interface for pretrained inference."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run inference with pretrained SegmentAnyTree model"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config.yaml",
        help="Path to config.yaml file"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to input LAS/LAZ/PLY file or directory"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output directory (defaults to config setting)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run on"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Load config
    config = load_config(args.config)

    # Run inference
    try:
        output_dir = run_pretrained_inference(
            config,
            args.input,
            args.output,
            args.device,
        )
        print(f"\nInference complete! Results saved to: {output_dir}")
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    run_inference_cli()
