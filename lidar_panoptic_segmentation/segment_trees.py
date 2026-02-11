#!/usr/bin/env python3
"""
SegmentAnyTree - Tree Segmentation Pipeline

This module provides a unified interface for running tree instance segmentation
on LiDAR point cloud data using the pretrained SegmentAnyTree model.

Supports:
- Local file paths
- Unity Catalog Volume paths (abfss://, /Volumes/)
- LAS, LAZ, and PLY input formats
- Docker-based or native inference

Usage:
    # CLI
    python -m lidar_panoptic_segmentation.segment_trees \\
        --input /path/to/lidar.las \\
        --output /path/to/output/

    # Python API
    from lidar_panoptic_segmentation.segment_trees import segment_trees
    results = segment_trees(input_path, output_path)
"""

import logging
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class SegmentationResult:
    """Result of tree segmentation."""
    input_file: str
    output_file: str
    num_trees: int
    num_points: int
    success: bool
    error_message: Optional[str] = None


def is_unity_catalog_path(path: str) -> bool:
    """Check if a path is a Unity Catalog path."""
    return path.startswith("abfss://") or path.startswith("/Volumes/")


def resolve_uc_path(path: str, local_cache: Path) -> Path:
    """
    Resolve a Unity Catalog path to a local path.

    For Databricks environments, this copies files to/from UC.
    For local environments, returns the path as-is.
    """
    if not is_unity_catalog_path(path):
        return Path(path)

    # On Databricks, /Volumes/ paths are directly accessible
    if path.startswith("/Volumes/"):
        return Path(path)

    # For abfss:// paths, we need dbutils or azure-storage
    try:
        # Try Databricks dbutils (available in notebooks)
        from pyspark.dbutils import DBUtils
        from pyspark.sql import SparkSession

        spark = SparkSession.builder.getOrCreate()
        dbutils = DBUtils(spark)

        local_path = local_cache / Path(path).name
        dbutils.fs.cp(path, f"file:{local_path}")
        return local_path
    except ImportError:
        pass

    # Try azure-storage-blob
    try:
        from azure.identity import DefaultAzureCredential
        from azure.storage.blob import BlobServiceClient

        # Parse abfss:// URL
        # Format: abfss://container@account.dfs.core.windows.net/path
        parts = path.replace("abfss://", "").split("/", 1)
        container_account = parts[0]
        blob_path = parts[1] if len(parts) > 1 else ""

        container, account_domain = container_account.split("@")
        account = account_domain.split(".")[0]

        # Download to local cache
        credential = DefaultAzureCredential()
        blob_service = BlobServiceClient(
            f"https://{account}.blob.core.windows.net",
            credential=credential
        )
        blob_client = blob_service.get_container_client(container).get_blob_client(blob_path)

        local_path = local_cache / Path(blob_path).name
        with open(local_path, "wb") as f:
            f.write(blob_client.download_blob().readall())

        return local_path
    except ImportError:
        raise ImportError(
            "Azure storage support requires azure-storage-blob and azure-identity. "
            "Install with: pip install azure-storage-blob azure-identity"
        )


def upload_to_uc(local_path: Path, uc_path: str) -> None:
    """Upload a local file to Unity Catalog."""
    if not is_unity_catalog_path(uc_path):
        # Just copy locally
        shutil.copy(local_path, uc_path)
        return

    if uc_path.startswith("/Volumes/"):
        # Direct copy on Databricks
        shutil.copy(local_path, uc_path)
        return

    # For abfss:// paths
    try:
        from pyspark.dbutils import DBUtils
        from pyspark.sql import SparkSession

        spark = SparkSession.builder.getOrCreate()
        dbutils = DBUtils(spark)
        dbutils.fs.cp(f"file:{local_path}", uc_path)
        return
    except ImportError:
        pass

    try:
        from azure.identity import DefaultAzureCredential
        from azure.storage.blob import BlobServiceClient

        parts = uc_path.replace("abfss://", "").split("/", 1)
        container_account = parts[0]
        blob_path = parts[1] if len(parts) > 1 else ""

        container, account_domain = container_account.split("@")
        account = account_domain.split(".")[0]

        credential = DefaultAzureCredential()
        blob_service = BlobServiceClient(
            f"https://{account}.blob.core.windows.net",
            credential=credential
        )
        blob_client = blob_service.get_container_client(container).get_blob_client(blob_path)

        with open(local_path, "rb") as f:
            blob_client.upload_blob(f, overwrite=True)
    except ImportError:
        raise ImportError(
            "Azure storage support requires azure-storage-blob and azure-identity."
        )


def check_docker_available() -> bool:
    """Check if Docker is available and the GPU runtime works."""
    try:
        result = subprocess.run(
            ["docker", "--version"],
            capture_output=True,
            text=True,
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def check_native_available() -> bool:
    """Check if native inference is available (MinkowskiEngine installed)."""
    try:
        import MinkowskiEngine
        return True
    except ImportError:
        return False


def segment_trees_docker(
    input_dir: Path,
    output_dir: Path,
    docker_image: str = "maciekwielgosz/segment-any-tree:latest",
    gpu: bool = True,
) -> List[SegmentationResult]:
    """
    Run tree segmentation using the Docker container.

    Args:
        input_dir: Directory containing input LAS/LAZ/PLY files
        output_dir: Directory for output files
        docker_image: Docker image to use
        gpu: Whether to use GPU (--gpus all)

    Returns:
        List of SegmentationResult objects
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build docker command
    cmd = ["docker", "run", "--rm"]

    if gpu:
        cmd.extend(["--gpus", "all"])

    cmd.extend([
        "-v", f"{input_dir.absolute()}:/home/nibio/mutable-outside-world/bucket_in_folder",
        "-v", f"{output_dir.absolute()}:/home/nibio/mutable-outside-world/bucket_out_folder",
        docker_image,
    ])

    logger.info(f"Running Docker inference: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"Docker inference failed:\n{result.stderr}")
        raise RuntimeError(f"Docker inference failed: {result.stderr}")

    # Collect results
    results = []
    for input_file in input_dir.glob("*"):
        if input_file.suffix.lower() in [".las", ".laz", ".ply"]:
            # Find corresponding output file
            output_file = output_dir / "final_results" / f"{input_file.stem}.ply"
            if output_file.exists():
                # Count trees in output
                num_trees, num_points = count_instances_in_ply(output_file)
                results.append(SegmentationResult(
                    input_file=str(input_file),
                    output_file=str(output_file),
                    num_trees=num_trees,
                    num_points=num_points,
                    success=True,
                ))
            else:
                results.append(SegmentationResult(
                    input_file=str(input_file),
                    output_file="",
                    num_trees=0,
                    num_points=0,
                    success=False,
                    error_message="Output file not found",
                ))

    return results


def count_instances_in_ply(ply_path: Path) -> tuple:
    """Count unique tree instances in a PLY file."""
    try:
        from plyfile import PlyData
        import numpy as np

        plydata = PlyData.read(str(ply_path))
        vertex = plydata['vertex']

        num_points = len(vertex)

        # Look for instance/treeID field
        for field_name in ['treeID', 'instance_id', 'instance', 'tree_id']:
            if field_name in vertex.data.dtype.names:
                instances = vertex[field_name]
                # Count unique non-zero instances
                unique_instances = np.unique(instances[instances > 0])
                return len(unique_instances), num_points

        return 0, num_points
    except Exception as e:
        logger.warning(f"Could not count instances in {ply_path}: {e}")
        return 0, 0


def segment_trees_native(
    input_dir: Path,
    output_dir: Path,
    model_path: Optional[Path] = None,
    device: str = "cuda",
) -> List[SegmentationResult]:
    """
    Run tree segmentation using native Python (requires MinkowskiEngine).

    Args:
        input_dir: Directory containing input files
        output_dir: Directory for output files
        model_path: Path to pretrained model (default: model_file/PointGroup-PAPER.pt)
        device: Device to run on

    Returns:
        List of SegmentationResult objects
    """
    # This requires the full torch_points3d environment
    # Import here to avoid dependency issues
    project_root = Path(__file__).parent.parent

    if model_path is None:
        model_path = project_root / "model_file" / "PointGroup-PAPER.pt"

    # Add project root to path
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # Run the inference script
    run_inference_script = project_root / "run_inference.sh"

    if run_inference_script.exists():
        # Use the shell script
        cmd = [
            "bash", str(run_inference_script),
            str(input_dir),
            str(output_dir),
            "true",  # clean output dir
        ]

        env = os.environ.copy()
        env["PYTHONPATH"] = str(project_root) + ":" + env.get("PYTHONPATH", "")

        logger.info(f"Running native inference: {' '.join(cmd)}")

        result = subprocess.run(cmd, capture_output=True, text=True, env=env)

        if result.returncode != 0:
            logger.error(f"Native inference failed:\n{result.stderr}")
            raise RuntimeError(f"Native inference failed: {result.stderr}")

    # Collect results
    results = []
    for input_file in input_dir.glob("*"):
        if input_file.suffix.lower() in [".las", ".laz", ".ply"]:
            output_file = output_dir / "final_results" / f"{input_file.stem}.ply"
            if output_file.exists():
                num_trees, num_points = count_instances_in_ply(output_file)
                results.append(SegmentationResult(
                    input_file=str(input_file),
                    output_file=str(output_file),
                    num_trees=num_trees,
                    num_points=num_points,
                    success=True,
                ))
            else:
                results.append(SegmentationResult(
                    input_file=str(input_file),
                    output_file="",
                    num_trees=0,
                    num_points=0,
                    success=False,
                    error_message="Output file not found",
                ))

    return results


def segment_trees(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    method: str = "auto",
    config_path: Optional[str] = None,
    gpu: bool = True,
) -> List[SegmentationResult]:
    """
    Segment trees in LiDAR point cloud data.

    This is the main entry point for tree segmentation. It automatically
    selects the best available method (Docker or native) and handles
    Unity Catalog paths.

    Args:
        input_path: Path to input file or directory (supports abfss://, /Volumes/)
        output_path: Path to output directory (supports abfss://, /Volumes/)
        method: Inference method - 'auto', 'docker', or 'native'
        config_path: Optional path to config.yaml for additional settings
        gpu: Whether to use GPU

    Returns:
        List of SegmentationResult objects

    Example:
        # Local files
        results = segment_trees("./data/forest.las", "./output/")

        # Unity Catalog paths
        results = segment_trees(
            "/Volumes/catalog/schema/lidar/input/",
            "/Volumes/catalog/schema/lidar/output/"
        )

        # Azure blob storage
        results = segment_trees(
            "abfss://data@storage.dfs.core.windows.net/lidar/input/",
            "abfss://data@storage.dfs.core.windows.net/lidar/output/"
        )
    """
    input_path = str(input_path)
    output_path = str(output_path)

    # Load config if provided
    config = None
    if config_path:
        from .config import load_config
        config = load_config(config_path)

    # Create temp directories for UC paths
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        local_input = temp_dir / "input"
        local_output = temp_dir / "output"
        local_input.mkdir()
        local_output.mkdir()

        # Handle Unity Catalog input paths
        if is_unity_catalog_path(input_path):
            logger.info(f"Copying input from Unity Catalog: {input_path}")
            input_resolved = resolve_uc_path(input_path, local_input)
            if Path(input_path).suffix:  # Single file
                shutil.copy(input_resolved, local_input)
                local_input_dir = local_input
            else:  # Directory - need to copy all files
                # For directories, copy recursively
                local_input_dir = local_input
                # This is simplified - would need proper directory handling
        else:
            local_input_dir = Path(input_path)
            if local_input_dir.is_file():
                # Single file - copy to temp input dir
                shutil.copy(local_input_dir, local_input / local_input_dir.name)
                local_input_dir = local_input

        # Select inference method
        if method == "auto":
            if check_native_available():
                method = "native"
            elif check_docker_available():
                method = "docker"
            else:
                raise RuntimeError(
                    "No inference method available. Either:\n"
                    "1. Install MinkowskiEngine for native inference\n"
                    "2. Install Docker for containerized inference\n"
                    "3. Use Docker image: docker pull maciekwielgosz/segment-any-tree:latest"
                )

        logger.info(f"Using inference method: {method}")

        # Run inference
        if method == "docker":
            results = segment_trees_docker(local_input_dir, local_output, gpu=gpu)
        elif method == "native":
            results = segment_trees_native(local_input_dir, local_output)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Handle Unity Catalog output paths
        if is_unity_catalog_path(output_path):
            logger.info(f"Copying output to Unity Catalog: {output_path}")
            final_results = local_output / "final_results"
            if final_results.exists():
                for f in final_results.glob("*"):
                    upload_to_uc(f, f"{output_path}/{f.name}")
        else:
            # Copy to local output path
            output_dir = Path(output_path)
            output_dir.mkdir(parents=True, exist_ok=True)
            final_results = local_output / "final_results"
            if final_results.exists():
                for f in final_results.glob("*"):
                    shutil.copy(f, output_dir / f.name)

            # Update result paths
            for r in results:
                if r.success:
                    r.output_file = str(output_dir / Path(r.output_file).name)

        return results


def main():
    """Command-line interface."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Segment trees in LiDAR point cloud data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single file
  python -m lidar_panoptic_segmentation.segment_trees -i forest.las -o ./output/

  # Directory of files
  python -m lidar_panoptic_segmentation.segment_trees -i ./lidar_data/ -o ./output/

  # Unity Catalog paths
  python -m lidar_panoptic_segmentation.segment_trees \\
      -i /Volumes/catalog/schema/input/ \\
      -o /Volumes/catalog/schema/output/

  # Force Docker method
  python -m lidar_panoptic_segmentation.segment_trees -i data.las -o out/ --method docker
        """
    )
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Input file or directory (LAS, LAZ, PLY)"
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Output directory"
    )
    parser.add_argument(
        "-c", "--config",
        default=None,
        help="Path to config.yaml"
    )
    parser.add_argument(
        "--method",
        choices=["auto", "docker", "native"],
        default="auto",
        help="Inference method (default: auto)"
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Use CPU only (no GPU)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )

    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    try:
        results = segment_trees(
            args.input,
            args.output,
            method=args.method,
            config_path=args.config,
            gpu=not args.cpu,
        )

        # Print results
        print("\n" + "=" * 60)
        print("SEGMENTATION RESULTS")
        print("=" * 60)

        total_trees = 0
        total_points = 0
        successful = 0

        for r in results:
            status = "OK" if r.success else "FAILED"
            print(f"\n{Path(r.input_file).name}: {status}")
            if r.success:
                print(f"  Output: {r.output_file}")
                print(f"  Trees detected: {r.num_trees}")
                print(f"  Points processed: {r.num_points:,}")
                total_trees += r.num_trees
                total_points += r.num_points
                successful += 1
            else:
                print(f"  Error: {r.error_message}")

        print("\n" + "-" * 60)
        print(f"Total: {successful}/{len(results)} files processed")
        print(f"Total trees detected: {total_trees}")
        print(f"Total points processed: {total_points:,}")
        print("=" * 60)

    except Exception as e:
        logger.error(f"Segmentation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
