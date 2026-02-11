#!/usr/bin/env python3
"""
Data Download Script for LiDAR Panoptic Segmentation

Downloads training/validation/test datasets from Zenodo and other sources,
organizing them for use with the SegmentAnyTree training pipeline.

Supported Datasets:
- FOR-instance: Forest instance segmentation benchmark (https://zenodo.org/records/8287792)
- NIBIO-MLS: Norwegian mobile laser scanning forest data
- Custom: Support for user-provided datasets

Usage:
    # Download all available data to local directory
    python scripts/download_data.py --output ./data

    # Download to Unity Catalog Volume (Databricks)
    python scripts/download_data.py --output /Volumes/catalog/schema/training_data

    # Download specific datasets only
    python scripts/download_data.py --datasets for-instance --output ./data

    # Verify existing data without downloading
    python scripts/download_data.py --verify-only --output ./data
"""

import os
import sys
import argparse
import hashlib
import zipfile
import tarfile
import shutil
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from urllib.request import urlretrieve, Request, urlopen
from urllib.error import HTTPError, URLError
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger(__name__)

# =============================================================================
# Dataset Registry
# =============================================================================
# Each dataset entry contains:
#   - url: Direct download URL
#   - zenodo_id: Zenodo record ID (for API-based downloads)
#   - files: List of expected files/directories after extraction
#   - size_gb: Approximate download size in GB
#   - description: Human-readable description

DATASETS = {
    "for-instance": {
        "name": "FOR-instance Benchmark",
        "description": "Forest instance segmentation benchmark with ALS, TLS, and MLS data",
        "zenodo_id": "8287792",
        "zenodo_url": "https://zenodo.org/records/8287792",
        "size_gb": 15.0,
        "files": [
            "FOR-instanceV2/",  # Main directory
        ],
        "splits": {
            "train": ["CULS", "NIBIO", "RMIT", "SCION", "TUWIEN"],
            "val": ["FOR-instanceV2_val"],
            "test": ["FOR-instanceV2_test"],
        },
        "format": "ply",
        "labels": {
            "semantic_seg": "1=non-tree, 2=tree",
            "treeID": "Instance ID per tree",
        },
    },
    "nibio-mls": {
        "name": "NIBIO Mobile Laser Scanning",
        "description": "Norwegian forest MLS data for tree segmentation",
        "zenodo_id": None,  # Not on Zenodo
        "manual_url": "https://github.com/maciekwielgosz/SegmentAnyTree",
        "size_gb": 2.0,
        "files": [],
        "note": "Contact NIBIO for data access",
    },
    "sample": {
        "name": "Sample Data",
        "description": "Small sample dataset for testing the pipeline",
        "size_gb": 0.01,
        "generated": True,
        "files": ["sample/train/", "sample/val/", "sample/test/"],
    },
}

# =============================================================================
# Download Progress
# =============================================================================

class DownloadProgress:
    """Progress reporter for downloads."""

    def __init__(self, filename: str, total_size: Optional[int] = None):
        self.filename = filename
        self.total_size = total_size
        self.downloaded = 0
        self.last_percent = -1

    def __call__(self, block_num: int, block_size: int, total_size: int):
        if total_size > 0:
            self.total_size = total_size

        self.downloaded = block_num * block_size

        if self.total_size and self.total_size > 0:
            percent = int(100 * self.downloaded / self.total_size)
            if percent != self.last_percent:
                self.last_percent = percent
                size_mb = self.downloaded / (1024 * 1024)
                total_mb = self.total_size / (1024 * 1024)
                print(f"\r  Downloading {self.filename}: {percent}% ({size_mb:.1f}/{total_mb:.1f} MB)", end="", flush=True)
        else:
            size_mb = self.downloaded / (1024 * 1024)
            print(f"\r  Downloading {self.filename}: {size_mb:.1f} MB", end="", flush=True)

# =============================================================================
# Zenodo API Helpers
# =============================================================================

def get_zenodo_files(record_id: str) -> List[Dict[str, Any]]:
    """
    Get list of files from a Zenodo record via API.

    Returns:
        List of file metadata dicts with 'filename', 'size', 'checksum', 'links' keys
    """
    api_url = f"https://zenodo.org/api/records/{record_id}"

    try:
        log.info(f"Querying Zenodo API for record {record_id}...")
        with urlopen(api_url) as response:
            data = json.loads(response.read().decode())

        files = data.get("files", [])
        log.info(f"Found {len(files)} files in Zenodo record")

        for f in files:
            size_mb = f.get("size", 0) / (1024 * 1024)
            log.info(f"  - {f['key']} ({size_mb:.1f} MB)")

        return files

    except HTTPError as e:
        log.error(f"Failed to query Zenodo API: HTTP {e.code}")
        return []
    except URLError as e:
        log.error(f"Failed to query Zenodo API: {e.reason}")
        return []
    except Exception as e:
        log.error(f"Failed to query Zenodo API: {e}")
        return []

def download_zenodo_file(
    file_info: Dict[str, Any],
    output_dir: Path,
    verify_checksum: bool = True
) -> Optional[Path]:
    """
    Download a single file from Zenodo.

    Args:
        file_info: File metadata from Zenodo API
        output_dir: Directory to save the file
        verify_checksum: Whether to verify MD5 checksum after download

    Returns:
        Path to downloaded file, or None if download failed
    """
    filename = file_info["key"]
    download_url = file_info["links"]["self"]
    expected_size = file_info.get("size", 0)
    expected_checksum = file_info.get("checksum", "").replace("md5:", "")

    output_path = output_dir / filename

    # Skip if already downloaded and verified
    if output_path.exists():
        if verify_checksum and expected_checksum:
            log.info(f"Verifying existing file: {filename}")
            actual_checksum = compute_md5(output_path)
            if actual_checksum == expected_checksum:
                log.info(f"  File already exists and checksum matches: {filename}")
                return output_path
            else:
                log.warning(f"  Checksum mismatch, re-downloading: {filename}")
        else:
            log.info(f"  File already exists: {filename}")
            return output_path

    # Download the file
    try:
        log.info(f"Downloading: {filename}")
        progress = DownloadProgress(filename, expected_size)
        urlretrieve(download_url, output_path, reporthook=progress)
        print()  # Newline after progress

        # Verify checksum
        if verify_checksum and expected_checksum:
            log.info(f"Verifying checksum...")
            actual_checksum = compute_md5(output_path)
            if actual_checksum != expected_checksum:
                log.error(f"Checksum mismatch for {filename}")
                log.error(f"  Expected: {expected_checksum}")
                log.error(f"  Actual:   {actual_checksum}")
                output_path.unlink()
                return None
            log.info("  Checksum OK")

        return output_path

    except Exception as e:
        log.error(f"Failed to download {filename}: {e}")
        if output_path.exists():
            output_path.unlink()
        return None

def compute_md5(filepath: Path, chunk_size: int = 8192) -> str:
    """Compute MD5 checksum of a file."""
    md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        while chunk := f.read(chunk_size):
            md5.update(chunk)
    return md5.hexdigest()

# =============================================================================
# Extraction Helpers
# =============================================================================

def extract_archive(archive_path: Path, output_dir: Path) -> bool:
    """
    Extract a zip or tar archive.

    Returns:
        True if extraction succeeded, False otherwise
    """
    log.info(f"Extracting: {archive_path.name}")

    try:
        if archive_path.suffix == ".zip":
            with zipfile.ZipFile(archive_path, "r") as zf:
                zf.extractall(output_dir)
        elif archive_path.suffix in (".tar", ".gz", ".tgz", ".bz2"):
            mode = "r:*"
            with tarfile.open(archive_path, mode) as tf:
                tf.extractall(output_dir)
        else:
            log.warning(f"Unknown archive format: {archive_path.suffix}")
            return False

        log.info(f"  Extracted to: {output_dir}")
        return True

    except Exception as e:
        log.error(f"Failed to extract {archive_path}: {e}")
        return False

# =============================================================================
# Dataset Organization
# =============================================================================

def organize_for_instance_data(data_dir: Path) -> bool:
    """
    Organize FOR-instance data into train/val/test splits.

    Expected structure after organization:
        data_dir/
            raw/
                CULS/
                    *.ply
                NIBIO/
                    *.ply
                ...
            train/
                *.ply (symlinks or copies)
            val/
                *.ply
            test/
                *.ply
    """
    raw_dir = data_dir / "raw"

    if not raw_dir.exists():
        log.warning(f"Raw data directory not found: {raw_dir}")
        return False

    # Create split directories
    for split in ["train", "val", "test"]:
        split_dir = data_dir / split
        split_dir.mkdir(exist_ok=True)

    # Count files by pattern
    train_count = 0
    val_count = 0
    test_count = 0

    for ply_file in raw_dir.rglob("*.ply"):
        filename = ply_file.stem.lower()

        # Determine split based on filename suffix
        if filename.endswith("_val") or filename.endswith("val"):
            dest_dir = data_dir / "val"
            val_count += 1
        elif filename.endswith("_test") or filename.endswith("test"):
            dest_dir = data_dir / "test"
            test_count += 1
        else:
            dest_dir = data_dir / "train"
            train_count += 1

        # Create symlink (or copy for cross-device)
        dest_path = dest_dir / ply_file.name
        if not dest_path.exists():
            try:
                dest_path.symlink_to(ply_file)
            except OSError:
                # Symlinks may not work across devices
                shutil.copy2(ply_file, dest_path)

    log.info(f"Organized data splits:")
    log.info(f"  Train: {train_count} files")
    log.info(f"  Val:   {val_count} files")
    log.info(f"  Test:  {test_count} files")

    return True

# =============================================================================
# Sample Data Generation
# =============================================================================

def generate_sample_data(output_dir: Path) -> bool:
    """
    Generate small sample PLY files for testing the pipeline.
    """
    try:
        import numpy as np
    except ImportError:
        log.error("NumPy is required to generate sample data")
        return False

    try:
        from plyfile import PlyData, PlyElement
    except ImportError:
        log.error("plyfile is required to generate sample data. Install with: pip install plyfile")
        return False

    log.info("Generating sample data...")

    for split in ["train", "val", "test"]:
        split_dir = output_dir / "sample" / "raw" / split
        split_dir.mkdir(parents=True, exist_ok=True)

        # Generate 2-3 sample files per split
        n_files = 3 if split == "train" else 2

        for i in range(n_files):
            filename = f"sample_{split}_{i+1}"
            if split == "val":
                filename += "_val"
            elif split == "test":
                filename += "_test"
            filename += ".ply"

            filepath = split_dir / filename

            # Generate random point cloud with ~5000 points
            n_points = np.random.randint(4000, 6000)

            # Create a simple forest scene with ground and trees
            # Ground points (class 0 - non-tree)
            n_ground = n_points // 3
            ground_x = np.random.uniform(0, 50, n_ground)
            ground_y = np.random.uniform(0, 50, n_ground)
            ground_z = np.random.uniform(0, 0.5, n_ground)
            ground_sem = np.zeros(n_ground, dtype=np.int32) + 1  # semantic: 1 = non-tree
            ground_inst = np.zeros(n_ground, dtype=np.int32)

            # Tree points (class 1 - tree)
            n_trees = np.random.randint(3, 8)
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
            tree_xyz = np.vstack(tree_points) if tree_points else np.zeros((0, 3))
            tree_sem = np.concatenate(tree_sems) if tree_sems else np.array([], dtype=np.int32)
            tree_inst = np.concatenate(tree_insts) if tree_insts else np.array([], dtype=np.int32)

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

            log.info(f"  Created: {filepath.relative_to(output_dir)} ({len(all_x)} points, {n_trees} trees)")

    return True

# =============================================================================
# Main Download Functions
# =============================================================================

def download_for_instance(output_dir: Path, skip_existing: bool = True) -> bool:
    """
    Download the FOR-instance dataset from Zenodo.
    """
    dataset_info = DATASETS["for-instance"]
    zenodo_id = dataset_info["zenodo_id"]

    log.info("=" * 60)
    log.info(f"Downloading: {dataset_info['name']}")
    log.info(f"Description: {dataset_info['description']}")
    log.info(f"Estimated size: {dataset_info['size_gb']:.1f} GB")
    log.info("=" * 60)

    # Create output directory
    data_dir = output_dir / "for-instance"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Get file list from Zenodo
    files = get_zenodo_files(zenodo_id)
    if not files:
        log.error("Failed to get file list from Zenodo")
        log.info(f"Please download manually from: {dataset_info['zenodo_url']}")
        return False

    # Download each file
    downloads_dir = data_dir / "downloads"
    downloads_dir.mkdir(exist_ok=True)

    downloaded_files = []
    for file_info in files:
        filepath = download_zenodo_file(file_info, downloads_dir, verify_checksum=True)
        if filepath:
            downloaded_files.append(filepath)

    if not downloaded_files:
        log.error("No files were downloaded")
        return False

    # Extract archives
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(exist_ok=True)

    for archive_path in downloaded_files:
        if archive_path.suffix in (".zip", ".tar", ".gz", ".tgz", ".bz2"):
            extract_archive(archive_path, raw_dir)

    # Organize into train/val/test splits
    organize_for_instance_data(data_dir)

    log.info("FOR-instance download complete!")
    return True

def download_dataset(
    dataset_name: str,
    output_dir: Path,
    skip_existing: bool = True
) -> bool:
    """
    Download a dataset by name.
    """
    if dataset_name not in DATASETS:
        log.error(f"Unknown dataset: {dataset_name}")
        log.info(f"Available datasets: {', '.join(DATASETS.keys())}")
        return False

    dataset_info = DATASETS[dataset_name]

    if dataset_name == "for-instance":
        return download_for_instance(output_dir, skip_existing)

    elif dataset_name == "sample":
        return generate_sample_data(output_dir)

    elif dataset_info.get("zenodo_id"):
        # Generic Zenodo download
        log.info(f"Downloading {dataset_info['name']} from Zenodo...")
        # Implement similar to FOR-instance
        log.warning("Generic Zenodo download not yet implemented")
        return False

    elif dataset_info.get("manual_url"):
        log.info(f"Dataset '{dataset_name}' requires manual download:")
        log.info(f"  URL: {dataset_info['manual_url']}")
        if dataset_info.get("note"):
            log.info(f"  Note: {dataset_info['note']}")
        return False

    else:
        log.error(f"No download method available for: {dataset_name}")
        return False

def verify_data(output_dir: Path) -> Dict[str, bool]:
    """
    Verify that expected data files exist.

    Returns:
        Dict mapping dataset names to verification status
    """
    results = {}

    for dataset_name, dataset_info in DATASETS.items():
        data_dir = output_dir / dataset_name

        if not data_dir.exists():
            results[dataset_name] = False
            continue

        # Check for expected files/directories
        expected_files = dataset_info.get("files", [])
        all_found = True

        for expected in expected_files:
            expected_path = data_dir / expected
            if not expected_path.exists():
                all_found = False
                break

        # Also check for any PLY files
        ply_files = list(data_dir.rglob("*.ply"))

        results[dataset_name] = all_found or len(ply_files) > 0

    return results

def list_datasets():
    """Print information about available datasets."""
    print("\nAvailable Datasets:")
    print("=" * 60)

    for name, info in DATASETS.items():
        print(f"\n{name}:")
        print(f"  Name: {info['name']}")
        print(f"  Description: {info['description']}")
        print(f"  Size: ~{info.get('size_gb', '?')} GB")

        if info.get("zenodo_id"):
            print(f"  Source: Zenodo ({info['zenodo_url']})")
        elif info.get("manual_url"):
            print(f"  Source: {info['manual_url']}")
        elif info.get("generated"):
            print(f"  Source: Generated locally")

    print()

# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Download training data for LiDAR Panoptic Segmentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all datasets to ./data
  python scripts/download_data.py --output ./data

  # Download to Unity Catalog Volume (Databricks)
  python scripts/download_data.py --output /Volumes/catalog/schema/training_data

  # Download only FOR-instance dataset
  python scripts/download_data.py --datasets for-instance --output ./data

  # Generate sample data for testing
  python scripts/download_data.py --datasets sample --output ./data

  # List available datasets
  python scripts/download_data.py --list

  # Verify existing data
  python scripts/download_data.py --verify-only --output ./data
        """
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        default="./data",
        help="Output directory for downloaded data (default: ./data)"
    )

    parser.add_argument(
        "--datasets", "-d",
        type=str,
        nargs="+",
        default=["sample"],
        help="Datasets to download (default: sample). Use 'all' for all datasets."
    )

    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available datasets and exit"
    )

    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing data without downloading"
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if files exist"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # List datasets
    if args.list:
        list_datasets()
        return 0

    output_dir = Path(args.output)

    # Verify only
    if args.verify_only:
        log.info(f"Verifying data in: {output_dir}")
        results = verify_data(output_dir)

        print("\nData Verification Results:")
        print("-" * 40)
        for name, found in results.items():
            status = "OK" if found else "MISSING"
            print(f"  {name}: {status}")

        return 0 if all(results.values()) else 1

    # Determine which datasets to download
    datasets_to_download = args.datasets
    if "all" in datasets_to_download:
        datasets_to_download = list(DATASETS.keys())

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Output directory: {output_dir}")

    # Download each dataset
    success_count = 0
    for dataset_name in datasets_to_download:
        try:
            success = download_dataset(
                dataset_name,
                output_dir,
                skip_existing=not args.force
            )
            if success:
                success_count += 1
        except Exception as e:
            log.error(f"Failed to download {dataset_name}: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()

    # Summary
    print()
    log.info(f"Download complete: {success_count}/{len(datasets_to_download)} datasets succeeded")

    return 0 if success_count == len(datasets_to_download) else 1

if __name__ == "__main__":
    sys.exit(main())
