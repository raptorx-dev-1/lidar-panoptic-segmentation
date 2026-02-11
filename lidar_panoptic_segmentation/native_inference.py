#!/usr/bin/env python3
"""
Native Inference Pipeline for SegmentAnyTree

Runs tree segmentation using the pretrained SegmentAnyTree model natively
WITHOUT Docker. Requires MinkowskiEngine to be installed.

This module uses the EXACT same loading mechanism as the original codebase
to ensure the pretrained model works correctly.

IMPORTANT: You must have the actual model file (not Git LFS pointer):
    git lfs pull  # or download from GitHub manually

Usage:
    from lidar_panoptic_segmentation.native_inference import run_inference

    results = run_inference(
        input_path="/path/to/lidar/files",
        output_path="/path/to/output",
        model_path="/path/to/PointGroup-PAPER.pt"
    )
"""

import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

# Setup logging
logger = logging.getLogger(__name__)


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


def setup_paths():
    """Add project root to path for torch_points3d imports."""
    project_root = get_project_root()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


# Setup paths immediately on import
setup_paths()


@dataclass
class SegmentationOutput:
    """Output from tree segmentation."""
    points: np.ndarray  # (N, 3) XYZ coordinates
    semantic_pred: np.ndarray  # (N,) semantic labels (0=non-tree, 1=tree)
    instance_pred: np.ndarray  # (N,) instance IDs
    num_trees: int
    metadata: Dict[str, Any] = field(default_factory=dict)


def check_minkowski_available() -> bool:
    """Check if MinkowskiEngine is available."""
    try:
        import MinkowskiEngine
        return True
    except ImportError:
        return False


def validate_model_file(model_path: Path) -> None:
    """Validate that the model file is real (not LFS pointer)."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            f"Please ensure you have downloaded the model."
        )

    # Check if it's an LFS pointer (small text file)
    size = model_path.stat().st_size
    if size < 10000:  # Real model is ~600MB, pointer is ~130 bytes
        with open(model_path, 'r') as f:
            content = f.read(100)
        if 'git-lfs' in content or 'version https' in content:
            raise ValueError(
                f"Model file is a Git LFS pointer, not the actual weights.\n"
                f"File: {model_path}\n"
                f"Size: {size} bytes (should be ~665MB)\n\n"
                f"To fix, run:\n"
                f"  cd {model_path.parent.parent}\n"
                f"  git lfs pull\n\n"
                f"Or download manually from:\n"
                f"  https://github.com/maciekwielgosz/SegmentAnyTree"
            )


class NativeInferencePipeline:
    """
    Native inference pipeline using the original torch_points3d model loading.

    This class replicates exactly what the Docker container does.
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        device: str = "cuda",
    ):
        self.model_path = Path(model_path)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Validate
        if not check_minkowski_available():
            raise ImportError(
                "MinkowskiEngine is required for native inference.\n"
                "Install using: bash scripts/install_minkowski.sh"
            )

        validate_model_file(self.model_path)

        # Load model using original mechanism
        self.model = None
        self.run_config = None
        self.dataset_properties = None
        self._load_model()

    def _load_model(self):
        """Load the pretrained model using original codebase mechanism."""
        from omegaconf import OmegaConf
        from torch_points3d.models.model_factory import instantiate_model

        logger.info(f"Loading model from {self.model_path}")

        # Load checkpoint (handle both old and new PyTorch versions)
        try:
            checkpoint = torch.load(self.model_path, map_location="cpu", weights_only=False)
        except TypeError:
            # Older PyTorch versions don't support weights_only parameter
            checkpoint = torch.load(self.model_path, map_location="cpu")

        # Validate checkpoint structure
        required_keys = ['run_config', 'models']
        for key in required_keys:
            if key not in checkpoint:
                raise ValueError(f"Invalid checkpoint: missing '{key}'")

        self.run_config = checkpoint['run_config']
        self.dataset_properties = checkpoint.get('dataset_properties', {})

        # Create dataset mock with properties from checkpoint
        class DatasetFromCheckpoint:
            def __init__(self, properties, run_config):
                # Use properties from checkpoint
                self.feature_dimension = properties.get('feature_dimension', 4)
                self.num_classes = properties.get('num_classes', 2)

                # Stuff/thing classes for panoptic
                self.stuff_classes = torch.tensor([0])  # non-tree
                self.thing_classes = torch.tensor([1])  # tree

                # Copy any other properties
                for k, v in properties.items():
                    if not hasattr(self, k):
                        setattr(self, k, v)

        dataset = DatasetFromCheckpoint(self.dataset_properties, self.run_config)

        # Instantiate model using original factory
        run_config_omega = OmegaConf.create(self.run_config)
        self.model = instantiate_model(run_config_omega, dataset)

        # Load weights
        models = checkpoint['models']
        available_weights = list(models.keys())
        logger.info(f"Available weights: {available_weights}")

        # Prefer best_miou, then latest
        if 'best_miou' in models:
            state_dict = models['best_miou']
            logger.info("Loading best_miou weights")
        elif 'latest' in models:
            state_dict = models['latest']
            logger.info("Loading latest weights")
        else:
            state_dict = models[available_weights[0]]
            logger.info(f"Loading {available_weights[0]} weights")

        self.model.load_state_dict(state_dict, strict=False)
        self.model = self.model.to(self.device)
        self.model.eval()

        logger.info(f"Model loaded successfully on {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    def preprocess_point_cloud(
        self,
        points: np.ndarray,
        grid_size: float = 0.2,
    ) -> Tuple[Any, np.ndarray]:
        """
        Preprocess point cloud for inference.

        Args:
            points: (N, 3+) array with XYZ and optional features
            grid_size: Voxel grid size for subsampling

        Returns:
            Tuple of (torch_geometric Data, offset for restoring coordinates)
        """
        from torch_geometric.data import Data
        from torch_points3d.core.data_transform import GridSampling3D

        # Extract coordinates
        pos = points[:, :3].astype(np.float32)

        # Center and normalize
        offset = pos.min(axis=0)
        pos = pos - offset

        # Create features (z-relative, absolute positions)
        center = pos.mean(axis=0)
        x_rel = pos[:, 0] - center[0]
        y_rel = pos[:, 1] - center[1]
        z_rel = pos[:, 2] - center[2]
        z_abs = pos[:, 2]

        features = np.column_stack([x_rel, y_rel, z_rel, z_abs]).astype(np.float32)

        n_points = len(pos)

        # Create torch_geometric Data with dummy labels for inference
        # The model's set_input requires these fields even during inference
        data = Data(
            pos=torch.from_numpy(pos),
            x=torch.from_numpy(features),
            # Dummy labels required by PointGroup3heads.set_input()
            center_label=torch.zeros(n_points, 3, dtype=torch.float32),
            y=torch.zeros(n_points, dtype=torch.long),
            num_instances=torch.tensor([0]),
            instance_labels=torch.zeros(n_points, dtype=torch.long),
            instance_mask=torch.zeros(n_points, dtype=torch.bool),
            vote_label=torch.zeros(n_points, 3, dtype=torch.float32),
        )

        # Apply grid sampling
        sampler = GridSampling3D(size=grid_size, quantize_coords=True, mode='last')
        data = sampler(data)

        # Add batch index
        data.batch = torch.zeros(data.pos.shape[0], dtype=torch.long)

        return data, offset

    def run_inference(self, data: Any) -> Dict:
        """Run model inference on preprocessed data."""
        data = data.to(self.device)

        with torch.no_grad():
            self.model.set_input(data, self.device)
            self.model.forward(epoch=999)  # High epoch enables clustering
            output = self.model.get_output()

        return output

    def postprocess_output(
        self,
        output: Any,
        data: Any,
        offset: np.ndarray,
    ) -> SegmentationOutput:
        """Convert model output to SegmentationOutput."""
        # Get positions back to original coordinates
        points = data.pos.cpu().numpy() + offset

        # Semantic predictions
        if hasattr(output, 'semantic_logits'):
            semantic_logits = output.semantic_logits.cpu()
            semantic_pred = torch.argmax(semantic_logits, dim=1).numpy()
        else:
            semantic_pred = np.zeros(len(points), dtype=np.int32)

        # Instance predictions from clusters
        instance_pred = np.zeros(len(points), dtype=np.int32)
        num_trees = 0

        if hasattr(output, 'clusters') and output.clusters is not None:
            clusters = output.clusters
            num_trees = len(clusters)
            for i, cluster in enumerate(clusters):
                if cluster is not None:
                    cluster_idx = cluster.cpu().numpy()
                    instance_pred[cluster_idx] = i + 1

        return SegmentationOutput(
            points=points,
            semantic_pred=semantic_pred,
            instance_pred=instance_pred,
            num_trees=num_trees,
        )

    def segment_points(
        self,
        points: np.ndarray,
        grid_size: float = 0.2,
    ) -> SegmentationOutput:
        """
        Segment trees in a point cloud array.

        Args:
            points: (N, 3+) array of points
            grid_size: Voxel size for preprocessing

        Returns:
            SegmentationOutput
        """
        # Preprocess
        data, offset = self.preprocess_point_cloud(points, grid_size)

        # Run inference
        output = self.run_inference(data)

        # Postprocess
        result = self.postprocess_output(output, data, offset)

        return result

    def segment_file(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
    ) -> SegmentationOutput:
        """
        Segment trees in a LAS/PLY file.

        Args:
            input_path: Path to input file
            output_path: Optional path to save output

        Returns:
            SegmentationOutput
        """
        input_path = Path(input_path)
        logger.info(f"Processing: {input_path.name}")

        # Read point cloud
        points, metadata = self._read_file(input_path)

        # Run segmentation
        result = self.segment_points(points)
        result.metadata = metadata

        # Save if requested
        if output_path:
            self._save_output(result, output_path)

        logger.info(f"  Found {result.num_trees} trees")
        return result

    def segment_files(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        recursive: bool = False,
    ) -> List[SegmentationOutput]:
        """
        Segment trees in multiple files.

        Args:
            input_path: Input file or directory
            output_path: Output directory
            recursive: Search recursively

        Returns:
            List of SegmentationOutput
        """
        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # Find files
        if input_path.is_file():
            files = [input_path]
        else:
            patterns = ['*.las', '*.laz', '*.ply', '*.LAS', '*.LAZ', '*.PLY']
            files = []
            for p in patterns:
                if recursive:
                    files.extend(input_path.rglob(p))
                else:
                    files.extend(input_path.glob(p))

        if not files:
            logger.warning(f"No LiDAR files found in {input_path}")
            return []

        logger.info(f"Processing {len(files)} files")

        results = []
        for i, f in enumerate(files):
            logger.info(f"[{i+1}/{len(files)}] {f.name}")
            try:
                out_file = output_path / f"{f.stem}_segmented.ply"
                result = self.segment_file(f, out_file)
                results.append(result)
            except Exception as e:
                logger.error(f"  Failed: {e}")

        return results

    def _read_file(self, path: Path) -> Tuple[np.ndarray, Dict]:
        """Read a LAS/PLY file."""
        suffix = path.suffix.lower()

        if suffix in ['.las', '.laz']:
            return self._read_las(path)
        elif suffix == '.ply':
            return self._read_ply(path)
        else:
            raise ValueError(f"Unsupported format: {suffix}")

    def _read_las(self, path: Path) -> Tuple[np.ndarray, Dict]:
        """Read LAS/LAZ file."""
        try:
            import laspy
        except ImportError:
            raise ImportError("laspy required: pip install laspy")

        las = laspy.read(str(path))
        points = np.column_stack([
            las.x.astype(np.float32),
            las.y.astype(np.float32),
            las.z.astype(np.float32),
        ])

        return points, {'source': str(path), 'num_points': len(points)}

    def _read_ply(self, path: Path) -> Tuple[np.ndarray, Dict]:
        """Read PLY file."""
        try:
            from plyfile import PlyData
        except ImportError:
            raise ImportError("plyfile required: pip install plyfile")

        ply = PlyData.read(str(path))
        v = ply['vertex']
        points = np.column_stack([
            v['x'].astype(np.float32),
            v['y'].astype(np.float32),
            v['z'].astype(np.float32),
        ])

        return points, {'source': str(path), 'num_points': len(points)}

    def _save_output(self, result: SegmentationOutput, path: Union[str, Path]):
        """Save segmentation output."""
        from plyfile import PlyData, PlyElement

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        dtype = [
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('semantic_seg', 'i4'),
            ('treeID', 'i4'),
        ]
        arr = np.zeros(len(result.points), dtype=dtype)
        arr['x'] = result.points[:, 0]
        arr['y'] = result.points[:, 1]
        arr['z'] = result.points[:, 2]
        arr['semantic_seg'] = result.semantic_pred
        arr['treeID'] = result.instance_pred

        el = PlyElement.describe(arr, 'vertex')
        PlyData([el], text=True).write(str(path))

        logger.info(f"Saved: {path}")


def run_inference(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    model_path: Optional[Union[str, Path]] = None,
    device: str = "cuda",
) -> List[SegmentationOutput]:
    """
    Run tree segmentation inference.

    Args:
        input_path: Input file or directory
        output_path: Output directory
        model_path: Path to model (default: model_file/PointGroup-PAPER.pt)
        device: Device to use

    Returns:
        List of SegmentationOutput
    """
    if model_path is None:
        model_path = get_project_root() / "model_file" / "PointGroup-PAPER.pt"

    pipeline = NativeInferencePipeline(model_path, device)
    return pipeline.segment_files(input_path, output_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run tree segmentation")
    parser.add_argument("-i", "--input", required=True, help="Input path")
    parser.add_argument("-o", "--output", required=True, help="Output path")
    parser.add_argument("-m", "--model", help="Model path")
    parser.add_argument("--device", default="cuda", help="Device")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    results = run_inference(args.input, args.output, args.model, args.device)

    total_trees = sum(r.num_trees for r in results)
    print(f"\nProcessed {len(results)} files, found {total_trees} trees")
