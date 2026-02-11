# Databricks notebook source
# MAGIC %md
# MAGIC # Tree Segmentation on Databricks
# MAGIC
# MAGIC This notebook demonstrates how to run tree instance segmentation on LiDAR data
# MAGIC using the pretrained SegmentAnyTree model on Azure Databricks.
# MAGIC
# MAGIC ## Prerequisites
# MAGIC - Databricks Runtime 15.4 LTS GPU (single node)
# MAGIC - Unity Catalog access configured
# MAGIC - LiDAR data in LAS/LAZ/PLY format uploaded to a Volume
# MAGIC
# MAGIC ## Workflow
# MAGIC 1. Install MinkowskiEngine (first run only)
# MAGIC 2. Upload pretrained model to Volume
# MAGIC 3. Configure paths
# MAGIC 4. Run inference
# MAGIC 5. View results

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Install MinkowskiEngine
# MAGIC
# MAGIC This only needs to be run once per cluster. The installation takes about 10-15 minutes.
# MAGIC Skip this cell if MinkowskiEngine is already installed.

# COMMAND ----------

# Check if MinkowskiEngine is already installed
try:
    import MinkowskiEngine
    print(f"MinkowskiEngine {MinkowskiEngine.__version__} is already installed!")
    ME_INSTALLED = True
except ImportError:
    print("MinkowskiEngine not found. Running installation...")
    ME_INSTALLED = False

# COMMAND ----------

# Install MinkowskiEngine if not present
if not ME_INSTALLED:
    # Run the comprehensive installation script from the repository
    # This handles CUDA 12.x patches, distutils vendoring, and proper build configuration

    import subprocess
    import os

    # Path to the install script in the repo
    REPO_PATH = "/Workspace/Repos/your_user/SegmentAnyTree"
    INSTALL_SCRIPT = f"{REPO_PATH}/scripts/install_minkowski.sh"

    # Set environment variables for the build
    env = os.environ.copy()
    env["MAX_JOBS"] = "4"
    env["ME_REPO"] = "https://github.com/NVIDIA/MinkowskiEngine.git"
    env["ME_REF"] = "master"

    print("Running MinkowskiEngine installation script...")
    print("This may take 10-15 minutes on first run.")
    print("-" * 50)

    # Run the install script
    result = subprocess.run(
        ["bash", INSTALL_SCRIPT],
        env=env,
        capture_output=False,  # Show output in real-time
        text=True
    )

    if result.returncode != 0:
        print(f"\nInstallation script exited with code {result.returncode}")
        print("Check the output above for errors.")

    # Verify installation
    try:
        import MinkowskiEngine
        print(f"\nMinkowskiEngine {MinkowskiEngine.__version__} installed successfully!")
    except ImportError as e:
        print(f"\nInstallation failed: {e}")
        print("You may need to restart the cluster and try again.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Configure Paths
# MAGIC
# MAGIC Update these paths to match your Unity Catalog setup.

# COMMAND ----------

# Configuration - UPDATE THESE FOR YOUR ENVIRONMENT
CATALOG = "your_catalog"           # Your Unity Catalog name
SCHEMA = "your_schema"             # Your schema name
VOLUME_DATA = "lidar_data"         # Volume containing LiDAR files
VOLUME_MODELS = "models"           # Volume containing the pretrained model

# Derived paths
INPUT_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME_DATA}/input"
OUTPUT_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME_DATA}/output"
MODEL_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME_MODELS}/PointGroup-PAPER.pt"

print(f"Input path:  {INPUT_PATH}")
print(f"Output path: {OUTPUT_PATH}")
print(f"Model path:  {MODEL_PATH}")

# COMMAND ----------

# Verify paths exist
import os

paths_ok = True

if not os.path.exists(INPUT_PATH):
    print(f"❌ Input path does not exist: {INPUT_PATH}")
    print("   Please create the Volume and upload your LiDAR files.")
    paths_ok = False
else:
    files = [f for f in os.listdir(INPUT_PATH) if f.endswith(('.las', '.laz', '.ply', '.LAS', '.LAZ', '.PLY'))]
    print(f"✓ Input path exists with {len(files)} LiDAR files")

if not os.path.exists(MODEL_PATH):
    print(f"❌ Model not found: {MODEL_PATH}")
    print("   Please upload the PointGroup-PAPER.pt model to the Volume.")
    paths_ok = False
else:
    print(f"✓ Model found: {MODEL_PATH}")

# Create output directory if needed
os.makedirs(OUTPUT_PATH, exist_ok=True)
print(f"✓ Output path ready: {OUTPUT_PATH}")

if not paths_ok:
    print("\n⚠️ Please fix the path issues above before proceeding.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Upload Pretrained Model (if needed)
# MAGIC
# MAGIC If you haven't uploaded the pretrained model yet, run this cell.
# MAGIC It will copy the model from the repository to your Volume.

# COMMAND ----------

# Upload model from repo to Volume (run once)
import shutil

# Path to model in the repo
REPO_MODEL_PATH = "/Workspace/Repos/your_user/SegmentAnyTree/model_file/PointGroup-PAPER.pt"

if not os.path.exists(MODEL_PATH):
    if os.path.exists(REPO_MODEL_PATH):
        print(f"Copying model from {REPO_MODEL_PATH} to {MODEL_PATH}...")
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        shutil.copy(REPO_MODEL_PATH, MODEL_PATH)
        print("Done!")
    else:
        print(f"Model not found at {REPO_MODEL_PATH}")
        print("Please download the model manually and upload to your Volume.")
        print("Model URL: https://github.com/maciekwielgosz/SegmentAnyTree (use git lfs pull)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Run Tree Segmentation

# COMMAND ----------

# Import the inference pipeline
import sys
sys.path.insert(0, "/Workspace/Repos/your_user/SegmentAnyTree")

from lidar_panoptic_segmentation.native_inference import NativeInferencePipeline

# COMMAND ----------

# Initialize the pipeline
print("Loading model...")
pipeline = NativeInferencePipeline(
    model_path=MODEL_PATH,
    device="cuda"
)
print("Model loaded successfully!")

# COMMAND ----------

# Run segmentation on all files in the input directory
print(f"Processing files from: {INPUT_PATH}")
print(f"Output will be saved to: {OUTPUT_PATH}")
print("-" * 50)

results = pipeline.segment_files(
    input_path=INPUT_PATH,
    output_path=OUTPUT_PATH,
)

print("-" * 50)
print(f"Processed {len(results)} files")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: View Results

# COMMAND ----------

# Summary of results
import pandas as pd

summary_data = []
for r in results:
    summary_data.append({
        "File": r.metadata.get("source_file", "Unknown"),
        "Trees Detected": r.num_trees,
        "Points": len(r.points),
    })

summary_df = pd.DataFrame(summary_data)
display(summary_df)

# COMMAND ----------

# Total statistics
total_trees = sum(r.num_trees for r in results)
total_points = sum(len(r.points) for r in results)

print(f"Total files processed: {len(results)}")
print(f"Total trees detected: {total_trees}")
print(f"Total points processed: {total_points:,}")

# COMMAND ----------

# List output files
output_files = os.listdir(OUTPUT_PATH)
print(f"Output files in {OUTPUT_PATH}:")
for f in sorted(output_files):
    size = os.path.getsize(os.path.join(OUTPUT_PATH, f))
    print(f"  {f} ({size/1024/1024:.2f} MB)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Optional: Visualize a Result
# MAGIC
# MAGIC This cell loads one of the output files and creates a basic visualization.

# COMMAND ----------

# Load and visualize first result
if results:
    result = results[0]

    # Create a simple scatter plot
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # Subsample for visualization
    n_viz = min(50000, len(result.points))
    idx = np.random.choice(len(result.points), n_viz, replace=False)

    fig = plt.figure(figsize=(15, 5))

    # Semantic segmentation view
    ax1 = fig.add_subplot(131, projection='3d')
    colors = ['brown' if s == 0 else 'green' for s in result.semantic_pred[idx]]
    ax1.scatter(
        result.points[idx, 0],
        result.points[idx, 1],
        result.points[idx, 2],
        c=colors, s=0.1
    )
    ax1.set_title('Semantic Segmentation\n(brown=non-tree, green=tree)')

    # Instance segmentation view
    ax2 = fig.add_subplot(132, projection='3d')
    instance_colors = plt.cm.tab20(result.instance_pred[idx] % 20)
    ax2.scatter(
        result.points[idx, 0],
        result.points[idx, 1],
        result.points[idx, 2],
        c=instance_colors, s=0.1
    )
    ax2.set_title(f'Instance Segmentation\n({result.num_trees} trees)')

    # Top-down view
    ax3 = fig.add_subplot(133)
    ax3.scatter(
        result.points[idx, 0],
        result.points[idx, 1],
        c=instance_colors, s=0.1
    )
    ax3.set_title('Top-down View')
    ax3.set_aspect('equal')

    plt.tight_layout()
    plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps
# MAGIC
# MAGIC 1. **Download Results**: Copy output files from the Volume to your local machine
# MAGIC 2. **Generate Polygons**: Use the postprocessing module to extract tree crown polygons
# MAGIC 3. **Evaluate Accuracy**: Compare predictions with ground truth if available
# MAGIC 4. **Train Custom Model**: Fine-tune on your own annotated data

# COMMAND ----------

# Example: Generate tree crown polygons
from lidar_panoptic_segmentation.postprocess import extract_polygons, save_geojson

if results:
    result = results[0]

    # Extract polygons from instance predictions
    polygons = extract_polygons(
        points=result.points,
        instance_labels=result.instance_pred,
        min_points=50,
    )

    # Save as GeoJSON
    output_geojson = f"{OUTPUT_PATH}/tree_crowns.geojson"
    save_geojson(polygons, output_geojson)
    print(f"Saved {len(polygons)} tree crown polygons to {output_geojson}")
