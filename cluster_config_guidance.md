# Azure Databricks Cluster Configuration Guide

This guide provides configuration recommendations for running LiDAR Panoptic Segmentation on Azure Databricks GPU clusters.

## Recommended Cluster Configuration

### Databricks Runtime
- **Runtime Version**: 15.4 LTS GPU
- **Spark Version**: 3.5.0
- **Scala Version**: 2.12

### Node Configuration (Single-Node)

For single-node GPU workloads:

| Setting | Recommended Value |
|---------|-------------------|
| Worker Type | Standard_NC6s_v3 (or Standard_NC12s_v3) |
| Driver Type | Same as worker |
| Mode | Single Node |
| Min Workers | 0 |
| Max Workers | 0 |

### GPU Options

| GPU Type | VRAM | Use Case |
|----------|------|----------|
| V100 (NC6s_v3) | 16 GB | Development, small datasets |
| V100 (NC12s_v3) | 32 GB | Medium datasets |
| A100 (NC24ads_A100_v4) | 80 GB | Large datasets, production |

## Cluster Policies

### Recommended Spark Configuration

```json
{
    "spark.databricks.delta.preview.enabled": "true",
    "spark.sql.execution.arrow.enabled": "true",
    "spark.sql.shuffle.partitions": "auto"
}
```

### Environment Variables

Set these as cluster environment variables:

```bash
# CUDA paths
CUDA_HOME=/usr/local/cuda
PATH=${CUDA_HOME}/bin:${PATH}
LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# PyTorch settings
TORCH_CUDA_ARCH_LIST=7.0  # V100
# TORCH_CUDA_ARCH_LIST=8.0  # A100

# MinkowskiEngine
MAX_JOBS=8
```

## Unity Catalog Configuration

### Storage Locations

Configure Unity Catalog external locations for data access:

```sql
-- Create external location for forest data
CREATE EXTERNAL LOCATION forest_data
URL 'abfss://forest-data@yourstorageaccount.dfs.core.windows.net/'
WITH (STORAGE CREDENTIAL your_credential);

-- Grant access
GRANT READ FILES ON EXTERNAL LOCATION forest_data TO `data_scientists`;
GRANT WRITE FILES ON EXTERNAL LOCATION forest_data TO `data_scientists`;
```

### Volume Configuration

Create volumes for model artifacts:

```sql
-- Create catalog and schema
CREATE CATALOG IF NOT EXISTS ml_catalog;
CREATE SCHEMA IF NOT EXISTS ml_catalog.lidar_segmentation;

-- Create volume for models
CREATE VOLUME IF NOT EXISTS ml_catalog.lidar_segmentation.models;

-- Create volume for outputs
CREATE VOLUME IF NOT EXISTS ml_catalog.lidar_segmentation.outputs;
```

## Library Installation

### Cluster Libraries

Install these libraries at the cluster level:

**PyPI Packages:**
- pydantic>=2.0
- shapely>=2.0
- geopandas>=0.12
- laspy>=2.4
- plyfile>=0.8
- hdbscan>=0.8.29
- mlflow>=2.5

### Notebook-Based Installation

MinkowskiEngine must be installed via notebook (not cluster libraries):

```python
# Cell 1: Get PAT from secrets
ADO_PAT = dbutils.secrets.get("azure", "ado_pat")
with open("/tmp/ado_pat", "w") as f:
    f.write(ADO_PAT)
```

```bash
# Cell 2: Install MinkowskiEngine
%sh
export ADO_PAT=$(cat /tmp/ado_pat)
export ME_REPO="https://dev.azure.com/org/project/_git/MinkowskiEngine"
export ME_REF="main"
bash ./scripts/install_minkowski.sh
```

## Memory Management

### GPU Memory

For large point clouds, monitor GPU memory:

```python
import torch

def get_gpu_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        cached = torch.cuda.memory_reserved() / 1e9
        print(f"GPU Memory: {allocated:.2f} GB allocated, {cached:.2f} GB cached")
        return allocated, cached
    return 0, 0

# Clear cache if needed
torch.cuda.empty_cache()
```

### Batch Size Tuning

Adjust batch size based on available GPU memory:

| GPU VRAM | Recommended Batch Size |
|----------|------------------------|
| 16 GB | 4-8 |
| 32 GB | 8-16 |
| 80 GB | 16-32 |

## Autoscaling (Not Recommended)

For this workload, single-node is preferred over autoscaling because:

1. MinkowskiEngine is installed per-node
2. GPU models don't benefit from multi-node scaling
3. Simpler debugging and monitoring

If you must use multi-node:

```json
{
    "autoscale": {
        "min_workers": 0,
        "max_workers": 2
    },
    "cluster_log_conf": {
        "dbfs": {
            "destination": "dbfs:/cluster-logs"
        }
    }
}
```

## Monitoring

### Enable Ganglia Metrics

Monitor GPU utilization:

1. Go to Cluster → Metrics
2. Enable Ganglia metrics
3. Monitor GPU utilization, memory, and temperature

### MLflow Experiment Tracking

All training runs are logged to MLflow:

```python
import mlflow

# View experiments
experiments = mlflow.search_experiments()
for exp in experiments:
    print(f"{exp.name}: {exp.experiment_id}")

# View runs
runs = mlflow.search_runs(experiment_names=["LidarPanopticSegmentation_Experiments"])
print(runs[["run_id", "metrics.val/loss", "params.epochs"]])
```

## Troubleshooting

### CUDA Out of Memory

If you encounter CUDA OOM errors:

1. Reduce batch size in config.yaml
2. Enable tile-based processing for large files
3. Clear GPU cache between runs

```python
import torch
import gc

torch.cuda.empty_cache()
gc.collect()
```

### MinkowskiEngine Build Failures

Common issues:

1. **Missing CUDA toolkit**: Ensure cuda-toolkit-12-1 is installed
2. **Wrong TORCH_CUDA_ARCH_LIST**: Match your GPU capability
3. **Setuptools version**: Must be <60.0 for numpy.distutils

Check the install script logs for specific errors.

### Unity Catalog Access Denied

Verify permissions:

```sql
-- Check your access
SHOW GRANTS ON EXTERNAL LOCATION forest_data;

-- Request access from admin if needed
```

## Best Practices

1. **Use Init Scripts Sparingly**: Prefer notebook-based installation
2. **Pin Library Versions**: Ensure reproducibility
3. **Use Volumes**: Store models in Unity Catalog volumes
4. **Enable Logging**: Configure MLflow and cluster logs
5. **Test Locally First**: Use Conda environment before Databricks
6. **Monitor GPU**: Watch for OOM and thermal throttling
7. **Use Checkpoints**: Save model checkpoints regularly

## Sample Cluster JSON

```json
{
    "cluster_name": "lidar-panoptic-gpu",
    "spark_version": "15.4.x-gpu-ml-scala2.12",
    "node_type_id": "Standard_NC6s_v3",
    "driver_node_type_id": "Standard_NC6s_v3",
    "num_workers": 0,
    "spark_conf": {
        "spark.databricks.cluster.profile": "singleNode",
        "spark.master": "local[*]"
    },
    "spark_env_vars": {
        "CUDA_HOME": "/usr/local/cuda",
        "TORCH_CUDA_ARCH_LIST": "7.0"
    },
    "enable_elastic_disk": true,
    "cluster_log_conf": {
        "dbfs": {
            "destination": "dbfs:/cluster-logs/lidar-panoptic"
        }
    }
}
```
