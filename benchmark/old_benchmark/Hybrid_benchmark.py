"""
Hybrid benchmark settings.
Defines model paths and default parameters for each dataset.
"""
from pathlib import Path

# Assume we are running from within the repo, find the root
# Location: datasets/loghub-2.0/benchmark/old_benchmark/Hybrid_benchmark.py
# Root:     (3 levels up) -> datasets -> loghub-2.0 -> benchmark 
# RepoRoot: (3 more levels up?) -> master
# Actually, let's just use absolute paths based on what the user provided or relative to this file?
# User said: "The final models for each pipeline part can currently be found in @[thesis_work/data]"
# We need to construct absolute paths dynamically or assume a fixed structure relative to repo root.

# For robustness, we will try to find the repo root.
# Current file: .../datasets/loghub-2.0/benchmark/old_benchmark/Hybrid_benchmark.py
# Repo root: .../ (containing thesis_work, datasets, etc.)
# Count parents: 1(old_benchmark), 2(benchmark), 3(loghub-2.0), 4(datasets), 5(master/repo_root)

_file = Path(__file__).resolve()
REPO_ROOT = _file.parents[4]

LOG_GROUPER_DIR = REPO_ROOT / "thesis_work/data/log_grouper"
PARAM_ID_DIR = REPO_ROOT / "thesis_work/data/param_identifier"

hybrid_settings = {}

datasets = [
    "Proxifier", "Linux", "Apache", "Zookeeper", "Hadoop", "HealthApp", "OpenStack",
    "HPC", "Mac", "OpenSSH", "Spark", "Thunderbird", "BGL", "HDFS"
]

for ds in datasets:
    hybrid_settings[ds] = {
        # Checkpoint for Log Grouper (Model .pt file)
        "checkpoint_dir": str(LOG_GROUPER_DIR / f"model_{ds}.pt"),
        
         # Checkpoint for Param Identifier (Directory or File)
        "param_id_checkpoint_dir": str(PARAM_ID_DIR / ds),
        
        "device": "cpu", # Default device, override with --device
        "min_match_prob": 0.5,
        
        # Default flags (Overridden by CLI)
        "use_grouper": False, 
        "use_param_identifier": False,
        "preload_ground_truth": False,
    }
