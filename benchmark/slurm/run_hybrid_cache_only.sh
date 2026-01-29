#!/bin/bash
#SBATCH --output=Log/%x_%j.out
#SBATCH --error=Log/%x_%j.err
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --partition=rtx2080ti
#SBATCH --chdir=/srv/GadM/Datasets/Tmp/inferring_body_weight_from_ct_scans/tmp/master

# Note: Hybrid runs are CPU-only as requested.
# We use the '2080' partition but do not request GPUs via --gres=gpu

CONDA_ENV_PREFIX="/srv/GadM/Datasets/Tmp/inferring_body_weight_from_ct_scans/tmp/local_conda_envs"
CONDA_ENV_NAME="xlstm"

source ~/miniconda/etc/profile.d/conda.sh
conda activate "$CONDA_ENV_PREFIX/$CONDA_ENV_NAME" || {
    echo "[ERROR] Failed to activate Conda env: $CONDA_ENV_NAME" >&2
    exit 1
}

echo "[JOB] Hybrid Parser: Cache Only (Baseline)"
python datasets/loghub-2.0/benchmark/evaluation/Hybrid_eval.py --full_data --run_name Run_CacheOnly --device cpu --no_grouper --preload_ground_truth
