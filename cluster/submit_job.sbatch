#!/bin/bash
# SLURM job template for train_rnn.py.
#
# Usage (from the repo root on the cluster):
#   sbatch cluster/submit_job.sbatch <net-type> <hidden-size> [extra train_rnn.py args...]
# Examples:
#   sbatch cluster/submit_job.sbatch direct    1500
#   sbatch cluster/submit_job.sbatch curric    1500 --n-epochs 300
#   sbatch cluster/submit_job.sbatch full-rank 1500

#SBATCH --job-name=lowrank_rnn
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=CHANGE_ME        # e.g. `sinfo` to list partitions on your cluster
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
# SBATCH --gres=gpu:1                # uncomment if your cluster has GPU nodes and you want one
                                      # (this code runs fine on CPU; only worth it for hidden_size >> 1500)

set -euo pipefail

# --- environment setup: adjust to how Python is provisioned on your cluster ---
# module load python/3.11 2>/dev/null || true
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate parametric_manifolds
source .venv/bin/activate            # assumes a venv already created at the repo root (see run_on_cluster.sh)

mkdir -p logs

NET_TYPE=$1
HIDDEN_SIZE=$2
shift 2

OUT_DIR="results/${SLURM_JOB_ID}_${NET_TYPE}_${HIDDEN_SIZE}"
mkdir -p "$OUT_DIR"

python train_rnn.py \
    --net-type "$NET_TYPE" \
    --hidden-size "$HIDDEN_SIZE" \
    --out-dir "$OUT_DIR" \
    "$@"
