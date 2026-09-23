#!/bin/bash
# Run from your LOCAL machine. Syncs code to the cluster via git, then submits SLURM jobs.
#
# PREREQUISITE (not automated, do this once):
#   `origin` in this repo currently points to the upstream paper author's repo
#   (github.com/emebeiran/parametric_manifolds), which you cannot push to. Before using
#   this script, push your own commits to a repo you control (a personal fork, or an
#   internal git server) and set GIT_REMOTE_URL below to that URL.
#
# Usage:
#   ./cluster/run_on_cluster.sh direct    1500
#   ./cluster/run_on_cluster.sh curric    1500 --n-epochs 300
#   ./cluster/run_on_cluster.sh full-rank 1500

set -euo pipefail

# ================== fill these in ==================
SSH_HOST="CHANGE_ME"                          # e.g. openmind7.mit.edu, or a Host alias from ~/.ssh/config
SSH_USER="CHANGE_ME"                          # your cluster username
REMOTE_DIR="~/parametric_manifolds"           # where the repo should live on the cluster
GIT_REMOTE_URL="CHANGE_ME"                    # YOUR fork/remote, not emebeiran/parametric_manifolds
GIT_BRANCH="main"
# ====================================================

if [[ "$SSH_HOST" == "CHANGE_ME" || "$SSH_USER" == "CHANGE_ME" || "$GIT_REMOTE_URL" == "CHANGE_ME" ]]; then
    echo "Edit the config block at the top of $0 first (SSH_HOST, SSH_USER, GIT_REMOTE_URL)." >&2
    exit 1
fi

NET_TYPE=${1:?"usage: $0 <net-type: direct|gener|curric|full-rank> <hidden-size> [extra args...]"}
HIDDEN_SIZE=${2:?"usage: $0 <net-type> <hidden-size> [extra args...]"}
shift 2 || true
EXTRA_ARGS=("$@")

SSH_TARGET="${SSH_USER}@${SSH_HOST}"

echo "==> Syncing code to $SSH_TARGET:$REMOTE_DIR"
ssh "$SSH_TARGET" "
    set -euo pipefail
    if [ -d '$REMOTE_DIR/.git' ]; then
        cd '$REMOTE_DIR' && git fetch origin '$GIT_BRANCH' && git checkout '$GIT_BRANCH' && git pull origin '$GIT_BRANCH'
    else
        git clone --branch '$GIT_BRANCH' '$GIT_REMOTE_URL' '$REMOTE_DIR'
    fi
    cd '$REMOTE_DIR'
    if [ ! -d .venv ]; then
        python3 -m venv .venv
        source .venv/bin/activate
        pip install --upgrade pip
        pip install torch numpy scipy matplotlib
    fi
    mkdir -p logs results
"

echo "==> Submitting SLURM job: net-type=$NET_TYPE hidden-size=$HIDDEN_SIZE ${EXTRA_ARGS[*]}"
ssh "$SSH_TARGET" "cd '$REMOTE_DIR' && sbatch cluster/submit_job.sbatch '$NET_TYPE' '$HIDDEN_SIZE' ${EXTRA_ARGS[*]}"

echo "==> Submitted. Check status with:"
echo "    ssh $SSH_TARGET squeue -u $SSH_USER"
echo "==> Once finished, fetch results with:"
echo "    ./cluster/fetch_results.sh"
