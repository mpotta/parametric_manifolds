#!/bin/bash
# Run from your LOCAL machine. Copies trained checkpoints/logs back from the cluster.
#
# Usage:
#   ./cluster/fetch_results.sh

set -euo pipefail

# ================== fill these in (match run_on_cluster.sh) ==================
SSH_HOST="CHANGE_ME"
SSH_USER="CHANGE_ME"
REMOTE_DIR="~/parametric_manifolds"
LOCAL_DIR="ClusterResults"
# ===============================================================================

if [[ "$SSH_HOST" == "CHANGE_ME" || "$SSH_USER" == "CHANGE_ME" ]]; then
    echo "Edit the config block at the top of $0 first (SSH_HOST, SSH_USER)." >&2
    exit 1
fi

SSH_TARGET="${SSH_USER}@${SSH_HOST}"
mkdir -p "$LOCAL_DIR"

echo "==> Pulling results/ and logs/ from $SSH_TARGET:$REMOTE_DIR"
rsync -avz --progress "${SSH_TARGET}:${REMOTE_DIR}/results/" "${LOCAL_DIR}/results/"
rsync -avz --progress "${SSH_TARGET}:${REMOTE_DIR}/logs/" "${LOCAL_DIR}/logs/"

echo "==> Done. Results in ${LOCAL_DIR}/"
