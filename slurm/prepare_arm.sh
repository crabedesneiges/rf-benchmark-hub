#!/bin/bash
# prepare_arm.sh — Build canonical split indices for an ALREADY-DOWNLOADED dataset on an
# ARM compute node (the frontend has internet but not numpy/h5py; the compute node has the
# ARM venv). Raw data must already sit in $RFBENCH_CACHE (fetched on the frontend). Split
# indices are written into the repo's leaderboard/splits/ (versioned).
#
# Usage: sbatch slurm/prepare_arm.sh <dataset>
#SBATCH --job-name=rfbench_prep
#SBATCH --output=/lustre/work/pdl16831/udl79f933/logs/rfbench_prep_%j.out
#SBATCH --error=/lustre/work/pdl16831/udl79f933/logs/rfbench_prep_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:40:00

set -uo pipefail
WORK=/lustre/work/pdl16831/udl79f933
DS="${1:?usage: sbatch prepare_arm.sh <dataset>}"
REPO="$WORK/projets/rf-benchmark-hub"
VENV="$WORK/envs/rfbench-arm"          # .[dev,data] : numpy/h5py present
export RFBENCH_CACHE="$WORK/data/rfbench_cache"

echo "=== node=$(hostname) arch=$(uname -m) dataset=$DS date=$(date -Is) ==="
cd "$REPO" || { echo "REPO NOT FOUND"; exit 2; }
echo "RFBENCH_CACHE=$RFBENCH_CACHE"
ls -la "$RFBENCH_CACHE/$DS" 2>/dev/null | head

echo "=== rfbench data prepare $DS (indices -> repo leaderboard/splits) ==="
"$VENV/bin/rfbench" data prepare "$DS" --out "$REPO/leaderboard"
rc=$?

echo "=== split indices produits ==="
find "$REPO/leaderboard/splits" -name '*.idx.json' 2>/dev/null | head -20

echo "=================================================="
[ "$rc" -eq 0 ] && echo "RESULT: SUCCESS — $DS split indices written" || echo "RESULT: PREPARE FAILED (rc=$rc)"
exit "$rc"
