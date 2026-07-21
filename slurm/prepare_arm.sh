#!/bin/bash
# prepare_arm.sh — Build canonical split indices for an ALREADY-DOWNLOADED dataset on an
# ARM compute node (the frontend has internet but not numpy/h5py; the compute node has the
# ARM venv). Raw data must already sit in $RFBENCH_CACHE (fetched on the frontend). Split
# indices are written into the repo's leaderboard/splits/ (versioned).
#
# Usage: sbatch slurm/prepare_arm.sh <dataset>
#SBATCH --job-name=rfbench_prep
#SBATCH --output=logs/rfbench_prep_%j.out
#SBATCH --error=logs/rfbench_prep_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:40:00
# cluster mono-partition (defq*, GB200/ARM uniquement) -- pas de contrainte d'architecture requise
# (confirmé via `sinfo -o "%P %f %c %G"`, seule feature reportée: location=local)

set -uo pipefail
# --- Portable config (override via environment; see slurm/README.md) -----------------
#   WORK                Lustre work root (REQUIRED; usually pre-set by the cluster).
#   RFBENCH_REPO        repo/worktree checkout to run       (default: $WORK/projets/rf-benchmark-hub[...]).
#   RFBENCH_VENV_CPU    CPU venv  .[dev,data]               (default: $WORK/envs/rfbench-arm).
#   RFBENCH_VENV_GPU    GPU venv  .[dev,data,tasks,torch]   (default: $WORK/envs/rfbench-arm-gpu).
#   RFBENCH_VENV_DETECTION  detection venv .[dev,detection] (default: $WORK/envs/rfbench-arm-detection).
#   RFBENCH_UV          uv binary for this arch             (default: $WORK/envs/uv-arm/uv).
#   RFBENCH_CACHE       dataset cache root                  (default: $WORK/data/rfbench_cache).
# SLURM logs go to logs/ relative to the submit dir: create it first (mkdir -p logs) or
# override with `sbatch --output=... --error=...`.
# ------------------------------------------------------------------------------------
WORK="${WORK:?set \$WORK to your Lustre work dir (e.g. /lustre/work/<project>/<user>)}"
DS="${1:?usage: sbatch prepare_arm.sh <dataset>}"
REPO="${RFBENCH_REPO:-$WORK/projets/rf-benchmark-hub}"
VENV="${RFBENCH_VENV_CPU:-$WORK/envs/rfbench-arm}"          # .[dev,data] : numpy/h5py present
export RFBENCH_CACHE="${RFBENCH_CACHE:-$WORK/data/rfbench_cache}"

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
