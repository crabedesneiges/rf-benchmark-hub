#!/bin/bash
# download_prepare_arm.sh — Fetch a REAL published dataset + build its canonical split
# indices on an ARM compute node. CPU-only (download + prepare need no GPU).
#
# Raw data lands in $RFBENCH_CACHE (never git); the split-index sidecars land in the repo's
# leaderboard/splits/ (versioned) so they can be committed.
#
# Usage: sbatch slurm/download_prepare_arm.sh <dataset>
#   e.g. sbatch slurm/download_prepare_arm.sh radioml_2016_10a
#SBATCH --job-name=rfbench_dl
#SBATCH --output=logs/rfbench_dl_%j.out
#SBATCH --error=logs/rfbench_dl_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
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
DS="${1:?usage: sbatch download_prepare_arm.sh <dataset>}"
REPO="${RFBENCH_REPO:-$WORK/projets/rf-benchmark-hub}"
VENV="${RFBENCH_VENV_CPU:-$WORK/envs/rfbench-arm}"          # built with .[dev,data] (numpy/h5py/requests)
export RFBENCH_CACHE="${RFBENCH_CACHE:-$WORK/data/rfbench_cache}"
export UV_CACHE_DIR="$WORK/.uv_cache_arm"

echo "=== node=$(hostname) arch=$(uname -m) dataset=$DS date=$(date -Is) ==="
mkdir -p "$RFBENCH_CACHE"
cd "$REPO" || { echo "REPO NOT FOUND"; exit 2; }
echo "RFBENCH_CACHE=$RFBENCH_CACHE  (free: $(df -h "$WORK" | awk 'NR==2{print $4}'))"

echo "=== [1/2] rfbench data download $DS ==="
"$VENV/bin/rfbench" data download "$DS"
rc_dl=$?
if [ "$rc_dl" -ne 0 ]; then echo "RESULT: DOWNLOAD FAILED (rc=$rc_dl)"; exit 4; fi

echo "=== [2/2] rfbench data prepare $DS (indices -> repo leaderboard/splits) ==="
"$VENV/bin/rfbench" data prepare "$DS" --out "$REPO/leaderboard"
rc_prep=$?

echo "=== resulting split indices ==="
find "$REPO/leaderboard/splits" -name '*.idx.json' -newermt '-10 min' 2>/dev/null | head

echo "=================================================="
if [ "$rc_prep" -eq 0 ]; then
    echo "RESULT: SUCCESS — $DS downloaded to cache + split indices written under leaderboard/splits/"
else
    echo "RESULT: PREPARE FAILED (rc=$rc_prep)"
fi
exit "$rc_prep"
