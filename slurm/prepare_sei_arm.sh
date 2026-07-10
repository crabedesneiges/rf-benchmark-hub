#!/bin/bash
# prepare_sei_arm.sh — Build the SEI canonical split indices (+ checksums) for a dataset whose
# raw captures are already under $RFBENCH_CACHE (feat/sei-complete). CPU-only, ARM, no GPU.
#
# Use this AFTER manually placing the SigMF captures for ORACLE / POWDER (both need a manual,
# browser-side download — see rfbench.data.download.sei_{oracle,powder}). WiSig is already
# prepared (indices committed), so you only need this for oracle / powder.
#
#   sbatch slurm/prepare_sei_arm.sh <dataset>       # dataset: powder | oracle | wisig
#   e.g. sbatch slurm/prepare_sei_arm.sh powder
#
# Writes only leaderboard/splits/<dataset>/*.idx.json + *.manifest.json in THIS worktree — never
# raw data (D3). Commit those indices, then train with slurm/train_sei_arm.sh.
#
#SBATCH --job-name=sei_prepare
#SBATCH --output=/lustre/work/pdl16831/udl79f933/logs/rfbench_sei_prepare_%j.out
#SBATCH --error=/lustre/work/pdl16831/udl79f933/logs/rfbench_sei_prepare_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00

set -uo pipefail
WORK=/lustre/work/pdl16831/udl79f933
REPO="$WORK/projets/rf-benchmark-hub-sei"      # the feat/sei-complete worktree
VENV="$WORK/envs/rfbench-arm"                  # .[dev,data]: numpy/h5py present (no torch needed)
DATASET="${1:-powder}"

export RFBENCH_CACHE="$WORK/data/rfbench_cache"
# Force this worktree's rfbench onto the path (the venv is editable-installed against MAIN, which
# does not know the 'powder' dataset) — same override the training script uses.
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

echo "=== node=$(hostname) arch=$(uname -m) dataset=$DATASET date=$(date -Is) ==="
cd "$REPO" || { echo "REPO NOT FOUND: $REPO"; exit 2; }
echo "RFBENCH_CACHE=$RFBENCH_CACHE"
[ -d "$RFBENCH_CACHE/$DATASET" ] || { echo "NO DATA at $RFBENCH_CACHE/$DATASET — download it first (manual)."; exit 3; }

# Pre-flight: confirm this worktree's rfbench (with the powder dataset) is what loads.
"$VENV/bin/python" -c "
import rfbench
from rfbench.data.prepare.sei import CANONICAL_SPLIT_IDS
assert '$DATASET' in CANONICAL_SPLIT_IDS, 'unknown SEI dataset: $DATASET'
print('rfbench (sei worktree) =', rfbench.__file__)
" || { echo "PREFLIGHT FAILED: wrong rfbench / unknown dataset"; exit 5; }

echo "=== rfbench data prepare $DATASET (indices -> $REPO/leaderboard/splits/$DATASET) ==="
"$VENV/bin/rfbench" data prepare "$DATASET" --out "$REPO/leaderboard"
rc=$?

echo "=== written indices ==="
ls -la "$REPO/leaderboard/splits/$DATASET/" 2>&1 | tail -6

echo "=================================================="
if [ "$rc" -eq 0 ]; then
  echo "RESULT: SUCCESS — $DATASET split prepared; commit leaderboard/splits/$DATASET/, then train."
else
  echo "RESULT: prepare FAILED (rc=$rc)"
fi
exit "$rc"
