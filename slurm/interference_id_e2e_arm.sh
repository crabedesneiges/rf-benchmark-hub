#!/bin/bash
# interference_id_e2e_arm.sh — Download + prepare + train interf_cnn on the GNSS-jamming
# (interf_gnss6) set, end to end, on an ARM compute node. Data is public (Zenodo 4629685,
# CC-BY-4.0, ~1.9 GB, no login) so this needs no credentials.
#
# Usage: sbatch slurm/interference_id_e2e_arm.sh [EPOCHS [SEED]]
#   EPOCHS default: 100 — training epochs
#   SEED   default: 42  — RNG seed (use 42/43/44 for multi-seed runs)
#
# Output locations (no direct write to leaderboard/results/ by this script):
#   result.json  → $WORK/logs/multiseed/interference_id/interf_cnn-seed<seed>.json
#   checkpoint   → $WORK/checkpoints/multiseed/interf_cnn-seed<seed>.pt
#
#SBATCH --job-name=rfbench_interf
#SBATCH --output=logs/rfbench_interf_%j.out
#SBATCH --error=logs/rfbench_interf_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
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
# Derive REPO from SLURM_SUBMIT_DIR so that the correct worktree's code is used regardless of
# which rfbench editable install is registered in the venv's .pth.
REPO="${SLURM_SUBMIT_DIR:-$WORK/projets/rf-benchmark-hub}"
# Strip trailing /slurm si sbatch lancé depuis le sous-répertoire slurm/.
REPO="${REPO%/slurm}"
VENV_CPU="${RFBENCH_VENV_CPU:-$WORK/envs/rfbench-arm}"          # .[dev,data]: numpy/h5py/requests
VENV_GPU="${RFBENCH_VENV_GPU:-$WORK/envs/rfbench-arm-gpu}"      # .[dev,data,tasks,torch]: + torch/CUDA
DATASET="interf_gnss6"
TASK="interference_id"
MODEL="interf_cnn"
EPOCHS="${1:-100}"
SEED="${2:-42}"                            # RNG seed; override as $2 (use 42/43/44)
REGIME="from_scratch"

# Multi-seed staging — no direct write to leaderboard/results/
OUT_DIR="$WORK/logs/multiseed/${TASK}"
OUT="$OUT_DIR/${MODEL}-seed${SEED}.json"
CKPT_DIR="$WORK/checkpoints/multiseed"
CKPT="$CKPT_DIR/${MODEL}-seed${SEED}.pt"

# Garantir que le code du repo de soumission précède le .pth de l'install editable.
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export RFBENCH_CACHE="${RFBENCH_CACHE:-$WORK/data/rfbench_cache}"
export RFBENCH_HARDWARE="1x NVIDIA GB200"

echo "=== node=$(hostname) arch=$(uname -m) model=$MODEL epochs=$EPOCHS seed=$SEED date=$(date -Is) ==="
echo "REPO=$REPO"
cd "$REPO" || { echo "REPO NOT FOUND: $REPO"; exit 2; }
mkdir -p "$OUT_DIR" "$CKPT_DIR"
echo "RFBENCH_CACHE=$RFBENCH_CACHE (free: $(df -h "$WORK" | awk 'NR==2{print $4}'))"

# Rename le job SLURM pour inclure le seed (best-effort, non bloquant).
[ -n "${SLURM_JOB_ID:-}" ] && \
  scontrol update JobId="$SLURM_JOB_ID" JobName="rfbench_interf_s${SEED}" 2>/dev/null || true

echo "=== [1/3] rfbench data download $DATASET (Zenodo 4629685, ~1.9 GB, no login) ==="
"$VENV_CPU/bin/rfbench" data download "$DATASET"
rc=$?
if [ "$rc" -ne 0 ]; then echo "RESULT: DOWNLOAD FAILED (rc=$rc)"; exit 4; fi

echo "=== [2/3] rfbench data prepare $DATASET (80/10/10 stratified, seed 42) ==="
"$VENV_CPU/bin/rfbench" data prepare "$DATASET" --out "$REPO/leaderboard"
rc=$?
if [ "$rc" -ne 0 ]; then echo "RESULT: PREPARE FAILED (rc=$rc)"; exit 5; fi
find "$REPO/leaderboard/splits/$DATASET" -name '*.idx.json' 2>/dev/null

echo "=== [3/3] rfbench train $MODEL on $TASK/$DATASET ($REGIME, $EPOCHS epochs, seed $SEED) ==="
"$VENV_GPU/bin/rfbench" train \
  --task "$TASK" \
  --dataset "$DATASET" \
  --model "$MODEL" \
  --regime "$REGIME" \
  --epochs "$EPOCHS" \
  --batch-size 256 \
  --lr 1e-3 \
  --seed "$SEED" \
  --device cuda \
  --out-checkpoint "$CKPT" \
  --out "$OUT"
rc=$?

echo "=== emitted result.json ==="
[ -f "$OUT" ] && head -c 2000 "$OUT"
echo ""
echo "=================================================="
if [ "$rc" -eq 0 ]; then
  echo "RESULT: SUCCESS — $MODEL seed=$SEED trained on $DATASET; result.json at $OUT ; checkpoint at $CKPT"
else
  echo "RESULT: TRAIN FAILED (rc=$rc)"
fi
exit "$rc"
