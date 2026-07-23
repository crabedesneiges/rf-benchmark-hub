#!/bin/bash
# compute_model_sizes_arm.sh — measure n_params + FLOPs of every IMPLEMENTED (registered) model
# and patch the numbers into leaderboard/results/**/*.json. Pure measurement, no training.
# Runs on the ARM/GB200 nodes (torch required; the Intel frontend has no torch).
#
#   mkdir -p logs && sbatch slurm/compute_model_sizes_arm.sh
#
# It runs against the SUBMIT directory (this checkout/worktree) via PYTHONPATH, using the
# pre-built ARM GPU venv ($WORK/envs/rfbench-arm-gpu, .[dev,data,tasks,torch]) + fvcore.
# Review `git diff leaderboard/results` on the frontend afterwards, then commit.
#
#SBATCH --job-name=rfbench_sizes
#SBATCH --output=logs/rfbench_sizes_%j.out
#SBATCH --error=logs/rfbench_sizes_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
# cluster mono-partition (defq*, GB200/ARM) -- no architecture constraint required.
set -uo pipefail

WORK="${WORK:?set \$WORK to your Lustre work dir}"
REPO="${RFBENCH_REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}"          # this worktree by default
VENV="${RFBENCH_VENV_GPU:-$WORK/envs/rfbench-arm-gpu}"       # .[dev,data,tasks,torch] — torch+CUDA
UV="${RFBENCH_UV:-$WORK/envs/uv-arm/uv}"
export UV_CACHE_DIR="$WORK/.uv_cache_arm"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"          # import rfbench + the model registry from THIS checkout
export RFBENCH_CACHE="${RFBENCH_CACHE:-$WORK/data/rfbench_cache}"

echo "=== node=$(hostname) arch=$(uname -m) repo=$REPO date=$(date -Is) ==="
cd "$REPO" || { echo "REPO NOT FOUND: $REPO"; exit 2; }
[ -x "$VENV/bin/python" ] || { echo "GPU venv missing: $VENV (run slurm/setup_gpu_venv.sh)"; exit 3; }

echo "=== ensure fvcore in the GPU venv (torch already present) ==="
"$UV" pip install --python "$VENV/bin/python" "fvcore>=0.1.5" >/dev/null || echo "(fvcore install warn -> params-only fallback)"

echo "=== measure params + FLOPs, patch result.json (--write) ==="
"$VENV/bin/python" scripts/compute_model_sizes.py --write

echo "=== done; review 'git diff leaderboard/results' on the frontend, then commit ==="
