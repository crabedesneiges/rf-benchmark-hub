#!/bin/bash
# compute_model_sizes_arm.sh — measure n_params + FLOPs of every IMPLEMENTED (registered) model
# and patch the numbers into leaderboard/results/**/*.json. Pure measurement, no training.
# Runs on the ARM/GB200 nodes (torch required; the Intel frontend has no torch).
#
#   sbatch slurm/compute_model_sizes_arm.sh
#
# Commit the resulting result.json changes from the frontend afterwards (git add/commit/push).
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
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

# Ensure the measurement deps are present in the project venv (fvcore + torch).
uv pip install -e ".[size]" >/dev/null

echo "== computing model sizes (params + FLOPs) =="
uv run python scripts/compute_model_sizes.py --write

echo "== done; review 'git diff leaderboard/results' on the frontend, then commit =="
