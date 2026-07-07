#!/bin/bash
# setup_gpu_venv.sh — Build the GPU training venv on an ARM node AND verify torch sees the
# GB200. This is the gate for M3 baselines (real training). Requests 1 GPU.
#
# Usage: sbatch slurm/setup_gpu_venv.sh
#SBATCH --job-name=rfbench_gpu_setup
#SBATCH --output=/lustre/work/pdl16831/udl79f933/logs/rfbench_gpu_setup_%j.out
#SBATCH --error=/lustre/work/pdl16831/udl79f933/logs/rfbench_gpu_setup_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
# cluster mono-partition (defq*, GB200/ARM uniquement) -- pas de contrainte d'architecture requise
# (confirmé via `sinfo -o "%P %f %c %G"`, seule feature reportée: location=local)

set -uo pipefail
WORK=/lustre/work/pdl16831/udl79f933
REPO="$WORK/projets/rf-benchmark-hub"
VENV="$WORK/envs/rfbench-arm-gpu"
UV="$WORK/envs/uv-arm/uv"
export UV_CACHE_DIR="$WORK/.uv_cache_arm"
export UV_PYTHON_INSTALL_DIR="$WORK/.uv_python"

echo "=== node=$(hostname) arch=$(uname -m) date=$(date -Is) ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>&1 | head -4 || echo "(nvidia-smi indispo)"
cd "$REPO" || { echo "REPO NOT FOUND"; exit 2; }

echo "=== build venv @ $VENV + install .[dev,data,tasks,torch] ==="
"$UV" venv --python 3.11 "$VENV" || { echo "VENV FAILED"; exit 3; }
"$UV" pip install --python "$VENV/bin/python" -e ".[dev,data,tasks,torch]" || { echo "INSTALL FAILED"; exit 4; }

echo "=== torch / CUDA / GB200 ==="
"$VENV/bin/python" - <<'PY'
import torch, platform
print("arch", platform.machine(), "| torch", torch.__version__)
ok = torch.cuda.is_available()
print("cuda_available:", ok)
if ok:
    print("device:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
    x = torch.randn(1024, 1024, device="cuda")
    y = (x @ x).sum().item()
    print("matmul on GPU OK, sum=", round(y, 2))
else:
    print("NO CUDA DEVICE VISIBLE")
PY
rc=$?
echo "=================================================="
[ "$rc" -eq 0 ] && echo "RESULT: gpu venv built; see cuda_available above" || echo "RESULT: FAILED rc=$rc"
exit "$rc"
