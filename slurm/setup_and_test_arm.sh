#!/bin/bash
# setup_and_test_arm.sh — Build the rfbench venv on an ARM compute node and run the
# full test suite there. CPU-only (no --gres): base install + tests + synthetic
# prepare need no GPU, so this does NOT compete with training jobs for GB200s.
#
# Why: the login node is Intel x86_64 but compute nodes are ARM aarch64 — deps must
# be compiled/installed on the target arch. This proves the whole harness (splits,
# evaluate, regimes, data prepare on synthetic fixtures, tasks) is green on aarch64.
#
# Usage: sbatch slurm/setup_and_test_arm.sh
#SBATCH --job-name=rfbench_arm_test
#SBATCH --output=/lustre/work/pdl16831/udl79f933/logs/rfbench_arm_test_%j.out
#SBATCH --error=/lustre/work/pdl16831/udl79f933/logs/rfbench_arm_test_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00

set -uo pipefail
WORK=/lustre/work/pdl16831/udl79f933
REPO="$WORK/projets/rf-benchmark-hub"
VENV="$WORK/envs/rfbench-arm"
export UV_CACHE_DIR="$WORK/.uv_cache_arm"
export UV_PYTHON_INSTALL_DIR="$WORK/.uv_python"

echo "=== node=$(hostname) arch=$(uname -m) date=$(date -Is) ==="
# Use the ARM uv by ABSOLUTE path: PATH may otherwise resolve `uv` to the Intel
# login-node binary ($WORK/envs/uv/bin/uv) -> "Exec format error" on aarch64.
UV="$WORK/envs/uv-arm/uv"
echo "uv: $UV — $("$UV" --version 2>&1)"

cd "$REPO" || { echo "REPO NOT FOUND: $REPO"; exit 2; }
echo "HEAD=$(git rev-parse --short HEAD 2>/dev/null)"

echo "=== build ARM venv @ $VENV (python 3.11) ==="
"$UV" venv --python 3.11 "$VENV" || { echo "VENV BUILD FAILED"; exit 3; }

echo "=== install -e .[dev,data] (ARM wheels) ==="
"$UV" pip install --python "$VENV/bin/python" -e ".[dev,data]" || { echo "INSTALL FAILED"; exit 4; }

echo "=== interpreter arch ==="
"$VENV/bin/python" -c "import platform; print('python', platform.python_version(), platform.machine())"

echo "=== full test suite on aarch64 ==="
"$VENV/bin/python" -m pytest -q
rc=$?

echo "=== entrypoint + ARM data wheels ==="
"$VENV/bin/rfbench" --help >/dev/null && echo "rfbench --help OK"
"$VENV/bin/python" -c "import numpy, h5py, platform; print('numpy', numpy.__version__, '| h5py', h5py.__version__, '| arch', platform.machine())"

echo "=================================================="
if [ "$rc" -eq 0 ]; then
    echo "RESULT: SUCCESS — ARM venv built, full pytest suite green on aarch64"
else
    echo "RESULT: PYTEST FAILED (rc=$rc)"
fi
exit "$rc"
