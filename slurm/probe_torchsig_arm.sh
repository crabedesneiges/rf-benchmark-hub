#!/bin/bash
# probe_torchsig_arm.sh — Recon on an ARM node: does `.[detection]` (torchsig+torch)
# install on aarch64, and what is the WBSig53 API surface? De-risks writing the real
# WBSig53 loader (rfbench/data/download/detection_wbsig53.py, currently a lazy stub).
# CPU-only (no --gres): torchsig signal generation is CPU; we only PROBE here.
#
# Usage: sbatch slurm/probe_torchsig_arm.sh
#SBATCH --job-name=rfbench_torchsig_probe
#SBATCH --output=/lustre/work/pdl16831/udl79f933/logs/rfbench_torchsig_probe_%j.out
#SBATCH --error=/lustre/work/pdl16831/udl79f933/logs/rfbench_torchsig_probe_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:40:00

set -uo pipefail
WORK=/lustre/work/pdl16831/udl79f933
REPO="$WORK/projets/rf-benchmark-hub"
VENV="$WORK/envs/rfbench-arm-detection"
UV="$WORK/envs/uv-arm/uv"
export UV_CACHE_DIR="$WORK/.uv_cache_arm"
export UV_PYTHON_INSTALL_DIR="$WORK/.uv_python"

echo "=== node=$(hostname) arch=$(uname -m) date=$(date -Is) ==="
echo "uv: $UV — $("$UV" --version 2>&1)"
cd "$REPO" || { echo "REPO NOT FOUND"; exit 2; }

echo "=== build venv @ $VENV ==="
"$UV" venv --python 3.11 "$VENV" || { echo "VENV BUILD FAILED"; exit 3; }

echo "=== install -e .[dev,detection] (torchsig + torch, aarch64) — heavy step ==="
"$UV" pip install --python "$VENV/bin/python" -e ".[dev,detection]"
rc_install=$?
if [ "$rc_install" -ne 0 ]; then echo "DETECTION INSTALL FAILED rc=$rc_install"; exit 4; fi

echo "=== probe torch / torchsig / WBSig53 API ==="
"$VENV/bin/python" - <<'PY'
import platform, importlib, pkgutil
print("arch", platform.machine(), "python", platform.python_version())
import torch
print("torch", torch.__version__, "| cuda_available", torch.cuda.is_available())
import torchsig
print("torchsig", getattr(torchsig, "__version__", "?"))
# Surface: where do WBSig53 / wideband generators live?
for modname in ("torchsig.datasets", "torchsig.datasets.wideband", "torchsig.datasets.datasets"):
    try:
        m = importlib.import_module(modname)
        names = [n for n in dir(m) if not n.startswith("_")]
        hits = [n for n in names if any(k in n.lower() for k in ("wide", "wbsig", "sig53", "detect"))]
        print(f"[{modname}] candidates: {hits[:20]}")
    except Exception as e:  # noqa: BLE001
        print(f"[{modname}] import failed: {type(e).__name__}: {e}")
# List torchsig submodules to help locate the WBSig53 builder + annotation format
try:
    import torchsig.datasets as d
    subs = [x.name for x in pkgutil.iter_modules(d.__path__)]
    print("torchsig.datasets submodules:", subs)
except Exception as e:  # noqa: BLE001
    print("submodule scan failed:", e)
PY
rc=$?
echo "=================================================="
if [ "$rc" -eq 0 ]; then echo "RESULT: SUCCESS — .[detection] installs on aarch64; WBSig53 API probed (see above)"; else echo "RESULT: PROBE FAILED rc=$rc"; fi
exit "$rc"
