#!/bin/bash
# train_raddet_detection_arm.sh — Train the RadDet YOLOv3 wideband-DETECTION baseline (ultralytics)
# on the real RadDet spectrogram tree and emit a schema-valid result.json on the OFFICIAL test
# split (M3/WP-22; detection, not classification — a whole different stack).
#
# This is DETECTION (T-F bounding boxes, mAP), so it needs the raddet extra (ultralytics + torch),
# NOT the torchsig `detection` extra. RadDet must already be downloaded + extracted under
# $RFBENCH_CACHE/raddet/<variant>/{images,labels}/{train,val,test} (Kaggle, credential-gated;
# see rfbench/data/download/detection_wbsig53.py). The canonical split
# `detect-raddet-detection-official-v1` is committed; we adopt it verbatim.
#
#   sbatch slurm/train_raddet_detection_arm.sh [VARIANT [EPOCHS [SEED]]]
#     VARIANT default: 512_9T  — RadDet resolution/density subtree (RFBENCH_RADDET_VARIANT)
#     EPOCHS  default: 100     — YOLOv3 from-scratch training epochs
#     SEED    default: 42      — RNG seed
#
# Output locations (NO write to leaderboard/results/ — result is self_reported, pending review):
#   result.json  → $WORK/logs/detection/raddet/raddet_yolov3-<variant>-seed<seed>.json
#   checkpoints  → $WORK/checkpoints/detection/raddet_yolov3-<variant>-seed<seed>/  (ultralytics run dir)
#
#SBATCH --job-name=rfbench_raddet_det
#SBATCH --output=logs/rfbench_raddet_det_%j.out
#SBATCH --error=logs/rfbench_raddet_det_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
# cluster mono-partition (defq*, GB200/ARM uniquement) -- pas de contrainte d'architecture requise.

set -uo pipefail
# --- Portable config (override via environment; see slurm/README.md) -----------------
#   WORK                   Lustre work root (REQUIRED; usually pre-set by the cluster).
#   RFBENCH_VENV_RADDET    detection venv .[dev,data,tasks,torch,raddet]  (ultralytics + torch).
#                          Build it on an ARM compute node:
#                            $RFBENCH_UV venv $WORK/envs/rfbench-arm-raddet --python 3.10
#                            $RFBENCH_UV pip install -p $WORK/envs/rfbench-arm-raddet \
#                              -e "$REPO[dev,data,tasks,torch,raddet]"
#   RFBENCH_CACHE          dataset cache root (RadDet lives under $RFBENCH_CACHE/raddet).
#   RFBENCH_RADDET_VARIANT RadDet variant subtree (default 512_9T; also passed as $1).
# ------------------------------------------------------------------------------------
WORK="${WORK:?set \$WORK to your Lustre work dir (e.g. /lustre/work/<project>/<user>)}"
REPO="${SLURM_SUBMIT_DIR:-$WORK/projets/rf-benchmark-hub}"
REPO="${REPO%/slurm}"                       # strip trailing /slurm if submitted from there
VENV="${RFBENCH_VENV_RADDET:-$WORK/envs/rfbench-arm-raddet}"   # .[dev,data,tasks,torch,raddet]
UV="${RFBENCH_UV:-$WORK/envs/uv-arm/uv}"    # ARM uv (login-node uv is x86 — never use it here)
export UV_CACHE_DIR="$WORK/.uv_cache_arm"
export UV_PYTHON_INSTALL_DIR="$WORK/.uv_python"
VARIANT="${1:-${RFBENCH_RADDET_VARIANT:-512_9T}}"
EPOCHS="${2:-100}"
SEED="${3:-42}"

# Repo code precedes any editable-install .pth so the submitted worktree wins.
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export RFBENCH_CACHE="${RFBENCH_CACHE:-$WORK/data/rfbench_cache}"
export RFBENCH_RADDET_VARIANT="$VARIANT"
export RFBENCH_HARDWARE="1x NVIDIA GB200"

OUT_DIR="$WORK/logs/detection/raddet"
OUT="$OUT_DIR/raddet_yolov3-${VARIANT}-seed${SEED}.json"
RUN_DIR="$WORK/checkpoints/detection"
RUN_NAME="raddet_yolov3-${VARIANT}-seed${SEED}"

echo "=== node=$(hostname) arch=$(uname -m) variant=$VARIANT epochs=$EPOCHS seed=$SEED date=$(date -Is) ==="
echo "REPO=$REPO"
echo "VENV=$VENV"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>&1 | head -4 \
  || echo "(nvidia-smi indispo)"
cd "$REPO" || { echo "REPO NOT FOUND: $REPO"; exit 2; }
echo "RFBENCH_CACHE=$RFBENCH_CACHE  RFBENCH_RADDET_VARIANT=$RFBENCH_RADDET_VARIANT"
echo "OUT=$OUT"
mkdir -p "$OUT_DIR" "$RUN_DIR"

# --- Self-provision the raddet venv on the ARM node if absent (idempotent) -----------
# ultralytics + torch must be built HERE (ARM aarch64), never on the x86 login node. Mirrors
# slurm/setup_gpu_venv.sh; skipped when the venv already exists.
if [ ! -x "$VENV/bin/python" ]; then
  echo "=== raddet venv missing at $VENV; building .[dev,data,tasks,torch,raddet] via $UV ==="
  "$UV" venv --python 3.11 "$VENV" || { echo "VENV FAILED"; exit 3; }
  "$UV" pip install --python "$VENV/bin/python" -e "$REPO[dev,data,tasks,torch,raddet]" \
    || { echo "INSTALL FAILED"; exit 4; }
fi
"$VENV/bin/python" -c "import ultralytics, torch; print('ultralytics', ultralytics.__version__, '| torch', torch.__version__, '| cuda', torch.cuda.is_available())" \
  || { echo "raddet venv unusable (ultralytics/torch import failed)"; exit 5; }

[ -n "${SLURM_JOB_ID:-}" ] && \
  scontrol update JobId="$SLURM_JOB_ID" JobName="rfbench_raddet_det_${VARIANT}_s${SEED}" 2>/dev/null || true

echo "=== train + eval RadDet YOLOv3 (from_scratch, $EPOCHS epochs, seed $SEED) ==="
"$VENV/bin/python" slurm/train_raddet_detection.py \
  --variant "$VARIANT" \
  --arch yolov3.yaml \
  --epochs "$EPOCHS" \
  --imgsz 512 \
  --seed "$SEED" \
  --device 0 \
  --track detection \
  --project "$RUN_DIR" \
  --run-name "$RUN_NAME" \
  --out "$OUT"
rc=$?

echo "=== emitted result.json ==="
[ -f "$OUT" ] && head -c 2000 "$OUT"

echo ""
echo "=================================================="
if [ "$rc" -eq 0 ]; then
  echo "RESULT: SUCCESS — RadDet YOLOv3 seed=$SEED ; self_reported result.json at $OUT"
  echo "        (NOT committed to leaderboard/results — review before submit.)"
else
  echo "RESULT: TRAIN/EVAL FAILED (rc=$rc)"
fi
exit "$rc"
