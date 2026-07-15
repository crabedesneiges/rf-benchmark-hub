#!/bin/bash
# train_sei_multiseed_arm.sh — Multi-seed SEI training for the tier-`verified` upgrade (LOT 3).
#
# Same recipe as slurm/train_sei_arm.sh (rfbench sei-train, per-model LR/batch/patience/window),
# but PARAMETRIZED BY SEED and writing to a STAGING path instead of overwriting the board rows:
#   $WORK/logs/multiseed/sei/<MODEL>-<TRACK>-seed<SEED>.json
# scripts/aggregate_multiseed.py then promotes seeds 42/43/44 to a multi_seed_std board row (adds the
# ±1σ band, like AMC/interf), and a fresh seed (45) re-run is checked within 2σ by `rfbench verify`.
#
#   sbatch slurm/train_sei_multiseed_arm.sh [MODEL [EPOCHS [TRACKS [SEED [DATASET]]]]]
#     MODEL   default wisig_cnn_paper  (also: complex_cnn, resnet1d_sei)
#     EPOCHS  default 100 (early stopping usually stops sooner)
#     TRACKS  default "closed_set cross_receiver cross_day"
#     SEED    default 42  (use 42/43/44 for the board mean, 45 for the verify re-run)
#     DATASET default wisig
#   e.g. sbatch slurm/train_sei_multiseed_arm.sh complex_cnn 100 "closed_set cross_receiver cross_day" 43
#
#SBATCH --job-name=sei_ms
#SBATCH --output=/lustre/work/pdl16831/udl79f933/logs/rfbench_sei_ms_%j.out
#SBATCH --error=/lustre/work/pdl16831/udl79f933/logs/rfbench_sei_ms_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00

set -uo pipefail
WORK=/lustre/work/pdl16831/udl79f933
# The GPU venv is editable-installed against the MAIN repo, where the SEI code (models + training_sei
# + sei-train CLI) now lives after the J3 merge — so point REPO at main (override with RFBENCH_REPO).
REPO="${RFBENCH_REPO:-$WORK/projets/rf-benchmark-hub}"
VENV="$WORK/envs/rfbench-arm-gpu"
UV="$WORK/envs/uv-arm/uv"
MODEL="${1:-wisig_cnn_paper}"
EPOCHS="${2:-100}"
TRACKS="${3:-closed_set cross_receiver cross_day}"
SEED="${4:-42}"
DATASET="${5:-wisig}"

OUT_DIR="$WORK/logs/multiseed/sei"
export RFBENCH_CACHE="$WORK/data/rfbench_cache"
export RFBENCH_HARDWARE="1x NVIDIA GB200"
export UV_PROJECT_ENVIRONMENT="$VENV"
export UV_CACHE_DIR="$WORK/.uv_cache_arm"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

# Per-model recipe — MUST match slurm/train_sei_arm.sh so the seeds are comparable to the board row.
case "$MODEL" in
  wisig_cnn_paper|wisig_cnn) LR="5e-4"; BATCH=32; PATIENCE=5; WINDOW=256 ;;
  complex_cnn) LR="1e-3"; BATCH=64; PATIENCE=8; WINDOW=256 ;;
  resnet1d_sei) LR="1e-3"; BATCH=128; PATIENCE=8; WINDOW=256 ;;
  *) LR="5e-4"; BATCH=32; PATIENCE=5; WINDOW=256 ;;
esac
if [ "$MODEL" = "resnet1d_sei" ]; then WD="1e-4"; L2="0.0"; else WD="0.0"; L2="1e-4"; fi

echo "=== node=$(hostname) arch=$(uname -m) model=$MODEL epochs=$EPOCHS tracks='$TRACKS' seed=$SEED date=$(date -Is) ==="
cd "$REPO" || { echo "REPO NOT FOUND: $REPO"; exit 2; }
mkdir -p "$OUT_DIR"
echo "RFBENCH_CACHE=$RFBENCH_CACHE  recipe: lr=$LR batch=$BATCH patience=$PATIENCE window=$WINDOW l2=$L2 wd=$WD  out=$OUT_DIR"

# Pre-flight: confirm the loaded rfbench carries the SEI model + sei-train.
"$UV" run --no-sync python -c "
import rfbench, importlib
from rfbench.core.registry import MODELS
importlib.import_module('rfbench.models.baselines.$MODEL')
assert '$MODEL' in MODELS, 'model not registered: $MODEL'
import rfbench.training_sei
print('rfbench =', rfbench.__file__)
" || { echo "PREFLIGHT FAILED"; exit 5; }

rc_all=0
for TRACK in $TRACKS; do
  OUT="$OUT_DIR/${MODEL}-${TRACK}-seed${SEED}.json"
  echo ""
  echo "=== rfbench sei-train $MODEL on $DATASET track=$TRACK ($EPOCHS ep, seed $SEED) ==="
  "$UV" run --no-sync rfbench sei-train \
    --dataset "$DATASET" \
    --model "$MODEL" \
    --track "$TRACK" \
    --regime from_scratch \
    --window "$WINDOW" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH" \
    --lr "$LR" \
    --l2-lambda "$L2" \
    --weight-decay "$WD" \
    --patience "$PATIENCE" \
    --seed "$SEED" \
    --device cuda \
    --out "$OUT"
  rc=$?
  [ "$rc" -ne 0 ] && rc_all=$rc
  echo "--- emitted ($TRACK seed $SEED) ---"; [ -f "$OUT" ] && head -c 600 "$OUT"; echo ""
done

echo ""
if [ "$rc_all" -eq 0 ]; then
  echo "RESULT: SUCCESS — $MODEL seed $SEED tracks '$TRACKS' -> $OUT_DIR"
else
  echo "RESULT: at least one track FAILED (rc=$rc_all)"
fi
exit "$rc_all"
