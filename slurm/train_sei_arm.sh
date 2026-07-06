#!/bin/bash
# train_sei_arm.sh — Train an SEI baseline from scratch on the REAL WiSig ManyTx across the three
# closed-set conditions (closed_set / cross_receiver / cross_day), each scored + emitted as its own
# schema-valid result.json under leaderboard/results/sei/ (M3, feat/sei-complete).
#
# Uses the SEI-specific training loop (rfbench.training_sei via `rfbench sei-train`): class-weighted
# CE, explicit L2 on the model's regularised kernels, best checkpoint + early stop on validation
# LOSS. The shared AMC loop (rfbench.training) is untouched. The WiSig split indices are already
# committed, so NO data-prepare step is needed here (only ManyTx.pkl must be in $RFBENCH_CACHE).
#
#   sbatch slurm/train_sei_arm.sh [MODEL] [EPOCHS] [TRACKS]
#     MODEL   default wisig_cnn_paper (also: complex_cnn, resnet1d_sei, oracle_cnn[oracle only])
#     EPOCHS  default 100 (WiSig d006 max; early stopping usually stops sooner). Use a SMALL value
#             (e.g. 3) for a quick validation run before the full one.
#     TRACKS  default "closed_set cross_receiver cross_day"
#   e.g. VALIDATION: sbatch slurm/train_sei_arm.sh wisig_cnn_paper 3 closed_set
#        FULL:       sbatch slurm/train_sei_arm.sh wisig_cnn_paper 100
#
#SBATCH --job-name=sei_train
#SBATCH --output=/lustre/work/pdl16831/udl79f933/logs/rfbench_sei_%j.out
#SBATCH --error=/lustre/work/pdl16831/udl79f933/logs/rfbench_sei_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00

set -uo pipefail
WORK=/lustre/work/pdl16831/udl79f933
REPO="$WORK/projets/rf-benchmark-hub-sei"      # the feat/sei-complete worktree
VENV="$WORK/envs/rfbench-arm-gpu"              # .[dev,data,tasks,torch] — torch + CUDA present
UV="$WORK/envs/uv-arm/uv"
MODEL="${1:-wisig_cnn_paper}"
EPOCHS="${2:-100}"
TRACKS="${3:-closed_set cross_receiver cross_day}"
DATASET="wisig"

export RFBENCH_CACHE="$WORK/data/rfbench_cache"
export RFBENCH_HARDWARE="1x NVIDIA GB200"
export UV_PROJECT_ENVIRONMENT="$VENV"           # reuse the prebuilt GPU venv (no re-sync)
export UV_CACHE_DIR="$WORK/.uv_cache_arm"
# The GPU venv is editable-installed against the MAIN repo, so force this worktree's rfbench onto
# the path (PYTHONPATH overrides the editable .pth) — otherwise the SEI code below never loads.
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

# Per-model recipe (WiSig d006 for wisig_cnn_paper; ORACLE 1e-4/patience-10; others sensible).
case "$MODEL" in
  oracle_cnn) LR="1e-4"; BATCH=32;  PATIENCE=10; WINDOW=128; DATASET="oracle" ;;
  wisig_cnn_paper|wisig_cnn) LR="5e-4"; BATCH=32; PATIENCE=5; WINDOW=256 ;;
  complex_cnn) LR="1e-3"; BATCH=64; PATIENCE=8; WINDOW=256 ;;
  resnet1d_sei) LR="1e-3"; BATCH=128; PATIENCE=8; WINDOW=256 ;;
  *) LR="5e-4"; BATCH=32; PATIENCE=5; WINDOW=256 ;;
esac
# ResNet-1D has no l2_penalty hook -> use Adam weight_decay for its L2 instead.
if [ "$MODEL" = "resnet1d_sei" ]; then WD="1e-4"; L2="0.0"; else WD="0.0"; L2="1e-4"; fi

echo "=== node=$(hostname) arch=$(uname -m) model=$MODEL epochs=$EPOCHS tracks='$TRACKS' date=$(date -Is) ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>&1 | head -4 \
  || echo "(nvidia-smi indispo)"
cd "$REPO" || { echo "REPO NOT FOUND: $REPO"; exit 2; }
echo "RFBENCH_CACHE=$RFBENCH_CACHE  recipe: lr=$LR batch=$BATCH patience=$PATIENCE window=$WINDOW l2=$L2 wd=$WD"

# Pre-flight: confirm THIS worktree's rfbench (with the SEI models + sei-train) is what loads.
"$UV" run --no-sync python -c "
import rfbench, importlib
from rfbench.core.registry import MODELS
importlib.import_module('rfbench.models.baselines.$MODEL')
assert '$MODEL' in MODELS, 'model not registered: $MODEL'
import rfbench.training_sei  # SEI loop must exist
print('rfbench (sei worktree) =', rfbench.__file__)
" || { echo "PREFLIGHT FAILED: wrong rfbench or missing SEI code"; exit 5; }

rc_all=0
for TRACK in $TRACKS; do
  OUT="$REPO/leaderboard/results/sei/${MODEL}-${TRACK}.json"
  mkdir -p "$(dirname "$OUT")"
  echo ""
  echo "=== rfbench sei-train $MODEL on $DATASET track=$TRACK ($EPOCHS ep, seed 42) ==="
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
    --seed 42 \
    --device cuda \
    --out "$OUT"
  rc=$?
  [ "$rc" -ne 0 ] && rc_all=$rc
  echo "--- emitted ($TRACK) ---"; [ -f "$OUT" ] && head -c 1200 "$OUT"; echo ""
done

echo ""
echo "=================================================="
if [ "$rc_all" -eq 0 ]; then
  echo "RESULT: SUCCESS — $MODEL trained on tracks '$TRACKS'; result.json under leaderboard/results/sei/"
else
  echo "RESULT: at least one track FAILED (rc=$rc_all)"
fi
exit "$rc_all"
