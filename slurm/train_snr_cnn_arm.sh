#!/bin/bash
# train_snr_cnn_arm.sh — Multi-seed GPU training du CNN régresseur SNR (snr_cnn) sur RadioML 2016.10a.
#
# Régression raw-IQ -> SNR dB (perte MSE, early-stop val, primary rmse_db). Sortie en STAGING
# (pas d'écrasement board) ; scripts/aggregate_multiseed.py promeut 42/43/44 en multi_seed_std.
# Cible : passer sous le RMSE 7.64 dB du snr_moment_ridge (DSP).
#
#   sbatch slurm/train_snr_cnn_arm.sh [MODEL [EPOCHS [SEED]]]
#     MODEL  default snr_cnn
#     EPOCHS default 100 (early-stop patience 10)
#     SEED   default 42 (utiliser 42/43/44 pour le board, 45 pour le verify)
#
#SBATCH --job-name=rfbench_snrcnn
#SBATCH --output=/lustre/work/pdl16831/udl79f933/logs/rfbench_snrcnn_%j.out
#SBATCH --error=/lustre/work/pdl16831/udl79f933/logs/rfbench_snrcnn_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00

set -uo pipefail
WORK=/lustre/work/pdl16831/udl79f933
REPO="${RFBENCH_REPO:-$WORK/projets/rf-benchmark-hub}"
VENV="$WORK/envs/rfbench-arm-gpu"
UV="$WORK/envs/uv-arm/uv"
MODEL="${1:-snr_cnn}"
EPOCHS="${2:-100}"
SEED="${3:-42}"
DATASET="radioml_2016_10a"

OUT_DIR="$WORK/logs/multiseed/snr_estimation"
OUT="$OUT_DIR/${MODEL}-seed${SEED}.json"
export RFBENCH_CACHE="$WORK/data/rfbench_cache"
export RFBENCH_HARDWARE="1x NVIDIA GB200"
export UV_PROJECT_ENVIRONMENT="$VENV"
export UV_CACHE_DIR="$WORK/.uv_cache_arm"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

echo "=== node=$(hostname) arch=$(uname -m) model=$MODEL epochs=$EPOCHS seed=$SEED date=$(date -Is) ==="
cd "$REPO" || { echo "REPO NOT FOUND: $REPO"; exit 2; }
mkdir -p "$OUT_DIR"

"$UV" run --no-sync python -c "
import rfbench, importlib
from rfbench.core.registry import MODELS
importlib.import_module('rfbench.models.baselines.$MODEL')
assert '$MODEL' in MODELS, 'model not registered: $MODEL'
import rfbench.training_snr
print('rfbench =', rfbench.__file__)
" || { echo "PREFLIGHT FAILED"; exit 5; }

echo "=== rfbench snr-train $MODEL on $DATASET (from_scratch, $EPOCHS ep, seed $SEED) ==="
"$UV" run --no-sync rfbench snr-train \
  --model "$MODEL" \
  --dataset "$DATASET" \
  --regime from_scratch \
  --epochs "$EPOCHS" \
  --batch-size 256 \
  --lr 1e-3 \
  --weight-decay 1e-4 \
  --patience 10 \
  --seed "$SEED" \
  --no-bootstrap \
  --device cuda \
  --out "$OUT"
rc=$?
echo "OUT=$OUT (rc=$rc)"; [ -f "$OUT" ] && head -c 800 "$OUT"; echo ""
exit "$rc"
