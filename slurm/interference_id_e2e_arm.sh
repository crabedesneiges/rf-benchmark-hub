#!/bin/bash
# interference_id_e2e_arm.sh — Download + prepare + train interf_cnn on the GNSS-jamming
# (interf_gnss6) set, end to end, on an ARM compute node. Data is public (Zenodo 4629685,
# CC-BY-4.0, ~1.9 GB, no login) so this needs no credentials.
#
# Usage: sbatch slurm/interference_id_e2e_arm.sh [EPOCHS]
#SBATCH --job-name=rfbench_interf
#SBATCH --output=/lustre/work/pdl16831/udl79f933/logs/rfbench_interf_%j.out
#SBATCH --error=/lustre/work/pdl16831/udl79f933/logs/rfbench_interf_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00

set -uo pipefail
WORK=/lustre/work/pdl16831/udl79f933
REPO="$WORK/projets/rf-benchmark-hub"
VENV_CPU="$WORK/envs/rfbench-arm"          # .[dev,data]: numpy/h5py/requests
VENV_GPU="$WORK/envs/rfbench-arm-gpu"      # .[dev,data,tasks,torch]: + torch/CUDA
DATASET="interf_gnss6"
TASK="interference_id"
MODEL="interf_cnn"
EPOCHS="${1:-100}"
REGIME="from_scratch"
OUT="$REPO/leaderboard/results/${TASK}/${MODEL}.json"

export RFBENCH_CACHE="$WORK/data/rfbench_cache"
export RFBENCH_HARDWARE="1x NVIDIA GB200"

echo "=== node=$(hostname) arch=$(uname -m) date=$(date -Is) ==="
cd "$REPO" || { echo "REPO NOT FOUND"; exit 2; }
mkdir -p "$(dirname "$OUT")"
echo "RFBENCH_CACHE=$RFBENCH_CACHE (free: $(df -h "$WORK" | awk 'NR==2{print $4}'))"

echo "=== [1/3] rfbench data download $DATASET (Zenodo 4629685, ~1.9 GB, no login) ==="
"$VENV_CPU/bin/rfbench" data download "$DATASET"
rc=$?
if [ "$rc" -ne 0 ]; then echo "RESULT: DOWNLOAD FAILED (rc=$rc)"; exit 4; fi

echo "=== [2/3] rfbench data prepare $DATASET (80/10/10 stratified, seed 42) ==="
"$VENV_CPU/bin/rfbench" data prepare "$DATASET" --out "$REPO/leaderboard"
rc=$?
if [ "$rc" -ne 0 ]; then echo "RESULT: PREPARE FAILED (rc=$rc)"; exit 5; fi
find "$REPO/leaderboard/splits/$DATASET" -name '*.idx.json' 2>/dev/null

echo "=== [3/3] rfbench train $MODEL on $TASK/$DATASET ($REGIME, $EPOCHS epochs) ==="
"$VENV_GPU/bin/rfbench" train \
  --task "$TASK" \
  --dataset "$DATASET" \
  --model "$MODEL" \
  --regime "$REGIME" \
  --epochs "$EPOCHS" \
  --batch-size 256 \
  --lr 1e-3 \
  --seed 42 \
  --device cuda \
  --out "$OUT"
rc=$?

echo "=== emitted result.json ==="
[ -f "$OUT" ] && head -c 2000 "$OUT"
echo ""
echo "=================================================="
if [ "$rc" -eq 0 ]; then
  echo "RESULT: SUCCESS — $MODEL trained on $DATASET; result.json at $OUT"
else
  echo "RESULT: TRAIN FAILED (rc=$rc)"
fi
exit "$rc"
