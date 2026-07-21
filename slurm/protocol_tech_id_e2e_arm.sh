#!/bin/bash
# protocol_tech_id_e2e_arm.sh — Download + prepare + train the tprime baseline on T-PRIME
# DS 3.0 (tprime_wifi4: real OTA 802.11 b/g/n/ax, single-protocol captures), end to end, on
# an ARM compute node.
#
# DS 3.0 is the CONFIRMED single-protocol capture (item neu:h989s8519; see
# rfbench/data/download/protocol_tprime.py header) -- do NOT swap in DS 3.1-3.4, which are
# multi-protocol overlap-detection captures for a different task. No login required, but the
# DRS item-download link can rot: if step [1/3] 404s, resolve
# http://hdl.handle.net/2047/D20621423 by hand and re-run with:
#   sbatch protocol_tech_id_e2e_arm.sh [EPOCHS] <resolved_url>
#
# Usage: sbatch slurm/protocol_tech_id_e2e_arm.sh [EPOCHS] [SOURCE_URL]
#SBATCH --job-name=rfbench_tprime
#SBATCH --output=logs/rfbench_tprime_%j.out
#SBATCH --error=logs/rfbench_tprime_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
# cluster mono-partition (defq*, GB200/ARM uniquement) -- pas de contrainte d'architecture requise
# (confirmé via `sinfo -o "%P %f %c %G"`, seule feature reportée: location=local)

set -uo pipefail
# --- Portable config (override via environment; see slurm/README.md) -----------------
#   WORK                Lustre work root (REQUIRED; usually pre-set by the cluster).
#   RFBENCH_REPO        repo/worktree checkout to run       (default: $WORK/projets/rf-benchmark-hub[...]).
#   RFBENCH_VENV_CPU    CPU venv  .[dev,data]               (default: $WORK/envs/rfbench-arm).
#   RFBENCH_VENV_GPU    GPU venv  .[dev,data,tasks,torch]   (default: $WORK/envs/rfbench-arm-gpu).
#   RFBENCH_VENV_DETECTION  detection venv .[dev,detection] (default: $WORK/envs/rfbench-arm-detection).
#   RFBENCH_UV          uv binary for this arch             (default: $WORK/envs/uv-arm/uv).
#   RFBENCH_CACHE       dataset cache root                  (default: $WORK/data/rfbench_cache).
# SLURM logs go to logs/ relative to the submit dir: create it first (mkdir -p logs) or
# override with `sbatch --output=... --error=...`.
# ------------------------------------------------------------------------------------
WORK="${WORK:?set \$WORK to your Lustre work dir (e.g. /lustre/work/<project>/<user>)}"
REPO="${RFBENCH_REPO:-$WORK/projets/rf-benchmark-hub}"
VENV_CPU="${RFBENCH_VENV_CPU:-$WORK/envs/rfbench-arm}"          # .[dev,data]: numpy/h5py/requests
VENV_GPU="${RFBENCH_VENV_GPU:-$WORK/envs/rfbench-arm-gpu}"      # .[dev,data,tasks,torch]: + torch/CUDA
DATASET="tprime_wifi4"
TASK="protocol_tech_id"
MODEL="tprime"
EPOCHS="${1:-50}"
SOURCE_URL="${2:-}"                        # optional DRS-link override (see header)
REGIME="from_scratch"
OUT="$REPO/leaderboard/results/${TASK}/${MODEL}.json"

export RFBENCH_CACHE="${RFBENCH_CACHE:-$WORK/data/rfbench_cache}"
export RFBENCH_HARDWARE="1x NVIDIA GB200"

echo "=== node=$(hostname) arch=$(uname -m) date=$(date -Is) ==="
cd "$REPO" || { echo "REPO NOT FOUND"; exit 2; }
mkdir -p "$(dirname "$OUT")"
echo "RFBENCH_CACHE=$RFBENCH_CACHE (free: $(df -h "$WORK" | awk 'NR==2{print $4}'))"

echo "=== [1/4] rfbench data download $DATASET (T-PRIME DS 3.0, item neu:h989s8519, ~GBs) ==="
if [ -n "$SOURCE_URL" ]; then
  "$VENV_CPU/bin/rfbench" data download "$DATASET" --source-url "$SOURCE_URL"
else
  "$VENV_CPU/bin/rfbench" data download "$DATASET"
fi
rc=$?
if [ "$rc" -ne 0 ]; then
  echo "RESULT: DOWNLOAD FAILED (rc=$rc) -- if this is a 404 on the DRS link, resolve"
  echo "http://hdl.handle.net/2047/D20621423 by hand and re-run with the resolved URL as \$2."
  exit 4
fi

echo "=== [2/4] confirm extracted layout (class sub-dirs, .bin dtype) before spending compute ==="
find "$RFBENCH_CACHE/$DATASET" -maxdepth 3 -type d 2>/dev/null | head -30
find "$RFBENCH_CACHE/$DATASET" -type f 2>/dev/null | head -5
echo "^^^ if class sub-dir names don't match rfbench.data.prepare.protocol._class_dir_names(),"
echo "    update that map before prepare/train will find any files."

echo "=== [3/4] rfbench data prepare $DATASET (80/10/10 stratified, seed 42) ==="
"$VENV_CPU/bin/rfbench" data prepare "$DATASET" --out "$REPO/leaderboard"
rc=$?
if [ "$rc" -ne 0 ]; then echo "RESULT: PREPARE FAILED (rc=$rc)"; exit 5; fi
find "$REPO/leaderboard/splits/$DATASET" -name '*.idx.json' 2>/dev/null

echo "=== [4/4] rfbench train $MODEL on $TASK/$DATASET ($REGIME, $EPOCHS epochs, SM variant) ==="
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
