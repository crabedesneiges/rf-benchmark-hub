#!/bin/bash
# prepare_openset_sei.sh — Generate ONLY the WiSig open-set split (held-out transmitters) in the
# ecstatic integration worktree, from the already-cached ManyTx.pkl. CPU-only, ARM, no GPU.
#
# The three closed-set WiSig splits are already committed and untouched: this writes just
#   leaderboard/splits/wisig/sei-wisig-openset-heldouttx-8010-seed42-v1.{idx,manifest}.json
# via a direct prepare_sei("wisig", "open_set", ...) call (no `rfbench data prepare`, which would
# regenerate every condition). Commit the two new files, then score with train_sei_arm.sh open_set.
#
#SBATCH --job-name=sei_openset_prep
#SBATCH --output=logs/rfbench_sei_openset_prep_%j.out
#SBATCH --error=logs/rfbench_sei_openset_prep_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
# NB: Dalia forbids --mem (RAM is auto-allocated proportional to cores); 16 cores gives ample
# headroom for the ~4.2 GB ManyTx.pkl load.

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
REPO="${RFBENCH_REPO:-$WORK/projets/rf-benchmark-hub/.claude/worktrees/ecstatic-torvalds-a6ced8}"
VENV="${RFBENCH_VENV_CPU:-$WORK/envs/rfbench-arm}"   # .[dev,data]: numpy present (no torch needed for split generation)

export RFBENCH_CACHE="${RFBENCH_CACHE:-$WORK/data/rfbench_cache}"
# Force THIS worktree's rfbench (with the open_set condition) ahead of any editable install.
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

echo "=== node=$(hostname) arch=$(uname -m) date=$(date -Is) ==="
cd "$REPO" || { echo "REPO NOT FOUND: $REPO"; exit 2; }
echo "RFBENCH_CACHE=$RFBENCH_CACHE"
[ -f "$RFBENCH_CACHE/wisig/ManyTx.pkl" ] || { echo "NO WiSig ManyTx.pkl in cache"; exit 3; }

"$VENV/bin/python" - <<'PY'
import rfbench
from rfbench.data.prepare.sei import CANONICAL_SPLIT_IDS, load_wisig_records, prepare_sei

print("rfbench:", rfbench.__file__)
assert "open_set" in CANONICAL_SPLIT_IDS["wisig"], "this worktree lacks the open_set condition"

records = load_wisig_records()  # reads $RFBENCH_CACHE/wisig/ManyTx.pkl (lazy numpy)
print("wisig records:", len(records))

split, manifest = prepare_sei("wisig", "open_set", out_dir="leaderboard", records=records)
sizes = {name: len(idx) for name, idx in split.indices.items()}
print("split id     :", split.canonical_split_id)
print("checksum     :", split.checksum)
print("partition    :", sizes)
print("n_items      :", manifest.n_items)
PY
rc=$?

echo "=== written open-set split ==="
ls -la "$REPO/leaderboard/splits/wisig/" | grep openset

echo "=================================================="
if [ "$rc" -eq 0 ]; then
  echo "RESULT: SUCCESS — open-set split written; commit it, then sbatch train_sei_arm.sh <model> <ep> open_set"
else
  echo "RESULT: prepare FAILED (rc=$rc)"
fi
exit "$rc"
