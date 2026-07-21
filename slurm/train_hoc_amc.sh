#!/bin/bash
# train_hoc_amc.sh — Fit a DSP / floor AMC baseline on the REAL RadioML 2016.10a train split and
# emit a schema-valid result.json under the multi-seed staging area (J2). CPU-ONLY: HOC+LR and the
# trivial floors have no GPU code, so this job requests NO --gres gpu.
#
# These baselines are DETERMINISTIC at seed 42 (HOC features are closed-form; LogisticRegression is
# seeded; majority_class is a fixed prior; chance uses a seeded RNG), so ONE run per model is the
# canonical row — no multi-seed sweep. The orchestrator decides the board path from the staging file.
#
# Models (registry names): hoc_lr | majority_class | chance
#
# The dataset must already be downloaded + prepared (split indices committed):
#   sbatch slurm/download_prepare_arm.sh radioml_2016_10a   # once
# then:
#   sbatch slurm/train_hoc_amc.sh [MODEL]
#     MODEL  default: hoc_lr   — one of hoc_lr | majority_class | chance
#
# Output (no write to leaderboard/results/ by this script):
#   result.json → $WORK/logs/multiseed/amc/<model>-seed42.json
#
#SBATCH --job-name=rfbench_hoc
#SBATCH --output=logs/rfbench_hoc_%j.out
#SBATCH --error=logs/rfbench_hoc_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
# cluster mono-partition (defq*, GB200/ARM uniquement) -- pas de contrainte d'architecture requise
# (confirmé via `sinfo -o "%P %f %c %G"`, seule feature reportée: location=local). CPU-only: no gpu.

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
# Derive REPO from SLURM_SUBMIT_DIR so the correct worktree's code is used regardless of which
# rfbench editable install is registered in the venv's .pth. Fall back to $PWD outside SLURM.
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
# Strip trailing /slurm if sbatch was run from the slurm/ subdirectory.
REPO="${REPO%/slurm}"
VENV="${RFBENCH_VENV_GPU:-$WORK/envs/rfbench-arm-gpu}"          # .[dev,data,tasks,torch] — numpy + sklearn present
MODEL="${1:-hoc_lr}"                        # hoc_lr | majority_class | chance; override as $1
DATASET="radioml_2016_10a"
SEED=42                                     # deterministic baselines — a single seed-42 run suffices

# Multi-seed staging — no direct write to leaderboard/results/ (orchestrator promotes rows).
OUT_DIR="$WORK/logs/multiseed/amc"
OUT="$OUT_DIR/${MODEL}-seed${SEED}.json"

# PYTHONPATH must precede the editable-install .pth so `import rfbench` resolves to THIS worktree.
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export RFBENCH_CACHE="${RFBENCH_CACHE:-$WORK/data/rfbench_cache}"
export RFBENCH_HARDWARE="CPU (frontend DSP baseline)"

echo "=== node=$(hostname) arch=$(uname -m) model=$MODEL seed=$SEED date=$(date -Is) ==="
echo "REPO=$REPO"
cd "$REPO" || { echo "REPO NOT FOUND: $REPO"; exit 2; }
echo "RFBENCH_CACHE=$RFBENCH_CACHE"
echo "OUT=$OUT"
mkdir -p "$OUT_DIR"

# Rename the SLURM log retroactively to include the model (best-effort, non-blocking).
[ -n "${SLURM_JOB_ID:-}" ] && \
  scontrol update JobId="$SLURM_JOB_ID" JobName="rfbench_hoc_${MODEL}" 2>/dev/null || true

echo "=== rfbench fit+eval $MODEL on $DATASET (from_scratch, seed $SEED, full SNR range) ==="
MODEL="$MODEL" OUT="$OUT" DATASET="$DATASET" "$VENV/bin/python" - <<'PY'
import os

# Explicit imports register the J2 baselines (@register_model) and the AMC task — the CLI's static
# _MODEL_MODULES table is not used here, mirroring eval_fm_arm.sh's explicit foundation imports.
import rfbench.models.baselines.hoc_amc  # noqa: F401  (registers 'hoc_lr')
import rfbench.models.baselines.trivial_amc  # noqa: F401  (registers 'majority_class' + 'chance')
import rfbench.tasks.amc  # noqa: F401  (registers 'amc')
from rfbench.core.evaluate import evaluate
from rfbench.core.registry import MODELS, get_task

model_name = os.environ["MODEL"]
out = os.environ["OUT"]
dataset = os.environ["DATASET"]

task = get_task("amc")
model = MODELS.get(model_name)()
ds = next(d for d in task.datasets() if d.name == dataset)

print(f"[hoc] {model_name} on amc/{ds.name} — fitting on train split...")
model.fit(ds.load("train"))
print("[hoc] evaluating on test split (full SNR range, bootstrap CI on)...")
# device is irrelevant for these CPU baselines (forward returns pure Python lists); evaluate's
# 'device' arg is only forwarded to torch models, so 'cpu' here is a harmless no-op.
res = evaluate(model, task, "test", model.regime, dataset=ds.name, device="cpu", out_path=out)
vals = res["metrics"]["values"]
print(f"RESULT-HOC acc_overall={vals.get('accuracy_overall')} macro_f1={vals.get('macro_f1')} -> {out}")
PY
rc=$?

echo "=== emitted result.json ==="
[ -f "$OUT" ] && head -c 2000 "$OUT"

echo ""
echo "=================================================="
if [ "$rc" -eq 0 ]; then
  echo "RESULT: SUCCESS — $MODEL seed=$SEED fit+evaluated; result.json at $OUT"
else
  echo "RESULT: FIT/EVAL FAILED (rc=$rc)"
fi
exit "$rc"
