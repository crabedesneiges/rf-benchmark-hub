#!/bin/bash
# eval_fm_arm.sh — Evaluate a foundation model on AMC (RadioML 2016.10a) under linear_probe on
# an ARM GB200 node, emitting a schema-valid result.json. The FM weights must already be in
# $RFBENCH_CACHE (fetched on the frontend: python -m rfbench.models.foundation._download_lwm_spectro).
# The FM wrapper reads the cached checkpoint via torch.load (no huggingface_hub needed at eval time).
#
# Usage: sbatch slurm/eval_fm_arm.sh [MODEL] [REGIME]   # default lwm-spectro linear_probe
#SBATCH --job-name=rfbench_eval_fm
#SBATCH --output=/lustre/work/pdl16831/udl79f933/logs/rfbench_eval_fm_%j.out
#SBATCH --error=/lustre/work/pdl16831/udl79f933/logs/rfbench_eval_fm_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00

set -uo pipefail
WORK=/lustre/work/pdl16831/udl79f933
REPO="$WORK/projets/rf-benchmark-hub"
VENV="$WORK/envs/rfbench-arm-gpu"
export RFBENCH_CACHE="$WORK/data/rfbench_cache"
MODEL="${1:-lwm-spectro}"
REGIME="${2:-linear_probe}"
OUT="$REPO/leaderboard/results/amc/${MODEL}-${REGIME}.json"

echo "=== node=$(hostname) arch=$(uname -m) model=$MODEL regime=$REGIME date=$(date -Is) ==="
cd "$REPO" || { echo "REPO NOT FOUND"; exit 2; }

MODEL="$MODEL" REGIME="$REGIME" OUT="$OUT" "$VENV/bin/python" - <<'PY'
import os
import rfbench.models.foundation.lwm_spectro  # noqa: F401  (registers 'lwm-spectro')
import rfbench.tasks.amc  # noqa: F401  (registers 'amc')
from rfbench.core.registry import MODELS, get_task
from rfbench.core.model import Regime, RegimeSpec
from rfbench.models.foundation.base import run_regime
from rfbench.core.evaluate import evaluate

model_name = os.environ["MODEL"]
regime = Regime(os.environ["REGIME"])
out = os.environ["OUT"]

task = get_task("amc")
fm = MODELS.get(model_name)()
ds = next(d for d in task.datasets() if d.name == "radioml_2016_10a")
print(f"[eval-fm] {model_name} / {regime.value} on amc/{ds.name} — fitting probe on train split...")
train = ds.load("train")
adapted = run_regime(fm, RegimeSpec(regime), train)
print("[eval-fm] evaluating on test split...")
res = evaluate(adapted, task, "test", adapted.regime, dataset=ds.name, out_path=out)
vals = res["metrics"]["values"]
print(f"RESULT-FM acc_overall={vals.get('accuracy_overall')} macro_f1={vals.get('macro_f1')} -> {out}")
PY
rc=$?
echo "=================================================="
[ "$rc" -eq 0 ] && echo "RESULT: SUCCESS — $MODEL $REGIME evaluated -> $OUT" || echo "RESULT: EVAL FAILED (rc=$rc)"
exit "$rc"
