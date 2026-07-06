#!/bin/bash
# eval_fm_arm.sh — Evaluate a foundation model on AMC (RadioML 2016.10a, seed 42, FULL SNR range)
# on an ARM GB200 node, emitting a schema-valid result.json.
#
# ONLY the frozen-embedding regimes produce a meaningful FM row here:
#   * linear_probe (default) — fit a head on the frozen encoder embeddings (validated chain).
#   * few_shot K             — same, on a K-per-class support set (K is the 3rd arg).
# from_scratch / full_finetune are REFUSED: they read model.forward(), which for lwm-spectro is a
# FRESH UNTRAINED head over the frozen encoder (~chance). A real full_finetune needs a separate
# training loop (not yet implemented) that fits + saves the head/encoder the wrapper then loads.
#
# PREREQUISITES (run once, not in this job):
#   1. Weights in $RFBENCH_CACHE (fetched with huggingface_hub — pure Python, OK on the frontend):
#        python -m rfbench.models.foundation._download_lwm_spectro   # needs rfbench[foundation]
#      -> $RFBENCH_CACHE/lwm-spectro/checkpoints/checkpoint.pth (+ moe + experts).
#   2. RadioML 2016.10a downloaded + prepared (split index committed):
#        sbatch slurm/download_prepare_arm.sh radioml_2016_10a
# The FM wrapper reads the cached checkpoint via torch.load (no huggingface_hub needed at eval time).
#
# NOTE: the IQ->STFT preprocessing is UNVERIFIED (upstream ships no IQ->spectrogram code; the exact
# 512-FFT recipe is unpublished). Any score from this job is PROVISIONAL and must not be published
# as a faithful LWM-Spectro figure until the upstream spectrogram-generation config is confirmed —
# embed() logs a loud warning to that effect.
#
# Usage: sbatch slurm/eval_fm_arm.sh [MODEL] [REGIME] [K_SHOT]
#   sbatch slurm/eval_fm_arm.sh                       # lwm-spectro linear_probe
#   sbatch slurm/eval_fm_arm.sh lwm-spectro few_shot 5
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
export RFBENCH_HARDWARE="1x NVIDIA GB200"
MODEL="${1:-lwm-spectro}"
REGIME="${2:-linear_probe}"
K_SHOT="${3:-}"
if [ "$REGIME" = "few_shot" ] && [ -n "$K_SHOT" ]; then
  OUT="$REPO/leaderboard/results/amc/${MODEL}-${REGIME}-k${K_SHOT}.json"
else
  OUT="$REPO/leaderboard/results/amc/${MODEL}-${REGIME}.json"
fi

echo "=== node=$(hostname) arch=$(uname -m) model=$MODEL regime=$REGIME k_shot=${K_SHOT:-n/a} date=$(date -Is) ==="
cd "$REPO" || { echo "REPO NOT FOUND"; exit 2; }

MODEL="$MODEL" REGIME="$REGIME" K_SHOT="$K_SHOT" OUT="$OUT" "$VENV/bin/python" - <<'PY'
import os
import sys

import rfbench.models.foundation.lwm_spectro  # noqa: F401  (registers 'lwm-spectro')
import rfbench.tasks.amc  # noqa: F401  (registers 'amc')
from rfbench.core.registry import MODELS, get_task
from rfbench.core.model import Regime, RegimeSpec
from rfbench.models.foundation.base import run_regime
from rfbench.core.evaluate import evaluate

model_name = os.environ["MODEL"]
regime = Regime(os.environ["REGIME"])
k_shot = os.environ.get("K_SHOT", "").strip()
out = os.environ["OUT"]

if regime in (Regime.FROM_SCRATCH, Regime.FULL_FINETUNE):
    sys.exit(
        f"[eval-fm] REFUSED: regime '{regime.value}' evaluates model.forward(), which for "
        "lwm-spectro is a FRESH UNTRAINED head over the frozen encoder (~chance). This job only "
        "produces meaningful rows for linear_probe / few_shot (frozen embeddings + fitted head). "
        "A real full_finetune needs a separate training loop that fits + saves the head/encoder."
    )
if regime is Regime.FEW_SHOT:
    if not k_shot:
        sys.exit("[eval-fm] few_shot requires K_SHOT (3rd arg), e.g. `sbatch ... few_shot 5`")
    spec = RegimeSpec(regime, k_shot=int(k_shot))
else:
    spec = RegimeSpec(regime)

task = get_task("amc")
fm = MODELS.get(model_name)()
ds = next(d for d in task.datasets() if d.name == "radioml_2016_10a")
print(f"[eval-fm] {model_name} / {spec.name.value} on amc/{ds.name} — fitting head on train split...")
train = ds.load("train")
adapted = run_regime(fm, spec, train)
print("[eval-fm] evaluating on test split (full SNR range)...")
res = evaluate(adapted, task, "test", adapted.regime, dataset=ds.name, out_path=out)
vals = res["metrics"]["values"]
print(f"RESULT-FM acc_overall={vals.get('accuracy_overall')} macro_f1={vals.get('macro_f1')} -> {out}")
print("[eval-fm] REMINDER: score is PROVISIONAL — IQ->STFT preprocessing is UNVERIFIED.")
PY
rc=$?
echo "=================================================="
[ "$rc" -eq 0 ] && echo "RESULT: SUCCESS — $MODEL $REGIME evaluated -> $OUT" || echo "RESULT: EVAL FAILED (rc=$rc)"
exit "$rc"
