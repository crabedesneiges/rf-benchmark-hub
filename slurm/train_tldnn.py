"""Train the TLDNN AMC baseline from scratch on a RadioML dataset and emit a result.json.

Driver used by ``slurm/train_tldnn_arm.sh``. The generic ``rfbench train`` AMC path instantiates
baselines with no arguments (``MODELS.get("tldnn")()``), which builds the 2016.10a config
(11-class, window 128, K=2). This driver instead constructs TLDNN with the dataset's real class
count AND window so that 2018.01a (24-class, window 1024, K=4) trains correctly -- exactly the
pattern of ``$WORK/slurm_scripts/train_amc2018.py`` for MCLDNN.

Environment:
    DATASET   radioml_2016_10a | radioml_2018_01a   (default: radioml_2016_10a)
    EPOCHS    training epochs                        (default: 150, the paper's budget)
    BATCH     batch size                             (default: 128 for 2016, 512 for 2018)
    LR        learning rate                          (default: 2e-4 -- see the collapse note)
    SEED      RNG seed                               (default: 42)
    OUT       result.json path                       (default: $WORK/logs/multiseed/amc/...)
    CKPT      checkpoint path                        (default: $WORK/checkpoints/multiseed/...)

lr note: TLDNN contains attention. Under the rfbench from-scratch loop (plain Adam, NO warmup) a
transformer-bearing model collapses at lr=1e-3 (documented: tprime 0.259->0.995 val). Train with
LR=2e-4, NOT the paper's AdamW-1e-3 (which relies on a warmup/schedule this loop does not run).
"""

from __future__ import annotations

import os

import rfbench  # noqa: F401  (ensures the installed package resolves)

print("rfbench:", rfbench.__file__, flush=True)

import rfbench.models.baselines.tldnn  # noqa: E402,F401  (registers "tldnn")
import rfbench.tasks.amc  # noqa: E402,F401  (registers the amc task)
from rfbench.core.model import Regime, RegimeSpec  # noqa: E402
from rfbench.core.registry import MODELS, get_task  # noqa: E402
from rfbench.training import resolve_amc_dataset, train_baseline  # noqa: E402
from rfbench.training_sei import count_classes  # noqa: E402

WORK = os.environ["WORK"]
DATASET = os.environ.get("DATASET", "radioml_2016_10a")
SEED = int(os.environ.get("SEED", "42"))
EPOCHS = int(os.environ.get("EPOCHS", "150"))
LR = float(os.environ.get("LR", "2e-4"))
# Per-dataset window + paper batch size.
WINDOW = 1024 if DATASET == "radioml_2018_01a" else 128
BATCH = int(os.environ.get("BATCH", "512" if DATASET == "radioml_2018_01a" else "128"))

OUT = os.environ.get("OUT", f"{WORK}/logs/multiseed/amc/tldnn-{DATASET}-seed{SEED}.json")
CKPT = os.environ.get("CKPT", f"{WORK}/checkpoints/multiseed/tldnn-{DATASET}-seed{SEED}.pt")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
os.makedirs(os.path.dirname(CKPT), exist_ok=True)

task = get_task("amc")
dataset = resolve_amc_dataset(task, DATASET)
num_classes = count_classes(dataset)
print(f"{DATASET}: num_classes={num_classes} window={WINDOW}", flush=True)

# window drives the conv depth K (2 for 128, 4 for 1024) inside TLDNNNet.
model = MODELS.get("tldnn")(num_classes=num_classes, window=WINDOW, device="cuda")
spec = RegimeSpec(name=Regime("from_scratch"), k_shot=None)

print(
    f"=== train tldnn on {DATASET} (from_scratch, {num_classes}-class, {EPOCHS} ep, "
    f"bs={BATCH}, lr={LR}, seed={SEED}) ===",
    flush=True,
)
_model, result = train_baseline(
    task,
    model,
    dataset,
    regime=spec,
    epochs=EPOCHS,
    batch_size=BATCH,
    lr=LR,
    seed=SEED,
    device="cuda",
    out_path=OUT,
    checkpoint_out=CKPT,
)
print("accuracy_overall =", result["metrics"]["values"]["accuracy_overall"], flush=True)
print(f"OUT={OUT}")
print("TLDNN_TRAIN_OK")
