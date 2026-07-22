"""Cluster runner: train + evaluate the RadDet YOLOv3 wideband-detection baseline.

Thin argparse wrapper over :mod:`rfbench.training_detection`. Trains a YOLOv3 detector on the
RadDet YOLO tree (ultralytics) then scores it through the canonical
:func:`rfbench.core.evaluate.evaluate` on RadDet's OFFICIAL test split, emitting a
``self_reported`` ``result.json`` (NOT committed to the leaderboard pending review).

Heavy deps (ultralytics/torch) load lazily inside the driver, so ``--skip-train`` +
``--weights`` re-scores an existing checkpoint, and ``--skip-eval`` trains only. Run on an ARM
compute node (never the login node) via the raddet venv:

    RFBENCH_CACHE=$WORK/data/rfbench_cache \\
    RFBENCH_RADDET_VARIANT=512_9T \\
    $WORK/envs/rfbench-arm-raddet/bin/python slurm/train_raddet_detection.py \\
        --epochs 100 --out $WORK/logs/detection/raddet/raddet_yolov3-seed42.json

See ``slurm/train_raddet_detection_arm.sh`` for the sbatch wrapper.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from rfbench.training_detection import (
    DEFAULT_EPOCHS,
    DEFAULT_EVAL_BATCH,
    DEFAULT_TRAIN_BATCH,
    evaluate_raddet_detector,
    train_raddet_detector,
)

logger = logging.getLogger("train_raddet_detection")


def _resolve_variant_root(variant: str) -> Path:
    """Return ``<RFBENCH_CACHE>/raddet/<variant>`` (or the flat tree) for the training data."""
    os.environ["RFBENCH_RADDET_VARIANT"] = variant
    from rfbench.data.download.detection_wbsig53 import (  # noqa: PLC0415
        _RADDET_SUBDIR,
        _raddet_variant_root,
    )
    from rfbench.data.prepare._common import resolve_cache_dir  # noqa: PLC0415

    return _raddet_variant_root(resolve_cache_dir() / _RADDET_SUBDIR)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default="512_9T", help="RadDet variant subtree (density/res)")
    parser.add_argument(
        "--arch", default="yolov3.yaml", help="ultralytics arch (yaml=from scratch)"
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--train-batch", type=int, default=DEFAULT_TRAIN_BATCH)
    parser.add_argument("--eval-batch", type=int, default=DEFAULT_EVAL_BATCH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None, help="ultralytics device spec, e.g. '0' or 'cpu'")
    parser.add_argument("--track", default="detection", choices=("detection", "recognition"))
    parser.add_argument("--project", default="runs/raddet", help="ultralytics run dir root")
    parser.add_argument("--run-name", default="yolov3_from_scratch")
    parser.add_argument("--weights", default=None, help="skip training; score this checkpoint")
    parser.add_argument("--out", required=True, help="result.json path (self_reported staging)")
    parser.add_argument("--no-torchmetrics", action="store_true", help="use the stdlib mAP path")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = _parse_args()
    eval_device = args.device if args.device is not None else "cuda"

    weights = args.weights
    if not args.skip_train and weights is None:
        variant_root = _resolve_variant_root(args.variant)
        logger.info("RadDet variant root: %s", variant_root)
        weights = str(
            train_raddet_detector(
                variant_root=variant_root,
                arch=args.arch,
                epochs=args.epochs,
                imgsz=args.imgsz,
                batch=args.train_batch,
                seed=args.seed,
                device=args.device,
                project=args.project,
                run_name=args.run_name,
            )
        )
    if weights is None:
        logger.error("no weights: pass --weights or drop --skip-train")
        return 2

    if args.skip_eval:
        logger.info("training done; skipping eval. weights=%s", weights)
        return 0

    result = evaluate_raddet_detector(
        weights=weights,
        track=args.track,
        variant=args.variant,
        out_path=args.out,
        use_torchmetrics=not args.no_torchmetrics,
        seed=args.seed,
        batch_size=args.eval_batch,
        device=eval_device,
        imgsz=args.imgsz,
    )
    values = result["metrics"]["values"]
    logger.info(
        "DONE track=%s mAP=%.4f mAR=%.4f IoU=%.4f -> %s (%s)",
        args.track,
        values.get("mAP", float("nan")),
        values.get("mAR", float("nan")),
        values.get("IoU", float("nan")),
        args.out,
        result["verification"]["status"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
