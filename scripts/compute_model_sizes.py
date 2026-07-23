"""Compute exact ``n_params`` and FLOPs for RF-Benchmark-Hub's registered (implemented) models.

RUN ON AN ARM GPU NODE -- torch is required and is absent on the Intel frontend. This is a pure
MEASUREMENT tool; it never trains. For every model registered in ``rfbench.core.registry.MODELS``
that also appears on the board (has a ``leaderboard/results`` row), it derives the task's input
shape from the committed data, instantiates the model, and reports:

* ``n_params`` -- ``int(model.n_params)`` (the :class:`rfbench.core.model.Model` property),
* ``n_flops``  -- ``fvcore`` ``FlopCountAnalysis`` total for ONE forward pass: a hardware-
  INDEPENDENT compute proxy (fvcore counts fused multiply-adds ~= MACs; multiply by 2 for raw
  FLOPs). Reported verbatim as returned by fvcore.

Usage (from the repo root, on an ARM node with torch + fvcore installed)::

    uv pip install -e ".[size]"                          # once: fvcore (+ torch)
    uv run python scripts/compute_model_sizes.py         # print a JSON report to stdout
    uv run python scripts/compute_model_sizes.py --write  # + patch leaderboard/results/**/*.json

Input shapes are DATA-DRIVEN (task modality + dataset window, from the result rows) with the small
documented override tables below; tweak ``_DATASET_SPEC`` / ``_TASK_LAYOUT`` if a model needs a
different dummy input. Each model is measured independently and failures are reported per-model
(the tool never aborts the whole run), so a single unsupported model does not block the rest.
"""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parent.parent
_RESULTS = _REPO / "leaderboard" / "results"

#: task -> dummy-input layout. IQ tasks feed ``(B, 2, W)`` EXCEPT SEI which is time-major
#: ``(B, W, 2)``; detection feeds a 2-channel spectrogram image ``(B, 2, H, W)``.
_TASK_LAYOUT: dict[str, str] = {
    "amc": "iq_cw",
    "snr_estimation": "iq_cw",
    "protocol_tech_id": "iq_cw",
    "interference_id": "iq_cw",
    "spectrum_sensing": "iq_cw",
    "sei": "iq_wc",
    "wideband_detection": "spectrogram",
}

#: dataset -> (n_classes, window). Covers the datasets currently on the board; a ``None`` falls
#: back to the task default below.
_DATASET_SPEC: dict[str, tuple[int | None, int | None]] = {
    "radioml_2016_10a": (11, 128),
    "radioml_2018_01a": (24, 1024),
    "wisig": (None, 256),
    "oracle": (16, 128),
    "powder": (None, 256),
    "interf_gnss6": (6, None),
    "tprime_wifi4": (4, None),
    "raddet": (11, None),
}

#: task -> default window when the dataset does not pin one.
_TASK_DEFAULT_WINDOW: dict[str, int] = {
    "amc": 128,
    "snr_estimation": 128,
    "protocol_tech_id": 512,
    "interference_id": 512,
    "spectrum_sensing": 32,
    "sei": 256,
    "wideband_detection": 512,
}


def _model_specs() -> dict[str, dict[str, Any]]:
    """Map ``model_name -> {task, dataset, n_classes, window}`` from the committed board rows."""
    out: dict[str, dict[str, Any]] = {}
    for path in sorted(_RESULTS.rglob("*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        name = row.get("model", {}).get("name")
        if not name or name in out:
            continue
        task = row.get("task", {}).get("name", "")
        dataset = row.get("dataset", {}).get("name", "")
        n_classes, window = _DATASET_SPEC.get(dataset, (None, None))
        if window is None:
            window = _TASK_DEFAULT_WINDOW.get(task, 128)
        out[name] = {"task": task, "dataset": dataset, "n_classes": n_classes, "window": window}
    return out


def _build_dummy(task: str, window: int) -> object:
    """Build a batch-1 dummy input tensor for ``task`` (see ``_TASK_LAYOUT``)."""
    import torch

    layout = _TASK_LAYOUT.get(task, "iq_cw")
    if layout == "iq_wc":
        return torch.randn(1, window, 2)
    if layout == "spectrogram":
        return torch.randn(1, 2, window, window)
    win = 32 if task == "spectrum_sensing" else window
    return torch.randn(1, 2, win)


def _instantiate(cls: type, n_classes: int | None) -> object:
    """Instantiate ``cls`` passing ``num_classes`` (under whatever name it accepts) when known."""
    params = inspect.signature(cls.__init__).parameters
    kwargs: dict[str, Any] = {}
    if n_classes:
        for cname in ("num_classes", "n_classes", "num_subbands", "out_dim"):
            if cname in params:
                kwargs[cname] = n_classes
                break
    return cls(**kwargs)


def measure() -> dict[str, dict[str, Any]]:
    """Measure every registered model that appears on the board (ARM-only: imports torch)."""
    import importlib
    import pkgutil

    import torch

    import rfbench.models.baselines as _baselines
    import rfbench.models.foundation as _foundation
    from rfbench.core.registry import MODELS

    # Models register via @register_model on IMPORT of their (torch) module; the package
    # __init__ files are empty, so import every concrete model module to populate MODELS.
    for _pkg in (_baselines, _foundation):
        for _info in pkgutil.iter_modules(_pkg.__path__, _pkg.__name__ + "."):
            if _info.name.rsplit(".", 1)[-1].startswith("_"):
                continue  # helpers/templates (_template, _download_*, base) register nothing
            try:
                importlib.import_module(_info.name)
            except Exception:  # noqa: BLE001 - optional-dep models just do not register
                continue

    try:
        from fvcore.nn import FlopCountAnalysis
    except ImportError:
        FlopCountAnalysis = None

    report: dict[str, dict[str, Any]] = {}
    for name, spec in _model_specs().items():
        if name not in MODELS:
            continue  # a from_paper row with no in-repo implementation -> nothing to measure
        entry: dict[str, Any] = {"task": spec["task"], "dataset": spec["dataset"]}
        try:
            model = _instantiate(MODELS[name], spec["n_classes"]).eval()
            entry["n_params"] = int(model.n_params)
            dummy = _build_dummy(spec["task"], int(spec["window"]))
            entry["input_shape"] = list(dummy.shape)
            if FlopCountAnalysis is None:
                entry["warning"] = "fvcore missing -> params only (uv pip install fvcore)"
            else:
                with torch.no_grad():
                    entry["n_flops"] = int(FlopCountAnalysis(model, (dummy,)).total())
        except Exception as exc:  # noqa: BLE001 - best-effort tool: record and keep going
            entry["error"] = f"{type(exc).__name__}: {exc}"
        report[name] = entry
    return report


def _write_back(report: dict[str, dict[str, Any]]) -> int:
    """Patch ``model.n_params`` / ``model.n_flops`` into every matching board row (in place)."""
    changed = 0
    for path in sorted(_RESULTS.rglob("*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        measured = report.get(row.get("model", {}).get("name", ""))
        if not measured or "error" in measured:
            continue
        model = row["model"]
        updated = False
        for key in ("n_params", "n_flops"):
            if key in measured and model.get(key) != measured[key]:
                model[key] = measured[key]
                updated = True
        if updated:
            path.write_text(json.dumps(row, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            changed += 1
    return changed


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: print the JSON report, optionally patch the result rows (``--write``)."""
    parser = argparse.ArgumentParser(description="Compute n_params + FLOPs for implemented models.")
    parser.add_argument(
        "--write",
        action="store_true",
        help="patch leaderboard/results/**/*.json in place with the measured sizes",
    )
    args = parser.parse_args(argv)
    report = measure()
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.write:
        print(f"# wrote {_write_back(report)} result file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
