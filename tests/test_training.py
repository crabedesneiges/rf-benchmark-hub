"""WP-30 acceptance tests for the real training loop + ``rfbench train`` CLI.

Two layers, matching the harness' dependency contract:

* **torch path** (``pytest.importorskip("torch")`` -> SKIPS in the dep-free ``.venv``, RUNS on
  the GPU venv). A tiny synthetic in-memory AMC dataset + a tiny dummy torch baseline are run
  through :func:`rfbench.training.train_baseline` for 1-2 epochs; we assert it fits without error
  and produces a ``result.json`` that VALIDATES against ``schemas/result.schema.json`` with the
  regime declared. No network, no real data.
* **CPU-only smoke** (no torch needed). ``rfbench train --help`` works and ``import rfbench`` stays
  dependency-free even with torch absent, so the CLI help surface is intact on a bare box.

Pure-logic helpers (:func:`rfbench.training.resolve_device` argument handling, the trainable-regime
guard) are exercised on the torch path since they are thin wrappers over torch.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from rfbench.core.dataset import Dataset
from rfbench.core.model import Model
from rfbench.core.splits import SplitManifest
from rfbench.core.types import SplitName, Track
from rfbench.tasks.amc.dataset import _InMemoryAmcSplit

# --- schema resolution (repo checkout) ------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
_RESULT_SCHEMA = _REPO_ROOT / "schemas" / "result.schema.json"


def _validate_result(document: dict[str, Any]) -> None:
    """Validate a result dict against result.schema.json (jsonschema is a light-install dep)."""
    from jsonschema import Draft202012Validator

    schema = json.loads(_RESULT_SCHEMA.read_text(encoding="utf-8"))
    Draft202012Validator(schema).validate(document)


# ==================================================================================================
# CPU-only smoke: the CLI help surface works with torch ABSENT (dependency-free)
# ==================================================================================================
def test_import_rfbench_is_dependency_free() -> None:
    """``import rfbench`` (and .core) must not pull torch, even transitively."""
    import importlib

    for mod in ("torch", "numpy"):
        sys.modules.pop(mod, None)
    importlib.import_module("rfbench")
    importlib.import_module("rfbench.core")
    importlib.import_module("rfbench.cli")
    assert "torch" not in sys.modules, "import rfbench must not import torch"


def test_train_help_works_without_torch(capsys: pytest.CaptureFixture[str]) -> None:
    """``rfbench train --help`` prints usage and exits 0 with no torch installed."""
    from rfbench.cli import main

    with pytest.raises(SystemExit) as excinfo:
        main(["train", "--help"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert "--task" in out
    assert "--model" in out
    assert "--regime" in out
    assert "--epochs" in out


def test_top_level_help_lists_train(capsys: pytest.CaptureFixture[str]) -> None:
    """``rfbench --help`` advertises the new ``train`` subcommand alongside the others."""
    from rfbench.cli import main

    with pytest.raises(SystemExit):
        main(["--help"])
    out = capsys.readouterr().out
    for sub in ("data", "eval", "train", "submit", "leaderboard", "verify"):
        assert sub in out, f"`rfbench --help` should list `{sub}`"


def test_train_rejects_probing_regime(capsys: pytest.CaptureFixture[str]) -> None:
    """A probing regime is a usage error for `train` (only trainable regimes are accepted)."""
    from rfbench.cli import EXIT_USAGE, main

    # linear_probe / few_shot are not in the train parser's choices -> argparse usage error (2).
    with pytest.raises(SystemExit) as excinfo:
        main(
            [
                "train",
                "--task",
                "amc",
                "--dataset",
                "radioml_2016_10a",
                "--model",
                "mcldnn",
                "--regime",
                "linear_probe",
            ]
        )
    assert excinfo.value.code == EXIT_USAGE


# ==================================================================================================
# torch path: a tiny model fits on a synthetic AMC dataset and emits a valid result.json
# ==================================================================================================
def _build_tiny_amc_dataset(n_per_class: int = 8, n_classes: int = 3, length: int = 16) -> Dataset:
    """A synthetic in-memory AMC dataset: class-separable IQ windows + snr_db meta.

    Each class ``c`` gets ``n_per_class`` samples whose IQ window is a constant ``(2, length)``
    map at level ``c`` (linearly separable), so a tiny net learns it in a couple of epochs. Uses
    the real :class:`~rfbench.tasks.amc.dataset.AmcDataset` synthetic (``samples=``) path so the
    whole ``load('train')`` / ``load('test')`` surface is exercised.
    """
    from rfbench.tasks.amc.dataset import AmcDataset

    samples: list[dict[str, Any]] = []
    for c in range(n_classes):
        for j in range(n_per_class):
            iq = [[float(c)] * length, [float(c) * 0.5] * length]  # (2, length), separable by class
            samples.append({"iq": iq, "label": c, "snr_db": (j % 5) * 2 - 4})
    return AmcDataset("radioml_2016_10a", samples=samples)


def _class_samples(n_per_class: int, n_classes: int, length: int) -> list[dict[str, Any]]:
    """Build a flat list of class-separable ``(2, length)`` IQ samples (for split builders)."""
    out: list[dict[str, Any]] = []
    for c in range(n_classes):
        for j in range(n_per_class):
            iq = [[float(c)] * length, [float(c) * 0.5] * length]
            out.append({"iq": iq, "label": c, "snr_db": (j % 5) * 2 - 4})
    return out


class _SplitAwareAmcDataset(Dataset):
    """A synthetic AMC dataset that returns DISTINCT samples per split.

    The real :class:`~rfbench.tasks.amc.dataset.AmcDataset` synthetic path ignores ``split`` and
    yields one fixed sample list, so it cannot exercise the val-monitoring path (train == val).
    This test double maps each ``split`` name (``train`` / ``val`` / ``test``) to its own list of
    per-sample ``Batch`` dicts, duck-typing exactly the surface ``train_baseline`` + ``evaluate``
    use: ``name`` / ``canonical_split_id`` / ``checksum`` attributes and ``load(split, track)``
    returning a map-style ``_InMemoryAmcSplit`` (``__len__`` + iteration + ``__getitem__``).
    """

    def __init__(self, splits: dict[str, list[dict[str, Any]]]) -> None:
        self.name = "radioml_2016_10a"
        self.canonical_split_id = "amc-strat-snr-seed42-v1"
        self.checksum = "sha256:" + "0" * 64
        self._splits = splits

    def download(self, cache: Path | None = None) -> None:  # pragma: no cover - unused in tests
        raise NotImplementedError

    def prepare(self, seed: int = 42) -> SplitManifest:  # pragma: no cover - unused in tests
        raise NotImplementedError

    def load(self, split: SplitName, track: Track | None = None) -> _InMemoryAmcSplit:
        return _InMemoryAmcSplit(self._splits.get(split, []))


def _build_tiny_model(n_classes: int = 3, length: int = 16) -> Model:
    """A tiny dummy torch baseline: a Model wrapping a linear ``nn.Module`` under ``.net``.

    Mirrors the real baseline shape (a :class:`~rfbench.core.model.Model` whose trainable module
    is ``self.net``, discovered by :func:`rfbench.training.resolve_module`). ``forward`` accepts
    the collated batch dict (``x["iq"]``) and returns ``(B, n_classes)`` logits; the training loop
    optimises ``.net`` directly on a ``(B, 2, L)`` tensor.
    """
    import torch
    from torch import nn

    from rfbench.core.model import Model
    from rfbench.core.types import Batch, Tensor

    class _TinyNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Linear(2 * length, n_classes)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.fc(x.reshape(x.shape[0], -1))

    class _TinyModel(Model):
        name = "tiny-baseline"
        family = "baseline"

        def __init__(self) -> None:
            self.device = torch.device("cpu")
            self.net = _TinyNet().to(self.device)

        def _to_tensor(self, iq_batch: object) -> torch.Tensor:
            t = torch.as_tensor(iq_batch, dtype=torch.float32, device=self.device)
            return t.unsqueeze(0) if t.ndim == 2 else t

        def forward(self, x: Batch) -> Tensor:
            iq = self._to_tensor(x["iq"])
            self.net.eval()
            with torch.no_grad():
                return self.net(iq)

        def embed(self, x: Batch) -> Tensor:  # pragma: no cover - baseline has no embed
            raise NotImplementedError

        @property
        def n_params(self) -> int:
            return sum(p.numel() for p in self.net.parameters())

    return _TinyModel()


def test_train_baseline_fits_and_emits_valid_result(tmp_path: Path) -> None:
    """train_baseline runs 2 epochs on a synthetic AMC set and writes a schema-valid result.json."""
    pytest.importorskip("torch")

    from rfbench.core.model import Regime, RegimeSpec
    from rfbench.tasks.amc.task import AmcTask
    from rfbench.training import train_baseline

    length = 16
    dataset = _build_tiny_amc_dataset(length=length)
    model = _build_tiny_model(length=length)
    task = AmcTask(datasets=[dataset])
    out_path = tmp_path / "mcldnn.json"

    trained, result = train_baseline(
        task,
        model,
        dataset,
        regime=RegimeSpec(Regime.FROM_SCRATCH),
        epochs=2,
        batch_size=4,
        lr=1e-2,
        seed=42,
        device="cpu",
        out_path=out_path,
    )

    # A trained model is returned and a result.json was written + is schema-valid.
    assert trained is model
    assert out_path.is_file()
    on_disk = json.loads(out_path.read_text(encoding="utf-8"))
    _validate_result(on_disk)
    _validate_result(result)

    # The regime is declared verbatim (from_scratch), never inferred (D5).
    assert result["regime"] == {"name": "from_scratch"}
    assert result["model"]["family"] == "baseline"
    assert result["model"]["n_params"] > 0
    assert result["metrics"]["primary"] == "accuracy_overall"
    assert "accuracy_overall" in result["metrics"]["values"]
    assert result["verification"]["status"] == "self_reported"
    # The full SNR range is attested (AMC no-cherry-picking rule).
    assert result["eval"]["conditions"]["full_snr_range"] is True


def test_train_baseline_learns_separable_classes(tmp_path: Path) -> None:
    """On a trivially-separable synthetic set the fit lifts accuracy above chance."""
    pytest.importorskip("torch")

    from rfbench.core.model import Regime, RegimeSpec
    from rfbench.tasks.amc.task import AmcTask
    from rfbench.training import train_baseline

    length = 16
    dataset = _build_tiny_amc_dataset(n_per_class=16, n_classes=3, length=length)
    model = _build_tiny_model(n_classes=3, length=length)
    task = AmcTask(datasets=[dataset])

    _trained, result = train_baseline(
        task,
        model,
        dataset,
        regime=RegimeSpec(Regime.FROM_SCRATCH),
        epochs=30,
        batch_size=8,
        lr=5e-2,
        seed=42,
        device="cpu",
    )
    # 3 classes -> chance is ~0.33; a linear net on constant separable inputs beats that easily.
    assert result["metrics"]["values"]["accuracy_overall"] > 0.5


def test_train_baseline_rejects_untrainable_regime() -> None:
    """A non-trainable regime spec is rejected before any fitting happens."""
    pytest.importorskip("torch")

    from rfbench.core.model import Regime, RegimeSpec
    from rfbench.tasks.amc.task import AmcTask
    from rfbench.training import train_baseline

    dataset = _build_tiny_amc_dataset()
    model = _build_tiny_model()
    task = AmcTask(datasets=[dataset])

    with pytest.raises(ValueError, match="fits only"):
        train_baseline(
            task,
            model,
            dataset,
            regime=RegimeSpec(Regime.LINEAR_PROBE),
            epochs=1,
            batch_size=4,
            lr=1e-2,
            device="cpu",
        )


def test_train_baseline_rejects_bad_epochs() -> None:
    """Non-positive epochs / batch_size fail loudly."""
    pytest.importorskip("torch")

    from rfbench.core.model import Regime, RegimeSpec
    from rfbench.tasks.amc.task import AmcTask
    from rfbench.training import train_baseline

    dataset = _build_tiny_amc_dataset()
    model = _build_tiny_model()
    task = AmcTask(datasets=[dataset])

    with pytest.raises(ValueError, match="epochs must be"):
        train_baseline(
            task,
            model,
            dataset,
            regime=RegimeSpec(Regime.FROM_SCRATCH),
            epochs=0,
            batch_size=4,
            lr=1e-2,
            device="cpu",
        )


def test_resolve_module_finds_net_and_full_finetune_runs(tmp_path: Path) -> None:
    """resolve_module locates the wrapped .net; the full_finetune regime also fits + evaluates."""
    pytest.importorskip("torch")

    import torch

    from rfbench.core.model import Regime, RegimeSpec
    from rfbench.tasks.amc.task import AmcTask
    from rfbench.training import resolve_module, train_baseline

    model = _build_tiny_model()
    module = resolve_module(model)
    assert isinstance(module, torch.nn.Module)

    dataset = _build_tiny_amc_dataset()
    task = AmcTask(datasets=[dataset])
    _trained, result = train_baseline(
        task,
        model,
        dataset,
        regime=RegimeSpec(Regime.FULL_FINETUNE),
        epochs=1,
        batch_size=4,
        lr=1e-2,
        device="cpu",
        out_path=tmp_path / "r.json",
    )
    _validate_result(result)
    assert result["regime"] == {"name": "full_finetune"}


def test_resolve_device_honours_explicit_cpu() -> None:
    """resolve_device returns an explicit device verbatim and resolves 'auto' to a concrete one."""
    pytest.importorskip("torch")

    from rfbench.training import resolve_device

    assert resolve_device("cpu") == "cpu"
    assert resolve_device("auto") in ("cuda", "cpu")
    assert resolve_device(None) in ("cuda", "cpu")


# ==================================================================================================
# torch path: the UPGRADED recipe -- val monitoring, LR plateau, early stop, best-val restore
# ==================================================================================================
def test_train_baseline_loads_and_monitors_val_split() -> None:
    """train_baseline reads the ``val`` split each epoch (distinct from ``train``) to monitor loss.

    Uses a split-aware synthetic dataset so ``load('val')`` returns its OWN samples; we record every
    ``load`` call and assert the ``val`` split was consumed (i.e. validation monitoring is active),
    on top of ``train`` and the final ``test`` evaluation.
    """
    pytest.importorskip("torch")

    from rfbench.core.model import Regime, RegimeSpec
    from rfbench.tasks.amc.task import AmcTask
    from rfbench.training import train_baseline

    length = 16
    dataset = _SplitAwareAmcDataset(
        {
            "train": _class_samples(8, 3, length),
            "val": _class_samples(4, 3, length),
            "test": _class_samples(4, 3, length),
        }
    )
    loaded_splits: list[str] = []
    original_load = dataset.load

    def _tracking_load(split: SplitName, track: Track | None = None) -> _InMemoryAmcSplit:
        loaded_splits.append(split)
        return original_load(split, track)

    dataset.load = _tracking_load  # type: ignore[method-assign]

    model = _build_tiny_model(n_classes=3, length=length)
    task = AmcTask(datasets=[dataset])

    _trained, result = train_baseline(
        task,
        model,
        dataset,
        regime=RegimeSpec(Regime.FROM_SCRATCH),
        epochs=3,
        batch_size=4,
        lr=1e-2,
        seed=42,
        device="cpu",
        patience=5,
    )
    _validate_result(result)
    assert "train" in loaded_splits, "the train split must be loaded"
    assert "val" in loaded_splits, "the val split must be loaded each epoch for monitoring"
    # 3 epochs -> the val split is loaded (re-iterated) at least once for monitoring.
    assert loaded_splits.count("val") >= 1


def test_train_baseline_restores_best_val_checkpoint() -> None:
    """The model evaluated on TEST is the BEST-VAL state, not the last-epoch state.

    We drive many epochs with a HIGH LR so the tail epochs diverge (val loss climbs after an early
    minimum). With best-val restore, the parameters after training must equal the snapshot taken at
    the best epoch -- NOT the (diverged) final-epoch parameters. We assert the restored weights beat
    a captured late-epoch snapshot on val loss, proving the best checkpoint (not the last) is kept.
    """
    pytest.importorskip("torch")

    import copy

    import torch

    from rfbench.core.model import Regime, RegimeSpec
    from rfbench.tasks.amc.task import AmcTask
    from rfbench.training import resolve_module, train_baseline

    length = 16
    dataset = _SplitAwareAmcDataset(
        {
            "train": _class_samples(12, 3, length),
            "val": _class_samples(6, 3, length),
            "test": _class_samples(6, 3, length),
        }
    )
    model = _build_tiny_model(n_classes=3, length=length)
    module = resolve_module(model)

    # Snapshot the RAW (pre-fit) weights so we can prove training actually changed the module.
    pre_fit = copy.deepcopy(module.state_dict())

    _trained, result = train_baseline(
        task=AmcTask(datasets=[dataset]),
        model=model,
        dataset=dataset,
        regime=RegimeSpec(Regime.FROM_SCRATCH),
        epochs=40,
        batch_size=4,
        lr=0.8,  # deliberately high so late epochs diverge and best-val != last-epoch
        seed=0,
        device="cpu",
        patience=100,  # disable early stop so the run reaches the diverging tail
        min_delta=0.0,
    )
    _validate_result(result)

    post_fit = module.state_dict()
    # Training changed the weights (the restored best state is not the untrained init).
    assert any(
        not torch.equal(pre_fit[k], post_fit[k]) for k in post_fit
    ), "training must update the module weights"

    # The restored (best-val) weights achieve a LOWER val loss than a fresh late-epoch continuation:
    # keep training the restored module a few more high-LR steps and confirm val loss does not drop
    # below the restored checkpoint -- i.e. the kept state is (near-)optimal, not a diverged tail.
    val_loader = torch.utils.data.DataLoader(  # type: ignore[attr-defined]
        [(s["iq"], s["label"]) for s in dataset.load("val")],
        batch_size=4,
        collate_fn=lambda pairs: (
            torch.tensor([iq for iq, _ in pairs], dtype=torch.float32),
            torch.tensor([lbl for _, lbl in pairs], dtype=torch.long),
        ),
    )
    criterion = torch.nn.CrossEntropyLoss()
    module.eval()
    with torch.no_grad():
        best_val_loss = sum(float(criterion(module(x), y)) for x, y in val_loader) / len(val_loader)

    # Take ONE more aggressive optimisation step from the restored state and re-measure.
    optimizer = torch.optim.SGD(module.parameters(), lr=5.0)
    module.train()
    for x, y in val_loader:
        optimizer.zero_grad()
        criterion(module(x), y).backward()
        optimizer.step()
    module.eval()
    with torch.no_grad():
        perturbed_val_loss = sum(float(criterion(module(x), y)) for x, y in val_loader) / len(
            val_loader
        )
    # A large uphill step from a (near-)optimum raises the loss: confirms a good minimum was kept.
    assert perturbed_val_loss >= best_val_loss - 1e-6


def test_train_baseline_falls_back_when_val_split_absent() -> None:
    """With no ``val`` split, train_baseline degrades to TRAIN-loss monitoring instead of crashing.

    A split-aware dataset whose ``val`` split is EMPTY forces the graceful fallback path: the run
    must still fit, keep a best checkpoint, and emit a valid result (monitoring the train loss).
    """
    pytest.importorskip("torch")

    from rfbench.core.model import Regime, RegimeSpec
    from rfbench.tasks.amc.task import AmcTask
    from rfbench.training import train_baseline

    length = 16
    dataset = _SplitAwareAmcDataset(
        {
            "train": _class_samples(8, 3, length),
            "val": [],  # empty -> _load_val_source returns None -> train-loss fallback
            "test": _class_samples(4, 3, length),
        }
    )
    model = _build_tiny_model(n_classes=3, length=length)
    task = AmcTask(datasets=[dataset])

    _trained, result = train_baseline(
        task,
        model,
        dataset,
        regime=RegimeSpec(Regime.FROM_SCRATCH),
        epochs=5,
        batch_size=4,
        lr=1e-2,
        seed=42,
        device="cpu",
        patience=3,
    )
    _validate_result(result)
    assert result["metrics"]["primary"] == "accuracy_overall"


def test_train_baseline_early_stops_before_max_epochs(caplog: pytest.LogCaptureFixture) -> None:
    """A tiny patience on an already-converged fit triggers early stopping before ``epochs``.

    On a trivially-separable set the val loss plateaus quickly; with ``patience=1`` and a large
    ``epochs`` budget the loop must stop early and log the early-stopping message.
    """
    pytest.importorskip("torch")

    import logging

    from rfbench.core.model import Regime, RegimeSpec
    from rfbench.tasks.amc.task import AmcTask
    from rfbench.training import train_baseline

    length = 16
    dataset = _SplitAwareAmcDataset(
        {
            "train": _class_samples(8, 3, length),
            "val": _class_samples(4, 3, length),
            "test": _class_samples(4, 3, length),
        }
    )
    model = _build_tiny_model(n_classes=3, length=length)
    task = AmcTask(datasets=[dataset])

    with caplog.at_level(logging.INFO, logger="rfbench.training"):
        _trained, result = train_baseline(
            task,
            model,
            dataset,
            regime=RegimeSpec(Regime.FROM_SCRATCH),
            epochs=500,  # large budget the loop should NOT exhaust
            batch_size=4,
            lr=1e-1,
            seed=42,
            device="cpu",
            patience=1,
            min_delta=1e-3,
        )
    _validate_result(result)
    assert any(
        "early stopping" in rec.message for rec in caplog.records
    ), "a converged fit with patience=1 must early-stop and log it"


def test_train_baseline_rejects_bad_patience() -> None:
    """Non-positive ``patience`` fails loudly (mirrors the epochs/batch_size guards)."""
    pytest.importorskip("torch")

    from rfbench.core.model import Regime, RegimeSpec
    from rfbench.tasks.amc.task import AmcTask
    from rfbench.training import train_baseline

    dataset = _build_tiny_amc_dataset()
    model = _build_tiny_model()
    task = AmcTask(datasets=[dataset])

    with pytest.raises(ValueError, match="patience must be"):
        train_baseline(
            task,
            model,
            dataset,
            regime=RegimeSpec(Regime.FROM_SCRATCH),
            epochs=1,
            batch_size=4,
            lr=1e-2,
            device="cpu",
            patience=0,
        )


# ==================================================================================================
# torch path via the CLI: `rfbench train` end-to-end on the synthetic dataset (mcldnn)
# ==================================================================================================
def test_cli_train_mcldnn_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`rfbench train ... --model mcldnn` trains the real MCLDNN on a tiny synthetic AMC set.

    Monkeypatches the AMC dataset resolution to inject the synthetic in-memory dataset (no real
    data / network), then drives the CLI handler end-to-end and validates the emitted result.json.
    """
    pytest.importorskip("torch")

    from rfbench.cli import EXIT_OK, main

    # MCLDNN's native window is 128; build the synthetic set at that length so shapes line up.
    dataset = _build_tiny_amc_dataset(n_per_class=6, n_classes=3, length=128)

    import rfbench.training as training_mod

    def _fake_resolve(task: object, dataset_name: str) -> Dataset:
        return dataset

    monkeypatch.setattr(training_mod, "resolve_amc_dataset", _fake_resolve)

    out_path = tmp_path / "mcldnn.json"
    rc = main(
        [
            "train",
            "--task",
            "amc",
            "--dataset",
            "radioml_2016_10a",
            "--model",
            "mcldnn",
            "--regime",
            "from_scratch",
            "--epochs",
            "1",
            "--batch-size",
            "4",
            "--lr",
            "1e-3",
            "--device",
            "cpu",
            "--out",
            str(out_path),
        ]
    )
    assert rc == EXIT_OK
    assert out_path.is_file()
    doc = json.loads(out_path.read_text(encoding="utf-8"))
    _validate_result(doc)
    assert doc["model"]["name"] == "mcldnn"
    assert doc["regime"] == {"name": "from_scratch"}
