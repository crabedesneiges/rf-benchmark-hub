"""WP-60/61 acceptance tests for the foundation-model wrappers.

The example :class:`~rfbench.models.foundation.dummy.DummyFoundationModel` is exercised
end-to-end WITHOUT ``torch``/``numpy``:

* it satisfies the :class:`~rfbench.core.model.Model` contract (``forward`` / ``embed`` /
  ``n_params`` + ``name`` / ``family``) and is registered in ``MODELS`` under ``"dummy-fm"``;
* ``embed()`` returns one fixed-width vector per sample (the expected shape), deterministically;
* the SAME wrapped FM runs through :func:`~rfbench.models.foundation.base.run_regime` for all
  four regimes (``from_scratch`` / ``full_finetune`` / ``linear_probe`` / ``few_shot``);
* an :func:`rfbench.core.evaluate.evaluate` end-to-end on a synthetic AMC/SEI-shaped task emits
  a dict that independently validates against ``schemas/result.schema.json`` with the regime
  declared verbatim.

Must pass with only ``pytest`` + ``jsonschema`` installed (no ``torch``/``numpy``/``sklearn``).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from rfbench.core.dataset import Dataset
from rfbench.core.evaluate import _resolve_schema_path, evaluate
from rfbench.core.metric import Metric
from rfbench.core.model import Model, Regime, RegimeSpec
from rfbench.core.registry import MODELS
from rfbench.core.splits import SplitManifest
from rfbench.core.task import Task
from rfbench.core.types import Batch, SplitName, Tensor, Track
from rfbench.models.foundation import (
    DEFAULT_EMBED_DIM,
    DummyFoundationModel,
    FoundationModel,
    as_vectors,
    build_example_fm,
    run_regime,
)
from rfbench.models.foundation.base import _AdaptedModel

# A byte-stable checksum satisfying the schema pattern ``^sha256:[0-9a-f]{64}$``.
_CHECKSUM = "sha256:" + "cd" * 32


# --------------------------------------------------------------------------------------------------
# Pure-Python task/dataset/metric fixtures (no torch/numpy), AMC-shaped
# --------------------------------------------------------------------------------------------------
class _InMemorySplit:
    """A tiny map-style dataset: a list of per-sample ``Batch`` dicts."""

    def __init__(self, samples: list[Batch]) -> None:
        self._samples = samples

    def __len__(self) -> int:
        return len(self._samples)

    def __iter__(self) -> Iterator[Batch]:
        return iter(self._samples)


class _DummyDataset(Dataset):
    """A dataset variant backed by an in-memory ``{iq,label,snr_db}`` split."""

    name = "radioml_2016_10a"
    canonical_split_id = "amc-strat-snr-seed42-v1"
    checksum = _CHECKSUM

    def __init__(self, samples: list[Batch]) -> None:
        self._samples = samples

    def download(self, cache: Path | None = None) -> None:  # pragma: no cover - unused
        raise NotImplementedError

    def prepare(self, seed: int = 42) -> SplitManifest:  # pragma: no cover - unused
        raise NotImplementedError

    def load(self, split: SplitName, track: Track | None = None) -> _InMemorySplit:
        return _InMemorySplit(self._samples)


class _Accuracy(Metric):
    """Streaming top-1 accuracy (pure Python), with the AMC full-SNR-range conditions hook."""

    name = "accuracy"
    primary_key = "accuracy_overall"

    def __init__(self) -> None:
        self._correct = 0
        self._total = 0

    def reset(self) -> None:
        self._correct = 0
        self._total = 0

    def update(self, pred: Tensor, target: Tensor, meta: dict[str, Any] | None = None) -> None:
        for predicted, expected in zip(pred, target, strict=True):
            self._total += 1
            if predicted == expected:
                self._correct += 1

    def compute(self) -> dict[str, float | list[dict[str, float]]]:
        accuracy = self._correct / self._total if self._total else 0.0
        return {"accuracy_overall": accuracy, "macro_f1": accuracy}

    def eval_conditions(self) -> dict[str, Any]:
        return {"snr_db_min": -20, "snr_db_max": 18, "full_snr_range": True}


class _AmcTask(Task):
    """An AMC-shaped task over a single in-memory dataset."""

    name = "amc"
    version = "v1"

    def __init__(self, samples: list[Batch]) -> None:
        self._dataset = _DummyDataset(samples)

    def datasets(self) -> list[Dataset]:
        return [self._dataset]

    def metrics(self) -> list[Metric]:
        return [_Accuracy()]

    def default_split(self) -> SplitName:
        return "test"

    def tracks(self) -> list[Track]:
        return ["closed_set"]

    def build_targets(self, batch: Batch) -> Tensor:
        return batch["label"]


# --------------------------------------------------------------------------------------------------
# Fixtures: a separable synthetic split so the probe head recovers labels
# --------------------------------------------------------------------------------------------------
def _train_split() -> list[Batch]:
    """A label-balanced train split: 4 samples per class, 3 classes, distinct IQ per sample.

    Each sample's IQ is distinct so the hash embedding gives distinct vectors; the eval set
    reuses the exact IQ of one train exemplar per class, so a nearest-centroid probe on the
    (deterministic) embeddings recovers the label -- letting us assert perfect accuracy.
    """
    samples: list[Batch] = []
    for label in (0, 1, 2):
        for j in range(4):
            samples.append(
                {"iq": [0.1 * label + 0.01 * j, 0.2 - 0.01 * j], "label": label, "snr_db": j}
            )
    return samples


def _eval_samples() -> list[Batch]:
    """Eval samples reusing the first train exemplar of each class (probe should recover labels)."""
    train = _train_split()
    per_class_first: dict[int, Batch] = {}
    for sample in train:
        per_class_first.setdefault(sample["label"], sample)
    return [dict(per_class_first[label]) for label in (0, 1, 2)]


def _load_schema() -> dict[str, Any]:
    schema_path = _resolve_schema_path("result.schema.json")
    assert schema_path is not None, "result.schema.json must be locatable"
    schema: dict[str, Any] = json.loads(schema_path.read_text(encoding="utf-8"))
    return schema


# --------------------------------------------------------------------------------------------------
# 1. Contract: the example FM IS a Model, registered, family foundation
# --------------------------------------------------------------------------------------------------
def test_example_fm_implements_model_contract() -> None:
    """DummyFoundationModel satisfies the Model ABC surface with the foundation family."""
    fm = DummyFoundationModel()
    assert isinstance(fm, Model)
    assert isinstance(fm, FoundationModel)
    assert fm.name == "dummy-fm"
    assert fm.family == "foundation"
    assert isinstance(fm.n_params, int)
    # forward + embed are callable and return per-sample outputs.
    batch = {"iq": [s["iq"] for s in _eval_samples()]}
    assert len(list(fm.forward(batch))) == 3
    assert len(list(fm.embed(batch))) == 3


def test_example_fm_is_registered() -> None:
    """The example FM is registered under 'dummy-fm' and resolves to the class."""
    assert "dummy-fm" in MODELS
    assert MODELS.get("dummy-fm") is DummyFoundationModel
    built = build_example_fm()
    assert isinstance(built, DummyFoundationModel)


# --------------------------------------------------------------------------------------------------
# 2. embed() shape + determinism
# --------------------------------------------------------------------------------------------------
def test_embed_shape_is_one_vector_per_sample() -> None:
    """embed() returns one DEFAULT_EMBED_DIM-wide vector per sample."""
    fm = DummyFoundationModel()
    batch = {"iq": [s["iq"] for s in _train_split()]}
    vectors = as_vectors(fm.embed(batch))
    assert len(vectors) == len(_train_split())
    assert all(len(v) == DEFAULT_EMBED_DIM for v in vectors)
    # coordinates are finite floats in [0, 1) by construction
    assert all(0.0 <= coord < 1.0 for v in vectors for coord in v)


def test_embed_is_deterministic() -> None:
    """The hash embedding is byte-stable across calls and instances (no numpy/random)."""
    batch = {"iq": [s["iq"] for s in _train_split()]}
    a = as_vectors(DummyFoundationModel().embed(batch))
    b = as_vectors(DummyFoundationModel().embed(batch))
    assert a == b


def test_embed_dim_is_configurable() -> None:
    """A custom embed_dim changes the vector width."""
    fm = DummyFoundationModel(embed_dim=4)
    batch = {"iq": [s["iq"] for s in _eval_samples()]}
    vectors = as_vectors(fm.embed(batch))
    assert all(len(v) == 4 for v in vectors)


# --------------------------------------------------------------------------------------------------
# 3. The same wrapped FM runs in ALL FOUR regimes via run_regime
# --------------------------------------------------------------------------------------------------
def _forward_batch() -> Batch:
    """A collated eval batch (dict of field -> list)."""
    samples = _eval_samples()
    return {
        "iq": [s["iq"] for s in samples],
        "label": [s["label"] for s in samples],
        "snr_db": [s["snr_db"] for s in samples],
    }


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        (RegimeSpec(Regime.FROM_SCRATCH), Regime.FROM_SCRATCH),
        (RegimeSpec(Regime.FULL_FINETUNE), Regime.FULL_FINETUNE),
        (RegimeSpec(Regime.LINEAR_PROBE), Regime.LINEAR_PROBE),
        (RegimeSpec(Regime.FEW_SHOT, k_shot=2), Regime.FEW_SHOT),
    ],
)
def test_run_regime_all_four(spec: RegimeSpec, expected: Regime) -> None:
    """run_regime adapts the FM under every regime and yields a scoreable Model."""
    fm = DummyFoundationModel()
    adapted = run_regime(fm, spec, _train_split())
    assert isinstance(adapted, _AdaptedModel)
    assert isinstance(adapted, Model)
    # regime declared verbatim, carried from the adapter
    assert adapted.regime.name is expected
    assert adapted.name == "dummy-fm"
    assert adapted.family == "foundation"
    assert adapted.n_params == fm.n_params
    preds = list(adapted.forward(_forward_batch()))
    assert len(preds) == 3


def test_probe_regimes_predict_trained_classes() -> None:
    """linear_probe / few_shot fit a head on the FM embeddings and predict known classes.

    The example FM's hash embedding is a plumbing fixture, not a separable representation, so
    we assert the honest, regime-level property: the probe fits on the FM's ``embed`` features
    and returns one prediction per eval sample drawn from the trained label set (0/1/2). Probe
    *quality* is a property of a real backbone, tested there -- not of this hash embedding.
    """
    fm = DummyFoundationModel()
    trained_labels = {0, 1, 2}
    for spec in (RegimeSpec(Regime.LINEAR_PROBE), RegimeSpec(Regime.FEW_SHOT, k_shot=4)):
        adapted = run_regime(fm, spec, _train_split())
        preds = list(adapted.forward(_forward_batch()))
        assert len(preds) == 3, f"one prediction per eval sample under {spec.name}"
        assert set(preds) <= trained_labels, f"probe under {spec.name} predicts trained classes"


def test_probe_recovers_labels_on_separable_embedding() -> None:
    """A probe on a separable injected embedding recovers labels (proves the probe path works).

    Uses an injected ``embed_fn`` that bakes the label into coordinate 0 (linearly separable),
    so the nearest-centroid head recovers it perfectly. This isolates the probe machinery from
    the hash embedding: it confirms ``run_regime`` -> fit -> predict is correct end-to-end.
    """
    fm = FoundationModel(
        "separable-fm",
        embed_fn=lambda x: [[float(lbl), 0.0] for lbl in x["label"]],
        n_params=1,
    )
    for spec in (RegimeSpec(Regime.LINEAR_PROBE), RegimeSpec(Regime.FEW_SHOT, k_shot=4)):
        adapted = run_regime(fm, spec, _train_split())
        preds = list(adapted.forward(_forward_batch()))
        assert preds == [0, 1, 2], f"separable probe under {spec.name} recovers labels"


def test_few_shot_carries_k_shot() -> None:
    """few_shot via run_regime keeps k_shot on the declared regime."""
    adapted = run_regime(
        DummyFoundationModel(), RegimeSpec(Regime.FEW_SHOT, k_shot=3), _train_split()
    )
    assert adapted.regime.k_shot == 3


# --------------------------------------------------------------------------------------------------
# 4. evaluate() end-to-end -> schema-valid result.json with the regime declared
# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "spec",
    [
        RegimeSpec(Regime.FROM_SCRATCH),
        RegimeSpec(Regime.FULL_FINETUNE),
        RegimeSpec(Regime.LINEAR_PROBE),
        RegimeSpec(Regime.FEW_SHOT, k_shot=2),
    ],
)
def test_evaluate_end_to_end_is_schema_valid(spec: RegimeSpec) -> None:
    """The wrapped FM, adapted under each regime, produces a schema-valid result.json."""
    from jsonschema import Draft202012Validator

    fm = DummyFoundationModel()
    task = _AmcTask(_train_split())
    adapted = run_regime(fm, spec, _train_split())

    result = evaluate(adapted, task, "test", adapted.regime, batch_size=2)

    Draft202012Validator(_load_schema()).validate(result)
    # regime declared verbatim, never inferred
    assert result["regime"]["name"] == spec.name.value
    if spec.k_shot is not None:
        assert result["regime"]["k_shot"] == spec.k_shot
    else:
        assert "k_shot" not in result["regime"]
    # attributed to the FM, in the foundation bucket
    assert result["model"]["name"] == "dummy-fm"
    assert result["model"]["family"] == "foundation"
    # AMC full-protocol condition is recorded
    assert result["eval"]["conditions"]["full_snr_range"] is True


def test_evaluate_writes_valid_file(tmp_path: Path) -> None:
    """evaluate(out_path=...) writes a file that re-validates against the schema."""
    from jsonschema import Draft202012Validator

    fm = DummyFoundationModel()
    task = _AmcTask(_train_split())
    adapted = run_regime(fm, RegimeSpec(Regime.LINEAR_PROBE), _train_split())

    out_path = tmp_path / "amc" / "dummy-fm.json"
    result = evaluate(adapted, task, "test", adapted.regime, batch_size=2, out_path=out_path)
    assert out_path.is_file()
    on_disk = json.loads(out_path.read_text(encoding="utf-8"))
    assert on_disk == result
    Draft202012Validator(_load_schema()).validate(on_disk)


# --------------------------------------------------------------------------------------------------
# 5. Generic wrapper: injected embed_fn / forward_fn, and the require_torch hint
# --------------------------------------------------------------------------------------------------
def test_injected_backbone_functions() -> None:
    """FoundationModel wraps injected embed_fn / forward_fn without a subclass."""
    fm = FoundationModel(
        "injected-fm",
        embed_fn=lambda x: [[float(v) for v in iq] for iq in x["iq"]],
        forward_fn=lambda x: list(x["label"]),
        n_params=7,
    )
    assert fm.family == "foundation"
    assert fm.n_params == 7
    assert fm.forward({"label": [1, 2]}) == [1, 2]
    assert fm.embed({"iq": [[0.0, 1.0]]}) == [[0.0, 1.0]]


def test_forward_defaults_to_embed() -> None:
    """With no forward_fn, forward falls back to embed (pass-through regimes still run)."""
    fm = FoundationModel("embed-only", embed_fn=lambda x: [[1.0] for _ in x["iq"]])
    assert fm.forward({"iq": [[0.0], [0.0]]}) == [[1.0], [1.0]]


def test_missing_embed_raises_clear_error() -> None:
    """A wrapper with neither embed override nor embed_fn fails loudly."""
    fm = FoundationModel("no-embed")
    with pytest.raises(NotImplementedError, match="provides no embed"):
        fm.embed({"iq": [[0.0]]})


def test_empty_name_rejected() -> None:
    """A FoundationModel needs a non-empty name."""
    with pytest.raises(ValueError, match="non-empty name"):
        FoundationModel("")


# --------------------------------------------------------------------------------------------------
# 6. Dependency-freedom
# --------------------------------------------------------------------------------------------------
def test_import_is_dependency_free() -> None:
    """Importing the foundation package must not pull torch/numpy/sklearn."""
    import importlib
    import sys

    for mod in ("torch", "numpy", "sklearn"):
        sys.modules.pop(mod, None)
    importlib.import_module("rfbench.models.foundation")
    for mod in ("torch", "numpy", "sklearn"):
        assert mod not in sys.modules, f"rfbench.models.foundation must not import {mod}"


def test_require_torch_gives_actionable_hint() -> None:
    """require_torch raises with the rfbench[torch] hint when torch is absent."""
    import sys

    from rfbench.models.foundation.base import require_torch

    if "torch" in sys.modules:  # pragma: no cover - torch not installed in the light env
        pytest.skip("torch is installed; the missing-extra path is not exercised here")
    with pytest.raises(ModuleNotFoundError, match=r"rfbench\[torch\]"):
        require_torch()
