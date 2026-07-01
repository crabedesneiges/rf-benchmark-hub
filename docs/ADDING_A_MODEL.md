# Adding a model (WP-61)

This guide shows how to add a model to RF-Benchmark-Hub **without touching the frozen core**
(`rfbench/core/`), the JSON schemas, the regime adapters, or `evaluate()`. A new model is
purely a new *wrapper* under `rfbench/models/`. If you find yourself editing anything in
`rfbench/core/` or `schemas/`, stop — that is a contract change (see
[CONTRIBUTING.md](../CONTRIBUTING.md#frozen-contracts)), not a model addition.

There are two flavours:

- **Baseline** (`family = "baseline"`) — a task-specific model trained from scratch. It only
  needs `forward()`; `embed()` may raise `NotImplementedError`. Lives in
  `rfbench/models/baselines/`.
- **Foundation model** (`family = "foundation"`) — a pretrained RF backbone you adapt to a
  task under one of the four regimes. It **must** expose `embed()` so the `linear_probe` and
  `few_shot` regimes work. Lives in `rfbench/models/foundation/`. This guide focuses on FMs.

The complete, runnable reference is
[`rfbench/models/foundation/dummy.py`](../rfbench/models/foundation/dummy.py) — a
dependency-free example FM. Copy
[`rfbench/models/foundation/_template.py`](../rfbench/models/foundation/_template.py) to start.

---

## The `Model` contract

Every model implements four things (see
[`rfbench/core/model.py`](../rfbench/core/model.py)):

| Member | What it is |
|--------|-----------|
| `name: str` | Leaderboard / registry name, e.g. `"iqfm-base"`. Written to `result.json.model.name`. |
| `family: "baseline" \| "foundation"` | Board bucket. FM wrappers use `"foundation"`. |
| `forward(x) -> Tensor` | Task-head output (logits / boxes / scores) for a collated batch. |
| `embed(x) -> Tensor` | Frozen representation, **one vector per sample**. Required for `linear_probe` / `few_shot`. |
| `n_params -> int` | Total parameter count including any adapter head. Written to `result.json.model.n_params`. |

`x` is the **collated batch** — a `dict` of field → list, e.g.
`{"iq": [...], "label": [...], "snr_db": [...]}` — exactly what `evaluate()` passes to
`forward`. `embed` reads the input field (usually `"iq"`) and returns one representation per
sample.

> **The regime is not a model attribute.** A model is evaluated under a regime, but the
> regime is carried by `RegimeSpec` and written verbatim into `result.json` — never inferred
> from the model. See [step 4](#4-pick-a-regime-and-evaluate).

---

## Step-by-step

### 1. Subclass `FoundationModel`

`FoundationModel`
([`rfbench/models/foundation/base.py`](../rfbench/models/foundation/base.py)) is the generic
wrapper. It fixes `family = "foundation"` and implements the contract; you override `embed`
(and `forward` if your FM ships a task head).

```python
from rfbench.core.registry import register_model
from rfbench.core.types import Batch, Tensor
from rfbench.models.foundation.base import FoundationModel, require_torch


@register_model("iqfm-base")
class IqfmBase(FoundationModel):
    def __init__(self, *, name: str = "iqfm-base", checkpoint: str | None = None) -> None:
        super().__init__(name, n_params=0, backbone=checkpoint, pretrained=True)
        self._checkpoint = checkpoint
        self._backbone = None  # loaded lazily
```

The constructor must work with **no required positional arguments** so the registry can
build it (`MODELS.get("iqfm-base")()`). Put any config behind defaults, an env var, or a
config file. Keep construction cheap — load weights lazily.

### 2. Implement `embed()` (load the backbone lazily)

`embed()` runs your encoder and returns one vector per sample. Import the heavy stack
**lazily** inside the method via `require_torch()`, so `import rfbench.models.foundation`
stays dependency-free and torch is only pulled when the model actually runs:

```python
    def _load(self) -> object:
        if self._backbone is None:
            torch = require_torch()  # raises with the `pip install rfbench[torch]` hint
            self._backbone = torch.load(self._checkpoint, map_location="cpu").eval()
            self._n_params = sum(p.numel() for p in self._backbone.parameters())
        return self._backbone

    def embed(self, x: Batch) -> Tensor:
        torch = require_torch()
        backbone = self._load()
        iq = torch.as_tensor(x["iq"])          # (batch, ...) per your input layout
        with torch.no_grad():
            return backbone(iq)                # (batch, embed_dim) — one vector per sample
```

The probing regimes normalise `embed`'s output to `list[float]` vectors, so any per-sample
2-D shape works.

### 3. Implement `forward()` only if your FM has a task head

- If your FM ships a fine-tuned head (needed for `from_scratch` / `full_finetune`), override
  `forward()` to return logits/boxes/scores.
- If you only ever **probe** it, delete the override and inherit
  `FoundationModel.forward`, which falls back to `embed()`.

### 4. Pick a regime and evaluate

The regime is an **adapter around the model**, resolved by
`make_adapter(RegimeSpec)` ([`rfbench/regimes/`](../rfbench/regimes/)):

- `from_scratch`, `full_finetune` — thin pass-throughs to `forward()`.
- `linear_probe` — freezes the backbone, fits a head on `embed()` features.
- `few_shot(k)` — same, on a `k`-per-class support set.

`rfbench.models.foundation` ships a one-call bridge, `run_regime`, that fits the adapter and
returns a `Model` you hand straight to `evaluate()`:

```python
from rfbench.core.evaluate import evaluate
from rfbench.core.model import Regime, RegimeSpec
from rfbench.core.registry import get_task
from rfbench.models.foundation import run_regime

fm = IqfmBase(checkpoint="/path/to/encoder.pt")
task = get_task("amc")
train = task.datasets()[0].load("train")     # fit split for the probe/finetune

for spec in (
    RegimeSpec(Regime.LINEAR_PROBE),
    RegimeSpec(Regime.FULL_FINETUNE),
    RegimeSpec(Regime.FEW_SHOT, k_shot=5),
):
    adapted = run_regime(fm, spec, train)     # fit under the regime
    result = evaluate(adapted, task, "test", adapted.regime)  # regime declared verbatim
```

Each call emits a schema-valid `result.json` with the regime declared, never inferred — so
the same wrapped FM yields one leaderboard row per regime (`linear_probe`, `full_finetune`,
`few_shot`), and the board keeps them in separate columns.

From the CLI (once your model is importable/registered):

```bash
rfbench eval amc --model iqfm-base --regime linear_probe
rfbench eval amc --model iqfm-base --regime few_shot --k-shot 5
```

### 5. Register + make it importable

`@register_model("<name>")` adds your class to `rfbench.core.registry.MODELS`. Registration
fires as a side effect of importing the module, so re-export your class from
`rfbench/models/foundation/__init__.py` (as `DummyFoundationModel` is) — that ensures the
`@register_model` decorator runs when the package is imported.

---

## Checklist

- [ ] Subclass `FoundationModel` (baseline: subclass `Model` with `family = "baseline"`).
- [ ] Implement `embed()` — one vector per sample; **required** for `linear_probe` / `few_shot`.
- [ ] Implement `forward()` if the FM has a task head; else inherit the `embed()` fallback.
- [ ] Set `n_params` (from the loaded backbone).
- [ ] Import torch/heavy deps **lazily** via `require_torch()` — package import stays dependency-free.
- [ ] `@register_model("<name>")` and re-export from the package `__init__` so it registers.
- [ ] Did **not** touch `rfbench/core/`, `schemas/`, `rfbench/regimes/`, or `evaluate()`.
- [ ] `ruff check .`, `mypy`, and `pytest -q` stay green; add a test for your wrapper.

## What you must not touch

The core contracts (`rfbench/core/`) and the JSON schemas (`schemas/`) are **frozen**
(Sprint 0). A model addition never edits them, the regime adapters, or `evaluate()`. If your
model needs a genuinely new capability from the core, that is a separate, reviewed contract
change with a version bump — see [CONTRIBUTING.md](../CONTRIBUTING.md#frozen-contracts).
