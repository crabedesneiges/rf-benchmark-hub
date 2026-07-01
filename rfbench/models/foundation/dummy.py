"""WP-60 -- a concrete, dependency-free example foundation model.

:class:`DummyFoundationModel` is a *runnable* example FM that needs no ``torch``/``numpy``:
it turns each IQ window into a small fixed-width feature vector with a deterministic,
pure-stdlib hash embedding. It exists so the whole FM path -- wrap a backbone, ``embed()``,
adapt under each regime, ``evaluate()`` to a schema-valid ``result.json`` -- runs end-to-end
in unit tests (light install: only ``pytest`` + ``jsonschema``) and so contributors have a
working reference next to the copy-me :mod:`~rfbench.models.foundation._template`.

Real RF backbones (e.g. a pretrained TorchSig/XCiT encoder, an IQ foundation model) plug in
the same way behind the ``rfbench[torch]`` extra: subclass
:class:`~rfbench.models.foundation.base.FoundationModel`, load the checkpoint in an
``embed()`` that calls :func:`~rfbench.models.foundation.base.require_torch`, and register
the wrapper. Nothing about the harness, the regimes, or the schema changes.

The embedding is intentionally simple, deterministic and dependency-free -- NOT a good RF
representation. It is a *plumbing fixture*, not a baseline.
"""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Sequence

from rfbench.core.model import Model
from rfbench.core.registry import register_model
from rfbench.core.types import Batch, Tensor
from rfbench.models.foundation.base import FoundationModel

#: Default embedding width for the example FM (kept tiny; the stdlib head is dimension-agnostic).
DEFAULT_EMBED_DIM = 8


def _to_floats(sample_iq: object) -> list[float]:
    """Flatten one sample's IQ payload into a flat ``list[float]`` (pure stdlib).

    Accepts the shapes the dependency-free path produces: a flat ``[i0, q0, i1, ...]`` list,
    or a list of ``[i, q]`` pairs. Nested pairs are flattened so the hash sees a stable byte
    stream regardless of the caller's layout.
    """
    if not isinstance(sample_iq, Sequence) or isinstance(sample_iq, (str, bytes)):
        raise TypeError(f"iq sample must be a sequence of numbers, got {type(sample_iq).__name__}")
    flat: list[float] = []
    for element in sample_iq:
        if isinstance(element, (list, tuple)):
            flat.extend(float(v) for v in element)
        else:
            flat.append(float(element))
    return flat


def _hash_embed(values: Sequence[float], dim: int) -> list[float]:
    """Map a flat float vector to a deterministic ``dim``-D embedding in [0, 1) (pure stdlib).

    Serialises the values to bytes, hashes them (BLAKE2b), and expands the digest into ``dim``
    floats. Deterministic across runs/platforms (fixed struct format + digest size), so the
    example FM's ``embed`` is reproducible without numpy. Distinct inputs almost always map to
    distinct vectors, which is all the plumbing needs.
    """
    payload = struct.pack(f"<{len(values)}d", *(float(v) for v in values))
    digest = hashlib.blake2b(payload, digest_size=dim * 2).digest()
    # Two bytes per dimension -> a 16-bit unsigned int scaled into [0, 1).
    return [int.from_bytes(digest[2 * i : 2 * i + 2], "little") / 65535.0 for i in range(dim)]


@register_model("dummy-fm")
class DummyFoundationModel(FoundationModel):
    """A pure-stdlib example FM: deterministic hash embedding, no ``torch``/``numpy``.

    Registered as ``"dummy-fm"`` in :data:`rfbench.core.registry.MODELS`. Subclasses
    :class:`~rfbench.models.foundation.base.FoundationModel`, so it is a
    ``family == "foundation"`` model exposing :meth:`embed` (the frozen representation the
    ``linear_probe`` / ``few_shot`` adapters fit on) and inheriting :meth:`forward` (falls
    back to :meth:`embed`, enough for the pass-through regimes). Constructed with no arguments
    by ``MODELS.get("dummy-fm")()`` on the registry path.
    """

    def __init__(self, *, name: str = "dummy-fm", embed_dim: int = DEFAULT_EMBED_DIM) -> None:
        """Build the example FM with an ``embed_dim``-wide hash embedding."""
        if embed_dim < 1:
            raise ValueError(f"embed_dim must be >= 1, got {embed_dim}")
        super().__init__(
            name,
            n_params=0,  # a hash has no learnable parameters
            backbone="stdlib-hash-embed",
            pretrained=False,
        )
        self.embed_dim = embed_dim

    def embed(self, x: Batch) -> Tensor:
        """Return one deterministic ``embed_dim``-D vector per sample in the collated batch ``x``.

        Reads the canonical ``"iq"`` field (a list of per-sample IQ payloads), flattens each
        sample and hashes it into a fixed-width vector. The output is a list of per-sample
        float vectors -- exactly what the probing adapters normalise and fit a head on.
        """
        iq_batch = x["iq"]
        return [_hash_embed(_to_floats(sample_iq), self.embed_dim) for sample_iq in iq_batch]


def build_example_fm() -> Model:
    """Instantiate the registered example FM (sugar for tests / the CLI's model registry path)."""
    from rfbench.core.registry import MODELS

    cls = MODELS.get("dummy-fm")
    return cls()  # type: ignore[no-any-return]


__all__ = ["DummyFoundationModel", "DEFAULT_EMBED_DIM", "build_example_fm"]
