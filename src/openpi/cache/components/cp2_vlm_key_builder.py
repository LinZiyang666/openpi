"""CP2 post-backbone single-key builder (ActionCache-style baseline).

The key is the backbone's prefix output (``Stage2Output.prefix_out``,
``[B, prefix_len, emb_dim]`` final-norm hidden states) flattened to one vector
and compressed by a fixed sparse ternary random projection
``R in {-1, 0, +1}^{d x D}`` (Li, Hastie & Church 2006): every row holds
``floor(p*D/2)`` entries of +1 and the same number of -1 at positions drawn
without replacement from a seeded generator. This is the key construction of
ActionCache (arXiv 2607.06370, Eq. 2) reproduced on the openpi staged API.

Public interface
----------------
- ``ProjectionSpec`` / ``get_projection_spec`` — process-wide immutable
  projection resource keyed by ``(seed, d, p, input_dim)``; the two int32
  index tables (~40 MB at d=500, p=0.01, D=1,982,464) are built once per
  process and shared by every connection's builder, with a lazily populated
  per-device copy cache.
- ``project`` — the single numeric implementation (float32 accumulation of the
  +1 and -1 gathers, float32 subtraction) shared by the online builder, the
  offline artifact builder and the artifact verifier.
- ``CP2VlmTernaryKeyBuilder`` — the ``QueryKeyBuilder`` for CP2: ``collect``
  takes ``stage2=Stage2Output`` and ``build`` returns ``{"vlm_out": [d]}``.

Key dependencies: ``openpi.cache.types`` (``CheckpointID``, ``VLM_OUT``),
``Stage2Output.prefix_out`` (``models_pytorch/pi0_pytorch.py``).
"""

from __future__ import annotations

import dataclasses
import hashlib
import threading

import numpy as np
import torch

from openpi.cache.types import VLM_OUT, CheckpointID

KEY_BUILDER_TYPE = "cp2_vlm_ternary"
ACCUMULATION_DTYPE = "float32"
# Pi0.5 prefix: 3 x 256 image tokens + 200 text tokens, 2048-dim -> 1,982,464.
DEFAULT_INPUT_DIM = 968 * 2048
DEFAULT_D = 500
DEFAULT_P = 0.01


# ------------------------------------------------------------------
# Projection resource (process-wide, immutable)
# ------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ProjectionSpec:
    """Frozen sparse ternary projection: seed, shape and the two index tables."""

    seed: int
    d: int
    p: float
    input_dim: int
    idx_pos: torch.Tensor  # int32 [d, nnz], CPU, sorted per row
    idx_neg: torch.Tensor  # int32 [d, nnz], CPU, sorted per row
    digest: str

    @property
    def nnz_per_sign(self) -> int:
        return int(self.idx_pos.shape[1])

    @property
    def key(self) -> tuple[int, int, float, int]:
        return (self.seed, self.d, self.p, self.input_dim)

    def meta(self) -> dict:
        """Artifact / binding metadata (what the storage binding check compares)."""
        return {
            "seed": self.seed,
            "d": self.d,
            "p": self.p,
            "D": self.input_dim,
            "nnz_per_sign": self.nnz_per_sign,
            "accumulation_dtype": ACCUMULATION_DTYPE,
            "digest": self.digest,
        }


def _make_spec(seed: int, d: int, p: float, input_dim: int) -> ProjectionSpec:
    nnz = int(np.floor(p * input_dim / 2.0))
    if d < 1 or input_dim < 2 or nnz < 1 or 2 * nnz > input_dim:
        raise ValueError(
            f"invalid projection shape: d={d}, p={p}, D={input_dim} -> nnz_per_sign={nnz}"
        )
    rng = np.random.default_rng(int(seed))
    idx_pos = np.empty((d, nnz), dtype=np.int32)
    idx_neg = np.empty((d, nnz), dtype=np.int32)
    for row in range(d):
        chosen = rng.choice(input_dim, size=2 * nnz, replace=False)
        idx_pos[row] = np.sort(chosen[:nnz])
        idx_neg[row] = np.sort(chosen[nnz:])
    h = hashlib.sha256()
    h.update(f"cp2_vlm_ternary:v1:seed={int(seed)}:d={d}:p={p!r}:D={input_dim}:nnz={nnz}".encode())
    h.update(idx_pos.tobytes())
    h.update(idx_neg.tobytes())
    return ProjectionSpec(
        seed=int(seed), d=int(d), p=float(p), input_dim=int(input_dim),
        idx_pos=torch.from_numpy(idx_pos), idx_neg=torch.from_numpy(idx_neg),
        digest=h.hexdigest(),
    )


_SPEC_LOCK = threading.Lock()
_SPECS: dict[tuple[int, int, float, int], ProjectionSpec] = {}
_DEVICE_INDICES: dict[tuple[tuple[int, int, float, int], str], tuple[torch.Tensor, torch.Tensor]] = {}


def get_projection_spec(seed: int, d: int = DEFAULT_D, p: float = DEFAULT_P,
                        input_dim: int = DEFAULT_INPUT_DIM) -> ProjectionSpec:
    """Return the process-wide spec for ``(seed, d, p, input_dim)``, building it once."""
    key = (int(seed), int(d), float(p), int(input_dim))
    with _SPEC_LOCK:
        spec = _SPECS.get(key)
        if spec is None:
            spec = _make_spec(*key)
            _SPECS[key] = spec
        return spec


def _device_indices(spec: ProjectionSpec, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-device copy of the index tables, populated lazily and shared."""
    dkey = (spec.key, str(device))
    with _SPEC_LOCK:
        cached = _DEVICE_INDICES.get(dkey)
        if cached is None:
            cached = (spec.idx_pos.to(device), spec.idx_neg.to(device))
            _DEVICE_INDICES[dkey] = cached
        return cached


def project(h: torch.Tensor, spec: ProjectionSpec) -> torch.Tensor:
    """Apply the sparse ternary projection: ``[D]`` or ``[B, D]`` -> float32 ``[d]`` / ``[B, d]``.

    Numeric contract (frozen): the +1 gather and the -1 gather are each summed
    with ``dtype=torch.float32`` and subtracted in float32, whatever the input
    dtype (bf16 / fp16 / fp32). This is the one implementation the online
    builder, the offline artifact builder and the verifier all call.
    """
    squeeze = h.dim() == 1
    h2 = h.reshape(1, -1) if squeeze else h
    if h2.dim() != 2 or h2.shape[-1] != spec.input_dim:
        raise ValueError(
            f"project: expected input dim {spec.input_dim}, got shape {tuple(h.shape)}"
        )
    idx_pos, idx_neg = _device_indices(spec, h2.device)
    pos = h2[:, idx_pos].sum(-1, dtype=torch.float32)  # [B, d]
    neg = h2[:, idx_neg].sum(-1, dtype=torch.float32)  # [B, d]
    out = pos - neg
    return out[0] if squeeze else out


def dense_projection_matrix(spec: ProjectionSpec) -> torch.Tensor:
    """Dense ``[d, D]`` float32 form of the projection (tests / small oracles only)."""
    R = torch.zeros(spec.d, spec.input_dim, dtype=torch.float32)
    rows = torch.arange(spec.d).unsqueeze(1)
    R[rows, spec.idx_pos.long()] = 1.0
    R[rows, spec.idx_neg.long()] = -1.0
    return R


# ------------------------------------------------------------------
# KeyBuilder
# ------------------------------------------------------------------


class CP2VlmTernaryKeyBuilder:
    """QueryKeyBuilder for CP2: projected backbone prefix output as the single key.

    ``collect(CP2, stage2=...)`` keeps a reference to ``stage2.prefix_out`` on
    its own device; ``build(CP2)`` flattens it and applies ``project`` (the only
    D2H transfer is the final ``.cpu()`` of the [d] key). Any other checkpoint
    is a programming error: config validation makes this builder CP2-only.
    """

    def __init__(self, seed: int, d: int = DEFAULT_D, p: float = DEFAULT_P,
                 input_dim: int = DEFAULT_INPUT_DIM) -> None:
        self._spec = get_projection_spec(seed, d, p, input_dim)
        self._cache: dict[str, torch.Tensor] = {}

    @property
    def spec(self) -> ProjectionSpec:
        return self._spec

    def projection_meta(self) -> dict:
        return self._spec.meta()

    def collect(self, checkpoint_id: CheckpointID, **stage_outputs) -> None:
        self._cache.clear()
        if checkpoint_id is not CheckpointID.CP2:
            raise ValueError(
                f"cp2_vlm_ternary serves CP2 only, collect() called for {checkpoint_id}"
            )
        stage2 = stage_outputs.get("stage2")
        if stage2 is None:
            raise ValueError("cp2_vlm_ternary.collect requires stage2=Stage2Output")
        prefix_out = getattr(stage2, "prefix_out", None)
        if prefix_out is None:
            # Fail loud: a silent MISS here would look like a cold cache.
            raise RuntimeError(
                "Stage2Output.prefix_out is None: the stage-2 forward did not capture "
                "the backbone output (direct path must use run_stage2_capture; the "
                "coordinator path must submit with requires_stage2_capture=True)."
            )
        self._cache["prefix_out"] = prefix_out

    def build(self, checkpoint_id: CheckpointID) -> dict[str, torch.Tensor]:
        if checkpoint_id is not CheckpointID.CP2:
            raise ValueError(
                f"cp2_vlm_ternary serves CP2 only, build() called for {checkpoint_id}"
            )
        prefix_out = self._cache["prefix_out"]
        if prefix_out.dim() == 3:
            flat = prefix_out.reshape(prefix_out.shape[0], -1)
        elif prefix_out.dim() == 2:
            flat = prefix_out.reshape(1, -1)
        else:
            raise ValueError(f"prefix_out must be [B, L, E] or [L, E], got {tuple(prefix_out.shape)}")
        if flat.shape[0] != 1:
            raise ValueError(f"build() expects a single request (B=1), got B={flat.shape[0]}")
        key = project(flat, self._spec)[0]
        return {VLM_OUT: key.detach().cpu().float().contiguous()}

    @property
    def cached_data(self) -> dict[str, torch.Tensor]:
        return self._cache

    def clear(self) -> None:
        self._cache.clear()


__all__ = [
    "ACCUMULATION_DTYPE",
    "CP2VlmTernaryKeyBuilder",
    "DEFAULT_D",
    "DEFAULT_INPUT_DIM",
    "DEFAULT_P",
    "KEY_BUILDER_TYPE",
    "ProjectionSpec",
    "dense_projection_matrix",
    "get_projection_spec",
    "project",
]
