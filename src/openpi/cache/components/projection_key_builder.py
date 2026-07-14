"""Outcome-compatible projection key builder (TRACER M1 skeleton).

This module implements the M1 mechanism from the *Action-Compatible
Failure-Aware Retrieval* proposal: a per-modality linear projection head
``z = x W^T (+ b)`` applied on top of an existing pooled query key, so that
cache similarity is measured in a learned projected space (proposal Eq 8-9)
rather than in the raw pooled-feature space.

Design contract
---------------
- ``ProjectionKeyBuilder`` WRAPS an inner (stateless pool) ``QueryKeyBuilder``
  and projects only the cosine fields (vision_0/1/2, prompt_emb). ``robot_state``
  is an L2-distance field and is never projected (projecting it would destroy
  the ``exp(-d/tau)`` distance semantics the backend relies on).
- With NO projection params loaded the builder is the identity: ``build()``
  returns the inner builder's exact output dict, so it is value-for-value
  equal to the wrapped pool builder (the non-regression anchor).
- Both the online factory (``config._build_key_builder``) and the offline
  artifact factory (``build_in_memory_cache_artifact._create_builder``)
  construct this wrapper, so library keys and query keys pass through the SAME
  head and the backend stays a plain cosine store.

Parameter / fit separation (roadmap D3)
---------------------------------------
- ``ProjectionParams`` mirrors the ``ScoreNormalizer`` load/fit split: runtime
  only loads frozen weights; fitting is offline. The serialized payload is a
  tensor state dict (``torch.save``/``torch.load``), not the scalar JSON that
  ``ScoreNormalizer`` uses, because projection heads are weight matrices.
- ``ProjectionKeyBuilder.fit`` implements the InfoNCE fitting MECHANISM over
  already-prepared per-field training batches. Phase 2 ships this mechanism and
  exercises it on synthetic data only; the real compatibility-label
  construction (``c^A``/``c^X`` from ``CachePayload``) and the offline driver
  that persists weights land in Phase 6.

Coupling map
------------
  DEPENDS ON:  torch, openpi.cache.types field constants; wraps any
               ``QueryKeyBuilder`` (Protocol in key_builder.py).
  CONSUMED BY: config._build_key_builder (online),
               build_in_memory_cache_artifact._create_builder (offline).
  FEEDS INTO:  CacheStorage.search() via QuerySpec (projected out_dim must
               match backend vector_dims).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from openpi.cache.types import (
    PROMPT_EMB,
    ROBOT_STATE,
    VISION_0,
    VISION_1,
    VISION_2,
    CheckpointID,
)

# Cosine fields eligible for projection. robot_state is intentionally excluded:
# it is an L2-distance field and must keep its raw vector.
_PROJECTED_FIELDS = frozenset({VISION_0, VISION_1, VISION_2, PROMPT_EMB})

# ------------------------------------------------------------------
# Canonical inner (pool) output dimensions
# ------------------------------------------------------------------
# SigLIP multi_modal_projector output width; every pool builder emits vision /
# prompt features in multiples of this. Mirrors the emb_dim used in
# key_builder.py and the offline `_VECTOR_DIMS` table in
# exp/common/build_in_memory_cache_artifact.py — kept here so src-side
# validation does not import from exp/.
_EMB_DIM = 2048
_STATE_DIM = 32

# Vision field output dim per inner pool builder type.
_VISION_POOL_DIM: dict[str, int] = {
    "cp1_mean_pool": _EMB_DIM,
    "cp1_max_pool": _EMB_DIM,
    "cp1_spatial_pool_16": 16 * _EMB_DIM,  # 4x4 tokens * 2048 = 32768
    "cp1_spatial_pool_4": 4 * _EMB_DIM,    # 2x2 tokens * 2048 = 8192
    "cp1_spatial_pool_64": 4 * _EMB_DIM,   # legacy alias of cp1_spatial_pool_4
}

# The stateless pool builders that may serve as a projection inner.
STATELESS_POOL_INNER_TYPES = frozenset(_VISION_POOL_DIM)


def _pool_inner_dim(inner_type: str, field: str) -> int:
    """Return the inner pool builder's output dim for ``field``.

    Function (not a static per-field table) so any vision field (including
    vision_2) and any enabled-field subset resolves consistently. prompt_emb is
    mean-pooled to emb_dim by every pool builder; robot_state stays raw.
    """
    if field == ROBOT_STATE:
        return _STATE_DIM
    if field == PROMPT_EMB:
        return _EMB_DIM
    if field in (VISION_0, VISION_1, VISION_2):
        try:
            return _VISION_POOL_DIM[inner_type]
        except KeyError:
            raise ValueError(
                f"unknown projection inner_type {inner_type!r}; "
                f"valid: {sorted(_VISION_POOL_DIM)}"
            ) from None
    raise ValueError(f"field {field!r} has no known pool output dim")


# ------------------------------------------------------------------
# Projection parameters (load / save; offline fit produces these)
# ------------------------------------------------------------------
@dataclass(frozen=True)
class ProjectionHead:
    """A single field's linear projection ``z = x W^T (+ b)``.

    weight: [out_dim, in_dim] CPU float32.
    bias:   [out_dim] CPU float32, or None (no bias).
    """

    weight: torch.Tensor
    bias: Optional[torch.Tensor] = None


@dataclass(frozen=True)
class ProjectionParams:
    """Per-field projection heads. A field absent from ``heads`` is identity."""

    heads: dict[str, ProjectionHead]

    def head(self, field: str) -> Optional[ProjectionHead]:
        return self.heads.get(field)

    def to_state_dict(self) -> dict[str, dict[str, torch.Tensor]]:
        """Serialize to a torch-saveable nested tensor dict."""
        sd: dict[str, dict[str, torch.Tensor]] = {}
        for field, h in self.heads.items():
            entry: dict[str, torch.Tensor] = {"weight": h.weight}
            if h.bias is not None:
                entry["bias"] = h.bias
            sd[field] = entry
        return sd

    @classmethod
    def from_state_dict(cls, sd: dict[str, dict[str, torch.Tensor]]) -> "ProjectionParams":
        heads: dict[str, ProjectionHead] = {}
        for field, entry in sd.items():
            weight = entry["weight"].detach().cpu().float().contiguous()
            bias = entry.get("bias")
            if bias is not None:
                bias = bias.detach().cpu().float().contiguous()
            heads[field] = ProjectionHead(weight=weight, bias=bias)
        return cls(heads=heads)

    def save(self, path) -> None:
        torch.save(self.to_state_dict(), path)

    @classmethod
    def load(cls, path) -> "ProjectionParams":
        return cls.from_state_dict(torch.load(path, map_location="cpu"))


def validate_projection_params(
    params: Optional[ProjectionParams],
    inner_type: str,
    vector_dims: dict[str, int],
) -> None:
    """Fail loud at load time if projection heads do not fit the pipeline.

    Called by both the online (``config._build_key_builder``) and offline
    (``_create_builder``) factories before constructing the builder, so a bad
    weights file cannot pass config validation and then die at the matmul.

    Contract (over the backend's stored ``vector_dims`` fields):
      - a field WITH a head must be a projectable cosine field, present in
        vector_dims, and have ``weight.shape == (vector_dims[f], inner_dim)``
        and ``bias in (None, [vector_dims[f]])``;
      - a field WITHOUT a head passes through raw, so its stored dim must equal
        the inner output dim;
      - a head for a non-projectable / unknown / robot_state field is rejected.
    """
    if params is None:
        return

    for field, head in params.heads.items():
        if field not in _PROJECTED_FIELDS:
            raise ValueError(
                f"projection head for non-projectable field {field!r}; "
                f"projectable cosine fields: {sorted(_PROJECTED_FIELDS)} "
                "(robot_state is an L2 field and is never projected)"
            )
        if field not in vector_dims:
            raise ValueError(
                f"projection head for field {field!r} not in backend "
                f"vector_dims {sorted(vector_dims)}"
            )
        out_dim = vector_dims[field]
        in_dim = _pool_inner_dim(inner_type, field)
        if tuple(head.weight.shape) != (out_dim, in_dim):
            raise ValueError(
                f"projection weight for {field!r} must be {(out_dim, in_dim)} "
                f"(out_dim=vector_dims, in_dim=inner {inner_type}), "
                f"got {tuple(head.weight.shape)}"
            )
        if head.bias is not None and tuple(head.bias.shape) != (out_dim,):
            raise ValueError(
                f"projection bias for {field!r} must be {(out_dim,)}, "
                f"got {tuple(head.bias.shape)}"
            )

    # Fields the backend stores but that have no head pass through unchanged;
    # their stored dim must therefore equal the inner output dim.
    for field, dim in vector_dims.items():
        if field in params.heads:
            continue
        expected = _pool_inner_dim(inner_type, field)
        if dim != expected:
            raise ValueError(
                f"field {field!r} has no projection head (passthrough) but "
                f"vector_dims[{field!r}]={dim} != inner {inner_type} dim {expected}"
            )


def _project(vec: torch.Tensor, head: ProjectionHead) -> torch.Tensor:
    """Apply a projection head to a single [in_dim] feature vector.

    Backstop shape check gives a clear field-level error if the canonical dim
    table ever drifts from the real inner output, instead of a bare matmul fail.
    """
    if head.weight.shape[1] != vec.shape[0]:
        raise ValueError(
            f"projection in_dim {head.weight.shape[1]} != feature dim {vec.shape[0]}"
        )
    z = vec.to(torch.float32) @ head.weight.t()  # [out_dim]
    if head.bias is not None:
        z = z + head.bias
    return z.cpu().float().contiguous()


# ------------------------------------------------------------------
# Projection key builder (wraps a stateless pool inner)
# ------------------------------------------------------------------
class ProjectionKeyBuilder:
    """Wrap an inner pool ``QueryKeyBuilder`` and project its cosine fields.

    With ``params=None`` this is the identity: ``build()`` returns the inner
    builder's exact dict (value-for-value equal to the wrapped pool builder).
    With params, each projectable field that has a head is mapped through it;
    every other field (including robot_state) passes through unchanged.

    All ``QueryKeyBuilder`` protocol calls delegate to the inner builder, so the
    inner is never modified.
    """

    def __init__(self, inner, params: Optional[ProjectionParams] = None) -> None:
        self._inner = inner
        self._params = params

    def collect(self, checkpoint_id: CheckpointID, **stage_outputs) -> None:
        self._inner.collect(checkpoint_id, **stage_outputs)

    def build(self, checkpoint_id: CheckpointID) -> dict[str, torch.Tensor]:
        keys = self._inner.build(checkpoint_id)
        if self._params is None:
            # Identity: return the inner dict untouched (exact non-regression).
            return keys
        out: dict[str, torch.Tensor] = {}
        for field, vec in keys.items():
            head = self._params.head(field) if field in _PROJECTED_FIELDS else None
            out[field] = vec if head is None else _project(vec, head)
        return out

    @property
    def cached_data(self) -> dict[str, torch.Tensor]:
        return self._inner.cached_data

    def clear(self) -> None:
        self._inner.clear()

    def on_episode_start(self) -> None:
        # Forward to the inner builder if it is stateful. The Phase 2 inner set
        # is stateless, so this is a no-op today, but forwarding keeps the
        # wrapper correct if a stateful inner is ever allowed.
        fn = getattr(self._inner, "on_episode_start", None)
        if fn is not None:
            fn()

    # --------------------------------------------------------------
    # Offline fit mechanism (NOT invoked in Phase 2)
    # --------------------------------------------------------------
    @classmethod
    def fit(
        cls,
        batches: dict[str, "FieldTrainingBatch"],
        *,
        out_dim: int,
        epochs: int = 200,
        lr: float = 1e-2,
        temperature: float = 0.07,
        loss: str = "auto",
    ) -> ProjectionParams:
        """Fit per-field linear projection heads over prepared batches.

        OFFLINE ONLY. ``batches`` is already-prepared training signal (per-field
        pooled features + a compatibility grouping/masks), NOT raw CachePayloads.
        ``loss`` selects the objective per TRACER §B:
        - "masked" -> threshold-gated ``proj_infonce_loss`` (needs pos/neg masks);
        - "group"  -> legacy supervised ``infonce_loss`` (needs group_labels);
        - "auto"   -> dispatch on batch contents (masks -> masked, group_labels ->
          group; both or neither -> error). Default "auto" keeps existing
          group_labels-only callers on the unchanged "group" path.
        """
        heads: dict[str, ProjectionHead] = {}
        for field, batch in batches.items():
            if field not in _PROJECTED_FIELDS:
                raise ValueError(
                    f"cannot fit a projection head for non-projectable field "
                    f"{field!r}; projectable: {sorted(_PROJECTED_FIELDS)}"
                )
            mode = _resolve_loss_mode(batch, loss)
            heads[field] = _fit_one_head(
                batch, mode=mode, out_dim=out_dim, epochs=epochs, lr=lr, temperature=temperature
            )
        return ProjectionParams(heads=heads)


# ------------------------------------------------------------------
# Offline fit internals (synthetic-tested mechanism; no real data in Phase 2)
# ------------------------------------------------------------------
@dataclass
class FieldTrainingBatch:
    """Prepared per-field training signal for ``ProjectionKeyBuilder.fit``.

    features:     [N, in_dim] pooled feature vectors (required).
    group_labels: [N] int; rows sharing a label are compatible positives. Feeds the
                  legacy supervised-contrastive ``infonce_loss`` (the "group" loss).
    pos_mask:     [N, N] bool; ``pos_mask[a, b]`` marks b a positive of anchor a.
    neg_mask:     [N, N] bool; ``neg_mask[a, b]`` marks b a hard negative of anchor a.
                  ``pos_mask``/``neg_mask`` feed the threshold-gated ``proj_infonce_loss``
                  (the "masked" loss, TRACER Eq 13-15); gray-zone pairs are in neither.

    A batch carries EITHER ``group_labels`` (legacy path) OR ``pos_mask`` + ``neg_mask``
    (method path); ``fit(..., loss=...)`` selects which. New optional fields default to
    ``None`` so existing group_labels-only callers are unaffected.
    """

    features: torch.Tensor
    group_labels: torch.Tensor | None = None
    pos_mask: torch.Tensor | None = None
    neg_mask: torch.Tensor | None = None


def infonce_loss(z: torch.Tensor, group_labels: torch.Tensor, temperature: float) -> torch.Tensor:
    """Supervised InfoNCE over projected features (higher compat -> closer).

    z is L2-normalized internally; positives are same-label off-diagonal pairs.
    Returns the mean negative log-probability of positives over anchors that
    have at least one positive.
    """
    z = F.normalize(z.float(), dim=1)  # [N, D]
    n = z.shape[0]
    sim = (z @ z.t()) / temperature  # [N, N]
    eye = torch.eye(n, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(eye, float("-inf"))  # drop self-similarity
    same = group_labels.view(-1, 1) == group_labels.view(1, -1)  # [N, N]
    pos_mask = same & ~eye
    log_prob = F.log_softmax(sim, dim=1)  # [N, N]; masked self position is -inf
    pos_counts = pos_mask.sum(dim=1)  # [N]
    valid = pos_counts > 0
    if not bool(valid.any()):
        raise ValueError("infonce_loss needs at least one anchor with a positive pair")
    # Select positive log-probs via where (not multiply) so the masked -inf self
    # position never forms a (-inf * 0) NaN.
    pos_log_prob = torch.where(pos_mask, log_prob, torch.zeros_like(log_prob))
    per_anchor = pos_log_prob.sum(dim=1)[valid] / pos_counts[valid].float()
    return -per_anchor.mean()


def _validate_masks(pos_mask: torch.Tensor, neg_mask: torch.Tensor, n: int, device: torch.device) -> None:
    """Fail-loud check that a masked-loss batch is well formed (TRACER §B invariants)."""
    for name, m in (("pos_mask", pos_mask), ("neg_mask", neg_mask)):
        if tuple(m.shape) != (n, n):
            raise ValueError(f"{name} must be square [N, N]=[{n}, {n}], got {tuple(m.shape)}")
        if m.dtype != torch.bool:
            raise ValueError(f"{name} must be bool, got {m.dtype}")
        if m.device != device:
            raise ValueError(f"{name} device {m.device} != features device {device}")
        if bool(torch.diagonal(m).any()):
            raise ValueError(f"{name} must have a zero diagonal (no self pairs)")
        if not bool((m == m.t()).all()):
            raise ValueError(f"{name} must be symmetric")
    if bool((pos_mask & neg_mask).any()):
        raise ValueError("pos_mask and neg_mask must be disjoint (no pair is both positive and negative)")


def proj_infonce_loss(
    z: torch.Tensor,
    pos_mask: torch.Tensor,
    neg_mask: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Threshold-gated masked InfoNCE (TRACER Eq 15).

    For each anchor with >=1 positive AND >=1 negative, the numerator sums exp(sim)
    over P(a) and the denominator over P(a) u N(a) ONLY -- gray-zone pairs (in neither
    mask) are excluded from both. This is the point of departure from the supervised
    ``infonce_loss`` above, whose denominator spans every off-diagonal entry. Returns
    the mean over such anchors of ``-(logsumexp_P - logsumexp_{P u N})``.
    """
    z = F.normalize(z.float(), dim=1)  # [N, D]
    n = z.shape[0]
    sim = (z @ z.t()) / temperature  # [N, N]
    eye = torch.eye(n, dtype=torch.bool, device=z.device)
    pos = pos_mask & ~eye  # [N, N]
    cand = (pos_mask | neg_mask) & ~eye  # denominator support P u N
    valid = pos.any(dim=1) & (neg_mask & ~eye).any(dim=1)  # [N]
    if not bool(valid.any()):
        raise ValueError("proj_infonce_loss needs an anchor with both a positive and a negative")
    # Fill non-support positions with the dtype min so they vanish under logsumexp
    # without forming a (-inf * 0) NaN.
    neg_inf = torch.finfo(sim.dtype).min
    sim_pos = torch.where(pos, sim, torch.full_like(sim, neg_inf))
    sim_cand = torch.where(cand, sim, torch.full_like(sim, neg_inf))
    num = torch.logsumexp(sim_pos, dim=1)  # [N]
    den = torch.logsumexp(sim_cand, dim=1)  # [N]
    per_anchor = (num - den)[valid]
    return -per_anchor.mean()


def _resolve_loss_mode(batch: "FieldTrainingBatch", loss: str) -> str:
    """Resolve the fit objective for one batch (TRACER §B contract).

    Contract, enforced for EVERY selector (not just "auto"): a partial mask (exactly one
    of pos/neg) is malformed; carrying BOTH mask and group families is ambiguous. "auto"
    then routes masks -> "masked", group_labels -> "group"; "masked"/"group" require their
    own family. Any violation is a loud error, never a silent fallback.
    """
    has_pos = batch.pos_mask is not None
    has_neg = batch.neg_mask is not None
    has_masks = has_pos and has_neg
    has_group = batch.group_labels is not None
    if has_pos ^ has_neg:  # exactly one of pos/neg given
        raise ValueError("partial mask: provide BOTH pos_mask and neg_mask, or neither")
    if has_masks and has_group:  # ambiguous for every selector, including explicit ones
        raise ValueError("ambiguous batch carries BOTH masks and group_labels")
    if loss == "masked":
        if not has_masks:
            raise ValueError("loss='masked' requires pos_mask and neg_mask on the batch")
        return "masked"
    if loss == "group":
        if not has_group:
            raise ValueError("loss='group' requires group_labels on the batch")
        return "group"
    if loss != "auto":
        raise ValueError(f"unknown loss {loss!r}; expected 'auto' | 'masked' | 'group'")
    if has_masks:
        return "masked"
    if has_group:
        return "group"
    raise ValueError("batch carries neither group_labels nor pos/neg masks")


def _fit_one_head(
    batch: FieldTrainingBatch,
    *,
    mode: str,
    out_dim: int,
    epochs: int,
    lr: float,
    temperature: float,
) -> ProjectionHead:
    features = batch.features.float()  # [N, in_dim]
    n, in_dim = features.shape
    if mode == "masked":
        _validate_masks(batch.pos_mask, batch.neg_mask, n, features.device)
    # Scaled normal init keeps early logits in a sane range for any dim pair.
    weight = (torch.randn(out_dim, in_dim) * (in_dim ** -0.5)).detach().requires_grad_(True)
    opt = torch.optim.Adam([weight], lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        z = features @ weight.t()  # [N, out_dim]
        if mode == "masked":
            loss = proj_infonce_loss(z, batch.pos_mask, batch.neg_mask, temperature)
        else:
            loss = infonce_loss(z, batch.group_labels, temperature)
        loss.backward()
        opt.step()
    return ProjectionHead(weight=weight.detach().cpu().float().contiguous(), bias=None)
