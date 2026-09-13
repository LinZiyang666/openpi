"""CP2 single-key builder for GR00T N1.5 (ActionCache-style baseline, GR00T arm).

ActionCache (arXiv 2607.06370 §4.1, App. B.2 Table 5) keys GR00T on the
*encoded* VLM output embeddings concatenated with the *encoded* robot-state
features, then compresses with the same fixed sparse ternary projection as the
Pi0.5 arm. Here both encodings come from ``GrootStagedRunner.run_cp2_key_source``
(the action head's ``process_backbone_output`` and ``state_encoder``), so the
key describes exactly the conditioning the head sees.

Layout (``layout.kind == "groot_encoded_v1"``): the ``[N, C]`` encoded token
sequence is zero-padded on the token axis to ``token_len`` and flattened, then
the ``[S]`` encoded state is appended:

    D = token_len * feature_dim + state_feat_dim        (640*2048 + 1536 = 1,312,256)

``N`` varies with the instruction (LIBERO ~566 tokens), which is why the pad
exists; a sequence longer than ``token_len`` is a hard error rather than a
silent truncation. The projection itself is the Pi0.5 implementation
(``cp2_vlm_key_builder.project`` / ``get_projection_spec``): one numeric
contract (float32 accumulation) for both teachers.

Tensor lifetime (plan §3.2 R2-B11): ``collect`` keeps the read-only, possibly
inference-mode source for the current decision only; ``build`` materialises
the pad/concat buffer and the projected key as ordinary tensors (explicitly
under ``torch.inference_mode(False)``) so the stored key satisfies the storage
contract (``is_inference() == False``); ``clear`` drops the source reference.

Coupling map:
  DEPENDS ON:  openpi.cache.components.cp2_vlm_key_builder (project / spec),
               openpi.cache.groot.staged.GrootCP2KeySource (duck-typed)
  CONSUMED BY: CacheOrchestrator via config.py's key_builder factory
  IF CHANGED:  projection_meta() is the artifact/online binding key -- any
               layout change is a new ``layout.kind``
"""

from __future__ import annotations

import torch

from openpi.cache.components.cp2_vlm_key_builder import (
    ACCUMULATION_DTYPE,
    ProjectionSpec,
    get_projection_spec,
    project,
)
from openpi.cache.types import VLM_OUT, CheckpointID

KEY_BUILDER_TYPE = "cp2_groot_ternary"
LAYOUT_KIND = "groot_encoded_v1"
DEFAULT_TOKEN_LEN = 640
DEFAULT_FEATURE_DIM = 2048
DEFAULT_STATE_FEAT_DIM = 1536
DEFAULT_D = 500
DEFAULT_P = 0.01


def input_dim(token_len: int, feature_dim: int, state_feat_dim: int) -> int:
    """``D`` of the flattened key source: padded tokens times features, plus the state features."""
    return int(token_len) * int(feature_dim) + int(state_feat_dim)


def projection_meta_for(
    spec: ProjectionSpec, *, token_len: int, feature_dim: int, state_feat_dim: int
) -> dict:
    """The artifact / binding metadata: the Pi0.5 spec fields plus the GR00T layout."""
    meta = spec.meta()
    meta["layout"] = {
        "kind": LAYOUT_KIND,
        "token_len": int(token_len),
        "feature_dim": int(feature_dim),
        "state_feat_dim": int(state_feat_dim),
    }
    return meta


def flatten_source(
    vl_encoded: torch.Tensor,
    state_encoded: torch.Tensor,
    *,
    token_len: int,
    feature_dim: int,
    state_feat_dim: int,
) -> torch.Tensor:
    """``[N, C]`` + ``[S]`` -> float32 ``[D]`` (zero-padded tokens, then state).

    Ordinary (non-inference) tensor by construction; called outside the runner
    session by the builder and by the offline tooling alike.
    """
    if vl_encoded.dim() != 2 or vl_encoded.shape[1] != feature_dim:
        raise RuntimeError(
            f"cp2_groot_ternary: encoded VLM output must be [N, {feature_dim}], "
            f"got {tuple(vl_encoded.shape)}"
        )
    n_tokens = int(vl_encoded.shape[0])
    if n_tokens > token_len:
        raise RuntimeError(
            f"cp2_groot_ternary: {n_tokens} tokens exceed token_len={token_len}; "
            "raise token_len (a new layout / artifact) rather than truncating."
        )
    if state_encoded.dim() != 1 or state_encoded.shape[0] != state_feat_dim:
        raise RuntimeError(
            f"cp2_groot_ternary: encoded state must be [{state_feat_dim}], "
            f"got {tuple(state_encoded.shape)}"
        )
    with torch.inference_mode(False):
        buf = torch.zeros(
            input_dim(token_len, feature_dim, state_feat_dim),
            dtype=torch.float32,
            device=vl_encoded.device,
        )
        span = n_tokens * feature_dim
        buf[:span] = vl_encoded.detach().to(torch.float32).reshape(-1)
        buf[token_len * feature_dim :] = state_encoded.detach().to(torch.float32)
    if not torch.isfinite(buf).all():
        raise RuntimeError("cp2_groot_ternary: non-finite values in the key source")
    return buf


class GrootCP2TernaryKeyBuilder:
    """QueryKeyBuilder for CP2 on GR00T: encoded VLM + encoded state, projected.

    ``collect(CP2, cp2_source=GrootCP2KeySource, ...)`` keeps the source;
    ``build(CP2)`` returns ``{"vlm_out": [d]}`` (CPU float32, storable). Any
    other checkpoint or a missing source is a programming error, never a
    silent MISS.
    """

    def __init__(
        self,
        seed: int,
        d: int = DEFAULT_D,
        p: float = DEFAULT_P,
        token_len: int = DEFAULT_TOKEN_LEN,
        feature_dim: int = DEFAULT_FEATURE_DIM,
        state_feat_dim: int = DEFAULT_STATE_FEAT_DIM,
    ) -> None:
        self.token_len = int(token_len)
        self.feature_dim = int(feature_dim)
        self.state_feat_dim = int(state_feat_dim)
        self._spec = get_projection_spec(
            seed, d, p, input_dim(self.token_len, self.feature_dim, self.state_feat_dim)
        )
        self._source = None

    @property
    def spec(self) -> ProjectionSpec:
        """The frozen sparse ternary projection (index tables) this builder applies."""
        return self._spec

    def projection_meta(self) -> dict:
        """The binding metadata (spec fields + ``layout``) an artifact built with this builder must carry."""
        return projection_meta_for(
            self._spec,
            token_len=self.token_len,
            feature_dim=self.feature_dim,
            state_feat_dim=self.state_feat_dim,
        )

    def collect(self, checkpoint_id: CheckpointID, **stage_outputs) -> None:
        """Keep this decision's ``cp2_source`` (a ``GrootCP2KeySource``); CP2 only, source required."""
        self._source = None
        if checkpoint_id is not CheckpointID.CP2:
            raise ValueError(
                f"cp2_groot_ternary serves CP2 only, collect() called for {checkpoint_id}"
            )
        source = stage_outputs.get("cp2_source")
        if source is None:
            # Fail loud: a silent MISS here would look like a cold cache.
            raise RuntimeError(
                "cp2_groot_ternary.collect requires cp2_source=GrootCP2KeySource "
                "(the interceptor must call runner.run_cp2_key_source before check())."
            )
        for name in ("vl_encoded", "state_encoded"):
            if not isinstance(getattr(source, name, None), torch.Tensor):
                raise RuntimeError(f"cp2_groot_ternary: cp2_source.{name} is not a tensor")
        self._source = source

    def build(self, checkpoint_id: CheckpointID) -> dict[str, torch.Tensor]:
        """Pad / concatenate / project the collected source into ``{"vlm_out": [d]}`` (CPU float32, storable)."""
        if checkpoint_id is not CheckpointID.CP2:
            raise ValueError(
                f"cp2_groot_ternary serves CP2 only, build() called for {checkpoint_id}"
            )
        if self._source is None:
            raise RuntimeError("cp2_groot_ternary.build called before collect")
        h = flatten_source(
            self._source.vl_encoded,
            self._source.state_encoded,
            token_len=self.token_len,
            feature_dim=self.feature_dim,
            state_feat_dim=self.state_feat_dim,
        )
        with torch.inference_mode(False):
            key = project(h, self._spec).detach().cpu().float().contiguous()
            if key.is_inference():  # pragma: no cover - defensive; project() allocates fresh
                key = key.clone()
        return {VLM_OUT: key}

    @property
    def cached_data(self) -> dict[str, torch.Tensor]:
        """The collected source tensors (empty after ``clear``), for the orchestrator's bookkeeping."""
        if self._source is None:
            return {}
        return {"vl_encoded": self._source.vl_encoded, "state_encoded": self._source.state_encoded}

    def clear(self) -> None:
        """Drop the source reference: the inference-mode tensors must not outlive the decision."""
        self._source = None


__all__ = [
    "ACCUMULATION_DTYPE",
    "DEFAULT_D",
    "DEFAULT_FEATURE_DIM",
    "DEFAULT_P",
    "DEFAULT_STATE_FEAT_DIM",
    "DEFAULT_TOKEN_LEN",
    "KEY_BUILDER_TYPE",
    "LAYOUT_KIND",
    "GrootCP2TernaryKeyBuilder",
    "flatten_source",
    "input_dim",
    "projection_meta_for",
]
