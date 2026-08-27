"""Dispatch-surface judge: three-tier (s, v) verdict from a calibrated artifact.

Implements the collapsed dispatch surface of the dispatch note (v3): every CP1
verdict is FULL_HIT (tau = N), WARM_START at the single stored depth
start_t = 0.3 (tau = 0.7 N), or MISS (tau = 0), decided by two conformally
calibrated monotone boundaries over the retrieval statistics

    s = results[0].score            (fused similarity, backend-produced)
    v = weighted top-k action disagreement (local action disagreement)

The boundaries, the diagonal weighting W, the top-k width and every
calibration-time constant live in a ``SurfaceArtifact`` persisted as NPZ +
embedded JSON (never pickle). The artifact also carries a ``retrieval_contract``
that the config layer compares against the live YAML / loaded library at
assembly time, so a surface certificate can never be silently applied to a
retrieval space it was not calibrated for.

Public interface:
  - ``weighted_topk_disagreement`` / ``weighted_chunk_deviation``: the shared
    (s, v, Y) metric implementations. The offline calibration pipeline under
    ``exp/dispatch_surface/`` imports THESE functions, so calibration-side and
    verdict-side values agree by construction.
  - ``SurfaceArtifact`` with ``load_surface_artifact`` / ``save_surface_artifact``.
  - ``SurfaceJudge``: the pluggable verdict component (``judge.type:
    dispatch_surface``).

Coupling map:
  DEPENDS ON:  SearchResultLite.score semantics; PayloadView.get_many for
               candidate action chunks; CANONICAL_DENOISE_TIMESTEPS.
  CONSUMED BY: CacheOrchestrator.check() via the SimilarityJudge protocol;
               config.py factory + contract validation.
  Purity contract: read-only access via ``view``; never writes storage.
"""

from __future__ import annotations

import dataclasses
import json
import math
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch

from openpi.cache.components.judge import HitType, JudgeResult
from openpi.cache.storage_types import RetrievalSignals, SearchResultLite
from openpi.cache.types import CANONICAL_DENOISE_TIMESTEPS, CheckpointID

if TYPE_CHECKING:
    from openpi.cache.components.factors.base import HistoryView
    from openpi.cache.components.payload_view import PayloadView


SURFACE_ARTIFACT_SCHEMA_VERSION = 1

# The dispatch-surface line froze the single warm tier at start_t = 0.3
# (tau = 7 of N = 10). An artifact carrying any other tier is not a valid
# certificate for this line, canonical timestep or not.
PINNED_START_T_WS = 0.3

# Dimensions whose library-wide variance falls below this are treated as
# padding and excluded from v / Y (same constant as factors/topk.py).
_ACTIVE_VARIANCE_EPS = 1e-8


# ------------------------------------------------------------------
# Shared metric implementations (calibration imports these verbatim)
# ------------------------------------------------------------------


def weighted_topk_disagreement(
    chunks: torch.Tensor,
    w: torch.Tensor,
    active_mask: torch.Tensor,
    h_exec: int,
) -> float:
    """Local action disagreement v over the executed window of top-k chunks.

    Args:
        chunks: [K, H, D] float tensor of candidate action chunks.
        w: [D] per-dim diagonal weights (inverse library sigma).
        active_mask: [D] bool mask of non-padding dims.
        h_exec: number of leading chunk steps that are actually executed.

    Returns:
        Mean squared weighted deviation from the candidate mean, or NaN when
        fewer than two candidates are available (the caller treats NaN as
        fail-closed).
    """
    if chunks.ndim != 3:
        raise ValueError(f"chunks must be [K, H, D], got shape {tuple(chunks.shape)}")
    k_eff = chunks.shape[0]
    if k_eff < 2:
        return float("nan")
    window = chunks[:, :h_exec, :].to(torch.float32)
    mean = window.mean(dim=0, keepdim=True)
    diff = (window - mean) * w.to(torch.float32)
    diff = diff[..., active_mask]
    # Mean over candidates and executed steps of the squared weighted norm.
    return float(diff.pow(2).sum(dim=-1).mean())


def weighted_chunk_deviation(
    chunk_a: torch.Tensor,
    chunk_b: torch.Tensor,
    w: torch.Tensor,
    active_mask: torch.Tensor,
    h_exec: int,
) -> float:
    """Deviation Y between two action chunks over the executed window.

    Mean (over the first ``h_exec`` steps) of the weighted L2 distance across
    active dims. Both chunks are [H, D]. A shape mismatch raises rather than
    broadcasting (the [H,D] vs [1,H,D] broadcast bug documented in
    shadow_teacher.py motivates the strictness).
    """
    if chunk_a.shape != chunk_b.shape or chunk_a.ndim != 2:
        raise ValueError(
            f"chunks must share [H, D] shape, got {tuple(chunk_a.shape)} vs {tuple(chunk_b.shape)}"
        )
    a = chunk_a[:h_exec].to(torch.float32)
    b = chunk_b[:h_exec].to(torch.float32)
    diff = (a - b) * w.to(torch.float32)
    diff = diff[..., active_mask]
    return float(torch.linalg.vector_norm(diff, dim=-1).mean())


def surface_verdict(
    s: float,
    v: float | None,
    v_bin_edges: np.ndarray,
    s_min_full: np.ndarray,
    s_min_warm: np.ndarray,
    *,
    uses_disagreement: bool,
) -> str:
    """The deployed three-tier decision rule as one shared pure function.

    Returns "full" | "warm" | "miss". BOTH the online ``SurfaceJudge`` and the
    offline delta-selection / evaluation code call THIS function, so the
    fitted-versus-deployed verdict parity holds by construction (G2-B1).

    Semantics (identical to the judge's fail-closed table): non-finite s is a
    miss; with ``uses_disagreement`` a missing/non-finite v or v above the
    rightmost bin edge is a miss, v below the leftmost edge clamps to bin 0;
    without it v is ignored entirely.
    """
    if not math.isfinite(s):
        return "miss"
    if uses_disagreement:
        if v is None or not math.isfinite(v):
            return "miss"
        if v > v_bin_edges[-1]:
            return "miss"
        b = int(np.clip(np.searchsorted(v_bin_edges, v, side="right") - 1,
                        0, len(v_bin_edges) - 2))
    else:
        b = 0
    if s >= s_min_full[b]:
        return "full"
    if s >= s_min_warm[b]:
        return "warm"
    return "miss"


def compute_library_action_weights(
    chunks: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-dim inverse-sigma weights and active mask from library action chunks.

    Args:
        chunks: [N_entries, H, D] stacked library action chunks.

    Returns:
        (w [D] float32, active_mask [D] bool). Padding dims (variance below
        ``_ACTIVE_VARIANCE_EPS``) are masked out; their weight is set to 0 so
        a stale mask cannot resurrect them numerically.
    """
    flat = chunks.reshape(-1, chunks.shape[-1]).to(torch.float32)
    sigma = flat.std(dim=0, unbiased=False)
    active = flat.var(dim=0, unbiased=False) > _ACTIVE_VARIANCE_EPS
    w = 1.0 / sigma.clamp_min(1e-6)
    w = torch.where(active, w, torch.zeros_like(w))
    return w, active


# ------------------------------------------------------------------
# Artifact schema, persistence and validation
# ------------------------------------------------------------------


@dataclasses.dataclass
class SurfaceArtifact:
    """Calibrated dispatch surface plus its binding contract.

    ``uses_disagreement=False`` marks the s-only nested-ablation artifact: the
    judge then never fetches payloads nor computes v, ``v_bin_edges`` must be
    exactly [-inf, +inf], and ``k`` is ignored (effective required top-k is 1).
    """

    schema_version: int
    k: int
    h_exec: int
    w: np.ndarray
    active_mask: np.ndarray
    start_t_ws: float
    delta: float
    alpha: float
    uses_disagreement: bool
    v_bin_edges: np.ndarray
    s_min_full: np.ndarray
    s_min_warm: np.ndarray
    conformal_c: float
    n_calibration_episodes: int
    retrieval_contract: dict
    meta: dict

    def validate(self) -> None:
        """Fail-fast structural validation. Raises ValueError on any breach."""
        if self.schema_version != SURFACE_ARTIFACT_SCHEMA_VERSION:
            raise ValueError(
                f"surface artifact schema_version {self.schema_version} != "
                f"{SURFACE_ARTIFACT_SCHEMA_VERSION}"
            )
        w = np.asarray(self.w, dtype=np.float32)
        mask = np.asarray(self.active_mask, dtype=bool)
        if w.ndim != 1 or mask.shape != w.shape:
            raise ValueError("w and active_mask must be 1-D arrays of equal length")
        if not mask.any():
            raise ValueError("active_mask must have at least one active dim")
        if not np.isfinite(w).all() or (w[mask] <= 0).any():
            raise ValueError("w must be finite and strictly positive on active dims")

        edges = np.asarray(self.v_bin_edges, dtype=np.float64)
        full = np.asarray(self.s_min_full, dtype=np.float64)
        warm = np.asarray(self.s_min_warm, dtype=np.float64)
        if edges.ndim != 1 or full.ndim != 1 or warm.ndim != 1:
            raise ValueError("v_bin_edges / s_min_full / s_min_warm must be 1-D")
        if len(full) != len(edges) - 1 or len(warm) != len(full):
            raise ValueError("boundary arrays must have len(v_bin_edges) - 1 bins")

        if self.uses_disagreement:
            if self.k < 2:
                raise ValueError("k must be >= 2 when uses_disagreement is true")
            if not np.isfinite(edges).all() or not (np.diff(edges) > 0).all():
                raise ValueError("v_bin_edges must be finite and strictly increasing")
        else:
            # s-only sentinel: exactly one bin spanning the whole real line.
            if len(edges) != 2 or edges[0] != -np.inf or edges[1] != np.inf:
                raise ValueError(
                    "s-only artifact requires v_bin_edges == [-inf, +inf], "
                    f"got {edges.tolist()}"
                )

        for name, arr in (("s_min_full", full), ("s_min_warm", warm)):
            if np.isnan(arr).any() or (arr == -np.inf).any():
                raise ValueError(f"{name} must not contain NaN or -inf")
            if (np.diff(arr) < 0).any():
                raise ValueError(f"{name} must be nondecreasing in v")
        if (warm > full).any():
            raise ValueError("s_min_warm must be elementwise <= s_min_full")

        st = round(float(self.start_t_ws), 4)
        if st not in CANONICAL_DENOISE_TIMESTEPS:
            raise ValueError(
                f"start_t_ws must be a canonical denoise timestep, got {self.start_t_ws}"
            )
        if st != PINNED_START_T_WS:
            raise ValueError(
                f"start_t_ws is pinned to {PINNED_START_T_WS} for the dispatch-surface "
                f"line, got {self.start_t_ws}"
            )
        if self.h_exec <= 0:
            raise ValueError(f"h_exec must be positive, got {self.h_exec}")
        if not (math.isfinite(self.delta) and self.delta > 0):
            raise ValueError(f"delta must be finite and > 0, got {self.delta}")
        if not (0 < self.alpha <= 0.5):
            raise ValueError(f"alpha must be in (0, 0.5], got {self.alpha}")
        # conformal_c may be +inf, but only as the coherent fail-valid
        # degenerate artifact whose boundaries are all +inf (rule == all-MISS).
        if math.isnan(self.conformal_c) or self.conformal_c == -np.inf:
            raise ValueError("conformal_c must not be NaN or -inf")
        if self.conformal_c == np.inf and (np.isfinite(full).any() or np.isfinite(warm).any()):
            raise ValueError(
                "conformal_c == +inf requires all boundaries to be +inf (all-MISS artifact)"
            )
        if not isinstance(self.retrieval_contract, dict) or not self.retrieval_contract:
            raise ValueError("retrieval_contract must be a non-empty dict")
        # Cross-field contract invariants: an artifact whose contract disagrees
        # with its own arrays would make runner and judge use different
        # parameters (G2-B7). Fail at load, not at use.
        contract = self.retrieval_contract
        if contract.get("h_exec") != self.h_exec:
            raise ValueError(
                f"contract h_exec={contract.get('h_exec')} != artifact h_exec={self.h_exec}"
            )
        if self.uses_disagreement and contract.get("top_k") != self.k:
            raise ValueError(
                f"contract top_k={contract.get('top_k')} != artifact k={self.k}"
            )
        if contract.get("action_dim") != len(w):
            raise ValueError(
                f"contract action_dim={contract.get('action_dim')} != len(w)={len(w)}"
            )


def save_surface_artifact(artifact: SurfaceArtifact, path: str) -> None:
    """Persist an artifact as NPZ with JSON-encoded scalar/contract fields."""
    artifact.validate()
    scalars = {
        "schema_version": artifact.schema_version,
        "k": artifact.k,
        "h_exec": artifact.h_exec,
        "start_t_ws": artifact.start_t_ws,
        "delta": artifact.delta,
        "alpha": artifact.alpha,
        "uses_disagreement": artifact.uses_disagreement,
        # inf does not survive strict JSON; encode as string sentinel.
        "conformal_c": "inf" if artifact.conformal_c == np.inf else artifact.conformal_c,
        "n_calibration_episodes": artifact.n_calibration_episodes,
    }
    np.savez(
        path,
        w=np.asarray(artifact.w, dtype=np.float32),
        active_mask=np.asarray(artifact.active_mask, dtype=bool),
        v_bin_edges=np.asarray(artifact.v_bin_edges, dtype=np.float64),
        s_min_full=np.asarray(artifact.s_min_full, dtype=np.float64),
        s_min_warm=np.asarray(artifact.s_min_warm, dtype=np.float64),
        scalars_json=np.frombuffer(json.dumps(scalars).encode("utf-8"), dtype=np.uint8),
        contract_json=np.frombuffer(
            json.dumps(artifact.retrieval_contract, sort_keys=True).encode("utf-8"),
            dtype=np.uint8,
        ),
        meta_json=np.frombuffer(json.dumps(artifact.meta).encode("utf-8"), dtype=np.uint8),
    )


def load_surface_artifact(path: str) -> SurfaceArtifact:
    """Load and validate an artifact. ``allow_pickle=False`` is non-negotiable."""
    with np.load(path, allow_pickle=False) as data:
        required = {
            "w", "active_mask", "v_bin_edges", "s_min_full", "s_min_warm",
            "scalars_json", "contract_json", "meta_json",
        }
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"surface artifact missing fields: {sorted(missing)}")
        scalars = json.loads(bytes(data["scalars_json"]).decode("utf-8"))
        contract = json.loads(bytes(data["contract_json"]).decode("utf-8"))
        meta = json.loads(bytes(data["meta_json"]).decode("utf-8"))
        c_raw = scalars["conformal_c"]
        artifact = SurfaceArtifact(
            schema_version=int(scalars["schema_version"]),
            k=int(scalars["k"]),
            h_exec=int(scalars["h_exec"]),
            w=data["w"],
            active_mask=data["active_mask"],
            start_t_ws=float(scalars["start_t_ws"]),
            delta=float(scalars["delta"]),
            alpha=float(scalars["alpha"]),
            uses_disagreement=bool(scalars["uses_disagreement"]),
            v_bin_edges=data["v_bin_edges"],
            s_min_full=data["s_min_full"],
            s_min_warm=data["s_min_warm"],
            conformal_c=float("inf") if c_raw == "inf" else float(c_raw),
            n_calibration_episodes=int(scalars["n_calibration_episodes"]),
            retrieval_contract=contract,
            meta=meta,
        )
    artifact.validate()
    return artifact


# ------------------------------------------------------------------
# SurfaceJudge
# ------------------------------------------------------------------


class SurfaceJudge:
    """Three-tier dispatch-surface judge (``judge.type: dispatch_surface``).

    Verdict rule (per plan section 2): with v-bin index b(v),
    ``s >= s_min_full[b]`` -> FULL_HIT; else ``s >= s_min_warm[b]`` ->
    WARM_START(start_t_ws); else MISS.

    Fail-closed table for ``uses_disagreement=True`` (all -> MISS): empty
    results; non-finite s; view is None; payload fetch failure or fewer than
    2 candidates; non-finite v; v above the rightmost bin edge. The single
    non-MISS exception is v below the leftmost edge, which clamps to the
    leftmost bin (conservative under the (A1) monotonicity direction). For
    the s-only artifact only empty results / non-finite s remain, and neither
    the view nor v is ever touched. CP3 is a defensive MISS in both modes.
    """

    def __init__(self, artifact_path: str, *, export_factor_outputs: bool = False) -> None:
        self._artifact_path = artifact_path
        self.artifact = load_surface_artifact(artifact_path)
        self.export_factor_outputs = export_factor_outputs
        # Read by build_per_connection_components and forwarded to the search
        # strategy as min_top_k_hint (config.py:2856). The s-only artifact
        # needs only the winner.
        self.min_required_top_k = self.artifact.k if self.artifact.uses_disagreement else 1
        self._w = torch.from_numpy(np.asarray(self.artifact.w, dtype=np.float32))
        self._active = torch.from_numpy(np.asarray(self.artifact.active_mask, dtype=bool))

    # -- verdict ----------------------------------------------------

    def __call__(
        self,
        results: list[SearchResultLite],
        checkpoint_id: CheckpointID,
        cached_data: dict[str, torch.Tensor],
        *,
        view: Optional["PayloadView"] = None,
        history: Optional["HistoryView"] = None,
        retrieval_signals: Optional[RetrievalSignals] = None,
    ) -> JudgeResult:
        art = self.artifact
        if checkpoint_id != CheckpointID.CP1:
            # Surface semantics are CP1-only; config validation rejects other
            # placements, this branch is defensive.
            return self._emit(HitType.MISS, None, s=None, v=None, v_bin=None, src="cp_guard")
        if not results:
            return self._emit(HitType.MISS, None, s=None, v=None, v_bin=None, src="empty")
        s = float(results[0].score)
        if not math.isfinite(s):
            return self._emit(HitType.MISS, None, s=s, v=None, v_bin=None, src="s_nonfinite")
        winner_id = results[0].id

        if art.uses_disagreement:
            v = self._compute_v(results, view)
            if not math.isfinite(v):
                return self._emit(HitType.MISS, None, s=s, v=v, v_bin=None, src="v_failclosed")
        else:
            v = None

        # The decision itself is the shared pure function — the SAME code the
        # offline delta selection executes (G2-B1 parity by construction).
        verdict = surface_verdict(
            s, v, art.v_bin_edges, art.s_min_full, art.s_min_warm,
            uses_disagreement=art.uses_disagreement,
        )
        if art.uses_disagreement and v is not None and v > art.v_bin_edges[-1]:
            v_bin = None
            src = "v_oob_right"
        else:
            v_bin = (
                int(np.clip(np.searchsorted(art.v_bin_edges, v, side="right") - 1,
                            0, len(art.v_bin_edges) - 2))
                if art.uses_disagreement else 0
            )
            src = verdict if verdict != "miss" else "below"
        if verdict == "full":
            return self._emit(HitType.FULL_HIT, winner_id, s=s, v=v, v_bin=v_bin, src="full")
        if verdict == "warm":
            return self._emit(
                HitType.WARM_START, winner_id, s=s, v=v, v_bin=v_bin, src="warm",
                start_t=art.start_t_ws,
            )
        return self._emit(HitType.MISS, None, s=s, v=v, v_bin=v_bin, src=src)

    def _compute_v(
        self, results: list[SearchResultLite], view: Optional["PayloadView"]
    ) -> float:
        """Fetch top-k candidate chunks and compute v; NaN on any failure."""
        if view is None:
            return float("nan")
        ids = [r.id for r in results[: self.artifact.k]]
        if len(ids) < 2:
            return float("nan")
        try:
            payloads = view.get_many(ids)
            chunks = torch.stack(
                [torch.as_tensor(p.action_chunk, dtype=torch.float32) for p in payloads],
                dim=0,
            )
        except Exception:
            return float("nan")
        if chunks.ndim != 3 or chunks.shape[0] < 2:
            return float("nan")
        return weighted_topk_disagreement(chunks, self._w, self._active, self.artifact.h_exec)

    def _emit(
        self,
        hit_type: HitType,
        winner_id: str | None,
        *,
        s: float | None,
        v: float | None,
        v_bin: int | None,
        src: str,
        start_t: float | None = None,
    ) -> JudgeResult:
        factor_outputs = None
        if self.export_factor_outputs:
            # NaN -> None so the dict survives strict JSON consumers, same
            # convention as the composite judge wire schema.
            def _clean(x: float | None) -> float | None:
                return None if x is None or not math.isfinite(x) else x

            factor_outputs = {
                "s": _clean(s),
                "v": _clean(v),
                "v_bin": v_bin,
                "verdict_src": src,
            }
        return JudgeResult(hit_type, winner_id, start_t=start_t, factor_outputs=factor_outputs)

    # -- lifecycle --------------------------------------------------

    def on_episode_start(self) -> None:
        """Stateless judge; nothing to reset. Called by Orchestrator."""

    def record_action(self, action_chunk: torch.Tensor) -> None:
        """Stateless judge; broadcast action is ignored."""
