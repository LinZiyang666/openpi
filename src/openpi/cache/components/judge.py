"""Judge: decide whether search results constitute a cache hit.

Data flow: SearchResultLite.score (from CacheStorage) -> judge() -> JudgeResult

Coupling map:
  DEPENDS ON:  SearchResultLite.score semantics (Step 3 backend-dependent)
               * Single-field cosine: score in [-1, 1]
               * Multi-field RRF: small positive numbers, scale depends on RRF k param
               * IF backend changes: thresholds MUST be recalibrated
  MAY DEPEND ON: KeyBuilder.cached_data (for future re-scoring judges)
  CONSUMED BY: CacheOrchestrator.check()
  Purity contract:
    A judge MUST NOT write to CacheStorage. Read-only access via the
    optional `view` (PayloadView) parameter is permitted at verdict time
    so composite judges can compute factor descriptors over candidate
    payloads + neighbor entries. See `cache_system.md` §5.6 / §5.11 for
    the contract refinement.
  IF CHANGED:  Only affects hit/miss decision, no downstream structural impact
"""

from __future__ import annotations

import inspect
import logging
import math
import os
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Optional, Protocol, runtime_checkable

import torch

from openpi.cache.storage_types import RetrievalSignals, SearchResultLite
from openpi.cache.types import CANONICAL_DENOISE_TIMESTEPS, CheckpointID

# Verdict-pipeline debug instrumentation. Gated on env var so production
# runs pay zero cost. Toggle from server shell:
#   OPENPI_CACHE_VERDICT_DEBUG=1 uv run scripts/serve_policy.py ... |& tee /tmp/vd.log
# then `grep '\[vd ' /tmp/vd.log` inspects per-verdict structured lines.
_VERDICT_DEBUG = os.environ.get("OPENPI_CACHE_VERDICT_DEBUG") == "1"
_verdict_logger = logging.getLogger("openpi.cache.verdict_debug")

if TYPE_CHECKING:
    from openpi.cache.components.factors.base import HistoryView, LibraryStats
    from openpi.cache.components.payload_view import PayloadView


def judge_accepts_query_keys(judge, *, allow_var_keyword: bool = True) -> bool:
    """Probe once whether ``judge.__call__`` can receive ``query_keys=...``.

    The X14 router decides on the post-``build()`` query keys, but the
    ``SimilarityJudge`` protocol does not carry them: injecting unconditionally
    would ``TypeError`` on every judge with an explicit keyword-only signature
    and no ``**kwargs`` (``CompositeJudge``, ``DumpingJudge``). Callers probe
    once at construction time and pass the kwarg only when this returns True.

    ``allow_var_keyword`` selects the strictness:

      - ``True`` (Orchestrator): a ``**kwargs`` judge counts as accepting. Such
        judges silently swallow the kwarg, so injection is safe and the rule
        stays a single signature test.
      - ``False`` (DumpingJudge's inner forward): only an explicit ``query_keys``
        parameter counts. A legacy inner judge would merely swallow the kwarg
        and gain nothing, so keeping it out preserves a byte-identical inner
        call for every dump-wrapped legacy / composite config.
    """
    return judge_accepts_kwarg(judge, "query_keys", allow_var_keyword=allow_var_keyword)


def judge_accepts_kwarg(judge, name: str, *, allow_var_keyword: bool = True) -> bool:
    """Probe once whether ``judge.__call__`` can receive ``name=...``.

    The generic form of the X14 ``query_keys`` probe, reused by the X15
    ``step_features`` seam. Each additive judge kwarg is injected only for
    judges that declare it, so every legacy judge keeps a byte-identical call.

    ``allow_var_keyword=True`` counts a ``**kwargs`` judge as accepting (safe:
    it silently swallows the kwarg); ``False`` requires an explicit parameter,
    which is what keeps a dump-wrapped legacy inner judge untouched.
    """
    call = getattr(judge, "__call__", None)
    if call is None:
        return False
    try:
        params = inspect.signature(call).parameters
    except (TypeError, ValueError):
        # Un-introspectable callable (C extension / exotic proxy): stay on the
        # conservative side and do not inject.
        return False
    if name in params:
        return True
    if not allow_var_keyword:
        return False
    return any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


class HitType(Enum):
    """Cache hit classification.

    Coupling:
      - CONSUMED BY: CacheOrchestrator (packs into CheckResult), Interceptor (controls stage skip)
    """

    MISS = auto()
    FULL_HIT = auto()
    WARM_START = auto()


@dataclass
class JudgeResult:
    """Structured return type for SimilarityJudge.

    Payload invariant (X14 refinement): FULL_HIT and WARM_START must include
    winner_id — Orchestrator skips fetch when winner_id is None — **except**
    for a payloadless FULL_HIT, which is legal if and only if
    ``hit_override is True``. That single exception exists for the RL router's
    student arm, where the verdict routes execution to a sidecar and no cached
    payload is ever read. Every other path keeps the original invariant.

    ``factor_outputs`` is an optional diagnostic payload populated only by
    CompositeJudge when its config carries ``export_factor_outputs: true``.
    Schema: ``{"raw": dict[str, float|None], "norm": dict[str, float|None],
    "score": float|None, "sentinel": str|None}``. NaN values are pre-converted
    to ``None`` so the dict round-trips through strict JSON parsers (jq,
    pandas) without relying on Python's lax ``allow_nan=True``. Orchestrator
    forwards this on ``CheckResult``; Interceptor surfaces it through
    ``__hit_meta__`` for client-side per-step logging.

    ``hit_override`` is the tri-state executor selector consumed by
    ``InferenceInterceptor``'s FULL_HIT dispatch:

      - ``True``  — payloadless FULL_HIT; the wired ``hit_executor`` produces
        the action (router student arm). ``winner_id`` MUST be None.
      - ``False`` — force cached-action replay even when a ``hit_executor`` is
        wired (router cache arm). ``winner_id`` MUST be set.
      - ``None``  — legacy verdict; interceptor behaviour is unchanged.

    ``router_outputs`` is the RL router's per-verdict provenance dict
    (``{decision_idx, arm_sampled, arm_executed, probs, temperature,
    weights_version, seed_ep, fallback}``). Only ``MlpRouterJudge`` populates
    it; every other judge leaves it None so the ``__hit_meta__`` wire is
    unchanged. Features / logits never ride this channel.
    """

    hit_type: HitType
    winner_id: str | None = None
    start_t: float | None = None
    composer_score: Optional[float] = None
    factor_outputs: Optional[dict] = None
    hit_override: Optional[bool] = None
    router_outputs: Optional[dict] = None


@runtime_checkable
class SimilarityJudge(Protocol):
    """Judge whether a search result constitutes a cache hit.

    Data flow: SearchResultLite.score (from CacheStorage) -> judge() -> JudgeResult
    Coupling:
      - DEPENDS ON: SearchResultLite.score semantics (Step 3 backend-dependent)
        * Single-field cosine: score in [-1, 1]
        * Multi-field RRF: small positive numbers, scale depends on RRF k param
        * IF backend changes: thresholds MUST be recalibrated
      - MAY DEPEND ON: KeyBuilder.cached_data (for future re-scoring judges)
      - CONSUMED BY: CacheOrchestrator.check()
      - Purity contract: read-only via `view`; never writes storage.
      - IF CHANGED: Only affects hit/miss decision, no downstream structural impact

    The `view`, `history`, and `retrieval_signals` keyword-only parameters
    carry verdict-time objects injected by Orchestrator. Existing judges that
    do not need them accept and ignore them via `**kwargs`; Orchestrator only
    builds and injects the facades from B1 onward (CompositeJudge land) and the
    retrieval signals when a dual-retrieval strategy provides them (Phase 3).

    `query_keys` (X14) is NOT part of this signature: it is injected only into
    judges whose `__call__` declares it explicitly (or accepts `**kwargs`), as
    decided once at Orchestrator build time by `judge_accepts_query_keys`.
    CompositeJudge and DumpingJudge's inner-forward path are therefore never
    perturbed. Only `MlpRouterJudge` consumes it — it decides on the post-
    `build()` query keys and never sees the library side (results / view /
    history / retrieval_signals).
    """

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
        """Judge the top search results.

        Args:
            results: Search results sorted by descending score (from CacheStorage).
            checkpoint_id: CP1 or CP3.
            cached_data: Raw tensors from KeyBuilder.cached_data.
            view: Read-only facade over CacheStorage (B1+ injection;
                None for the B0 path and for legacy judges).
            history: Per-episode action / state snapshot (B1+ injection).
            retrieval_signals: Per-query failure-aware signals from a
                dual-retrieval strategy (TRACER M2 / Phase 3). None for
                strategies that do not produce them; only failure_aware_gate
                consumes it. Legacy judges accept and ignore it via **kwargs.

        Returns:
            JudgeResult with hit_type, winner_id, and optional start_t.
        """
        ...


class AlwaysHitJudge:
    """Always returns FULL_HIT for the top-1 result (if any results exist).

    Useful for testing / calibration: confirms the full hit path works
    end-to-end without threshold tuning.
    """

    def __call__(
        self,
        results: list[SearchResultLite],
        checkpoint_id: CheckpointID,
        cached_data: dict[str, torch.Tensor],
        **kwargs,
    ) -> JudgeResult:
        if not results:
            return JudgeResult(HitType.MISS)
        return JudgeResult(HitType.FULL_HIT, results[0].id)

    def on_episode_start(self) -> None:
        """Clear internal history buffer. Called by Orchestrator at episode start."""

    def record_action(self, action_chunk: torch.Tensor) -> None:
        """Receive Orchestrator-broadcast action. Pure local buffer op."""


class AlwaysWarmStartJudge:
    """Always returns WARM_START with a fixed start_t for the top-1 result.

    Used to sweep the success_rate ~ start_t curve under a constant (forced)
    warm-start regime, independent of similarity score. Empty result set
    falls back to MISS (cache truly empty / first step of episode).

    Restricted to CP1 — CP3 has no warm start support. Config-level
    validation rejects CP3 usage; the runtime FULL_HIT fallback below is
    defensive only and should be unreachable for validated configs.
    """

    def __init__(self, start_t: float) -> None:
        st = round(start_t, 4)
        if st not in CANONICAL_DENOISE_TIMESTEPS:
            raise ValueError(
                f"start_t must round to one of {sorted(CANONICAL_DENOISE_TIMESTEPS)}, "
                f"got {start_t}"
            )
        self._start_t = st

    def __call__(
        self,
        results: list[SearchResultLite],
        checkpoint_id: CheckpointID,
        cached_data: dict[str, torch.Tensor],
        **kwargs,
    ) -> JudgeResult:
        if not results:
            return JudgeResult(HitType.MISS)
        if checkpoint_id != CheckpointID.CP1:
            # Defensive fallback — config validation rejects always_warm_start on
            # non-CP1 checkpoints, so this branch should be unreachable in
            # validated configs. Keeps behaviour observable if someone bypasses
            # validation (e.g. direct instantiation in tests).
            return JudgeResult(HitType.FULL_HIT, results[0].id)
        return JudgeResult(HitType.WARM_START, results[0].id, start_t=self._start_t)

    def on_episode_start(self) -> None:
        """Clear internal history buffer. Called by Orchestrator at episode start."""

    def record_action(self, action_chunk: torch.Tensor) -> None:
        """Receive Orchestrator-broadcast action. Pure local buffer op."""


class ThresholdJudge:
    """Multi-tier threshold judge: FULL_HIT / WARM_START / MISS.

    Data flow: results[0].score -> compare threshold -> compare warm_tiers -> JudgeResult
    Coupling:
      - DEPENDS ON: score range from CacheStorage backend (see SimilarityJudge docstring)
      - IF backend or key builder changes: threshold value likely needs recalibration
    """

    def __init__(
        self,
        cp1_threshold: float = 0.98,
        cp3_threshold: float = 0.95,
        warm_tiers: list[dict[str, float]] | None = None,
        cp2_threshold: float | None = None,
    ) -> None:
        # CP2 (post-backbone single-key arm) shares CP1's FULL threshold unless
        # given its own; the factory always passes it explicitly.
        self._thresholds = {
            CheckpointID.CP1: cp1_threshold,
            CheckpointID.CP2: cp1_threshold if cp2_threshold is None else cp2_threshold,
            CheckpointID.CP3: cp3_threshold,
        }
        self._warm_tiers = warm_tiers or []

    def __call__(
        self,
        results: list[SearchResultLite],
        checkpoint_id: CheckpointID,
        cached_data: dict[str, torch.Tensor],
        **kwargs,
    ) -> JudgeResult:
        if not results:
            return JudgeResult(HitType.MISS)
        top = results[0]
        threshold = self._thresholds.get(checkpoint_id, 0.98)
        if top.score >= threshold:
            return JudgeResult(HitType.FULL_HIT, top.id)
        if checkpoint_id in (CheckpointID.CP1, CheckpointID.CP2) and self._warm_tiers:
            for tier in self._warm_tiers:
                if top.score >= tier["threshold"]:
                    return JudgeResult(HitType.WARM_START, top.id, start_t=tier["start_t"])
        return JudgeResult(HitType.MISS)

    def on_episode_start(self) -> None:
        """Clear internal history buffer. Called by Orchestrator at episode start."""

    def record_action(self, action_chunk: torch.Tensor) -> None:
        """Receive Orchestrator-broadcast action. Pure local buffer op."""


class FailureAwareGateJudge:
    """Failure-aware sigmoid gate judge (TRACER M2 / Phase 3 + Phase 5 u_t).

    Three-state gate over a dual-retrieval margin. The gate value is
    ``g = sigmoid(b0 + b1*margin + b2*u_t + b3*delta_pos)`` where ``margin`` and
    ``delta_pos`` come from the ``retrieval_signals`` side-channel produced by a
    dual-retrieval SearchStrategy (Eq 19-21) and ``u_t`` is an optional
    kinematic-quality descriptor of the positive winner d+ (Phase 5). Decision
    (thresholds are on ``g``; WARM_START is CP1-only):

      - ``g >= threshold``                     -> FULL_HIT(results[0].id)
      - CP1 and ``g >= tier["threshold"]``     -> WARM_START(start_t)
      - otherwise                              -> MISS

    Degenerate default (``u_t_factor=None, b2=0, b3=0, b0=-tau, b1=1,
    threshold=0.5``, no warm tiers) makes ``g = sigmoid(margin - tau)``, so
    ``g >= 0.5 <=> margin >= tau``. With an empty D- (``margin == s_pos ==
    results[0].score``) this is value-equivalent to ``ThresholdJudge(tau)`` — the
    Phase 3 non-regression anchor.

    u_t (Phase 5, activated when ``u_t_factor`` is configured): the gate builds
    one online kinematic Factor (``<descriptor>_online_<channel>``) over a single
    ``(past, future)`` window plus a Layer-1 ``ZScoreNormalization`` from the
    D+-only ``library_stats``, and computes the descriptor over the
    ``[history[-P:], winner, walk_next(F)]`` splice at verdict time. A NaN u_t
    (boundary / fork / short history / chain end) drops the ``b2*u_t`` term for
    that step, degrading the gate to margin-only (identical to
    ``u_t_factor=None``). ``config.validate_cache_config`` requires ``u_t_factor``
    (plus an in_memory backend with ``library_stats``) whenever
    ``gate_betas["b2"] != 0``.

    factor_outputs (opt-in via ``export_factor_outputs``, default False): when
    True every verdict carries the raw per-step signals
    ``{schema, s_pos, s_neg, margin, delta_pos, u_t, g}`` (NaN pre-converted to
    None) for offline diagnostics; when False ``factor_outputs`` stays None so the
    ``__hit_meta__`` wire is byte-identical to Phase 3.
    """

    def __init__(
        self,
        *,
        gate_betas: dict[str, float],
        threshold: float = 0.5,
        warm_tiers: list[dict[str, float]] | None = None,
        u_t_factor: Optional[dict] = None,
        library_stats: Optional["LibraryStats"] = None,
        export_factor_outputs: bool = False,
    ) -> None:
        self._b0 = float(gate_betas.get("b0", 0.0))
        self._b1 = float(gate_betas.get("b1", 0.0))
        # b2 is the kinematic u_t coefficient (Phase 5). It is inert unless a
        # u_t_factor is configured; the validator ties the two together.
        self._b2 = float(gate_betas.get("b2", 0.0))
        self._b3 = float(gate_betas.get("b3", 0.0))
        self._threshold = float(threshold)
        self._warm_tiers = warm_tiers or []
        self._export_factor_outputs = bool(export_factor_outputs)

        # ------------------------------------------------------------------
        # u_t kinematic channel (Phase 5). Lazy-imported so judge.py keeps no
        # hard dependency on the factors layer (mirrors _build_inner_judge's
        # in-function imports). None -> Phase 3 behaviour (no b2*u_t term).
        # ------------------------------------------------------------------
        self._u_t_factor = None
        self._u_t_norm = None
        self._u_t_key: Optional[str] = None
        if u_t_factor is not None:
            from openpi.cache.components.factors import registry
            from openpi.cache.components.factors.normalization.zscore import ZScoreNormalization

            if library_stats is None:
                raise ValueError(
                    "failure_aware_gate u_t_factor requires library_stats "
                    "(the D+-only normalization basis); got None"
                )
            name = f"{u_t_factor['descriptor']}_online_{u_t_factor['channel']}"
            self._u_t_factor = registry.build(
                name, windows=[{"past": int(u_t_factor["past"]), "future": int(u_t_factor["future"])}]
            )
            # Single window -> exactly one descriptor key; cache it so __call__
            # reads the one value without re-deriving the key string.
            (self._u_t_key,) = self._u_t_factor.descriptor_orientations.keys()
            self._u_t_norm = ZScoreNormalization(library_stats)

    def __call__(
        self,
        results: list[SearchResultLite],
        checkpoint_id: CheckpointID,
        cached_data: dict[str, torch.Tensor],
        *,
        view: Optional["PayloadView"] = None,
        history: Optional["HistoryView"] = None,
        retrieval_signals: Optional[RetrievalSignals] = None,
        **kwargs,
    ) -> JudgeResult:
        if not results:
            return JudgeResult(HitType.MISS)
        if retrieval_signals is None:
            # A failure_aware_gate must be paired with a dual-retrieval strategy
            # that supplies signals; config validation enforces the pairing, so
            # reaching here means a misconfiguration — fail loud, do not guess.
            raise ValueError(
                "failure_aware_gate requires retrieval_signals from a "
                "dual-retrieval strategy; got None (check judge<->strategy pairing)"
            )
        m = retrieval_signals.margin
        d = retrieval_signals.delta_pos
        # u_t: kinematic quality of the winner d+ (None when inactive). A NaN
        # value (short history / fork / chain end) drops the term -> margin-only.
        u_t = self._compute_u_t(results, view, history)
        u_term = 0.0 if (u_t is None or math.isnan(u_t)) else self._b2 * u_t
        z = self._b0 + self._b1 * m + u_term + self._b3 * d
        # Numerically stable sigmoid (avoids math.exp overflow for large |z|).
        if z >= 0.0:
            g = 1.0 / (1.0 + math.exp(-z))
        else:
            ez = math.exp(z)
            g = ez / (1.0 + ez)
        winner_id = results[0].id
        fo = self._factor_outputs(retrieval_signals, m, d, u_t, g)
        if g >= self._threshold:
            return JudgeResult(HitType.FULL_HIT, winner_id, composer_score=g, factor_outputs=fo)
        if checkpoint_id == CheckpointID.CP1 and self._warm_tiers:
            for tier in self._warm_tiers:
                if g >= tier["threshold"]:
                    return JudgeResult(
                        HitType.WARM_START, winner_id,
                        start_t=tier["start_t"], composer_score=g, factor_outputs=fo,
                    )
        return JudgeResult(HitType.MISS, composer_score=g, factor_outputs=fo)

    def _compute_u_t(
        self,
        results: list[SearchResultLite],
        view: Optional["PayloadView"],
        history: Optional["HistoryView"],
    ) -> Optional[float]:
        """Extract the winner's single kinematic descriptor value, or None.

        None when u_t is inactive or the verdict context lacks the view/history
        the online factor needs. A NaN return from the factor (boundary / fork /
        short-history window) is passed through and treated by ``__call__`` as
        "drop the b2*u_t term".
        """
        if self._u_t_factor is None:
            return None
        if view is None or history is None:
            # The searched path always injects both; guard so a bare unit call
            # degrades to margin-only instead of crashing.
            return None
        from openpi.cache.components.factors.base import FactorContext

        ctx = FactorContext(
            results=results, view=view, history=history, normalization=self._u_t_norm
        )
        return self._u_t_factor.extract(ctx)[self._u_t_key]

    def _factor_outputs(
        self,
        signals: RetrievalSignals,
        margin: float,
        delta_pos: float,
        u_t: Optional[float],
        g: float,
    ) -> Optional[dict]:
        """Opt-in diagnostic dump of the raw per-step gate signals (or None).

        None unless ``export_factor_outputs`` is set, keeping the Phase 3
        ``__hit_meta__`` wire byte-identical by default. NaN / None u_t both
        serialize to None via ``_nan_to_none``.
        """
        if not self._export_factor_outputs:
            return None
        return {
            "schema": "failure_gate_v1",
            **_nan_to_none({
                "s_pos": signals.s_pos, "s_neg": signals.s_neg,
                "margin": margin, "delta_pos": delta_pos, "u_t": u_t, "g": g,
            }),
        }

    def on_episode_start(self) -> None:
        """No-op; the gate holds no per-episode state (u_t reads live view/history)."""

    def record_action(self, action_chunk: torch.Tensor) -> None:
        """Receive Orchestrator-broadcast action. No-op (gate is stateless)."""


# ---------------------------------------------------------------------------
# Diagnostic factor-output helpers (used by CompositeJudge when
# export_factor_outputs=True). Pre-converts NaN to None so the dict
# round-trips through strict JSON parsers (jq, pandas) without relying on
# the producer's `allow_nan=True`.
# ---------------------------------------------------------------------------


def _nan_to_none(d: dict[str, float]) -> dict[str, Optional[float]]:
    out: dict[str, Optional[float]] = {}
    for k, v in d.items():
        if v is None:
            out[k] = None
            continue
        f = float(v)
        out[k] = None if math.isnan(f) else f
    return out


def _build_factor_outputs(
    raw: dict[str, float],
    norm: dict[str, float],
    *,
    composer_score: Optional[float],
    sentinel: Optional[str],
) -> dict:
    score: Optional[float]
    if composer_score is None:
        score = None
    else:
        s = float(composer_score)
        score = None if math.isnan(s) else s
    return {
        "raw":  _nan_to_none(raw),
        "norm": _nan_to_none(norm),
        "score": score,
        "sentinel": sentinel,
    }


# ---------------------------------------------------------------------------
# B5/B7 refactor — CompositeJudge + DumpingJudge moved out of this file.
# Facade re-exports preserve legacy import paths
#   from openpi.cache.components.judge import CompositeJudge, DumpingJudge
# while the implementations live in dedicated modules.
# ---------------------------------------------------------------------------

from openpi.cache.components.composite_judge import CompositeJudge  # noqa: E402, F401
from openpi.cache.components.dumping_judge import DumpingJudge      # noqa: E402, F401
from openpi.cache.components.surface_judge import SurfaceJudge      # noqa: E402, F401
