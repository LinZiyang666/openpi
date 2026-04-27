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

import math
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Optional, Protocol, runtime_checkable

import torch

from openpi.cache.storage_types import SearchResultLite
from openpi.cache.types import CANONICAL_DENOISE_TIMESTEPS, CheckpointID

if TYPE_CHECKING:
    from openpi.cache.components.factors.base import (
        HistoryView,
        OnlineExtractor,
    )
    from openpi.cache.components.factors.composers import Composer
    from openpi.cache.components.factors.normalizers import Normalizer
    from openpi.cache.components.payload_view import PayloadView


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

    FULL_HIT and WARM_START must include winner_id; Orchestrator skips
    fetch when winner_id is None.
    """

    hit_type: HitType
    winner_id: str | None = None
    start_t: float | None = None


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

    The `view` and `history` keyword-only parameters carry verdict-time
    facade objects injected by Orchestrator. Existing judges that do not
    need them accept and ignore both via `**kwargs`; Orchestrator only
    builds and injects the facades from B1 onward (CompositeJudge land).
    """

    def __call__(
        self,
        results: list[SearchResultLite],
        checkpoint_id: CheckpointID,
        cached_data: dict[str, torch.Tensor],
        *,
        view: Optional["PayloadView"] = None,
        history: Optional["HistoryView"] = None,
    ) -> JudgeResult:
        """Judge the top search results.

        Args:
            results: Search results sorted by descending score (from CacheStorage).
            checkpoint_id: CP1 or CP3.
            cached_data: Raw tensors from KeyBuilder.cached_data.
            view: Read-only facade over CacheStorage (B1+ injection;
                None for the B0 path and for legacy judges).
            history: Per-episode action / state snapshot (B1+ injection).

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
    ) -> None:
        self._thresholds = {
            CheckpointID.CP1: cp1_threshold,
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
        if checkpoint_id == CheckpointID.CP1 and self._warm_tiers:
            for tier in self._warm_tiers:
                if top.score >= tier["threshold"]:
                    return JudgeResult(HitType.WARM_START, top.id, start_t=tier["start_t"])
        return JudgeResult(HitType.MISS)

    def on_episode_start(self) -> None:
        """Clear internal history buffer. Called by Orchestrator at episode start."""

    def record_action(self, action_chunk: torch.Tensor) -> None:
        """Receive Orchestrator-broadcast action. Pure local buffer op."""


# ---------------------------------------------------------------------------
# CompositeJudge — verdict factor pipeline
# ---------------------------------------------------------------------------


class CompositeJudge:
    """Aggregate multiple verdict-factor extractors into a hit decision.

    Pipeline:
      1. Each `OnlineExtractor` returns a `dict[str, float]` — the keys
         it declared up front via `descriptor_orientations`.
      2. CompositeJudge unions all keys and asserts each extractor's
         runtime output matches its declared key set (key contract).
      3. Optional `Normalizer` maps raw values into a normalized space
         (typically [0, 1] percentile rank). When the normalizer is in
         cold-start `force_miss` mode and returns all-NaN, CompositeJudge
         short-circuits to MISS without invoking the Composer.
      4. `Composer` aggregates the normalized dict into a `JudgeResult`.

    The B0 ship is a structural skeleton: extractors / composer /
    normalizer all raise NotImplementedError in their algorithm bodies,
    and `_JUDGE_TYPES` excludes "composite" so user YAMLs are rejected at
    config-load time — preventing first-verdict NotImplementedError. B1
    flips the gate and lands the algorithm bodies.

    Coupling map:
      DEPENDS ON: components/factors/base.py (OnlineExtractor),
                  components/factors/composers (Composer),
                  components/factors/normalizers (Normalizer).
      CONSUMED BY: CacheOrchestrator.check (B1+ injection of view + history).
    """

    def __init__(
        self,
        extractors: list["OnlineExtractor"],
        composer: "Composer",
        normalizer: Optional["Normalizer"] = None,
    ) -> None:
        if not extractors:
            raise ValueError(
                "CompositeJudge requires at least one OnlineExtractor"
            )
        self._extractors = list(extractors)
        self._composer = composer
        self._normalizer = normalizer
        self.min_required_top_k = max(
            (getattr(e, "required_top_k", 0) for e in self._extractors),
            default=0,
        )

        # Collect static metadata: union of extractor descriptor_orientations.
        # A key claimed by two extractors with conflicting orientations is a
        # config bug — surface it loudly at construction time, not silently
        # at the first verdict.
        all_orientations: dict[str, str] = {}
        for ext in self._extractors:
            ext_orientations = getattr(ext, "descriptor_orientations", None)
            if ext_orientations is None:
                raise ValueError(
                    f"Extractor {type(ext).__name__} is missing "
                    "descriptor_orientations attribute"
                )
            for k, ori in ext_orientations.items():
                existing = all_orientations.get(k)
                if existing is not None and existing != ori:
                    raise ValueError(
                        f"Composite judge factor key {k!r} has conflicting "
                        f"orientations: {existing!r} vs {ori!r}"
                    )
                all_orientations[k] = ori
        self._all_orientations = all_orientations

        # Bind metadata into composer / normalizer. Composer cross-checks
        # `directions` config against non_monotonic keys (B1+); normalizer
        # uses the key list to pre-allocate per-key state.
        composer.bind_orientations(dict(all_orientations))
        if normalizer is not None:
            normalizer.bind_keys(list(all_orientations.keys()))

    def __call__(
        self,
        results: list[SearchResultLite],
        checkpoint_id: CheckpointID,
        cached_data: dict[str, torch.Tensor],
        *,
        view: Optional["PayloadView"] = None,
        history: Optional["HistoryView"] = None,
    ) -> JudgeResult:
        if not results:
            return JudgeResult(HitType.MISS)

        raw: dict[str, float] = {}
        for ext in self._extractors:
            out = ext.extract(results, view, history, cached_data)
            # Key contract enforcement: an extractor's runtime keys MUST
            # equal its declared `descriptor_orientations.keys()`. Drift
            # would silently desync Normalizer state, Composer weights,
            # and orientation lookup, so we fail loud at the boundary.
            expected = set(ext.descriptor_orientations.keys())
            actual = set(out.keys())
            if actual != expected:
                raise RuntimeError(
                    f"Extractor {type(ext).__name__} key contract violation: "
                    f"declared {sorted(expected)}, returned {sorted(actual)} "
                    f"(missing: {sorted(expected - actual)}, "
                    f"extra: {sorted(actual - expected)})"
                )
            raw.update(out)

        norm = self._normalizer(raw) if self._normalizer is not None else raw

        # Cold-start sentinel: all-NaN normalized output means the
        # normalizer signaled "not enough samples / force_miss". Short-
        # circuit to MISS rather than letting each Composer reinvent the
        # all-NaN handling rule.
        if norm and all(math.isnan(v) for v in norm.values()):
            return JudgeResult(HitType.MISS)

        return self._composer.compose(norm, winner_id=results[0].id)

    def on_episode_start(self) -> None:
        if self._normalizer is not None:
            self._normalizer.on_episode_start()

    def record_action(self, action_chunk: torch.Tensor) -> None:
        # CompositeJudge holds no per-episode action buffer of its own;
        # action / state history is owned by Orchestrator and injected
        # via HistoryView at verdict time.
        return None


# ---------------------------------------------------------------------------
# DumpingJudge — transparent wrapper that side-channels per-verdict factor
# logs to JSONL for offline calibration without altering verdict behaviour
# ---------------------------------------------------------------------------


class DumpingJudge:
    """Transparent wrapper that logs per-verdict factor descriptors to JSONL.

    Pipeline per verdict:
      1. Forward `(results, checkpoint_id, cached_data, view, history)` to the
         inner judge and capture its `JudgeResult`.
      2. Run the configured dump-side `OnlineExtractor` list on the same args
         to collect raw factor values + a per-key NaN flag map.
      3. Append one JSONL row containing `config_id, task_id, orig_init_state_idx,
         step_idx, winner_id, cp1_score, factor_raw, factor_nan` to `dump_path`.
      4. Return the inner `JudgeResult` unchanged — verdict behaviour is
         byte-identical to the inner judge.

    Identity (`task_id`, `orig_init_state_idx`) reaches the wrapper through
    `on_episode_start(extra_metadata=...)`, which the orchestrator broadcasts
    via `_safe_call_lifecycle` (signature filtering keeps existing kwargs-free
    judges unaffected).

    Surface contract preserved from the inner judge:
      - `min_required_top_k` is `max(inner_hint, max(dump-extractor required_top_k))`,
        so cache search returns enough candidates for both the inner verdict
        and dump-side factors (e.g. F2 needs K=5).
      - `on_episode_start` and `record_action` forward to the inner judge —
        the former through filtered dispatch (inner may not accept
        `extra_metadata`), the latter directly (4 existing judges share the
        same `(self, action_chunk)` signature).
      - Any other attribute access falls through to the inner judge via
        `__getattr__`, so future surface additions work without re-wrapping.

    Coupling map:
      DEPENDS ON:  any SimilarityJudge instance + a list of OnlineExtractor.
      CONSUMED BY: `_build_judge` when the YAML carries a `judge.dump` block.
      IF CHANGED:  recheck `Orchestrator._safe_call_lifecycle` filtering and
                   the JSONL column names referenced by Phase 5 calibration.
    """

    def __init__(
        self,
        inner: "SimilarityJudge",
        dump_extractors: list,
        dump_path: str,
        config_id: str,
    ) -> None:
        # Set the four attributes via object.__setattr__ so __getattr__ never
        # has a chance to recurse into the inner judge while __init__ is still
        # running and the instance dict is incomplete.
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_dump_extractors", list(dump_extractors))
        object.__setattr__(self, "_dump_path", str(dump_path))
        object.__setattr__(self, "_config_id", str(config_id))
        object.__setattr__(self, "_current_extra", {})
        object.__setattr__(self, "_step_idx", 0)

    # ------------------------------------------------------------------
    # Surface preserved from the inner judge
    # ------------------------------------------------------------------

    @property
    def min_required_top_k(self) -> int:
        """Combine inner hint with dump-side extractor top-k requirements."""
        inner_hint = int(getattr(self._inner, "min_required_top_k", 0))
        dump_hint = max(
            (int(getattr(ext, "required_top_k", 0)) for ext in self._dump_extractors),
            default=0,
        )
        return max(inner_hint, dump_hint)

    def __getattr__(self, name: str):
        # __getattr__ runs only on normal-attribute miss, so the
        # object.__setattr__ stash in __init__ keeps this from recursing.
        return getattr(self._inner, name)

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------

    def on_episode_start(self, *, extra_metadata: Optional[dict] = None) -> None:
        """Stash identity metadata + reset per-episode counters; forward to inner."""
        self._current_extra = dict(extra_metadata or {})
        self._step_idx = 0
        # Filtered forwarding — inner judge may not accept `extra_metadata`,
        # so use the same `inspect.signature` filtering as Orchestrator.
        # Local import avoids module-load cycle (orchestrator imports judge).
        from openpi.cache.orchestrator import CacheOrchestrator
        CacheOrchestrator._safe_call_lifecycle(
            self._inner, "on_episode_start", extra_metadata=extra_metadata,
        )

    def record_action(self, action_chunk: torch.Tensor) -> None:
        """Forward action broadcast to the inner judge.

        All four existing judges share `record_action(self, action_chunk)`,
        so a direct call is safe; wrap behind `_safe_call_lifecycle` only
        when the inner-judge surface widens here.
        """
        hook = getattr(self._inner, "record_action", None)
        if hook is not None:
            hook(action_chunk)

    # ------------------------------------------------------------------
    # Verdict path
    # ------------------------------------------------------------------

    def __call__(
        self,
        results: list[SearchResultLite],
        checkpoint_id: CheckpointID,
        cached_data: dict[str, torch.Tensor],
        *,
        view: Optional["PayloadView"] = None,
        history: Optional["HistoryView"] = None,
    ) -> JudgeResult:
        # 1) Forward verdict to inner — verdict behaviour byte-identical.
        judge_result = self._inner(
            results, checkpoint_id, cached_data, view=view, history=history,
        )

        # 2) Best-effort dump. Any extractor failure must not pollute the
        # verdict path; trap exceptions and write NaN rows instead.
        try:
            self._write_dump_row(results, view, history, cached_data)
        except Exception:  # noqa: BLE001
            # Log path failure is recoverable from the verdict's perspective;
            # surfacing here would corrupt experiment runs for an analysis
            # side-channel. Swallowing here is the documented contract.
            pass
        finally:
            self._step_idx += 1

        return judge_result

    # ------------------------------------------------------------------
    # Internal: JSONL row assembly
    # ------------------------------------------------------------------

    def _write_dump_row(
        self,
        results,
        view,
        history,
        cached_data,
    ) -> None:
        factor_raw: dict[str, float] = {}
        factor_nan: dict[str, bool] = {}
        for ext in self._dump_extractors:
            try:
                out = ext.extract(results, view, history, cached_data)
            except Exception:  # noqa: BLE001
                # Per-extractor isolation: declared keys go to NaN.
                declared = getattr(ext, "descriptor_orientations", {}) or {}
                for k in declared:
                    factor_raw[k] = float("nan")
                    factor_nan[k] = True
                continue
            for k, v in out.items():
                fv = float(v)
                factor_raw[k] = fv
                factor_nan[k] = math.isnan(fv)

        winner = results[0] if results else None
        row = {
            "config_id": self._config_id,
            "task_id": self._current_extra.get("task_id"),
            "orig_init_state_idx": self._current_extra.get("orig_init_state_idx"),
            "step_idx": self._step_idx,
            "winner_id": winner.id if winner is not None else None,
            "cp1_score": float(winner.score) if winner is not None else None,
            "factor_raw": factor_raw,
            "factor_nan": factor_nan,
        }

        # Append-only write. JSONL by line.
        import json
        with open(self._dump_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
