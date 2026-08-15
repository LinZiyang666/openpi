"""DumpingJudge — transparent verdict-time factor dumper (B5 refactor).

The new 4-layer DumpingJudge wraps any inner SimilarityJudge and side-
channels per-verdict factor raw values to a JSONL file:

  1. Forward the verdict request to ``inner`` and capture the JudgeResult.
  2. Run an *independent* set of Layer 2 factors (the "dump factors")
     over the same inputs, using a self-owned Layer 1 Normalization
     (configured to match ``inner``'s Normalization for wire comparability).
  3. Append a JSONL row with the dump-factor raw dict + inner verdict
     identity (hit_type / winner / cp1_score / etc.).
  4. Return ``inner``'s JudgeResult unchanged — verdict behaviour is
     byte-identical to the inner judge.

Plan §6.9 (option b):
  - Independent dump factor list lets warmup yamls over-collect across
    all 17 factors while different eval yamls each pick subsets.
  - Self-owned Normalization (a replica of the same config as inner) avoids
    reaching into inner's internal state; z-score is cheap so duplication
    is fine.

Coupling map:
  DEPENDS ON: components/factors/base.py (Factor, FactorContext, HistoryView),
              components/factors/normalization (Layer 1 replica),
              components/judge.py (SimilarityJudge contract).
  CONSUMED BY: cache/config.py (`_build_judge` wraps when yaml has `dump`).
"""

from __future__ import annotations

import json
import logging
import math
import os
from typing import TYPE_CHECKING, Optional

import torch

from openpi.cache.components.factors.base import Factor, FactorContext, HistoryView
from openpi.cache.components.judge import JudgeResult
from openpi.cache.storage_types import RetrievalSignals, SearchResultLite
from openpi.cache.types import CheckpointID

if TYPE_CHECKING:
    from openpi.cache.components.factors.normalization.base import Normalization
    from openpi.cache.components.judge import SimilarityJudge
    from openpi.cache.components.payload_view import PayloadView

logger = logging.getLogger("openpi.cache.dumping_judge")


class DumpingJudge:
    """Transparent wrapper that side-channels factor dump rows to JSONL.

    Construction:
        DumpingJudge(
            inner=<SimilarityJudge>,
            dump_normalization=<Normalization>,    # self-owned Layer 1 replica
            dump_factors=[<Factor>, ...],          # Layer 2 subset (any combination of the 17 factors)
            dump_path="/tmp/<config_id>.jsonl",
            config_id="<yaml stem>",
        )

    The wrapper forwards every SimilarityJudge surface to ``inner`` via
    ``__getattr__`` (so attributes like ``min_required_top_k`` propagate),
    while augmenting it with the dump-factor required_top_k contribution.
    """

    def __init__(
        self,
        inner: "SimilarityJudge",
        dump_normalization: "Normalization",
        dump_factors: list[Factor],
        dump_path: str,
        config_id: str,
    ) -> None:
        # Use object.__setattr__ so __getattr__ never recurses while __init__
        # is still running and self.__dict__ is incomplete.
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_dump_normalization", dump_normalization)
        object.__setattr__(self, "_dump_factors", list(dump_factors))
        object.__setattr__(self, "_dump_path", str(dump_path))
        object.__setattr__(self, "_config_id", str(config_id))
        object.__setattr__(self, "_current_extra", {})
        object.__setattr__(self, "_step_idx", 0)
        # X14: probe the inner judge ONCE for the query_keys seam. Strict mode
        # (explicit parameter only) — a legacy inner with **kwargs would merely
        # swallow the kwarg, so withholding it keeps every dump-wrapped legacy /
        # composite config on a byte-identical inner call, while a dump-wrapped
        # MlpRouterJudge (which declares the parameter) still receives it.
        from openpi.cache.components.judge import judge_accepts_query_keys

        object.__setattr__(
            self, "_forward_query_keys",
            judge_accepts_query_keys(inner, allow_var_keyword=False),
        )
        # Lazy file handle: opened on first verdict and reused; closed by GC
        # at server shutdown. Avoids 50w open/close/append per phase2 run
        # (plan §6.9 / G1 R1 P3 critique).
        object.__setattr__(self, "_fh", None)

    # ------------------------------------------------------------------
    # Surface preserved from the inner judge
    # ------------------------------------------------------------------

    @property
    def min_required_top_k(self) -> int:
        """Combine inner hint with dump-factor top-k requirement."""
        inner_hint = int(getattr(self._inner, "min_required_top_k", 0))
        dump_hint = max(
            (int(getattr(f, "required_top_k", 0)) for f in self._dump_factors),
            default=0,
        )
        return max(inner_hint, dump_hint)

    def __getattr__(self, name: str):
        # __getattr__ runs only on normal-attribute miss; the stash in
        # __init__ keeps this from recursing.
        return getattr(self._inner, name)

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------

    def on_episode_start(self, *, extra_metadata: Optional[dict] = None) -> None:
        """Stash identity metadata + reset per-episode counters; forward to inner."""
        object.__setattr__(self, "_current_extra", dict(extra_metadata or {}))
        object.__setattr__(self, "_step_idx", 0)
        # Filtered forward — inner judge may not accept extra_metadata.
        from openpi.cache.orchestrator import CacheOrchestrator
        CacheOrchestrator._safe_call_lifecycle(
            self._inner, "on_episode_start", extra_metadata=extra_metadata,
        )

    def record_action(self, action_chunk: torch.Tensor) -> None:
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
        history: Optional[HistoryView] = None,
        retrieval_signals: Optional[RetrievalSignals] = None,
        query_keys: Optional[dict[str, torch.Tensor]] = None,
    ) -> JudgeResult:
        # 1) Forward verdict to inner — verdict behaviour byte-identical. The
        #    Orchestrator injects retrieval_signals unconditionally (TRACER M2 /
        #    Phase 3 seam); forward it so an inner failure_aware_gate still
        #    receives it. The dump-factor path below does not use it.
        #    ``query_keys`` (X14) is declared here so the Orchestrator's probe
        #    admits the wrapper, but it is relayed only to an inner judge that
        #    asked for it — see the constructor's strict probe.
        extra_kwargs = {"query_keys": query_keys} if self._forward_query_keys else {}
        judge_result = self._inner(
            results, checkpoint_id, cached_data,
            view=view, history=history, retrieval_signals=retrieval_signals,
            **extra_kwargs,
        )

        # 2) Best-effort dump. Any extractor failure must not corrupt the
        #    verdict path; trap exceptions and log instead of swallowing
        #    silently (plan §6.9 G1 R1 P3 critique).
        try:
            self._write_dump_row(results, view, history, judge_result)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "DumpingJudge step=%d config_id=%s: failed to write dump row: %r",
                self._step_idx, self._config_id, exc,
            )
        finally:
            object.__setattr__(self, "_step_idx", self._step_idx + 1)
        return judge_result

    # ------------------------------------------------------------------
    # Internal: JSONL row assembly + persistent file handle
    # ------------------------------------------------------------------

    def _write_dump_row(
        self,
        results: list[SearchResultLite],
        view: Optional["PayloadView"],
        history: Optional[HistoryView],
        judge_result: JudgeResult,
    ) -> None:
        # Run the dump-factor list with a self-owned FactorContext.
        ctx = FactorContext(
            results=results,
            view=view,
            history=history if history is not None else HistoryView(actions=[], states=[]),
            normalization=self._dump_normalization,
        )
        factor_raw: dict[str, float] = {}
        factor_nan: dict[str, bool] = {}
        for f in self._dump_factors:
            try:
                out = f.extract(ctx)
            except Exception as exc:  # noqa: BLE001
                # Per-factor isolation: declared keys go to NaN, log the cause.
                logger.warning(
                    "DumpingJudge step=%d factor=%s extract raised: %r",
                    self._step_idx, type(f).__name__, exc,
                )
                declared = getattr(f, "descriptor_orientations", {}) or {}
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
            # Inner main-path diagnostic snapshot (plan §6.10 schema_version=2).
            "inner_hit_type": judge_result.hit_type.name,
            "inner_start_t": judge_result.start_t,
            "inner_factor_outputs": getattr(judge_result, "factor_outputs", None),
        }
        self._append_jsonl(row)

    def _append_jsonl(self, row: dict) -> None:
        """Persistent line-buffered append to the dump file.

        Lazy-opens on first call. Survives across verdicts; GC closes at
        server shutdown. Avoids 50w open/close/append per phase2 sweep.
        """
        if self._fh is None:
            # Ensure parent directory exists once.
            parent = os.path.dirname(self._dump_path)
            if parent and not os.path.isdir(parent):
                os.makedirs(parent, exist_ok=True)
            object.__setattr__(self, "_fh", open(self._dump_path, "a", encoding="utf-8", buffering=1))
        self._fh.write(json.dumps(row) + "\n")
