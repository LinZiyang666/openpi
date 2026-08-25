"""RiskRouterJudge — X15 verdict-layer risk gate (``judge.type: "risk_router"``).

Where X14's ``mlp_router`` samples an arm from the raw query keys and is
deliberately blind to the library, this judge does the opposite: it reads the
retrieval evidence the search just produced and asks a small calibrated model
one question — *how risky is replaying the cache at this step?* A scalar risk
crossing a threshold routes to teacher; otherwise the cached chunk replays.

Three properties matter and are enforced here rather than left to config:

* **Fail-safe.** Any missing input (no results, absent diagnostics, a feature
  that comes out non-finite) yields maximum risk, i.e. teacher. Degrading to
  the expensive-but-correct arm is the only safe direction for a gate whose
  whole purpose is catching cache failures.
* **Dwell.** Once teacher is chosen the gate holds it for ``dwell`` decisions.
  Cache failure is drift-shaped, so single-step flapping both wastes teacher
  budget and leaves the recovery half-finished.
* **No history in the A-tier feature set.** ``history`` is accepted (the
  Orchestrator injects it unconditionally) and never read, matching the frozen
  paper scope; the only temporal inputs are the scalar step fractions, which
  are part of the Markov state.

Key dependencies: ``StepRetrievalFeatures`` (openpi.cache.storage_types),
``PayloadView`` for neighbour payloads/entries, and the risk model artifact
written by ``exp/rl_router/train_risk_model.py``.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Optional

import torch

from openpi.cache.components.judge import HitType, JudgeResult
from openpi.cache.storage_types import SearchResultLite, StepRetrievalFeatures
from openpi.cache.types import CheckpointID

logger = logging.getLogger("openpi.cache.risk_router")

# Risk assigned when the gate cannot form a trustworthy opinion. Any finite
# threshold routes this to teacher.
MAX_RISK = 1.0


# ------------------------------------------------------------------
# Judge
# ------------------------------------------------------------------


class RiskRouterJudge:
    """Threshold a calibrated cache-risk score to pick teacher vs cache.

    Construction validates the artifact against the runtime feature builder:
    a checkpoint trained on a different feature schema is refused at yaml load
    rather than silently producing garbage risk mid-episode.
    """

    def __init__(
        self,
        *,
        feature_builder,
        risk_model,
        tau: float,
        dwell: int = 1,
        dump_dir: str = "",
    ) -> None:
        if dwell < 1:
            raise ValueError(f"risk_router: dwell must be >= 1, got {dwell}")
        self._features = feature_builder
        self._model = risk_model
        self._tau = float(tau)
        self._dwell = int(dwell)
        self._dump_dir = dump_dir

        # Per-episode state. ``_dwell_left`` is the executor-side policy state
        # the plan keeps out of the feature vector on purpose.
        self._decision_idx = 0
        self._dwell_left = 0
        self._fallback_count = 0

    # -- lifecycle ----------------------------------------------------

    def on_episode_start(self, *args, **kwargs) -> None:
        """Reset per-episode counters.

        This is the name the Orchestrator broadcasts; without it ``decision_idx``
        and the dwell countdown leak across episodes, which corrupts both the
        step-fraction feature and the first decisions of every episode after the
        first. Extra arguments are tolerated so a broadcast that grows a
        parameter does not break the judge.
        """
        self._decision_idx = 0
        self._dwell_left = 0

    # Alias kept for direct callers and tests.
    reset_episode = on_episode_start

    @property
    def fallback_count(self) -> int:
        """Number of decisions that fell back to teacher on a degraded input."""
        return self._fallback_count

    # -- verdict ------------------------------------------------------

    def __call__(
        self,
        results: list[SearchResultLite],
        checkpoint_id: CheckpointID,
        cached_data: dict[str, torch.Tensor],
        *,
        view=None,
        history=None,
        retrieval_signals=None,
        step_features: Optional[StepRetrievalFeatures] = None,
        query_keys: Optional[dict[str, torch.Tensor]] = None,
    ) -> JudgeResult:
        """Score this step's retrieval evidence and route one arm.

        The signature must accept every keyword ``CacheOrchestrator.check()``
        passes unconditionally — ``view`` / ``history`` / ``retrieval_signals``
        — or the first verdict raises TypeError. ``history`` and
        ``retrieval_signals`` are accepted and never read: the A-tier scope
        excludes trajectory history, and ``retrieval_signals`` is the
        dual-retrieval bundle that only the failure-aware strategy produces
        (always None on this arm pair, its content already covered by
        ``step_features``). ``cached_data`` is the query-side tensor bundle
        from the key builder, not the neighbour's content; neighbour payloads
        and entries come through ``view``.
        """
        # A-tier scope: query-side bundle, trajectory history and the
        # dual-retrieval signals are all deliberately unread.
        del cached_data, history, retrieval_signals

        decision_idx = self._decision_idx
        self._decision_idx += 1

        # Dwell holds teacher without re-scoring: the gate already decided the
        # trajectory needs the teacher for the next few decisions.
        top1 = results[0].id if results else None
        if self._dwell_left > 0:
            self._dwell_left -= 1
            return self._teacher(
                decision_idx, risk=None, reason="dwell", candidate_id=top1,
            )

        risk, degraded = self._risk(results, step_features, query_keys, view, decision_idx)
        if degraded:
            self._fallback_count += 1
            self._dwell_left = self._dwell - 1
            return self._teacher(
                decision_idx, risk=risk, reason="fail_safe", candidate_id=top1,
            )

        if risk >= self._tau:
            self._dwell_left = self._dwell - 1
            return self._teacher(
                decision_idx, risk=risk, reason="risk", candidate_id=top1,
            )

        return JudgeResult(
            hit_type=HitType.FULL_HIT,
            winner_id=results[0].id,
            hit_override=False,          # force cached replay over any hit_executor
            router_outputs={
                "decision_idx": decision_idx,
                "arm_sampled": "cache",
                "arm_executed": None,    # stamped by the Interceptor
                "risk": risk,
                "tau": self._tau,
                "reason": "risk",
            },
        )

    # -- internals ----------------------------------------------------

    def _risk(
        self,
        results: list[SearchResultLite],
        step_features: Optional[StepRetrievalFeatures],
        query_keys: Optional[dict[str, torch.Tensor]],
        view: Any,
        decision_idx: int,
    ) -> tuple[float, bool]:
        """Return ``(risk, degraded)``; ``degraded`` forces the fail-safe path."""
        if not results or step_features is None or query_keys is None:
            logger.debug(
                "risk_router: degraded input at decision %d "
                "(results=%d, features=%s, query_keys=%s)",
                decision_idx, len(results), step_features is not None,
                query_keys is not None,
            )
            return MAX_RISK, True
        try:
            x = self._features.build(
                results=results,
                step_features=step_features,
                query_keys=query_keys,
                view=view,
                decision_idx=decision_idx,
            )
            if not torch.isfinite(x).all():
                return MAX_RISK, True
            risk = float(self._model.risk(x))
        except Exception:  # noqa: BLE001 - any builder/model fault must fail safe
            logger.exception("risk_router: risk computation failed; routing to teacher")
            return MAX_RISK, True
        if not math.isfinite(risk):
            return MAX_RISK, True
        return risk, False

    def _teacher(
        self,
        decision_idx: int,
        *,
        risk: Optional[float],
        reason: str,
        candidate_id: Optional[str] = None,
    ) -> JudgeResult:
        # A MISS still carries the retrieval top-1. The verdict does not use it
        # — teacher executes either way — but the X15 shadow labeller needs to
        # know which cached chunk was passed over, and it was already retrieved,
        # so surfacing it costs nothing. Orchestrator relays it as
        # ``CheckResult.entry_id``.
        return JudgeResult(
            hit_type=HitType.MISS,
            winner_id=candidate_id,
            router_outputs={
                "decision_idx": decision_idx,
                "arm_sampled": "teacher",
                "arm_executed": None,
                "risk": risk,
                "tau": self._tau,
                "reason": reason,
            },
        )
