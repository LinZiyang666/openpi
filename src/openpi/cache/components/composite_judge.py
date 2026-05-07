"""CompositeJudge — 4-layer assembly shell (B5 of the verdict-factor refactor).

Pipeline (per verdict):
  1. Empty results → JudgeResult(MISS) early-return (plan §6.11 #5).
  2. Build a FactorContext and run every Layer 2 factor's ``extract``;
     enforce the per-factor key contract (declared keys == returned keys).
  3. Layer 3 calibrates the raw dict → calibrated dict (NaN propagates).
  4. Layer 4 composer maps calibrated dict → JudgeResult.
  5. Optional ``export_factor_outputs`` attaches a diagnostic
     ``factor_outputs`` payload (raw, calibrated, composer_score,
     schema_version=2) to the result for wire transport.

CompositeJudge does NOT consume Layer 1 directly — it injects the
Normalization instance into ``FactorContext.normalization`` so each
factor can call ``normalize_action`` / ``normalize_state`` on demand.

Coupling map:
  DEPENDS ON: components/factors/base.py (Factor + FactorContext + HistoryView),
              components/factors/normalization (Layer 1),
              components/factors/calibrations (Layer 3),
              components/factors/composers (Layer 4),
              components/judge.py (HitType, JudgeResult, SimilarityJudge),
              components/payload_view.py (PayloadView).
  CONSUMED BY: cache/config.py (`_build_judge`),
               cache/orchestrator.py (verdict path).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional

import torch

from openpi.cache.components.factors._debug import VERDICT_DEBUG, verdict_logger
from openpi.cache.components.factors.base import Factor, FactorContext, HistoryView
from openpi.cache.components.judge import (
    HitType,
    JudgeResult,
    _build_factor_outputs,
)
from openpi.cache.storage_types import SearchResultLite
from openpi.cache.types import CheckpointID

if TYPE_CHECKING:
    from openpi.cache.components.factors.calibrations.base import Calibration
    from openpi.cache.components.factors.composers.base import Composer
    from openpi.cache.components.factors.normalization.base import Normalization
    from openpi.cache.components.payload_view import PayloadView


class CompositeJudge:
    """4-layer assembly: Normalization → Factors → Calibration → Composer.

    Construction performs the static checks once:
      - Union of declared factor orientations is consistent (no key with
        two conflicting orientations).
      - Composer's ``declared_dependencies`` is a subset of the union of
        factor keys (else: missing dependency → fail-fast).
      - Calibration ``bind_keys`` is invoked with the union of factor
        keys; subclasses use this to fail-fast on undersized samples.
      - Composer ``bind_orientations`` receives the union map; subclasses
        cross-check that any non_monotonic key has a configured direction.

    After construction, ``min_required_top_k`` is exposed (max of every
    factor's ``required_top_k``) so the search-strategy ``min_top_k_hint``
    can lift fetch breadth.
    """

    def __init__(
        self,
        normalization: "Normalization",
        factors: list[Factor],
        calibration: "Calibration",
        composer: "Composer",
        *,
        export_factor_outputs: bool = False,
    ) -> None:
        if not factors:
            raise ValueError("CompositeJudge requires at least one Factor")

        # ------------------------------------------------------------------
        # 1. Collect the union of declared orientations + key contract preflight.
        # ------------------------------------------------------------------
        all_orientations: dict[str, str] = {}
        for f in factors:
            decl = getattr(f, "descriptor_orientations", None)
            if decl is None:
                raise ValueError(
                    f"Factor {type(f).__name__} is missing `descriptor_orientations`"
                )
            for k, ori in decl.items():
                existing = all_orientations.get(k)
                if existing is not None and existing != ori:
                    raise ValueError(
                        f"Composite judge factor key {k!r} has conflicting "
                        f"orientations: {existing!r} vs {ori!r}"
                    )
                all_orientations[k] = ori
        all_keys = list(all_orientations.keys())

        # ------------------------------------------------------------------
        # 2. Composer dependency check (plan §11.5 / §13.3 rule 4).
        # ------------------------------------------------------------------
        deps = getattr(composer, "declared_dependencies", None)
        if deps is None:
            raise ValueError(
                f"Composer {type(composer).__name__} is missing "
                "`declared_dependencies` instance attribute"
            )
        missing = set(deps) - set(all_keys)
        if missing:
            raise ValueError(
                f"Composer {type(composer).__name__} requires factor keys not "
                f"produced by the configured Layer 2 factors: {sorted(missing)}"
            )

        # ------------------------------------------------------------------
        # 3. Inject metadata into Layer 3 + Layer 4.
        # ------------------------------------------------------------------
        # Calibration bind_keys runs the startup-time fail-fast (no cold-start).
        calibration.bind_keys(all_keys)
        composer.bind_orientations(dict(all_orientations))

        # ------------------------------------------------------------------
        # 4. Search-strategy hint.
        # ------------------------------------------------------------------
        self.min_required_top_k: int = max(
            (int(getattr(f, "required_top_k", 0)) for f in factors),
            default=0,
        )

        # ------------------------------------------------------------------
        # State
        # ------------------------------------------------------------------
        self._normalization = normalization
        self._factors: list[Factor] = list(factors)
        self._calibration = calibration
        self._composer = composer
        self._export_factor_outputs = bool(export_factor_outputs)
        self._all_orientations = all_orientations
        # Per-instance verdict counter for debug log correlation.
        self._vd_step = 0

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
    ) -> JudgeResult:
        # Plan §6.11 #5: empty results short-circuit MISS, do NOT delegate
        # to composer (avoids WARM_START with winner_id=None edge case).
        if not results:
            return JudgeResult(HitType.MISS)

        ctx = FactorContext(
            results=results,
            view=view,
            history=history if history is not None else HistoryView(actions=[], states=[]),
            normalization=self._normalization,
        )

        raw: dict[str, float] = {}
        per_ext_keys: dict[str, list[str]] = {}
        for f in self._factors:
            out = f.extract(ctx)
            expected = set(f.descriptor_orientations.keys())
            actual = set(out.keys())
            if actual != expected:
                raise RuntimeError(
                    f"Factor {type(f).__name__} key contract violation: "
                    f"declared {sorted(expected)}, returned {sorted(actual)} "
                    f"(missing: {sorted(expected - actual)}, "
                    f"extra: {sorted(actual - expected)})"
                )
            if VERDICT_DEBUG:
                per_ext_keys[type(f).__name__] = sorted(out.keys())
            raw.update(out)

        calibrated = self._calibration(raw)
        result = self._composer.compose(calibrated, winner_id=results[0].id)

        if self._export_factor_outputs:
            # Plan §6.10: schema_version=2 lets new analysis scripts gate on
            # the new key naming; legacy jsonl readers fall back to v1.
            payload = _build_factor_outputs(
                raw, calibrated,
                composer_score=result.composer_score,
                sentinel=None,        # plan §6.5 rule 3: no framework sentinel
            )
            # Refactor schema renames: "norm" -> "calibrated", "score" ->
            # "composer_score", drop "sentinel" (no framework all-NaN concept).
            if "norm" in payload:
                payload["calibrated"] = payload.pop("norm")
            if "score" in payload:
                payload["composer_score"] = payload.pop("score")
            payload.pop("sentinel", None)
            payload["schema_version"] = 2
            result.factor_outputs = payload

        if VERDICT_DEBUG:
            self._vd_step += 1
            self._emit_vd(checkpoint_id, raw, calibrated, result, per_ext_keys=per_ext_keys)
        return result

    # ------------------------------------------------------------------
    # Debug
    # ------------------------------------------------------------------

    def _emit_vd(
        self,
        checkpoint_id: CheckpointID,
        raw: dict[str, float],
        calibrated: dict[str, float],
        result: JudgeResult,
        *,
        per_ext_keys: dict[str, list[str]],
    ) -> None:
        raw_nan_summary = []
        for ext_name, keys in per_ext_keys.items():
            n = len(keys)
            n_nan = sum(1 for k in keys if math.isnan(raw.get(k, float("nan"))))
            raw_nan_summary.append(f"{ext_name}={n_nan}/{n}")
        cal_nan = sum(1 for v in calibrated.values() if math.isnan(v))
        valid = len(calibrated) - cal_nan
        verdict_logger.info(
            "[vd step=%d cp=%s] raw_nan: %s | calibrated valid=%d nan=%d | hit=%s start_t=%s winner=%s",
            self._vd_step, checkpoint_id.name, " ".join(raw_nan_summary),
            valid, cal_nan,
            result.hit_type.name, result.start_t, result.winner_id,
        )

    # ------------------------------------------------------------------
    # Lifecycle (Orchestrator broadcasts these)
    # ------------------------------------------------------------------

    def on_episode_start(self, *, extra_metadata: Optional[dict] = None) -> None:
        # Layer 3 buffers persist across episodes by design (plan §6.11 #9).
        # The hook is forwarded for any subclass that wants to reset state.
        if hasattr(self._calibration, "on_episode_start"):
            self._calibration.on_episode_start()

    def record_action(self, action_chunk: torch.Tensor) -> None:
        # Plan §6.5 rule 6: history is owned by Orchestrator, not the
        # judge; CompositeJudge has no per-episode action buffer.
        return None
