"""KinematicSearchStrategy — minimal subclass of WeightSearchStrategy.

Inherits all lifecycle hooks (``plan`` / ``on_stage_begin`` /
``on_stage_complete`` / etc.) unchanged. The only addition is a
``_write_per_step`` method that ``run_phase2.py`` injects into
``ConductorDriver(per_step_writer=...)``. Per-step JSONL flushing is
driven entirely by ``ConductorDriver._complete_stage`` itself — see
``logs/weighted_sum_kinematic_phase5_replication.log.md`` §4.1 (G1 R2
B1 mandate: do NOT change the ``on_stage_complete`` hook signature
because 5 existing concrete overrides take 3 args).

The kinematic-on-weighted_sum eval pipeline uses
``samples_source.type=offline`` in every eval yaml (see
``spec.build_eval_yaml_for_cell``), so no ``preload_normalizer_buffer``
call is needed in ``on_stage_begin``: the server reads the super warmup
raw jsonl from disk at ``_bind_bundle`` time. The base
``WeightSearchStrategy.on_stage_begin`` (load_cache_config only) is
therefore exactly what we need; we do not override it.

Coupling map:
  DEPENDS ON:  exp.weighted_sum.weight_search_strategy.WeightSearchStrategy
  CONSUMED BY: exp.weighted_sum.run_phase2 (constructed when --strategy=kinematic),
               openpi.conductor.driver.ConductorDriver (receives
                                                       ``_write_per_step``
                                                       as ``per_step_writer``).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from exp.weighted_sum.weight_search_strategy import WeightSearchStrategy


logger = logging.getLogger(__name__)


class KinematicSearchStrategy(WeightSearchStrategy):
    """WeightSearchStrategy + per-yaml per_step writer for crash-survival.

    The strategy class itself owns only the file-format / path decision
    for per_step output. The actual flush trigger is in
    ``ConductorDriver._complete_stage`` (driver-internal call to
    ``_flush_per_step_for_stage``), so the strategy hook signature is
    NOT modified — preserving backward compatibility with the 5 existing
    ``on_stage_complete(self, stage, ctl, ctx)`` overrides in
    ``warmup_eval_strategy.py:169`` and ``tests/conductor/...``.

    Usage from run_phase2.py:

        strategy = KinematicSearchStrategy(..., per_step_out_dir=Path(...))
        driver = ConductorDriver(
            ...,
            strategy=strategy,
            per_step_writer=strategy._write_per_step,  # bound method
        )
    """

    def __init__(
        self,
        *args,
        per_step_out_dir: Path | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._per_step_out_dir = per_step_out_dir
        if per_step_out_dir is not None:
            per_step_out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Writer callable — injected into ConductorDriver(per_step_writer=...)
    # ------------------------------------------------------------------

    def _write_per_step(self, yaml_id: str, rows: list[dict]) -> None:
        """Per-yaml JSONL writer. Driver calls us when a stage completes.

        Intentionally idempotent in the trivial sense: writes the rows
        atomically by ``open(mode="w")`` truncating any prior partial
        file (the driver only calls us once per yaml at stage
        completion; mid-stage flushes are NOT issued).

        If ``per_step_out_dir`` is None this is a no-op — supports a
        configuration where per-step capture is disabled at the
        run_phase2 level (e.g. dry runs).
        """
        if self._per_step_out_dir is None:
            return
        out = self._per_step_out_dir / f"{yaml_id}.jsonl"
        with out.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
        logger.info(
            "[kinematic] flushed per_step for %s: %d rows -> %s",
            yaml_id,
            len(rows),
            out,
        )


__all__ = ["KinematicSearchStrategy"]
