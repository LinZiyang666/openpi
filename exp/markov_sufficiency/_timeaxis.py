"""Environment-step to inference-cycle conversion for per-step rollout logs.

The two sides of E2 live on different time axes: per-step JSONL rows carry the
physical environment step (0, 5, 10, ... for ``replan_steps=5``), while cache
entries are numbered by inference cycle (0, 1, 2, ...). Comparing them directly
mislabels the second half of every episode, so conversion happens here behind a
hard gate.

Public interface: :class:`QuarantineReport`, :func:`to_cycles`.

Key dependency: the row schema written by
``openpi.serving.per_step_recorder``.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Iterable

# ------------------------------------------------------------------
# Report
# ------------------------------------------------------------------


@dataclasses.dataclass
class QuarantineReport:
    """Counts of rows dropped on the way to a clean cycle axis.

    ``episode_summary`` rows are a schema feature, not an anomaly: the recorder
    emits one per episode and they legitimately carry no ``step_idx``. They are
    excluded first and counted separately so they never inflate the anomaly
    counters.
    """

    episode_summary: int = 0
    missing_step_idx: int = 0
    non_divisible: int = 0
    bad_spacing: set[str] = dataclasses.field(default_factory=set)
    non_contiguous: set[str] = dataclasses.field(default_factory=set)
    quarantined_yamls: set[str] = dataclasses.field(default_factory=set)
    kept_rows: int = 0

    def as_dict(self) -> dict[str, Any]:
        """Manifest-friendly view (sets become sorted lists)."""
        return {
            "episode_summary": self.episode_summary,
            "missing_step_idx": self.missing_step_idx,
            "non_divisible": self.non_divisible,
            "bad_spacing": sorted(self.bad_spacing),
            "non_contiguous": sorted(self.non_contiguous),
            "quarantined_yamls": sorted(self.quarantined_yamls),
            "kept_rows": self.kept_rows,
        }


# ------------------------------------------------------------------
# Conversion gate
# ------------------------------------------------------------------

_EPISODE_KEYS = ("yaml_id", "task_id", "subset_init_state_idx", "episode_id", "attempt")


def _episode_key(row: dict[str, Any]) -> tuple:
    return tuple(row.get(k) for k in _EPISODE_KEYS)


def to_cycles(
    rows: Iterable[dict[str, Any]],
    replan_steps: int,
) -> tuple[list[dict[str, Any]], QuarantineReport]:
    """Convert ``step_idx`` (env steps) to ``cycle`` and validate the axis.

    Checks, in order: drop ``_kind == "episode_summary"`` rows by schema; every
    remaining row must carry ``step_idx``; ``step_idx`` must be divisible by
    ``replan_steps``; within an episode the spacing must be constant and the
    resulting cycle sequence must start at 0 and be contiguous.

    A failure quarantines the whole ``yaml_id`` -- the axis is a property of a
    collection run, so a partially broken run cannot be silently patched up.
    Returns the surviving rows (each with an added ``cycle`` key) and the report.
    """
    if replan_steps <= 0:
        raise ValueError(f"replan_steps must be >= 1, got {replan_steps}")

    report = QuarantineReport()
    staged: dict[tuple, list[dict[str, Any]]] = {}

    for row in rows:
        if row.get("_kind") == "episode_summary":
            report.episode_summary += 1
            continue
        if "step_idx" not in row:
            report.missing_step_idx += 1
            report.quarantined_yamls.add(str(row.get("yaml_id")))
            continue
        step = row["step_idx"]
        if step % replan_steps:
            report.non_divisible += 1
            report.quarantined_yamls.add(str(row.get("yaml_id")))
            continue
        enriched = dict(row)
        enriched["cycle"] = step // replan_steps
        staged.setdefault(_episode_key(row), []).append(enriched)

    kept: list[dict[str, Any]] = []
    for key, items in staged.items():
        yaml_id = str(key[0])
        items.sort(key=lambda r: r["cycle"])
        cycles = [r["cycle"] for r in items]
        steps = [r["step_idx"] for r in items]
        diffs = {b - a for a, b in zip(steps, steps[1:])}
        if len(diffs) > 1:
            report.bad_spacing.add(yaml_id)
            report.quarantined_yamls.add(yaml_id)
            continue
        if cycles and (cycles[0] != 0 or cycles != list(range(len(cycles)))):
            report.non_contiguous.add(yaml_id)
            report.quarantined_yamls.add(yaml_id)
            continue
        kept.extend(items)

    # A yaml can fail on one episode and pass on another; the axis verdict is
    # per-yaml, so drop everything from a quarantined yaml.
    kept = [r for r in kept if str(r.get("yaml_id")) not in report.quarantined_yamls]
    report.kept_rows = len(kept)
    return kept, report
