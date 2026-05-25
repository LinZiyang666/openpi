"""Progress / throughput / health aggregation + compact rendering (plan §10).

Fed by the driver's progress/result stream. Renders an *aggregate* view (global
done/total/SR + per-server / worker-count summary) rather than one tqdm bar per
worker — at 96 workers that is unreadable and the driver often runs head-less
(plan §10, G1R1/self-review P16). Per-worker detail is available via
``health.snapshot()`` on demand.

Coupling map:
  DEPENDS ON:  health.py, scheduler.py (optional, for global progress)
  CONSUMED BY: driver.py (monitor hook)
"""

from __future__ import annotations

from openpi.conductor.health import HealthAggregator


class Monitor:
    """Driver-side monitor: collects progress/results, renders an aggregate view."""

    def __init__(self, scheduler=None, health: HealthAggregator | None = None) -> None:
        """``scheduler`` (optional; anything exposing ``progress()``) feeds the
        global done/total line. Without it ``render()`` shows only the worker
        summary."""
        self._scheduler = scheduler
        self._health = health or HealthAggregator()

    @property
    def health(self) -> HealthAggregator:
        return self._health

    def on_progress(self, payload: dict) -> None:
        self._health.on_progress(payload)

    def on_result(self, result) -> None:
        # ``result`` is an EpisodeResult.
        self._health.on_result(result.task_uid, success=result.success)

    def render(self) -> str:
        """A compact, head-less-friendly status string."""
        lines: list[str] = []
        if self._scheduler is not None:
            p = self._scheduler.progress()
            total = p["episodes_total"] or 1
            sr = p["episodes_done"] / total
            lines.append(
                f"episodes {p['episodes_done']}/{p['episodes_total']} done ({p['episodes_failed']} failed, {sr:.0%})"
            )
        snap = self._health.snapshot()
        running = sum(1 for w in snap.values() if w["state"] == "running")
        total_aps = round(sum(w["actions_per_s"] for w in snap.values()), 1)
        lines.append(f"workers: {running} running / {len(snap)} total, {total_aps} act/s")
        return "\n".join(lines)
