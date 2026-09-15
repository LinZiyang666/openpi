"""Constants and small helpers shared by the online RIT experiment scripts.

Everything here is CPU-only and importable in the main venv. The three-tier
ladder, the executed-dimension count, the feedback-cost ledger keys and the
table column names are defined once so the builder, the replays, the emitter
and the aggregate cannot drift apart.

Public interface: ``SUITES``, ``TIER_TS``, ``EXEC_DIMS``, ``TIER_COLUMNS``,
``tier_index_of``, ``load_jsonl``, ``write_jsonl``, ``write_json``,
``sha256_file``, ``canonical_sha256``, ``CostLedger``, ``load_ledger``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import pathlib
from typing import Any, Iterable

from openpi.cache.types import DenoiseSchedule, groot_n15_schedule

SUITES = ("libero_spatial", "libero_10")
SUITE_TAG = {"libero_spatial": "sp", "libero_10": "l10"}
DENOISING_STEPS = 8
SCHEDULE = groot_n15_schedule(DENOISING_STEPS)
#: Riskiest first: one, two and four remaining steps of the k=8 loop.
TIER_TS: tuple[float, ...] = (0.875, 0.75, 0.5)
#: LIBERO executes seven action dimensions (six arm + gripper); the rest is padding.
EXEC_DIMS = 7
H_EXEC = 5
ALPHA = 0.05
WINDOW = 128
N_MIN = 20
SNAPSHOT_EVERY = 200
TARGET_IRS = (50.0, 60.0, 70.0, 80.0, 90.0)
IR_TOLERANCE_PP = 1.0


def tier_index_of(start_t: float, schedule: DenoiseSchedule = SCHEDULE) -> int:
    return schedule.snapshot_index(round(float(start_t), 4))


def tier_name(start_t: float) -> str:
    return f"warm{int(round(float(start_t) * 1000)):03d}"


#: Column names of the disagreement table, per tier: (d column, D column).
TIER_COLUMNS = {
    0.875: ("d_7", "y_rem1"),
    0.75: ("d_6", "y_rem2"),
    0.5: ("d_4", "y_rem4"),
}


def load_jsonl(path: str | pathlib.Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | pathlib.Path, rows: Iterable[dict]) -> None:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def write_json(path: str | pathlib.Path, obj: Any) -> None:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def sha256_file(path: str | pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_sha256(obj: Any) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


# ------------------------------------------------------------------
# Cost ledger
# ------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class CostLedger:
    """Per-decision costs of the served teacher plus the feedback side steps.

    Formal v2 ledgers price the measured stage ladder directly, including
    capture, feedback, dispatch, commits and snapshot allowances. The linear
    head/step fit is used only by provisional legacy ledgers. FM1 pays a
    batch-2 side step on warm starts and batch-3 on candidate misses.
    """

    stage1_ms: float
    stage2_ms: float
    stage3_head_ms: float
    stage3_step_ms: float
    fb_batch_ms: dict[int, float]
    schedule: DenoiseSchedule = SCHEDULE
    source: str = ""
    stage3_ladder_ms: dict[int, float] = dataclasses.field(default_factory=dict)
    captured_stage3_ms: dict[int, float] = dataclasses.field(default_factory=dict)
    executed_feedback_ms: dict[int, float] = dataclasses.field(default_factory=dict)
    dispatch_ms: dict[str, float] = dataclasses.field(default_factory=dict)
    commit_ms: dict[str, float] = dataclasses.field(default_factory=dict)
    snapshot_ms: dict[str, float] = dataclasses.field(default_factory=dict)

    def stage3_ms(self, steps: int) -> float:
        if steps <= 0:
            return 0.0
        if self.stage3_ladder_ms:
            return self.stage3_ladder_ms[steps]
        return self.stage3_head_ms + self.stage3_step_ms * float(steps)

    @property
    def miss_ms(self) -> float:
        return self.stage1_ms + self.stage2_ms + self.stage3_ms(self.schedule.num_steps)

    def warm_ms(self, start_t: float) -> float:
        remaining = self.schedule.remaining_steps(round(float(start_t), 4))
        return self.stage1_ms + self.stage2_ms + self.stage3_ms(remaining)

    def fb_ms(self, batch: int) -> float:
        if batch <= 0:
            return 0.0
        if batch not in self.fb_batch_ms:
            raise KeyError(f"cost ledger has no side-step cost for batch {batch}")
        return float(self.fb_batch_ms[batch])

    def episode_end_ms(self, *, online: bool, update_enabled: bool = True) -> float:
        """Conservative allowance for one task-end snapshot per episode."""
        return self.snapshot_ms["learning" if update_enabled else "frozen"] if online and self.snapshot_ms else 0.0

    def decision_ms(self, hit_type: str, start_t: float | None, fb_batch: int, *,
                    online: bool = False, update_enabled: bool = True,
                    has_candidate: bool = True, include_feedback: bool = True) -> float:
        """Price actual stages plus measured dispatch and optional online feedback.

        Host costs use a measured bound over cold/partial/full windows. Periodic
        snapshots are amortized per learned batch; task-end snapshots are separate.
        """
        if hit_type == "MISS":
            base = self.miss_ms
        elif hit_type == "WARM_START":
            base = self.warm_ms(float(start_t))
        elif hit_type == "FULL_HIT":
            base = self.stage1_ms
        else:
            raise ValueError(f"unknown hit_type {hit_type!r}")
        if self.dispatch_ms:
            base += self.dispatch_ms["online" if online else "threshold"]
        if not include_feedback:
            return base
        if online and self.commit_ms:
            remaining = self.schedule.remaining_steps(start_t) if hit_type == "WARM_START" else None
            if remaining is not None:
                base += self.captured_stage3_ms[remaining] - self.stage3_ms(remaining)
                base += self.executed_feedback_ms[remaining]
            n_feedback = fb_batch + int(remaining is not None)
            if n_feedback not in (0, 1, 3):
                raise ValueError(f"unexpected online feedback count {n_feedback}")
            mode = "learning" if update_enabled else "frozen"
            base += self.commit_ms[f"{mode}:{n_feedback}"]
            if update_enabled and n_feedback and has_candidate:
                base += self.snapshot_ms[mode] / SNAPSHOT_EVERY
        return base + self.fb_ms(fb_batch)


def load_ledger(path: str | pathlib.Path, *, require_fb: bool = True) -> CostLedger:
    """Read the complete eager v2 cost ledger, rejecting missing measurements.

    ``require_fb=False`` admits legacy stage ledgers for provisional M1
    addressing; the caller records that provisional status.
    """
    d = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    schedule = groot_n15_schedule(int(d.get("num_steps", DENOISING_STEPS)))
    fb = {int(k): float(v) for k, v in (d.get("fb_batch_ms") or {}).items()}
    ladder = {int(k): float(v) for k, v in d.get("stage3_ladder_ms", {}).items()}
    captured = {int(k): float(v) for k, v in d.get("captured_stage3_ms", {}).items()}
    executed = {int(k): float(v) for k, v in d.get("executed_feedback_ms", {}).items()}
    dispatch, commit, snapshot = (d.get(k, {}) for k in ("dispatch_ms", "commit_ms", "snapshot_ms"))
    if require_fb:
        if (d.get("protocol") != "online_rit_cost_v2" or d.get("bench", {}).get("mode") != "eager"
                or not {1, 2, 3} <= set(fb) or not {1, 2, 4, 8} <= set(ladder)
                or not {1, 2, 4} <= set(captured) or not {1, 2, 4} <= set(executed)
                or not {"online", "threshold"} <= set(dispatch)
                or not {f"{m}:{n}" for m in ("frozen", "learning") for n in (0, 1, 3)} <= set(commit)
                or not {"frozen", "learning"} <= set(snapshot)):
            raise SystemExit(f"{path}: incomplete or incompatible cost ledger; run bench_fb_cost v2")
    costs = [d["stage1_ms"], d["stage2_ms"], *fb.values(), *ladder.values(), *captured.values(), *executed.values(), *dispatch.values(), *commit.values(), *snapshot.values()]
    if any(not math.isfinite(float(v)) or float(v) < 0 for v in costs):
        raise SystemExit(f"{path}: costs must be finite and nonnegative")
    return CostLedger(
        stage1_ms=float(d["stage1_ms"]),
        stage2_ms=float(d["stage2_ms"]),
        stage3_head_ms=float(d.get("stage3_head_ms", 0.0)),
        stage3_step_ms=float(d["stage3_step_ms"]),
        fb_batch_ms=fb,
        schedule=schedule,
        source=str(path), stage3_ladder_ms=ladder, captured_stage3_ms=captured,
        executed_feedback_ms=executed, dispatch_ms=dispatch, commit_ms=commit, snapshot_ms=snapshot,
    )
