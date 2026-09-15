"""Accepted eval evidence shared by experiment runners and their summaries.

Run ids fence dispatch counters that restart after a driver crash. Conflicting
terminal records or step traces are rejected rather than selected by file order.
This module uses only the standard library and does not alter scheduling.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib

TERMINAL = ("done", "failed")


# ------------------------------------------------------------------
# Terminal records (journal)
# ------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class Winner:
    """The accepted terminal record of one episode, with the run that produced it."""

    arm: str
    uid: str
    task_id: int
    episode_idx: int
    success: bool
    attempt: int
    run_id: str
    record: str = dataclasses.field(default="", repr=False)


def _split_uid(uid: str) -> tuple[str, str, int, int]:
    parts = uid.rsplit(":", 3)
    if len(parts) != 4:
        raise SystemExit(f"unrecognised task_uid {uid!r}")
    return parts[0], parts[1], int(parts[2]), int(parts[3])


def scan_journal(path: pathlib.Path, source: str, *, allowed: set[str] | None = None) -> list[dict]:
    """Terminal rows the scheduler did not reject, each checked for identity before it can count.

    A row counts only as what it claims to be: its ``task_uid`` must name the
    row's own ``yaml_id`` and phase, the phase must be ``eval`` (this design has
    no warm-up; any other phase is a foreign journal), and ``attempt`` /
    ``run_id`` must be present so the winner can be tied to the run that
    produced it.
    """
    if not path.exists():
        raise SystemExit(f"{source}: journal {path} missing")
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("status") not in TERMINAL or r.get("accepted") is False:
            continue
        if allowed is not None and r.get("yaml_id") not in allowed:
            continue
        uid, arm = r["task_uid"], r["yaml_id"]
        uid_arm, uid_phase, task_id, episode_idx = _split_uid(uid)
        if uid_arm != arm or uid_phase != r.get("phase"):
            raise SystemExit(f"{source}: {uid} does not match its row's yaml_id/phase ({arm}, {r.get('phase')})")
        if uid_phase != "eval":
            raise SystemExit(f"{source}: {uid} is a {uid_phase!r} record; only eval terminals may count")
        if r.get("attempt") is None or not r.get("run_id"):
            raise SystemExit(f"{source}: {uid} lacks attempt/run_id; it cannot be bound to a run")
        if (r.get("accepted") is not True or type(r.get("success")) is not bool
                or type(r["attempt"]) is not int or r["attempt"] < 1
                or not isinstance(r["run_id"], str)):
            raise SystemExit(f"{source}: {uid} has invalid accepted/success/attempt/run_id stamps")
        rows.append({"arm": arm, "uid": uid, "task_id": task_id, "episode_idx": episode_idx,
                     "success": bool(r["success"]), "attempt": int(r["attempt"]), "run_id": str(r["run_id"]),
                     "error": r.get("error"), "record": json.dumps(r, sort_keys=True, allow_nan=False)})
    return rows


def merge_source(merged: dict[str, dict[str, Winner]], source: str, journal: pathlib.Path,
                 allowed: set[str], *, restrict: bool) -> None:
    """Fold one journal's accepted winners for ``allowed`` arms into ``merged``.

    Within one run the highest accepted attempt wins. Attempts from different
    runs cannot be ordered; accepting an episode in two runs is a conflict.
    Same-attempt records and cross-file winners must be exact duplicates,
    including status and error. ``restrict``
    distinguishes a run journal (may only carry the arms its matrix declared)
    from the reference journal (other arms are present and simply not
    selected). A winner carrying a worker error is an infrastructure failure,
    not a rollout outcome.
    """
    rows = scan_journal(journal, source, allowed=None if restrict else allowed)
    foreign = {r["arm"] for r in rows} - allowed
    if restrict and foreign:
        raise SystemExit(f"{source}: journal {journal} carries arms outside the manifest: {sorted(foreign)}")
    local: dict[tuple[str, str], dict] = {}
    for r in rows:
        if r["arm"] not in allowed:
            continue
        key = (r["arm"], r["uid"])
        prev = local.get(key)
        if prev is not None and r["run_id"] != prev["run_id"]:
            raise SystemExit(f"{source}: {r['uid']} accepted run identities disagree "
                             f"({prev['run_id']} vs {r['run_id']}); attempts restart per run")
        if prev is None or r["attempt"] > prev["attempt"]:
            local[key] = r
        elif r["attempt"] == prev["attempt"] and r["record"] != prev["record"]:
            raise SystemExit(f"{source}: {r['uid']} has two accepted rows at attempt {r['attempt']} that disagree "
                             "in their terminal record (including status/error)")
    for (arm, uid), r in local.items():
        if r["error"]:
            raise SystemExit(f"{source}: {uid} accepted attempt {r['attempt']} carries a worker error "
                             f"({str(r['error'])[:120]}); an infrastructure failure is not a rollout outcome")
        w = Winner(arm, uid, r["task_id"], r["episode_idx"], r["success"], r["attempt"], r["run_id"], r["record"])
        prev = merged.setdefault(arm, {}).get(uid)
        if prev is None:
            merged[arm][uid] = w
        elif prev != w:
            raise SystemExit(f"{source}: {uid} already merged from another file with a different "
                             f"outcome/attempt/run ({prev} vs {w})")


# ------------------------------------------------------------------
# Per-step evidence (FULL_HIT traces)
# ------------------------------------------------------------------
def scan_per_step(path: str | pathlib.Path) -> dict[tuple[str, str, str, int], dict[int, dict]]:
    """``{(arm, uid, run_id, attempt): {step_idx: row}}`` from one per-step file.

    Every row must carry the driver's stamps (``attempt``, ``run_id``,
    ``step_idx``); a row the scheduler fenced (``accepted: false``) is not
    evidence. Two rows for the same step of the same run/attempt make the
    trace ambiguous and are refused outright.
    """
    path = pathlib.Path(path)
    if not path.exists():
        raise SystemExit(f"per-step file {path} missing")
    out: dict[tuple[str, str, str, int], dict[int, dict]] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        uid, hit = r.get("task_uid"), r.get("hit_type")
        if uid is None or hit is None:
            continue
        if r.get("accepted") is False:
            continue
        arm = r.get("yaml_id") or uid.rsplit(":", 3)[0]
        attempt, run_id, step = r.get("attempt"), r.get("run_id"), r.get("step_idx")
        if attempt is None or not run_id or step is None:
            raise SystemExit(f"{path}: per-step row for {uid} lacks attempt/run_id/step_idx; it cannot witness a run")
        if (r.get("accepted") is not True or type(attempt) is not int or attempt < 1
                or type(step) is not int or step < 0 or not isinstance(run_id, str)):
            raise SystemExit(f"{path}: per-step row for {uid} has invalid accepted/run/attempt/step stamps")
        uid_arm, phase, _, _ = _split_uid(uid)
        if uid_arm != arm or phase != "eval" or r.get("phase", phase) != phase:
            raise SystemExit(f"{path}: per-step row {uid} does not match its yaml_id/eval phase")
        trace = out.setdefault((arm, uid, str(run_id), int(attempt)), {})
        if int(step) in trace:
            raise SystemExit(f"{path}: duplicate per-step row for {uid} run {run_id} attempt {attempt} step {step}")
        trace[int(step)] = r
    return out


def merge_hits(per_step_paths: list[str]) -> dict[tuple[str, str, str, int], dict[int, dict]]:
    """Traces from one or more per-step files; the same (uid, run, attempt) must be step-for-step identical everywhere."""
    merged: dict[tuple[str, str, str, int], dict[int, dict]] = {}
    for path in per_step_paths:
        for key, trace in scan_per_step(path).items():
            if key in merged and merged[key] != trace:
                raise SystemExit(f"{key[0]}: {key[1]} run {key[2]} attempt {key[3]} has different traces in two per-step files")
            merged.setdefault(key, trace)
    return merged


def witness_full_hit(arm: str, winners: dict[str, Winner], hits: dict[tuple[str, str, str, int], dict[int, dict]]) -> dict:
    """Every accepted episode needs a non-empty, all-FULL_HIT trace from its own run and attempt.

    Rows from another run or attempt of the same episode are stale evidence:
    counted, never substituted. Rows for an episode this arm never accepted
    cannot belong to its result.
    """
    own = {k: v for k, v in hits.items() if k[0] == arm}
    unknown = sorted({k[1] for k in own} - set(winners))
    if unknown:
        raise SystemExit(f"{arm}: {len(unknown)} per-step episode(s) are absent from the accepted ledger "
                         f"(e.g. {unknown[:5]}); evidence for episodes this run never accepted cannot be part of its result")
    stale = 0
    steps = 0
    missing, offenders = [], {}
    for uid, w in winners.items():
        trace = own.get((arm, uid, w.run_id, w.attempt))
        stale += sum(len(v) for k, v in own.items() if k[1] == uid and (k[2], k[3]) != (w.run_id, w.attempt))
        if not trace:
            missing.append(uid)
            continue
        steps += len(trace)
        bad = sorted({r["hit_type"] for r in trace.values() if r["hit_type"] != "FULL_HIT"})
        if bad:
            offenders[uid] = bad
    if missing:
        stale_only = [u for u in missing if any(k[1] == u for k in own)]
        detail = (f" {len(stale_only)} of them only have rows from another run/attempt (e.g. {stale_only[:3]}),"
                  " which cannot witness the accepted run." if stale_only else "")
        raise SystemExit(f"{arm}: {len(missing)} accepted episode(s) have no inference rows at their accepted "
                         f"run/attempt (e.g. {missing[:5]})." + detail)
    if offenders:
        raise SystemExit(f"{arm}: {len(offenders)} episode(s) served non-FULL_HIT steps "
                         f"(e.g. {dict(list(offenders.items())[:5])}); the arm was not served entirely from cache")
    return {"episodes": len(winners), "steps": steps, "stale_rows_ignored": stale}
