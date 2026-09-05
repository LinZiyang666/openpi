"""Statistics of the CP2 baseline line: Wilson CIs, frontiers, two-sided bootstrap.

Pure functions over per-episode ledgers (``{"uid", "task", "success",
"cost_ms", "n_dec"}`` rows) so they are unit-testable without any run data:

- ``wilson`` — per-arm success-rate interval.
- ``reference_hull`` / ``interp_frontier`` — the reference **upper concave
  hull** over (cost, SR) arms (best SR per cost, non-dominated, then chord
  test) and its linear interpolation, ``None`` outside the support. The point
  estimate and every bootstrap replicate use this one implementation.
- ``load_episode_ledger`` — journal + per_step of a conductor run -> ledger,
  pricing each decision with the caller's cost function; duplicate terminal
  rows fail loud.
- ``audit_run`` — plan §3.11 completeness gate (fail-closed identity checks
  mirroring ``exp/rit_pareto/ops/audit_k3_group.py``).
- ``bootstrap_frontier_delta`` — plan §3.11: per replicate, resample the CP2
  arm's episodes and, independently, the reference cohort's episode
  identities (one draw shared by every reference arm), both stratified by
  task; rebuild the reference frontier, interpolate at the CP2 replicate's
  IR, record ``support_miss`` when outside the support, and decide with the
  95% percentile interval of the valid replicates.
"""

from __future__ import annotations

import collections
import json
import math
import pathlib
from typing import Callable, Sequence

import numpy as np

VERDICTS = ("FULL_HIT", "WARM_START", "MISS")


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score interval for ``k`` successes out of ``n``."""
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _cross(o: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def reference_hull(points: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    """Upper concave hull of (cost, SR) arms, sorted by cost (plan §3.11).

    1. one SR per cost — the highest;
    2. drop dominated arms — SR must strictly increase with cost;
    3. drop every arm that lies on or below the chord between its hull
       neighbours (monotone-chain upper hull), so the interpolated reference
       is the envelope the frontier could attain by mixing two arms, never a
       sagging non-dominated point (``[(0,.5),(1,.51),(2,.9)]`` -> ``.70`` at
       ``x=1``, not ``.51``).
    """
    best: dict[float, float] = {}
    for x, y in points:
        best[float(x)] = max(best.get(float(x), -math.inf), float(y))
    nd: list[tuple[float, float]] = []
    top = -math.inf
    for x in sorted(best):
        if best[x] > top:
            nd.append((x, best[x]))
            top = best[x]
    hull: list[tuple[float, float]] = []
    for p in nd:
        while len(hull) >= 2 and _cross(hull[-2], hull[-1], p) >= 0.0:
            hull.pop()
        hull.append(p)
    return hull


def interp_frontier(front: Sequence[tuple[float, float]], x: float) -> float | None:
    """Linear interpolation of the frontier at ``x``; None outside its support."""
    if not front or x < front[0][0] or x > front[-1][0]:
        return None
    xs = [p[0] for p in front]
    ys = [p[1] for p in front]
    return float(np.interp(x, xs, ys))


# ------------------------------------------------------------------
# Episode ledgers
# ------------------------------------------------------------------


def load_episode_ledger(run_dir: str | pathlib.Path, cost_fn: Callable[[str, float | None], float],
                        *, require_checkpoint: str | None = None) -> dict[str, list[dict]]:
    """``{yaml_id: [{"uid", "init", "task", "success", "cost_ms", "n_dec", "counts"}]}``.

    Only accepted terminal journal rows count; per-step rows are joined on the
    accepted attempt. A second terminal row for the same ``task_uid`` fails
    loud (a journal is one terminal row per uid; silently keeping the last
    one would hide a double-dispatch). ``require_checkpoint`` (e.g. ``"CP2"``)
    fails loud when a priced row does not carry that checkpoint — a run
    against an older server or the wrong arm would otherwise be priced with
    the wrong table.
    """
    run_dir = pathlib.Path(run_dir)
    accepted_attempt: dict[str, int] = {}
    episodes: dict[str, dict[str, dict]] = collections.defaultdict(dict)
    with (run_dir / "journal.jsonl").open(encoding="utf-8") as fh:
        for raw in fh:
            if not raw.strip():
                continue
            row = json.loads(raw)
            if row.get("status") not in ("done", "failed"):
                continue
            uid = row["task_uid"]
            if uid in accepted_attempt:
                raise SystemExit(f"{run_dir}: duplicate terminal journal row for {uid}")
            if not row.get("accepted"):
                continue
            accepted_attempt[uid] = row["attempt"]
            task = _task_of(row)
            episodes[row["yaml_id"]][uid] = {
                "uid": uid, "init": _init_of(row), "task": task,
                "success": row["status"] == "done",
                "cost_ms": 0.0, "n_dec": 0, "counts": collections.Counter(),
            }
    with (run_dir / "per_step.jsonl").open(encoding="utf-8") as fh:
        for raw in fh:
            if not raw.strip():
                continue
            row = json.loads(raw)
            hit_type = row.get("hit_type")
            if hit_type is None:
                continue
            if accepted_attempt.get(row["task_uid"]) != row.get("attempt"):
                continue
            if hit_type not in VERDICTS:
                raise SystemExit(f"unpriceable hit_type {hit_type!r} in {run_dir}")
            if require_checkpoint is not None and row.get("checkpoint") != require_checkpoint:
                raise SystemExit(
                    f"{run_dir}: per_step row of {row['task_uid']} carries checkpoint="
                    f"{row.get('checkpoint')!r}, expected {require_checkpoint!r}"
                )
            ep = episodes[row["yaml_id"]].get(row["task_uid"])
            if ep is None:
                continue
            if ep["task"] is None and row.get("task_id") is not None:
                ep["task"] = row["task_id"]
            key = hit_type if hit_type != "WARM_START" else f"WARM_START@{float(row.get('start_t')):g}"
            ep["counts"][key] += 1
            ep["n_dec"] += 1
            ep["cost_ms"] += cost_fn(hit_type, row.get("start_t"))
    # Sorted by the arm-independent init identity so every arm of one pool
    # lists its episodes in the same order (the bootstrap relies on it).
    return {y: sorted(episodes[y].values(), key=lambda e: e["init"]) for y in sorted(episodes)}


def audit_run(run_dir: str | pathlib.Path, *, step_cap: int, min_hit_rows: int,
              expect_episodes: int, allow_partial: bool = False,
              expected_arms: Sequence[str] | None = None,
              expected_library_sha256: str | None = None) -> dict:
    """Plan §3.11 completeness gate over the raw journal / per_step files.

    Returns ``{"problems": [...], ...counts}``; the caller fails closed on a
    non-empty ``problems``. ``allow_partial`` relaxes only the per-arm episode
    count — never identity, attempt, truncation, arm-set or library checks.
    Mirrors ``exp/rit_pareto/ops/audit_k3_group.py``:

    - terminal journal rows == unique ``task_uid`` (0 duplicates);
    - per_step ``(uid, attempt)`` set == terminal-journal ``(uid, attempt)`` set;
    - a ``failed`` episode whose ``client_timing.steps < step_cap`` was
      truncated by a client-side exception (excise + re-run), and one with
      fewer than ``min_hit_rows`` verdict rows is likewise suspicious;
    - arm set == the export record's arm set (missing / extra);
    - every verdict row carries the server's ``library_sha256`` and it equals
      the export record's (or, without a record, one single digest).
    """
    run_dir = pathlib.Path(run_dir)
    term: dict[str, dict] = {}
    dups: list[str] = []
    with (run_dir / "journal.jsonl").open(encoding="utf-8") as fh:
        for raw in fh:
            if not raw.strip():
                continue
            r = json.loads(raw)
            if r.get("status") not in ("done", "failed"):
                continue
            uid = r["task_uid"]
            if uid in term:
                dups.append(uid)
            term[uid] = r
    j_pairs = {(u, r["attempt"]) for u, r in term.items()}
    failed = {u for u, r in term.items() if r["status"] == "failed"}
    arm_counts: collections.Counter = collections.Counter(r["yaml_id"] for r in term.values())

    ps_pairs: set = set()
    hit_rows: collections.Counter = collections.Counter()
    timing_steps: dict[str, int | None] = {}
    digests: collections.Counter = collections.Counter()
    rows_without_digest = 0
    with (run_dir / "per_step.jsonl").open(encoding="utf-8") as fh:
        for raw in fh:
            if not raw.strip():
                continue
            r = json.loads(raw)
            key = (r["task_uid"], r.get("attempt"))
            ps_pairs.add(key)
            if r.get("_kind") == "client_timing":
                timing_steps[r["task_uid"]] = r.get("steps")
                continue
            if r.get("hit_type") is None:
                continue
            if key in j_pairs:
                hit_rows[r["task_uid"]] += 1
                d = r.get("library_sha256")
                if d is None:
                    rows_without_digest += 1
                else:
                    digests[d] += 1

    problems: list[str] = []
    if dups:
        problems.append(f"{len(dups)} duplicate terminal journal uid(s): {sorted(set(dups))[:5]}")
    missing_ps = sorted(u for u, _ in (j_pairs - ps_pairs))
    extra_ps = sorted(u for u, _ in (ps_pairs - j_pairs))
    if missing_ps:
        problems.append(f"{len(missing_ps)} terminal (uid, attempt) without per_step rows: {missing_ps[:5]}")
    if extra_ps:
        problems.append(f"{len(extra_ps)} per_step (uid, attempt) without a terminal journal row: {extra_ps[:5]}")
    truncated = sorted(u for u in failed if timing_steps.get(u) is not None and timing_steps[u] < step_cap)
    if truncated:
        problems.append(f"{len(truncated)} failed episode(s) truncated below {step_cap} steps (excise + re-run): {truncated[:5]}")
    short = sorted(u for u in failed if u not in truncated and hit_rows[u] < min_hit_rows)
    if short:
        problems.append(f"{len(short)} failed episode(s) with < {min_hit_rows} verdict rows: {short[:5]}")
    if expected_arms is not None:
        exp_set, got = set(expected_arms), set(arm_counts)
        if exp_set - got:
            problems.append(f"arms in export record but not in run: {sorted(exp_set - got)}")
        if got - exp_set:
            problems.append(f"arms in run but not in export record: {sorted(got - exp_set)}")
    bad = {a: n for a, n in arm_counts.items() if n != expect_episodes}
    if bad and not allow_partial:
        problems.append(f"arms not at {expect_episodes} accepted episodes: {bad}")
    if rows_without_digest:
        problems.append(f"{rows_without_digest} verdict rows carry no library_sha256 (server did not report its library)")
    if expected_library_sha256 is not None:
        wrong = {d: n for d, n in digests.items() if d != expected_library_sha256}
        if wrong:
            problems.append(f"server library_sha256 != export record {expected_library_sha256[:16]}...: {wrong}")
    elif len(digests) > 1:
        problems.append(f"verdict rows come from {len(digests)} different libraries: {dict(digests)}")
    return {
        "problems": problems, "terminal_rows": len(term) + len(dups), "unique_uid": len(term),
        "dup_terminal_uid": len(dups), "arms": dict(arm_counts), "failed": len(failed),
        "truncated_failed_uids": truncated, "short_failed_uids": short,
        "journal_pairs_without_per_step": len(missing_ps), "per_step_pairs_not_terminal": len(extra_ps),
        "library_sha256": dict(digests), "rows_without_digest": rows_without_digest,
    }


def _init_of(journal_row: dict) -> str:
    """Episode identity shared across arms: ``task_uid`` without its yaml prefix
    (``<phase>:<task_id>:<episode_idx>``), i.e. the paired init of the pool."""
    parts = str(journal_row.get("task_uid", "")).split(":")
    return ":".join(parts[1:]) if len(parts) > 1 else parts[0]


def _task_of(journal_row: dict):
    """task id of a journal row: explicit field, else the third ``task_uid`` segment."""
    if journal_row.get("task_id") is not None:
        return journal_row["task_id"]
    parts = str(journal_row.get("task_uid", "")).split(":")
    if len(parts) >= 4 and parts[2].isdigit():
        return int(parts[2])
    return parts[2] if len(parts) >= 3 else None


def summarize(eps: Sequence[dict], miss_ms: float) -> dict:
    """Point estimates of one arm: SR, Wilson CI, IR (ratio of sums)."""
    n = len(eps)
    k = sum(1 for e in eps if e["success"])
    n_dec = sum(e["n_dec"] for e in eps)
    cost = sum(e["cost_ms"] for e in eps)
    return {
        "n_ep": n, "successes": k, "success_rate": k / n if n else float("nan"),
        "wilson95": list(wilson(k, n)), "decisions": n_dec,
        "ir_percent": 100.0 * cost / (n_dec * miss_ms) if n_dec else float("nan"),
    }


# ------------------------------------------------------------------
# Two-sided stratified bootstrap
# ------------------------------------------------------------------


def _by_task(eps: Sequence[dict]) -> dict:
    groups: dict = collections.defaultdict(list)
    for i, e in enumerate(eps):
        groups[e["task"]].append(i)
    return groups


def _resample(rng: np.random.Generator, groups: dict) -> list[int]:
    out: list[int] = []
    for _task, idx in sorted(groups.items(), key=lambda kv: str(kv[0])):
        arr = np.asarray(idx)
        out.extend(rng.choice(arr, size=arr.size, replace=True).tolist())
    return out


def _sr_ir(eps: Sequence[dict], idx: Sequence[int], miss_ms: float) -> tuple[float, float]:
    k = 0
    n_dec = 0
    cost = 0.0
    for i in idx:
        e = eps[i]
        k += 1 if e["success"] else 0
        n_dec += e["n_dec"]
        cost += e["cost_ms"]
    return k / len(idx), 100.0 * cost / (n_dec * miss_ms)


def bootstrap_frontier_delta(cp2_eps: Sequence[dict], ref_arms: dict[str, Sequence[dict]], *,
                             miss_ms: float, B: int = 2000, seed: int = 0,
                             support_miss_max: float = 0.01) -> dict:
    """ΔSR = SR_cp2 − frontier_ref(IR_cp2) with a two-sided stratified bootstrap.

    ``ref_arms`` must share one episode identity set (same ``uid`` list per
    arm, e.g. the pruned-500 pool); one per-task resample of those identities
    is applied to every reference arm within a replicate so the arms keep
    their cohort correlation.
    """
    ref_names = sorted(ref_arms)
    if not ref_names:
        raise ValueError("no reference arms")
    init_lists = [tuple(e.get("init", e["uid"]) for e in ref_arms[a]) for a in ref_names]
    if any(u != init_lists[0] for u in init_lists[1:]):
        raise ValueError("reference arms must share the same episode identity list (same order)")
    ref0 = ref_arms[ref_names[0]]

    # Point estimate. ``_sr_ir`` returns (sr, ir); the frontier wants (cost, sr).
    sr_hat, x_hat = _sr_ir(cp2_eps, range(len(cp2_eps)), miss_ms)
    ref_points = []
    for a in ref_names:
        sr_a, ir_a = _sr_ir(ref_arms[a], range(len(ref0)), miss_ms)
        ref_points.append((ir_a, sr_a))
    front_hat = reference_hull(ref_points)
    ref_hat = interp_frontier(front_hat, x_hat)
    delta_hat = None if ref_hat is None else sr_hat - ref_hat

    rng = np.random.default_rng(seed)
    cp2_groups = _by_task(cp2_eps)
    ref_groups = _by_task(ref0)
    deltas: list[float] = []
    n_miss = 0
    for _ in range(B):
        ci = _resample(rng, cp2_groups)
        ri = _resample(rng, ref_groups)
        sr_b, x_b = _sr_ir(cp2_eps, ci, miss_ms)
        pts = []
        for a in ref_names:
            sr_a, ir_a = _sr_ir(ref_arms[a], ri, miss_ms)
            pts.append((ir_a, sr_a))
        y_b = interp_frontier(reference_hull(pts), x_b)
        if y_b is None:
            n_miss += 1
            continue
        deltas.append(sr_b - y_b)
    miss_frac = n_miss / B
    if delta_hat is None:
        decision = "outside_reference_support"
        ci95 = None
    elif miss_frac > support_miss_max or not deltas:
        decision = "descriptive_only"
        ci95 = [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))] if deltas else None
    else:
        lo, hi = float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))
        ci95 = [lo, hi]
        decision = "cp2_higher" if lo > 0 else ("reference_higher" if hi < 0 else "indistinguishable")
    return {
        "sr_cp2": sr_hat, "ir_cp2": x_hat, "sr_reference_at_ir": ref_hat, "delta_sr": delta_hat,
        "delta_ci95": ci95, "B": B, "valid_replicates": len(deltas), "support_miss_frac": miss_frac,
        "support_miss_max": support_miss_max, "decision": decision,
        "reference_frontier": front_hat, "reference_arms": ref_names,
    }


__all__ = [
    "VERDICTS",
    "bootstrap_frontier_delta",
    "audit_run",
    "interp_frontier",
    "load_episode_ledger",
    "reference_hull",
    "summarize",
    "wilson",
]
