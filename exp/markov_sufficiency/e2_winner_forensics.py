"""E2: winner forensics on d1 failures, inside an equal-exposure window.

Asks where the d1 ceiling comes from: aliasing the retrieval could in principle
fix (winners landing at the wrong phase), or library coverage it cannot.

Two design constraints dominate the implementation:

  * **Time axis.** Per-step rows carry environment steps; library entries carry
    inference cycles. Conversion runs first, behind :mod:`_timeaxis`'s gate.
  * **Exposure.** Successful episodes break on ``done`` while failures run to
    the step cap, so a plain "at least one deviation" comparison would read
    episode length as aliasing. The primary estimand is therefore restricted to
    the first ``K`` cycles, with ``K`` frozen from an independent batch.

"Wrong task" is not treated as evidence: production always applies a
``task_key`` filter, so a non-zero rate means broken provenance, and it is
handled as a data-integrity gate.

Public interface: :func:`load_rows`, :func:`label_rows`, :func:`episode_table`,
:func:`analyse`, :func:`main`.

Key dependencies: :mod:`_library`, :mod:`_timeaxis`, :mod:`_stats`.
"""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any, Iterable, Mapping

import numpy as np

from exp.markov_sufficiency import DELTA_E2, K_WINDOW, W_PHASE, _library, _stats, _timeaxis

DEFAULT_REPLAN_STEPS = 5
_HIT_TYPES = ("FULL_HIT", "WARM_START")

# ------------------------------------------------------------------
# Input
# ------------------------------------------------------------------


def assert_join_gate(join: dict[str, Any], suite: str) -> None:
    """Fail closed on the join rate, for **every** suite.

    NaN means "no eligible winner at all" and must stop the run rather than
    slip through a ``< 0.999`` comparison, which is False for NaN. Applying
    this to only the first suite would let a second suite with broken
    provenance reach the Holm family.
    """
    rate = join["join_rate"]
    if not (rate == rate) or rate < 0.999:
        raise SystemExit(
            f"{suite}: winner join rate {rate} did not clear 0.999 -- refusing to drop rows silently"
        )


def load_rows(path: str | pathlib.Path) -> list[dict[str, Any]]:
    """Read a per-step JSONL log (either the always_hit or the threshold source)."""
    with pathlib.Path(path).open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def dedupe_attempts(rows: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Keep only the last attempt per ``(yaml_id, task_id, init_idx)``."""
    best: dict[tuple, int] = {}
    for r in rows:
        key = (r.get("yaml_id"), r.get("task_id"), r.get("subset_init_state_idx"))
        best[key] = max(best.get(key, -1), r.get("attempt") or 0)
    kept = [r for r in rows if best[(r.get("yaml_id"), r.get("task_id"), r.get("subset_init_state_idx"))] == (r.get("attempt") or 0)]
    return kept, len(list(rows)) - len(kept) if isinstance(rows, list) else 0


# ------------------------------------------------------------------
# Labelling
# ------------------------------------------------------------------


def label_rows(
    rows: list[dict[str, Any]],
    lib: _library.Library,
    suite: str,
    *,
    require_searched: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Attach ``wrong_task`` / ``wrong_phase`` labels by joining winners to the library.

    ``require_searched`` is True for the threshold source (where only accepted
    hits carry a winner) and False for always_hit, where every searched step
    has one. The join rate is reported: below 99.9% the run stops rather than
    dropping rows silently.
    """
    w = W_PHASE[suite]
    labelled: list[dict[str, Any]] = []
    joined = considered = 0
    for r in rows:
        if require_searched and (not r.get("searched") or r.get("hit_type") not in _HIT_TYPES):
            continue
        winner_id = r.get("winner_id")
        if winner_id is None:
            continue
        considered += 1
        entry = lib.by_id.get(winner_id)
        if entry is None:
            continue
        joined += 1
        labelled.append(
            {
                **r,
                "winner_task_key": entry.payload.task_key,
                "winner_cycle": entry.step_idx,
                "wrong_task": None,  # filled by the caller-supplied task map
                "wrong_phase": abs(entry.step_idx - r["cycle"]) > w,
            }
        )
    report = {
        "considered": considered,
        "joined": joined,
        "join_rate": (joined / considered) if considered else float("nan"),
        "W": w,
    }
    return labelled, report


def apply_task_gate(
    rows: list[dict[str, Any]],
    task_key_by_id: dict[int, str],
) -> dict[str, Any]:
    """Data-integrity gate: the wrong-task rate must be exactly zero.

    Production filters candidates by ``task_key``, so a wrong-task winner is
    impossible under a healthy pipeline; a non-zero rate points at broken
    provenance, not at aliasing, and must stop the analysis.
    """
    mismatches = []
    unmapped: set = set()
    for r in rows:
        expected = task_key_by_id.get(r.get("task_id"))
        if expected is None:
            # Fail closed: an unmapped task id means the gate cannot vouch for
            # that row, which is not the same as the row being fine.
            unmapped.add(r.get("task_id"))
            continue
        wrong = expected.strip() != str(r["winner_task_key"]).strip()
        r["wrong_task"] = wrong
        if wrong:
            mismatches.append({"task_id": r.get("task_id"), "winner": r["winner_task_key"], "expected": expected})
    n_checked = sum(1 for r in rows if r.get("wrong_task") is not None)
    return {
        "n_checked": n_checked,
        "n_rows": len(rows),
        "n_wrong_task": len(mismatches),
        "unmapped_task_ids": sorted(t for t in unmapped if t is not None),
        "examples": mismatches[:5],
        # Zero rows checked is a failure, not a pass: an empty check proves
        # nothing about provenance.
        "passed": bool(rows) and not mismatches and not unmapped and n_checked == len(rows),
    }


# ------------------------------------------------------------------
# Episode aggregation
# ------------------------------------------------------------------


def episode_table(rows: list[dict[str, Any]], suite: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Collapse labelled steps into the equal-exposure episode outcome ``Y_e^K``.

    Episodes shorter than ``K`` are excluded *by schema*: the target population
    is "episodes surviving at least K cycles". Their count and outcome mix are
    returned so the report can state the restriction instead of hiding it.
    """
    k = K_WINDOW[suite]
    by_ep: dict[tuple, dict[str, Any]] = {}
    for r in rows:
        key = (r.get("yaml_id"), r.get("task_id"), r.get("subset_init_state_idx"), r.get("episode_id"))
        ep = by_ep.setdefault(key, {"task_id": r.get("task_id"), "success": bool(r.get("success")), "n_cycle": 0, "dev_in_window": False, "deviations": 0})
        ep["n_cycle"] = max(ep["n_cycle"], r["cycle"] + 1)
        if r["wrong_phase"]:
            ep["deviations"] += 1
            if r["cycle"] < k:
                ep["dev_in_window"] = True

    all_episodes = list(by_ep.values())
    kept = [e for e in all_episodes if e["n_cycle"] >= k]
    dropped = [e for e in all_episodes if e["n_cycle"] < k]
    exposure = {
        "K": k,
        "n_episodes": len(by_ep),
        "n_kept": len(kept),
        "n_dropped_short": len(dropped),
        "dropped_success_fraction": (sum(1 for e in dropped if e["success"]) / len(dropped)) if dropped else 0.0,
        "cycle_median_success": _median([e["n_cycle"] for e in by_ep.values() if e["success"]]),
        "cycle_median_failure": _median([e["n_cycle"] for e in by_ep.values() if not e["success"]]),
    }
    # The secondary estimand covers *every* episode, short ones included: the
    # K-window restriction belongs to the primary estimand only.
    return kept, all_episodes, exposure


def _median(xs: list[int]) -> float:
    return float(np.median(xs)) if xs else float("nan")


# ------------------------------------------------------------------
# Verdict
# ------------------------------------------------------------------


def secondary_quasi_binomial(all_episodes: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-cycle deviation rate over **all** episodes, length in the denominator.

    Fitted as a quasi-binomial GLM in spirit: the response is
    ``(deviations, n_cycle - deviations)`` with a logit link and task fixed
    effects, and over-dispersion inflates the standard error. It is fitted by
    IRLS here rather than pulled from statsmodels so the offline analysis keeps
    its numpy-only dependency. No ``log(n_cycle)`` offset -- that is the
    Poisson parameterisation and would double-count the denominator.
    """
    if not all_episodes:
        return {"n": 0, "coef_success": float("nan"), "ci": [float("nan"), float("nan")], "dispersion": float("nan")}

    tasks = sorted({e["task_id"] for e in all_episodes})
    cols = [tasks.index(e["task_id"]) for e in all_episodes]
    x = np.zeros((len(all_episodes), 1 + len(tasks)), dtype=np.float64)
    x[:, 0] = [1.0 if e["success"] else 0.0 for e in all_episodes]
    for i, c in enumerate(cols):
        x[i, 1 + c] = 1.0
    trials = np.array([max(1, e["n_cycle"]) for e in all_episodes], dtype=np.float64)
    events = np.array([min(e["deviations"], e["n_cycle"]) for e in all_episodes], dtype=np.float64)

    beta = np.zeros(x.shape[1])
    w = np.ones_like(trials)
    for _ in range(50):
        eta = x @ beta
        mu = 1.0 / (1.0 + np.exp(-np.clip(eta, -30, 30)))
        w = trials * mu * (1 - mu) + 1e-9
        z = eta + (events - trials * mu) / w
        xtwx = x.T @ (x * w[:, None])
        try:
            beta_new = np.linalg.solve(xtwx + 1e-8 * np.eye(x.shape[1]), x.T @ (w * z))
        except np.linalg.LinAlgError:
            break
        if np.max(np.abs(beta_new - beta)) < 1e-9:
            beta = beta_new
            break
        beta = beta_new

    mu = 1.0 / (1.0 + np.exp(-np.clip(x @ beta, -30, 30)))
    resid = (events - trials * mu) / np.sqrt(trials * mu * (1 - mu) + 1e-12)
    dof = max(1, len(all_episodes) - x.shape[1])
    dispersion = float(np.sum(resid**2) / dof)
    cov = np.linalg.pinv(x.T @ (x * w[:, None])) * max(dispersion, 1.0)
    se = float(np.sqrt(max(cov[0, 0], 0.0)))
    return {
        "n": len(all_episodes),
        "coef_success": float(beta[0]),
        "ci": [float(beta[0] - 1.959964 * se), float(beta[0] + 1.959964 * se)],
        "dispersion": dispersion,
    }


def analyse(
    episodes: list[dict[str, Any]],
    all_episodes: list[dict[str, Any]] | None = None,
    seed: int = 0,
    level: float = 0.95,
) -> dict[str, Any]:
    """CMH test, the cluster-bootstrap intervals, and the secondary model.

    ``level`` is supplied by the family driver so the interval the verdict
    consumes is the Holm-adjusted one; calling this directly yields the
    descriptive 95% interval.
    """
    tables = []
    for task_id in sorted({e["task_id"] for e in episodes}):
        grp = [e for e in episodes if e["task_id"] == task_id]
        fail = [e for e in grp if not e["success"]]
        succ = [e for e in grp if e["success"]]
        tables.append(
            (
                sum(1 for e in fail if e["dev_in_window"]),
                sum(1 for e in fail if not e["dev_in_window"]),
                sum(1 for e in succ if e["dev_in_window"]),
                sum(1 for e in succ if not e["dev_in_window"]),
            )
        )
    cmh = _stats.cmh_test(tables)

    def rate_diff(items: list[dict[str, Any]]) -> float:
        fail = [e["dev_in_window"] for e in items if not e["success"]]
        succ = [e["dev_in_window"] for e in items if e["success"]]
        if not fail or not succ:
            return float("nan")
        return float(np.mean(fail) - np.mean(succ))

    def aligned_share(items: list[dict[str, Any]]) -> float:
        fail = [e for e in items if not e["success"]]
        return float(np.mean([not e["dev_in_window"] for e in fail])) if fail else float("nan")

    strata = [e["task_id"] for e in episodes]
    ci = _stats.cluster_bootstrap_ci(episodes, rate_diff, strata=strata, seed=seed, level=level)
    ci_aligned = _stats.cluster_bootstrap_ci(episodes, aligned_share, strata=strata, seed=seed + 1, level=level)
    return {
        "cmh_p": cmh.p_value,
        "ci_level": level,
        "rate_diff": ci.estimate,
        "rate_diff_ci": [ci.low, ci.high],
        "aligned_share": ci_aligned.estimate,
        "aligned_share_ci": [ci_aligned.low, ci_aligned.high],
        "secondary": secondary_quasi_binomial(all_episodes if all_episodes is not None else episodes),
        # Deliberately no verdict: a single suite has no Holm correction, so a
        # scientific call here would bypass the registered family. Only
        # analyse_family() may produce one.
        "descriptive_only": True,
    }


def analyse_family(
    per_suite: Mapping[str, tuple[list[dict[str, Any]], list[dict[str, Any]]]],
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, Any]:
    """Two-suite Holm family: adjusted intervals plus a Holm-consistent verdict.

    The verdict may only use the Holm-adjusted interval **and** must agree with
    the Holm-adjusted CMH decision; otherwise an interval-based rule would
    quietly bypass the multiplicity correction that §3.3.2 keeps in force.
    """
    suites = sorted(per_suite)
    prelim = {s: analyse(per_suite[s][0], per_suite[s][1], seed=seed) for s in suites}
    p_values = [prelim[s]["cmh_p"] for s in suites]
    rejected = _stats.holm(p_values, alpha=alpha)
    levels = _stats.holm_adjusted_levels(p_values, alpha=alpha)

    cells = {}
    for suite, rej, level in zip(suites, rejected, levels):
        adjusted = analyse(per_suite[suite][0], per_suite[suite][1], seed=seed, level=level)
        adjusted["cmh_rejected_holm"] = bool(rej)
        adjusted["holm_level"] = level
        ci = _stats.IntervalResult(adjusted["rate_diff"], *adjusted["rate_diff_ci"], level, 0)
        ci_a = _stats.IntervalResult(adjusted["aligned_share"], *adjusted["aligned_share_ci"], level, 0)
        interval_verdict = verdict(ci, ci_a)
        # Both directions must agree with the Holm-adjusted CMH decision:
        # "aliasing" needs the rejection, and an equivalence call ("coverage")
        # may not be made while the corrected test still rejects.
        if interval_verdict == "aliasing" and not rej:
            interval_verdict = "inconclusive"
        elif interval_verdict == "library_coverage" and rej:
            interval_verdict = "inconclusive"
        adjusted["verdict"] = interval_verdict
        cells[suite] = adjusted
    return {"alpha": alpha, "family_size": len(suites), "cells": cells}


def verdict(ci: _stats.IntervalResult, ci_aligned: _stats.IntervalResult) -> str:
    """Map intervals to the three pre-registered outcomes."""
    if ci.low > 0 and ci.estimate >= DELTA_E2:
        return "aliasing"
    if ci.high < DELTA_E2 and ci_aligned.low > 0.70:
        return "library_coverage"
    return "inconclusive"


def main() -> None:
    ap = argparse.ArgumentParser(description="E2 winner forensics")
    ap.add_argument("--suite", required=True)
    ap.add_argument("--gate-rows", required=True, help="absolute path to the per-step JSONL")
    ap.add_argument("--library", required=True)
    ap.add_argument(
        "--replan-steps",
        type=int,
        required=True,
        help="taken from the collection manifest/runner config; §3.3.0 forbids guessing it",
    )
    ap.add_argument(
        "--replan-provenance",
        required=True,
        help="where that value came from (manifest path or runner flag), recorded in the output",
    )
    ap.add_argument("--suite-b", help="second suite name; required for the family verdict")
    ap.add_argument("--gate-rows-b", help="second suite per-step JSONL")
    ap.add_argument("--library-b", help="second suite artifact")
    ap.add_argument("--task-map-b", help="second suite task map")
    ap.add_argument("--source", choices=("always_hit", "threshold"), required=True)
    ap.add_argument(
        "--task-map",
        required=True,
        help='JSON mapping task_id -> canonical task_key, e.g. {"0": "pick up the black bowl ..."}. '
        "Required: the wrong-task rate is a data-integrity gate, not an optional extra.",
    )
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = load_rows(args.gate_rows)
    cycles, quarantine = _timeaxis.to_cycles(rows, args.replan_steps)
    deduped, _ = dedupe_attempts(cycles)
    lib = _library.load_library(args.library)
    labelled, join = label_rows(deduped, lib, args.suite, require_searched=(args.source == "threshold"))
    assert_join_gate(join, args.suite)

    with pathlib.Path(args.task_map).open() as fh:
        task_map = {int(k): v for k, v in json.load(fh).items()}
    gate = apply_task_gate(labelled, task_map)
    if not gate["passed"]:
        raise SystemExit(
            f"task-key integrity gate failed: {gate['n_wrong_task']} wrong-task winners. "
            "Production always filters by task_key, so this is broken provenance, "
            "not aliasing -- investigate before drawing any conclusion."
        )

    episodes, all_episodes, exposure = episode_table(labelled, args.suite)
    sensitivity = {}
    for delta in (-2, 2):
        w = W_PHASE[args.suite] + delta
        shifted = [{**r, "wrong_phase": abs(r["winner_cycle"] - r["cycle"]) > w} for r in labelled]
        kept_s, all_s, _ = episode_table(shifted, args.suite)
        sensitivity[f"W{w}"] = analyse(kept_s, all_s)["rate_diff"]

    result = {
        "suite": args.suite,
        "source": args.source,
        "quarantine": quarantine.as_dict(),
        "join": join,
        "task_gate": gate,
        "exposure": exposure,
        "analysis": analyse(episodes, all_episodes),
        "w_sensitivity": sensitivity,
        "replan_steps": {"value": args.replan_steps, "provenance": args.replan_provenance},
    }
    suite_b_args = (args.suite_b, args.gate_rows_b, args.library_b, args.task_map_b)
    if any(suite_b_args) and not all(suite_b_args):
        raise SystemExit("--suite-b/--gate-rows-b/--library-b/--task-map-b must be given together")
    if args.suite_b:
        rows_b = load_rows(args.gate_rows_b)
        cyc_b, quar_b = _timeaxis.to_cycles(rows_b, args.replan_steps)
        dedup_b, _ = dedupe_attempts(cyc_b)
        lib_b = _library.load_library(args.library_b)
        lab_b, join_b = label_rows(dedup_b, lib_b, args.suite_b, require_searched=(args.source == "threshold"))
        assert_join_gate(join_b, args.suite_b)  # same gate as suite A, before any analysis
        with pathlib.Path(args.task_map_b).open() as fh:
            gate_b = apply_task_gate(lab_b, {int(k): v for k, v in json.load(fh).items()})
        if not gate_b["passed"]:
            raise SystemExit(f"task-key integrity gate failed for {args.suite_b}")
        kept_b, all_b, exp_b = episode_table(lab_b, args.suite_b)
        result["suite_b"] = {"suite": args.suite_b, "quarantine": quar_b.as_dict(), "join": join_b,
                             "task_gate": gate_b, "exposure": exp_b}
        result["family"] = analyse_family(
            {args.suite: (episodes, all_episodes), args.suite_b: (kept_b, all_b)}
        )
    else:
        # Without the second suite there is no family and therefore no verdict.
        result["family"] = None
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        json.dump(result, fh, indent=2)
    print(json.dumps(result["analysis"], indent=2))


if __name__ == "__main__":
    main()
