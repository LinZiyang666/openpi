"""Paired analysis of the E4 and E5 rollout arms.

The emitters produce the yamls; this module consumes the resulting per-episode
outcomes and produces the registered statistics. It is the analysis half that
plan §3.5 / §5.4 / §5.5 require -- the file list in plan §6 named only the
emitters, so this module is an addition flagged to the reviewer rather than a
silent deviation.

E4: ``A3 - A1`` is the primary contrast (exact McNemar, Holm across the two
suites, Wilson intervals per arm, paired bootstrap interval for the risk
difference). The interaction ``(A3-A1)-(A2-A0)`` is permanently estimation
only, bootstrapped from the four-arm **joint** episode outcome so its variance
reflects the actual dependence between arms. A discordance above the proxy
Q75 downgrades the suite's primary to estimation, since n = 950 is the
held-out ceiling and there is no room to enlarge the sample.

E5: estimation only by design -- the interval decides among four mutually
exclusive outcomes, so no ordering of ``if`` statements can influence the call.

Public interface: :func:`load_arm`, :func:`paired_table`, :func:`analyse_e4`,
:func:`analyse_e5`.

Key dependency: :mod:`_stats`.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any, Mapping, Sequence

import numpy as np

from exp.markov_sufficiency import DELTA_E4, _stats

#: Discordance proxies from plan §3.5.1 (no-filter historical contrasts). Above
#: these the primary verdict for that suite degrades to estimation.
PI_D_PROXY_Q75 = {"libero_spatial": 0.12, "libero_10": 0.31}

#: E5 target effect: the originally screened d3-trough size.
E5_TARGET = 0.04

#: Effect floor for E4's primary call (plan §5.4): Holm rejection alone is not
#: enough, the point estimate must also reach 5pp.
E4_EFFECT_FLOOR = 0.05


# ------------------------------------------------------------------
# Input
# ------------------------------------------------------------------


def load_arm(path: str | pathlib.Path) -> dict[tuple[int, int], bool]:
    """Read one arm's episode results into ``{(task_id, init_idx): success}``."""
    with pathlib.Path(path).open() as fh:
        rows = json.load(fh)
    return {(int(r["task_id"]), int(r["init_state_idx"])): bool(r["success"]) for r in rows}


def paired_table(arm_a: Mapping[tuple[int, int], bool], arm_b: Mapping[tuple[int, int], bool]) -> dict[str, Any]:
    """Discordant counts on the episodes both arms actually ran."""
    shared = sorted(set(arm_a) & set(arm_b))
    b = sum(1 for k in shared if arm_a[k] and not arm_b[k])
    c = sum(1 for k in shared if arm_b[k] and not arm_a[k])
    return {"n_pairs": len(shared), "b": b, "c": c, "keys": shared,
            "pi_d_obs": (b + c) / len(shared) if shared else float("nan")}


# ------------------------------------------------------------------
# E4
# ------------------------------------------------------------------


def _risk_diff_ci(arm_a: Mapping, arm_b: Mapping, keys: Sequence, seed: int = 0, level: float = 0.95) -> _stats.IntervalResult:
    """Paired bootstrap interval for ``SR(a) - SR(b)`` (episode is the unit)."""
    items = [{"a": arm_a[k], "b": arm_b[k]} for k in keys]
    return _stats.cluster_bootstrap_ci(
        items,
        lambda xs: float(np.mean([i["a"] for i in xs]) - np.mean([i["b"] for i in xs])),
        seed=seed,
        level=level,
    )


def analyse_e4(arms_by_suite: Mapping[str, Mapping[str, Mapping[tuple[int, int], bool]]], alpha: float = 0.05) -> dict[str, Any]:
    """Primary ``A3-A1`` per suite with Holm, plus the estimation-only interaction."""
    suites = sorted(arms_by_suite)
    primary = {}
    for suite in suites:
        arms = arms_by_suite[suite]
        table = paired_table(arms["A3"], arms["A1"])
        primary[suite] = {
            "table": {k: v for k, v in table.items() if k != "keys"},
            "mcnemar": _stats.mcnemar_exact(table["b"], table["c"], n_pairs=table["n_pairs"]),
        }

    p_values = [primary[s]["mcnemar"].p_value for s in suites]
    rejected = _stats.holm(p_values, alpha=alpha)
    levels = _stats.holm_adjusted_levels(p_values, alpha=alpha)

    cells = {}
    for suite, rej, level in zip(suites, rejected, levels):
        arms = arms_by_suite[suite]
        table = paired_table(arms["A3"], arms["A1"])
        ci = _risk_diff_ci(arms["A3"], arms["A1"], table["keys"], level=level)
        desc = _risk_diff_ci(arms["A3"], arms["A1"], table["keys"], level=0.95)
        downgraded = table["pi_d_obs"] > PI_D_PROXY_Q75.get(suite, float("inf"))
        cells[suite] = {
            "n_pairs": table["n_pairs"],
            "pi_d_obs": table["pi_d_obs"],
            "pi_d_proxy_q75": PI_D_PROXY_Q75.get(suite),
            "downgraded_to_estimation": bool(downgraded),
            "risk_diff": ci.estimate,
            "holm_ci": [ci.low, ci.high],
            "descriptive_ci_95": [desc.low, desc.high],
            "wilson": {
                arm: [_stats.wilson_ci(sum(arms[arm].values()), len(arms[arm])).low,
                      _stats.wilson_ci(sum(arms[arm].values()), len(arms[arm])).high]
                for arm in sorted(arms)
            },
            "p_value": primary[suite]["mcnemar"].p_value,
            "holm_rejected": bool(rej),
            "verdict": _e4_verdict(ci, bool(rej), downgraded),
            "interaction": interaction_estimate(arms),
            "a4_descriptive": _a4_descriptive(arms),
            # Registered descriptive contrasts, each with its own paired CI.
            "reproducibility_A2_minus_A0": _contrast(arms, "A2", "A0"),
            "filter_effect_A1_minus_A0": _contrast(arms, "A1", "A0"),
        }
    return {"alpha": alpha, "delta_equivalence": DELTA_E4, "cells": cells}


def _e4_verdict(ci: _stats.IntervalResult, holm_rejected: bool, downgraded: bool) -> str:
    """Four mutually exclusive outcomes; equivalence needs the bound, not a null."""
    if downgraded:
        return "estimation_only_discordance_above_proxy"
    # Plan §5.4 requires the effect floor as well: a significant 4pp improvement
    # is not the registered "filter improves" call.
    if holm_rejected and ci.low > 0 and ci.estimate >= E4_EFFECT_FLOOR:
        return "filter_improves"
    if ci.low > -DELTA_E4 and ci.high < DELTA_E4:
        return "practically_equivalent"
    return "no_improvement_found_inconclusive"


def interaction_estimate(arms: Mapping[str, Mapping[tuple[int, int], bool]], seed: int = 0) -> dict[str, Any]:
    """``(A3-A1)-(A2-A0)`` from the four-arm joint outcome, estimation only.

    Resampling the joint vector (rather than each contrast separately) is what
    keeps the between-arm dependence in the interval. No p-value is produced:
    the interaction is permanently estimation-only.
    """
    keys = sorted(set(arms["A0"]) & set(arms["A1"]) & set(arms["A2"]) & set(arms["A3"]))
    items = [{a: arms[a][k] for a in ("A0", "A1", "A2", "A3")} for k in keys]

    def theta(xs: list[dict[str, bool]]) -> float:
        m = {a: float(np.mean([x[a] for x in xs])) for a in ("A0", "A1", "A2", "A3")}
        return (m["A3"] - m["A1"]) - (m["A2"] - m["A0"])

    ci = _stats.cluster_bootstrap_ci(items, theta, seed=seed)
    return {"n_pairs": len(keys), "theta": ci.estimate, "ci": [ci.low, ci.high], "p_value": None}


def _contrast(arms: Mapping[str, Mapping[tuple[int, int], bool]], a: str, b: str, seed: int = 0) -> dict[str, Any]:
    """Paired estimate + bootstrap CI for one descriptive arm contrast."""
    if a not in arms or b not in arms:
        return {"present": False}
    table = paired_table(arms[a], arms[b])
    ci = _risk_diff_ci(arms[a], arms[b], table["keys"], seed=seed)
    return {"present": True, "n_pairs": table["n_pairs"], "risk_diff": ci.estimate, "ci": [ci.low, ci.high]}


def _a4_descriptive(arms: Mapping[str, Mapping[tuple[int, int], bool]]) -> dict[str, Any]:
    """A4 (exact filter) stays exploratory: hit-rate collapse is the thing to watch."""
    if "A4" not in arms:
        return {"present": False}
    w = _stats.wilson_ci(sum(arms["A4"].values()), len(arms["A4"]))
    vs_a2 = _contrast(arms, "A4", "A2")
    return {
        "present": True,
        "sr": w.estimate,
        "sr_ci": [w.low, w.high],
        "vs_A2": vs_a2,
        "note": "exploratory; excluded from the Holm family",
    }


# ------------------------------------------------------------------
# E5
# ------------------------------------------------------------------


def analyse_e5(
    anchor: Mapping[tuple[int, int], bool],
    shapes: Mapping[str, Mapping[tuple[int, int], bool]],
    high_adr_tasks: Sequence[int] = (),
    seed: int = 0,
) -> dict[str, Any]:
    """Estimation-only confirmatory analysis against the same-base d1 anchor."""
    out = {}
    for name, arm in sorted(shapes.items()):
        table = paired_table(arm, anchor)
        ci = _risk_diff_ci(arm, anchor, table["keys"], seed=seed)
        cell = {
            "n_pairs": table["n_pairs"],
            "risk_diff": ci.estimate,
            "ci": [ci.low, ci.high],
            "verdict": _e5_verdict(ci.low, ci.high),
        }
        if high_adr_tasks:
            subset = [k for k in table["keys"] if k[0] in set(high_adr_tasks)]
            if subset:
                sub_ci = _risk_diff_ci(arm, anchor, subset, seed=seed + 1)
                cell["high_adr_subset"] = {
                    "n_pairs": len(subset),
                    "risk_diff": sub_ci.estimate,
                    "ci": [sub_ci.low, sub_ci.high],
                    "note": "secondary; not in any comparison family",
                }
        out[name] = cell
    return {"target_effect": E5_TARGET, "shapes": out}


def _e5_verdict(low: float, high: float) -> str:
    """Four mutually exclusive interval outcomes -- no ``if`` ordering decides.

    The overlapping region (interval strictly inside ``(0, target)``) gets its
    own label rather than being claimed by whichever branch runs first.
    """
    if not (low == low and high == high):
        return "inconclusive"
    positive = low > 0.0
    below_target = high < E5_TARGET
    if positive and not below_target:
        return "replicated"
    if positive and below_target:
        return "positive_but_below_target"
    if not positive and below_target:
        return "not_supported"
    return "inconclusive"


# ------------------------------------------------------------------
# Interfaces and CLI
# ------------------------------------------------------------------


def task_ids_from_adr_ranking(
    ranking: Sequence[Mapping[str, Any]],
    task_key_to_id: Mapping[str, int],
    top_n: int = 3,
) -> list[int]:
    """Translate E3's ``task_key``-keyed ADR ranking into E5's integer task ids.

    E3 reports per-task ADR keyed by the canonical task string while the rollout
    outcomes are keyed by integer task id; without this explicit mapping the two
    secondary analyses cannot be connected at all.
    """
    out: list[int] = []
    for row in ranking:
        if not row.get("reported"):
            continue
        tid = task_key_to_id.get(str(row["task_key"]).strip())
        if tid is not None:
            out.append(int(tid))
        if len(out) >= top_n:
            break
    return out


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="E4 / E5 paired rollout analysis")
    ap.add_argument("--mode", choices=("e4", "e5"), required=True)
    ap.add_argument(
        "--arms",
        required=True,
        help="e4: 'suite:ARM=path,...' for both suites; e5: 'anchor=path,shape0=path,...'",
    )
    ap.add_argument("--adr-ranking", help="e5 only: E3 output JSON providing the per-task ADR ranking")
    ap.add_argument("--task-map", help="e5 only: JSON mapping task_id -> task_key (inverted internally)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.mode == "e4":
        by_suite: dict[str, dict[str, Any]] = {}
        for chunk in args.arms.split(","):
            suite_arm, _, path = chunk.partition("=")
            suite, _, arm = suite_arm.partition(":")
            by_suite.setdefault(suite, {})[arm] = load_arm(path)
        if len(by_suite) < 2:
            raise SystemExit("E4's Holm family covers both suites; pass arms for each")
        result = analyse_e4(by_suite)
    else:
        arms = {}
        for chunk in args.arms.split(","):
            name, _, path = chunk.partition("=")
            arms[name] = load_arm(path)
        anchor = arms.pop("anchor", None)
        if anchor is None:
            raise SystemExit("E5 needs the same-base d1 anchor (anchor=<path>)")
        high_adr: list[int] = []
        if args.adr_ranking and args.task_map:
            with pathlib.Path(args.adr_ranking).open() as fh:
                ranking = json.load(fh).get("by_task", [])
            with pathlib.Path(args.task_map).open() as fh:
                key_to_id = {v.strip(): int(k) for k, v in json.load(fh).items()}
            high_adr = task_ids_from_adr_ranking(ranking, key_to_id)
        result = analyse_e5(anchor, arms, high_adr_tasks=high_adr)

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        json.dump(result, fh, indent=2, default=str)
    print(json.dumps(result, indent=2, default=str)[:2000])


if __name__ == "__main__":
    main()
