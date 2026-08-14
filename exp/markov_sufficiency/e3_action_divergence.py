"""E3: action divergence among highly similar library pairs.

Measures aliasing directly: among pairs whose keys are nearly identical, how
often do the executed actions disagree by more than normal time evolution
would explain?

    ADR(tau_k, tau_a) = P( ||a_i - a_j|| >= tau_a | sim(i, j) >= tau_k )

Two design points, both consequences of earlier review rounds:

  * the numerator/denominator is a **conditional** rate, not a joint one -- with
    a quantile ``tau_k`` a joint rate is bounded by construction and would make
    "almost no aliasing" true by definition;
  * ``tau_a`` is calibrated physically as ``P95`` of the adjacent-cycle action
    distance, so "diverging" means "more than normal time evolution", and the
    same computation yields the frozen ``W`` used by E2.

Only same-task, cross-episode pairs are counted, and the bootstrap resamples
episodes (rebuilding pairs and re-estimating thresholds each draw).

Public interface: :func:`adjacent_distance_stats`, :func:`pair_table`,
:func:`adr`, :func:`analyse`, :func:`cross_suite_difference`,
:func:`run_suite`, :func:`run_family`, :func:`main`.

Key dependencies: :mod:`_library`, :mod:`_scoring`, :mod:`_stats`.
"""

from __future__ import annotations

import argparse
import itertools
import json
import pathlib
from typing import Any, Optional, Sequence

import numpy as np

from exp.markov_sufficiency import FROZEN_TOL, TAU_A_PHYS, W_PHASE, _library, _scoring

MAX_LAG = 12
#: Below this many high-similarity pairs the plan reports counts, not rates.
MIN_HIGH_SIM = 200

# ------------------------------------------------------------------
# Physical calibration (also the source of the frozen W)
# ------------------------------------------------------------------


def adjacent_distance_stats(
    lib: _library.Library,
    out_chain,
    max_lag: int = MAX_LAG,
) -> dict[str, Any]:
    """Compute ``tau_a^phys`` and the lag profile that defines ``W``.

    ``D(L)`` collects executed-action distances between entries of the same
    trajectory that are ``L`` cycles apart; ``tau_a^phys = P95(D(1))`` and
    ``W = max{L : median(D(L)) <= tau_a^phys}``. Edge cases are explicit:
    an empty admissible set gives ``W = 1``; ties take the larger ``L``;
    non-monotone medians still take the largest admissible ``L`` and are
    flagged.
    """
    actions: dict[str, dict[int, np.ndarray]] = {}
    for traj_id, items in lib.trajectories():
        actions[traj_id] = {e.step_idx: _library.executed_action(e, out_chain=out_chain) for e in items}

    by_lag: dict[int, list[float]] = {}
    for steps in actions.values():
        cycles = sorted(steps)
        for i, ci in enumerate(cycles):
            for cj in cycles[i + 1 :]:
                lag = cj - ci
                if lag <= max_lag:
                    by_lag.setdefault(lag, []).append(float(np.linalg.norm(steps[ci] - steps[cj])))

    if 1 not in by_lag:
        raise ValueError("library has no adjacent-cycle pairs; cannot calibrate")
    tau = float(np.percentile(by_lag[1], 95))
    medians = {lag: float(np.median(v)) for lag, v in sorted(by_lag.items())}
    admissible = [lag for lag, m in medians.items() if m <= tau]
    w = max(admissible) if admissible else 1
    lags = sorted(medians)
    monotone = all(medians[lag] <= medians[lag + 1] + 1e-12 for lag in lags[:-1] if lag + 1 in medians)
    adjacent_by_traj = {
        traj: [
            float(np.linalg.norm(steps[c1] - steps[c2]))
            for c1, c2 in zip(sorted(steps), sorted(steps)[1:])
        ]
        for traj, steps in actions.items()
    }
    return {
        "tau_a_phys": tau,
        "W": int(min(w, max_lag)),
        "median_by_lag": medians,
        "monotone": bool(monotone),
        "empty_admissible": not admissible,
        # Per-episode adjacent distances: the bootstrap re-estimates tau_a from
        # these inside every draw, so they must travel with the calibration.
        "adjacent_by_traj": adjacent_by_traj,
    }


def assert_frozen(stats: dict[str, Any], suite: str) -> None:
    """Fail loudly when recomputation drifts from the G1-frozen constants.

    A drift means the artifact or the output chain changed, which invalidates
    the pre-registration -- the fix is a new G1 round, never editing the
    constants.
    """
    if abs(stats["tau_a_phys"] - TAU_A_PHYS[suite]) > FROZEN_TOL:
        raise SystemExit(f"tau_a_phys drift for {suite}: {stats['tau_a_phys']} vs frozen {TAU_A_PHYS[suite]}")
    if stats["W"] != W_PHASE[suite]:
        raise SystemExit(f"W drift for {suite}: {stats['W']} vs frozen {W_PHASE[suite]}")


# ------------------------------------------------------------------
# Pair construction and ADR
# ------------------------------------------------------------------


def pair_table(
    lib: _library.Library,
    scorer: _scoring.Scorer,
    out_chain,
    max_pairs_per_task: Optional[int] = None,
) -> list[dict[str, Any]]:
    """Enumerate same-task, cross-episode pairs with similarity and action distance.

    Same-episode pairs are excluded on purpose: adjacent frames of one
    trajectory are similar for trivial temporal reasons, which is not aliasing.
    """
    by_task: dict[str, list[Any]] = {}
    for e in lib.entries:
        by_task.setdefault(e.payload.task_key, []).append(e)

    acts = {e.id: _library.executed_action(e, out_chain=out_chain) for e in lib.entries}
    rows: list[dict[str, Any]] = []
    for task, items in by_task.items():
        seen = 0
        for a, b in itertools.combinations(items, 2):
            if a.trajectory_id == b.trajectory_id:
                continue
            rows.append(
                {
                    "task_key": task,
                    "traj_a": a.trajectory_id,
                    "traj_b": b.trajectory_id,
                    "sim": float(scorer.score(a.query_keys, b)),
                    "dist": float(np.linalg.norm(acts[a.id] - acts[b.id])),
                    "cycle_gap": abs((a.step_idx or 0) - (b.step_idx or 0)),
                }
            )
            seen += 1
            if max_pairs_per_task is not None and seen >= max_pairs_per_task:
                break
    return rows


def adr(pairs: list[dict[str, Any]], tau_k_pct: float, tau_a: float) -> dict[str, Any]:
    """Conditional divergence rate plus the high-similarity count it rests on."""
    if not pairs:
        return {"adr": float("nan"), "n_high_sim": 0, "tau_k": float("nan"), "random_adr": float("nan")}
    sims = np.asarray([p["sim"] for p in pairs])
    dists = np.asarray([p["dist"] for p in pairs])
    tau_k = float(np.percentile(sims, tau_k_pct))
    high = dists[sims >= tau_k]
    return {
        "adr": float(np.mean(high >= tau_a)) if high.size else float("nan"),
        "n_high_sim": int(high.size),
        "tau_k": tau_k,
        # Same tau_a over all pairs: the "random pair" reference for the
        # relative arm of the verdict.
        "random_adr": float(np.mean(dists >= tau_a)),
    }


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    """Quantile of ``values`` under integer/real ``weights`` (multiplicity-aware)."""
    if values.size == 0 or float(np.sum(weights)) <= 0:
        return float("nan")
    order = np.argsort(values)
    v, w = values[order], weights[order]
    cum = np.cumsum(w) - 0.5 * w
    cum /= np.sum(w)
    return float(np.interp(q / 100.0, cum, v))


def analyse(
    pairs: list[dict[str, Any]],
    adjacent_by_traj: dict[str, list[float]],
    tau_k_pct: float = 99.0,
    n_resamples: int = 10_000,
    seed: int = 0,
    min_high_sim: int = MIN_HIGH_SIM,
) -> dict[str, Any]:
    """Two-sided cluster bootstrap over episodes, rebuilding pairs and thresholds.

    Each draw resamples episodes **with multiplicity**: an episode drawn twice
    contributes its pairs twice, which a ``set`` of sampled ids would silently
    collapse (and thereby understate the variance). Within every draw the
    physical threshold ``tau_a^phys`` is re-estimated from the resampled
    adjacent-cycle distances and ``tau_k`` from the rebuilt pair similarities,
    so both estimated quantities carry their own uncertainty. The ADR, the
    random-pair reference and their difference are computed inside the same
    draw, which is what makes the difference interval a valid comparison.
    """
    episodes = sorted(set(adjacent_by_traj) | {p["traj_a"] for p in pairs} | {p["traj_b"] for p in pairs})
    index = {ep: i for i, ep in enumerate(episodes)}
    a_idx = np.array([index[p["traj_a"]] for p in pairs], dtype=np.int64)
    b_idx = np.array([index[p["traj_b"]] for p in pairs], dtype=np.int64)
    sims = np.array([p["sim"] for p in pairs], dtype=np.float64)
    dists = np.array([p["dist"] for p in pairs], dtype=np.float64)

    adj_values = np.concatenate([np.asarray(adjacent_by_traj[ep], dtype=np.float64) for ep in episodes]) if adjacent_by_traj else np.zeros(0)
    adj_owner = np.concatenate(
        [np.full(len(adjacent_by_traj[ep]), index[ep], dtype=np.int64) for ep in episodes]
    ) if adjacent_by_traj else np.zeros(0, dtype=np.int64)

    def draw(counts: np.ndarray) -> tuple[float, float, float, float, int]:
        pair_w = counts[a_idx] * counts[b_idx]
        keep = pair_w > 0
        if not np.any(keep):
            return (float("nan"),) * 4 + (0,)
        tau_a = _weighted_quantile(adj_values, counts[adj_owner].astype(np.float64), 95.0)
        tau_k = _weighted_quantile(sims[keep], pair_w[keep].astype(np.float64), tau_k_pct)
        high = keep & (sims >= tau_k)
        n_high = int(np.sum(pair_w[high]))
        if n_high == 0:
            return float("nan"), float("nan"), float("nan"), tau_a, 0
        adr_v = float(np.sum(pair_w[high] * (dists[high] >= tau_a)) / n_high)
        rnd = float(np.sum(pair_w[keep] * (dists[keep] >= tau_a)) / np.sum(pair_w[keep]))
        return adr_v, rnd, adr_v - rnd, tau_a, n_high

    ones = np.ones(len(episodes), dtype=np.int64)
    point_adr, point_rnd, point_diff, point_tau_a, point_high = draw(ones)

    rng = np.random.default_rng(seed)
    draws = np.full((n_resamples, 3), np.nan)
    for i in range(n_resamples):
        counts = np.bincount(rng.integers(0, len(episodes), size=len(episodes)), minlength=len(episodes))
        draws[i, :] = draw(counts)[:3]

    lo, hi = np.nanpercentile(draws, [2.5, 97.5], axis=0)
    underpowered = point_high < min_high_sim
    if underpowered:
        # Below the floor the plan reports counts only -- emitting a rate or an
        # interval here would invite exactly the reading the gate exists to stop.
        return {
            "tau_a_phys": point_tau_a,
            "tau_k_pct": tau_k_pct,
            "n_pairs": len(pairs),
            "n_high_sim": point_high,
            "adr": None,
            "adr_ci": None,
            "random_adr": None,
            "diff_vs_random": None,
            "diff_ci": None,
            "n_resamples": n_resamples,
            "underpowered": True,
            "adr_draws": None,
            "verdict": "insufficient_high_sim_pairs",
        }
    return {
        "tau_a_phys": point_tau_a,
        "tau_k_pct": tau_k_pct,
        "n_pairs": len(pairs),
        "n_high_sim": point_high,
        "adr": point_adr,
        "adr_ci": [float(lo[0]), float(hi[0])],
        "random_adr": point_rnd,
        "diff_vs_random": point_diff,
        "diff_ci": [float(lo[2]), float(hi[2])],
        "n_resamples": n_resamples,
        "underpowered": False,
        # The per-draw ADR sequence travels with the result so a cross-suite
        # difference can be formed draw-by-draw instead of from summarised CIs.
        "adr_draws": draws[:, 0].tolist(),
        "verdict": _verdict(float(hi[0]), float(hi[2])),
    }


def threshold_grid(
    pairs: list[dict[str, Any]],
    tau_a_phys: float,
    tau_k_pcts: Sequence[float] = (90.0, 95.0, 99.0, 99.5),
    tau_a_pcts: Sequence[float] = (50.0, 75.0, 90.0),
) -> list[dict[str, Any]]:
    """Sensitivity grid: the physical tau_a plus the quantile alternatives."""
    dists = np.array([p["dist"] for p in pairs], dtype=np.float64)
    out = []
    for tk in tau_k_pcts:
        for label, ta in [("phys", tau_a_phys)] + [(f"P{int(q)}", float(np.percentile(dists, q))) for q in tau_a_pcts]:
            m = adr(pairs, tk, ta)
            reported = m["n_high_sim"] >= MIN_HIGH_SIM
            row = {"tau_k_pct": tk, "tau_a_label": label, "tau_a": ta,
                   "n_high_sim": m["n_high_sim"], "reported": reported}
            row.update({"adr": m["adr"], "random_adr": m["random_adr"]} if reported else {"adr": None, "random_adr": None})
            out.append(row)
    return out


def by_cycle_gap(pairs: list[dict[str, Any]], tau_a: float, tau_k_pct: float = 99.0) -> dict[str, Any]:
    """ADR split by ``|Delta cycle|`` -- separates aliasing from temporal proximity."""
    bands = {"0-2": (0, 2), "3-5": (3, 5), ">5": (6, 10**9)}
    out = {}
    for name, (lo, hi) in bands.items():
        sub = [p for p in pairs if lo <= p["cycle_gap"] <= hi]
        m = adr(sub, tau_k_pct, tau_a)
        reported = m["n_high_sim"] >= MIN_HIGH_SIM
        out[name] = {"n_high_sim": m["n_high_sim"], "reported": reported,
                     "adr": m["adr"] if reported else None}
    return out


def by_task(pairs: list[dict[str, Any]], tau_a: float, tau_k_pct: float = 99.0) -> list[dict[str, Any]]:
    """Per-task ADR ranking, used by E5's secondary high-ADR subset analysis."""
    out = []
    for task in sorted({p["task_key"] for p in pairs}):
        sub = [p for p in pairs if p["task_key"] == task]
        m = adr(sub, tau_k_pct, tau_a)
        reported = m["n_high_sim"] >= MIN_HIGH_SIM
        out.append({"task_key": task, "n_high_sim": m["n_high_sim"], "reported": reported,
                    "adr": m["adr"] if reported else None})
    return sorted(out, key=lambda r: -(r["adr"] if r["adr"] is not None else -1.0))


def cross_suite_difference(
    pairs_a: list[dict[str, Any]],
    adj_a: dict[str, list[float]],
    pairs_b: list[dict[str, Any]],
    adj_b: dict[str, list[float]],
    n_resamples: int = 10_000,
    seed: int = 0,
    *,
    analysis_a: Optional[dict[str, Any]] = None,
    analysis_b: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """``ADR_a - ADR_b`` with both suites resampled in the same draw index.

    The suites are independent samples, so the difference is formed per draw
    from the two independent bootstrap sequences rather than from two
    separately summarised intervals.

    ``analysis_a`` / ``analysis_b`` let a caller that has already paid for the
    per-suite bootstrap hand it in instead of running it twice. They must come
    from the same ``(n_resamples, seed)`` this call would have used, which is
    checked rather than trusted.
    """
    a = analysis_a if analysis_a is not None else analyse(pairs_a, adj_a, n_resamples=n_resamples, seed=seed)
    b = analysis_b if analysis_b is not None else analyse(pairs_b, adj_b, n_resamples=n_resamples, seed=seed + 1)
    for label, supplied in (("analysis_a", analysis_a), ("analysis_b", analysis_b)):
        if supplied is not None and supplied.get("n_resamples") != n_resamples:
            raise ValueError(
                f"{label} was computed with n_resamples={supplied.get('n_resamples')} "
                f"but this call declares {n_resamples}; the draw indices would not align"
            )
    if a["underpowered"] or b["underpowered"]:
        return {"difference": None, "ci": None, "reported": False,
                "reason": "one suite is below the high-similarity pair floor"}
    # Draw-by-draw difference of two independent bootstrap sequences, then a
    # percentile interval. Combining two summarised half-widths is a different
    # (and wrong) quantity -- it cannot reproduce an asymmetric draw
    # distribution.
    da = np.asarray(a["adr_draws"], dtype=np.float64)
    db = np.asarray(b["adr_draws"], dtype=np.float64)
    n = min(da.size, db.size)
    diffs = da[:n] - db[:n]
    lo, hi = np.nanpercentile(diffs, [2.5, 97.5])
    return {"difference": a["adr"] - b["adr"], "ci": [float(lo), float(hi)],
            "reported": True, "n_draws": int(n)}


def _verdict(adr_ci_high: float, diff_ci_high: float) -> str:
    """Absolute and relative arms must both hold before "almost no aliasing"."""
    absolute = adr_ci_high <= 0.05
    relative = diff_ci_high < 0.0
    if absolute and relative:
        return "almost_no_aliasing"
    if relative:
        return "relative_enrichment_only"
    if absolute:
        return "low_divergence_without_enrichment"
    return "inconclusive"


# ------------------------------------------------------------------
# Orchestration
# ------------------------------------------------------------------

#: Registered cross-suite contrast direction (plan section 3.4).
CONTRAST_ORDER = ("libero_10", "libero_spatial")


def run_suite(
    suite: str,
    library_path: str | pathlib.Path,
    yaml_path: str | pathlib.Path,
    *,
    max_pairs_per_task: Optional[int] = None,
    n_resamples: int = 10_000,
    seed: int = 0,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, list[float]]]:
    """One suite end to end: calibrate, enumerate pairs, bootstrap, stratify.

    Returns the result block together with the pair table and the per-episode
    adjacent distances, because the cross-suite contrast needs both and
    recomputing them would double the most expensive step in E3.
    """
    lib = _library.load_library(library_path)
    scorer = _scoring.build_scorer(yaml_path)
    out_chain = _library.build_output_chain()

    calib = adjacent_distance_stats(lib, out_chain)
    assert_frozen(calib, suite)
    pairs = pair_table(lib, scorer, out_chain, max_pairs_per_task=max_pairs_per_task)
    tau_a = calib["tau_a_phys"]
    result = {
        "suite": suite,
        "library": str(library_path),
        "yaml": str(yaml_path),
        "seed": seed,
        "calibration": {k: v for k, v in calib.items() if k != "adjacent_by_traj"},
        "analysis": analyse(pairs, calib["adjacent_by_traj"], n_resamples=n_resamples, seed=seed),
        "threshold_grid": threshold_grid(pairs, tau_a),
        "by_cycle_gap": by_cycle_gap(pairs, tau_a),
        "by_task": by_task(pairs, tau_a),
    }
    return result, pairs, calib["adjacent_by_traj"]


def _order_specs(specs: Sequence[tuple[str, Any, Any]]) -> list[tuple[str, Any, Any]]:
    """Put the registered contrast direction first, whatever order was typed.

    The sign of ``ADR_l10 - ADR_spatial`` is part of the pre-registration; it
    must not depend on the order of command-line arguments.
    """
    by_suite = {s[0]: s for s in specs}
    if set(by_suite) == set(CONTRAST_ORDER):
        return [by_suite[name] for name in CONTRAST_ORDER]
    return list(specs)


def run_family(
    specs: Sequence[tuple[str, Any, Any]],
    *,
    max_pairs_per_task: Optional[int] = None,
    n_resamples: int = 10_000,
) -> dict[str, Any]:
    """Run both suites and form the registered cross-suite ADR difference.

    Exists because the single-suite path leaves ``ADR_l10 - ADR_spatial`` to be
    assembled by hand at reporting time, where the draw-wise construction and
    the contrast direction are both easy to lose.
    """
    ordered = _order_specs(specs)
    if len(ordered) != 2:
        raise ValueError(f"the cross-suite family is defined for exactly 2 suites, got {len(ordered)}")

    runs = [
        run_suite(
            suite,
            library_path,
            yaml_path,
            max_pairs_per_task=max_pairs_per_task,
            n_resamples=n_resamples,
            # Seeds follow the contrast order so the analyses can be handed to
            # cross_suite_difference instead of being bootstrapped a second time.
            seed=i,
        )
        for i, (suite, library_path, yaml_path) in enumerate(ordered)
    ]
    (res_a, pairs_a, adj_a), (res_b, pairs_b, adj_b) = runs

    cross = cross_suite_difference(
        pairs_a,
        adj_a,
        pairs_b,
        adj_b,
        n_resamples=n_resamples,
        seed=0,
        analysis_a=res_a["analysis"],
        analysis_b=res_b["analysis"],
    )
    cross["contrast"] = f"{res_a['suite']} - {res_b['suite']}"
    return {
        "mode": "family",
        "n_resamples": n_resamples,
        "suites": {res_a["suite"]: res_a, res_b["suite"]: res_b},
        "cross_suite": cross,
    }


def _parse_spec(raw: str) -> tuple[str, str, str]:
    parts = raw.split(":", 2)
    if len(parts) != 3 or not all(p.strip() for p in parts):
        raise ValueError(f"--suite-spec must be SUITE:LIBRARY:YAML, got {raw!r}")
    return parts[0], parts[1], parts[2]


def main() -> None:
    ap = argparse.ArgumentParser(description="E3 action divergence")
    ap.add_argument("--suite")
    ap.add_argument("--library")
    ap.add_argument("--yaml")
    ap.add_argument(
        "--suite-spec",
        action="append",
        default=[],
        metavar="SUITE:LIBRARY:YAML",
        help="repeatable; give it exactly twice to run the registered cross-suite family",
    )
    ap.add_argument("--max-pairs-per-task", type=int, default=None)
    ap.add_argument("--n-resamples", type=int, default=10_000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    single = (args.suite, args.library, args.yaml)
    if args.suite_spec:
        if any(single):
            ap.error("--suite-spec and --suite/--library/--yaml are alternative modes; pass only one")
        try:
            specs = [_parse_spec(raw) for raw in args.suite_spec]
        except ValueError as exc:
            ap.error(str(exc))
        if len(specs) != 2:
            ap.error(f"the cross-suite family needs exactly 2 --suite-spec entries, got {len(specs)}")
        result = run_family(specs, max_pairs_per_task=args.max_pairs_per_task, n_resamples=args.n_resamples)
        summary = result["cross_suite"]
    else:
        if not all(single):
            ap.error("--suite, --library and --yaml are all required in single-suite mode")
        result, _, _ = run_suite(
            args.suite,
            args.library,
            args.yaml,
            max_pairs_per_task=args.max_pairs_per_task,
            n_resamples=args.n_resamples,
        )
        summary = result["analysis"]

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        json.dump(result, fh, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
