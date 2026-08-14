"""E1 / E1-O: leave-one-episode-out action-prediction residuals (A/B/C).

Answers the discriminating question behind the two-condition framework: is the
history uninformative for the retrieval target (a is false), or is it
informative but inexpressible by the additive smoothing operator (b is false)?

  A  current key only (the production baseline)
  B  raw history, scored with the production weights **of that depth**
  C  difference features weighted per modality as ``w_f (x) [1, gamma, gamma]``
  O  oracle phase alignment on normalised progress, run for both A and B --
     the honest test of the time-elasticity competitor, offline only

Inference unit is the episode; every step-level number here is descriptive.
The registered verdict lives in :func:`family_analysis`, which applies Holm
across the four primary cells and falls back to estimation when the pilot
variance says the test is underpowered.

Public interface: :func:`run_suite`, :func:`aggregate`, :func:`family_analysis`,
:func:`dose_response`, :func:`main`.

Key dependencies: :mod:`_library` (artifacts, output chain), :mod:`_scoring`
(production-parity scoring, per-modality difference features), :mod:`_stats`.
"""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from exp.markov_sufficiency import _library, _scoring, _stats

DEFAULT_LIBRARY_ROOT = "/home/weiland/projects/openpi/exp/common/data/cache_artifacts"

# Primary cell (plan §3.1): production uses top_k=1, and cp1_spatial_pool_16 is
# the key builder behind both suites' strongest base config. Everything else in
# the 192-cell grid is exploratory.
PRIMARY_KEY_BUILDER = "cp1_spatial_pool_16"
PRIMARY_K = 1
PRIMARY_GROUPS = ("B-d3", "C-g1.0")

#: Effect floor and equivalence margin from plan §3.1 (relative residual).
EFFECT_FLOOR = 0.05
TOST_MARGIN = 0.03
#: Number of library-side pairs used to calibrate the difference blocks.
DIFF_CALIBRATION_PAIRS = 200

# ------------------------------------------------------------------
# Frames and candidate pools
# ------------------------------------------------------------------


def _history_frames(lib: _library.Library, entry: Any, depth: int) -> list[Optional[dict]]:
    """Newest-first query frames ``[k_t, k_{t-1}, ...]`` for one entry."""
    frames: list[Optional[dict]] = [entry.query_keys]
    for anc_id in _library.walk_ancestors(lib, entry.id, depth - 1):
        frames.append(None if anc_id is None else lib.by_id[anc_id].query_keys)
    return frames


def _progress(lib: _library.Library, entry: Any) -> float:
    """Normalised progress ``cycle / (n_cycles - 1)`` inside the entry's episode."""
    items = lib.by_traj[entry.trajectory_id]
    last = max(e.step_idx for e in items)
    return 0.0 if last <= 0 else entry.step_idx / last


def _knn_predict(scores: np.ndarray, candidates: Sequence[Any], out_chain, k: int) -> np.ndarray:
    """Similarity-weighted kNN aggregation, identical for every feature group.

    Groups A, B and C must share this so that a difference in residuals is a
    difference in the *feature set*, not in how neighbours were pooled.
    """
    order = np.argsort(-scores)[:k]
    acts = np.stack([_library.executed_action(candidates[i], out_chain=out_chain) for i in order])
    if k == 1:
        return acts[0]
    w = np.asarray(scores)[order].astype(np.float64)
    w = np.ones_like(w) if float(w.sum()) <= 0 else w / w.sum()
    return (acts * w[:, None]).sum(axis=0)


# ------------------------------------------------------------------
# Group C: per-modality difference scoring with a library-side calibration
# ------------------------------------------------------------------


def calibrate_diff_normalizers(
    scorer: _scoring.Scorer,
    lib: _library.Library,
    pool: Sequence[Any],
    held_out: str,
    n_pairs: int = DIFF_CALIBRATION_PAIRS,
    seed: int = 0,
) -> dict[str, _scoring.DiffNormalizer]:
    """Fit one difference normalizer per modality from **library-side pairs only**.

    The sample is built from pairs of pool entries, never from the held-out
    query: a scale fitted on held-out similarities leaks the fold, and the leak
    is invisible to a check that only looks at trajectory names.
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pool), size=(min(n_pairs, len(pool) * (len(pool) - 1) // 2), 2))
    samples: dict[str, list[float]] = {}
    sources: dict[str, list[str]] = {}
    for i, j in idx:
        if i == j:
            continue
        a, b = pool[int(i)], pool[int(j)]
        if a.trajectory_id == held_out or b.trajectory_id == held_out:
            continue  # defensive: the pool is already fold-clean
        sims = _scoring.diff_similarity_per_field(
            scorer, _history_frames(lib, a, 3), _history_frames(lib, b, 3)
        )
        for field, (d1, d2) in sims.items():
            samples.setdefault(field, []).extend([d1, d2])
            sources.setdefault(field, []).extend([a.trajectory_id, a.trajectory_id])

    fit_trajectories = sorted({e.trajectory_id for e in pool})
    return {
        field: _scoring.fit_diff_normalizer(
            values, fit_trajectories, held_out, source_trajectories=sources[field]
        )
        for field, values in samples.items()
        if values
    }


def score_group_c(
    scorer: _scoring.Scorer,
    lib: _library.Library,
    query_frames: Sequence[Optional[Mapping[str, np.ndarray]]],
    candidates: Sequence[Any],
    normalizers: Mapping[str, _scoring.DiffNormalizer],
    gamma: float,
    base_scores: np.ndarray,
) -> np.ndarray:
    """``base + sum_f w_f * gamma * (norm_f(delta) + norm_f(delta^2))``."""
    extra = np.zeros(len(candidates), dtype=np.float64)
    weights = {f: w for f, w, _ in scorer.active_fields}
    for ci, cand in enumerate(candidates):
        sims = _scoring.diff_similarity_per_field(scorer, query_frames, _history_frames(lib, cand, 3))
        acc = 0.0
        for field, (d1, d2) in sims.items():
            norm = normalizers.get(field)
            if norm is None:
                continue
            acc += weights.get(field, 0.0) * gamma * (norm(d1) + norm(d2))
        extra[ci] = acc
    return base_scores + extra


# ------------------------------------------------------------------
# Driver
# ------------------------------------------------------------------


def run_suite(
    suite: str,
    key_builder: str,
    scorer_by_depth: Mapping[int, _scoring.Scorer],
    library_root: str = DEFAULT_LIBRARY_ROOT,
    gammas: tuple[float, ...] = (0.5, 1.0),
    ks: tuple[int, ...] = (1, 5),
    oracle_eps: tuple[float, ...] = (0.05, 0.10),
    max_episodes: Optional[int] = None,
) -> dict[str, Any]:
    """Run LOEO over one (suite, key builder).

    ``scorer_by_depth`` must map every depth to the scorer built from *that
    depth's* production yaml: depth 3 and depth 5 ship different
    ``trajectory_weights``, and truncating one to serve the other would score a
    configuration that was never run.
    """
    depths = tuple(sorted(d for d in scorer_by_depth if d > 1))
    base_scorer = scorer_by_depth[1] if 1 in scorer_by_depth else scorer_by_depth[depths[0]]

    lib = _library.load_library(f"{library_root}/{suite}/{key_builder}.pkl")
    out_chain = _library.build_output_chain()

    rows: list[dict[str, Any]] = []
    traj_ids = sorted(lib.by_traj)
    if max_episodes is not None:
        traj_ids = traj_ids[:max_episodes]

    n_padding = 0
    for held_out in traj_ids:
        pool = [e for e in lib.entries if e.trajectory_id != held_out]
        normalizers = calibrate_diff_normalizers(base_scorer, lib, pool, held_out)
        cache: dict = {}
        for entry in lib.by_traj[held_out]:
            task_key = entry.payload.task_key
            cands = [e for e in pool if e.payload.task_key == task_key]
            if not cands:
                continue
            truth = _library.executed_action(entry, out_chain=out_chain)
            frames = _history_frames(lib, entry, max(depths, default=1) + 2)
            _, _, padding = _scoring.diff_features(frames[:3], base_scorer.active_fields[0][0])
            n_padding += int(padding)

            base = {
                "suite": suite,
                "key_builder": key_builder,
                "trajectory_id": held_out,
                "step_idx": entry.step_idx,
                "task_key": task_key,
                "padding": bool(padding),
            }
            base_scores = base_scorer.score_batch(entry.query_keys, cands, cache=cache)

            for k in ks:
                rows.append({**base, "group": "A", "k": k,
                             "residual": float(np.linalg.norm(_knn_predict(base_scores, cands, out_chain, k) - truth))})

                for d in depths:
                    scorer_d = scorer_by_depth[d]
                    weights = scorer_d.trajectory_weights
                    if weights is None or len(weights) != d:
                        raise ValueError(
                            f"depth {d} scorer must carry exactly {d} production trajectory weights, "
                            f"got {weights!r} from {scorer_d.yaml_path}"
                        )
                    q_hist = frames[:d]
                    traj_scores = np.array(
                        [
                            scorer_d.score_trajectory(
                                q_hist,
                                [c] + [None if a is None else lib.by_id[a] for a in _library.walk_ancestors(lib, c.id, d - 1)],
                                list(weights),
                            )
                            for c in cands
                        ]
                    )
                    rows.append({**base, "group": f"B-d{d}", "k": k,
                                 "residual": float(np.linalg.norm(_knn_predict(traj_scores, cands, out_chain, k) - truth))})

                for gamma in gammas:
                    c_scores = score_group_c(base_scorer, lib, frames[:3], cands, normalizers, gamma, base_scores)
                    rows.append({**base, "group": f"C-g{gamma}", "k": k,
                                 "residual": float(np.linalg.norm(_knn_predict(c_scores, cands, out_chain, k) - truth))})

                # E1-O: the same A and B contrast, restricted to phase-aligned
                # candidates. Running only A here would not test H-B at all.
                p_query = _progress(lib, entry)
                for eps in oracle_eps:
                    keep = [i for i, c in enumerate(cands) if abs(_progress(lib, c) - p_query) <= eps]
                    if not keep:
                        continue
                    aligned = [cands[i] for i in keep]
                    rows.append({**base, "group": f"O-A-e{eps}", "k": k,
                                 "residual": float(np.linalg.norm(_knn_predict(base_scores[keep], aligned, out_chain, k) - truth))})
                    for d in depths:
                        scorer_d = scorer_by_depth[d]
                        q_hist = frames[:d]
                        traj_scores = np.array(
                            [
                                scorer_d.score_trajectory(
                                    q_hist,
                                    [c] + [None if a is None else lib.by_id[a] for a in _library.walk_ancestors(lib, c.id, d - 1)],
                                    list(scorer_d.trajectory_weights),
                                )
                                for c in aligned
                            ]
                        )
                        rows.append({**base, "group": f"O-B-d{d}-e{eps}", "k": k,
                                     "residual": float(np.linalg.norm(_knn_predict(traj_scores, aligned, out_chain, k) - truth))})

    manifest = {
        "suite": suite,
        "key_builder": key_builder,
        "library_meta": lib.meta,
        "yaml_by_depth": {str(d): s.yaml_path for d, s in scorer_by_depth.items()},
        "n_episodes": len(traj_ids),
        "n_rows": len(rows),
        "padding_steps": n_padding,
        "diff_calibration_pairs": DIFF_CALIBRATION_PAIRS,
    }
    return {"rows": rows, "manifest": manifest}


# ------------------------------------------------------------------
# Episode-level aggregation and the registered family verdict
# ------------------------------------------------------------------


def aggregate(rows: list[dict[str, Any]], group: str, k: int, include_padding: bool = False) -> dict[str, Any]:
    """Episode-level median residuals for group A and ``group``, then the paired test."""
    per_ep: dict[str, dict[str, list[float]]] = {}
    for r in rows:
        if r["k"] != k or (r["padding"] and not include_padding):
            continue
        if r["group"] not in ("A", group):
            continue
        per_ep.setdefault(r["trajectory_id"], {}).setdefault(r["group"], []).append(r["residual"])

    diffs, med_a = [], []
    for groups in per_ep.values():
        if "A" not in groups or group not in groups:
            continue
        a = float(np.median(groups["A"]))
        x = float(np.median(groups[group]))
        diffs.append(a - x)
        med_a.append(a)

    test = _stats.cluster_sign_permutation(diffs)
    baseline = float(np.median(med_a)) if med_a else float("nan")
    # Hodges-Lehmann is the median of Walsh averages, not the median of the
    # differences, and its interval comes from inverting the same permutation
    # test that produced the p-value.
    hl = _stats.hodges_lehmann(diffs)
    ci = _stats.sign_permutation_ci(diffs)
    return {
        "group": group,
        "k": k,
        "n_episodes": len(diffs),
        "median_residual_A": baseline,
        "hodges_lehmann": hl,
        "relative_delta": (hl / baseline) if baseline else float("nan"),
        "p_value": test.p_value,
        "hl_ci": [ci.low, ci.high],
        "paired_diffs": diffs,
    }


def family_analysis(
    per_suite_rows: Mapping[str, list[dict[str, Any]]],
    groups: Sequence[str] = PRIMARY_GROUPS,
    k: int = PRIMARY_K,
    alpha: float = 0.05,
    pilot_power_by_cell: Mapping[tuple[str, str], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """The four primary cells: Holm across suites x groups, plus TOST.

    Everything outside this family is exploratory and reported without a
    p-value verdict.
    """
    cells = []
    for suite, rows in sorted(per_suite_rows.items()):
        for group in groups:
            cell = aggregate(rows, group, k)
            cell["suite"] = suite
            cells.append(cell)

    p_values = [c["p_value"] for c in cells]
    rejected = _stats.holm(p_values, alpha=alpha)
    levels = _stats.holm_adjusted_levels(p_values, alpha=alpha)

    for cell, rej, level in zip(cells, rejected, levels):
        adj = _stats.sign_permutation_ci(cell["paired_diffs"], level=level)
        cell["holm_rejected"] = bool(rej)
        cell["holm_level"] = level
        cell["holm_ci"] = [adj.low, adj.high]
        base = cell["median_residual_A"]
        margin = TOST_MARGIN * base if base == base else float("nan")
        # Equivalence only when the adjusted interval sits inside +-margin;
        # a non-rejection on its own never establishes it.
        cell["tost_equivalent"] = bool(adj.low > -margin and adj.high < margin) if margin == margin else False
        power = pilot_power_by_cell.get((cell["suite"], cell["group"])) if pilot_power_by_cell else None
        cell["pilot_power"] = power
        cell["verdict"] = "estimation_only_underpowered" if (power and power["underpowered"]) else _cell_verdict(cell)
        cell.pop("paired_diffs", None)
    return {"alpha": alpha, "k": k, "cells": cells}


def pilot_power(rows: list[dict[str, Any]], group: str, k: int = PRIMARY_K, effect: float = EFFECT_FLOOR) -> dict[str, Any]:
    """Episode-level pilot power for the registered effect floor.

    Plan §3.1 requires the primary verdict to degrade to estimation when the
    cluster-level test is underpowered, so the number has to be produced by the
    driver rather than assumed.
    """
    cell = aggregate(rows, group, k)
    diffs = np.asarray(cell["paired_diffs"], dtype=np.float64)
    n = diffs.size
    baseline = cell["median_residual_A"]
    if n < 3 or not (baseline == baseline) or baseline == 0:
        return {"n_episodes": int(n), "power": float("nan"), "underpowered": True}
    sd = float(np.std(diffs, ddof=1))
    if sd <= 0:
        return {"n_episodes": int(n), "power": 1.0, "underpowered": False}
    # Normal approximation to the sign-flip test at the registered effect size.
    z = (effect * baseline) / (sd / np.sqrt(n))
    power = float(_stats._phi(z - 1.959964) + _stats._phi(-z - 1.959964))
    return {"n_episodes": int(n), "power": power, "underpowered": bool(power < 0.8)}


def _cell_verdict(cell: dict[str, Any]) -> str:
    if cell["holm_rejected"] and cell["relative_delta"] >= EFFECT_FLOOR:
        return "history_helps"
    if cell["tost_equivalent"]:
        return "equivalent_no_history_value"
    return "inconclusive"


def dose_response(
    by_key_builder: Mapping[str, list[dict[str, Any]]],
    group: str,
    k: int = PRIMARY_K,
    *,
    fold_assignment: Mapping[str, int] | None = None,
    swap: bool = False,
) -> dict[str, Any]:
    """Cross-fitted key-quality vs history-gain trend (exploratory, n = builders).

    ``x`` (key quality) and ``y`` (relative gain) are computed on disjoint
    episode halves so they do not share the same ``r_A`` term; sharing it would
    manufacture correlation independent of the hypothesis.

    ``fold_assignment`` maps ``trajectory_id -> 0 | 1`` so the caller can supply
    the plan's task-stratified random split; without it the halves fall back to
    a deterministic split of the sorted episode ids. ``swap`` exchanges the two
    folds, which is how the plan's "run it again with the folds reversed and
    report both" is obtained without re-running LOEO.
    """
    xs, ys = [], []
    for builder, rows in sorted(by_key_builder.items()):
        episodes = sorted({r["trajectory_id"] for r in rows})
        if fold_assignment is not None:
            fold1 = {e for e in episodes if fold_assignment.get(e, 0) == 0}
            fold2 = {e for e in episodes if fold_assignment.get(e, 0) == 1}
        else:
            half = len(episodes) // 2
            fold1, fold2 = set(episodes[:half]), set(episodes[half:])
        if swap:
            fold1, fold2 = fold2, fold1
        a1 = aggregate([r for r in rows if r["trajectory_id"] in fold1], group, k)
        a2 = aggregate([r for r in rows if r["trajectory_id"] in fold2], group, k)
        if a1["median_residual_A"] != a1["median_residual_A"] or a2["relative_delta"] != a2["relative_delta"]:
            continue
        xs.append(a1["median_residual_A"])
        ys.append(a2["relative_delta"])
    if len(xs) < 3:
        return {"n": len(xs), "spearman": float("nan"), "p_value": float("nan")}

    rho = _spearman(xs, ys)
    rng = np.random.default_rng(0)
    perm = np.array([_spearman(xs, list(rng.permutation(ys))) for _ in range(10_000)])
    return {"n": len(xs), "spearman": rho, "p_value": float((np.sum(np.abs(perm) >= abs(rho)) + 1) / 10_001)}


def _spearman(x: Sequence[float], y: Sequence[float]) -> float:
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = float(np.linalg.norm(rx) * np.linalg.norm(ry))
    return 0.0 if denom == 0 else float(np.dot(rx, ry) / denom)


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------


def _parse_yaml_by_depth(spec: str) -> dict[int, str]:
    """Parse ``1=path,3=path,5=path`` into a depth -> yaml mapping."""
    out: dict[int, str] = {}
    for chunk in spec.split(","):
        depth, _, path = chunk.partition("=")
        out[int(depth)] = path
    if not out:
        raise ValueError("expected at least one <depth>=<yaml> pair")
    return out


def write_rows(rows: list[dict[str, Any]], out_base: pathlib.Path) -> str:
    """Write the per-step rows as parquet when possible, else JSONL.

    The plan registers a parquet artifact; pandas/pyarrow are not guaranteed in
    every project venv, so the fallback is recorded in the manifest rather than
    silently changing the deliverable.
    """
    try:
        import pandas as pd

        path = out_base.with_suffix(".parquet")
        pd.DataFrame(rows).to_parquet(path, index=False)
        return str(path)
    except Exception:
        path = out_base.with_suffix(".jsonl")
        with path.open("w") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
        return str(path)


def run_family(
    suite_specs: Mapping[str, Mapping[int, str]],
    out_dir: pathlib.Path,
    key_builder: str = PRIMARY_KEY_BUILDER,
    library_root: str = DEFAULT_LIBRARY_ROOT,
    max_episodes: Optional[int] = None,
) -> dict[str, Any]:
    """Registered E1 delivery: both suites, pilot power, family verdict, artifacts.

    This is the only path that may emit a family verdict. Pilot power is
    computed for all four primary cells and fed into ``family_analysis`` so an
    underpowered cell degrades to estimation automatically instead of relying
    on the caller to remember.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    per_suite_rows: dict[str, list[dict[str, Any]]] = {}
    manifests: dict[str, Any] = {}
    artifacts: dict[str, str] = {}

    for suite, yaml_by_depth in suite_specs.items():
        scorers = {d: _scoring.build_scorer(p) for d, p in yaml_by_depth.items()}
        result = run_suite(suite, key_builder, scorers, library_root=library_root, max_episodes=max_episodes)
        per_suite_rows[suite] = result["rows"]
        manifests[suite] = result["manifest"]
        artifacts[suite] = write_rows(result["rows"], out_dir / f"e1_{suite}_{key_builder}")

    pilot = {
        (suite, group): pilot_power(rows, group, PRIMARY_K)
        for suite, rows in per_suite_rows.items()
        for group in PRIMARY_GROUPS
    }
    family = family_analysis(per_suite_rows, pilot_power_by_cell=pilot)

    manifest = {
        "suites": manifests,
        "row_artifacts": artifacts,
        "pilot_power": {f"{s}|{g}": v for (s, g), v in pilot.items()},
        "primary_groups": list(PRIMARY_GROUPS),
        "primary_k": PRIMARY_K,
        "effect_floor": EFFECT_FLOOR,
        "tost_margin": TOST_MARGIN,
    }
    manifest_path = out_dir / "e1_manifest.json"
    with manifest_path.open("w") as fh:
        json.dump(manifest, fh, indent=2)

    family_path = out_dir / "e1_family.json"
    with family_path.open("w") as fh:
        json.dump(family, fh, indent=2)
    return {"family": family, "manifest": manifest, "manifest_path": str(manifest_path),
            "family_path": str(family_path)}


def main() -> None:
    ap = argparse.ArgumentParser(description="E1 / E1-O LOEO residuals (two-suite registered driver)")
    ap.add_argument(
        "--suite-yaml",
        action="append",
        required=True,
        help="repeatable: '<suite>:<depth>=<yaml>[,<depth>=<yaml>...]'; both suites are required",
    )
    ap.add_argument("--key-builder", default=PRIMARY_KEY_BUILDER)
    ap.add_argument("--library-root", default=DEFAULT_LIBRARY_ROOT)
    ap.add_argument("--max-episodes", type=int, default=None, help="smoke runs only")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    specs: dict[str, dict[int, str]] = {}
    for chunk in args.suite_yaml:
        suite, _, rest = chunk.partition(":")
        specs[suite] = _parse_yaml_by_depth(rest)
    if len(specs) < 2:
        raise SystemExit(
            "the registered E1 verdict is a two-suite Holm family; pass --suite-yaml for each suite"
        )

    result = run_family(
        specs,
        pathlib.Path(args.out_dir),
        key_builder=args.key_builder,
        library_root=args.library_root,
        max_episodes=args.max_episodes,
    )
    print(json.dumps(result["family"], indent=2))


if __name__ == "__main__":
    main()
