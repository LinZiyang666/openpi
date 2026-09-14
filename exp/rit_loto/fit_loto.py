"""Fit the RIT ladders on the LOTO table and compare them with the deployed shadow fit.

Four fits, one audit, all through the unchanged ``emit_rit_rc.fit_ladders`` LP:

  loto_all             every LOTO row (frozen ``fit_source`` of the verification arm)
  loto_in_library      rows whose trajectory sits in the library (searched with it left out)
  loto_out_of_library  rows whose trajectory never entered the library (full library)
  shadow               the deployed rollout calibration, re-fitted from ``shadow_rows.jsonl``
  audit                the shadow re-fit must reproduce ``arm_record.json`` knot for knot and
                       cut for cut (``+inf`` <-> JSON ``null``); otherwise the "original
                       curve" on the comparison figure would not be the deployed one.

Outputs (all under ``--out-dir``, run products per the artifact layout):
  fits.json            per source and per K: knots, per-tier knot values, tiers with
                       measured cost, alpha, eps_total, segment counts, row counts, score
                       quantiles, attainable IR range, and the addressing score sample --
                       enough to rebuild the ``PLFitK`` without refitting
  compare.json         exact max |q_a - q_b| on the common support (end points and the
                       union of both knot sets), per-segment differences, score marginals,
                       the engineering scale tau_a and the row-count table
  bootstrap_band.json  pointwise reference variability of the shadow fit under a
                       task-stratified episode bootstrap (descriptive, not a test)

Main venv (numpy / scipy / HiGHS); no model, no GPU. The parity gate of the
table must be an explicit PASS bound to the table's identity or nothing is fitted.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import pathlib
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import numpy as np

from exp.libero_groot.emit_rit_arms import DEFAULT_ALPHA, WARM_TS, load_cost
from exp.rit_pareto.rit_k import PLFitK, fit_from_record, fit_record_fields, predict
from exp.robocasa365 import emit_rit_rc as er
from exp.robocasa365 import rit_cost_rc as rc

from exp.rit_loto.build_loto_table import read_jsonl, require_code_identity, require_pass_gate, sha256_file

LEGACY_ALPHA = float(DEFAULT_ALPHA)
Y_KEYS = ("y_full", "y_rem2", "y_rem4")
DEFAULT_KS = (1, 2, 3)
SOURCES = ("loto_all", "loto_in_library", "loto_out_of_library", "shadow")
DEFAULT_N_BOOT = 200
DEFAULT_MIN_VALID_FRACTION = 0.9


# ------------------------------------------------------------------
# Rows
# ------------------------------------------------------------------


def load_rows(path: str | pathlib.Path, *, in_library: bool | None = None) -> list[dict]:
    """LOTO rows, optionally restricted to in-library or out-of-library trajectories."""
    rows = read_jsonl(path)
    if in_library is not None:
        rows = [r for r in rows if bool(r.get("in_library")) == in_library]
    return rows


def validate_rows(rows: list[dict], y_keys: tuple[str, ...] = Y_KEYS) -> None:
    """Refuse missing or non-finite calibration values instead of dropping them silently."""
    bad: dict[str, int] = {}
    for r in rows:
        for key in ("s", *y_keys):
            v = r.get(key)
            if v is None or not math.isfinite(float(v)):
                bad[key] = bad.get(key, 0) + 1
    if bad:
        raise SystemExit(f"rows with missing / non-finite values: {bad}")


def scores(rows: list[dict]) -> np.ndarray:
    """The retrieval scores of ``rows`` as a float64 array."""
    return np.array([float(r["s"]) for r in rows], dtype=np.float64)


def validate_table_record(rows: list[dict], record: dict) -> None:
    """Require the complete 500-episode table before fitting the formal calibration."""
    stats, ident = record.get("stats", {}), record.get("identity", {})
    if record.get("protocol") != "rit_loto_table_v1" or record.get("smoke") is not False:
        raise SystemExit("refusing an unknown or smoke table record")
    if ident.get("corpus_files") != 500 or stats.get("episodes") != 500 or stats.get("rows") != len(rows):
        raise SystemExit("formal calibration requires the complete 500-episode table; a task shard is incomplete")
    seen, episodes, coordinates = set(), {}, {}
    for row in rows:
        key = (row["trajectory_id"], int(row["decision_id"]))
        if key in seen or row.get("suite") != ident.get("suite") or type(row.get("in_library")) is not bool:
            raise SystemExit("calibration rows contain duplicate keys or inconsistent suite/membership")
        seen.add(key)
        coord = (int(row["task_id"]), int(row["orig_init_state_idx"]))
        trajectory = row["trajectory_id"]
        if trajectory in coordinates and coordinates[trajectory] != coord:
            raise SystemExit("one calibration trajectory has inconsistent task/init coordinates")
        if coord in episodes and episodes[coord] != trajectory:
            raise SystemExit("calibration table repeats an episode under another trajectory id")
        coordinates[trajectory], episodes[coord] = coord, trajectory
    if set(episodes) != {(t, i) for t in range(10) for i in range(50)}:
        raise SystemExit("calibration table does not represent all 10 x 50 task/init episodes")


# ------------------------------------------------------------------
# Fits and their serialisation
# ------------------------------------------------------------------


def _quantiles(s: np.ndarray) -> dict[str, float]:
    return {str(q): float(np.quantile(s, q)) for q in (0.0, 0.05, 0.15, 0.5, 0.85, 0.95, 1.0)}


def fit_source(rows: list[dict], cost: rc.StageCost, *, source: str, input_sha256: str,
               warm_ts: tuple[float, ...] = WARM_TS, ks: tuple[int, ...] = DEFAULT_KS,
               alpha: float = LEGACY_ALPHA) -> dict:
    """Fit every K on ``rows``; a knot-ladder stop-loss is recorded as ``fit_unavailable``, never faked."""
    out: dict[str, Any] = {"source": source, "input_sha256": input_sha256, "n_rows": len(rows),
                           "alpha": float(alpha), "warm_ts": [float(t) for t in warm_ts], "fits": {}}
    if not rows:
        out["fit_unavailable"] = True
        out["reason"] = "no rows"
        return out
    s = scores(rows)
    out["s_quantiles"] = _quantiles(s)
    out["s_sample"] = s.tolist()
    try:
        fits = er.fit_ladders(rows, cost, list(warm_ts), list(ks), float(alpha), ir_sample=s)
    except SystemExit as exc:
        out["fit_unavailable"] = True
        out["reason"] = str(exc)
        return out
    for k, blob in fits.items():
        rec = fit_record_fields(blob["fit"])
        rec.update({"k": int(k), "n_rows": int(blob["n_rows"]), "n_seg_req": int(blob["n_seg_req"]),
                    "ir_range": [float(blob["ir_range"][0]), float(blob["ir_range"][1])],
                    "s_range": [float(blob["s"].min()), float(blob["s"].max())]})
        out["fits"][str(k)] = rec
    return out


def deserialize_fit(rec: dict, cost: rc.StageCost, warm_ts: tuple[float, ...] = WARM_TS) -> PLFitK:
    """A frozen fit read back with RoboCasa-priced tiers, so IR addressing prices it correctly.

    ``fit_from_record`` restores pi0.5 ``Tier`` objects whose ``cost_ms`` is the
    LIBERO x pi0.5 constant; ``rit_cost_rc`` prices ``fit.tiers`` directly, so the
    ladder is rebuilt from the measured cost and checked against the complete tier metadata.
    """
    fit = fit_from_record(rec)
    tiers = rc.ladder(cost, list(warm_ts[: int(rec["k"]) - 1]))
    recorded = [t["name"] for t in rec["tiers"]]
    if [t.name for t in tiers] != recorded:
        raise ValueError(f"tier names {recorded} do not match the ladder {[t.name for t in tiers]}")
    for old, tier in zip(rec["tiers"], tiers):
        expected = dataclasses.asdict(tier) | {"cost_ms": tier.cost_ms}
        if old != expected:
            raise ValueError(f"tier metadata differs from the measured ladder: {tier.name}")
    knots = np.asarray(fit.knots)
    if knots.ndim != 1 or len(knots) < 2 or not np.isfinite(knots).all() or not (np.diff(knots) > 0).all():
        raise ValueError("fit knots must be finite and strictly increasing")
    if not math.isfinite(fit.alpha) or not 0 < fit.alpha < 1 or not math.isfinite(fit.eps_total) or fit.eps_total <= 0:
        raise ValueError("invalid fit alpha/eps_total")
    for values in fit.q.values():
        if values.shape != knots.shape or not np.isfinite(values).all():
            raise ValueError("curve ordinates must be finite and match the knot grid")
    return dataclasses.replace(fit, tiers=tiers)


# ------------------------------------------------------------------
# Shadow re-fit audit
# ------------------------------------------------------------------


def _cut_matches(recorded, mine: float, tol: float = 1e-9) -> bool:
    if isinstance(mine, float) and math.isnan(mine):
        raise ValueError("cut is NaN")
    if mine == -math.inf:
        raise ValueError("cut is -inf")
    if recorded is None:
        return mine == math.inf
    recorded = float(recorded)
    if not math.isfinite(recorded):
        raise ValueError(f"recorded cut is not finite: {recorded!r}")
    return math.isfinite(mine) and abs(mine - recorded) <= tol


def refit_shadow_and_check(rows: list[dict], cost: rc.StageCost, arm_record: dict, *, table_sha: str,
                           template_sha: str, legacy_alpha: float = LEGACY_ALPHA) -> dict:
    """Re-fit the deployed shadow table and prove it is the deployed curve, or stop."""
    if float(legacy_alpha) != LEGACY_ALPHA:
        raise SystemExit(f"legacy_alpha must equal the frozen {LEGACY_ALPHA}")
    problems: list[str] = []
    if arm_record.get("shadow_sha256") != table_sha:
        problems.append(f"shadow table sha {table_sha[:12]} != record {str(arm_record.get('shadow_sha256'))[:12]}")
    if arm_record.get("template_sha256") != template_sha:
        problems.append(f"template sha {template_sha[:12]} != record {str(arm_record.get('template_sha256'))[:12]}")
    warm_ts = tuple(round(float(t), 4) for t in arm_record.get("warm_ts", []))
    if warm_ts != tuple(WARM_TS):
        problems.append(f"record warm_ts {warm_ts} != {WARM_TS}")
    if problems:
        raise SystemExit("shadow audit identity failed: " + "; ".join(problems))
    ks = sorted(int(k) for k in arm_record["fits"])
    if tuple(ks) != tuple(DEFAULT_KS):
        raise SystemExit(f"arm_record fits K set {ks} != frozen {list(DEFAULT_KS)}")
    if "alpha" in arm_record:
        alpha, alpha_source = float(arm_record["alpha"]), "record"
        if alpha != float(legacy_alpha):
            raise SystemExit(f"arm_record alpha {alpha} != frozen {legacy_alpha}: a self-consistent record at "
                             "another quantile level is not the deployed curve family")
    else:
        alpha, alpha_source = float(legacy_alpha), "legacy_reconstruction"
    validate_rows(rows)
    s = scores(rows)
    fits = er.fit_ladders(rows, cost, list(warm_ts), ks, alpha, ir_sample=s)
    for k in ks:
        rec = arm_record["fits"][str(k)]
        mine = np.asarray(fits[k]["knots"], dtype=np.float64)
        theirs = np.asarray(rec["knots"], dtype=np.float64)
        if mine.shape != theirs.shape or not np.array_equal(mine, theirs):
            raise SystemExit(f"k={k}: re-fitted knots differ from arm_record ({mine.shape} vs {theirs.shape})")
        names = [t.name for t in fits[k]["tiers"]]
        expected_tiers = [t.name for t in rc.ladder(cost, list(warm_ts[: k - 1]))]
        if names != list(rec["tiers"]) or names != expected_tiers:
            raise SystemExit(f"k={k}: tier names {names} != record {rec['tiers']} / ladder {expected_tiers}")
        y_keys = [t.y_key for t in fits[k]["tiers"]]
        if y_keys != list(Y_KEYS[:k]):
            raise SystemExit(f"k={k}: tier y_keys {y_keys} != frozen {list(Y_KEYS[:k])}")
    n_arms, n_cuts = 0, 0
    for name, arm in sorted(arm_record["arms"].items()):
        if arm.get("rule") != "rit":
            continue
        k = int(arm["k"])
        mine = rc.cuts_for(fits[k]["fit"], float(arm["delta"]))
        recorded = list(arm["cuts"])
        if len(mine) != len(recorded):
            raise SystemExit(f"{name}: {len(mine)} cuts vs {len(recorded)} recorded")
        for i, (r, m) in enumerate(zip(recorded, mine)):
            if not _cut_matches(r, float(m)):
                raise SystemExit(f"{name}: cut[{i}] recorded {r!r} vs re-fitted {m!r}")
            n_cuts += 1
        n_arms += 1
    if n_arms == 0:
        raise SystemExit("arm_record carries no rit arms to audit against")
    return {"alpha": alpha, "alpha_source": alpha_source, "ks": ks, "n_arms_checked": n_arms,
            "n_cuts_checked": n_cuts, "n_rows": len(rows), "fits": fits}


# ------------------------------------------------------------------
# Curve comparison
# ------------------------------------------------------------------


def common_support(fit_a: PLFitK, fit_b: PLFitK) -> tuple[float, float] | None:
    """Intersection of the two fitted knot ranges, or None when they do not overlap."""
    lo = max(float(fit_a.knots[0]), float(fit_b.knots[0]))
    hi = min(float(fit_a.knots[-1]), float(fit_b.knots[-1]))
    return (lo, hi) if lo < hi else None


def curve_max_diff(fit_a: PLFitK, fit_b: PLFitK, tier: str, s_lo: float, s_hi: float) -> dict:
    """Exact ``max |q_a - q_b|`` on ``[s_lo, s_hi]``: both curves are piecewise linear, so the
    difference is too and its extremum sits on an end point or a knot of either fit."""
    if s_lo >= s_hi:
        raise ValueError("empty support")
    pts = {float(s_lo), float(s_hi)}
    for fit in (fit_a, fit_b):
        pts.update(float(k) for k in fit.knots if s_lo <= float(k) <= s_hi)
    grid = np.array(sorted(pts), dtype=np.float64)
    d = predict(fit_a, grid, tier) - predict(fit_b, grid, tier)
    i = int(np.argmax(np.abs(d)))
    segments = []
    kb = np.asarray(fit_b.knots, dtype=np.float64)
    for j in range(len(kb) - 1):
        lo, hi = max(float(kb[j]), s_lo), min(float(kb[j + 1]), s_hi)
        if lo >= hi:
            continue
        m = (grid >= lo) & (grid <= hi)
        segments.append({"s_lo": lo, "s_hi": hi, "max_abs": float(np.abs(d[m]).max()),
                         "mean_signed": float(d[m].mean())})
    return {"max_abs": float(abs(d[i])), "at_s": float(grid[i]), "signed_at_max": float(d[i]),
            "n_points": int(grid.size), "support": [float(s_lo), float(s_hi)], "segments": segments}


def score_marginal(s_a: np.ndarray, s_b: np.ndarray) -> dict:
    """Quantiles, mass at s >= 0.99 and the two-sample KS statistic of two score samples."""
    from scipy.stats import ks_2samp

    s_a, s_b = np.asarray(s_a, dtype=np.float64), np.asarray(s_b, dtype=np.float64)
    ks = ks_2samp(s_a, s_b) if s_a.size and s_b.size else None
    return {
        "a": {"n": int(s_a.size), **_quantiles(s_a), "mass_ge_0.99": float(np.mean(s_a >= 0.99))} if s_a.size else {"n": 0},
        "b": {"n": int(s_b.size), **_quantiles(s_b), "mass_ge_0.99": float(np.mean(s_b >= 0.99))} if s_b.size else {"n": 0},
        "ks_statistic": None if ks is None else float(ks.statistic),
        "ks_pvalue": None if ks is None else float(ks.pvalue),
    }


def tau_scale(rows: list[dict], fit: PLFitK, tier: str, y_key: str) -> float:
    """Engineering scale: p90 of |D - q_hat(s)| on the rows the reference curve was fitted on."""
    s = scores(rows)
    y = np.array([float(r[y_key]) for r in rows], dtype=np.float64)
    return float(np.quantile(np.abs(y - predict(fit, s, tier)), 0.9))


# ------------------------------------------------------------------
# Episode bootstrap of the shadow fit (descriptive band)
# ------------------------------------------------------------------


def _group_episodes(rows: list[dict]) -> dict[str, dict[Any, list[dict]]]:
    by_task: dict[str, dict[Any, list[dict]]] = {}
    for r in rows:
        by_task.setdefault(str(r.get("task", "")), {}).setdefault(r.get("episode_id"), []).append(r)
    return by_task


def _boot_one(args: tuple) -> dict:
    rows, cost_json, warm_ts, k, alpha, grid, seed, b = args
    cost = _cost_from_json(cost_json)
    by_task = _group_episodes(rows)
    rng = np.random.default_rng([int(seed), int(b)])
    sample: list[dict] = []
    multiplicity: dict[str, int] = {}
    for task in sorted(by_task):
        eps = sorted(by_task[task], key=str)
        draw = rng.choice(len(eps), size=len(eps), replace=True)
        for i in draw.tolist():
            multiplicity[f"{task}|{eps[i]}"] = multiplicity.get(f"{task}|{eps[i]}", 0) + 1
            sample.extend(by_task[task][eps[i]])
    s = scores(sample)
    try:
        fits = er.fit_ladders(sample, cost, list(warm_ts), [int(k)], float(alpha), ir_sample=s)
    except SystemExit as exc:
        return {"b": int(b), "ok": False, "reason": str(exc), "n_rows": len(sample)}
    fit = fits[int(k)]["fit"]
    g = np.asarray(grid, dtype=np.float64)
    inside = (g >= float(fit.knots[0])) & (g <= float(fit.knots[-1]))
    pred = {}
    for t in fit.tiers:
        vals = predict(fit, g, t.name)
        vals = np.where(inside, vals, np.nan)
        pred[t.name] = vals.tolist()
    return {"b": int(b), "ok": True, "n_rows": len(sample), "n_distinct_episodes": len(multiplicity),
            "support": [float(fit.knots[0]), float(fit.knots[-1])], "pred": pred}


def _cost_to_json(cost: rc.StageCost) -> dict:
    return {"teacher": cost.teacher, "num_steps": cost.schedule.num_steps, "schedule_id": cost.schedule.schedule_id,
            "stage1_ms": cost.stage1_ms, "stage2_ms": cost.stage2_ms, "stage3_head_ms": cost.stage3_head_ms,
            "stage3_step_ms": cost.stage3_step_ms, "provenance": cost.provenance, "linear_stage3": cost.linear_stage3}


def _cost_from_json(d: dict) -> rc.StageCost:
    from openpi.cache.types import groot_n15_schedule

    return rc.StageCost(teacher=d["teacher"], schedule=groot_n15_schedule(int(d["num_steps"])),
                        stage1_ms=float(d["stage1_ms"]), stage2_ms=float(d["stage2_ms"]),
                        stage3_head_ms=float(d["stage3_head_ms"]), stage3_step_ms=float(d["stage3_step_ms"]),
                        provenance=str(d["provenance"]), linear_stage3=bool(d.get("linear_stage3", False)))


def episode_bootstrap_band(rows: list[dict], cost: rc.StageCost, warm_ts: tuple[float, ...], k: int,
                           alpha: float, grid: list[float], *, n_boot: int = DEFAULT_N_BOOT, seed: int = 0,
                           jobs: int = 1, min_valid_fraction: float = DEFAULT_MIN_VALID_FRACTION) -> dict:
    """Task-stratified episode bootstrap of one fit: pointwise 2.5 / 97.5 quantiles where enough
    replicates cover the point. Failed replicates are recorded, never redrawn."""
    cost_json = _cost_to_json(cost)
    work = [(rows, cost_json, tuple(warm_ts), int(k), float(alpha), list(grid), int(seed), b) for b in range(int(n_boot))]
    if jobs > 1:
        with ProcessPoolExecutor(max_workers=int(jobs)) as pool:
            results = list(pool.map(_boot_one, work, chunksize=1))
    else:
        results = [_boot_one(w) for w in work]
    ok = [r for r in results if r["ok"]]
    failures = [r for r in results if not r["ok"]]
    tiers = sorted({name for r in ok for name in r["pred"]})
    g = np.asarray(grid, dtype=np.float64)
    min_valid = int(math.ceil(min_valid_fraction * n_boot))
    out: dict[str, Any] = {"k": int(k), "alpha": float(alpha), "n_boot": int(n_boot), "seed": int(seed),
                           "grid": g.tolist(), "min_valid": min_valid, "n_ok": len(ok), "n_failures": len(failures),
                           "failures": failures, "tiers": {}}
    for name in tiers:
        mat = np.array([r["pred"][name] for r in ok], dtype=np.float64) if ok else np.empty((0, g.size))
        valid = np.isfinite(mat).sum(axis=0) if ok else np.zeros(g.size, dtype=int)
        lo, hi, med = [], [], []
        for j in range(g.size):
            col = mat[:, j][np.isfinite(mat[:, j])] if ok else np.array([])
            if col.size >= min_valid:
                lo.append(float(np.quantile(col, 0.025)))
                hi.append(float(np.quantile(col, 0.975)))
                med.append(float(np.quantile(col, 0.5)))
            else:
                lo.append(None)
                hi.append(None)
                med.append(None)
        out["tiers"][name] = {"lo": lo, "hi": hi, "median": med, "valid_per_point": valid.tolist()}
    return out


def display_grid(s_a: np.ndarray, s_b: np.ndarray, n: int) -> list[float]:
    """Empirical-quantile grid over both score samples (unique, sorted)."""
    pooled = np.concatenate([np.asarray(s_a, dtype=np.float64), np.asarray(s_b, dtype=np.float64)])
    return np.unique(np.quantile(pooled, np.linspace(0.0, 1.0, int(n)))).tolist()


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------


def _counts_table(rows: list[dict]) -> dict:
    table: dict[str, int] = {}
    for r in rows:
        key = f"in_library={bool(r.get('in_library'))},success={bool(r.get('episode_success'))}"
        table[key] = table.get(key, 0) + 1
    return table


def _compare_pair(name_a: str, name_b: str, fits_a: dict, fits_b: dict, rows_b: list[dict] | None, cost, k: int,
                  warm_ts) -> dict:
    rec_a, rec_b = fits_a["fits"].get(str(k)), fits_b["fits"].get(str(k))
    if rec_a is None or rec_b is None:
        return {"pair": [name_a, name_b], "k": k, "unavailable": True}
    fit_a, fit_b = deserialize_fit(rec_a, cost, warm_ts), deserialize_fit(rec_b, cost, warm_ts)
    sup = common_support(fit_a, fit_b)
    if sup is None:
        return {"pair": [name_a, name_b], "k": k, "no_common_support": True}
    s_a, s_b = np.asarray(fits_a["s_sample"]), np.asarray(fits_b["s_sample"])
    out = {"pair": [name_a, name_b], "k": k, "support": list(sup),
           "rows_outside_support": {name_a: int(((s_a < sup[0]) | (s_a > sup[1])).sum()),
                                    name_b: int(((s_b < sup[0]) | (s_b > sup[1])).sum())},
           "tiers": {}}
    for t in fit_b.tiers:
        diff = curve_max_diff(fit_a, fit_b, t.name, sup[0], sup[1])
        if rows_b is not None:
            tau = tau_scale(rows_b, fit_b, t.name, t.y_key)
            diff["tau_b"] = tau
            diff["max_abs_over_tau"] = None if tau == 0.0 else diff["max_abs"] / tau
            diff["tau_zero"] = tau == 0.0
        out["tiers"][t.name] = diff
    return out


def main() -> None:
    """CLI: gate-bound fits, shadow audit, curve comparison and the descriptive bootstrap band."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", required=True)
    ap.add_argument("--loto-table", required=True)
    ap.add_argument("--loto-record", required=True)
    ap.add_argument("--parity-gate", required=True)
    ap.add_argument("--shadow-rows", required=True)
    ap.add_argument("--arm-record", required=True)
    ap.add_argument("--template-yaml", required=True)
    ap.add_argument("--cost", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--ks", default="1,2,3")
    ap.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT)
    ap.add_argument("--boot-seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--grid-points", type=int, default=200)
    ap.add_argument("--skip-bootstrap", action="store_true")
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ks = tuple(int(x) for x in args.ks.split(",") if x.strip())
    if len(ks) != len(DEFAULT_KS) or set(ks) != set(DEFAULT_KS):
        raise SystemExit("formal LOTO calibration requires K={1,2,3}")
    record = json.loads(pathlib.Path(args.loto_record).read_text(encoding="utf-8"))
    identity = record["identity"]
    require_code_identity(identity.get("code_sha256"))
    if sha256_file(args.template_yaml) != identity.get("template_sha256"):
        raise SystemExit("template differs from the LOTO table identity")
    if identity.get("suite") != args.suite:
        raise SystemExit(f"loto record is for {identity.get('suite')!r}, run is for {args.suite!r}")
    gate = require_pass_gate(args.parity_gate, identity)
    if sha256_file(args.parity_gate) != record.get("parity_gate_sha256"):
        raise SystemExit("parity gate differs from the one that admitted the LOTO table")
    table_sha = sha256_file(args.loto_table)
    if table_sha != record.get("out_sha256"):
        raise SystemExit("loto table bytes differ from the record that describes them")
    if record.get("smoke"):
        raise SystemExit("refusing to fit a smoke table")
    cost = load_cost(pathlib.Path(args.cost))
    rows_all = load_rows(args.loto_table)
    validate_table_record(rows_all, record)
    validate_rows(rows_all)
    rows_in = [r for r in rows_all if bool(r.get("in_library"))]
    rows_out = [r for r in rows_all if not bool(r.get("in_library"))]
    shadow_rows = er.load_shadow(args.shadow_rows)
    shadow_sha = sha256_file(args.shadow_rows)
    template_sha = sha256_file(args.template_yaml)
    arm_record = json.loads(pathlib.Path(args.arm_record).read_text(encoding="utf-8"))
    t0 = time.time()
    audit = refit_shadow_and_check(shadow_rows, cost, arm_record, table_sha=shadow_sha, template_sha=template_sha)

    fits_out = {
        "protocol": "rit_loto_fits_v1", "suite": args.suite, "identity": identity, "cost": _cost_to_json(cost),
        "loto_table_sha256": table_sha, "parity_gate_status": gate["status"], "shadow_rows_sha256": shadow_sha,
        "arm_record_sha256": sha256_file(args.arm_record),
        "loto_all": fit_source(rows_all, cost, source="loto_all", input_sha256=table_sha, ks=ks),
        "loto_in_library": fit_source(rows_in, cost, source="loto_in_library", input_sha256=table_sha, ks=ks),
        "loto_out_of_library": fit_source(rows_out, cost, source="loto_out_of_library", input_sha256=table_sha, ks=ks),
    }
    shadow_block: dict[str, Any] = {"source": "shadow", "input_sha256": shadow_sha, "n_rows": len(shadow_rows),
                                    "alpha": audit["alpha"], "alpha_source": audit["alpha_source"],
                                    "warm_ts": list(WARM_TS), "audit": {k: v for k, v in audit.items() if k != "fits"},
                                    "s_quantiles": _quantiles(scores(shadow_rows)),
                                    "s_sample": scores(shadow_rows).tolist(), "fits": {}}
    for k, blob in audit["fits"].items():
        rec = fit_record_fields(blob["fit"])
        rec.update({"k": int(k), "n_rows": int(blob["n_rows"]), "n_seg_req": int(blob["n_seg_req"]),
                    "ir_range": [float(blob["ir_range"][0]), float(blob["ir_range"][1])],
                    "s_range": [float(blob["s"].min()), float(blob["s"].max())]})
        shadow_block["fits"][str(k)] = rec
    fits_out["shadow"] = shadow_block
    for src in ("loto_all", "shadow"):
        if fits_out[src].get("fit_unavailable"):
            raise SystemExit(f"{src} fit unavailable: {fits_out[src].get('reason')}; nothing downstream may run")
    (out_dir / "fits.json").write_text(json.dumps(fits_out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    compare: dict[str, Any] = {"protocol": "rit_loto_compare_v1", "suite": args.suite, "pairs": [],
                               "score_marginals": {}, "row_counts": {}}
    pairs = (("loto_all", "shadow", shadow_rows), ("loto_out_of_library", "shadow", shadow_rows),
             ("loto_in_library", "loto_out_of_library", rows_out))
    for k in ks:
        for name_a, name_b, rows_b in pairs:
            compare["pairs"].append(_compare_pair(name_a, name_b, fits_out[name_a], fits_out[name_b], rows_b, cost, k, WARM_TS))
    compare["score_marginals"]["loto_all_vs_shadow"] = score_marginal(scores(rows_all), scores(shadow_rows))
    compare["score_marginals"]["loto_in_vs_out_of_library"] = score_marginal(scores(rows_in), scores(rows_out))
    compare["row_counts"] = {"loto": _counts_table(rows_all), "loto_total": len(rows_all), "loto_in_library": len(rows_in),
                             "loto_out_of_library": len(rows_out), "shadow": len(shadow_rows),
                             "shadow_success_rows": int(sum(bool(r.get("episode_success")) for r in shadow_rows))}
    compare["tau_definition"] = "tau_b := p90_i |D_{i,a} - q_hat_b(s_i)| over the rows fit b was fitted on (engineering scale, not a test)"
    (out_dir / "compare.json").write_text(json.dumps(compare, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    if not args.skip_bootstrap:
        grid = display_grid(scores(rows_all), scores(shadow_rows), args.grid_points)
        bands = {"protocol": "rit_loto_bootstrap_band_v1", "suite": args.suite, "source": "shadow",
                 "note": "pointwise reference variability of the shadow fit under a task-stratified episode bootstrap; "
                         "not a simultaneous band and not a test of the LOTO-vs-shadow difference",
                 "bands": {}}
        for k in ks:
            bands["bands"][str(k)] = episode_bootstrap_band(shadow_rows, cost, WARM_TS, k, audit["alpha"], grid,
                                                            n_boot=args.n_boot, seed=args.boot_seed, jobs=args.jobs)
        (out_dir / "bootstrap_band.json").write_text(json.dumps(bands, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"fit_loto {args.suite}: rows all/in/out={len(rows_all)}/{len(rows_in)}/{len(rows_out)} shadow={len(shadow_rows)} "
          f"audit arms={audit['n_arms_checked']} cuts={audit['n_cuts_checked']} in {time.time() - t0:.0f}s -> {out_dir}")


if __name__ == "__main__":
    main()
