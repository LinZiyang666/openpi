"""Equal-NFE pairing and the pre-registered Q-B verdict of the step-vs-warm-start line (plan §2 Q-B).

Inputs per policy: the driver outputs of every arm (``<arms-root>/<teacher>/<arm_id>/`` with
``launch_*.json``, ``journal_*.jsonl``, ``per_step_*.jsonl``) and the server-side evidence rows of
every arm (``<server-rows>/<teacher>/<arm_id>/rows_*.jsonl``).

Admission per cell (task x arm): the launch manifest's expected episode set must be matched one
to one by accepted terminal journal records (``status`` done/failed, ``accepted`` not False, no
``error``, exactly one record per identity -- a second accepted record is a reported conflict,
never "the latest wins"); the server rows must hold a terminal finalize for every expected
episode whose client stamp (driver launch id / arm / config sha, forwarded by the worker's client
proxy) names this arm and one of its launches, with the worker's ``episode_summary`` row agreeing
on the decision count (its ``n_env_steps`` answers Q-C.1); and the strict equal-NFE gate must hold on *every* decision — plain arms: ``hit_type == MISS`` and
``executed_steps == m``; warm arms: ``hit_type == WARM_START`` and ``executed_steps == m``. Any
MISS fallback, extra stage-3 call, missing count or missing row marks the cell
``non_equal_nfe`` / ``evidence_missing``; its outcomes are still reported (descriptive) but the
policy's main verdict becomes inconclusive.

Statistics (fixed cliff group, plan §2 Q-B): per task the paired triples ``(F_i, P_i, W_i)`` over
the environment identities; ``g = mean(F-P)``, ``Delta = mean(W-P)``, ``H50 = Delta - 0.5 g``,
``H25 = Delta - 0.25 g``; the main estimate is the equal-weight macro over the fixed cliff group,
bootstrapped by resampling identities within each task with the three arms sharing indices
(``--boot`` default 100000, seed 20260919); 12 pre-registered intervals at alpha = 0.05/12 (8
macro quantities over two policies + 4 flat Deltas). Flat Deltas use the discordant-pair
Clopper-Pearson construction ``[L10 - U01, U10 - L01]`` at 1 - alpha/2 each side.

Verdict per policy: supported (cliff macro g, Delta, H50 lower bounds > 0 and every flat task
Delta lower > -0.10) / not supported (g lower > 0 and H25 upper < 0) / inconclusive; independent
``harmful_on_flat`` flag (any flat Delta upper < -0.10).
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import hashlib
import pathlib
from typing import Dict, List, Optional, Tuple

import numpy as np

from exp.step_diag import envs as _envs
from exp.step_diag import evidence as _evidence

ALPHA_FAMILY = 0.05 / 12
ANALYSIS_SEED = 20260919


# ------------------------------------------------------------------
# Loading
# ------------------------------------------------------------------


def load_arm(arm_dir: pathlib.Path) -> dict:
    """Expected identities (launch manifest) + accepted terminal outcomes (journal) of one arm."""
    launches = sorted(arm_dir.glob("launch_*.json"))
    if not launches:
        raise FileNotFoundError(f"{arm_dir}: no launch manifest")
    expected: Dict[str, dict] = {}
    launch_ids: set = set()
    arm_ids: set = set()
    launch_map = {}
    manifest_problems = []
    for lp in launches:
        launch = json.loads(lp.read_text())
        lid = launch.get("launch_id")
        if not lid or (lid in launch_map and launch_map[lid] != launch):
            manifest_problems.append("duplicate_or_missing_launch_id")
        launch_map[lid] = launch
        launch_ids.add(launch.get("launch_id"))
        arm_ids.add(launch.get("arm_id"))
        for ident in launch["expected"]:
            if ident["task_uid"] in expected and expected[ident["task_uid"]] != ident:
                manifest_problems.append("expected_identity_conflict")
            expected[ident["task_uid"]] = ident
    if len(arm_ids) > 1:
        raise ValueError(f"{arm_dir}: launch manifests of different arms {sorted(map(str, arm_ids))}")
    outcomes: Dict[str, dict] = {}
    conflicts: set = set()
    for jp in sorted(arm_dir.glob("journal_*.jsonl")):
        for line in jp.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("accepted") is not True or rec.get("status") not in ("done", "failed"):
                continue
            uid = rec["task_uid"]
            if uid in outcomes:
                # Two accepted terminal records for one identity (two launches, or a journal that
                # was not resumed idempotently): a conflict to report, never "take the latest".
                conflicts.add(uid)
                continue
            outcomes[uid] = rec
    summaries: Dict[str, dict] = {}
    for pp in sorted(arm_dir.glob("per_step_*.jsonl")):
        for line in pp.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("row") == "episode_summary" and row.get("accepted") is True:
                attempts = summaries.setdefault(row["task_uid"], {})
                attempt = int(row.get("attempt", 1))
                if attempt in attempts:
                    conflicts.add(row["task_uid"])
                attempts[attempt] = row
    if set(outcomes) - set(expected):
        manifest_problems.append("unexpected_accepted_terminal")
    return {"dir": str(arm_dir), "expected": expected, "outcomes": outcomes, "conflicts": conflicts,
            "summaries": summaries, "launch_ids": launch_ids, "arm_id": next(iter(arm_ids), None),
            "launches": launch_map, "manifest_problems": manifest_problems}


def load_server_rows(rows_dir: pathlib.Path) -> Dict[Tuple[str, int], dict]:
    """``{(task_uid, attempt): {"finalize": row, "decisions": [rows]}}`` from the recorder rows."""
    eps: Dict[Tuple[str, int], dict] = collections.defaultdict(lambda: {"finalize": None, "decisions": []})
    manifests = _evidence.load_manifests(rows_dir)
    for p in sorted(rows_dir.glob("rows_*.jsonl")):
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            key = (r["task_uid"], int(r.get("attempt", 1)))
            if r.get("status") == "finalize":
                if eps[key]["finalize"] is not None:
                    eps[key]["duplicate_finalize"] = True
                eps[key]["finalize"] = r
                eps[key]["manifest"] = manifests.get(r.get("config_sha"))
            else:
                eps[key]["decisions"].append(r)
    for ep in eps.values():
        fin = ep["finalize"] or {}
        path = rows_dir / (fin.get("arrays") or "__missing__")
        ep["arrays_valid"] = False
        if not path.is_file() or not fin.get("arrays_sha256"):
            continue
        if hashlib.sha256(path.read_bytes()).hexdigest() != fin["arrays_sha256"]:
            continue
        env = _envs.ENVS.get(fin.get("env_id"))
        if env is None:
            continue
        try:
            with np.load(path, allow_pickle=False) as arrs:
                ep["arrays_valid"] = all(
                    arrs[f"a_exec_{d['decision_idx']:04d}"].shape == (env.action_horizon, env.action_dim)
                    and np.isfinite(arrs[f"a_exec_{d['decision_idx']:04d}"]).all()
                    and d.get("arrays") == fin["arrays"] and d.get("arrays_sha256") == fin["arrays_sha256"]
                    for d in ep["decisions"])
        except (ValueError, KeyError, OSError):
            pass
    return eps


# ------------------------------------------------------------------
# Admission + equal-NFE gate
# ------------------------------------------------------------------


def cell_admission(arm: dict, server: Dict[Tuple[str, int], dict], task: str, *, kind: str, m: int,
                   env_id: str | None = None) -> dict:
    """Join terminal, worker and server evidence; every missing binding fails closed."""
    exp_ids = {uid: ident for uid, ident in arm["expected"].items() if ident["task"] == task}
    outcomes, totals = {}, {}
    comparisons = set()
    worker_identities = set()
    runtime_notes = set()
    problems = collections.Counter(arm.get("manifest_problems", []))
    steps, env_steps, worker_counts = [], [], []
    n_miss = 0
    for uid, ident in exp_ids.items():
        rec = arm["outcomes"].get(uid)
        if rec is None:
            problems["missing_terminal"] += 1
            continue
        if uid in arm.get("conflicts", ()):
            problems["duplicate_accepted_terminal"] += 1
        if rec.get("error") or type(rec.get("success")) is not bool:
            problems["terminal_error"] += 1
        if type(rec.get("success")) is bool:
            outcomes[uid] = {**ident, "success": rec["success"], "attempt": rec.get("attempt")}
        attempt = rec.get("attempt")
        ep = server.get((uid, attempt))
        if not ep or not ep.get("finalize") or ep["finalize"].get("terminal") is not True:
            problems["server_evidence_missing"] += 1
            continue
        fin = ep["finalize"]
        if not ep.get("arrays_valid"):
            problems["arrays_missing_or_invalid"] += 1
        if ep.get("duplicate_finalize"):
            problems["duplicate_finalize"] += 1
        stamp = fin.get("client_stamp") or {}
        teacher = pathlib.Path(arm["dir"]).parent.name
        policy = "groot" if teacher == "groot_tp" else "pi05"
        expected_env = env_id or f"{policy}_rc"
        if fin.get("env_id") != expected_env:
            problems["environment_identity_mismatch"] += 1
        manifest = ep.get("manifest")
        problems.update(_evidence.manifest_problems(manifest, env_id=expected_env, arm_id=arm["arm_id"],
                                                     config_sha=stamp.get("config_sha")))
        if manifest:
            comparisons.add(_evidence.comparison_identity(manifest))
            runtime_notes.add(_envs.sha256_json(_evidence.runtime_note(manifest)))
        launch = arm.get("launches", {}).get(stamp.get("launch_id"))
        if (not launch or any(not stamp.get(k) for k in ("launch_id", "arm_id", "experiment_id", "config_sha"))
                or any(stamp.get(k) != launch.get(k) for k in ("arm_id", "experiment_id", "config_sha"))):
            problems["arm_stamp_mismatch"] += 1
        if not launch or not launch.get("driver_run_id") or rec.get("run_id") != launch["driver_run_id"]:
            problems["driver_run_mismatch"] += 1
        if launch and uid not in {i["task_uid"] for i in launch["expected"]}:
            problems["driver_episode_mismatch"] += 1
        if fin.get("stamp_mismatch") or fin.get("config_sha") != stamp.get("config_sha"):
            problems["config_mismatch"] += 1
        if type(fin.get("outcome")) is not bool or fin.get("outcome") != rec.get("success"):
            problems["outcome_mismatch"] += 1
        n = fin.get("n_decisions")
        summ = arm.get("summaries", {}).get(uid, {}).get(attempt)
        if summ is None:
            problems["worker_summary_missing"] += 1
        else:
            # recorded, not gated: which simulator island produced the episode
            wid = _evidence.worker_identity(summ.get("worker_runtime"))
            if wid is not None:
                worker_identities.add(wid)
            if (summ.get("n_decisions") != n or type(summ.get("n_env_steps")) is not int
                    or summ.get("n_env_steps", 0) < 1):
                problems["decision_count_mismatch"] += 1
            if summ.get("run_id") != rec.get("run_id") or summ.get("success") != rec.get("success") or summ.get("error"):
                problems["worker_identity_mismatch"] += 1
            if type(summ.get("n_env_steps")) is int:
                env_steps.append(summ["n_env_steps"])
            if type(summ.get("n_decisions")) is int:
                worker_counts.append(summ["n_decisions"])
        decs = sorted(ep["decisions"], key=lambda r: r.get("decision_idx", -1))
        if type(n) is not int or n < 1 or [d.get("decision_idx") for d in decs] != list(range(n)):
            problems["server_decision_gap"] += 1
        # Teacher identity comes from the manifest/directory, never from the evidence being checked.
        teacher = (launch or {}).get("teacher") or pathlib.Path(arm["dir"]).parent.name
        policy = "groot" if teacher == "groot_tp" else "pi05"
        schedule = "pi05_v1" if policy == "pi05" else f"groot_n15_k{4 if kind == 'warm' else m}_v1"
        want_t = float(arm["arm_id"].removeprefix("warm_t")) if kind == "warm" else None
        ep_steps = []
        for d in decs:
            st = d.get("executed_steps")
            if d.get("status") != "ok":
                problems["decision_error"] += 1
            if d.get("config_sha") != stamp.get("config_sha"):
                problems["config_mismatch"] += 1
            if d.get("env_id") != expected_env:
                problems["environment_identity_mismatch"] += 1
            for key in ("task", "init_idx", "env_seed", "lane", "pin_id", "layout", "style", "init_pool_sha256"):
                if key in ident and d.get(key) != ident[key]:
                    problems["environment_identity_mismatch"] += 1
                if key in ident and fin.get(key) != ident[key]:
                    problems["environment_identity_mismatch"] += 1
            if d.get("schedule_id") != schedule or d.get("start_t") != want_t:
                problems["schedule_mismatch"] += 1
            want = "MISS" if kind == "plain" else "WARM_START"
            if d.get("hit_type") != want:
                n_miss += int(kind == "warm")
                problems["hit_type_mismatch"] += 1
            if type(st) is not int or st != m:
                problems["steps_mismatch"] += 1
            if type(d.get("n_stage3_calls")) is not int or d["n_stage3_calls"] != 1:
                problems["extra_stage3_calls"] += 1
            if type(st) is int and st >= 0:
                steps.append(st)
                ep_steps.append(st)
        totals[uid] = sum(ep_steps) if len(ep_steps) == len(decs) == n else None
    complete = bool(exp_ids) and len(outcomes) == len(exp_ids) and not any(
        problems[k] for k in ("missing_terminal", "terminal_error", "duplicate_accepted_terminal"))
    n_dec = sum(len(server.get((uid, rec.get("attempt")), {}).get("decisions", []))
                for uid, rec in outcomes.items())
    return {"task": task, "kind": kind, "m": m, "n_expected": len(exp_ids), "n_outcomes": len(outcomes),
            "complete": complete, "equal_nfe": complete and not any(problems.values()),
            "problems": {k: v for k, v in problems.items() if v}, "n_decisions": n_dec,
            "miss_fraction": n_miss / n_dec if n_dec else None,
            "mean_executed_steps": float(np.mean(steps)) if steps else None,
            "episode_total_nfe": totals,
            "comparison_identities": sorted(comparisons),
            "worker_identities": sorted(worker_identities),
            "runtime_notes": sorted(runtime_notes),
            "env_steps_mean": float(np.mean(env_steps)) if env_steps else None,
            "decisions_per_episode_mean": float(np.mean(worker_counts)) if worker_counts else None,
            "outcomes": outcomes}


# ------------------------------------------------------------------
# Statistics
# ------------------------------------------------------------------


def pair_identity(v: dict) -> tuple:
    return tuple(v.get(k) for k in ("task", "init_idx", "env_seed", "lane", "pin_id", "layout", "style"))


def paired_outcomes(plain: dict, warm: dict) -> list:
    p = {pair_identity(v): int(v["success"]) for v in plain["outcomes"].values()}
    w = {pair_identity(v): int(v["success"]) for v in warm["outcomes"].values()}
    return [(p[i], w[i]) for i in sorted(p.keys() & w.keys())]


def wilson(outcomes: list) -> dict:
    n = len(outcomes)
    if not n:
        return {"n": 0, "sr": None, "lower": None, "upper": None}
    p, z = float(np.mean(outcomes)), 1.959963984540054
    center = (p+z*z/(2*n))/(1+z*z/n)
    half = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return {"n": n, "sr": p, "lower": center-half, "upper": center+half}


def comparison_problems(cells: dict, arms: dict) -> list:
    """Cross-arm gates of one comparison: one model + environment contract, one experiment namespace.

    Worker islands and serving runtimes (host / GPU / torch / source digests) are reported by
    ``comparison_notes``; they do not gate (owner ruling 2026-09-12).
    """
    problems = []
    values = [c["comparison_identities"] for c in cells.values()]
    if any(len(v) != 1 for v in values) or any(v != values[0] for v in values):
        problems.append("comparison_identities_mismatch")
    experiments = {launch.get("experiment_id") for aid in cells for launch in arms[aid]["launches"].values()}
    if len(experiments) != 1 or not next(iter(experiments), None):
        problems.append("experiment_identity_mismatch")
    return problems


def comparison_notes(cells: dict) -> dict:
    """Descriptive: how many distinct worker islands / serving runtimes the arms of a task used."""
    workers = {w for c in cells.values() for w in c.get("worker_identities", [])}
    runtimes = {r for c in cells.values() for r in c.get("runtime_notes", [])}
    return {"worker_islands": len(workers), "serving_runtimes": len(runtimes),
            "uniform": len(workers) <= 1 and len(runtimes) <= 1}


def paired_triples(full: dict, plain: dict, warm: dict) -> List[Tuple[int, int, int]]:
    """``(F, P, W)`` per identity (init_idx) present in all three cells, ordered by init_idx."""
    by = lambda cell: {pair_identity(v): int(v["success"]) for v in cell["outcomes"].values()}  # noqa: E731
    f, p, w = by(full), by(plain), by(warm)
    ids = sorted(set(f) & set(p) & set(w))
    return [(f[i], p[i], w[i]) for i in ids]


def task_stats(triples: List[Tuple[int, int, int]]) -> dict:
    a = np.asarray(triples, dtype=float)
    if a.size == 0:
        return {"n": 0, "g": None, "delta": None, "H50": None, "H25": None, "recovery": None}
    g = float(np.mean(a[:, 0] - a[:, 1]))
    d = float(np.mean(a[:, 2] - a[:, 1]))
    return {"n": int(a.shape[0]), "sr_full": float(a[:, 0].mean()), "sr_plain": float(a[:, 1].mean()),
            "sr_warm": float(a[:, 2].mean()), "g": g, "delta": d, "H50": d - 0.5 * g, "H25": d - 0.25 * g,
            "recovery": (d / g) if g > 0 else None}


def macro_bootstrap(cliff: Dict[str, List[Tuple[int, int, int]]], boot: int, alpha: float,
                    seed: int = ANALYSIS_SEED) -> dict:
    """Equal-weight macro of g / Delta / H50 / H25 with within-task resampling, shared indices across arms."""
    rng = np.random.default_rng(seed)
    arrays = {t: np.asarray(v, dtype=float) for t, v in cliff.items() if len(v)}
    if not arrays:
        return {"n_tasks": 0}
    if boot < 1:
        raise ValueError("bootstrap count must be positive")
    points, draws = [], np.zeros((boot, 4))
    for arr in arrays.values():
        g, d = arr[:, 0] - arr[:, 1], arr[:, 2] - arr[:, 1]
        points.append([g.mean(), d.mean(), (d-.5*g).mean(), (d-.25*g).mean()])
        # All four quantities use the same sampled triples. Batching bounds memory.
        for lo_idx in range(0, boot, 10000):
            hi_idx = min(boot, lo_idx + 10000)
            idx = rng.integers(0, len(arr), size=(hi_idx-lo_idx, len(arr)))
            gb, db = g[idx].mean(axis=1), d[idx].mean(axis=1)
            draws[lo_idx:hi_idx] += np.column_stack([gb, db, db-.5*gb, db-.25*gb])
    point = np.mean(points, axis=0)
    draws /= len(arrays)
    lo = np.percentile(draws, 100 * alpha / 2, axis=0)
    hi = np.percentile(draws, 100 * (1 - alpha / 2), axis=0)
    names = ("g", "delta", "H50", "H25")
    out = {"n_tasks": len(arrays), "boot": boot, "alpha": alpha, "seed": seed}
    for i, name in enumerate(names):
        degenerate = bool(np.ptp(draws[:, i]) == 0.0)
        out[name] = {"point": point[i], "lower": float(lo[i]), "upper": float(hi[i]), "degenerate": degenerate}
    return out


def clopper_pearson(x: int, n: int, conf: float) -> Tuple[float, float]:
    from scipy.stats import beta

    lo = 0.0 if x == 0 else float(beta.ppf((1 - conf) / 2, x, n - x + 1))
    hi = 1.0 if x == n else float(beta.ppf(1 - (1 - conf) / 2, x + 1, n - x))
    return lo, hi


def flat_delta_interval(pw: List[Tuple[int, int]], alpha: float) -> dict:
    """``Delta = mean(W - P)`` with the discordant-pair construction ``[L10 - U01, U10 - L01]``."""
    n = len(pw)
    n10 = sum(1 for p, w in pw if w == 1 and p == 0)
    n01 = sum(1 for p, w in pw if w == 0 and p == 1)
    if not n:
        return {"n": 0, "n10": 0, "n01": 0, "delta": None, "lower": -1.0, "upper": 1.0, "conf_each_side": 1-alpha/2}
    conf = 1 - alpha / 2
    l10, u10 = clopper_pearson(n10, n, conf)
    l01, u01 = clopper_pearson(n01, n, conf)
    return {"n": n, "n10": n10, "n01": n01, "delta": (n10 - n01) / n if n else None,
            "lower": l10 - u01, "upper": u10 - l01, "conf_each_side": conf}


def verdict(macro: dict, flats: Dict[str, dict], cells_ok: bool) -> dict:
    flags = {"harmful_on_flat": any(f["upper"] < -0.10 for f in flats.values())}
    if not cells_ok or macro.get("n_tasks", 0) != 4 or not flats or any(macro[q]["degenerate"] for q in ("g", "delta", "H50", "H25")):
        return {"verdict": "inconclusive", "reason": "cells not admissible / too few cliff tasks / degenerate bootstrap",
                **flags}
    flat_ok = all(f["lower"] > -0.10 for f in flats.values())
    if macro["g"]["lower"] > 0 and macro["delta"]["lower"] > 0 and macro["H50"]["lower"] > 0 and flat_ok:
        return {"verdict": "supported", **flags}
    if macro["g"]["lower"] > 0 and macro["H25"]["upper"] < 0:
        return {"verdict": "not_supported", **flags}
    return {"verdict": "inconclusive", "reason": "gap not reproduced / intervals too wide / flat non-inferiority unproven",
            **flags}


# ------------------------------------------------------------------
# Similarity bins (plan §3.3, descriptive)
# ------------------------------------------------------------------

MIN_BIN = 4
SHADOW_IDX = range(10)


def similarity_bins(shadow_episodes: List[dict], cells: Dict[str, Dict[str, dict]], plain_id: str, warm_id: str) -> dict:
    """Two score bins per task over the identities the shadow run scored (init idx 0..9 only).

    Score = the teacher shadow's per-episode median top-1 score (frozen BEFORE the warm outcomes
    are read). Split at the per-task median of the scored identities that also have both plain
    and warm outcomes; ties go to the low bin (equal scores are never split). A bin shows its
    episode Delta = mean(W - P) only with >= MIN_BIN identities; the rest is reported as
    unavailable. This is an association on the fixed teacher trajectories, not a risk curve.
    """
    scores = collections.defaultdict(dict)
    for ep in shadow_episodes:
        task_cells = cells.get(ep["task"], {})
        if any(ep.get("comparison_identity") not in task_cells.get(a, {}).get("comparison_identities", [])
               for a in (plain_id, warm_id)):
            continue
        if ep.get("init_idx") in SHADOW_IDX and ep.get("top1_score_median") is not None:
            scores[ep["task"]][pair_identity(ep)] = float(ep["top1_score_median"])
    out = {}
    for task, c in cells.items():
        if plain_id not in c or warm_id not in c:
            out[task] = {"status": "arm_missing"}
            continue
        p = {pair_identity(v): int(v["success"]) for v in c[plain_id]["outcomes"].values() if v["init_idx"] in SHADOW_IDX}
        w = {pair_identity(v): int(v["success"]) for v in c[warm_id]["outcomes"].values() if v["init_idx"] in SHADOW_IDX}
        ids = sorted(i for i in scores.get(task, {}) if i in p and i in w)
        rec = {"n_scored": len(scores.get(task, {})), "n_joined": len(ids), "n_shadow_idx": len(SHADOW_IDX)}
        if len(ids) < 2 * MIN_BIN:
            rec["status"] = "too_few"
            out[task] = rec
            continue
        # Freeze the cut from all admitted pre-intervention scores before joining outcomes.
        vals = sorted(scores[task].values())
        cut = float(np.median(vals))
        low = [i for i in ids if scores[task][i] <= cut]
        high = [i for i in ids if scores[task][i] > cut]
        rec.update({"status": "ok", "cut": cut, "score_range": [vals[0], vals[-1]]})
        for name, members in (("low", low), ("high", high)):
            if len(members) >= MIN_BIN:
                rec[name] = {"n": len(members), "delta": float(np.mean([w[i] - p[i] for i in members])),
                             "sr_plain": float(np.mean([p[i] for i in members])),
                             "sr_warm": float(np.mean([w[i] for i in members]))}
            else:
                rec[name] = {"n": len(members), "delta": None}
        out[task] = rec
    return out


# ------------------------------------------------------------------
# Driver
# ------------------------------------------------------------------


def aggregate(policy: str, arms_root: pathlib.Path, server_root: pathlib.Path, *, boot: int,
              shadow_json: Optional[pathlib.Path] = None) -> dict:
    teacher = "pi05" if policy == "pi05" else "groot_tp"
    m_star, t_star = _envs.QB_MAIN_M[policy], _envs.QB_MAIN_T[policy]
    full_id, plain_id, warm_id = "full", f"plain_k{m_star}", f"warm_t{t_star:g}"
    arm_specs = [("full", "plain", _envs.ENVS[f"{policy}_rc"].k_full)]
    arm_specs += [(f"plain_k{k}", "plain", k) for k in _envs.QB_PLAIN_KS[policy]]
    arm_specs += [(f"warm_t{t:g}", "warm", _envs.ENVS[f"{policy}_rc"].remaining_steps(t))
                  for t in _envs.QB_WARM_TS[policy]]
    arms = {aid: load_arm(arms_root / teacher / aid) for aid, _, _ in arm_specs
            if (arms_root / teacher / aid).is_dir()}
    servers = {aid: load_server_rows(server_root / teacher / aid) for aid in arms if (server_root / teacher / aid).is_dir()}
    missing = [aid for aid in (full_id, plain_id, warm_id) if aid not in arms]
    cells: Dict[str, Dict[str, dict]] = {}
    for task in _envs.qb_tasks(policy):
        cells[task] = {}
        for aid, kind, m in arm_specs:
            if aid in arms:
                cell = cell_admission(arms[aid], servers.get(aid, {}), task, kind=kind, m=m)
                expected = [v for v in arms[aid]["expected"].values() if v["task"] == task]
                count = _envs.qb_episode_count(policy, task, aid)
                if (len(expected) != count or {v.get("init_idx") for v in expected} != set(range(count))
                        or any(v.get("env_seed") != _envs.RC_FORMAL_BASE_SEED + v.get("init_idx", -1)
                               or v.get("lane") != _envs.lane_of(task)
                               or v.get("pin_id") != (_envs.canonical_pin_id() if _envs.lane_of(task) == "pnp" else None)
                               or v.get("layout") != 1 or v.get("style") != 1 for v in expected)):
                    cell["complete"] = cell["equal_nfe"] = False
                    cell["problems"]["preregistered_identity_set_mismatch"] = 1
                cells[task][aid] = cell
    per_task = {}
    cliff_triples = {}
    flats = {}
    cells_ok = not missing
    for task in _envs.qb_tasks(policy):
        c = {a: cells[task][a] for a in (full_id, plain_id, warm_id) if a in cells[task]}
        if len(c) < 3:
            per_task[task] = {"status": "arm_missing"}
            cells_ok = False
            continue
        ok = all(c[a]["complete"] and c[a]["equal_nfe"] for a in c)
        cross_problems = comparison_problems(c, arms)
        ok = ok and not cross_problems
        triples = paired_triples(c[full_id], c[plain_id], c[warm_id])
        expected_pairs = _envs.qb_episode_count(policy, task, plain_id)
        if len(paired_outcomes(c[plain_id], c[warm_id])) != expected_pairs or len(triples) != _envs.QB_EPISODES:
            ok = False
        st = task_stats(triples)
        st["exploratory_ci95"] = macro_bootstrap({task: triples}, min(boot, 20000), .05)
        st["sr_wilson95"] = {a: wilson([v["success"] for v in cell["outcomes"].values()]) for a, cell in c.items()}
        per_task[task] = {"status": "ok" if ok else "not_admissible", "comparison_problems": cross_problems,
                          "comparison_notes": comparison_notes(c),
                          "cells": {a: {k: v for k, v in c[a].items() if k != "outcomes"} for a in c}, **st}
        if task in _envs.QB_CLIFF[policy]:
            cliff_triples[task] = triples
            cells_ok = cells_ok and ok
        else:
            pw = paired_outcomes(c[plain_id], c[warm_id])
            flats[task] = flat_delta_interval(pw, ALPHA_FAMILY)
            per_task[task].update(n=len(pw), n_triples=len(triples), delta=flats[task]["delta"],
                                  sr_plain=float(np.mean([p for p, _ in pw])) if pw else None,
                                  sr_warm=float(np.mean([w for _, w in pw])) if pw else None,
                                  g=None, H50=None, H25=None, recovery=None)
            cells_ok = cells_ok and ok
    exploratory = {}
    for t in _envs.QB_WARM_TS[policy]:
        m = _envs.ENVS[f"{policy}_rc"].remaining_steps(t)
        pid, wid = f"plain_k{m}", f"warm_t{t:g}"
        exploratory[str(m)] = {}
        for task, all_cells in cells.items():
            if any(a not in all_cells for a in (full_id, pid, wid)):
                exploratory[str(m)][task] = {"status": "arm_missing"}
                continue
            c = {a: all_cells[a] for a in (full_id, pid, wid)}
            triples = paired_triples(c[full_id], c[pid], c[wid])
            cross_problems = comparison_problems(c, arms)
            exploratory[str(m)][task] = {**task_stats(triples),
                "status": "ok" if (all(v["equal_nfe"] for v in c.values()) and not cross_problems
                                    and len(triples) == 50) else "not_admissible",
                "comparison_problems": cross_problems,
                "ci95": macro_bootstrap({task: triples}, min(boot, 20000), .05),
                "sr_wilson95": {a: wilson([v["success"] for v in cell["outcomes"].values()]) for a, cell in c.items()},
                "cells": {a: {k: v for k, v in cell.items() if k != "outcomes"} for a, cell in c.items()}}
    macro = macro_bootstrap(cliff_triples, boot, ALPHA_FAMILY)
    v = verdict(macro, flats, cells_ok)
    bins = None
    if shadow_json is not None and shadow_json.is_file():
        shadow = json.loads(shadow_json.read_text())
        if shadow.get("env_id") != f"{policy}_rc":
            raise ValueError("similarity scores must come from this policy's RC teacher shadow")
        bins = similarity_bins(shadow.get("episodes", []), cells, plain_id, warm_id)
    return {"policy": policy, "main_m": m_star, "main_t": t_star, "arms": {aid: a["dir"] for aid, a in arms.items()},
            "missing_arms": missing, "per_task": per_task, "cliff_macro": macro, "flat": flats, "verdict": v,
            "exploratory_budgets": exploratory, "alpha_family": ALPHA_FAMILY, "historical_gap": _envs.QB_HISTORICAL_GAP[policy], "similarity_bins": bins}


def markdown(res: dict) -> str:
    f = lambda v: "–" if v is None else f"{v:+.3f}"  # noqa: E731
    lines = [f"## Q-B {res['policy']} (m*={res['main_m']}, t*={res['main_t']}) — verdict **{res['verdict']['verdict']}**"
             + (" ⚠ harmful_on_flat" if res["verdict"].get("harmful_on_flat") else ""),
             "", "| task | group | n | SR full | SR plain | SR warm | g | Δ | H50 | H25 | recovery | admissible |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for task, r in res["per_task"].items():
        group = "cliff" if task in res["historical_gap"] else "flat"
        if r.get("status") == "arm_missing":
            lines.append(f"| {task} | {group} | – | arm missing |||||||||")
            continue
        lines.append(f"| {task} | {group} | {r['n']} | {f(r.get('sr_full'))} | {f(r.get('sr_plain'))} | {f(r.get('sr_warm'))} | "
                     f"{f(r['g'])} | {f(r['delta'])} | {f(r['H50'])} | {f(r['H25'])} | {f(r['recovery'])} | {r['status']} |")
    m = res["cliff_macro"]
    if m.get("n_tasks"):
        lines += ["", f"cliff macro (n_tasks={m['n_tasks']}, boot={m['boot']}, α={m['alpha']:.5f}): "
                  + ", ".join(f"{q} {f(m[q]['point'])} [{f(m[q]['lower'])}, {f(m[q]['upper'])}]" for q in ("g", "delta", "H50", "H25"))]
    for task, fl in res["flat"].items():
        lines.append(f"flat {task}: Δ {f(fl['delta'])} [{f(fl['lower'])}, {f(fl['upper'])}] (n10={fl['n10']}, n01={fl['n01']}, n={fl['n']})")
    lines += ["", "Flat n counts all primary P/W pairs (100); its full reference has 50 episodes. "
              "Per-arm Wilson intervals, per-task paired bootstrap intervals and episode NFE totals are in the JSON."]
    for budget, tasks in res.get("exploratory_budgets", {}).items():
        if int(budget) == res["main_m"]:
            continue
        lines += ["", f"Exploratory budget m={budget} (95% intervals; no formal verdict):",
                  "", "| task | paired n | g | Δ | Δ CI95 | status |", "|---|---|---|---|---|---|"]
        for task, rec in tasks.items():
            ci = rec.get("ci95", {}).get("delta", {})
            lines.append(f"| {task} | {rec.get('n', 0)} | {f(rec.get('g'))} | {f(rec.get('delta'))} | "
                         f"[{f(ci.get('lower'))}, {f(ci.get('upper'))}] | {rec['status']} |")
    bins = res.get("similarity_bins")
    if bins:
        lines += ["", "similarity bins (shadow idx 0..9 only, descriptive): task | joined | low Δ (n) | high Δ (n) | cut"]
        for task, b in bins.items():
            if b.get("status") != "ok":
                lines.append(f"- {task}: {b.get('status')} (scored {b.get('n_scored', 0)}, joined {b.get('n_joined', 0)})")
                continue
            lo, hi = b["low"], b["high"]
            lines.append(f"- {task}: {b['n_joined']} | {f(lo['delta'])} ({lo['n']}) | {f(hi['delta'])} ({hi['n']}) | {b['cut']:.3f}")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", required=True, choices=("pi05", "groot"))
    ap.add_argument("--arms-root", required=True, help="driver outputs: <root>/<teacher>/<arm_id>/")
    ap.add_argument("--server-rows", required=True, help="pulled server rows: <root>/<teacher>/<arm_id>/rows_*.jsonl")
    ap.add_argument("--boot", type=int, default=100000)
    ap.add_argument("--shadow-json", default=None, help="analyze_shadow output of the RC shadow run (similarity bins)")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", default=None)
    a = ap.parse_args()
    if a.boot != 100000:
        ap.error("formal Q-B analysis freezes --boot=100000")
    res = aggregate(a.policy, pathlib.Path(a.arms_root), pathlib.Path(a.server_rows), boot=a.boot,
                    shadow_json=None if a.shadow_json is None else pathlib.Path(a.shadow_json))
    pathlib.Path(a.out_json).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(a.out_json).write_text(json.dumps(res, indent=1, default=str))
    if a.out_md:
        pathlib.Path(a.out_md).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(a.out_md).write_text(markdown(res))
    print(f"AGGREGATE_ARMS {a.policy}: {res['verdict']} -> {a.out_json}")


if __name__ == "__main__":
    main()
