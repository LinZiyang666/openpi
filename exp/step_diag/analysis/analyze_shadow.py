"""Shadow-channel analysis of the step-vs-warm-start line (plan §2 Q-A, §3.3).

Input: conductor launch/journal/worker evidence via ``--driver-dir``, the recorder's
``rows_*.jsonl`` files (+ their ``.npz`` arrays) of one environment, the
frozen action weights (``--weights`` npz with ``w``, ``sigma``, ``active_mask`` written by
``freeze_weights``) and, for the correlation, the historical ladder gaps per task
(``--gaps`` json ``{task: g}``; see ``ladder_gaps``).

Admission (per episode = ``(task_uid, attempt)``): a terminal ``finalize`` row; decision indices
contiguous ``0..n_decisions-1``; ``ok`` rows only, with the arrays file present and its sha256
matching; ``N_full >= n_primary`` and every ``k`` of ``k_set`` with ``n_primary`` paired samples;
finite values. Rejections are counted per reason (never silently dropped). Episode coverage
(fraction of decisions with valid primary metrics) must be >= ``--min-coverage`` (0.9) and a task
needs >= ``--min-episodes`` (8) admitted episodes to publish task metrics.

Per decision: ``d_k`` / ``r_k`` / ``disp_K`` / ``d_w(t)`` (``exp.step_diag.metrics``); dense
decisions add the PCA-2D / GMM fit. Task metric = median over episodes of the per-episode median.

Correlation (exploratory): Spearman rho between the task ``median d_1`` and the ladder gap ``g``,
with a task-cluster bootstrap (``--boot``, seed 20260919). Reading: rho >= 0.6 with 95% lower
bound > 0.2 -> "consistent association on these tasks"; rho < 0.4, gap range < 0.15 or constant
inputs -> "no conclusion"; in between -> "grey". LOTO (leave-one-task-out threshold on d_1 for
gap >= 0.15) is a descriptive feasibility table only.

usage:
  python -m exp.step_diag.analysis.analyze_shadow --env-id pi05_rc --rows <dir or files...> \\
      --driver-dir <conductor-dir> --weights <npz> [--gaps gaps.json] --out-json <json> --out-md <md>
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import pathlib
from typing import Dict, List, Optional

import numpy as np
import torch

from exp.step_diag import envs as _envs
from exp.step_diag import metrics as M
from exp.step_diag import recorder as R

ANALYSIS_SEED = 20260919
CLIFF_GAP = 0.15


# ------------------------------------------------------------------
# Frozen weights and ladder gaps (inputs)
# ------------------------------------------------------------------


def freeze_weights(library_pkl: str, env_id: str, out_npz: str) -> dict:
    """``w / sigma / active_mask`` of an environment from its library pickle (plan §2.0)."""
    import pickle

    env = _envs.resolve_env(env_id)
    with open(library_pkl, "rb") as f:
        lib = pickle.load(f)
    chunks = torch.stack([torch.as_tensor(e.payload.action_chunk, dtype=torch.float32) for e in lib["entries"]])
    if chunks.shape[-1] != env.action_dim:
        raise ValueError(f"library action dim {chunks.shape[-1]} != env {env.action_dim}")
    mask = M.executed_mask(env.action_dim, env.n_executed)
    w, sigma, degenerate = M.frozen_action_weights(chunks, mask)
    np.savez(out_npz, w=w.numpy(), sigma=sigma.numpy(), active_mask=mask.numpy(), degenerate=np.asarray(degenerate),
             library_sha256=np.asarray(_envs.library_digest(library_pkl)), env_id=np.asarray(env_id),
             n_entries=np.asarray(int(chunks.shape[0])))
    return {"env_id": env_id, "n_entries": int(chunks.shape[0]), "degenerate_dims": degenerate,
            "library_sha256": _envs.library_digest(library_pkl)}


def load_weights(npz_path: str) -> tuple[torch.Tensor, torch.Tensor, dict]:
    z = np.load(npz_path)
    w = torch.as_tensor(z["w"], dtype=torch.float32)
    mask = torch.as_tensor(z["active_mask"], dtype=torch.bool)
    info = {k: (z[k].tolist() if z[k].ndim else z[k].item()) for k in z.files if k not in ("w", "sigma", "active_mask")}
    return w, mask, info


def ladder_gaps(agg_json: str, ref_json: str, k: int = 1) -> Dict[str, float]:
    """``g_task = SR(K, reference) - SR(k, ladder)`` from the baseline line's aggregate + reference."""
    agg = json.loads(pathlib.Path(agg_json).read_text())
    ref = json.loads(pathlib.Path(ref_json).read_text())
    per_task = agg["per_k"][str(k)]["per_task"]
    gaps = {}
    for task, rec in per_task.items():
        r = ref["tasks"][task]
        sr_ref = r["sr"] if isinstance(r, dict) else float(r)
        gaps[task] = float(sr_ref) - float(rec["success_rate"])
    return gaps


# ------------------------------------------------------------------
# Admission
# ------------------------------------------------------------------


def read_rows(paths: List[pathlib.Path]) -> List[dict]:
    rows = []
    for p in paths:
        for line in p.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                row["_rows_file"] = str(p)
                rows.append(row)
    return rows


def group_episodes(rows: List[dict]) -> Dict[tuple, dict]:
    """``{(task_uid, attempt): {"decisions": {idx: row}, "finalize": row|None, "dupes": n}}``."""
    eps: Dict[tuple, dict] = collections.defaultdict(lambda: {"decisions": {}, "finalize": None, "dupes": 0})
    for r in rows:
        key = (r["task_uid"], int(r.get("attempt", 1)))
        e = eps[key]
        if r.get("status") == "finalize":
            if e["finalize"] is not None:
                e["dupes"] += 1
            e["finalize"] = r
        else:
            idx = int(r["decision_idx"])
            if idx in e["decisions"]:
                e["dupes"] += 1
            e["decisions"][idx] = r
    return eps


def _arrays_for(row: dict, root: pathlib.Path, cache: dict) -> Optional[dict]:
    rel = row.get("arrays")
    digest = row.get("arrays_sha256")
    if not rel or not digest:
        return None
    path = root / rel
    cache_key = (path, digest)
    if cache_key in cache:
        return cache[cache_key]
    if not path.is_file():
        cache[path] = None
        return None
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != digest:
        cache[cache_key] = None
        return None
    with np.load(path, allow_pickle=False) as z:
        cache[cache_key] = {k: z[k] for k in z.files}
    return cache[cache_key]


def admit_and_measure(rows: List[dict], arrays_root: pathlib.Path, w, mask, *, h_exec: int, n_primary: int,
                      min_coverage: float, min_episodes: int, journal_accepted: Optional[set] = None) -> dict:
    """Admission ledger + per-decision metrics (see module docstring)."""
    eps = group_episodes(rows)
    reasons: collections.Counter = collections.Counter()
    decisions_out: List[dict] = []
    episodes_out: List[dict] = []
    cache: dict = {}
    for (task_uid, attempt), e in sorted(eps.items()):
        fin = e["finalize"]
        if journal_accepted is not None and (task_uid, attempt) not in journal_accepted:
            reasons["not_accepted_terminal"] += 1
            continue
        if fin is None or not fin.get("terminal"):
            reasons["no_terminal_finalize"] += 1
            continue
        if e["dupes"]:
            reasons["duplicate_rows"] += 1
            continue
        identity_keys = ("config_sha", "env_id", "experiment_id", "task", "init_idx", "env_seed", "lane", "pin_id")
        if (not fin.get("config_sha") or fin.get("init_idx") is None
                or (fin.get("env_id", "").endswith("_rc") and fin.get("env_seed") is None)
                or any(any(r.get(k) != fin.get(k) for k in identity_keys) for r in e["decisions"].values())):
            reasons["unbound_identity"] += 1
            continue
        n = int(fin.get("n_decisions", 0))
        idxs = sorted(e["decisions"])
        if idxs != list(range(n)):
            reasons["decision_index_gap"] += 1
            continue
        task = fin["task"]
        valid = []
        for idx in idxs:
            r = e["decisions"][idx]
            env = _envs.resolve_env(r["env_id"])
            if (r.get("k_set") != list(env.k_set) or r.get("k_full") != env.k_full
                    or r.get("n_primary") != 4 or r.get("warm_ts") != list(env.warm_ts)):
                reasons["sampling_contract_mismatch"] += 1
                continue
            if r.get("status") != "ok":
                reasons["decision_error_row"] += 1
                continue
            arrs = _arrays_for(r, arrays_root, cache)
            if arrs is None:
                reasons["arrays_missing_or_sha_mismatch"] += 1
                continue
            tag = f"{idx:04d}"
            a_full = arrs.get(f"a_full_{tag}")
            expected_full = 32 if r.get("dense") else 4
            if r.get("dense") != R.is_dense_decision(r["env_id"], task, fin["init_idx"], idx):
                reasons["dense_selection_mismatch"] += 1
                continue
            if a_full is None or a_full.shape != (expected_full, env.action_horizon, env.action_dim):
                reasons["insufficient_full_samples"] += 1
                continue
            if (r.get("n_full") != expected_full or len(r.get("noise_ids", [])) != expected_full
                    or len(set(r["noise_ids"])) != expected_full):
                reasons["noise_contract_mismatch"] += 1
                continue
            want_noise = [R.noise_seed(r["experiment_id"], r["env_id"], task,
                (fin.get("env_seed"), fin["init_idx"], fin.get("init_pool_sha256")), attempt, idx, i)
                for i in range(expected_full)]
            if r["noise_ids"] != want_noise:
                reasons["noise_identity_mismatch"] += 1
                continue
            a_k = {}
            missing_k = False
            for k in r.get("k_set", []):
                arr = arrs.get(f"a_k{int(k)}_{tag}")
                if arr is None or arr.shape != (4, env.action_horizon, env.action_dim):
                    missing_k = True
                    break
                a_k[int(k)] = list(arr)
            if missing_k:
                reasons["missing_paired_k_samples"] += 1
                continue
            a_warm = {}
            warm_bad = False
            for t in r.get("warm_ts", []):
                key = f"{float(t):.4f}"
                a_warm[float(t)] = arrs.get(f"a_warm_{key}_{tag}")
                status = r.get("warm_status", {}).get(key)
                value = a_warm[float(t)]
                if (status not in ("ok", "no_candidate", "no_snapshot")
                        or (status == "ok") != (value is not None)
                        or (value is not None and value.shape != (env.action_horizon, env.action_dim))):
                    warm_bad = True
            values = [a_full, *(np.asarray(v) for v in a_k.values()), *(v for v in a_warm.values() if v is not None)]
            if warm_bad or not all(np.isfinite(v).all() for v in values):
                reasons["invalid_sample_array"] += 1
                continue
            try:
                m = M.decision_metrics(list(a_full), a_k, a_warm, w, mask, h_exec, n_primary=n_primary)
            except ValueError:
                reasons["metric_error"] += 1
                continue
            if not all(math.isfinite(v) for v in [m["disp_K"], *m["d"].values(),
                                                 *(v for v in m["d_w"].values() if v is not None)]):
                reasons["non_finite"] += 1
                continue
            rec = {"task_uid": task_uid, "attempt": attempt, "task": task, "init_idx": fin.get("init_idx"),
                   "env_seed": fin.get("env_seed"), "decision_idx": idx, "dense": bool(r.get("dense")),
                   "top1_score": r.get("top1_score"), "outcome": fin.get("outcome"), **m}
            if r.get("dense") and a_full.shape[0] > n_primary:
                rec["mixture"] = M.mixture_fit(list(a_full), w, mask, h_exec, seed=ANALYSIS_SEED)
            valid.append(rec)
        coverage = len(valid) / n if n else 0.0
        if coverage < min_coverage:
            reasons["episode_low_coverage"] += 1
            continue
        decisions_out.extend(valid)
        episodes_out.append({"task_uid": task_uid, "attempt": attempt, "task": task, "n_decisions": n,
                             "n_valid": len(valid), "coverage": coverage, "outcome": fin.get("outcome"),
                             "init_idx": fin.get("init_idx"), "env_seed": fin.get("env_seed"),
                             **{k: fin.get(k) for k in ("lane", "pin_id", "layout", "style", "config_sha", "env_id", "experiment_id")},
                             "top1_score_median": _median([d["top1_score"] for d in valid])})
    per_task = _task_table(decisions_out, episodes_out, min_episodes)
    return {"episodes": episodes_out, "decisions": decisions_out, "per_task": per_task,
            "rejections": dict(reasons), "n_episodes_seen": len(eps), "stratified": _stratified(eps, episodes_out)}


def _stratified(eps: Dict[tuple, dict], admitted: List[dict]) -> dict:
    """Missing-label rates split by episode outcome (labels may be missing at random or not)."""
    adm = {(e["task_uid"], e["attempt"]) for e in admitted}
    out: Dict[str, dict] = {}
    for key, e in eps.items():
        fin = e["finalize"]
        if fin is None or not fin.get("terminal"):
            continue
        stratum = {True: "success", False: "failure"}.get(fin.get("outcome"), "unknown")
        rec = out.setdefault(stratum, {"episodes": 0, "admitted": 0, "decisions": 0, "error_rows": 0})
        rec["episodes"] += 1
        rec["admitted"] += int(key in adm)
        rows = list(e["decisions"].values())
        rec["decisions"] += len(rows)
        rec["error_rows"] += sum(1 for r in rows if r.get("status") != "ok")
    for rec in out.values():
        rec["error_rate"] = (rec["error_rows"] / rec["decisions"]) if rec["decisions"] else None
        rec["rejected"] = rec["episodes"] - rec["admitted"]
    return out


def authoritative_stratified(arm: dict, rows: List[dict], admitted: List[dict]) -> dict:
    """Use the frozen driver cohort, including absent server evidence, and exclude retries.

    Outcome comes from the accepted conductor terminal. Missing or conflicting authority
    has its own unknown stratum; observed-row error rates never stand in for missing rows.
    """
    eps = group_episodes(rows)
    adm = {(e["task_uid"], e["attempt"]): e for e in admitted}
    out = {}
    for uid in arm["expected"]:
        rec = arm["outcomes"].get(uid) or {}
        key = (uid, rec.get("attempt"))
        launches = [launch for launch in arm["launches"].values()
                    if rec.get("run_id") and launch.get("driver_run_id") == rec["run_id"]
                    and uid in {i["task_uid"] for i in launch["expected"]}]
        known = (bool(launches) and uid not in arm["conflicts"] and not rec.get("error")
                 and type(rec.get("success")) is bool)
        stratum = ("success" if rec["success"] else "failure") if known else "unknown"
        s = out.setdefault(stratum, {"episodes": 0, "admitted": 0, "decisions": 0, "error_rows": 0,
            "expected_decisions": 0, "missing_decision_rows": 0, "valid_labels": 0,
            "unknown_decision_count_episodes": 0, "missing_finalize_episodes": 0})
        ep = eps.get(key) or {"decisions": {}, "finalize": None}
        decisions = list(ep["decisions"].values())
        s["episodes"] += 1
        s["admitted"] += int(key in adm)
        s["valid_labels"] += adm.get(key, {}).get("n_valid", 0)
        s["decisions"] += len(decisions)
        s["error_rows"] += sum(r.get("status") != "ok" for r in decisions)
        s["missing_finalize_episodes"] += int(not (ep["finalize"] or {}).get("terminal"))
        summ = arm["summaries"].get(uid, {}).get(rec.get("attempt")) or {}
        n = summ.get("n_decisions")
        if (known and summ.get("run_id") == rec.get("run_id") and summ.get("success") == rec.get("success")
                and not summ.get("error") and type(n) is int and n > 0):
            s["expected_decisions"] += n
            s["missing_decision_rows"] += len(set(range(n)) - set(ep["decisions"]))
        else:
            s["unknown_decision_count_episodes"] += 1
    for s in out.values():
        s["rejected"] = s["episodes"] - s["admitted"]
        s["error_rate"] = s["error_rows"] / s["decisions"] if s["decisions"] else None
        s["episode_missing_rate"] = s["rejected"] / s["episodes"]
    return out


def _median(values) -> Optional[float]:
    vals = M.finite(values)
    return None if not vals else float(np.median(vals))


def _task_table(decisions: List[dict], episodes: List[dict], min_episodes: int) -> Dict[str, dict]:
    by_task: Dict[str, dict] = collections.defaultdict(lambda: {"episodes": collections.defaultdict(list)})
    for d in decisions:
        by_task[d["task"]]["episodes"][(d["task_uid"], d["attempt"])].append(d)
    out = {}
    for task, rec in sorted(by_task.items()):
        eps = rec["episodes"]
        ks = sorted({k for e in eps.values() for d in e for k in d["d"]})
        ts = sorted({t for e in eps.values() for d in e for t in d["d_w"]})
        row: dict = {"n_episodes": len(eps), "n_decisions": sum(len(e) for e in eps.values()),
                     "published": len(eps) >= min_episodes, "d": {}, "r": {}, "d_w": {}}
        # per-episode medians first, then the median across episodes (long episodes do not dominate)
        row["disp_K"] = _median([_median([d["disp_K"] for d in e]) for e in eps.values()])
        for k in ks:
            row["d"][k] = _median([_median([d["d"][k] for d in e]) for e in eps.values()])
            row["r"][k] = _median([_median([d["r"][k] for d in e]) for e in eps.values()])
        for t in ts:
            row["d_w"][f"{t:.4f}"] = _median([_median([d["d_w"][t] for d in e]) for e in eps.values()])
        dense = [d["mixture"] for e in eps.values() for d in e if d.get("mixture")]
        bic = M.finite([m.get("delta_bic_2_vs_1") for m in dense])
        row["dense"] = {"n": len(dense), "n_fitted": len(bic),
                        "frac_delta_bic_gt_10": (float(np.mean([b > 10 for b in bic])) if len(bic) >= 8 else None),
                        "pca_explained_2d_median": _median([m.get("pca_explained_2d") for m in dense])}
        successes = [ep["outcome"] for ep in episodes if ep["task"] == task and ep["outcome"] is not None]
        row["sr_shadow"] = (float(np.mean([bool(s) for s in successes])) if successes else None)
        out[task] = row
    return out


# ------------------------------------------------------------------
# Correlation + LOTO
# ------------------------------------------------------------------


def spearman(x: List[float], y: List[float]) -> Optional[float]:
    if len(x) < 3 or len(set(x)) < 2 or len(set(y)) < 2:
        return None
    from scipy.stats import spearmanr

    return float(spearmanr(x, y).correlation)


def correlation(per_task: Dict[str, dict], gaps: Dict[str, float], k: int = 1, boot: int = 20000,
                seed: int = ANALYSIS_SEED) -> dict:
    """Task-cluster bootstrap of Spearman rho(median d_k, g) with the pre-registered reading."""
    tasks = sorted(t for t in per_task if per_task[t]["published"] and t in gaps and per_task[t]["d"].get(k) is not None)
    x = [per_task[t]["d"][k] for t in tasks]
    y = [gaps[t] for t in tasks]
    rho = spearman(x, y)
    out = {"k": k, "tasks": tasks, "d": x, "g": y, "rho": rho, "n": len(tasks)}
    if rho is None or len(tasks) < 10:
        out["reading"] = "no_conclusion_constant_or_too_few"
        return out
    rng = np.random.default_rng(seed)
    idx = np.arange(len(tasks))
    samples = []
    for _ in range(boot):
        b = rng.choice(idx, size=len(idx), replace=True)
        r = spearman([x[i] for i in b], [y[i] for i in b])
        if r is not None:
            samples.append(r)
    lo, hi = (float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))) if samples else (None, None)
    out.update({"boot": boot, "ci95": [lo, hi], "gap_range": float(max(y) - min(y))})
    if out["gap_range"] < CLIFF_GAP:
        out["reading"] = "no_conclusion_gap_range"
    elif rho >= 0.6 and lo is not None and lo > 0.2:
        out["reading"] = "consistent_association"
    elif rho < 0.4:
        out["reading"] = "no_conclusion"
    else:
        out["reading"] = "grey"
    return out


def loto(per_task: Dict[str, dict], gaps: Dict[str, float], k: int = 1) -> dict:
    """Leave-one-task-out threshold on d_k for cliff (g >= 0.15): a feasibility table, not a rule."""
    tasks = sorted(t for t in per_task if per_task[t]["published"] and t in gaps and per_task[t]["d"].get(k) is not None)
    d = {t: per_task[t]["d"][k] for t in tasks}
    cliff = {t: gaps[t] >= CLIFF_GAP for t in tasks}
    rows = []
    for held in tasks:
        train = [t for t in tasks if t != held]
        if len(set(cliff[t] for t in train)) < 2:
            rows.append({"held": held, "threshold": None, "pred_cliff": None, "true_cliff": cliff[held]})
            continue
        ds = sorted(d[t] for t in train)
        cands = [-math.inf, *[(a + b) / 2 for a, b in zip(ds, ds[1:])], math.inf]
        best, best_acc = None, -1.0
        for th in cands:
            tp = sum(1 for t in train if cliff[t] and d[t] > th)
            fn = sum(1 for t in train if cliff[t] and d[t] <= th)
            tn = sum(1 for t in train if not cliff[t] and d[t] <= th)
            fp = sum(1 for t in train if not cliff[t] and d[t] > th)
            acc = 0.5 * (tp / max(tp + fn, 1) + tn / max(tn + fp, 1))
            if acc > best_acc or (acc == best_acc and best is not None and th < best):
                best, best_acc = th, acc
        rows.append({"held": held, "threshold": best, "pred_cliff": d[held] > best, "true_cliff": cliff[held]})
    confusion = collections.Counter((r["pred_cliff"], r["true_cliff"]) for r in rows if r["threshold"] is not None)
    return {"rows": rows, "missed_cliffs": confusion[(False, True)], "false_cliffs": confusion[(True, False)],
            "hits": confusion[(True, True)], "correct_flat": confusion[(False, False)]}


# ------------------------------------------------------------------
# Report
# ------------------------------------------------------------------


def markdown(env_id: str, res: dict, corr: Optional[dict], loto_tab: Optional[dict]) -> str:
    strat = res.get("stratified") or {}
    lines = [f"## shadow diagnostics — {env_id}", "",
             f"episodes seen {res['n_episodes_seen']}, admitted {len(res['episodes'])}; rejections: "
             + (", ".join(f"{k}={v}" for k, v in sorted(res["rejections"].items())) or "none"),
             "missing labels by outcome: " + ("; ".join(
                 f"{k}: {v['episodes']} eps ({v['rejected']} rejected), error rows {v['error_rows']}/{v['decisions']}, "
                 f"missing decision rows {v.get('missing_decision_rows', 'n/a')}, "
                 f"unknown decision counts {v.get('unknown_decision_count_episodes', 'n/a')} eps"
                 for k, v in sorted(strat.items())) or "n/a"), "",
             "| task | eps | decisions | disp_K | " + " | ".join(f"d_{k}" for k in _all_ks(res)) + " | "
             + " | ".join(f"r_{k}" for k in _all_ks(res)) + " | ΔBIC>10 frac (n dense) | SR(shadow) |",
             "|---|---|---|---|" + "---|" * (2 * len(_all_ks(res))) + "---|---|"]
    for task, row in res["per_task"].items():
        f = lambda v: "–" if v is None else f"{v:.3f}"  # noqa: E731
        lines.append(f"| {task}{'' if row['published'] else ' (unpublished)'} | {row['n_episodes']} | {row['n_decisions']} | "
                     f"{f(row['disp_K'])} | " + " | ".join(f(row["d"].get(k)) for k in _all_ks(res)) + " | "
                     + " | ".join(f(row["r"].get(k)) for k in _all_ks(res))
                     + f" | {f(row['dense']['frac_delta_bic_gt_10'])} ({row['dense']['n']}) | {f(row['sr_shadow'])} |")
    if corr:
        lines += ["", f"Spearman rho(d_{corr['k']}, g) = {corr.get('rho')} n={corr['n']} ci95={corr.get('ci95')} "
                      f"gap_range={corr.get('gap_range')} → **{corr['reading']}**"]
    if loto_tab:
        lines += ["", f"LOTO (descriptive): hits {loto_tab['hits']}, missed cliffs {loto_tab['missed_cliffs']}, "
                      f"false cliffs {loto_tab['false_cliffs']}, correct flat {loto_tab['correct_flat']}"]
    return "\n".join(lines) + "\n"


def _all_ks(res: dict) -> List[int]:
    return sorted({k for row in res["per_task"].values() for k in row["d"]})


def accepted_shadow_episodes(driver_dir: pathlib.Path, rows_dir: pathlib.Path, env_id: str,
                             weights_info: dict) -> tuple[set, dict]:
    """Independent conductor terminal authority, including server resource and worker bindings."""
    from exp.step_diag.analysis import aggregate_arms as B

    arm = B.load_arm(driver_dir)
    server = B.load_server_rows(rows_dir)
    env = _envs.resolve_env(env_id)
    accepted, rejected = set(), collections.Counter()
    if (len({x.get("config_sha") for x in arm["launches"].values()}) != 1
            or len({x.get("experiment_id") for x in arm["launches"].values()}) != 1):
        return set(), {"mixed_shadow_configuration": len(arm["expected"])}
    design = collections.defaultdict(list)
    for ident in arm["expected"].values():
        design[ident["task"]].append(ident)
    if weights_info.get("env_id") != env_id or not weights_info.get("library_sha256"):
        raise ValueError("weights must bind this environment and a library content digest")
    for uid, ident in arm["expected"].items():
        task_design = design[ident["task"]]
        if len(task_design) != 10 or {i.get("init_idx") for i in task_design} != set(range(10)):
            rejected["shadow_identity_set_mismatch"] += 1
            continue
        rec = arm["outcomes"].get(uid)
        if not rec:
            rejected["missing_terminal"] += 1
            continue
        key = (uid, rec.get("attempt"))
        ep = server.get(key, {})
        manifest = ep.get("manifest") or {}
        if (manifest.get("env", {}).get("env_id") != env_id
                or manifest.get("library_sha256") != weights_info["library_sha256"]):
            rejected["weights_or_environment_binding"] += 1
            continue
        idx = ident.get("init_idx")
        if type(idx) is not int or idx not in range(10):
            rejected["unexpected_init"] += 1
            continue
        if env.benchmark == "robocasa365":
            valid = (ident.get("task") in (*_envs.RC_MAIN_LANE, *_envs.RC_PNP_LANE)
                     and ident.get("env_seed") == _envs.RC_FORMAL_BASE_SEED + idx)
        else:
            valid = bool(ident.get("init_pool_sha256")) and ident.get("env_seed") == 7
        if not valid:
            rejected["environment_identity_mismatch"] += 1
            continue
        one = {**arm, "expected": {uid: ident}}
        audit = B.cell_admission(one, server, ident["task"], kind="plain", m=env.k_full, env_id=env_id)
        problems = {k: v for k, v in audit["problems"].items() if k != "decision_error"}
        if audit["complete"] and not problems:
            accepted.add(key)
        else:
            rejected.update(problems or {"incomplete": 1})
    return accepted, dict(rejected)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    fw = sub.add_parser("freeze-weights")
    fw.add_argument("--library", required=True)
    fw.add_argument("--env-id", required=True)
    fw.add_argument("--out", required=True)
    gp = sub.add_parser("gaps")
    gp.add_argument("--agg", required=True)
    gp.add_argument("--ref", required=True)
    gp.add_argument("--k", type=int, default=1)
    gp.add_argument("--out", required=True)
    an = sub.add_parser("analyze")
    an.add_argument("--env-id", required=True)
    an.add_argument("--rows", nargs="+", required=True)
    an.add_argument("--weights", required=True)
    an.add_argument("--gaps", default=None)
    an.add_argument("--driver-dir", required=True, help="conductor launch/journal/per_step directory for this shadow arm")
    an.add_argument("--n-primary", type=int, default=4)
    an.add_argument("--min-coverage", type=float, default=0.9)
    an.add_argument("--min-episodes", type=int, default=8)
    an.add_argument("--boot", type=int, default=20000)
    an.add_argument("--out-json", required=True)
    an.add_argument("--out-md", default=None)
    a = ap.parse_args()
    if a.cmd == "freeze-weights":
        print(json.dumps(freeze_weights(a.library, a.env_id, a.out)))
        return
    if a.cmd == "gaps":
        gaps = ladder_gaps(a.agg, a.ref, a.k)
        pathlib.Path(a.out).write_text(json.dumps(gaps, indent=1, sort_keys=True))
        print(f"gaps -> {a.out} ({len(gaps)} tasks)")
        return
    paths = []
    for r in a.rows:
        p = pathlib.Path(r)
        paths.extend(sorted(p.glob("rows_*.jsonl")) if p.is_dir() else [p])
    rows = read_rows(paths)
    w, mask, winfo = load_weights(a.weights)
    if not paths:
        raise ValueError("no server rows")
    if a.n_primary != 4 or a.min_coverage != .9 or a.min_episodes != 8:
        raise ValueError("formal analysis requires N=4, coverage=.9 and eight admitted episodes")
    if a.boot != 20000:
        raise ValueError("the exploratory correlation freezes bootstrap=20000")
    if len({p.parent.resolve() for p in paths}) != 1:
        raise ValueError("one environment's rows must be pulled into one directory")
    env = _envs.resolve_env(a.env_id)
    if not torch.equal(mask, M.executed_mask(env.action_dim, env.n_executed)) or not torch.isfinite(w).all():
        raise ValueError("weights/mask do not match the frozen executed dimensions")
    accepted, authority_rejections = accepted_shadow_episodes(pathlib.Path(a.driver_dir), paths[0].parent,
                                                              a.env_id, winfo)
    res = admit_and_measure(rows, paths[0].parent, w, mask, h_exec=_envs.H_EXEC, n_primary=a.n_primary,
                            min_coverage=a.min_coverage, min_episodes=a.min_episodes, journal_accepted=accepted)
    res["rejections"].update({f"authority_{k}": v for k, v in authority_rejections.items()})
    from exp.step_diag.analysis import aggregate_arms as B
    arm = B.load_arm(pathlib.Path(a.driver_dir))
    res["n_server_episodes_seen"] = res["n_episodes_seen"]
    res["n_episodes_seen"] = len(arm["expected"])
    res["stratified"] = authoritative_stratified(arm, rows, res["episodes"])
    from exp.step_diag import evidence
    manifests = evidence.load_manifests(paths[0].parent)
    for ep in res["episodes"]:
        ep["comparison_identity"] = evidence.comparison_identity(manifests[ep["config_sha"]])
    corr = loto_tab = None
    if a.gaps:
        gaps = json.loads(pathlib.Path(a.gaps).read_text())
        corr = correlation(res["per_task"], gaps, k=1, boot=a.boot)
        loto_tab = loto(res["per_task"], gaps, k=1)
    out = {"env_id": a.env_id, "weights": winfo, "rejections": res["rejections"], "n_episodes_seen": res["n_episodes_seen"],
           "n_server_episodes_seen": res["n_server_episodes_seen"],
           "stratified": res["stratified"], "episodes": res["episodes"], "per_task": res["per_task"],
           "correlation": corr, "loto": loto_tab, "decisions": res["decisions"]}
    pathlib.Path(a.out_json).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(a.out_json).write_text(json.dumps(out, indent=1, default=str))
    if a.out_md:
        pathlib.Path(a.out_md).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(a.out_md).write_text(markdown(a.env_id, res, corr, loto_tab))
    print(f"ANALYZE_SHADOW {a.env_id}: admitted {len(res['episodes'])}/{res['n_episodes_seen']} episodes, "
          f"{len(res['decisions'])} decisions; rho={None if not corr else corr.get('rho')} -> {a.out_json}")


if __name__ == "__main__":
    main()
