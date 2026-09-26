"""Paired analysis of warm reset entry runs, in the step_diag analysis shapes.

Reads one or more run directories of ONE environment (segments of one
experiment, merged): ``plan.json`` (arms, tasks, rollout; YAMLs re-verified),
``execution.json`` (the dispatched EpisodeTasks) and ``admission.json`` (the
per-episode verdict of ``exp.warm_reset.run admit``, whose server evidence and
journal / per-step joins it already checked). Every dispatched episode becomes
one outcome keyed by its pairing identity -- task, the environment adapter's
fields (LIBERO: init idx, env seed, init pool; RoboCasa: init idx, env seed,
lane, pin id, layout, style) and the env id -- with its measured NFE and
decision count, and the statistics are those of
``exp.step_diag.analysis.warm_variants`` / ``success_length`` (imported, not
copied), so the JSON / Markdown have the same shapes the report pipeline reads:

* ``--out-json`` / ``--out-md``: ``{"m<m>": panel}`` per equal-continuation
  panel, each ``tasks -> {cells, paired_deltas, comparison_problems, ...}`` plus
  ``macro`` (``warm_variants._macro``) -- the ``analyze_libero`` layout;
* ``--success-length-out``: ``{"decisions": {"<env>_m<m>": ...}}``
  (``success_length.analyse``; decisions only -- the journal carries no
  environment-step count, so the ``env_steps`` unit is not produced).

A cell is ``complete`` when every planned episode of that arm x task in the
loaded runs was admitted, and ``equal_nfe`` when every admitted episode
executed exactly the arm's continuation budget per decision (server-measured;
the ``warm_t*`` exact resume is a worker-only reference, reported as declared).
Runs whose model / rollout contract differs, or one arm frozen differently in
two runs, are reported and kept out of the macro.

Framework cells never pair with step_diag cells unless ``--allow-cross-framework``
is given: step_diag arms loaded with ``--step-diag-arms`` appear as
``step_diag:<arm>`` cells, and only the same-arm framework-vs-step_diag
difference is computed, labelled ``[cross-framework]`` and never in a macro.
Descriptive only.

usage::

    python -m exp.warm_reset.analysis --run-dir RUN_A --run-dir RUN_B \\
        --out-json wr.json --out-md wr.md --success-length-out wr_len.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
from typing import Optional

import numpy as np

from exp.step_diag import envs as E
from exp.step_diag.analysis import aggregate_arms as A
from exp.step_diag.analysis import success_length as SL
from exp.step_diag.analysis import warm_variants as W
from exp.warm_reset.conductor import manifest_digest
from exp.warm_reset.envs import RoboCasaAdapter, get_env
from exp.warm_reset.plan import KIND_MISS, arm_kind_of, miss_steps_of, read_plan
from openpi.conductor.task import EpisodeTask

FRAMEWORK = "warm_reset"
STEP_DIAG = "step_diag"
SD_PREFIX = "step_diag:"


# ------------------------------------------------------------------
# Loading
# ------------------------------------------------------------------


def _sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode("utf-8")).hexdigest()


def load_run(root: pathlib.Path) -> dict:
    """One admitted run: plan (YAMLs verified), dispatched tasks and its admission."""
    root = pathlib.Path(root)
    plan = read_plan(root)
    execution = json.loads((root / "execution.json").read_text())
    if execution.get("manifest_sha256") != manifest_digest(plan) or execution.get("token") != plan["token"]:
        raise ValueError(f"{root}: execution does not belong to this plan")
    path = root / "admission.json"
    if not path.is_file():
        raise ValueError(f"{root}: no admission.json; run `python -m exp.warm_reset.run admit` first")
    admission = json.loads(path.read_text())
    if admission.get("schema") != "warm_reset_admission_v1" or admission.get("run_id") != execution.get("run_id"):
        raise ValueError(f"{root}: admission.json is not this execution's")
    return {"root": str(root), "plan": plan, "tasks": [EpisodeTask(**t) for t in execution["tasks"]],
            "admission": {r["task_uid"]: r for r in admission["episodes"]},
            "global_problems": dict(admission.get("global_problems") or {})}


def arm_steps(env, arm: str) -> int:
    """Continuation Euler steps per decision of an arm (its equal-NFE budget m)."""
    steps = miss_steps_of(env, arm)
    if steps is not None:
        return steps
    n = E.warm_steps_of(arm)
    return int(n) if n is not None else int(env.schedule.remaining_steps(E.warm_t_of(arm)))


def _is_self(arm: str) -> bool:
    try:
        return E.is_self_mode(E.warm_mode_of(arm))
    except ValueError:
        return False


def _contract(run: dict) -> str:
    """The run-level comparison contract: environment, schedule, rollout, pins and pool."""
    plan = run["plan"]
    rollout = {k: v for k, v in plan["rollout"].items() if k != "init_states_dir"}
    return _sha([plan["env_id"], plan["schedule_id"], plan["k"], rollout, plan.get("pin_id"),
                 plan.get("init_pool_sha256")])[:16]


def _arm_contract(run: dict, record: dict) -> str:
    """An arm's frozen definition, independent of its run token, evidence dir and namespace."""
    text = record["yaml"].replace(run["plan"]["evidence_dir"], "<evidence>")
    text = text.replace(run["plan"]["namespace"], "<namespace>")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def framework_arms(runs: list[dict], *, min_idx: int = 0) -> dict[str, dict]:
    """``arm -> {kind, steps, self_start, episodes: [..], contracts, run_contracts, problems}`` over every run."""
    env_ids = {run["plan"]["env_id"] for run in runs}
    if len(env_ids) != 1:
        raise ValueError(f"run dirs of different environments cannot merge: {sorted(env_ids)}")
    env = get_env(env_ids.pop())
    arms: dict[str, dict] = {}
    for run in runs:
        plan = run["plan"]
        records = {a["yaml_id"]: a for a in plan["arms"]}
        names = {t["task_id"]: t["name"] for t in plan["tasks"]}
        for task in run["tasks"]:
            record = records[task.yaml_id]
            aid = record["arm"]
            arm = arms.setdefault(aid, {"kind": arm_kind_of(record), "steps": arm_steps(env, aid),
                                        "self_start": _is_self(aid), "episodes": [], "contracts": set(),
                                        "run_contracts": set(), "problems": [],
                                        "server_measured": bool(record["spec_digest"])})
            arm["contracts"].add(_arm_contract(run, record))
            arm["run_contracts"].add(_contract(run))
            ident = {"task": names[task.task_id], "env_id": env.env_id, "lane": None, "pin_id": None,
                     "layout": None, "style": None, "init_pool_sha256": None,
                     **env.adapter.pairing(env, plan, task)}
            if int(ident["init_idx"]) < int(min_idx):
                continue
            rep = run["admission"].get(task.task_uid)
            arm["episodes"].append({"uid": task.task_uid, "run": run["root"], "ident": ident, "report": rep,
                                    "global_problems": bool(run["global_problems"])})
    for aid, arm in arms.items():
        if len(arm["contracts"]) > 1:
            arm["problems"].append("arm_contract_mismatch_across_runs")
    return arms


# ------------------------------------------------------------------
# Cells
# ------------------------------------------------------------------


def _mean(values) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return float(np.mean(vals)) if vals else None


def framework_cell(env, aid: str, arm: dict, task: str) -> dict:
    """The ``warm_variants`` cell of one framework arm on one task (with ``_outcomes``)."""
    episodes = [e for e in arm["episodes"] if e["ident"]["task"] == task]
    problems: dict[str, int] = {}
    outcomes, cont, selfs, totals, per_dec, self_per_dec, dec_success = {}, [], [], [], [], [], []
    for ep in episodes:
        rep = ep["report"]
        if rep is None or not rep.get("admitted") or ep["global_problems"]:
            problems["episode_not_admitted"] = problems.get("episode_not_admitted", 0) + 1
            continue
        outcomes[f"{ep['run']}::{ep['uid']}"] = {**ep["ident"], "success": bool(rep["success"]),
                                                  "attempt": rep.get("attempt")}
        n_dec = rep.get("n_decisions")
        if arm["kind"] == KIND_MISS:
            c = rep.get("miss_nfe")
            s = 0
        elif arm["server_measured"]:
            c, s = rep.get("continuation_nfe"), rep.get("self_start_nfe")
        else:
            c, s = (arm["steps"] * n_dec if n_dec else None), 0  # declared (worker reference)
        cont.append(c)
        selfs.append(s)
        totals.append(None if c is None or s is None else c + s)
        if c is not None and n_dec:
            per_dec.append(c / n_dec)
        if arm["self_start"] and s is not None and n_dec:
            self_per_dec.append(s / n_dec)
        if rep["success"] and n_dec:
            dec_success.append(n_dec)
    equal = bool(per_dec) and all(v == arm["steps"] for v in per_dec) and len(per_dec) == len(outcomes)
    if outcomes and not equal:
        problems["continuation_steps_mismatch"] = 1
    for p in arm["problems"]:
        problems[p] = problems.get(p, 0) + 1
    n = len(outcomes)
    k_self = _mean(self_per_dec) if arm["self_start"] else None
    steps_mean = _mean(per_dec)
    return {
        "n": n,
        "n_identities": len({A.pair_identity(v) for v in outcomes.values()}),
        "replicated": n != len({A.pair_identity(v) for v in outcomes.values()}),
        "sr": W._identity_sr({"_outcomes": outcomes}),
        "comparison_identities": sorted(arm["run_contracts"]),
        "worker_identities": [],
        "runtime_notes": [],
        "complete": bool(episodes) and n == len(episodes) and not arm["problems"],
        "equal_nfe": equal,
        "problems": sorted(problems),
        "miss_fraction": 1.0 if arm["kind"] == KIND_MISS else 0.0,
        "mean_executed_steps": steps_mean,
        "self_start": arm["self_start"],
        "nfe_per_decision": {"continuation": steps_mean, "self_start": k_self if arm["self_start"] else 0,
                             "total": None if steps_mean is None else steps_mean + (k_self or 0)},
        "episode_nfe_mean": {"continuation": _mean(cont), "self_start": _mean(selfs), "total": _mean(totals)},
        "decisions_per_success_mean": _mean(dec_success),
        "nfe_source": "server" if arm["server_measured"] else "declared (worker_reference)",
        "framework": FRAMEWORK,
        "stray_sessions": 0,
        "_outcomes": outcomes,
    }


def _identity_problems(cells: dict) -> list[str]:
    """One model / rollout contract across the task's arms, one pool and seed per init idx."""
    comp = [tuple(c["comparison_identities"]) for c in cells.values() if c["n"]]
    problems = [] if comp and all(len(v) == 1 and v == comp[0] for v in comp) else ["comparison_identities_mismatch"]
    idents = [[v for v in c["_outcomes"].values()] for c in cells.values()]
    return problems + W.cross_identity_problems(idents)


def _delta(deltas: dict, macro_pairs: dict, cells: dict, var: str, ref: str, task: str, cross_problems: list,
           rng: np.random.Generator, *, allow_cross: bool) -> None:
    """``var - ref``; a cross-framework pair only when allowed, labelled, never in a macro."""
    frameworks = {cells[var].get("framework", STEP_DIAG), cells[ref].get("framework", STEP_DIAG)}
    if len(frameworks) == 1:
        W._add_delta(deltas, macro_pairs, cells, var, ref, task, cross_problems, rng)
    elif allow_cross:
        deltas[f"{var} - {ref} [cross-framework]"] = W._boot_delta(W._paired(cells[var], cells[ref]), rng)


def panels(env, arm_ids) -> list[int]:
    """Equal-continuation panels m: the plain arms' budgets, else every warm budget present."""
    plain = sorted({arm_steps(env, a) for a in arm_ids if a.startswith("plain_k")})
    if plain:
        return plain
    return sorted({arm_steps(env, a) for a in arm_ids if a != "full"})


def _step_diag_cells(env, sd: dict, task: str, m: int, min_idx: int) -> dict:
    """step_diag cells of the requested arms on one task (``warm_variants._task_cells``), prefixed."""
    specs = [(aid, "plain" if miss_steps_of(env, aid) is not None else "warm", arm_steps(env, aid))
             for aid in sd["arms"] if aid == "full" or arm_steps(env, aid) == m]
    cells, _, _ = W._task_cells(sd["loaded"], sd["servers"], specs, task, min_idx,
                                env_id=sd["env_id"])
    return {f"{SD_PREFIX}{aid}": {**c, "framework": STEP_DIAG} for aid, c in cells.items()}


def analyze_runs(runs: list[dict], *, min_idx: int = 0, step_diag: Optional[dict] = None,
                 allow_cross_framework: bool = False) -> dict:
    """``{"m<m>": panel}`` over merged framework runs (see module docstring)."""
    arms = framework_arms(runs, min_idx=min_idx)
    env = get_env(runs[0]["plan"]["env_id"])
    tasks = sorted({(t["task_id"], t["name"]) for run in runs for t in run["plan"]["tasks"]})
    out = {}
    for m in panels(env, arms):
        rng = np.random.default_rng(W.SEED)
        specs = [(aid, "plain" if arm["kind"] == KIND_MISS else "warm", arm["steps"])
                 for aid, arm in arms.items() if aid == "full" or arm["steps"] == m]
        in_panel = [a for a, _, _ in specs]
        plain = f"plain_k{m}"
        ours = next((a for a in in_panel if a != "full" and miss_steps_of(env, a) is None
                     and E.warm_mode_of(a) == "warm"), None)
        res = {"env_id": env.env_id, "policy": env.policy, "m": m, "t": None, "min_idx": int(min_idx),
               "framework": FRAMEWORK, "runs": [r["root"] for r in runs], "tasks": {},
               "arms": in_panel, "missing_arms": [a for a in ("full", plain) if a not in arms]}
        macro_pairs: dict[str, dict[str, list[tuple]]] = {}
        for task_id, task in tasks:
            cells = {aid: framework_cell(env, aid, arms[aid], task) for aid in in_panel}
            cells = {aid: c for aid, c in cells.items() if c["n"] or c["problems"]}
            cross = _identity_problems(cells)
            if step_diag:
                cells.update(_step_diag_cells(env, step_diag, task, m, min_idx))
            deltas: dict = {}
            for var in [a for a in in_panel if a in cells and miss_steps_of(env, a) is None]:
                if _is_self(var):
                    refs = (var[len("self"):], "full", plain)
                elif var == ours:
                    refs = (plain, "full")
                else:
                    refs = (plain, ours, "full")
                for ref in refs:
                    if ref and ref in cells:
                        _delta(deltas, macro_pairs, cells, var, ref, task, cross, rng,
                               allow_cross=allow_cross_framework)
            for aid in list(in_panel):
                if aid in cells and f"{SD_PREFIX}{aid}" in cells:
                    _delta(deltas, macro_pairs, cells, aid, f"{SD_PREFIX}{aid}", task, cross, rng,
                           allow_cross=allow_cross_framework)
            for c in cells.values():
                c.pop("_outcomes", None)
            res["tasks"][task] = {"task_id": task_id, "cells": cells, "paired_deltas": deltas,
                                  "comparison_problems": cross, "comparison_notes": A.comparison_notes(cells)}
        complete = all(c["complete"] and c["equal_nfe"] for r in res["tasks"].values()
                       for aid, c in r["cells"].items() if not aid.startswith(SD_PREFIX))
        res["status"] = "complete" if complete and not res["missing_arms"] else "partial"
        if len(tasks) > 1:
            res["macro"] = W._macro(res, specs, macro_pairs, rng)
        out[f"m{m}"] = res
    return out


def success_lengths(runs: list[dict], *, min_idx: int = 0) -> dict:
    """``{"decisions": {"<env>_m<m>": success_length.analyse(...)}}`` against ``full``."""
    arms = framework_arms(runs, min_idx=min_idx)
    env = get_env(runs[0]["plan"]["env_id"])
    # RoboCasa panels are one teacher each (no env level), as in success_length.
    env_dir = "" if isinstance(env.adapter, RoboCasaAdapter) else env.env_id
    data: dict[str, dict] = {}
    problems: list[str] = []
    for aid, arm in arms.items():
        per: dict[tuple, list] = {}
        for ep in arm["episodes"]:
            rep = ep["report"]
            if rep is None or not rep.get("admitted") or ep["global_problems"]:
                problems.append(f"{ep['run']}:{ep['uid']}: not admitted")
                continue
            key = SL.identity_key(ep["ident"], env_dir)
            per.setdefault(key, []).append((bool(rep["success"]), None, int(rep["n_decisions"])))
        data[aid] = per
    if "full" not in data:
        return {"decisions": {}, "note": "no full arm: nothing to pair lengths against"}
    out = {}
    for m in panels(env, arms):
        in_panel = [a for a, arm in arms.items() if a != "full" and arm["steps"] == m]
        rng = np.random.default_rng(SL.SEED)
        out[f"{env.env_id}_m{m}"] = SL.analyse(data, problems + SL.identity_mismatches(data, ["full", *in_panel]),
                                               in_panel, SL.UNITS["decisions"], rng, vs_cache=True)
    return {"decisions": out}


def markdown(results: dict) -> str:
    """Markdown of ``analyze_runs``: the ``warm_variants`` body per panel with a framework header."""
    lines = []
    for res in results.values():
        lines += [f"## Warm reset entry runs — {res['env_id']} (m={res['m']}; descriptive, 95% paired bootstrap"
                  + (f"; init_idx >= {res['min_idx']}" if res.get("min_idx") else "") + ")", "",
                  f"status: **{res['status'].upper()}** (framework cells; runs: {len(res['runs'])})", ""]
        lines += W._markdown_body(res)
        lines += [("Framework cells (`warm_reset` entry, production server): NFE are server-measured per decision "
                   "(continuation + self start; `full` / `plain_k` the executed MISS steps); `warm_t*` is a worker-only "
                   "reference with declared steps. Pairs are environment identities (task, init idx, env seed, pool / "
                   "lane / pin / layout / style, env). Cells never pair with step_diag cells unless explicitly allowed "
                   "(`[cross-framework]`, never in a macro)."), ""]
    return "\n".join(lines)


def _step_diag_source(args, env) -> Optional[dict]:
    if not args.step_diag_arms:
        return None
    if not args.step_diag_root:
        raise ValueError("--step-diag-arms needs --step-diag-root")
    libero = not isinstance(env.adapter, RoboCasaAdapter)
    roots = [pathlib.Path(x) for x in args.step_diag_root.split(",") if x]
    servers = [pathlib.Path(x) for x in (args.step_diag_server_rows or "").split(",") if x]
    loaded, rows = {}, {}
    for aid in (a for a in args.step_diag_arms.split(",") if a):
        arm, srv = W._load_arm_roots(env.teacher, aid, roots, servers, env_dir=env.env_id if libero else "")
        if arm is not None:
            loaded[aid], rows[aid] = arm, srv
    return {"arms": list(loaded), "loaded": loaded, "servers": rows,
            "env_id": env.env_id if libero else None}


def main(argv: Optional[list[str]] = None) -> int:
    """CLI: see the module docstring."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", action="append", required=True, type=pathlib.Path,
                    help="an admitted run directory; repeat to merge segments of one environment")
    ap.add_argument("--out-json", required=True, type=pathlib.Path)
    ap.add_argument("--out-md", required=True, type=pathlib.Path)
    ap.add_argument("--success-length-out", type=pathlib.Path, default=None)
    ap.add_argument("--min-idx", type=int, default=0)
    ap.add_argument("--step-diag-root", default="", help="step_diag driver out roots (comma-separated)")
    ap.add_argument("--step-diag-server-rows", default="", help="step_diag server out roots (comma-separated)")
    ap.add_argument("--step-diag-arms", default="", help="step_diag arms to load as `step_diag:<arm>` cells")
    ap.add_argument("--allow-cross-framework", action="store_true",
                    help="compute framework - step_diag same-arm differences (labelled, never in a macro)")
    args = ap.parse_args(argv)
    runs = [load_run(r) for r in args.run_dir]
    env = get_env(runs[0]["plan"]["env_id"])
    results = analyze_runs(runs, min_idx=args.min_idx, step_diag=_step_diag_source(args, env),
                           allow_cross_framework=args.allow_cross_framework)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(results, indent=1))
    args.out_md.write_text(markdown(results))
    if args.success_length_out:
        args.success_length_out.write_text(json.dumps(success_lengths(runs, min_idx=args.min_idx), indent=1))
    for key, res in results.items():
        print(f"WARM_RESET_ANALYSIS {res['env_id']} {key}: status={res['status']} tasks={len(res['tasks'])} "
              f"missing={res['missing_arms']} -> {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
