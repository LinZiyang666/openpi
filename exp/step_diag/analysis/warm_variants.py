"""Warm-start continuation variants vs the formal Q-B arms (pi0.5 RoboCasa, 2026-09-21 follow-up).

Compares, on the tasks given, the equal-NFE arms ``full`` / ``plain_k<m>`` / ``warm_t<t>`` (formal Q-B
data) with ``warmreset_t<t>`` and ``warmshoot_t<t>`` (``exp.step_diag.pi05.warm_variant_stage3``).
Every cell goes through the same admission as the formal analysis (``aggregate_arms.cell_admission``:
complete accepted set, WARM_START on every decision, executed steps == m, one stage-3 call); the
outcomes are paired on the frozen environment identity (``aggregate_arms.pair_identity``: task, init_idx,
env_seed, lane, pin, layout, style, LIBERO pool, environment) and the paired difference of each variant against
``plain_k<m>`` and against ``warm_t<t>`` gets a percentile bootstrap interval (episodes resampled, 20000 draws,
seed 20260919). A task whose arms disagree on the model / environment contract, the initial-state pool or the
env seed of an init_idx is reported and kept out of the macro. Descriptive only: these arms were not in the
pre-registered family, so no verdict is issued.

Equal NFE here means equal *continuation* NFE: every arm of a panel executes m Euler steps per decision. A
self-start arm (``self<variant>``) also runs a K-step direct inference per decision to produce its start
(``self_direct_nfe``), so its action-head cost is K + m per decision against m for its cache arm; the cells
report continuation, self-start and total NFE separately.

With ``--env-id`` naming a LIBERO environment, the LIBERO self-start round (``analyze_libero``): the
suite's tasks, one report per equal-continuation-NFE panel m (pi0.5 m = 2; GR00T m = 1, 2). An arm is a
formal result only when it covers the frozen identity set of ``envs.LIBERO_SELF_EXPERIMENT_ID`` (10 tasks x
init_idx 0..49, env seed 7, one pool) with every cell admitted; the macro uses formal arms only, and a run in
flight gets a separate progress macro labelled partial.

usage::

    python -m exp.step_diag.analysis.warm_variants --policy pi05 --tasks CloseFridge,PickPlaceCounterToStove \
        --arms-root exp/step_diag/data/rc --server-rows exp/step_diag/data/server \
        --out-json exp/step_diag/data/analysis/warm_variants_pi05.json --out-md exp/step_diag/analysis/warm_variants_pi05.md
    python -m exp.step_diag.analysis.warm_variants --env-id groot_libero_10 \
        --arms-root exp/step_diag/data/libero_self --server-rows exp/step_diag/data/server_libero_self \
        --out-json exp/step_diag/data/analysis/libero_self_groot_libero_10.json \
        --out-md exp/step_diag/data/analysis/libero_self_groot_libero_10.md
"""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Dict, List

import numpy as np

from exp.step_diag import envs as _envs
from exp.step_diag.analysis import aggregate_arms as A

BOOT = 20000
SEED = 20260919


Z95 = 1.959963984540054


def _by_identity(cell: dict) -> Dict[tuple, List[int]]:
    """Outcomes grouped by frozen environment identity; a repeated evaluation keeps every replicate."""
    out: Dict[tuple, List[int]] = {}
    for v in cell["_outcomes"].values():
        out.setdefault(A.pair_identity(v), []).append(int(v["success"]))
    return out


def _identity_sr(cell: dict) -> float | None:
    """Mean over identities of the replicate mean (equals the pooled SR when replicates are balanced)."""
    g = _by_identity(cell)
    return float(np.mean([np.mean(r) for r in g.values()])) if g else None


def _paired(a: dict, b: dict) -> List[tuple]:
    """(a_i, b_i) per shared environment identity, each side the mean over its replicates.

    Order independent: when an arm was evaluated twice on the same identity (two out roots), both
    replicates enter the mean instead of the later one silently overwriting the earlier one.
    """
    ka, kb = _by_identity(a), _by_identity(b)
    keys = sorted(set(ka) & set(kb))
    return [(float(np.mean(ka[k])), float(np.mean(kb[k]))) for k in keys]


def _degenerate_interval(point: float, n: int) -> tuple:
    """Fallback when every paired difference is identical (bootstrap collapses to a point): the Wilson
    95% upper bound on the fraction of pairs that could disagree, z^2/(n+z^2), around the point."""
    u = Z95 * Z95 / (n + Z95 * Z95)
    return max(-1.0, point - u), min(1.0, point + u)


def _boot_delta(pairs: List[tuple], rng: np.random.Generator) -> dict:
    if not pairs:
        return {"n": 0, "point": None, "lower": None, "upper": None}
    arr = np.asarray(pairs, dtype=float)
    diff = arr[:, 0] - arr[:, 1]
    idx = rng.integers(0, len(diff), size=(BOOT, len(diff)))
    means = diff[idx].mean(axis=1)
    point = float(diff.mean())
    lower, upper = float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
    degenerate = bool(np.all(diff == diff[0]))
    if degenerate:
        lower, upper = _degenerate_interval(point, len(diff))
    return {"n": len(diff), "point": point, "lower": lower, "upper": upper, "degenerate": degenerate,
            "n10": int((arr[:, 0] > arr[:, 1]).sum()), "n01": int((arr[:, 0] < arr[:, 1]).sum())}


def _merge_arms(parts: List[dict]) -> dict:
    """Union of one arm's launches/outcomes over several out roots (disjoint task_uids by construction)."""
    out = {"dir": ";".join(p["dir"] for p in parts), "expected": {}, "outcomes": {}, "conflicts": set(), "summaries": {},
           "launch_ids": set(), "arm_id": parts[0]["arm_id"], "launches": {}, "manifest_problems": []}
    for p in parts:
        if p["arm_id"] != out["arm_id"]:
            raise ValueError(f"arm dirs of different arms: {out['arm_id']} vs {p['arm_id']}")
        for key in ("expected", "outcomes", "summaries", "launches"):
            dup = set(out[key]) & set(p[key])
            if dup:
                out["manifest_problems"].append(f"cross_root_duplicate_{key}")
            out[key].update(p[key])
        out["conflicts"] |= set(p["conflicts"]); out["launch_ids"] |= set(p["launch_ids"])
        out["manifest_problems"] += list(p["manifest_problems"])
    return out


def _load_arm_roots(teacher: str, aid: str, arms_roots: List[pathlib.Path], server_roots: List[pathlib.Path],
                    env_dir: str = ""):
    """One arm's cells under every root; LIBERO cells live one level deeper, in ``<teacher>/<env_id>/<arm>``."""
    parts, servers = [], {}
    for ar in arms_roots:
        d = pathlib.Path(ar) / teacher / env_dir / aid
        if d.is_dir() and any(d.glob("launch_*.json")):
            parts.append(A.load_arm(d))
    for sr in server_roots:
        d = pathlib.Path(sr) / teacher / env_dir / aid
        if d.is_dir():
            servers.update(A.load_server_rows(d))
    if not parts:
        return None, {}
    return (parts[0] if len(parts) == 1 else _merge_arms(parts)), servers


def _boot_macro(per_task_pairs: Dict[str, List[tuple]], rng: np.random.Generator) -> dict:
    """Macro paired difference: mean over tasks of the per-task mean; episodes resampled within task."""
    tasks = [t for t, p in per_task_pairs.items() if p]
    if not tasks:
        return {"n_tasks": 0, "point": None, "lower": None, "upper": None}
    diffs = {t: np.asarray(per_task_pairs[t], dtype=float)[:, 0] - np.asarray(per_task_pairs[t], dtype=float)[:, 1] for t in tasks}
    point = float(np.mean([d.mean() for d in diffs.values()]))
    acc = np.zeros(BOOT)
    for d in diffs.values():
        idx = rng.integers(0, len(d), size=(BOOT, len(d)))
        acc += d[idx].mean(axis=1)
    acc /= len(tasks)
    lower, upper = float(np.percentile(acc, 2.5)), float(np.percentile(acc, 97.5))
    degenerate = all(bool(np.all(d == d[0])) for d in diffs.values())
    if degenerate:
        lower, upper = _degenerate_interval(point, int(sum(len(d) for d in diffs.values())))
    return {"n_tasks": len(tasks), "point": point, "lower": lower, "upper": upper, "degenerate": degenerate}


def analyze(policy: str, tasks: List[str], arms_root, server_root, *, min_idx: int = 0,
            t: float | None = None) -> dict:
    arms_roots = list(arms_root) if isinstance(arms_root, (list, tuple)) else [arms_root]
    server_roots = list(server_root) if isinstance(server_root, (list, tuple)) else [server_root]
    teacher = "pi05" if policy == "pi05" else "groot_tp"
    env = _envs.ENVS[f"{policy}_rc"]
    t = _envs.QB_MAIN_T[policy] if t is None else float(t)
    if t not in _envs.QB_WARM_TS[policy]:
        raise ValueError(f"t must be one of {_envs.QB_WARM_TS[policy]}")
    m = int(env.schedule.remaining_steps(t))  # the resume's step count under the policy's own schedule direction
    arm_specs = [("full", "plain", env.k_full), (f"plain_k{m}", "plain", m), (f"warm_t{t:g}", "warm", m),
                 (f"warmreset_t{t:g}", "warm", m), (f"warmshoot_t{t:g}", "warm", m), (f"resetfinal_t{t:g}", "warm", m),
                 (f"midfinal_t{t:g}", "warm", m), (f"midfinal50_t{t:g}", "warm", m),
                 (f"midreset_t{t:g}", "warm", m), (f"midreset50_t{t:g}", "warm", m),
                 (f"midshoot_t{t:g}", "warm", m), (f"midshoot50_t{t:g}", "warm", m)]  # GR00T shoot ablation
    # self-start ablation (owner 2026-09-24): the same variants with the start from a direct inference, not the cache
    arm_specs += [(f"{mode}_t{t:g}", "warm", m) for mode in (*_envs.SELF_VARIANT_MODES, *_envs.SELF_SHOOT_MODES)
                  if f"{mode}_t{t:g}" in _envs.SELF13_ARMS_BY_POLICY[policy]]
    arms, servers = {}, {}
    for aid, _, _ in arm_specs:
        arm, srv = _load_arm_roots(teacher, aid, arms_roots, server_roots)
        if arm is not None:
            arms[aid], servers[aid] = arm, srv
    rng = np.random.default_rng(SEED)
    out = {"policy": policy, "m": m, "t": t, "min_idx": int(min_idx), "tasks": {},
           "missing_arms": [a for a, _, _ in arm_specs if a not in arms]}
    macro_pairs: Dict[str, Dict[str, List[tuple]]] = {}
    for task in tasks:
        cells, cross_problems, notes = _task_cells(arms, servers, arm_specs, task, min_idx)
        deltas = {}
        for var in [a for a, _, _ in arm_specs if a.startswith("self") and a in cells]:
            for ref in (var[len("self"):], "full", f"plain_k{m}"):  # the paired cache arm first
                if ref in cells:
                    _add_delta(deltas, macro_pairs, cells, var, ref, task, cross_problems, rng)
        for var in (f"warmreset_t{t:g}", f"warmshoot_t{t:g}", f"resetfinal_t{t:g}", f"midfinal_t{t:g}", f"midfinal50_t{t:g}", f"midreset_t{t:g}", f"midreset50_t{t:g}", f"midshoot_t{t:g}", f"midshoot50_t{t:g}", f"warm_t{t:g}"):
            if var not in cells:
                continue
            for ref in (f"plain_k{m}", f"warm_t{t:g}", f"warmreset_t{t:g}", f"resetfinal_t{t:g}", f"midfinal_t{t:g}", f"midreset_t{t:g}", "full"):
                if ref in cells and ref != var:
                    _add_delta(deltas, macro_pairs, cells, var, ref, task, cross_problems, rng)
        for c in cells.values():
            c.pop("_outcomes")
        out["tasks"][task] = {"cells": cells, "paired_deltas": deltas, "comparison_problems": cross_problems,
                              "comparison_notes": notes}
    if len(tasks) > 1:
        out["macro"] = _macro(out, arm_specs, macro_pairs, rng)
    return out


def _mean_of(values) -> float | None:
    vals = [v for v in values if v is not None]
    return float(np.mean(vals)) if vals else None


def cross_identity_problems(identity_sets: List[List[dict]]) -> List[str]:
    """Cross-arm identity gate of one task: every arm's declared identities must come from ONE initial-state pool
    (``init_pool_sha256``; None on RoboCasa) and name ONE env seed per init_idx. The same init_idx of another pool
    or seed is another initial state: it never pairs (``aggregate_arms.pair_identity``), and the task is reported
    and kept out of the macro instead of silently losing its pairs."""
    pools = {ident.get("init_pool_sha256") for idents in identity_sets for ident in idents}
    seeds: Dict[tuple, set] = {}
    for idents in identity_sets:
        for ident in idents:
            seeds.setdefault((ident.get("task"), ident.get("init_idx")), set()).add(ident.get("env_seed"))
    problems = []
    if len(pools) > 1:
        problems.append("init_pool_mismatch")
    if any(len(s) > 1 for s in seeds.values()):
        problems.append("env_seed_mismatch")
    return problems


def _task_cells(arms: dict, servers: dict, arm_specs: list, task: str, min_idx: int, env_id: str | None = None):
    """Admitted cells of one task (``aggregate_arms.cell_admission``), the cross-arm gate and the notes."""
    cells = {}
    for aid, kind, mm in arm_specs:
        if aid in arms:
            c = A.cell_admission(arms[aid], servers.get(aid, {}), task, kind=kind, m=mm, env_id=env_id)
            # sensitivity analysis: admission is judged on the whole cell, the estimate on init_idx >= min_idx
            c["outcomes"] = {u: v for u, v in c["outcomes"].items() if int(v["init_idx"]) >= int(min_idx)}
            n = len(c["outcomes"])
            n_ids = len({A.pair_identity(v) for v in c["outcomes"].values()})
            steps, self_nfe = c["mean_executed_steps"], c["mean_self_direct_nfe"]
            cells[aid] = {"n": n, "n_identities": n_ids, "replicated": n != n_ids,
                          "sr": _identity_sr({"_outcomes": c["outcomes"]}),
                          "comparison_identities": c["comparison_identities"],
                          "worker_identities": c["worker_identities"], "runtime_notes": c["runtime_notes"],
                          "complete": c["complete"], "equal_nfe": c["equal_nfe"], "problems": c["problems"],
                          "miss_fraction": c["miss_fraction"], "mean_executed_steps": c["mean_executed_steps"],
                          # action-head NFE per decision: the continuation (the panel's m), the self-start direct
                          # inference (K on a self arm, 0 otherwise) and their sum; per episode likewise
                          "self_start": c["self_start"],
                          "nfe_per_decision": {"continuation": steps, "self_start": self_nfe,
                                               "total": None if steps is None or self_nfe is None else steps + self_nfe},
                          "episode_nfe_mean": {"continuation": _mean_of(c["episode_continuation_nfe"].values()),
                                               "self_start": _mean_of(c["episode_self_start_nfe"].values()),
                                               "total": _mean_of(c["episode_total_nfe"].values())},
                          "stray_sessions": c["stray_sessions"], "_outcomes": c["outcomes"]}
    # cross-arm gate (as in the formal analysis): one model + environment contract across the task's arms
    comp = [tuple(c["comparison_identities"]) for c in cells.values() if c["n"]]  # arms not run on this task carry none
    cross_problems = [] if comp and all(len(v) == 1 and v == comp[0] for v in comp) else ["comparison_identities_mismatch"]
    # and one initial-state pool / one env seed per init_idx across the arms (the pairing identity)
    cross_problems += cross_identity_problems(
        [[v for v in arms[aid]["expected"].values() if v["task"] == task] for aid in cells])
    return cells, cross_problems, A.comparison_notes(cells)


def _add_delta(deltas: dict, macro_pairs: dict, cells: dict, var: str, ref: str, task: str, cross_problems: list,
               rng: np.random.Generator) -> None:
    """Paired difference ``var - ref`` on one task; admissible pairs also enter the macro."""
    pairs = _paired(cells[var], cells[ref])
    deltas[f"{var} - {ref}"] = _boot_delta(pairs, rng)
    if (not cross_problems and cells[var]["complete"] and cells[var]["equal_nfe"]
            and cells[ref]["complete"] and cells[ref]["equal_nfe"]):
        macro_pairs.setdefault(f"{var} - {ref}", {})[task] = pairs


def _macro(out: dict, arm_specs: list, macro_pairs: dict, rng: np.random.Generator) -> dict:
    """Equal-weight macro SR over the admissible tasks of every arm, and the macro paired differences."""
    macro_sr = {}
    for aid, _, _ in arm_specs:
        srs = [r["cells"][aid]["sr"] for r in out["tasks"].values()
               if aid in r["cells"] and not r["comparison_problems"] and r["cells"][aid]["complete"] and r["cells"][aid]["equal_nfe"] and r["cells"][aid]["sr"] is not None]
        macro_sr[aid] = {"n_tasks": len(srs), "macro_sr": float(np.mean(srs)) if srs else None}
    return {"sr": macro_sr, "paired_deltas": {k: _boot_macro(v, rng) for k, v in macro_pairs.items()}}


def libero_arm_specs(env_id: str, m: int) -> List[tuple]:
    """``(arm_id, kind, m)`` of one equal-continuation-NFE panel of the LIBERO self-start round: ``full``,
    ``plain_k<m>`` and every warm-family arm executing ``m`` Euler steps per decision (the exact resume, the cache
    and the self variants; a self arm also runs its K-step direct inference, K + m in total)."""
    specs = []
    for aid in _envs.LIBERO_SELF_ARMS_BY_POLICY[_envs.resolve_env(env_id).policy]:
        steps = _envs.executed_steps_of(env_id, aid)
        if aid == "full":
            specs.append((aid, "plain", steps))
        elif steps == m:
            specs.append((aid, "plain" if aid.startswith("plain_k") else "warm", steps))
    return specs


def libero_panels(env_id: str) -> tuple:
    """The continuation budgets m of an environment's LIBERO round: its plain arms' step counts (pi0.5 2, GR00T 1
    and 2)."""
    arms = _envs.LIBERO_SELF_ARMS_BY_POLICY[_envs.resolve_env(env_id).policy]
    return tuple(sorted(_envs.executed_steps_of(env_id, a) for a in arms if a.startswith("plain_k")))


def _libero_tasks(arms: dict) -> List[tuple]:
    """``(task_id, task)`` of every expected identity of the loaded arms, by task id (one name per id)."""
    names: Dict[int, set] = {}
    for arm in arms.values():
        for ident in arm["expected"].values():
            names.setdefault(int(ident["task_id"]), set()).add(ident["task"])
    clash = {tid: sorted(v) for tid, v in names.items() if len(v) != 1}
    if clash:
        raise ValueError(f"launch manifests name one LIBERO task id differently: {clash}")
    return [(tid, next(iter(v))) for tid, v in sorted(names.items())]


def libero_formal_problems(env_id: str, arm: dict) -> List[str]:
    """Why one loaded arm's manifests are not the frozen design of the LIBERO self-start round (empty = they are).

    Formal means: every launch belongs to ``envs.LIBERO_SELF_EXPERIMENT_ID`` (a smoke or any other experiment id
    is not), names this environment and env seed 7, and the declared identities are exactly the suite's
    ``LIBERO_N_TASKS`` task ids x init_idx ``0..LIBERO_SELF_EPISODES-1``, once each, env seed 7, from one
    initial-state pool. A task subset or a truncated manifest is a run in flight (partial), never formal.
    """
    problems = []
    launches = list(arm.get("launches", {}).values())
    if not launches or any(launch.get("experiment_id") != _envs.LIBERO_SELF_EXPERIMENT_ID for launch in launches):
        problems.append("experiment_not_formal")
    if any(launch.get("env_id") != env_id or launch.get("env_seed") != _envs.LIBERO_ENV_SEED for launch in launches):
        problems.append("launch_environment_mismatch")
    idents = list(arm.get("expected", {}).values())
    by_task: Dict[object, List[object]] = {}
    for ident in idents:
        by_task.setdefault(ident.get("task_id"), []).append(ident.get("init_idx"))
    frozen = list(range(_envs.LIBERO_SELF_EPISODES))
    if set(by_task) != set(range(_envs.LIBERO_N_TASKS)) or any(sorted(v) != frozen for v in by_task.values()):
        problems.append("identity_set_not_frozen")
    if any(ident.get("env_seed") != _envs.LIBERO_ENV_SEED for ident in idents):
        problems.append("env_seed_not_frozen")
    pools = {ident.get("init_pool_sha256") for ident in idents}
    if len(pools) != 1 or not next(iter(pools)):
        problems.append("init_pool_not_unique")
    return problems + list(arm.get("manifest_problems", []))


def _libero_formal(env_id: str, arms: dict, arm_specs: list, tasks: dict) -> dict:
    """Formal status of every arm of one panel; marks each loaded cell ``formal`` in place (see ``analyze_libero``)."""
    status, pools = {}, set()
    for aid, _, _ in arm_specs:
        if aid not in arms:
            status[aid] = {"formal": False, "problems": ["arm_missing"]}
            continue
        problems = libero_formal_problems(env_id, arms[aid])
        pools |= {ident.get("init_pool_sha256") for ident in arms[aid]["expected"].values()}
        for task, r in tasks.items():
            c = r["cells"].get(aid)
            if not (c and c["n"] > 0 and c["complete"] and c["equal_nfe"] and not r["comparison_problems"]):
                problems.append(f"cell_not_admitted:{task}")
        status[aid] = {"formal": not problems, "problems": problems}
    if len(pools) > 1:  # arms of one panel on different pools: none of their pairs is formal
        for rec in status.values():
            rec["formal"] = False
            rec["problems"].append("init_pool_mismatch_across_arms")
    for r in tasks.values():  # a cell is formal only as part of a formal arm
        for aid, c in r["cells"].items():
            c["formal"] = status[aid]["formal"]
        r["status"] = "formal" if r["cells"] and all(c["formal"] for c in r["cells"].values()) else "partial"
    complete = all(rec["formal"] for rec in status.values())
    return {"experiment_id": _envs.LIBERO_SELF_EXPERIMENT_ID, "complete": complete, "arms": status}


def _libero_macro(out: dict, arm_specs: list, macro_pairs: dict, rng: np.random.Generator, formal_arms: set) -> dict:
    """Formal macro: equal-weight SR over the suite's tasks of every formal arm, and the macro paired difference
    of every pair of formal arms. Arms that are not formal get no macro value here."""
    macro_sr = {}
    for aid, _, _ in arm_specs:
        srs = [r["cells"][aid]["sr"] for r in out["tasks"].values()] if aid in formal_arms else []
        macro_sr[aid] = {"n_tasks": len(srs), "macro_sr": float(np.mean(srs)) if srs else None}
    pairs = {k: v for k, v in macro_pairs.items() if all(a in formal_arms for a in k.split(" - "))}
    return {"status": "formal", "sr": macro_sr, "paired_deltas": {k: _boot_macro(v, rng) for k, v in pairs.items()}}


def analyze_libero(env_id: str, arms_root, server_root, *, m: int, min_idx: int = 0) -> dict:
    """LIBERO self-start round (plan §4.6), one environment and one equal-continuation-NFE panel ``m``.

    Tasks are the suite's tasks found in the arms' launch manifests; outcomes pair on the full environment
    identity (``aggregate_arms.pair_identity``: task, init_idx, env seed, initial-state pool, environment), and a
    task whose arms disagree on the pool or an env seed is reported (``comparison_problems``) and kept out of
    every macro. Per task: every self arm against its cache arm, ``full`` and ``plain_k<m>``; every cache variant
    against ``plain_k<m>``, the exact resume of the panel and ``full``; the exact resume against ``plain_k<m>``
    and ``full``. Admission is the formal one (``cell_admission`` with the LIBERO environment: WARM_START on
    every decision, m executed steps, one stage-3 call, and on a self arm the self-start evidence).

    Completeness (``formal``): an arm is a formal result only if ``libero_formal_problems`` is empty and every
    one of its task cells is admitted; ``status`` is ``formal`` when every arm of the panel is, else
    ``partial``. ``macro`` averages formal arms only (all suite tasks); while the round is in flight,
    ``macro_partial`` is the progress view over the admitted task subsets, explicitly not a formal result.
    Descriptive only.
    """
    env = _envs.resolve_env(env_id)
    if env.benchmark == "robocasa365" or m not in libero_panels(env_id):
        raise ValueError(f"{env_id} / m={m} is not a panel of the LIBERO self-start round")
    arms_roots = list(arms_root) if isinstance(arms_root, (list, tuple)) else [arms_root]
    server_roots = list(server_root) if isinstance(server_root, (list, tuple)) else [server_root]
    teacher = "pi05" if env.policy == "pi05" else "groot_tp"
    arm_specs = libero_arm_specs(env_id, m)
    arms, servers = {}, {}
    for aid, _, _ in arm_specs:
        arm, srv = _load_arm_roots(teacher, aid, arms_roots, server_roots, env_dir=env_id)
        if arm is not None:
            arms[aid], servers[aid] = arm, srv
    plain = f"plain_k{m}"
    ours = next(a for a, kind, _ in arm_specs if kind == "warm" and _envs.warm_mode_of(a) == "warm")
    rng = np.random.default_rng(SEED)
    out = {"env_id": env_id, "policy": env.policy, "m": m, "t": None, "min_idx": int(min_idx), "tasks": {},
           "arms": [a for a, _, _ in arm_specs], "missing_arms": [a for a, _, _ in arm_specs if a not in arms]}
    macro_pairs: Dict[str, Dict[str, List[tuple]]] = {}
    tasks = _libero_tasks(arms)
    for task_id, task in tasks:
        cells, cross_problems, notes = _task_cells(arms, servers, arm_specs, task, min_idx, env_id=env_id)
        deltas = {}
        for var in [a for a, kind, _ in arm_specs if kind == "warm" and a in cells]:
            if var.startswith("self"):
                refs = (var[len("self"):], "full", plain)  # the paired cache arm first
            elif var == ours:
                refs = (plain, "full")
            else:
                refs = (plain, ours, "full")
            for ref in refs:
                if ref in cells:
                    _add_delta(deltas, macro_pairs, cells, var, ref, task, cross_problems, rng)
        for c in cells.values():
            c.pop("_outcomes")
        out["tasks"][task] = {"task_id": task_id, "cells": cells, "paired_deltas": deltas,
                              "comparison_problems": cross_problems, "comparison_notes": notes}
    out["formal"] = _libero_formal(env_id, arms, arm_specs, out["tasks"])
    out["status"] = "formal" if out["formal"]["complete"] else "partial"
    formal_arms = {aid for aid, rec in out["formal"]["arms"].items() if rec["formal"]}
    if formal_arms:
        out["macro"] = _libero_macro(out, arm_specs, macro_pairs, rng, formal_arms)
    if not out["formal"]["complete"] and len(tasks) > 1:
        out["macro_partial"] = {**_macro(out, arm_specs, macro_pairs, rng),
                                "status": "partial (progress over admitted task subsets; not a formal result)"}
    return out


SELF_NFE_NOTE = ("NFE: `steps` is the continuation's Euler steps per decision (the equal-NFE gate); a `self<variant>` arm "
                 "also runs a K-step direct inference per decision to produce its start (`self_direct_nfe`, shown as "
                 "`+ K self`), so its action-head cost is K + m per decision against m for its cache arm. The self vs "
                 "cache comparison is at equal continuation, not equal total, NFE.")


def _has_self_arm(res: dict) -> bool:
    return any(c.get("self_start") for r in res["tasks"].values() for c in r["cells"].values())


def markdown(res: dict) -> str:
    """Markdown report of ``analyze`` (RoboCasa): per-task cell table and paired differences, then the macro."""
    m, t = res["m"], res["t"]
    lines = [f"## Warm-start continuation variants — {res['policy']} (m={m}, t={t:g}; descriptive, 95% paired bootstrap"
             + (f"; init_idx >= {res['min_idx']}" if res.get("min_idx") else "") + ")", ""]
    lines += _markdown_body(res)
    lines += ["Variants: `warmreset` restarts the flow time at 1 with dt = −1/remaining (the cache is fed as if it were noise); "
              "`warmshoot` keeps the cache's start_t with dt = −1/remaining (t crosses 0). Both run the same number of "
              "Euler steps as `warm_t` and `plain_k`; `resetfinal` is the `warmreset` loop started from the cache's final action chunk (t = 0) "
              "instead of the snapshot, so `start_t` only sets the step budget; `midfinal` feeds that final chunk as-is one full-schedule grid step "
              "below pure noise (flow time 0.9 for pi0.5, 0.75 for GR00T; not 1) and walks to 0 in the same number of steps. n10 = variant success / reference failure; n01 the reverse. "
              "An arm evaluated twice on the same environment identity (two out roots) enters as the mean of its replicates, "
              "both in SR and in the paired difference. † = every paired difference identical, so the bootstrap collapses; the "
              "interval shown is the point ± the Wilson 95% upper bound on the disagreeing fraction, z²/(n+z²).", ""]
    if _has_self_arm(res):
        lines += [SELF_NFE_NOTE, ""]
    return "\n".join(lines)


def markdown_libero(res: dict) -> str:
    """Markdown report of one ``analyze_libero`` panel: the formal / partial status of the round and of every arm,
    per-task cell tables and paired differences, the formal macro (formal arms only) and, while the round is in
    flight, the progress macro labelled partial."""
    lines = [f"## LIBERO self-start round — {res['env_id']} (m={res['m']}; descriptive, 95% paired bootstrap"
             + (f"; init_idx >= {res['min_idx']}" if res.get("min_idx") else "") + ")", ""]
    formal = res.get("formal", {})
    status = res.get("status", "partial")
    lines += [f"status: **{status.upper()}**" + ("" if status == "formal" else
              f" (not a formal result: {formal.get('experiment_id')} needs every arm on 10 tasks x init_idx 0..49, "
              "env seed 7, one pool, every cell admitted)"), ""]
    not_formal = {aid: rec["problems"] for aid, rec in formal.get("arms", {}).items() if not rec["formal"]}
    if not_formal:
        lines += ["arms not formal: " + "; ".join(f"{aid} ({', '.join(p[:3])}{', ...' if len(p) > 3 else ''})"
                                                 for aid, p in not_formal.items()), ""]
    lines += _markdown_body(res)
    lines += ["Arms run m Euler steps per decision in the executed continuation (`full`: K). `warm_t` is the exact resume on the "
              "native grid; the cache variants feed the retrieved entry's snapshot (`warmreset` at flow time 1, `midreset` at 0.75, "
              "`midreset50` at 0.5) or its final action (`resetfinal` / `midfinal` / `midfinal50`, same entries); `self<variant>` "
              "feeds the same start produced by a direct full inference on the decision (private noise) instead of the cache, on "
              "the cache arm's yaml, so a self arm costs K + m action-head steps per decision (K = 10 pi0.5, 8 GR00T) against m "
              "for its cache arm: the panel is equal in continuation NFE, not in total NFE. "
              "GR00T arm ids name the native snapshot time and N (`_t0.75_n1`: T = 0.25, one step; `_t0.5_n2`: T = 0.5, two steps). "
              "Pairs are environment identities (task, init_idx, env seed, initial-state pool); n10 = variant success / reference "
              "failure, n01 the reverse; † = degenerate bootstrap, interval = point ± the Wilson 95% upper bound z²/(n+z²).", ""]
    return "\n".join(lines)


def _macro_lines(title: str, macro: dict) -> List[str]:
    lines = [title, "", "| arm | tasks | macro SR |", "|---|---|---|"]
    for aid, m in macro["sr"].items():
        msr = "–" if m["macro_sr"] is None else f"{m['macro_sr']:.3f}"
        lines.append(f"| {aid} | {m['n_tasks']} | {msr} |")
    lines += ["", "| macro paired difference | tasks | Δ | 95% CI |", "|---|---|---|---|"]
    for name, d in macro["paired_deltas"].items():
        if d["n_tasks"]:
            dg = " †" if d.get("degenerate") else ""
            lines.append(f"| {name} | {d['n_tasks']} | {d['point']:+.3f} | [{d['lower']:+.3f}, {d['upper']:+.3f}]{dg} |")
    lines.append("")
    return lines


def _markdown_body(res: dict) -> List[str]:
    lines = []
    if res["missing_arms"]:
        lines += [f"missing arms: {', '.join(res['missing_arms'])}", ""]
    for task, r in res["tasks"].items():
        lines += [f"### t{r['task_id']}: {task}" if "task_id" in r else f"### {task}", ""]
        if "status" in r:
            lines += [f"task status: {r['status']}", ""]
        cp, cn = r.get("comparison_problems", []), r.get("comparison_notes", {})
        lines += [f"cross-arm model/env identity: {'MISMATCH (task excluded from macro)' if cp else 'identical'}; "
                  f"worker islands {cn.get('worker_islands', '–')}, serving runtimes {cn.get('serving_runtimes', '–')} (recorded, not gated)", ""]
        if cp:
            lines += [f"cross-arm problems: {', '.join(cp)}", ""]
        lines += ["| arm | n | SR | admissible | steps | miss | problems |", "|---|---|---|---|---|---|---|"]
        for aid, c in r["cells"].items():
            adm = "ok" if c["complete"] and c["equal_nfe"] else "NO"
            if "formal" in c and adm == "ok":
                adm = "formal" if c["formal"] else "partial"
            sr = "–" if c["sr"] is None else f"{c['sr']:.2f}"
            st = "–" if c["mean_executed_steps"] is None else f"{c['mean_executed_steps']:.1f}"
            if c.get("self_start"):
                k = (c.get("nfe_per_decision") or {}).get("self_start")
                st += " + " + ("–" if k is None else f"{k:g}") + " self"
            nn = f"{c['n']} ({c['n_identities']} ids × replicates)" if c.get("replicated") else f"{c['n']}"
            lines.append(f"| {aid} | {nn} | {sr} | {adm} | {st} | {c['miss_fraction']} | {', '.join(c['problems']) or '–'} |")
        lines += ["", "| paired difference | n | Δ | 95% CI | n10 / n01 |", "|---|---|---|---|---|"]
        for name, d in r["paired_deltas"].items():
            if d["n"]:
                dg = " †" if d.get("degenerate") else ""
                lines.append(f"| {name} | {d['n']} | {d['point']:+.2f} | [{d['lower']:+.2f}, {d['upper']:+.2f}]{dg} | {d['n10']} / {d['n01']} |")
        lines.append("")
    if "formal" not in res:  # RoboCasa: one macro over the admissible tasks
        if res.get("macro"):
            lines += _macro_lines("### Macro (mean over admissible tasks)", res["macro"])
        return lines
    if res.get("macro"):
        lines += _macro_lines("### Macro — formal (all suite tasks; formally complete arms and pairs only)", res["macro"])
    if res.get("macro_partial"):
        lines += _macro_lines("### Progress macro — PARTIAL (admitted task subsets of a round in flight; not a formal result)",
                              res["macro_partial"])
    return lines


def main() -> None:
    """CLI: the RoboCasa variant report (``--tasks``) or, with ``--env-id``, the LIBERO round's panels."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", default="pi05", choices=("pi05", "groot"))
    ap.add_argument("--tasks", default="", help="comma-separated task names (RoboCasa; required there)")
    ap.add_argument("--env-id", default="", help="a LIBERO environment: the LIBERO self-start round (tasks from the manifests)")
    ap.add_argument("--m", type=int, default=None, help="LIBERO: one equal-continuation-NFE panel m (default: every panel of the environment)")
    ap.add_argument("--arms-root", default=None, help="one or more (comma-separated) driver out roots; merged per arm "
                    "(default: exp/step_diag/data/rc, LIBERO exp/step_diag/data/libero_self)")
    ap.add_argument("--server-rows", default=None, help="one or more (comma-separated) server out roots "
                    "(default: exp/step_diag/data/server, LIBERO exp/step_diag/data/server_libero_self)")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--min-idx", type=int, default=0, help="sensitivity: keep only init_idx >= this (admission still on the whole cell)")
    ap.add_argument("--t", type=float, default=None, help="warm-start ladder point (default: the Q-B main t); m = floor(t*K+0.5)")
    a = ap.parse_args()
    libero = bool(a.env_id)
    arms_root = a.arms_root or ("exp/step_diag/data/libero_self" if libero else "exp/step_diag/data/rc")
    server_rows = a.server_rows or ("exp/step_diag/data/server_libero_self" if libero else "exp/step_diag/data/server")
    arms_roots = [pathlib.Path(x) for x in arms_root.split(",") if x]
    server_roots = [pathlib.Path(x) for x in server_rows.split(",") if x]
    pathlib.Path(a.out_json).parent.mkdir(parents=True, exist_ok=True)
    if libero:
        panels = libero_panels(a.env_id) if a.m is None else (a.m,)
        results = {f"m{m}": analyze_libero(a.env_id, arms_roots, server_roots, m=m, min_idx=a.min_idx) for m in panels}
        pathlib.Path(a.out_json).write_text(json.dumps(results, indent=1))
        pathlib.Path(a.out_md).write_text("\n".join(markdown_libero(r) for r in results.values()))
        for key, r in results.items():
            print(f"WARM_VARIANTS {a.env_id} {key}: status={r['status']} tasks={len(r['tasks'])} "
                  f"missing={r['missing_arms']} -> {a.out_json}")
        return
    if not a.tasks:
        ap.error("--tasks is required for RoboCasa")
    res = analyze(a.policy, [s for s in a.tasks.split(",") if s], arms_roots, server_roots, min_idx=a.min_idx, t=a.t)
    pathlib.Path(a.out_json).write_text(json.dumps(res, indent=1))
    pathlib.Path(a.out_md).write_text(markdown(res))
    print(f"WARM_VARIANTS {a.policy}: tasks={list(res['tasks'])} missing={res['missing_arms']} -> {a.out_json}")


if __name__ == "__main__":
    main()
