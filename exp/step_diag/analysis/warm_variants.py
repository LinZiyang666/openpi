"""Warm-start continuation variants vs the formal Q-B arms (pi0.5 RoboCasa, 2026-09-21 follow-up).

Compares, on the tasks given, the equal-NFE arms ``full`` / ``plain_k<m>`` / ``warm_t<t>`` (formal Q-B
data) with ``warmreset_t<t>`` and ``warmshoot_t<t>`` (``exp.step_diag.pi05.warm_variant_stage3``).
Every cell goes through the same admission as the formal analysis (``aggregate_arms.cell_admission``:
complete accepted set, WARM_START on every decision, executed steps == m, one stage-3 call); the
outcomes are paired on the frozen environment identity (task, init_idx, env_seed, lane, pin, layout,
style) and the paired difference of each variant against ``plain_k<m>`` and against ``warm_t<t>`` gets
a percentile bootstrap interval (episodes resampled, 20000 draws, seed 20260919). Descriptive only:
these arms were not in the pre-registered family, so no verdict is issued.

usage::

    python -m exp.step_diag.analysis.warm_variants --policy pi05 --tasks CloseFridge,PickPlaceCounterToStove \
        --arms-root exp/step_diag/data/rc --server-rows exp/step_diag/data/server \
        --out-json exp/step_diag/data/analysis/warm_variants_pi05.json --out-md exp/step_diag/analysis/warm_variants_pi05.md
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


def _load_arm_roots(teacher: str, aid: str, arms_roots: List[pathlib.Path], server_roots: List[pathlib.Path]):
    parts, servers = [], {}
    for ar in arms_roots:
        d = ar / teacher / aid
        if d.is_dir() and any(d.glob("launch_*.json")):
            parts.append(A.load_arm(d))
    for sr in server_roots:
        d = sr / teacher / aid
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
                 (f"midreset_t{t:g}", "warm", m), (f"midreset50_t{t:g}", "warm", m)]
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
        cells = {}
        for aid, kind, mm in arm_specs:
            if aid in arms:
                c = A.cell_admission(arms[aid], servers.get(aid, {}), task, kind=kind, m=mm)
                # sensitivity analysis: admission is judged on the whole cell, the estimate on init_idx >= min_idx
                c["outcomes"] = {u: v for u, v in c["outcomes"].items() if int(v["init_idx"]) >= int(min_idx)}
                n = len(c["outcomes"])
                n_ids = len({A.pair_identity(v) for v in c["outcomes"].values()})
                cells[aid] = {"n": n, "n_identities": n_ids, "replicated": n != n_ids,
                              "sr": _identity_sr({"_outcomes": c["outcomes"]}),
                              "comparison_identities": c["comparison_identities"],
                              "worker_identities": c["worker_identities"], "runtime_notes": c["runtime_notes"],
                              "complete": c["complete"], "equal_nfe": c["equal_nfe"], "problems": c["problems"],
                              "miss_fraction": c["miss_fraction"], "mean_executed_steps": c["mean_executed_steps"],
                              "stray_sessions": c["stray_sessions"], "_outcomes": c["outcomes"]}
        # cross-arm gate (as in the formal analysis): one model + environment contract across the task's arms
        comp = [tuple(c["comparison_identities"]) for c in cells.values() if c["n"]]  # arms not run on this task carry none
        cross_problems = [] if comp and all(len(v) == 1 and v == comp[0] for v in comp) else ["comparison_identities_mismatch"]
        notes = A.comparison_notes(cells)
        deltas = {}
        for var in (f"warmreset_t{t:g}", f"warmshoot_t{t:g}", f"resetfinal_t{t:g}", f"midfinal_t{t:g}", f"midfinal50_t{t:g}", f"midreset_t{t:g}", f"midreset50_t{t:g}", f"warm_t{t:g}"):
            if var not in cells:
                continue
            for ref in (f"plain_k{m}", f"warm_t{t:g}", f"warmreset_t{t:g}", f"resetfinal_t{t:g}", f"midfinal_t{t:g}", f"midreset_t{t:g}", "full"):
                if ref in cells and ref != var:
                    pairs = _paired(cells[var], cells[ref])
                    deltas[f"{var} - {ref}"] = _boot_delta(pairs, rng)
                    if (not cross_problems and cells[var]["complete"] and cells[var]["equal_nfe"]
                            and cells[ref]["complete"] and cells[ref]["equal_nfe"]):
                        macro_pairs.setdefault(f"{var} - {ref}", {})[task] = pairs
        for c in cells.values():
            c.pop("_outcomes")
        out["tasks"][task] = {"cells": cells, "paired_deltas": deltas, "comparison_problems": cross_problems,
                              "comparison_notes": notes}
    if len(tasks) > 1:
        macro_sr = {}
        for aid, _, _ in arm_specs:
            srs = [r["cells"][aid]["sr"] for r in out["tasks"].values()
                   if aid in r["cells"] and not r["comparison_problems"] and r["cells"][aid]["complete"] and r["cells"][aid]["equal_nfe"] and r["cells"][aid]["sr"] is not None]
            macro_sr[aid] = {"n_tasks": len(srs), "macro_sr": float(np.mean(srs)) if srs else None}
        out["macro"] = {"sr": macro_sr, "paired_deltas": {k: _boot_macro(v, rng) for k, v in macro_pairs.items()}}
    return out


def markdown(res: dict) -> str:
    m, t = res["m"], res["t"]
    lines = [f"## Warm-start continuation variants — {res['policy']} (m={m}, t={t:g}; descriptive, 95% paired bootstrap"
             + (f"; init_idx >= {res['min_idx']}" if res.get("min_idx") else "") + ")", ""]
    if res["missing_arms"]:
        lines += [f"missing arms: {', '.join(res['missing_arms'])}", ""]
    for task, r in res["tasks"].items():
        lines += [f"### {task}", ""]
        cp, cn = r.get("comparison_problems", []), r.get("comparison_notes", {})
        lines += [f"cross-arm model/env identity: {'MISMATCH (task excluded from macro)' if cp else 'identical'}; "
                  f"worker islands {cn.get('worker_islands', '–')}, serving runtimes {cn.get('serving_runtimes', '–')} (recorded, not gated)", ""]
        lines += ["| arm | n | SR | admissible | steps | miss | problems |", "|---|---|---|---|---|---|---|"]
        for aid, c in r["cells"].items():
            adm = "ok" if c["complete"] and c["equal_nfe"] else "NO"
            sr = "–" if c["sr"] is None else f"{c['sr']:.2f}"
            st = "–" if c["mean_executed_steps"] is None else f"{c['mean_executed_steps']:.1f}"
            nn = f"{c['n']} ({c['n_identities']} ids × replicates)" if c.get("replicated") else f"{c['n']}"
            lines.append(f"| {aid} | {nn} | {sr} | {adm} | {st} | {c['miss_fraction']} | {', '.join(c['problems']) or '–'} |")
        lines += ["", "| paired difference | n | Δ | 95% CI | n10 / n01 |", "|---|---|---|---|---|"]
        for name, d in r["paired_deltas"].items():
            if d["n"]:
                dg = " †" if d.get("degenerate") else ""
                lines.append(f"| {name} | {d['n']} | {d['point']:+.2f} | [{d['lower']:+.2f}, {d['upper']:+.2f}]{dg} | {d['n10']} / {d['n01']} |")
        lines.append("")
    if res.get("macro"):
        lines += ["### Macro (mean over admissible tasks)", "", "| arm | tasks | macro SR |", "|---|---|---|"]
        for aid, m in res["macro"]["sr"].items():
            msr = "–" if m["macro_sr"] is None else f"{m['macro_sr']:.3f}"
            lines.append(f"| {aid} | {m['n_tasks']} | {msr} |")
        lines += ["", "| macro paired difference | tasks | Δ | 95% CI |", "|---|---|---|---|"]
        for name, d in res["macro"]["paired_deltas"].items():
            if d["n_tasks"]:
                dg = " †" if d.get("degenerate") else ""
                lines.append(f"| {name} | {d['n_tasks']} | {d['point']:+.3f} | [{d['lower']:+.3f}, {d['upper']:+.3f}]{dg} |")
        lines.append("")
    lines += ["Variants: `warmreset` restarts the flow time at 1 with dt = −1/remaining (the cache is fed as if it were noise); "
              "`warmshoot` keeps the cache's start_t with dt = −1/remaining (t crosses 0). Both run the same number of "
              "Euler steps as `warm_t` and `plain_k`; `resetfinal` is the `warmreset` loop started from the cache's final action chunk (t = 0) "
              "instead of the snapshot, so `start_t` only sets the step budget; `midfinal` feeds that final chunk as-is one full-schedule grid step "
              "below pure noise (flow time 0.9 for pi0.5, 0.75 for GR00T; not 1) and walks to 0 in the same number of steps. n10 = variant success / reference failure; n01 the reverse. "
              "An arm evaluated twice on the same environment identity (two out roots) enters as the mean of its replicates, "
              "both in SR and in the paired difference. † = every paired difference identical, so the bootstrap collapses; the "
              "interval shown is the point ± the Wilson 95% upper bound on the disagreeing fraction, z²/(n+z²).", ""]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", default="pi05", choices=("pi05", "groot"))
    ap.add_argument("--tasks", required=True, help="comma-separated task names")
    ap.add_argument("--arms-root", default="exp/step_diag/data/rc", help="one or more (comma-separated) driver out roots; merged per arm")
    ap.add_argument("--server-rows", default="exp/step_diag/data/server", help="one or more (comma-separated) server out roots")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--min-idx", type=int, default=0, help="sensitivity: keep only init_idx >= this (admission still on the whole cell)")
    ap.add_argument("--t", type=float, default=None, help="warm-start ladder point (default: the Q-B main t); m = floor(t*K+0.5)")
    a = ap.parse_args()
    res = analyze(a.policy, [s for s in a.tasks.split(",") if s], [pathlib.Path(x) for x in a.arms_root.split(",") if x],
                  [pathlib.Path(x) for x in a.server_rows.split(",") if x], min_idx=a.min_idx, t=a.t)
    pathlib.Path(a.out_json).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(a.out_json).write_text(json.dumps(res, indent=1))
    pathlib.Path(a.out_md).write_text(markdown(res))
    print(f"WARM_VARIANTS {a.policy}: tasks={list(res['tasks'])} missing={res['missing_arms']} -> {a.out_json}")


if __name__ == "__main__":
    main()
