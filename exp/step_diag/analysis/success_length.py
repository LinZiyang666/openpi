"""Length of successful episodes: warm start / warm reset / plain step reduction vs full inference (RoboCasa365, LIBERO).

Episode length, in two units from the accepted attempt's ``episode_summary`` row in ``per_step``: ``n_decisions``
(policy inference calls) and ``n_env_steps`` (environment steps; each call executes 5). A length is admitted only
from an accepted journal terminal joined to exactly one accepted summary of the same task_uid, attempt, driver run,
arm and configuration, with the same success and no error (``arm_episodes``); everything else is reported under
``problems``. Where the worker also logs one row per decision (pi0.5), those
rows are checked against it (contiguous ``step_idx`` 0, 5, ..., count == ``n_decisions``); every summary must satisfy
5 (n_decisions - 1) < n_env_steps <= 5 n_decisions. Only successful episodes have a meaningful length (failures run to
the step limit).

Two readings per arm, both against ``full`` of the same teacher:
* unpaired: mean decisions over each arm's own successes, per task, then the equal-weight mean over tasks;
* paired (the fair one): environment identities (task, init_idx, env seed, initial-state pool) on which BOTH the arm
  and full succeeded, mean of (arm - full) per task, equal-weight mean over tasks, bootstrap over identities within
  task. The same (task, init_idx) with another seed or pool never pairs and is reported (``identity_mismatches``).
An identity evaluated twice (two out roots) enters with the mean length of its successful replicates.
The task roster is the union of the loaded arms, so an unfinished reference cannot hide another arm's successes;
an unpaired macro is available only when that arm has a successful length on every task in the loaded roster.

``--benchmark libero``: the LIBERO self-start round, one panel per environment and continuation budget m
(``warm_variants.libero_arm_specs``; a self arm's decisions cost K + m action-head steps, its cache arm's m), cells
under ``<root>/<teacher>/<env_id>/<arm>``; environment steps exclude the settle steps LIBERO runs before the first
decision (``num_steps_wait`` of the summary row), and every self arm is also paired against its cache arm
(``paired_vs_cache``). Each panel and arm carries ``status``: ``formal`` only when every arm covers the frozen
10 tasks x init_idx 0..49 of ``envs.LIBERO_SELF_EXPERIMENT_ID`` with one accepted length each (``libero_arm_status``),
else ``partial`` (a run in flight; its numbers are progress, not a result). A paired statistic has its own status,
which is formal only when both its arm and reference are formal.

    uv run python -m exp.step_diag.analysis.success_length
    uv run python -m exp.step_diag.analysis.success_length --benchmark libero --out exp/step_diag/data/analysis/success_length_libero.json
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
from typing import Dict, List, Tuple

import numpy as np

from exp.step_diag import envs as _envs
from exp.step_diag.analysis import aggregate_arms as A
from exp.step_diag.analysis import warm_variants as W

ROOTS = ("exp/step_diag/data/rc", "exp/step_diag/data/rc_macro13", "exp/step_diag/data/rc_self13")
LIBERO_ROOTS = ("exp/step_diag/data/libero_self",)
LIBERO_ENV_IDS = ("pi05_libero_spatial", "pi05_libero_10", "groot_libero_spatial", "groot_libero_10")
PANELS = {
    "pi05": ("pi05", ["full", "plain_k2", "warm_t0.2", "warmreset_t0.2", "resetfinal_t0.2", "midfinal_t0.2",
                      "selfwarmreset_t0.2", "selfresetfinal_t0.2", "selfmidfinal_t0.2"]),
    "groot_n1": ("groot_tp", ["full", "plain_k1", "warm_t0.75", "warmreset_t0.75", "midreset_t0.75", "midreset50_t0.75",
                              "resetfinal_t0.75", "midfinal_t0.75", "midfinal50_t0.75",
                              "selfwarmreset_t0.75", "selfmidreset_t0.75", "selfmidreset50_t0.75",
                              "selfresetfinal_t0.75", "selfmidfinal_t0.75", "selfmidfinal50_t0.75"]),
    "groot_n2": ("groot_tp", ["full", "plain_k2", "midreset50_t0.5", "warmreset_t0.5", "midreset_t0.5", "resetfinal_t0.5",
                              "midfinal_t0.5", "midfinal50_t0.5",
                              "selfwarmreset_t0.5", "selfmidreset_t0.5", "selfresetfinal_t0.5", "selfmidfinal_t0.5",
                              "selfmidfinal50_t0.5"]),
}
BOOT = 20000
SEED = 20260923


def identity_key(ident: dict, env_dir: str = "") -> tuple:
    """The environment identity lengths pair on: ``(task, init_idx, env_seed, init_pool_sha256, env)``. The pool is
    None on RoboCasa and ``env`` the LIBERO environment id (None on RoboCasa, whose panels are one teacher each); the
    same init_idx of another pool or seed is another initial state and never pairs."""
    return (ident["task"], int(ident["init_idx"]), ident.get("env_seed"), ident.get("init_pool_sha256"), env_dir or None)


def summary_problems(summ: dict, rec: dict, launch: dict, ident: dict, arm_id, *, libero: bool = False) -> List[str]:
    """Why one accepted ``episode_summary`` cannot give the length of the accepted terminal ``rec`` (empty = it can).

    The summary must name the terminal's arm, experiment and served configuration (as its launch declares them),
    carry the same boolean success and no error, and hold positive integer counts. LIBERO requires all identity
    fields its worker emits (init_idx, task id, seed, pool) and a nonnegative settle-step count; legacy RoboCasa
    summaries are checked on the identity fields they carry."""
    out = []
    if summ.get("error"):
        out.append("summary error")
    if type(summ.get("success")) is not bool or summ.get("success") != rec.get("success"):
        out.append("summary success disagrees with the terminal")
    if summ.get("arm_id") != launch.get("arm_id") or summ.get("arm_id") != arm_id:
        out.append("summary arm mismatch")
    if any(not launch.get(k) or summ.get(k) != launch[k] for k in ("experiment_id", "config_sha")):
        out.append("summary experiment / config mismatch")
    if ident not in launch.get("expected", []):
        out.append("identity is not declared by the summary's launch")
    for skey, ikey in (("init_idx", "init_idx"), ("task_id", "task_id"), ("seed", "env_seed"),
                       ("init_pool_sha256", "init_pool_sha256")):
        if libero and (summ.get(skey) is None or ident.get(ikey) is None):
            out.append(f"summary / expected {skey} missing")
        elif summ.get(skey) is not None and ikey in ident and summ[skey] != ident[ikey]:
            out.append(f"summary {skey} differs from the expected identity")
    if any(type(summ.get(k)) is not int or summ[k] < 1 for k in ("n_decisions", "n_env_steps")):
        out.append("summary counts missing or nonpositive")
    if libero and (type(summ.get("num_steps_wait")) is not int or summ["num_steps_wait"] < 0):
        out.append("summary settle count missing or invalid")
    return out


def arm_episodes(teacher: str, arm: str, roots=ROOTS,
                 env_dir: str = "") -> Tuple[Dict[tuple, List[Tuple[bool, int, int]]], List[str]]:
    """``identity_key`` -> [(success, n_env_steps, n_decisions)] over every accepted replicate of one arm, and the
    problems found (LIBERO: ``env_dir`` = the environment id, the cells' extra directory level).

    A length comes only from an accepted journal terminal (done / failed, boolean success, no error, one per
    task_uid) joined to exactly ONE accepted ``episode_summary`` of the same task_uid and attempt, stamped with the
    terminal's driver run (its ``run_id``, and a launch of this arm whose ``driver_run_id`` it is) and passing
    ``summary_problems``. Anything else is reported and contributes no length: a summary the driver fenced
    (``accepted`` not True), a success that disagrees with the terminal, an error terminal or summary, another
    run's or attempt's summary, a conflicting terminal, and every manifest problem of the arm.
    """
    out: Dict[tuple, List[Tuple[bool, int, int]]] = collections.defaultdict(list)
    problems: List[str] = []
    for root in roots:
        d = pathlib.Path(root) / teacher / env_dir / arm
        if not (d.is_dir() and any(d.glob("launch_*.json"))):
            continue
        a = A.load_arm(d)
        tag = f"{root}:{arm}"
        manifest_problems = [f"{tag}: manifest problem {p}" for p in a["manifest_problems"]]
        launches = a["launches"]  # launch_id -> launch; the journal's run_id is a launch's driver_run_id
        if a["arm_id"] != arm:
            manifest_problems.append(f"{tag}: launch arm differs from the requested arm")
        if env_dir and any(launch.get("env_id") != env_dir for launch in launches.values()):
            manifest_problems.append(f"{tag}: a launch manifest names another environment than {env_dir}")
        if manifest_problems:
            problems += manifest_problems
            continue
        steps: Dict[Tuple[str, str, int], List[int]] = collections.defaultdict(list)
        summaries: Dict[Tuple[str, int], List[dict]] = collections.defaultdict(list)
        for f in d.glob("per_step_*.jsonl"):
            for line in f.read_text().splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("row") == "episode_summary":
                    summaries[(r["task_uid"], int(r.get("attempt", 1)))].append(r)
                elif r.get("step_idx") is not None and int(r["step_idx"]) >= 0:  # LIBERO client_timing rows carry none
                    steps[(r["task_uid"], r.get("run_id"), int(r.get("attempt", 1)))].append(int(r["step_idx"]))
        for uid, rec in a["outcomes"].items():
            ident = a["expected"].get(uid)
            if ident is None or uid in a["conflicts"]:
                problems.append(f"{tag}:{uid}: no expected identity or conflicting terminal")
                continue
            if rec.get("error") or type(rec.get("success")) is not bool:
                problems.append(f"{tag}:{uid}: terminal error or non-boolean success")
                continue
            att, run = int(rec.get("attempt", 1)), rec.get("run_id")
            cands = [r for r in summaries.get((uid, att), [])
                     if r.get("accepted") is True and run and r.get("run_id") == run
                     and (launches.get(r.get("launch_id")) or {}).get("driver_run_id") == run]
            if len(cands) != 1:
                problems.append(f"{tag}:{uid}: {len(cands)} accepted episode_summary rows of the terminal's run / attempt")
                continue
            summ = cands[0]
            bad = summary_problems(summ, rec, launches[summ["launch_id"]], ident, arm, libero=bool(env_dir))
            if bad:
                problems.append(f"{tag}:{uid}: {'; '.join(bad)}")
                continue
            idx = sorted(steps.get((uid, run, att), []))
            # LIBERO counts its settle steps before the first decision; RoboCasa summaries carry none
            n_dec, n_env = int(summ["n_decisions"]), int(summ["n_env_steps"]) - int(summ.get("num_steps_wait") or 0)
            # pi0.5 workers also log one row per decision; GR00T workers log only the summary
            if idx and (idx != list(range(0, 5 * len(idx), 5)) or len(idx) != n_dec):
                problems.append(f"{tag}:{uid}: decision rows ({len(idx)}) disagree with summary ({n_dec})")
                continue
            if not 5 * (n_dec - 1) < n_env <= 5 * n_dec:
                problems.append(f"{tag}:{uid}: n_env_steps {n_env} inconsistent with {n_dec} decisions of 5 steps")
                continue
            out[identity_key(ident, env_dir)].append((rec["success"], n_env, n_dec))
    return out, problems


def identity_mismatches(data: Dict[str, dict], arms: List[str]) -> List[str]:
    """Cross-arm identity gate of one panel: every ``(task, init_idx)`` must carry ONE env seed / pool / environment
    across the arms. A mismatched identity never pairs (``identity_key``); it is reported here, not dropped silently."""
    seen: Dict[tuple, set] = collections.defaultdict(set)
    owners: Dict[tuple, set] = collections.defaultdict(set)
    for arm in arms:
        for key in data.get(arm, {}):
            seen[key[:2]].add(key[2:])
            owners[key[:2]].add(arm)
    bad = sorted(k for k, v in seen.items() if len(v) > 1)
    if not bad:
        return []
    involved = sorted({a for k in bad for a in owners[k]})
    return [f"{len(bad)} (task, init_idx) identities carry different env seeds / initial-state pools / environments "
            f"across arms {involved} (unpaired)"]


def libero_arm_status(teacher: str, arm: str, roots, env_id: str, data_arm: dict, arm_problems: List[str]) -> dict:
    """Formal status of one LIBERO arm for the length analysis: its manifests are the frozen design
    (``warm_variants.libero_formal_problems``), no length problem was reported for it, and every one of the
    10 x 50 expected identities gave exactly one accepted length. Otherwise the arm is partial (a run in flight)."""
    parts = [A.load_arm(pathlib.Path(r) / teacher / env_id / arm) for r in roots
             if (pathlib.Path(r) / teacher / env_id / arm).is_dir()
             and any((pathlib.Path(r) / teacher / env_id / arm).glob("launch_*.json"))]
    if not parts:
        return {"formal": False, "problems": ["arm_missing"]}
    merged = parts[0] if len(parts) == 1 else W._merge_arms(parts)
    problems = W.libero_formal_problems(env_id, merged)
    if arm_problems:
        problems.append(f"length_problems:{len(arm_problems)}")
    n_expected = _envs.LIBERO_N_TASKS * _envs.LIBERO_SELF_EPISODES
    if len(data_arm) != n_expected or any(len(v) != 1 for v in data_arm.values()):
        problems.append(f"accepted_lengths:{sum(len(v) for v in data_arm.values())}/{n_expected}")
    return {"formal": not problems, "problems": problems}


UNITS = {"env_steps": 1, "decisions": 2}  # tuple position of each length unit


def success_length(reps: List[Tuple[bool, int, int]], pos: int):
    """Mean length (tuple position ``pos``: 1 env steps, 2 decisions) over the successful replicates of one
    identity; None when none of them succeeded (a failure's length is the step limit, not a length)."""
    lens = [r[pos] for r in reps if r[0]]
    return float(np.mean(lens)) if lens else None


def analyse(data: dict, problems: List[str], arms: List[str], pos: int, rng: np.random.Generator, *,
            ref: str = "full", vs_cache: bool = False) -> dict:
    """Per-arm success lengths, paired against ``ref``; ``vs_cache`` also pairs every self arm with its cache arm."""
    tasks = sorted({k[0] for aid in {ref, *arms} for k in data[aid]})
    full_len = {k: success_length(v, pos) for k, v in data[ref].items()}
    res = {"tasks": tasks, "arms": {}, "problems": problems}
    for arm in arms:
        own = {k: success_length(v, pos) for k, v in data[arm].items()}
        per_task = {}
        diffs: Dict[str, np.ndarray] = {}
        for t in tasks:
            succ = [v for k, v in own.items() if k[0] == t and v is not None]
            pairs = [(own[k], full_len[k]) for k in own if k[0] == t and own[k] is not None and full_len.get(k) is not None]
            d = np.array([a - f for a, f in pairs], dtype=float)
            per_task[t] = {"n_success": len(succ), "mean_length": float(np.mean(succ)) if succ else None,
                           "n_pairs": len(pairs), "paired_diff": float(d.mean()) if len(d) else None}
            if len(d):
                diffs[t] = d
        if not data[arm]:
            res["arms"][arm] = {"unpaired_macro_length": None, "n_success": 0, "per_task": per_task, "missing": True}
            continue
        means = [v["mean_length"] for v in per_task.values() if v["mean_length"] is not None]
        row = {"unpaired_macro_length": float(np.mean(means)) if len(means) == len(tasks) else None,
               "n_success": int(sum(v["n_success"] for v in per_task.values())), "per_task": per_task}
        if arm != ref and diffs:
            point = float(np.mean([d.mean() for d in diffs.values()]))
            acc = np.zeros(BOOT)
            for d in diffs.values():
                acc += d[rng.integers(0, len(d), size=(BOOT, len(d)))].mean(axis=1)
            acc /= len(diffs)
            pooled = np.concatenate(list(diffs.values()))
            ratios = [float(np.mean([own[k] for k in own if k[0] == t and own[k] is not None and full_len.get(k) is not None]) /
                            np.mean([full_len[k] for k in own if k[0] == t and own[k] is not None and full_len.get(k) is not None]))
                      for t in diffs]
            row["paired"] = {"n_tasks": len(diffs), "n_pairs": int(len(pooled)), "macro_diff": point,
                             "lower": float(np.percentile(acc, 2.5)), "upper": float(np.percentile(acc, 97.5)),
                             "macro_ratio": float(np.mean(ratios)),
                             "share_shorter": float(np.mean(pooled < 0)), "share_longer": float(np.mean(pooled > 0))}
        cache = arm[len("self"):]
        if vs_cache and arm.startswith("self") and data.get(cache):
            row["paired_vs_cache"] = analyse(data, [], [arm], pos, rng, ref=cache)["arms"][arm].get("paired")
        res["arms"][arm] = row
    return res


def libero_panels(env_ids=LIBERO_ENV_IDS) -> Dict[str, Tuple[str, str, List[str]]]:
    """``{<env_id>_m<m>: (teacher, env_id, arms)}``: the LIBERO self-start round's equal-continuation-NFE panels
    (a self arm also runs its K-step direct inference per decision: K + m action-head steps)."""
    panels = {}
    for env_id in env_ids:
        teacher = "pi05" if _envs.resolve_env(env_id).policy == "pi05" else "groot_tp"
        for m in W.libero_panels(env_id):
            panels[f"{env_id}_m{m}"] = (teacher, env_id, [a for a, _, _ in W.libero_arm_specs(env_id, m)])
    return panels


def main() -> None:
    """CLI: the RoboCasa panels (default) or, with ``--benchmark libero``, the LIBERO round's panels, each in two
    length units; writes ``--out`` and prints one line per arm (LIBERO panels carry their formal / partial status)."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="exp/step_diag/data/analysis/success_length.json")
    ap.add_argument("--benchmark", default="rc", choices=("rc", "libero"))
    ap.add_argument("--roots", default=",".join(LIBERO_ROOTS), help="LIBERO: comma-separated driver out roots")
    ap.add_argument("--env-ids", default=",".join(LIBERO_ENV_IDS), help="LIBERO: environments to report")
    a = ap.parse_args()
    loaded, formal = {}, {}
    if a.benchmark == "libero":
        roots = tuple(x for x in a.roots.split(",") if x)
        for k, (teacher, env_id, arms) in libero_panels(tuple(x for x in a.env_ids.split(",") if x)).items():
            data, problems, status = {}, [], {}
            for arm in arms:
                data[arm], p = arm_episodes(teacher, arm, roots, env_dir=env_id)
                problems += p
                status[arm] = libero_arm_status(teacher, arm, roots, env_id, data[arm], p)
            mismatch = identity_mismatches(data, arms)
            pools = {key[3] for arm in arms for key in data[arm]}
            if mismatch or len(pools) > 1:
                for rec in status.values():
                    rec["formal"] = False
                    rec["problems"].append("identity_mismatch_across_arms")
            loaded[k] = (data, problems + mismatch, arms)
            formal[k] = status
    else:
        for k, (teacher, arms) in PANELS.items():
            data, problems = {}, []
            for arm in arms:
                data[arm], p = arm_episodes(teacher, arm)
                problems += p
            loaded[k] = (data, problems + identity_mismatches(data, arms), arms)
    out = {}
    for unit, pos in UNITS.items():
        rng = np.random.default_rng(SEED)
        out[unit] = {k: analyse(data, problems, arms, pos, rng, vs_cache=a.benchmark == "libero")
                     for k, (data, problems, arms) in loaded.items()}
        for k, status in formal.items():  # LIBERO: formal only when every arm covers the frozen 10 x 50 design
            r = out[unit][k]
            r["status"] = "formal" if all(rec["formal"] for rec in status.values()) else "partial"
            r["formal"] = status
            for arm, row in r["arms"].items():
                row["status"] = "formal" if status[arm]["formal"] else "partial"
                for key, ref in (("paired", "full"), ("paired_vs_cache", arm.removeprefix("self"))):
                    if row.get(key):
                        row[key]["status"] = ("formal" if status[arm]["formal"] and status[ref]["formal"] else "partial")
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(a.out).write_text(json.dumps(out, indent=1))
    for unit, panels in out.items():
        for k, r in panels.items():
            print(f"== {unit} / {k}  tasks={len(r['tasks'])}  problems={len(r['problems'])}"
                  + (f"  status={r['status'].upper()}" if "status" in r else ""))
            for arm, row in r["arms"].items():
                p = row.get("paired")
                if row.get("missing"):
                    print(f"  {arm:18s} (no data)")
                    continue
                own = row["unpaired_macro_length"]
                txt = f"  {arm:18s} succ={row['n_success']:4d}  own-success mean=" + ("   –   " if own is None else f"{own:7.2f}")
                if row.get("status") == "partial":
                    txt += "  [PARTIAL]"
                if p:
                    txt += (f"  | paired vs full: n={p['n_pairs']:4d} diff={p['macro_diff']:+7.2f} [{p['lower']:+.2f}, {p['upper']:+.2f}]"
                            f" ratio={p['macro_ratio']:.3f} shorter={p['share_shorter']:.2f} longer={p['share_longer']:.2f}")
                    if p.get("status") == "partial":
                        txt += " [PARTIAL]"
                pc = row.get("paired_vs_cache")
                if pc:
                    txt += f"  | vs cache: n={pc['n_pairs']:4d} diff={pc['macro_diff']:+7.2f} [{pc['lower']:+.2f}, {pc['upper']:+.2f}]"
                    if pc.get("status") == "partial":
                        txt += " [PARTIAL]"
                print(txt)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
