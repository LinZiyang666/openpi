"""Strict accepted-attempt join, adapted from K5 estimate.load_arm.

Produces anchors.csv, neighbours.csv, action_steps.csv and accounting.json.
Malformed/incomplete/missing/contradictory logs fail closed. No silent dropping.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from exp.offline_search.rounds.r04.k5_rand.estimate import jsonl
from .assignment import assign


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"no rows for {path}")
    keys = sorted(set().union(*(r.keys() for r in rows)))
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def flatten(value, prefix="", out=None):
    out = {} if out is None else out
    for key, val in value.items():
        key = f"{prefix}{key}"
        if isinstance(val, dict):
            flatten(val, key + ".", out)
        elif isinstance(val, list):
            out[key] = json.dumps(val, separators=(",", ":"))
        else:
            out[key] = val
    return out


def identical(a, b):
    # Only transport/timing provenance may differ on retransmission.
    drop = {"ts", "conn", "tag", "timing", "infer_ms", "pre_ms", "post_ms", "q_us", "search_us",
            "native_us", "s1_ms", "s23_ms", "decision_index", "stage1_calls"}
    return {k: v for k, v in a.items() if k not in drop} == {k: v for k, v in b.items() if k not in drop}


def load(root, arm, require_inputs=False):
    directory = Path(root) / "runs" / arm
    accepted = {}
    for r in jsonl(directory / "client/journal.jsonl"):
        if r.get("accepted") and r.get("status") in ("done", "failed") and not r.get("error"):
            uid = r["task_uid"]
            if uid in accepted and accepted[uid] != r:
                raise ValueError(f"multiple accepted outcomes: {uid}")
            accepted[uid] = r
    if not accepted:
        raise ValueError("no accepted completed episodes")
    buckets = {ev: defaultdict(dict) for ev in ("dec", "p3_anchor")}
    ends = {}
    configs = {}
    for path in sorted(directory.glob("server_*/decisions_*.jsonl")):
        for r in jsonl(path):
            if r["ev"] == "p3_startup":
                key = (r["tag"], r["conn"])
                config = {k: r[k] for k in ("p", "seed", "replicate", "strata")}
                if key in configs and configs[key] != config:
                    raise ValueError("mixed profile configuration")
                configs[key] = config
            uid = r.get("uid")
            if uid not in accepted or int(r.get("attempt", 1) or 1) != int(accepted[uid].get("attempt", 1) or 1):
                continue
            ev = r["ev"]
            if ev == "episode" and r.get("reason") == "episode_end":
                if uid in ends and (ends[uid]["n_decisions"], ends[uid]["success"]) != (r["n_decisions"], r["success"]):
                    raise ValueError(f"conflicting end: {uid}")
                ends[uid] = r
            if ev in buckets:
                step = int(r["step"])
                old = buckets[ev][uid].get(step)
                if old is not None and not identical(old, r):
                    raise ValueError(f"conflicting {ev}: {uid}:{step}")
                buckets[ev][uid][step] = r
    anchors, neighbours, steps, episodes = [], [], [], []
    inputs = {}
    if require_inputs:
        for path in directory.glob("server_*/inputs/*.npz"):
            with np.load(path, allow_pickle=False) as data:
                meta = json.loads(str(data["meta"]))
                uid = meta["uid"]
                if uid not in accepted or int(meta["attempt"]) != int(accepted[uid].get("attempt", 1)):
                    continue
                if uid in inputs:
                    raise ValueError("duplicate accepted input archives")
                inputs[uid] = {k: data[k].copy() for k in ("step", "hit", "vision", "a_exec")}
    for uid, journal in sorted(accepted.items()):
        ds, ps = buckets["dec"][uid], buckets["p3_anchor"][uid]
        if not ds or sorted(ds) != list(range(len(ds))):
            raise ValueError(f"missing decision(s): {uid}")
        end = ends.get(uid)
        if (end is None or end["n_decisions"] != len(ds) or end["n_exec"] != len(ds)
                or bool(end["success"]) != bool(journal["success"])):
            raise ValueError(f"missing/mismatching episode_end: {uid}")
        if any(not d.get("ok", True) or d.get("error") for d in ds.values()):
            raise ValueError(f"failed accepted decision: {uid}")
        if set(ps) != {i for i, d in ds.items() if d["vision"]}:
            raise ValueError(f"profile coverage != vision anchors: {uid}")
        if require_inputs:
            tape = inputs.get(uid)
            if (tape is None or tape["step"].tolist() != list(range(len(ds)))
                    or tape["hit"].tolist() != [int(ds[j]["hit"]) for j in range(len(ds))]
                    or tape["vision"].tolist() != [bool(ds[j]["vision"]) for j in range(len(ds))]):
                raise ValueError(f"missing/inconsistent input archive: {uid}")
            for j in range(len(ds)):
                if not np.array_equal(tape["a_exec"][j, :5, :7], np.asarray(ds[j]["served_head"], np.float32)):
                    raise ValueError("archive action mismatch")
        M = sum(not d["hit"] for d in ds.values())
        if end["n_miss"] != M:
            raise ValueError("MISS totals disagree")
        first_profile = ps[min(ps)]
        episodes.append(dict(arm=arm, uid=uid, model=first_profile["model"], suite=first_profile["suite"],
                             lib=first_profile["lib"], decisions=len(ds), anchors=len(ps), misses=M))
        for i, r in sorted(ps.items()):
            if not r.get("ok") or r.get("error") or r.get("schema") != "r6p3.anchor.v1":
                raise ValueError("failed/unknown profile row")
            a, d = r["assignment"], ds[i]
            cfg = configs[r["tag"], r["conn"]]
            p, stratum = cfg["p"], 0
            if cfg["strata"] is not None:
                stratum = int(np.searchsorted(cfg["strata"]["edges"], r["retrieval"]["d1_loeo_quantile"], side="right"))
                p = cfg["strata"]["p"][stratum]
            if (a["seed"], a["replicate"], a["propensity"], a["stratum"]) != (cfg["seed"], cfg["replicate"], p, stratum):
                raise ValueError("assignment does not match configured propensity")
            expected = assign(a["seed"], r["task_id"], r["init"], a["replicate"], i, a["propensity"])
            if any(a.get(k) != v for k, v in expected.items()):
                raise ValueError("forged or inconsistent assignment")
            policy = (not a["baseline_hit"]) or a["assigned_call"]
            if (bool(d["hit"]) == policy or a["executed_policy"] != policy
                    or a["eligible"] != a["baseline_hit"]
                    or a["injected"] != (a["baseline_hit"] and a["assigned_call"])
                    or a["actual_propensity"] != (p if a["baseline_hit"] else 1.)):
                raise ValueError("treatment noncompliance")
            if (r["task_id"], r["init"]) != (d["task_id"], d["init"]):
                raise ValueError("profile identity differs from decision")
            cache, teacher, executed = [np.asarray(r[k], np.float32) for k in ("cache_chunk", "policy_chunk", "executed_chunk")]
            if (cache.shape != teacher.shape or cache.shape != executed.shape or cache.ndim != 2
                    or cache.shape[1] != 32 or not all(np.isfinite(x).all() for x in (cache, teacher, executed))):
                raise ValueError("invalid action shape/value")
            if executed.tobytes() != (teacher if policy else cache).tobytes():
                raise ValueError("selected chunk was not served")
            if not np.array_equal(executed[:5, :7], np.asarray(d["served_head"], np.float32)):
                raise ValueError("dec/profile action mismatch")
            if i + 1 in ds:
                from exp.offline_search.closed_loop.blind import policy_tail_chunk
                tail = ds[i + 1]
                if tail["vision"] or tail["src"] != ("policy_tail" if policy else "cache_blind"):
                    raise ValueError("10-control commit missing")
                if not np.array_equal(np.asarray(tail["served_head"], np.float32), policy_tail_chunk(executed)[:5, :7]):
                    raise ValueError("committed tail mismatch")
            ident = dict(arm=arm, uid=uid, attempt=r["attempt"], task_id=r["task_id"], init=r["init"], step=i,
                         model=r["model"], suite=r["suite"], lib=r["lib"])
            row = {**ident, "Y": int(bool(journal["success"])), "N": len(ds), "M": M,
                   "future_decisions": len(ds) - i, "future_misses": sum(not ds[j]["hit"] for j in range(i, len(ds)))}
            for group in ("assignment", "distance", "guards", "timing", "cost", "state", "extensions"):
                flatten(r[group], group + ".", row)
            flatten({k: v for k, v in r["retrieval"].items() if not isinstance(v, list)}, "retrieval.", row)
            for k in ("metric_code", "nearest_distances"):
                for j, value in enumerate(r["retrieval"][k]):
                    row[f"retrieval.{k}.{j}"] = value
            anchors.append(row)
            ret = r["retrieval"]
            for j in range(len(ret["rows"])):
                neighbours.append({**ident, "rank": j + 1, **{k: ret[k][j] for k in (
                    "rows", "distances", "ranked_distances", "weights", "episodes", "progress", "steps", "phase_rows", "phase_steps")}})
            for t in range(len(cache)):
                item = {**ident, "chunk_step": t, "in_commit10": t < 10,
                        "rms": r["distance"]["per_step_rms"][t], "l2": r["distance"]["per_step_l2"][t],
                        "neighbour_dispersion": ret["dispersion_per_step"][t]}
                for name, value in (("cache", cache), ("policy", teacher), ("executed", executed)):
                    item.update({f"{name}.{j}": float(value[t, j]) for j in range(7)})
                steps.append(item)
    return anchors, neighbours, steps, episodes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--require-inputs", action="store_true")
    ap.add_argument("--snapshot-root", type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    tables = [[], [], [], []]
    for arm in args.arms:
        for dst, src in zip(tables, load(args.run_root, arm, require_inputs=args.require_inputs)):
            dst.extend(src)
    if args.snapshot_root:
        index = {(r["uid"], r["attempt"], r["step"]): r for r in tables[0]}
        matched = set()
        for path in args.snapshot_root.rglob("step_*.npz"):
            with np.load(path, allow_pickle=False) as data:
                meta = json.loads(str(data["metadata_json"]))
                key = (meta["task_uid"], int(meta.get("attempt", 1)), meta["decision_step"])
                if key not in index:
                    continue  # other arm or rejected retry
                if key in matched:
                    raise ValueError("duplicate accepted simulator snapshot")
                matched.add(key)
                index[key].update(snapshot_path=str(path), snapshot_env_timestep=int(data["env_timestep"]),
                                  snapshot_restore_certified=meta["restore_certified"])
    for name, rows in zip(("anchors", "neighbours", "action_steps", "episodes"), tables):
        write_csv(args.out / f"{name}.csv", rows)
    a, _, _, eps = tables
    accounting = dict(episodes=len(eps), anchors=len(a), decisions=sum(e["decisions"] for e in eps),
        misses=sum(e["misses"] for e in eps), physical_full_policy_forwards=len(a),
        profiling_only_policy_calls=sum(r["cost.profiling_only_policy_calls"] for r in a),
        profiling_only_s23_wall_hours=sum(r["cost.profiling_only_s23_ms"] for r in a) / 3.6e6,
        note="stage wall times include queue/synchronization; ledger IR remains based only on dec vision/hit")
    accounting["by_arm"] = {}
    for arm in args.arms:
        group = [e for e in eps if e["arm"] == arm]
        N, V, M = [sum(e[k] for e in group) for k in ("decisions", "anchors", "misses")]
        c1 = {"pi05": .152, "groot": .148}[group[0]["model"]]
        accounting["by_arm"][arm] = dict(N=N, V=V, M=M, v=V / N, m=M / N, c1=c1,
            ledger_IR=(c1 * V + (1 - c1) * M) / N,
            profiling_only_IR=(1 - c1) * (V - M) / N, physical_IR=V / N)
    (args.out / "accounting.json").write_text(json.dumps(accounting, indent=2) + "\n")
    print(json.dumps(accounting))


if __name__ == "__main__":
    main()
