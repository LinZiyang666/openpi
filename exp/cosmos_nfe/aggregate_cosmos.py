"""Join the shard JSONs of the Cosmos Policy LIBERO ladder into one table.

usage: python -m exp.cosmos_nfe.aggregate_cosmos --root exp/cosmos_nfe/data/results --out exp/cosmos_nfe/data/aggregate.json

Reads <root>/<suite>/k<k>/shard*.json (written by run_libero_shard.py), requires every
shard to be complete, every task id 0..9 present exactly once and n episodes per task
equal across tasks, and reports per-k success rate, per-task rates and query latency.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics


def join_point(point_dir: pathlib.Path, expect_tasks: int = 10) -> dict:
    tasks: dict[int, dict] = {}
    lat: list[float] = []
    lat_summ = []
    for f in sorted(point_dir.glob("shard*.json")):
        d = json.loads(f.read_text())
        if not d.get("complete"):
            raise SystemExit(f"{f}: shard not complete")
        for tid, t in d["tasks"].items():
            tid = int(tid)
            if tid in tasks:
                raise SystemExit(f"{f}: task {tid} appears twice")
            tasks[tid] = t
        lat_summ.append(d["latency_ms"])
        lat.extend([])  # per-shard summaries only; raw samples stay in the shard files
    if sorted(tasks) != list(range(expect_tasks)):
        raise SystemExit(f"{point_dir}: tasks present {sorted(tasks)} != 0..{expect_tasks - 1}")
    ns = {t["n"] for t in tasks.values()}
    if len(ns) != 1:
        raise SystemExit(f"{point_dir}: unequal episodes per task {ns}")
    n = sum(t["n"] for t in tasks.values())
    ok = sum(t["successes"] for t in tasks.values())
    means = [s["mean"] for s in lat_summ if s.get("mean") is not None]
    return {
        "n_episodes": n,
        "n_successes": ok,
        "success_rate": ok / n,
        "per_task": {str(k): {"n": v["n"], "success_rate": v["successes"] / v["n"], "description": v.get("description")} for k, v in sorted(tasks.items())},
        "latency_ms_mean_of_shards": round(statistics.fmean(means), 2) if means else None,
        "latency_ms_p50_of_shards": round(statistics.median([s["p50"] for s in lat_summ if s.get("p50") is not None]), 2) if lat_summ else None,
        "n_shards": len(lat_summ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--expect-tasks", type=int, default=10)
    args = ap.parse_args()
    root = pathlib.Path(args.root)
    out: dict = {"policy": "cosmos_policy_libero_predict2_2b", "num_steps_default": 5, "suites": {}}
    for suite_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        per_k = {}
        for kdir in sorted(suite_dir.glob("k*")):
            if not any(kdir.glob("shard*.json")):
                continue
            try:
                per_k[int(kdir.name[1:])] = join_point(kdir, args.expect_tasks)
            except SystemExit as e:
                print(f"skip {kdir}: {e}")
        if per_k:
            out["suites"][suite_dir.name] = {str(k): v for k, v in sorted(per_k.items())}
            for k, v in sorted(per_k.items()):
                print(f"{suite_dir.name} k={k}: SR {v['success_rate']:.3f} (n={v['n_episodes']}) latency~{v['latency_ms_mean_of_shards']} ms")
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text(json.dumps(out, indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
