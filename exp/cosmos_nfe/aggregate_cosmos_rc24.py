"""Join the shard JSONs of the Cosmos Policy RoboCasa-2024 ladder (24 tasks x 50 trials per k) into one table.

usage: python -m exp.cosmos_nfe.aggregate_cosmos_rc24 --root exp/cosmos_nfe/data/results_rc24 [--root ...] \\
           --out exp/cosmos_nfe/data/aggregate_rc24.json

Reads <root>/k<k>/<task>_t<a>-<b>.json (written by run_robocasa_shard.py) from every root
(the trial ranges of one k are split across client boxes, so several roots are joined),
requires every shard complete, every task's trial ids 0..49 present exactly once, and
reports per-k macro success rate over the 24 tasks (equal task weights, the paper's
metric), the episode-pooled rate, per-task rates and query latency.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics

TASKS = [
    "PnPCounterToCab", "PnPCabToCounter", "PnPCounterToSink", "PnPSinkToCounter", "PnPCounterToMicrowave",
    "PnPMicrowaveToCounter", "PnPCounterToStove", "PnPStoveToCounter", "OpenSingleDoor", "CloseSingleDoor",
    "OpenDoubleDoor", "CloseDoubleDoor", "OpenDrawer", "CloseDrawer", "TurnOnStove", "TurnOffStove",
    "TurnOnSinkFaucet", "TurnOffSinkFaucet", "TurnSinkSpout", "CoffeeSetupMug", "CoffeeServeMug",
    "CoffeePressButton", "TurnOnMicrowave", "TurnOffMicrowave",
]


def join_point(files: list[pathlib.Path], trials: int = 50) -> dict:
    per_task: dict[str, dict[int, bool]] = {t: {} for t in TASKS}
    lat_summ, server_ms, walls = [], [], []
    for f in files:
        d = json.loads(f.read_text())
        if not d.get("complete"):
            raise SystemExit(f"{f}: shard not complete")
        t = d["task_name"]
        if t not in per_task:
            raise SystemExit(f"{f}: unknown task {t}")
        for tid, ok in d["task"]["trials"].items():
            tid = int(tid)
            if tid in per_task[t]:
                raise SystemExit(f"{f}: task {t} trial {tid} appears twice")
            per_task[t][tid] = bool(ok)
        lat_summ.append(d["latency_ms"])
        if d.get("server_ms_mean") is not None:
            server_ms.append(d["server_ms_mean"])
        walls.extend(d["task"].get("episode_s", []))
    for t, tr in per_task.items():
        if sorted(tr) != list(range(trials)):
            raise SystemExit(f"task {t}: trials present {len(tr)}/{trials} (missing {sorted(set(range(trials)) - set(tr))[:5]}...)")
    rates = {t: sum(tr.values()) / trials for t, tr in per_task.items()}
    n = trials * len(TASKS)
    ok = sum(sum(tr.values()) for tr in per_task.values())
    means = [s["mean"] for s in lat_summ if s.get("mean") is not None]
    return {
        "n_episodes": n,
        "n_successes": ok,
        "n_tasks": len(TASKS),
        "macro_success_rate": sum(rates.values()) / len(TASKS),
        "pooled_success_rate": ok / n,
        "per_task": {t: {"n": trials, "success_rate": rates[t]} for t in TASKS},
        "latency_ms_mean_of_shards": round(statistics.fmean(means), 2) if means else None,
        "server_ms_mean_of_shards": round(statistics.fmean(server_ms), 2) if server_ms else None,
        "episode_s_mean": round(statistics.fmean(walls), 1) if walls else None,
        "n_shards": len(files),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--trials", type=int, default=50)
    args = ap.parse_args()
    roots = [pathlib.Path(r) for r in args.root]
    ks = sorted({int(k.name[1:]) for r in roots for k in r.glob("k*") if k.is_dir()})
    out: dict = {"policy": "cosmos_policy_robocasa_predict2_2b", "benchmark": "robocasa_2024_24_tasks",
                 "num_steps_default": 5, "note": "k denoising steps = k+1 network evaluations (res_sampler sample_clean)",
                 "roots": [str(r) for r in roots], "per_k": {}}
    for k in ks:
        files = sorted(f for r in roots for f in (r / f"k{k}").glob("*_t*.json"))
        try:
            rec = join_point(files, args.trials)
        except SystemExit as e:
            print(f"skip k={k}: {e}")
            continue
        rec["k"] = k
        rec["nfe"] = k + 1
        out["per_k"][str(k)] = rec
        print(f"k={k}: macro SR {rec['macro_success_rate']:.4f}  pooled {rec['pooled_success_rate']:.4f}  (n={rec['n_episodes']}) "
              f"server~{rec['server_ms_mean_of_shards']} ms  shards={rec['n_shards']}")
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text(json.dumps(out, indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
