"""Summarise the fusion-weight ablation: per-arm success on the A pool with paired comparisons.

Inputs are the two lanes' native outputs, read as they are:

* Pi0.5: the conductor journal (``task_uid = <arm>:eval:<task>:<episode>``,
  one terminal row per uid with ``success``);
* GR00T: one per-episode JSON per cell from ``orchestrate_search``
  (``task_id`` / ``init_state_idx`` / ``success``), plus, optionally, the
  historical grid cells in the same shape so the leader can be paired in.

Per arm: n, success rate, Wilson 95% interval. Per suite: every pair of arms
compared on the shared (task, init) keys with an exact sign-flip test on the
discordant episodes (McNemar without continuity correction, two-sided).

Usage:
  python exp/weighted_sum/analysis/lcw_ablation_summary.py \
      --pi05-journal libero_spatial=<journal.jsonl> --pi05-journal libero_10=<journal.jsonl> \
      --groot-results libero_spatial=<abl_results dir> --groot-results libero_10=<abl_results dir> \
      --groot-grid libero_spatial=<r1_results/v0@3_v1@2_rs@1.json> ... --out <json>
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import pathlib


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def binom_two_sided(b: int, c: int) -> float:
    """Exact two-sided sign test on discordant counts (b wins for A, c wins for B)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def load_pi05(path: pathlib.Path) -> dict[str, dict[tuple[int, int], bool]]:
    arms: dict[str, dict[tuple[int, int], bool]] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if not r.get("accepted", True):
            continue
        arm, phase, task, ep = r["task_uid"].rsplit(":", 3)
        arms.setdefault(arm, {})[(int(task), int(ep))] = bool(r["success"])
    return arms


def load_groot(directory: pathlib.Path) -> dict[str, dict[tuple[int, int], bool]]:
    arms = {}
    for p in sorted(directory.glob("*.json")):
        if p.name.endswith(".partial.json"):
            continue
        rows = json.loads(p.read_text())
        arms[p.stem] = {(int(r["task_id"]), int(r["init_state_idx"])): bool(r["success"]) for r in rows}
    return arms


def load_grid_cell(path: pathlib.Path) -> dict[tuple[int, int], bool]:
    rows = json.loads(path.read_text())
    return {(int(r["task_id"]), int(r["init_state_idx"])): bool(r["success"]) for r in rows}


def summarise(suite: str, arms: dict[str, dict[tuple[int, int], bool]]) -> dict:
    out = {"suite": suite, "arms": {}, "pairs": []}
    for name, eps in arms.items():
        n, k = len(eps), sum(eps.values())
        lo, hi = wilson(k, n)
        per_task = {}
        for (t, _), s in eps.items():
            per_task.setdefault(t, [0, 0])
            per_task[t][0] += int(s)
            per_task[t][1] += 1
        out["arms"][name] = {"n": n, "successes": k, "sr": k / n if n else float("nan"),
                             "wilson95": [lo, hi],
                             "per_task_sr": {str(t): v[0] / v[1] for t, v in sorted(per_task.items())}}
        print(f"  {name:36s} n={n:4d} SR={k / n if n else float('nan'):.3f} [{lo:.3f}, {hi:.3f}]")
    for a, b in itertools.combinations(arms, 2):
        shared = set(arms[a]) & set(arms[b])
        wins_a = sum(arms[a][k] and not arms[b][k] for k in shared)
        wins_b = sum(arms[b][k] and not arms[a][k] for k in shared)
        diff = (sum(arms[a][k] for k in shared) - sum(arms[b][k] for k in shared)) / max(len(shared), 1)
        p = binom_two_sided(wins_a, wins_b)
        out["pairs"].append({"a": a, "b": b, "shared": len(shared), "diff_pp": 100 * diff,
                             "a_only": wins_a, "b_only": wins_b, "p_sign": p})
        print(f"    {a} vs {b}: shared={len(shared)} diff={100 * diff:+.1f} pp  discordant {wins_a}/{wins_b}  p={p:.3f}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pi05-journal", action="append", default=[], help="suite=path")
    ap.add_argument("--groot-results", action="append", default=[], help="suite=dir")
    ap.add_argument("--groot-grid", action="append", default=[], help="suite=path (historical leader cell json)")
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()
    report = {}
    for spec in args.pi05_journal:
        suite, path = spec.split("=", 1)
        print(f"== pi05 {suite}")
        report[f"pi05_{suite}"] = summarise(suite, load_pi05(pathlib.Path(path)))
    grids = dict(s.split("=", 1) for s in args.groot_grid)
    for spec in args.groot_results:
        suite, d = spec.split("=", 1)
        arms = load_groot(pathlib.Path(d))
        if suite in grids:
            arms["grid_leader:" + pathlib.Path(grids[suite]).stem] = load_grid_cell(pathlib.Path(grids[suite]))
        print(f"== groot {suite}")
        report[f"groot_{suite}"] = summarise(suite, arms)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1) + "\n")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
