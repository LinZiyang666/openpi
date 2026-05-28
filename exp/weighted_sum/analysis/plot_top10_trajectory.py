"""Dedicated trajectory figure for the global top-10 weighted_sum configs.

One panel, one line per top-10 base config, success rate vs depth (d1 baseline
from all_results.csv + d3/d4/d5/d6 from the trajectory journal). Mirrors
plot_trajectory_results but isolates Group 2 (the 10 best) per owner request.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import re
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_DEPTH_RE = re.compile(r"__d(\d+)$")


def main():
    repo = Path(__file__).resolve().parents[3]
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(repo / "exp/weighted_sum/data/trajectory/results.json"))
    ap.add_argument("--baseline", default=str(repo / "exp/weighted_sum/data/phase2/all_results.csv"))
    ap.add_argument("--top10-dir", default=str(repo / "exp/weighted_sum/config/top10"))
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "top10_trajectory.png"))
    args = ap.parse_args()

    top10 = sorted(p.stem for p in Path(args.top10_dir).glob("*.yaml"))

    base = collections.defaultdict(list)
    for r in csv.DictReader(open(args.baseline)):
        base[r["yaml_id"]].append(float(r["success_rate"]))
    d1 = {k: statistics.mean(v) for k, v in base.items()}

    traj = collections.defaultdict(dict)
    data = json.loads(Path(args.results).read_text())
    for yid, rec in data.items():
        m = _DEPTH_RE.search(yid)
        if m:
            traj[yid[: m.start()]][int(m.group(1))] = float(rec["success_rate"] if isinstance(rec, dict) else rec)

    depths = sorted({d for b in top10 for d in traj.get(b, {})})
    x = [1] + depths

    fig, ax = plt.subplots(figsize=(10, 6))
    deltas, perdepth = [], collections.defaultdict(list)
    for b in top10:
        kb = "spatial_16" if "spatial_pool_16" in b else "max_pool"
        w = b.split("grid3_", 1)[-1].replace("vision_0@", "v0:").replace("_vision_1@", " v1:").replace("_robot_state@", " rs:")
        ys = [d1.get(b)] + [traj[b].get(d) for d in depths]
        ys = [v * 100 if v is not None else None for v in ys]
        ax.plot(x, ys, marker="o", ls="-" if kb == "spatial_16" else "--", label=f"{kb} {w}", markersize=5)
        best = max(traj[b], key=lambda d: traj[b][d])
        deltas.append((traj[b][best] - d1[b]) * 100)
        for d in depths:
            perdepth[d].append(traj[b][d] * 100)

    d1_mean = statistics.mean([d1[b] * 100 for b in top10])
    ax.plot(x, [d1_mean] + [statistics.mean(perdepth[d]) for d in depths], color="k", lw=3, marker="s",
            label=f"TOP-10 MEAN (mean Δbest={statistics.mean(deltas):+.1f}pp)", zorder=10)
    ax.axhline(d1_mean, color="gray", ls=":", lw=1)
    ax.set_title("Top-10 weighted_sum configs under trajectory search (libero_spatial, always_hit)")
    ax.set_xlabel("trajectory depth (1 = weighted_sum baseline)")
    ax.set_ylabel("success rate (%)")
    ax.set_xticks(x)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2, loc="lower left")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(args.out.replace(".png", f".{ext}"), dpi=200)

    print("=== TOP-10 trajectory ===")
    print(f"{'config':58} {'d1':>5} " + " ".join(f"d{d:>3}" for d in depths) + "  Δbest")
    for b in top10:
        cells = " ".join(f"{traj[b][d]*100:4.0f}" for d in depths)
        best = max(traj[b], key=lambda d: traj[b][d])
        print(f"{b[:58]:58} {d1[b]*100:4.0f}% {cells}  {(traj[b][best]-d1[b])*100:+.0f}")
    print(f"\nTOP-10 mean: d1={d1_mean:.1f}% " + " ".join(f"d{d}={statistics.mean(perdepth[d]):.1f}%" for d in depths))
    print(f"TOP-10 mean Δ(best traj - d1) = {statistics.mean(deltas):+.1f}pp ; all-negative={all(x<0 for x in deltas)}")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
