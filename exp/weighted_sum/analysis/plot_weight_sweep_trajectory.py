"""Analyze the spatial_16 weight-sweep × depth trajectory re-search.

Answers H-3a vs H-mechanism:
- Best SR per depth (does a re-tuned weight at d3/d4/d5 recover the d1 ceiling?).
- Optimal weight vector (v0/v1/rs) drift across depths (does the optimum move
  when trajectory is on?).
- Per-depth weight→SR landscape.

Reads results.json ({stem__grid3_..._d{depth}: {success_rate,..}}) from the
weight-sweep journal summary.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_W_RE = {f: re.compile(rf"{f}@(\d+)") for f in ("vision_0", "vision_1", "robot_state")}
_DEPTH_RE = re.compile(r"__d(\d+)$")


def _parse(yaml_id):
    m = _DEPTH_RE.search(yaml_id)
    depth = int(m.group(1))
    w = {f: (int(r.search(yaml_id).group(1)) if r.search(yaml_id) else 0) / 100.0 for f, r in _W_RE.items()}
    return depth, w


def main():
    repo = Path(__file__).resolve().parents[3]
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--results",
        default=str(repo / "exp/weighted_sum/data/libero_spatial/trajectory_wsweep/results.json"),
    )
    ap.add_argument(
        "--out-dir",
        default=str(Path(__file__).resolve().parent / "libero_spatial" / "trajectory_wsweep"),
    )
    ap.add_argument("--d1-ceiling", type=float, default=74.0, help="reference d1 ceiling from weighted_sum")
    args = ap.parse_args()

    data = json.loads(Path(args.results).read_text())
    # depth -> list of (sr, weights, yaml_id)
    by_depth = collections.defaultdict(list)
    for yid, rec in data.items():
        depth, w = _parse(yid)
        sr = (rec["success_rate"] if isinstance(rec, dict) else rec) * 100
        by_depth[depth].append((sr, w, yid))

    depths = sorted(by_depth)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Best per depth + drift ──
    best = {}
    print(f"{'depth':>5} {'best SR':>8} {'v0':>5} {'v1':>5} {'rs':>5}  vs-d1ceil")
    for d in depths:
        sr, w, yid = max(by_depth[d], key=lambda t: t[0])
        best[d] = (sr, w)
        print(f"{d:>5} {sr:7.1f}% {w['vision_0']:5.2f} {w['vision_1']:5.2f} {w['robot_state']:5.2f}  {sr-args.d1_ceiling:+.1f}")
    d1b = best.get(1, (None,))[0]
    if d1b is not None:
        print(f"\nd1 best (this sweep) = {d1b:.1f}% ; trajectory-best recovers d1? " +
              " ".join(f"d{d}:{best[d][0]-d1b:+.1f}" for d in depths if d != 1))

    # ── Fig 1: best SR vs depth ──
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(depths, [best[d][0] for d in depths], marker="o", lw=2, label="best re-tuned SR per depth")
    ax.axhline(args.d1_ceiling, color="r", ls="--", label=f"weighted_sum d1 ceiling ({args.d1_ceiling:.0f}%)")
    if d1b is not None:
        ax.axhline(d1b, color="gray", ls=":", label=f"this-sweep d1 best ({d1b:.0f}%)")
    ax.set_xlabel("trajectory depth")
    ax.set_ylabel("best success rate over weight grid (%)")
    ax.set_title("spatial_16: best re-tuned SR per depth (H-3a recovery test)")
    ax.set_xticks(depths)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"wsweep_best_vs_depth.{ext}", dpi=200)
    plt.close(fig)

    # ── Fig 2: optimal weight drift across depth ──
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    for f in ("vision_0", "vision_1", "robot_state"):
        ax2.plot(depths, [best[d][1][f] for d in depths], marker="s", label=f)
    ax2.set_xlabel("trajectory depth")
    ax2.set_ylabel("optimal weight")
    ax2.set_title("spatial_16: optimal weight vector drift vs depth")
    ax2.set_xticks(depths)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    for ext in ("png", "pdf"):
        fig2.savefig(out / f"wsweep_weight_drift.{ext}", dpi=200)
    plt.close(fig2)

    print(f"\nsaved figures to {out}")


if __name__ == "__main__":
    main()
