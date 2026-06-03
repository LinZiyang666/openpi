"""Per-base Pareto plots for the threshold-pareto experiment.

For each of the 4 bases (d1/d3/d4/d5), plot eval yamls as (inf_ratio, SR)
scatter points + the upper-left Pareto frontier + the inf~0 always_hit anchor
Then a 5th overlay panel with the 4 frontiers on one plane for direct
cross-depth comparison. Also dumps a per-yaml CSV.

This answers the experiment's core question: does trajectory (higher depth) give
a better "retrieval score as confidence signal" — i.e. does its Pareto frontier
dominate d1's at any inf_ratio?
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# parse "..._d{N}__fh{FH}_ws{WS}_quantile" out of eval yaml_ids
_TAG_RE = re.compile(r"__d(\d+)__fh(\d+)_ws(\d+)_")
_ANCHOR_RE = re.compile(r"__d(\d+)__anchor_allmiss$")
# wsweep best always_hit SR per base (the inf~0 anchor)
_ALWAYS_HIT_SR = {1: 74.0, 3: 72.0, 4: 72.0, 5: 72.0}
_BASE_COLOR = {1: "#1f77b4", 3: "#ff7f0e", 4: "#2ca02c", 5: "#d62728"}


def _load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def _load_threshold_maps(sr_path: Path, inf_path: Path):
    if sr_path.exists() and inf_path.exists():
        return _load(sr_path), _load(inf_path)

    root = sr_path.parent
    sr = {}
    inf = {}
    for depth in (1, 3, 4, 5):
        droot = root / f"d{depth}"
        sr.update(_load(droot / "results.json"))
        inf.update(_load(droot / "inf_ratio.json"))
    return sr, inf


def _pareto_front(points):
    """Upper-left frontier: sort by inf asc; keep points with strictly higher SR."""
    pts = sorted(points, key=lambda p: (p[0], -p[1]))
    front, best = [], -1.0
    for inf, sr, *_ in pts:
        if sr > best:
            front.append((inf, sr))
            best = sr
    return front


def _parse_always_hit(spec: str) -> dict[int, float]:
    if not spec:
        return dict(_ALWAYS_HIT_SR)
    out = dict(_ALWAYS_HIT_SR)
    for item in spec.split(","):
        if not item.strip():
            continue
        key, value = item.split("=", 1)
        depth = int(key.strip().removeprefix("d"))
        out[depth] = float(value)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sr", default="exp/weighted_sum/data/libero_spatial/threshold_pareto/results.json")
    ap.add_argument("--inf-ratio", default="exp/weighted_sum/data/libero_spatial/threshold_pareto/inf_ratio.json")
    ap.add_argument("--suite-label", default="libero_spatial")
    ap.add_argument(
        "--always-hit-sr",
        default="",
        help="comma list of inf≈0 anchors in percent, e.g. d1=52,d3=51,d4=46,d5=49",
    )
    ap.add_argument(
        "--out-dir",
        default=str(Path(__file__).resolve().parent / "libero_spatial" / "threshold_pareto"),
    )
    args = ap.parse_args()
    always_hit_sr = _parse_always_hit(args.always_hit_sr)

    sr_data, inf_data = _load_threshold_maps(Path(args.sr), Path(args.inf_ratio))
    by_base: dict[int, list] = defaultdict(list)
    csv_rows = []
    for yid, sr_rec in sr_data.items():
        m = _TAG_RE.search(yid)
        anchor = _ANCHOR_RE.search(yid)
        if not (m or anchor) or yid not in inf_data:
            continue
        if m:
            d, fh, ws = int(m.group(1)), int(m.group(2)) / 100, int(m.group(3)) / 100
        else:
            d, fh, ws = int(anchor.group(1)), None, None
        sr_pct = (sr_rec["success_rate"] if isinstance(sr_rec, dict) else sr_rec) * 100
        ir_rec = inf_data[yid]
        ir = ir_rec.get("inference_ratio")
        if ir is None:
            continue
        by_base[d].append((ir, sr_pct, fh, ws, yid))
        csv_rows.append({
            "yaml_id": yid, "base_depth": d, "f_FH_target": fh, "f_WS_target": ws,
            "inf_ratio": round(ir, 4), "success_rate_pct": round(sr_pct, 2),
            "n_eval": ir_rec.get("n_steps", sr_rec.get("n") if isinstance(sr_rec, dict) else None),
            "n_FH": ir_rec.get("n_full_hit"), "n_WS": ir_rec.get("n_warm_start"), "n_MISS": ir_rec.get("n_miss"),
        })

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "threshold_pareto_per_yaml.csv"
    if csv_rows:
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            w.writeheader()
            w.writerows(sorted(csv_rows, key=lambda r: (r["base_depth"], r["inf_ratio"])))
        print(f"wrote {len(csv_rows)} rows to {csv_path}")

    # ── Figure 1: 2x2 per-base ──
    fig1, axes = plt.subplots(2, 2, figsize=(13, 10), sharex=True, sharey=True)
    for ax, d in zip(axes.flat, [1, 3, 4, 5]):
        pts = by_base.get(d, [])
        color = _BASE_COLOR[d]
        ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=22, alpha=0.55, color=color, label=f"d{d} cells (n={len(pts)})")
        front = _pareto_front(pts)
        if front:
            ax.plot([p[0] for p in front], [p[1] for p in front], "-o", color="red", lw=2, ms=5, label="Pareto frontier")
        ax.axhline(always_hit_sr[d], color="gray", ls="--", alpha=0.7, label=f"always_hit SR {always_hit_sr[d]:.0f}% (inf≈0)")
        ax.set_title(f"d{d} (n={len(pts)}, mean SR={sum(p[1] for p in pts)/max(len(pts),1):.1f}%)")
        ax.set_xlabel("inference_ratio (lower = more cache reuse)")
        ax.set_ylabel("success rate (%)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", fontsize=8)
    counts = {d: len(by_base.get(d, [])) for d in [1, 3, 4, 5]}
    n_label = next(iter(set(counts.values()))) if len(set(counts.values())) == 1 else "variable"
    fig1.suptitle(
        f"Threshold-pareto per-base: SR vs inference_ratio ({args.suite_label}, {n_label} yamls per base)",
        fontsize=12,
    )
    fig1.tight_layout()
    for ext in ("png", "pdf"):
        fig1.savefig(out / f"threshold_pareto_per_base.{ext}", dpi=180)
    plt.close(fig1)

    # ── Figure 2: overlay (4 frontiers + light scatter) ──
    fig2, ax = plt.subplots(figsize=(10, 7))
    for d in [1, 3, 4, 5]:
        pts = by_base.get(d, [])
        if not pts:
            continue
        c = _BASE_COLOR[d]
        ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=14, alpha=0.25, color=c)
        front = _pareto_front(pts)
        if front:
            ax.plot([p[0] for p in front], [p[1] for p in front], "-o", color=c, lw=2, ms=5, label=f"d{d} frontier (always_hit {always_hit_sr[d]:.0f}%)")
        ax.scatter([0], [always_hit_sr[d]], marker="*", s=140, color=c, edgecolor="black", linewidth=0.8, zorder=5)
    ax.set_xlabel("inference_ratio (lower = more cache reuse / less compute)")
    ax.set_ylabel("success rate (%)")
    ax.set_title("Threshold-pareto overlay: d1/d3/d4/d5 frontiers compared\n★ = wsweep always_hit anchor (inf≈0)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig2.tight_layout()
    for ext in ("png", "pdf"):
        fig2.savefig(out / f"threshold_pareto_overlay.{ext}", dpi=180)
    plt.close(fig2)

    # ── stdout summary ──
    print("\n=== per-base summary ===")
    print(f"{'depth':>5} {'n':>4} {'meanSR':>7} {'maxSR':>6} {'minInf':>7} {'frontier-pts':>13}")
    for d in [1, 3, 4, 5]:
        pts = by_base.get(d, [])
        if not pts:
            continue
        srs = [p[1] for p in pts]; irs = [p[0] for p in pts]
        ft = _pareto_front(pts)
        print(f"{d:>5} {len(pts):>4} {sum(srs)/len(srs):6.1f}% {max(srs):5.1f}% {min(irs):7.3f} {len(ft):>13}")
    print(f"\nsaved per-base + overlay figures to {out}")


if __name__ == "__main__":
    main()
