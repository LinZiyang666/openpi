"""weighted_sum result visualization (pure matplotlib, no mpltern dependency).

Reads all_results.csv (columns stage,keybuilder,v0,v1,rs,normalizer,n,success_rate)
and produces three figures:
  fig1_ternary.png   -- ternary heatmap, one triangle panel per keybuilder
                        (weight simplex v0/v1/rs, color = SR)
  fig2_keybuilder_box.png -- SR boxplot by keybuilder + per-keybuilder best-point scatter
  fig3_crossgpu.png  -- top10 cross-GPU paired slopes (jupyter H200 vs a100 A100);
                        needs top10_compare.csv (skipped if absent)

Usage (timan107, ephemeral matplotlib install, does not touch .venv):
  uv run --with matplotlib python exp/weighted_sum/plot_results.py \
      --all-results exp/weighted_sum/data/libero_spatial/phase2/all_results.csv \
      --compare exp/weighted_sum/data/libero_spatial/phase2/top10_compare.csv \
      --outdir exp/weighted_sum/analysis/libero_spatial/phase2
"""
from __future__ import annotations

import argparse
import csv
import math
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

# Triangle vertices: v0=bottom-left (0,0), v1=bottom-right (1,0), rs=top (0.5, sqrt(3)/2)
_RS_Y = math.sqrt(3) / 2.0


def _proj(v0: float, v1: float, rs: float) -> tuple[float, float]:
    s = v0 + v1 + rs
    if s <= 0:
        return 0.0, 0.0
    v0, v1, rs = v0 / s, v1 / s, rs / s
    return v1 + 0.5 * rs, rs * _RS_Y


def _read_all(path: str) -> list[dict]:
    rows = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            try:
                r["v0"], r["v1"], r["rs"] = float(r["v0"]), float(r["v1"]), float(r["rs"])
                r["success_rate"] = float(r["success_rate"])
                r["n"] = int(float(r.get("n", 0)))
            except (ValueError, KeyError):
                continue
            rows.append(r)
    return rows


def _tri_border(ax):
    xs = [0, 1, 0.5, 0]
    ys = [0, 0, _RS_Y, 0]
    ax.plot(xs, ys, color="black", lw=1.0)
    ax.text(-0.02, -0.04, "v0", ha="right", va="top", fontsize=9)
    ax.text(1.02, -0.04, "v1", ha="left", va="top", fontsize=9)
    ax.text(0.5, _RS_Y + 0.03, "rs", ha="center", va="bottom", fontsize=9)
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.12, _RS_Y + 0.1)
    ax.axis("off")
    ax.set_aspect("equal")


def fig_ternary(rows, outpath):
    # One panel per keybuilder; weight points projected onto the triangle, SR-colored (tricontourf smoothing + scatter).
    kbs = sorted({r["keybuilder"] for r in rows})
    # Put the 4 known keybuilders first, higher-SR keybuilders to the left
    order = ["cp1_max_pool", "cp1_spatial_pool_16", "cp1_spatial_pool_64", "cp1_mean_pool"]
    kbs = [k for k in order if k in kbs] + [k for k in kbs if k not in order]
    n = len(kbs)
    ncol = min(n, 4)
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 4.0 * nrow), squeeze=False)
    vmin = min(r["success_rate"] for r in rows) * 100
    vmax = max(r["success_rate"] for r in rows) * 100
    sc = None
    for i, kb in enumerate(kbs):
        ax = axes[i // ncol][i % ncol]
        pts = [r for r in rows if r["keybuilder"] == kb]
        xs, ys, srs = [], [], []
        for r in pts:
            x, y = _proj(r["v0"], r["v1"], r["rs"])
            xs.append(x)
            ys.append(y)
            srs.append(r["success_rate"] * 100)
        _tri_border(ax)
        if len(xs) >= 3:
            try:
                tri = mtri.Triangulation(xs, ys)
                ax.tricontourf(tri, srs, levels=12, cmap="viridis", vmin=vmin, vmax=vmax, alpha=0.85)
            except Exception:
                pass
        sc = ax.scatter(xs, ys, c=srs, cmap="viridis", vmin=vmin, vmax=vmax,
                        s=28, edgecolors="white", linewidths=0.4, zorder=5)
        # Mark the best point
        if srs:
            bi = max(range(len(srs)), key=lambda j: srs[j])
            ax.scatter([xs[bi]], [ys[bi]], marker="*", s=240, c="red", edgecolors="black", zorder=6)
            ax.set_title("%s\nbest %.0f%%" % (kb, srs[bi]), fontsize=10)
    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    if sc is not None:
        cbar = fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.7, pad=0.02)
        cbar.set_label("Success Rate (%)")
    fig.suptitle("weighted_sum: SR over weight simplex (v0/v1/rs) per keybuilder\n★=best; jupyter H200, always_hit, 100ep/config", fontsize=11)
    fig.savefig(outpath, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote", outpath)


def fig_box(rows, outpath):
    order = ["cp1_max_pool", "cp1_spatial_pool_16", "cp1_spatial_pool_64", "cp1_mean_pool"]
    by = defaultdict(list)
    for r in rows:
        by[r["keybuilder"]].append(r["success_rate"] * 100)
    kbs = [k for k in order if k in by] + [k for k in by if k not in order]
    data = [by[k] for k in kbs]
    fig, ax = plt.subplots(figsize=(1.6 * len(kbs) + 2, 5))
    bp = ax.boxplot(data, labels=[k.replace("cp1_", "") for k in kbs], showfliers=False, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#9ecae1")
    for i, k in enumerate(kbs):
        ys = by[k]
        xs = [i + 1 + (hash(str(j)) % 100 - 50) / 400.0 for j in range(len(ys))]
        ax.scatter(xs, ys, s=10, c="gray", alpha=0.5, zorder=3)
        ax.scatter([i + 1], [max(ys)], marker="*", s=200, c="red", edgecolors="black", zorder=5)
    ax.set_ylabel("Success Rate (%)")
    ax.set_title("SR distribution by keybuilder (all weight configs)\n★=best; box=spread across weights")
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(outpath, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote", outpath)


def fig_crossgpu(compare_path, outpath):
    if not os.path.exists(compare_path):
        print("skip fig3: no", compare_path)
        return
    rows = []
    with open(compare_path) as fh:
        for r in csv.DictReader(fh):
            try:
                rows.append((r["yaml"], float(r["jupyter_sr"]) * 100, float(r["a100_sr"]) * 100))
            except (ValueError, KeyError):
                continue
    if not rows:
        print("skip fig3: empty", compare_path)
        return
    rows.sort(key=lambda t: -t[1])
    fig, ax = plt.subplots(figsize=(7, 6))
    for i, (y, jsr, asr) in enumerate(rows):
        color = "tab:red" if abs(jsr - asr) >= 5 else "tab:gray"
        ax.plot([0, 1], [jsr, asr], "-o", color=color, alpha=0.8)
        ax.text(-0.03, jsr, y.split("__", 1)[-1][:22], ha="right", va="center", fontsize=7)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["jupyter (H200)", "a100 (A100)"])
    ax.set_ylabel("Success Rate (%)")
    ax.set_title("Top-10 configs: cross-GPU SR (H200 vs A100)\nred = |diff|>=5pp; same held-out 100ep/config")
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(outpath, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote", outpath)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all-results", required=True)
    ap.add_argument("--compare", default="")
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    rows = _read_all(a.all_results)
    print("loaded %d result rows" % len(rows))
    fig_ternary(rows, os.path.join(a.outdir, "fig1_ternary.png"))
    fig_box(rows, os.path.join(a.outdir, "fig2_keybuilder_box.png"))
    fig_crossgpu(a.compare, os.path.join(a.outdir, "fig3_crossgpu.png"))


if __name__ == "__main__":
    main()
