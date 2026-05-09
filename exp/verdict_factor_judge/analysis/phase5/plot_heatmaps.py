"""Phase 5 heatmaps — G1 (window × channel) + G5 ((FH, WS) × recipe).

For each base_recipe in G1: heatmap of SR over (window, channel) per desc.
For G5: per-recipe (FH × WS) SR heatmap with inf overlay.

Output: exp/verdict_factor_judge/analysis/phase5/heatmaps.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from exp.verdict_factor_judge.phase5.spec import (  # noqa: E402
    G1_BASE_RECIPES, G1_CHANNELS, G1_DESCS, G1_WINDOWS,
    G5_RECIPES, G5_THRESHOLD_GRID,
)


def _load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _g1_heatmap(rows: list[dict], ax, base: str, desc: str) -> None:
    """rows: g1 cells filtered to one (base, desc); plot SR (window × channel)."""
    grid = np.full((len(G1_WINDOWS), len(G1_CHANNELS)), np.nan)
    for r in rows:
        axis = r.get("axis_tag", "")
        # axis_tag like "p1_state_jerk__win-0-3"
        if not axis.startswith(f"{base}_"):
            continue
        if f"_{desc}" not in axis:
            continue
        ch = "state" if "_state_" in axis else "action"
        win_str = axis.split("__win-")[1]
        try:
            p, f = (int(x) for x in win_str.split("-"))
        except ValueError:
            continue
        if (p, f) not in G1_WINDOWS:
            continue
        i = G1_WINDOWS.index((p, f))
        j = G1_CHANNELS.index(ch)
        grid[i, j] = r.get("success_rate") or np.nan
    im = ax.imshow(grid, cmap="viridis", aspect="auto", vmin=0.5, vmax=1.0)
    ax.set_xticks(range(len(G1_CHANNELS)))
    ax.set_xticklabels(G1_CHANNELS)
    ax.set_yticks(range(len(G1_WINDOWS)))
    ax.set_yticklabels([f"({p},{f})" for (p, f) in G1_WINDOWS])
    ax.set_title(f"G1 SR — base={base} desc={desc}", fontsize=10)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            v = grid[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if v < 0.85 else "black", fontsize=8)
    return im


def _g5_heatmap(rows: list[dict], ax, recipe_short: str) -> None:
    fh_vals = sorted({fh for (fh, _) in G5_THRESHOLD_GRID})
    ws_vals = sorted({ws for (_, ws) in G5_THRESHOLD_GRID})
    grid = np.full((len(fh_vals), len(ws_vals)), np.nan)
    for r in rows:
        if r.get("base_recipe") != recipe_short:
            continue
        if r.get("group") != "g5":
            continue
        try:
            fh = float(r.get("fh_ratio"))
            ws = float(r.get("ws_ratio"))
        except (TypeError, ValueError):
            continue
        if fh not in fh_vals or ws not in ws_vals:
            continue
        i = fh_vals.index(fh)
        j = ws_vals.index(ws)
        grid[i, j] = r.get("success_rate") or np.nan
    ax.imshow(grid, cmap="viridis", aspect="auto", vmin=0.5, vmax=1.0)
    ax.set_xticks(range(len(ws_vals)))
    ax.set_xticklabels([f"WS={w}" for w in ws_vals], fontsize=8)
    ax.set_yticks(range(len(fh_vals)))
    ax.set_yticklabels([f"FH={f}" for f in fh_vals], fontsize=8)
    ax.set_title(f"G5 SR — recipe={recipe_short}", fontsize=10)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            v = grid[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if v < 0.85 else "black", fontsize=7)


def main() -> None:
    repo = Path(__file__).resolve().parents[4]
    rows = _load(
        repo / "exp/verdict_factor_judge/data/phase5_systematic/per_yaml_summary.jsonl"
    )
    if not rows:
        print("no phase5_systematic data found — skipping heatmaps")
        return

    # Layout: top half G1 (2 base × 2 desc = 4 plots), bottom half G5 (3 recipes)
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle("Phase 5 heatmaps — G1 (window × channel) per (base, desc)  +  G5 (FH × WS) per recipe",
                 fontsize=13, fontweight="bold")

    g1_rows = [r for r in rows if r.get("group") == "g1"]
    plots = [(b, d) for b in G1_BASE_RECIPES for d in G1_DESCS]
    for ax, (b, d) in zip(axes[0], plots):
        _g1_heatmap(g1_rows, ax, b, d)

    g5_rows = [r for r in rows if r.get("group") == "g5"]
    for ax, recipe_short in zip(axes[1][:3], G5_RECIPES):
        # G5_RECIPES are full ids; map to short
        short = "p1" if recipe_short.startswith("p1") else (
            "p2" if recipe_short.startswith("p2") else "g6"
        )
        _g5_heatmap(g5_rows, ax, short)
    axes[1][3].axis("off")

    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    out = repo / "exp/verdict_factor_judge/analysis/phase5/heatmaps.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=160, bbox_inches="tight")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
