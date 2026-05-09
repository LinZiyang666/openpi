"""Per-recipe 3D bar charts for Phase 3.

11 recipes (g1..g11) -> 11 subplots in a 4x3 grid. Each subplot:

    x = configured FH_ratio (warmup quantile cut producing fh_thr)
    y = configured WS_ratio (warmup quantile cut producing ws_thr)
    z = success_rate (eval, ~100 ep, 10 tasks)
    color = actual inference_ratio
            (0 · n_full_hit + 0.75 · n_warm_start + 1.0 · n_miss) / N
            with warm cost 0.75 = warm_start_t=0.5 lock-in.

Colormap is forward viridis: dark purple = LOW inf (cheap, good),
bright yellow = HIGH inf (expensive). Combined with bar height (SR), the
ideal winner cells are TALL + DARK PURPLE.

Input:  exp/verdict_factor_judge/data/phase3/per_yaml_summary.jsonl
Output: exp/verdict_factor_judge/analysis/phase3_3d_bars.png

Usage::

    MPLBACKEND=Agg uv run python -m exp.verdict_factor_judge.analysis.plot_3d_bars_phase3
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # noqa: E402  headless / no Qt
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from mpl_toolkits.mplot3d import Axes3D  # noqa: E402,F401  registers 3d projection


RATIOS = [0.2, 0.3, 0.4, 0.5]
BAR_DX = 0.07
BAR_DY = 0.07
GRID_NROWS = 3
GRID_NCOLS = 4

# Empirical data range (per_yaml_summary.jsonl, 176 cells):
#   SR  ∈ [0.52, 1.00], p05=0.59     → z floor 0.5 keeps full distribution
#                                       visible without flattening high-SR
#                                       bars against 0.
#   inf ∈ [0.00, 0.77], p95=0.735    → colorbar 0..0.8 brackets the data;
#                                       always-WARM @ t=0.5 baseline is at
#                                       inf=0.75 so 0.8 cap keeps that tier
#                                       distinguishable from the few cells
#                                       above it.
Z_FLOOR = 0.50
Z_CEIL = 1.00
INF_VMIN = 0.00
INF_VMAX = 0.80
WARM_COST = 0.75    # warm_start_t=0.5 → cost = 1 - 0.5*(1-0.5) = 0.75


def _load_rows(summary_path: Path) -> list[dict]:
    out: list[dict] = []
    for line in summary_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        if "error" in d or not d.get("n_eval_verdicts"):
            continue
        out.append(d)
    return out


def _recipe_sort_key(rid: str) -> int:
    # g1_f1b_... -> 1, g11_... -> 11
    return int(rid.split("_", 1)[0][1:])


def _short_label(rid: str) -> str:
    # "g2_f1b_t_w_long_risk_d_jerk" -> "g2 f1b·t·w_long_risk·d_jerk"
    head, _, tail = rid.partition("_")
    return f"{head}  {tail.replace('_', ' ')}"


def main() -> None:
    repo = Path(__file__).resolve().parents[3]
    summary_path = (
        repo / "exp/verdict_factor_judge/data/phase3/per_yaml_summary.jsonl"
    )
    out_path = repo / "exp/verdict_factor_judge/analysis/phase3_3d_bars.png"

    if not summary_path.exists():
        raise SystemExit(f"phase3 summary not found: {summary_path}")

    rows = _load_rows(summary_path)
    by_recipe: dict[str, list[dict]] = {}
    for r in rows:
        by_recipe.setdefault(r["recipe_id"], []).append(r)
    recipes = sorted(by_recipe, key=_recipe_sort_key)

    # Forward viridis: dark purple = LOW inf (cheap, good), bright yellow =
    # HIGH inf (expensive). Combined with bar height (SR), winner cells are
    # TALL + DARK PURPLE.
    cmap = plt.get_cmap("viridis")
    norm = Normalize(vmin=INF_VMIN, vmax=INF_VMAX)

    fig = plt.figure(figsize=(40, 30))
    fig.suptitle(
        "Phase 3 — Per-recipe (FH_ratio, WS_ratio) -> SR  "
        f"(spatial16, 11 recipes × 16 cells; z=SR ∈ [{Z_FLOOR:.2f}, {Z_CEIL:.2f}], "
        f"color = inference_ratio ∈ [{INF_VMIN:.2f}, {INF_VMAX:.2f}], "
        "purple=cheap / yellow=expensive)",
        fontsize=18, fontweight="bold", y=0.995,
    )

    for idx, rid in enumerate(recipes):
        ax = fig.add_subplot(GRID_NROWS, GRID_NCOLS, idx + 1, projection="3d")
        cells = by_recipe[rid]

        xs, ys, zs, colors = [], [], [], []
        for c in cells:
            n = c["n_eval_verdicts"]
            inf = (
                c["n_full_hit"] * 0.0
                + c["n_warm_start"] * WARM_COST
                + c["n_miss"] * 1.0
            ) / n if n else 0.0
            xs.append(c["fh_ratio"])
            ys.append(c["ws_ratio"])
            zs.append(c["success_rate"])
            colors.append(cmap(norm(inf)))

        # bar3d: bottom-left corners shifted so the bar is centered on the
        # grid; bottom raised to Z_FLOOR so the visible bar height encodes
        # (SR - Z_FLOOR), expanding the readable dynamic range.
        x0 = [x - BAR_DX / 2 for x in xs]
        y0 = [y - BAR_DY / 2 for y in ys]
        z0 = [Z_FLOOR] * len(xs)
        # Clip negatives in case any cell drifts below the floor.
        dz = [max(z - Z_FLOOR, 0.0) for z in zs]

        ax.bar3d(
            x0, y0, z0,
            [BAR_DX] * len(xs), [BAR_DY] * len(ys), dz,
            color=colors, edgecolor="black", linewidth=0.5, shade=True,
        )

        ax.set_xticks(RATIOS)
        ax.set_yticks(RATIOS)
        ax.set_zlim(Z_FLOOR, Z_CEIL)
        ax.set_xlabel("FH_ratio", fontsize=12, labelpad=8)
        ax.set_ylabel("WS_ratio", fontsize=12, labelpad=8)
        ax.set_zlabel("SR", fontsize=12, labelpad=6)
        ax.set_title(_short_label(rid), fontsize=14, fontweight="bold", pad=8)
        ax.tick_params(axis="both", which="major", labelsize=10)
        ax.view_init(elev=22, azim=-58)

    # Hide the unused 12th cell.
    if len(recipes) < GRID_NROWS * GRID_NCOLS:
        for j in range(len(recipes), GRID_NROWS * GRID_NCOLS):
            ax_empty = fig.add_subplot(GRID_NROWS, GRID_NCOLS, j + 1)
            ax_empty.axis("off")

    # Subplot spacing — tight_layout doesn't play well with 3D axes, so
    # use explicit subplots_adjust and leave room on the right for the
    # shared colorbar.
    fig.subplots_adjust(
        left=0.03, right=0.92, bottom=0.04, top=0.95, wspace=0.05, hspace=0.18,
    )

    # Shared colorbar for the actual warm-firing rate.
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.94, 0.25, 0.012, 0.5])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label(
        "inference_ratio  =  (0·n_FH + 0.75·n_WS + 1.0·n_miss) / N    "
        "[purple = cheap, yellow = expensive]",
        fontsize=13,
    )
    cbar.ax.tick_params(labelsize=11)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(
        f"saved -> {out_path}  size: {out_path.stat().st_size / 1024:.1f} KB  "
        f"({len(recipes)} recipes, {len(rows)} cells)"
    )


if __name__ == "__main__":
    main()
