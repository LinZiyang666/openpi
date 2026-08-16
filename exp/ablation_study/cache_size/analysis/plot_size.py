"""Plot the size curve for the cache-size ablation (plan §8.5).

Two panels:

*   **SR vs library size** with the task-cluster CI band, the teacher anchor as a
    horizontal reference, and the delta margin shaded. The x-axis is the
    **realized** mean trajectories per task, not the nominal tier target --
    low-success tasks top out below it, so plotting the nominal number would
    overstate the library at the upper tiers.
*   **retrieval cost vs library size**, because the same axis that buys success
    also buys latency; at the top tier retrieval approaches the cost of the
    teacher inference it is meant to replace.

The plot is descriptive. The confirmatory verdict comes from the Holm-adjusted
family in ``analyze_size.py``; annotations here therefore label the plateau axis
as descriptive rather than implying a test.

**Single source of truth**: every plotted number -- per-tier SR, its CI, the
teacher anchor, the verdict -- is read from the analysis JSON. They are not
accepted as arguments, because a caller who could pass them separately could
draw a curve that no analysis produced, and nothing would flag it. Only genuinely
independent artifacts (the size grid, entry counts, measured latency) come in
from outside.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TIERS = ("S1", "S2", "S3", "S4", "S5", "S6")


def _x_values(grid: dict | None) -> list[float]:
    if grid and "mean_realized" in grid:
        return [grid["mean_realized"][t] for t in TIERS]
    return [1, 2, 5, 10, 20, 45]


def authoritative_series(result: dict) -> tuple[dict, dict, float]:
    """Pull SR / CI / teacher out of the analysis JSON, or fail loudly.

    Missing or partial fields mean the JSON did not come from a current
    ``analyze_size`` run; drawing anyway would produce a figure with no
    traceable provenance.
    """
    for key in ("tier_sr", "tier_ci", "teacher_sr"):
        if key not in result:
            raise SystemExit(
                f"analysis JSON lacks {key!r}; regenerate it with the current "
                "analyze_size.py rather than supplying the numbers by hand"
            )
    tier_sr, tier_ci = result["tier_sr"], result["tier_ci"]
    missing = [t for t in TIERS if t not in tier_sr or t not in tier_ci]
    if missing:
        raise SystemExit(f"analysis JSON is missing tiers {missing}")
    return tier_sr, {t: tuple(tier_ci[t]) for t in TIERS}, float(result["teacher_sr"])


def plot(
    result: dict,
    *,
    grid: dict | None,
    entries: dict[str, int] | None,
    latency_ms: dict[str, float] | None,
    suite: str,
    out_path: pathlib.Path,
    delta: float = 0.05,
) -> None:
    tier_sr, tier_ci, teacher_sr = authoritative_series(result)
    x = _x_values(grid)
    y = [tier_sr[t] for t in TIERS]
    lo = [tier_ci[t][0] for t in TIERS]
    hi = [tier_ci[t][1] for t in TIERS]

    n_panels = 2 if latency_ms else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(6.5 * n_panels, 4.6))
    axes = [axes] if n_panels == 1 else list(axes)

    ax = axes[0]
    ax.fill_between(x, lo, hi, alpha=0.18, label="95% task-cluster CI")
    ax.plot(x, y, marker="o", label="pure cache (always_hit)")
    ax.axhline(teacher_sr, linestyle="--", label=f"teacher anchor ({teacher_sr:.3f})")
    ax.axhspan(teacher_sr - delta, teacher_sr, alpha=0.10,
               label=f"practical margin δ={delta:.0%}")
    ax.set_xscale("log")
    ax.set_xlabel("library size — realized successful trajectories per task (log)")
    ax.set_ylabel("episode success rate (A pool, 500 ep/arm)")

    v = result.get("verdict", {})
    branch = v.get("branch", "?")
    ax.set_title(
        f"{suite} — branch {branch}: Q={v.get('q','?')} / D={v.get('d','?')}"
        f"\nplateau P={v.get('p','?')} (descriptive)",
        fontsize=10,
    )
    for tier, xi, yi in zip(TIERS, x, y):
        label = tier if not entries else f"{tier}\n{entries.get(tier, 0):,}e"
        ax.annotate(label, (xi, yi), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=7)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)

    if latency_ms:
        ax2 = axes[1]
        ax2.plot(x, [latency_ms[t] for t in TIERS], marker="s", color="tab:red")
        ax2.set_xscale("log")
        ax2.set_xlabel("library size — realized successful trajectories per task (log)")
        ax2.set_ylabel("retrieval latency per step (ms)")
        ax2.set_title("cost of the same axis", fontsize=10)
        ax2.grid(alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--result-json", required=True,
                    help="analyze_size.py output -- the sole source of SR/CI/teacher")
    ap.add_argument("--grid", default=None, help="size grid, for realized x values")
    ap.add_argument("--entries", default=None, help='json: {"S1": 212, ...}')
    ap.add_argument("--latency-ms", default=None, help='json: {"S1": 2.7, ...}')
    ap.add_argument("--suite", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    grid = None
    if args.grid:
        import yaml

        grid = yaml.safe_load(pathlib.Path(args.grid).read_text())

    plot(
        json.loads(pathlib.Path(args.result_json).read_text()),
        grid=grid,
        entries=json.loads(args.entries) if args.entries else None,
        latency_ms=json.loads(args.latency_ms) if args.latency_ms else None,
        suite=args.suite,
        out_path=pathlib.Path(args.out),
    )
    print(args.out)


if __name__ == "__main__":
    main()
