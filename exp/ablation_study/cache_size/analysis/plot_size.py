"""Plot the size curve for the cache-size ablation (plan §8.5).

Two panels:

*   **SR vs library size** with the task-cluster CI band, the teacher anchor as a
    horizontal reference, and the delta margin shaded. The x-axis is the
    **realized** mean trajectories per task, not the nominal tier target --
    low-success tasks top out below it, so plotting the nominal number would
    overstate the library at the upper tiers.
*   **retrieval cost vs library size** (optional), because the same axis that
    buys success also buys latency.

The plot is descriptive. The confirmatory verdict comes from the Holm-adjusted
family in ``analyze_size.py``; annotations here therefore label the plateau axis
as descriptive rather than implying a test.

**Single source of truth**: the figure reads only ``plot_data.json`` (built by
``emit_plot_data.py``, which copies the analyzer output verbatim and records its
sha256). Numbers are not accepted as arguments, because a caller who could pass
them separately could draw a curve that no analysis produced, and nothing would
flag it. Supplementary measurements (e.g. latency re-runs on other hardware)
land in the same file under per-point labels instead of ad-hoc CLI JSON.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TIERS = ("S1", "S2", "S3", "S4", "S5", "S6")


def authoritative_family(data: dict, key: str) -> dict:
    """Pull one family block out of plot_data.json, or fail loudly.

    Missing or partial fields mean the file did not come from a current
    ``emit_plot_data`` run; drawing anyway would produce a figure with no
    traceable provenance.
    """
    families = data.get("families")
    if not families:
        raise SystemExit("plot data lacks 'families'; regenerate with emit_plot_data.py")
    fam = families.get(key)
    if fam is None:
        raise SystemExit(
            f"family {key!r} not in plot data (have: {sorted(families)}); "
            "collect it with emit_plot_data.py first"
        )
    for field in ("teacher_success_rate", "points"):
        if field not in fam:
            raise SystemExit(f"family {key!r} lacks {field!r}; regenerate the plot data")
    if len(fam["points"]) < 2:
        raise SystemExit(f"family {key!r} has {len(fam['points'])} point(s); a size "
                         "curve needs the full collected ladder")
    for p in fam["points"]:
        for field in ("trajectories_per_task", "success_rate", "success_rate_ci95"):
            if field not in p:
                raise SystemExit(
                    f"family {key!r} point {p.get('source_arm')!r} lacks {field!r}; "
                    "regenerate the plot data"
                )
    return fam


FAMILY_CURVE_LABEL = {
    "all": "all collected trajectories (primary)",
    "success": "successful trajectories only (secondary)",
    None: "pure cache (always_hit)",
}


def plot(
    data: dict,
    *,
    families: list[str],
    out_path: pathlib.Path,
    latency_labels: list[str] | None = None,
) -> None:
    fams = [authoritative_family(data, f) for f in families]
    suites = {f.get("suite", key.split("/")[0]) for f, key in zip(fams, families)}
    if len(suites) != 1:
        raise SystemExit(
            f"one figure overlays one suite's families, got suites {sorted(suites)}"
        )
    suite = suites.pop()
    teachers = {float(f["teacher_success_rate"]) for f in fams}
    if len(teachers) != 1:
        raise SystemExit(
            f"families disagree on the teacher anchor ({sorted(teachers)}); "
            "they cannot share one reference line"
        )
    teacher_sr = teachers.pop()

    # The axis names what the size counts. A single family keeps the wording of
    # its own filter (plan §3.1b ruling 1); an overlay of both filters uses the
    # neutral form and lets the legend carry the distinction.
    if len(fams) == 1:
        filt = fams[0].get("outcome_filter")
        kind = {"all": "collected", "success": "successful"}.get(filt, "successful")
        x_label = f"{kind} trajectories per task in the cache (log; counts on points)"
    else:
        x_label = "trajectories per task in the cache (log; counts on points)"

    def _ms(v):
        return float(v) if not isinstance(v, dict) else float(v.get("median", v.get("ms")))

    # One latency curve per label. A label may legitimately cover only part of
    # the tiers (e.g. a host that could not fit the top tier in RAM); those
    # points are simply absent -- a visible gap, never an invented number. A
    # label with NO points in any plotted family is a typo and fails loudly.
    latency_series: dict[str, list[tuple[float, float]]] = {}
    for label in (latency_labels or []):
        for fam in fams:
            pts = [(p["trajectories_per_task"], _ms(p["retrieval_latency_ms"][label]))
                   for p in sorted(fam["points"], key=lambda q: q["trajectories_per_task"])
                   if label in (p.get("retrieval_latency_ms") or {})]
            if pts:
                latency_series[label] = pts
                break
        else:
            raise SystemExit(
                f"no point carries latency under label {label!r}; attach it with "
                "emit_plot_data.py --attach-latency first"
            )

    n_panels = 2 if latency_series else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(6.5 * n_panels, 4.6))
    axes = [axes] if n_panels == 1 else list(axes)

    def _entries_rows(ax_, fams_colors):
        """One sub-axis row per family: total cache entries at each point's x.

        The axis itself carries no tick numbers, so these rows are the only
        place the absolute library size appears on the figure.
        """
        rows = [(color, {p["trajectories_per_task"]: p["library_entries_total"]
                         for p in fam["points"] if "library_entries_total" in p})
                for fam, color in fams_colors]
        rows = [(c, m) for c, m in rows if m]
        for r, (color, entmap) in enumerate(rows):
            ypos = -0.075 - 0.06 * r
            ax_.annotate("entries:", xy=(0, ypos), xycoords="axes fraction",
                         ha="right", va="top", fontsize=7, color=color,
                         annotation_clip=False)
            for xi, e in entmap.items():
                ax_.annotate(f"{e:,}", xy=(xi, ypos),
                             xycoords=("data", "axes fraction"), ha="center",
                             va="top", fontsize=7, color=color,
                             annotation_clip=False)
        return len(rows)

    ax = axes[0]
    fam_colors: list[tuple[dict, str]] = []
    for i, fam in enumerate(fams):
        points = sorted(fam["points"], key=lambda p: p["trajectories_per_task"])
        x = [p["trajectories_per_task"] for p in points]
        y = [p["success_rate"] for p in points]
        lo = [p["success_rate_ci95"][0] for p in points]
        hi = [p["success_rate_ci95"][1] for p in points]
        curve_label = FAMILY_CURVE_LABEL.get(fam.get("outcome_filter"),
                                             "pure cache (always_hit)")
        line, = ax.plot(x, y, marker="o", label=curve_label)
        fam_colors.append((fam, line.get_color()))
        ax.fill_between(x, lo, hi, alpha=0.12, color=line.get_color())
        dy = 10 if i % 2 == 0 else -20
        for p, xi, yi in zip(points, x, y):
            ax.annotate(f"{p['trajectories_per_task']:g} traj/task\nSR {yi:.3f}",
                        (xi, yi), textcoords="offset points", xytext=(0, dy),
                        ha="center", fontsize=7, color=line.get_color())
    ax.axhline(teacher_sr, linestyle="--", color="black", alpha=0.7,
               label=f"teacher anchor ({teacher_sr:.3f})")
    ax.set_xscale("log")
    ax.set_xlabel(x_label)
    ax.set_ylabel("episode success rate (A pool, 500 ep/arm)")

    # The confirmatory verdict belongs to the primary family; a secondary
    # (descriptive) family never titles the figure.
    primary = next((f for f in fams if f.get("family_role") == "primary"), fams[0])
    v = primary.get("verdict", {})
    ax.set_title(
        f"{suite} — branch {v.get('branch','?')}: Q={v.get('q','?')} / D={v.get('d','?')}"
        f"\nplateau P={v.get('p','?')} (descriptive)",
        fontsize=10,
    )
    ax.set_xticks([])
    ax.set_xticks([], minor=True)
    # Point annotations replace the tick numbers, so give them breathing room:
    # without margins the outermost labels clip on the axes frame.
    ax.margins(x=0.10, y=0.12)
    n_rows = _entries_rows(ax, fam_colors)
    # The entries rows sit where tick labels would; the axis label goes below them.
    ax.xaxis.labelpad = 12 + 26 * n_rows
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)

    if latency_series:
        ax2 = axes[1]
        # Same annotation contract as the SR panel: the axis carries no tick
        # numbers, so every point states its own trajectory count alongside the
        # measured value; alternate above/below per series so two close curves
        # stay readable at the small end of the axis.
        for i, (label, pts) in enumerate(latency_series.items()):
            xs2 = [xi for xi, _ in pts]
            ys2 = [yi for _, yi in pts]
            line2, = ax2.plot(xs2, ys2, marker="s", label=label)
            dy = 10 if i % 2 == 0 else -22
            for xi, yi in pts:
                ax2.annotate(f"{xi:g} traj/task\n{yi:.1f} ms", (xi, yi),
                             textcoords="offset points", xytext=(0, dy),
                             ha="center", fontsize=7, color=line2.get_color())
        ax2.set_xscale("log")
        ax2.set_xlabel(x_label)
        ax2.set_ylabel("retrieval latency per step (ms, median)")
        ax2.set_title("cost of the same axis", fontsize=10)
        ax2.set_xticks([])
        ax2.set_xticks([], minor=True)
        ax2.margins(x=0.10, y=0.18)
        # The latency labels attach to one family's points (the primary); its
        # entries row is the one that names each x position's library size.
        lat_fam = next((f for f in fams if any("retrieval_latency_ms" in p
                                               for p in f["points"])), fams[0])
        n_rows2 = _entries_rows(ax2, [(lat_fam, "dimgray")])
        ax2.xaxis.labelpad = 12 + 26 * n_rows2
        ax2.legend(fontsize=8, loc="upper left")
        ax2.grid(alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # bbox_inches="tight": the entries rows live below the axes frame and would
    # otherwise be clipped by the figure boundary.
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    # Record which figure(s) each family feeds, so the data file states its own
    # consumers. Names only -- content provenance stays with `source.sha256`.
    for fam in fams:
        figs = fam.setdefault("figures", [])
        if out_path.name not in figs:
            figs.append(out_path.name)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True,
                    help="plot_data.json from emit_plot_data.py -- the sole data source")
    ap.add_argument("--family", required=True, action="append",
                    help="family key, e.g. libero_10/all (repeatable: families of "
                         "one suite overlay in a single figure)")
    ap.add_argument("--latency-label", action="append", default=None,
                    help="render the latency panel from this per-point label "
                         "(repeatable: one curve per label)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    data_path = pathlib.Path(args.data)
    data = json.loads(data_path.read_text())
    plot(
        data,
        families=args.family,
        latency_labels=args.latency_label,
        out_path=pathlib.Path(args.out),
    )
    # plot() appended the figure name to the family block; persist it.
    data_path.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")
    print(args.out)


if __name__ == "__main__":
    main()
