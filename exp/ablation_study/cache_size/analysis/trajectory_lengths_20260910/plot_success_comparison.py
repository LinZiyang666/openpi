"""Compare S6 trajectory lengths before/after filtering by recorded success.

Lengths come from the previously extracted all-S6 PKL metadata. Outcome labels
come from the frozen success-S6 episode lists, which select HDF5 success attrs.
The selected episode and entry counts are checked against build manifests.
"""
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np

ROOT = Path(__file__).resolve().parent
CONFIG = ROOT.parent.parent / "config"
records = [json.loads(line) for line in (ROOT / "metadata.jsonl").read_text().splitlines()]
records.sort(key=lambda r: r["suite"] != "libero_spatial")
names = {"libero_spatial": "LIBERO Spatial", "libero_10": "LIBERO-10"}
colors = {"libero_spatial": "#16847d", "libero_10": "#4168b1"}


def episode_ids(path):
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    ids = [str(Path(line).with_suffix("")) for line in lines]
    assert len(set(ids)) == len(ids), f"Duplicate episodes: {path}"
    return set(ids)


def stats(values):
    return {
        "trajectories": int(len(values)),
        "total_entries": int(sum(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "min": int(min(values)),
        "max": int(max(values)),
    }


summary = {}
csv_rows = []
series = {}
for record in records:
    suite = record["suite"]
    rows = record["rows"]
    ids = {row["trajectory_id"] for row in rows}
    all_list = CONFIG / f"lists_all/episodes_{suite}_S6.txt"
    success_list = CONFIG / f"lists_success/episodes_{suite}_S6.txt"
    successes = episode_ids(success_list)
    assert len(ids) == len(rows) == 450
    assert ids == episode_ids(all_list)
    assert successes <= ids
    all_vals = np.array([row["length"] for row in rows])
    yes = np.array([row["length"] for row in rows if row["trajectory_id"] in successes])
    no = np.array([row["length"] for row in rows if row["trajectory_id"] not in successes])
    manifest = json.loads((CONFIG / f"entries_{suite}_success.json").read_text())
    s6 = next(tier for tier in manifest["tiers"] if tier["tier"] == "S6")
    assert len(yes) == s6["episodes"]
    assert int(yes.sum()) == s6["entries"]
    assert int(all_vals.sum()) == record["entries"]
    summary[suite] = {
        "source_pkl": record["path"],
        "success_episode_list": str(success_list),
        "length_unit": "cached policy calls",
        "all": stats(all_vals),
        "success": stats(yes),
        "failure": stats(no),
        "failure_length_counts": {int(v): int((no == v).sum()) for v in np.unique(no)},
    }
    series[suite] = (all_vals, yes)
    for row in rows:
        csv_rows.append({
            "suite": suite,
            "task_id": row["task_id"],
            "trajectory_id": row["trajectory_id"],
            "length": row["length"],
            "outcome": "success" if row["trajectory_id"] in successes else "failure",
        })

(ROOT / "success_comparison_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
with (ROOT / "trajectory_lengths_with_outcomes.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(csv_rows[0]))
    writer.writeheader()
    writer.writerows(csv_rows)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 15,
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#cad0d7",
    "xtick.color": "#425466",
    "ytick.color": "#425466",
    "text.color": "#233247",
    "axes.labelcolor": "#233247",
    "pdf.fonttype": 42,
})
fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.8))
for ax, record in zip(axes, records):
    suite = record["suite"]
    all_vals, yes = series[suite]
    s = summary[suite]
    width = 2 if suite == "libero_spatial" else 5
    bins = np.arange((all_vals.min() // width) * width - 0.5,
                     ((all_vals.max() // width) + 2) * width - 0.5, width)
    before, _ = np.histogram(all_vals, bins=bins)
    after, _ = np.histogram(yes, bins=bins)
    assert before.sum() == len(all_vals)
    assert after.sum() == len(yes)
    assert np.all(after <= before)
    ax.stairs(before, bins, fill=True, color="#e5e7eb", alpha=0.9, linewidth=0)
    ax.bar(bins[:-1], after, width=np.diff(bins), align="edge", color=colors[suite],
           alpha=0.88, edgecolor="white", linewidth=1.0)
    ax.stairs(before, bins, color="#6b7280", linestyle="--", linewidth=1.5, zorder=3)
    ax.set_title(names[suite], loc="left", fontweight="bold", pad=15)
    ax.set_xlabel("Trajectory length (cached policy calls)", labelpad=10)
    ax.set_ylabel("Number of trajectories", labelpad=9)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylim(0, max(before) * 1.52)
    ax.grid(axis="y", alpha=0.15)
    ax.set_axisbelow(True)
    ax.legend(handles=[
        Line2D([0], [0], color="#6b7280", linestyle="--", linewidth=1.5,
               label=f"Before: all {len(all_vals)} trajectories"),
        Patch(facecolor=colors[suite], alpha=0.88,
              label=f"After: {len(yes)} successful trajectories"),
    ], loc="upper left", frameon=False, fontsize=9.5, handlelength=2.3)
    ax.text(0.98, 0.96,
            f"After filtering\nMedian {s['success']['median']:g}  |  Mean {s['success']['mean']:.1f}\n"
            f"Range {s['success']['min']}–{s['success']['max']}",
            transform=ax.transAxes, ha="right", va="top", fontsize=10,
            linespacing=1.65,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.92, "pad": 3})
    failure_x = s["failure"]["max"]
    bin_index = np.searchsorted(bins, failure_x, side="right") - 1
    ax.annotate(f"{s['failure']['trajectories']} failures removed",
                xy=(failure_x, before[bin_index]), xycoords="data",
                xytext=(-8, 22), textcoords="offset points", ha="right", va="bottom",
                fontsize=9.5, color="#a15238",
                arrowprops={"arrowstyle": "->", "color": "#a15238", "lw": 1.1})
    ax.text(0, -0.28,
            f"Mean: {s['all']['mean']:.1f} → {s['success']['mean']:.1f} calls   ·   "
            f"Retained: {len(yes) / len(all_vals):.1%}   ·   Bin width: {width}",
            transform=ax.transAxes, fontsize=10, color="#64748b")

fig.suptitle("Trajectory lengths before and after removing failures",
             fontsize=18, fontweight="bold", x=0.065, ha="left", y=0.98)
fig.text(0.065, 0.905,
         "Largest S6 cache libraries  ·  Failures excluded using recorded success labels",
         fontsize=11, color="#64748b")
fig.text(0.065, 0.035,
         "Dashed outline: original distribution. Colored bars: successful trajectories only.\n"
         "Lengths are stored policy decisions, not individual environment actions; filtering uses outcomes, not a length cutoff.",
         fontsize=9, color="#64748b", linespacing=1.55)
fig.subplots_adjust(left=0.065, right=0.985, bottom=0.28, top=0.80, wspace=0.25)
fig.savefig(ROOT / "trajectory_length_success_comparison.png", dpi=190, facecolor="white")
fig.savefig(ROOT / "trajectory_length_success_comparison.pdf", facecolor="white")
print(json.dumps(summary, indent=2))
