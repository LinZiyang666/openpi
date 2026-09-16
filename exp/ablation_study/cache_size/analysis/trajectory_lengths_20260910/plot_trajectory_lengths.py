"""Plot trajectory lengths read directly from cache-size S6 artifacts."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

ROOT = Path(__file__).resolve().parent
records = [json.loads(line) for line in (ROOT / "metadata.jsonl").read_text().splitlines()]
assert {r["suite"] for r in records} == {"libero_spatial", "libero_10"}
records.sort(key=lambda r: r["suite"] != "libero_spatial")
names = {"libero_spatial": "LIBERO Spatial", "libero_10": "LIBERO-10"}
colors = {"libero_spatial": "#16847d", "libero_10": "#4168b1"}
summary = {}
all_rows = []
for record in records:
    rows = record["rows"]
    assert len(rows) == record["n_trajectories"] == 450
    assert sum(row["length"] for row in rows) == record["entries"]
    assert all(sum(row["task_id"] == t for row in rows) == 45 for t in range(10))
    for row in rows:
        assert row["first_step"] == 0
        assert row["last_step"] + 1 == row["length"]
        assert row["step_increments"] == [1]
        all_rows.append({"suite": record["suite"], **row})
    lengths = np.array([r["length"] for r in rows])
    summary[record["suite"]] = {
        "source_pkl": record["path"],
        "source_bytes": record["file_bytes"],
        "source_mtime_ns": record["mtime_ns"],
        "length_unit": "cached policy calls (one cache entry per recorded decision)",
        "trajectories": len(rows),
        "tasks": 10,
        "trajectories_per_task": 45,
        "total_entries": int(lengths.sum()),
        "min": int(lengths.min()),
        "q25": float(np.percentile(lengths, 25)),
        "median": float(np.median(lengths)),
        "q75": float(np.percentile(lengths, 75)),
        "mean": float(lengths.mean()),
        "p90": float(np.percentile(lengths, 90)),
        "p95": float(np.percentile(lengths, 95)),
        "max": int(lengths.max()),
        "step_indices_contiguous": True,
    }
(ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
with (ROOT / "trajectory_lengths.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["suite", "task_id", "trajectory_id", "length", "first_step", "last_step", "task"])
    writer.writeheader()
    for row in all_rows:
        writer.writerow({k: row[k] for k in ["suite", "task_id", "trajectory_id", "length", "first_step", "last_step"]} | {"task": " | ".join(row["task_keys"])})

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
fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.5))
for ax, record in zip(axes, records):
    suite = record["suite"]
    vals = np.array([r["length"] for r in record["rows"]])
    stats = summary[suite]
    width = 2 if suite == "libero_spatial" else 5
    bins = np.arange((vals.min() // width) * width - 0.5, ((vals.max() // width) + 2) * width - 0.5, width)
    counts, _, _ = ax.hist(vals, bins=bins, color=colors[suite], alpha=0.84, edgecolor="white", linewidth=1.1)
    assert int(counts.sum()) == len(vals)
    ax.axvline(stats["median"], color="#253449", linewidth=1.7, linestyle="--", label="Median", zorder=3)
    ax.axvline(stats["mean"], color="#c77328", linewidth=1.7, linestyle=":", label="Mean", zorder=3)
    ax.set_title(names[suite], loc="left", fontweight="bold", pad=13)
    ax.set_xlabel("Trajectory length (cached policy calls)", labelpad=10)
    ax.set_ylabel("Number of trajectories", labelpad=9)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(axis="y", alpha=0.17)
    ax.set_axisbelow(True)
    ax.set_ylim(0, max(counts) * 1.28)
    ax.text(0.97, 0.95,
            f"n = {stats['trajectories']}\nMedian  {stats['median']:g}\nMean  {stats['mean']:.1f}\nP90  {stats['p90']:g}\nRange  {stats['min']}–{stats['max']}",
            transform=ax.transAxes, ha="right", va="top", fontsize=10.5, linespacing=1.65,
            bbox={"facecolor":"white", "alpha":0.92, "edgecolor":"none", "pad":5})
    ax.text(0.01, 0.98, f"Bin width: {width} calls", transform=ax.transAxes, va="top", fontsize=9, color="#64748b")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.092), frameon=False, ncol=2, fontsize=10)
fig.suptitle("Trajectory lengths in the largest cache-size libraries", fontsize=18, fontweight="bold", x=0.065, ha="left", y=0.985)
fig.text(0.065, 0.905, "S6 / all trajectories  ·  45 trajectories per task  ·  10 tasks per suite", fontsize=11, color="#64748b")
fig.text(0.065, 0.012,
         "Source: cache_size_{suite}_all_S6.pkl. Length counts stored decisions, not individual environment actions.\n"
         "Both successful and failed source trajectories are included; step indices are contiguous in every trajectory.",
         fontsize=9, color="#64748b", linespacing=1.55)
fig.subplots_adjust(left=0.065, right=0.985, bottom=0.235, top=0.815, wspace=0.25)
fig.savefig(ROOT / "trajectory_length_distribution.png", dpi=190, facecolor="white")
fig.savefig(ROOT / "trajectory_length_distribution.pdf", facecolor="white")
print(json.dumps(summary, indent=2))
