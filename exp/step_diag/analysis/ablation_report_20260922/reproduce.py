"""Recompute the report's tables and figures from local launch manifests and journals.

Run from the repository root with `.venv/bin/python -B <this-file>`.
Only this report directory is written. No simulator, model, or Git command is run.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[3]
DATA = REPO / "exp/step_diag/data"
os.environ.setdefault("MPLCONFIGDIR", str(OUT / ".mplconfig"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

ARMS = ["full", "plain_k2", "warm_t0.2", "warmreset_t0.2", "resetfinal_t0.2", "warmshoot_t0.2"]
LABELS = ["Full (10)", "Plain (2)", "Warm resume (2)", "Warm reset (2)", "Reset final (2)", "Warm shoot (2)"]
COLORS = ["#64748b", "#3979b7", "#c08134", "#158a80", "#8661b0", "#c45560"]
TWO = ["CloseFridge", "PickPlaceCounterToStove"]
KEYS = ("task", "init_idx", "env_seed", "lane", "pin_id", "layout", "style")
BOOT, BOOT_SEED = 20000, 20260922
sources: dict[str, str] = {}
cache: dict[tuple, dict] = {}


def read_text(path: Path) -> str:
    """Read and record the exact input bytes used in this report."""
    raw = path.read_bytes()
    rel = str(path.relative_to(REPO))
    digest = hashlib.sha256(raw).hexdigest()
    if rel in sources:
        assert sources[rel] == digest, f"Input changed while reading: {rel}"
    sources[rel] = digest
    return raw.decode("utf-8")


def load(root: str, arm: str) -> dict:
    """Join terminal outcomes to their expected environment identities, without overwrites."""
    key = (root, arm)
    if key in cache:
        return cache[key]
    folder = DATA / root / "pi05" / arm
    expected, terminal = {}, {}
    for path in sorted(folder.glob("launch_*.json")):
        launch = json.loads(read_text(path))
        for ident in launch["expected"]:
            uid = ident["task_uid"]
            assert uid not in expected or expected[uid] == ident
            expected[uid] = ident
    for path in sorted(folder.glob("journal_*.jsonl")):
        for line_no, line in enumerate(read_text(path).splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("accepted") is not True or row.get("status") not in ("done", "failed"):
                continue
            uid = row["task_uid"]
            assert uid not in terminal, f"Duplicate accepted terminal: {uid}"
            assert uid in expected, f"Unexpected terminal: {uid}"
            assert isinstance(row.get("success"), bool), row
            terminal[uid] = {**expected[uid], "success": int(row["success"]),
                             "root": root, "arm": arm, "journal": str(path.relative_to(REPO)),
                             "journal_line": line_no, "status": row["status"]}
    assert set(expected) == set(terminal), f"Missing terminal outcomes in {folder}"
    result = {}
    for row in terminal.values():
        task = row["task"]
        identity = tuple(row.get(k) for k in KEYS)
        bucket = result.setdefault(task, {})
        assert identity not in bucket, f"Repeated environment in one root: {identity}"
        bucket[identity] = row
    cache[key] = result
    return result


def cell(root: str, arm: str, task: str, lo: int = 0, hi: int = 50) -> list[dict]:
    """Select a fixed interval of initialization indices, independently of outcomes."""
    rows = [v for v in load(root, arm).get(task, {}).values() if lo <= v["init_idx"] < hi]
    rows.sort(key=lambda v: v["init_idx"])
    assert len(rows) == hi - lo, (root, arm, task, lo, hi, len(rows))
    assert [v["init_idx"] for v in rows] == list(range(lo, hi))
    expected_base = 1000000 if root == "rc_x1m" else 2000000
    assert all(v["env_seed"] == expected_base + v["init_idx"] for v in rows)
    return rows


def choose_root(arm: str, task: str) -> str:
    """Use the complete macro run for reset/shoot; otherwise use disjoint task batches."""
    if arm in ("warmreset_t0.2", "warmshoot_t0.2"):
        return "rc_macro13"
    return "rc" if task in load("rc", arm) else "rc_macro13"


def sr(rows: list[dict]) -> float:
    """Return the binary success fraction."""
    return float(np.mean([r["success"] for r in rows]))


def wilson(p: float, n: int) -> tuple[float, float]:
    """Return a two-sided 95% Wilson interval for a single binomial cell."""
    z = 1.959963984540054
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return float(max(0.0, min(p, center - half))), float(min(1.0, max(p, center + half)))


def save(fig, name: str) -> None:
    """Save a shareable PNG and an editable vector SVG."""
    for ext in ("png", "svg"):
        fig.savefig(OUT / "figures" / f"{name}.{ext}", dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    """Build provenance, aligned analysis tables, and all report figures."""
    (OUT / "figures").mkdir(exist_ok=True)
    admission = json.loads(read_text(DATA / "analysis/warm_variants_pi05_macro13.json"))
    tasks = sorted(admission["tasks"])
    assert len(tasks) == 13
    for task in tasks:
        record = admission["tasks"][task]
        assert not record["comparison_problems"]
        for arm in ARMS:
            c = record["cells"][arm]
            assert c["complete"] and c["equal_nfe"] and not c["problems"]
            assert c["mean_executed_steps"] == (10 if arm == "full" else 2)

    selected, raw = [], np.zeros((len(tasks), 50, len(ARMS)), dtype=np.int8)
    task_table = []
    for ti, task in enumerate(tasks):
        identities = None
        out_row = {"task": task}
        for ai, arm in enumerate(ARMS):
            rows = cell(choose_root(arm, task), arm, task)
            ids = [tuple(r.get(k) for k in KEYS) for r in rows]
            assert identities is None or identities == ids, f"Unpaired environments: {task}/{arm}"
            identities = ids
            selected.extend(rows)
            raw[ti, :, ai] = [r["success"] for r in rows]
            out_row[arm] = sr(rows)
        task_table.append(out_row)
    assert raw.shape == (13, 50, 6)
    primary = {a: {"successes": int(raw[:, :, i].sum()), "n": 650,
                   "sr": float(raw[:, :, i].mean())} for i, a in enumerate(ARMS)}
    rng = np.random.default_rng(BOOT_SEED)
    draws = np.zeros((BOOT, len(ARMS)))
    for block in raw:
        idx = rng.integers(0, 50, size=(BOOT, 50))
        draws += block[idx].mean(axis=1) / 13
    for i, arm in enumerate(ARMS):
        primary[arm]["ci95"] = np.percentile(draws[:, i], [2.5, 97.5]).tolist()
    contrasts = [("warmreset_t0.2", "full"), ("warmreset_t0.2", "plain_k2"),
                 ("warmreset_t0.2", "warm_t0.2"), ("resetfinal_t0.2", "warmreset_t0.2"),
                 ("resetfinal_t0.2", "full"), ("plain_k2", "full")]
    differences = []
    for a, b in contrasts:
        ai, bi = ARMS.index(a), ARMS.index(b)
        difference = raw[:, :, ai] - raw[:, :, bi]
        differences.append({"a": a, "b": b, "delta": float(difference.mean()),
                            "ci95": np.percentile(draws[:, ai] - draws[:, bi], [2.5, 97.5]).tolist(),
                            "a_only_success": int((difference == 1).sum()),
                            "b_only_success": int((difference == -1).sum())})

    # Reconstruct the current repository's all-data, identity-balanced estimand independently.
    all_data = {}
    for arm in ARMS:
        task_rates = []
        for task in tasks:
            grouped = {}
            for root in ("rc", "rc_macro13"):
                for identity, row in load(root, arm).get(task, {}).items():
                    grouped.setdefault(identity, []).append(row["success"])
            task_rates.append(float(np.mean([np.mean(v) for v in grouped.values()])))
        estimate = float(np.mean(task_rates))
        stored = admission["macro"]["sr"][arm]["macro_sr"]
        assert abs(estimate - stored) < 1e-12, (arm, estimate, stored)
        all_data[arm] = estimate

    ladder = {}
    for task in TWO:
        ladder[task] = {"full": sr(cell("rc", "full", task))}
        for family in ("plain", "warm", "warmreset", "resetfinal"):
            ladder[task][family] = []
            for n in (1, 2, 3):
                arm = f"plain_k{n}" if family == "plain" else f"{family}_t0.{n}"
                ladder[task][family].append(sr(cell("rc", arm, task)))
    robust = {}
    robust_arms = ["plain_k2", "warm_t0.2", "warmreset_t0.2", "warmshoot_t0.2"]
    for task in TWO:
        robust[task] = {}
        for arm in robust_arms:
            robust[task][arm] = {"2m_50": sr(cell("rc", arm, task)),
                                 "1m_50": sr(cell("rc_x1m", arm, task)),
                                 "2m_500": sr(cell("rc500", arm, task, 0, 500)),
                                 "2m_idx50_499": sr(cell("rc500", arm, task, 50, 500)),
                                 "2m_idx100_499": sr(cell("rc500", arm, task, 100, 500))}
    eight_tasks = sorted(set(load("rc", "resetfinal_t0.1")) | set(load("rc_macro13", "resetfinal_t0.1")))
    assert len(eight_tasks) == 8
    eight = {"tasks": eight_tasks, "full": float(np.mean([primary_task["full"] for primary_task in task_table
                                                          if primary_task["task"] in eight_tasks]))}
    for n in (1, 2, 3):
        arm = f"resetfinal_t0.{n}"
        eight[arm] = float(np.mean([sr(cell(choose_root(arm, t), arm, t)) for t in eight_tasks]))

    with (OUT / "primary_episodes.csv").open("w", newline="") as f:
        fields = ["arm", "task", "init_idx", "env_seed", "lane", "pin_id", "layout", "style", "success",
                  "root", "task_uid", "journal", "journal_line"]
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(selected)
    with (OUT / "task_success_rates.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["task", *ARMS])
        writer.writeheader()
        writer.writerows(task_table)
    stats = {"scope": "pi0.5 / RoboCasa / fixed 13 tasks / layout 1 style 1",
             "selection": "init_idx 0..49; macro13-only warmreset/warmshoot; other arms use disjoint task batches",
             "bootstrap": {"draws": BOOT, "seed": BOOT_SEED, "method": "paired within-task resampling; fixed tasks; percentile 95%; unadjusted exploratory"},
             "arms": ARMS, "primary": primary, "contrasts": differences, "tasks": task_table,
             "all_data_identity_balanced": all_data, "ladder": ladder, "robustness": robust,
             "matched_eight_tasks": eight,
             "validation": {"primary_episode_rows": len(selected), "primary_missing": 0,
                            "primary_duplicate_identities_per_arm": 0, "cross_arm_pairing": "exact",
                            "stored_admission_artifact_checked": True,
                            "server_arrays_rehashed_in_report_generation": False}}
    (OUT / "statistics.json").write_text(json.dumps(stats, indent=2) + "\n")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.labelcolor": "#334155", "text.color": "#16283a",
                         "axes.titleweight": "bold", "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [1, 1.25]})
    values = [primary[a]["sr"] * 100 for a in ARMS]
    lows = np.array([primary[a]["ci95"][0] * 100 for a in ARMS])
    highs = np.array([primary[a]["ci95"][1] * 100 for a in ARMS])
    axes[0].barh(range(6), values, color=COLORS, height=.64)
    axes[0].errorbar(values, range(6), xerr=[np.array(values)-lows, highs-np.array(values)],
                     fmt="none", color="#23374d", capsize=3, lw=1)
    for i, v in enumerate(values):
        axes[0].text(max(highs[i], v) + 1.4, i, f"{v:.1f}%", va="center", fontsize=10)
    axes[0].set(yticks=range(6), yticklabels=LABELS, xlim=(0, 89), xlabel="Success rate (%)",
                title="A  |  Same 13 tasks, 50 episodes per arm/task")
    axes[0].invert_yaxis()
    axes[0].grid(axis="x", alpha=.15)
    labels = ["Warm reset - Full", "Warm reset - Plain", "Warm reset - Warm resume",
              "Reset final - Warm reset", "Reset final - Full", "Plain - Full"]
    for i, d in enumerate(differences):
        v, (lo, hi) = d["delta"]*100, np.array(d["ci95"])*100
        color = "#158a80" if v > 0 else "#8661b0" if abs(v) < 1e-12 else "#64748b"
        axes[1].errorbar(v, i, xerr=[[v-lo], [hi-v]], fmt="o", capsize=4, color=color, lw=2)
        axes[1].text(50, i, f"{v:+.1f} [{lo:+.1f}, {hi:+.1f}]", va="center", fontsize=9)
    axes[1].axvline(0, color="#94a3b8", lw=1, ls="--")
    axes[1].set(yticks=range(6), yticklabels=labels, xlim=(-14, 78),
                xlabel="Success-rate difference (percentage points)", title="B  |  Paired differences and 95% intervals")
    axes[1].invert_yaxis()
    axes[1].grid(axis="x", alpha=.15)
    fig.text(.5, -.005, "20,000 paired bootstrap draws within each fixed task; exploratory intervals, no multiplicity correction.",
             ha="center", fontsize=9, color="#64748b")
    fig.tight_layout(w_pad=3)
    save(fig, "01_main_comparison")

    matrix = raw.mean(axis=1) * 100
    heat_labels = [t.replace("PickPlace", "PickPlace\n") for t in tasks]
    fig, ax = plt.subplots(figsize=(11.5, 8.8))
    cmap = LinearSegmentedColormap.from_list("success", ["#f2f5f9", "#a9d9d0", "#107d74"])
    im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=100, aspect="auto")
    for r in range(13):
        for c in range(6):
            ax.text(c, r, f"{matrix[r,c]:.0f}%", ha="center", va="center",
                    color="white" if matrix[r,c] >= 70 else "#182f40", fontsize=11)
    ax.set(xticks=range(6), xticklabels=["Full\n10", "Plain\n2", "Warm resume\n2", "Warm reset\n2",
                                       "Reset final\n2", "Warm shoot\n2"], yticks=range(13),
           yticklabels=heat_labels, title="Task-level success rates | matched 50-episode cohort")
    ax.tick_params(length=0, pad=10)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.colorbar(im, ax=ax, shrink=.65, pad=.025, label="Success rate (%)")
    fig.tight_layout()
    save(fig, "02_task_heatmap")

    family_colors = {"plain": COLORS[1], "warm": COLORS[2], "warmreset": COLORS[3], "resetfinal": COLORS[4]}
    family_names = {"plain": "Plain", "warm": "Warm resume", "warmreset": "Warm reset", "resetfinal": "Reset final"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, task in zip(axes, TWO):
        for family, color in family_colors.items():
            ax.plot([1,2,3], np.array(ladder[task][family])*100, "o-", lw=2, color=color,
                    label=family_names[family], markersize=6)
        ax.axhline(ladder[task]["full"]*100, ls="--", color=COLORS[0], label="Full (10 steps)")
        ax.set(title=task, xticks=[1,2,3], ylim=(-3,105), xlabel="Denoising evaluations per decision")
        ax.grid(alpha=.18)
    axes[0].set_ylabel("Success rate (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, bbox_to_anchor=(.5, -.025), frameon=False)
    fig.suptitle("Step-budget ablation | original two-task runs, n = 50 per cell", y=1.02, fontweight="bold")
    fig.tight_layout(rect=(0,.06,1,1))
    save(fig, "03_step_ladder")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, task in zip(axes, TWO):
        for i, arm in enumerate(robust_arms):
            p = robust[task][arm]["2m_500"]
            lo, hi = wilson(p, 500)
            color = COLORS[ARMS.index(arm)]
            ax.bar(i, p*100, color=color, width=.65)
            ax.errorbar(i, p*100, yerr=[[(p-lo)*100],[(hi-p)*100]], fmt="none", color="#23374d", capsize=4)
            ax.text(i, hi*100+2, f"{p*100:.1f}%", ha="center", fontsize=10)
        ax.axhline(ladder[task]["full"]*100, ls="--", color=COLORS[0])
        ax.set(title=task, xticks=range(4), xticklabels=["Plain", "Warm\nresume", "Warm\nreset", "Warm\nshoot"],
               ylim=(0,108))
        ax.grid(axis="y", alpha=.15)
    axes[0].set_ylabel("Success rate (%)")
    fig.suptitle("Larger-sample follow-up | 500 episodes per arm/task", y=1.02, fontweight="bold")
    fig.text(.5, -.015, "Error bars: 95% Wilson intervals. Dashed line: original Full reference (n = 50, not 500).",
             ha="center", color="#64748b", fontsize=9)
    fig.tight_layout()
    save(fig, "04_500_episode_followup")

    fig, ax = plt.subplots(figsize=(12.5, 5.8))
    schedules = [[1-i/10 for i in range(10)], [1,.5], [.2,.1], [1,.5], [1,.5], [.2,-.3]]
    endpoints = [0,0,0,0,0,-.8]
    starts = ["noise", "noise", "cached x(0.2)", "cached x(0.2)", "cached x(0)", "cached x(0.2)"]
    for i, (times, end, color) in enumerate(zip(schedules, endpoints, COLORS)):
        ax.annotate("", xy=(end,i), xytext=(times[0],i), arrowprops={"arrowstyle":"->", "color":color,"lw":2})
        ax.scatter(times, [i]*len(times), color=color, s=58, zorder=4)
        ax.scatter([end], [i], color="white", edgecolor=color, s=45, zorder=5)
        ax.text(1.13, i, starts[i], va="center", fontsize=10, color=color)
    ax.axvline(0, ls="--", color="#94a3b8")
    ax.axvspan(-.9, 0, color="#f8e7e8", alpha=.6)
    ax.set(yticks=range(6), yticklabels=LABELS, xticks=[-.8,-.3,0,.1,.2,.5,1], xlim=(-.9,1.7),
           xlabel="Flow time t (decreases during inference)", title="What each arm changes: start tensor and time schedule")
    ax.invert_yaxis()
    ax.text(-.82, -.5, "negative time", color="#a64753", fontsize=9)
    fig.text(.5, -.015, "Filled dots = model evaluations. Open circles = final time after updates. Resetting t does not replace the cached tensor.",
             ha="center", color="#64748b", fontsize=9)
    fig.tight_layout()
    save(fig, "00_inference_schedules")

    for rel in ("exp/step_diag/pi05.py", "exp/step_diag/serve_diag_pi05.py", "exp/step_diag/envs.py",
                "exp/step_diag/analysis/warm_variants.py", "exp/step_diag/analysis/aggregate_arms.py",
                "exp/step_diag/analysis/step_vs_warmstart.md", "exp/step_diag/config/arms/pi05_rc/warm_t0.2.yaml",
                "src/openpi/models_pytorch/pi0_pytorch.py", "exp/robocasa365/episode_runner.py"):
        read_text(REPO / rel)
    # Detect any concurrent edits to files already consumed.
    for rel, digest in sources.items():
        assert hashlib.sha256((REPO / rel).read_bytes()).hexdigest() == digest, f"Input changed: {rel}"
    (OUT / "source_manifest.json").write_text(json.dumps({"repository": str(REPO),
        "note": "Working-tree inputs, no Git commands. Hashes describe this report's read snapshot.",
        "files": sources}, indent=2) + "\n")
    print(json.dumps({"primary": primary, "contrasts": differences, "ladder": ladder,
                      "robustness": robust, "matched_eight_tasks": eight,
                      "input_files": len(sources), "selected_rows": len(selected)}, indent=2))


if __name__ == "__main__":
    main()
