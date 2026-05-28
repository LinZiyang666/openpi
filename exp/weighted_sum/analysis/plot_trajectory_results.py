"""Plot trajectory-search results on top of the weighted-sum best configs.

Merges the reused depth-1 baseline (per base config, from the weighted-sum
``all_results.csv``) with the trajectory depths (from this experiment's
summarized journal) and reproduces the old trajectory analysis lenses:

- **Figure 1** (``trajectory_results.png/pdf``): per-keybuilder line chart,
  success rate vs depth (d1 baseline + d3/d4/d5/d6), one line per base config,
  styled by Group-1 role (top1 / top2 / 2nd-worst) and top-10 membership.
- **Figure 2** (``trajectory_delta.png/pdf``): the rescue/strong stratification
  — mean Δ (best trajectory depth − d1) grouped by Group-1 role, plus the
  per-depth aggregate over all base configs.

Also prints a markdown SR table (rows = base config, cols = d1..d6, best depth,
Δ vs d1) to stdout.

Usage:
    uv run exp/weighted_sum/analysis/plot_trajectory_results.py \
        --results  exp/weighted_sum/data/trajectory/results.json \
        --baseline exp/weighted_sum/data/phase2/all_results.csv
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import re
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from exp.weighted_sum.emit_trajectory_yamls import select_base_configs  # noqa: E402

_DEPTH_RE = re.compile(r"__d(\d+)$")
_ROLE_STYLE = {  # role -> (matplotlib marker, linestyle)
    "top1": ("o", "-"),
    "top2": ("s", "-"),
    "2nd-worst": ("^", "--"),
    "top10-only": ("D", ":"),
}


def _baseline_sr(results_csv: Path) -> dict[str, float]:
    """Per base_id depth-1 SR = mean over all stages it appears in."""
    agg: dict[str, list[float]] = collections.defaultdict(list)
    for r in csv.DictReader(results_csv.open()):
        agg[r["yaml_id"]].append(float(r["success_rate"]))
    return {y: statistics.mean(v) for y, v in agg.items()}


def _trajectory_sr(results_json: Path) -> dict[str, dict[int, float]]:
    """Load summarized trajectory results -> {base_id: {depth: success_rate}}."""
    data = json.loads(results_json.read_text(encoding="utf-8"))
    items = data.items() if isinstance(data, dict) else ((r["yaml_id"], r) for r in data)
    out: dict[str, dict[int, float]] = collections.defaultdict(dict)
    for yaml_id, rec in items:
        m = _DEPTH_RE.search(yaml_id)
        if not m:
            continue
        depth = int(m.group(1))
        base_id = yaml_id[: m.start()]
        sr = rec["success_rate"] if isinstance(rec, dict) else rec
        out[base_id][depth] = float(sr)
    return out


def _role_map(group1, top10_ids: set[str]) -> dict[str, tuple[str, str]]:
    """base_id -> (keybuilder, role). top10-only configs get role 'top10-only'."""
    roles: dict[str, tuple[str, str]] = {}
    for kb, role, yaml_id, _sr in group1:
        roles[yaml_id] = (kb, role)
    for yaml_id in top10_ids:
        roles.setdefault(yaml_id, (yaml_id.split("__", 1)[0], "top10-only"))
    return roles


def main():
    repo = Path(__file__).resolve().parents[3]
    ap = argparse.ArgumentParser(description="Plot trajectory results over weighted-sum bases")
    ap.add_argument("--results", default=str(repo / "exp/weighted_sum/data/trajectory/results.json"))
    ap.add_argument("--baseline", default=str(repo / "exp/weighted_sum/data/phase2/all_results.csv"))
    ap.add_argument("--results-csv", default=str(repo / "exp/weighted_sum/data/phase2/all_results.csv"))
    ap.add_argument("--top10-dir", default=str(repo / "exp/weighted_sum/config/top10"))
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parent))
    args = ap.parse_args()

    baseline = _baseline_sr(Path(args.baseline))
    traj = _trajectory_sr(Path(args.results))
    if not traj:
        print("no trajectory results to plot (is results.json populated?)")
        return

    group1, top10_ids, _base_ids = select_base_configs(Path(args.results_csv), Path(args.top10_dir))
    roles = _role_map(group1, top10_ids)

    depths = sorted({d for dd in traj.values() for d in dd})
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Stdout table ──
    print(f"{'base config':62} {'d1':>5} " + " ".join(f"{'d'+str(d):>5}" for d in depths) + f" {'best':>5} {'Δ':>6}")
    deltas_by_role: dict[str, list[float]] = collections.defaultdict(list)
    per_depth: dict[int, list[float]] = collections.defaultdict(list)
    for base_id in sorted(traj):
        kb, role = roles.get(base_id, (base_id.split("__", 1)[0], "?"))
        d1 = baseline.get(base_id)
        srs = traj[base_id]
        best_depth = max(srs, key=lambda d: srs[d])
        delta = (srs[best_depth] - d1) * 100 if d1 is not None else float("nan")
        deltas_by_role[role].append(delta)
        for d, v in srs.items():
            per_depth[d].append(v * 100)
        d1s = f"{d1 * 100:4.0f}" if d1 is not None else "  - "
        cells = " ".join(f"{srs[d] * 100:4.0f}%" if d in srs else "   - " for d in depths)
        print(f"{base_id:62} {d1s}% {cells} {'d' + str(best_depth):>5} {delta:+5.1f}")

    print("\n--- mean Δ (best traj depth − d1) by role ---")
    for role in ("2nd-worst", "top2", "top1", "top10-only"):
        if deltas_by_role[role]:
            print(f"  {role:11}: n={len(deltas_by_role[role]):2}  mean Δ = {statistics.mean(deltas_by_role[role]):+5.1f} pp")
    print("\n--- per-depth aggregate (mean SR over all bases) ---")
    print(f"  d1 (baseline): {statistics.mean([baseline[b] * 100 for b in traj if b in baseline]):.1f}%")
    for d in depths:
        print(f"  d{d}: {statistics.mean(per_depth[d]):.1f}%")

    # ── Figure 1: per-keybuilder SR vs depth ──
    kbs = sorted({kb for kb, _ in roles.values()})
    fig, axes = plt.subplots(len(kbs), 1, figsize=(9, 3 * len(kbs)), squeeze=False)
    x = [1] + depths
    for ax, kb in zip(axes[:, 0], kbs):
        for base_id in sorted(b for b in traj if roles.get(b, ("", ""))[0] == kb):
            _, role = roles[base_id]
            marker, ls = _ROLE_STYLE.get(role, ("x", "-"))
            d1 = baseline.get(base_id)
            ys = [d1 * 100 if d1 is not None else None] + [traj[base_id].get(d, None) and traj[base_id][d] * 100 for d in depths]
            label = f"{role} · {base_id.split('__', 1)[1][:28]}"
            ax.plot(x, ys, marker=marker, ls=ls, label=label, markersize=5)
        ax.set_title(kb)
        ax.set_xlabel("trajectory depth (1 = weighted-sum baseline)")
        ax.set_ylabel("success rate (%)")
        ax.set_xticks(x)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=6, ncol=2)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"trajectory_results.{ext}", dpi=200)
    plt.close(fig)

    # ── Figure 2: rescue/strong stratification ──
    fig2, (axa, axb) = plt.subplots(1, 2, figsize=(12, 4))
    roles_order = [r for r in ("2nd-worst", "top2", "top1", "top10-only") if deltas_by_role[r]]
    axa.bar(roles_order, [statistics.mean(deltas_by_role[r]) for r in roles_order])
    axa.axhline(0, color="k", lw=0.8)
    axa.set_ylabel("mean Δ (best traj − d1), pp")
    axa.set_title("Trajectory effect by base role (rescue vs strong)")
    d1_agg = statistics.mean([baseline[b] * 100 for b in traj if b in baseline])
    axb.plot([1] + depths, [d1_agg] + [statistics.mean(per_depth[d]) for d in depths], marker="o")
    axb.set_xlabel("trajectory depth")
    axb.set_ylabel("mean SR over all bases (%)")
    axb.set_title("Per-depth aggregate")
    axb.set_xticks([1] + depths)
    axb.grid(True, alpha=0.3)
    fig2.tight_layout()
    for ext in ("png", "pdf"):
        fig2.savefig(out_dir / f"trajectory_delta.{ext}", dpi=200)
    plt.close(fig2)

    print(f"\nsaved figures to {out_dir}")


if __name__ == "__main__":
    main()
