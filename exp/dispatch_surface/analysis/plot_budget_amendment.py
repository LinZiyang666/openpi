"""Figures for the budget-mixture amendment result (confirmation plan 3.1 / 7-4).

Reads ``budget_outcome_design.json`` (+ ``budget_cost_map.json``) and draws:

  1. ``family_frontiers``  measured (cost, SR) points of every arm and the
     budget-mixture value curve ``V_F(B)`` of each family, budget interval shaded;
  2. ``delta_curves``      pairwise family differences ``V_A(B) - V_B(B)`` on the
     interval with the H1 interval-average effect and its bootstrap 90 % band;
  3. ``effect_summary``    H1 / H2 / S0-T interval effects with bootstrap q05-q95
     bars, per-task descriptive effects and the audited replicate differences;
  4. ``tgrid_heatmap``     dense GST grid: SR and analytic cost per (fh, ws) cell;
  5. ``baseline_density``  exploratory: plug-in H1 vs the number of grid cells the baseline may tune over;
  6. ``pareto_hull_percent`` every arm, cost in % of always-full inference, Rev 1-style Pareto hulls
     (needs ``--phase0-summary`` for the anchor);
  6b. ``pareto_hull_percent_crd`` / ``_dense_crd``  the same chart with the H-CRD development
     points (``--crd-summary``, a ``sgrid_sweep summarize`` output of CRD arms) overlaid: the
     quantile sweep at fixed (gamma, beta-mult, J_bad, L_max) gets its own hull; ablation points
     (pure CRD, fuse-only, other beta multiples) are drawn but never enter a hull.

Pure post-hoc visualisation: nothing here feeds any verdict. Values are
recomputed with the frozen estimator (``budget_mixture.value_at``) from the
sufficient statistics stored in the outcome design.

Usage:
  python -m exp.dispatch_surface.analysis.plot_budget_amendment \
      --outcome-design exp/dispatch_surface/data/confirmation/libero_10/budget_outcome_design.json \
      --budget-cost-map exp/dispatch_surface/data/confirmation/libero_10/budget_cost_map.json \
      --out-dir exp/dispatch_surface/analysis/figures/libero_10
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from exp.dispatch_surface.analysis.budget_mixture import ArmStats, value_at  # noqa: E402

FAMILY_LABEL = {
    "sv": "RIT on (s, v) — ablation (Risk-Indexed Threshold with the disagreement statistic)",
    "s0": "RIT (Risk-Indexed Threshold, s-only ladder)",
        "threshold": "GST (Grid-Searched Threshold, fh / ws grid)",
    "s0_pl": "RIT-PL (piecewise-linear q̂, IR-addressed)",
}
FAMILY_COLOR = {"sv": "#1f5fbf", "s0": "#2a9d3f", "threshold": "#e07b1a", "s0_pl": "#0b7a75"}
FAMILY_MARKER = {"sv": "o", "s0": "s", "threshold": "D", "s0_pl": "P"}
#: Families drawn on the frontier figures, in this order, when present in the design.
FRONTIER_FAMILIES = ("threshold", "s0", "s0_pl", "sv")
CRD_COLOR = "#8e24aa"
CRD_ABL_COLOR = "#c2185b"
SYSGATE_COLOR = "#6a2d9e"
CRD_SWEEP = {"budget_mult": 2, "j_bad": 3, "l_max": 6, "gamma": 1.0}   # the frozen H-CRD development setting
HYP_LABEL = {"H1": "H1: RIT(s,v) − GST", "H2": "H2: RIT(s,v) − RIT(s-only)",
             "S0_minus_T": "RIT(s-only) − GST"}
_TG = re.compile(r"^dsp_tg_fh(\d+)_ws(\d+)$")
_REV1_T = re.compile(r"^dsp_t_fh(\d+)_ws(\d+)$")
_PL = re.compile(r"^dsp_s0_pl_(?:ir([0-9]+(?:p[0-9]+)?)|p([0-9]+))$")


def _stats(measured: dict) -> dict[str, ArmStats]:
    # Per-episode means are sufficient for value_at (E cancels); use E = 1.
    return {arm: ArmStats(T=float(m["t"]), D=float(m["d"]), S=float(m["sr"]), E=1.0) for arm, m in measured.items()}


def _curve(arms, stats, grid):
    ys = []
    for b in grid:
        v, _ = value_at(arms, stats, float(b))
        ys.append(np.nan if v is None else v)
    return np.asarray(ys, dtype=float)


def _short(arm: str) -> str:
    m = _TG.match(arm) or _REV1_T.match(arm)
    if m:
        return f"{m.group(1)}/{m.group(2)}"
    pl = _PL.match(arm)
    if pl:
        # ir82p5 -> IR82.5 (target inference ratio); p925 -> q.925 (delta-addressed)
        return f"IR{pl.group(1).replace('p', '.')}" if pl.group(1) else f"q.{pl.group(2)}"
    return arm.replace("dsp_", "")


def _present_families(fams: dict) -> list[str]:
    return [f for f in FRONTIER_FAMILIES if f in fams and fams[f]["measured_policies"]]


def _crd_groups(crd_summary: dict) -> tuple[dict, dict]:
    """Split a CRD summarize output into the quantile sweep (one hull) and the
    ablation / off-setting points (drawn, no hull)."""
    from exp.dispatch_surface.sgrid_sweep import crd_params

    sweep, ablation = {}, {}
    for arm, a in crd_summary["arms"].items():
        knobs = crd_params(arm)
        if knobs is None:
            raise SystemExit(f"{arm}: --crd-summary must contain CRD arms only")
        (sweep if knobs == CRD_SWEEP else ablation)[arm] = {**a, "knobs": knobs}
    return sweep, ablation


def _crd_short(arm: str, knobs: dict) -> str:
    q = arm.split("_crd_q", 1)[1].split("_", 1)[0]
    label = f"q0.{q}"
    if knobs == CRD_SWEEP:
        return label
    m, j, L = knobs["budget_mult"], knobs["j_bad"], knobs["l_max"]
    if m == float("inf"):
        return f"{label} fuse-only"
    if j == float("inf") and L is None:
        return f"{label} pure m{m}"
    return f"{label} m{m}"


def _dense_active(measured: dict, b_l: float, b_h: float) -> list[str]:
    """Arms that carry any piece of the family's mixture on the interval."""
    from exp.dispatch_surface.analysis.budget_mixture import active_basis_union

    arms = sorted(measured)
    return active_basis_union(arms, _stats(measured), b_l, b_h)["active"]


def merge_sgrid(design: dict, summary: dict) -> dict:
    """Deep-copy the outcome design and add every sweep arm to its family's
    measured policies (same {cost, sr, t, d} shape). Exploratory only."""
    dense = json.loads(json.dumps(design))
    if summary.get("suite") != design.get("suite"):
        raise SystemExit("sgrid summary suite != outcome design suite")
    for arm, a in summary["arms"].items():
        family = a["family"]
        if family not in dense["families"]:
            if family != "s0_pl":
                raise SystemExit(f"sweep arm {arm} has family {family!r}, unknown to the outcome design")
            # RIT-PL only ever appears on the exploratory frontier figures.
            dense["families"][family] = {"measured_policies": {}, "active": []}
        fam = dense["families"][family]["measured_policies"]
        if arm in fam:
            raise SystemExit(f"sweep arm {arm} already exists in the outcome design")
        fam[arm] = {"cost": a["cost"], "sr": a["sr"], "t": a["t"], "d": a["d"]}
    return dense


def fig_family_frontiers(design: dict, out: pathlib.Path, suffix: str = "") -> None:
    b_l, b_h = design["interval"]
    fams = design["families"]
    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    ax.axvspan(b_l, b_h, color="#bbbbbb", alpha=0.25, lw=0, label=f"budget interval [{b_l}, {b_h}] ms")
    for b in (design["B_1"], design["B_2"]):
        ax.axvline(b, color="#777777", ls=":", lw=1)
    xmin = min(m["cost"] for f in fams.values() for m in f["measured_policies"].values())
    xmax = max(m["cost"] for f in fams.values() for m in f["measured_policies"].values())
    grid = np.linspace(xmin - 0.5, xmax + 0.5, 1200)
    for fam in _present_families(fams):
        f = fams[fam]
        measured = f["measured_policies"]
        stats = _stats(measured)
        arms = sorted(measured)
        ys = _curve(arms, stats, grid)
        ax.plot(grid, ys, "-", color=FAMILY_COLOR[fam], lw=2.4,
                label=f"{FAMILY_LABEL[fam]} — budget-mixture value $V_F(B)$")
        xs = [measured[a]["cost"] for a in arms]
        sr = [measured[a]["sr"] for a in arms]
        active = set(f["active"]) if not suffix else set(_dense_active(measured, b_l, b_h))
        face = [FAMILY_COLOR[fam] if a in active else "white" for a in arms]
        ax.scatter(xs, sr, marker=FAMILY_MARKER[fam], s=46, facecolors=face, edgecolors=FAMILY_COLOR[fam],
                   linewidths=1.2, zorder=4)
        for a, x, y in zip(arms, xs, sr):
            # Surface-family labels sit above-left so they do not collide with grid labels.
            off = (3, 3) if fam == "threshold" else (-30, 7)
            ax.annotate(_short(a), (x, y), xytext=off, textcoords="offset points", fontsize=6.5,
                        color=FAMILY_COLOR[fam], alpha=0.9, fontweight="bold" if fam != "threshold" else "normal")
    ax.scatter([], [], marker="o", facecolors="k", edgecolors="k", s=40, label="measured arm (filled = active in the mixture on the interval)")
    ax.scatter([], [], marker="o", facecolors="white", edgecolors="k", s=40, label="measured arm (never active)")
    ax.set_xlabel("analytic compute cost per decision, B (ms)")
    ax.set_ylabel("success rate on the development set (A′ 30 inits × 10 tasks)")
    ax.set_title("libero_10 development set: family frontiers under the budget-mixture estimand"
                 + (" — with the RIT density sweep (exploratory)" if suffix else "") + "\n"
                 "GST labels = fh/ws percentiles; RIT labels = calibration quantile; dashed = B_1, B_2")
    ax.set_ylim(0.1, 0.9)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(out / f"family_frontiers{suffix}.png", dpi=170)
    fig.savefig(out / f"family_frontiers{suffix}.pdf")
    plt.close(fig)


def fig_delta_curves(design: dict, out: pathlib.Path, suffix: str = "") -> None:
    from exp.dispatch_surface.analysis.budget_mixture import auc_norm

    b_l, b_h = design["interval"]
    fams = design["families"]
    grid = np.linspace(b_l, b_h, 800)
    curves = {}
    for fam, f in fams.items():
        measured = f["measured_policies"]
        curves[fam] = _curve(sorted(measured), _stats(measured), grid)
    pairs = [("H1", "sv", "threshold"), ("H2", "sv", "s0"), ("S0_minus_T", "s0", "threshold")]
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    ax.axhline(0.0, color="black", lw=1)
    colors = {"H1": "#c0392b", "H2": "#1f5fbf", "S0_minus_T": "#2a9d3f"}
    plug = {}
    for name, a, b in pairs:
        ax.plot(grid, curves[a] - curves[b], "-", color=colors[name], lw=2.2,
                label=f"{HYP_LABEL[name]}: $V_{{{a}}}(B) - V_{{{b}}}(B)$")
        if suffix:
            ma = fams[a]["measured_policies"]
            mb = fams[b]["measured_policies"]
            plug[name] = auc_norm(sorted(ma), _stats(ma), b_l, b_h) - auc_norm(sorted(mb), _stats(mb), b_l, b_h)
            ax.axhline(plug[name], color=colors[name], lw=1, ls="--")
        else:
            h = design["hypotheses"][name]
            ax.axhline(h["effect_plugin"], color=colors[name], lw=1, ls="--")
            ax.axhspan(h["bootstrap_q05"], h["bootstrap_q95"], color=colors[name], alpha=0.08, lw=0)
    if suffix:
        ax.annotate("dense families, plug-in interval effects (no bootstrap):\n"
                    + "\n".join(f"{HYP_LABEL[n]}: {plug[n]:+.4f}" for n, _, _ in pairs),
                    xy=(b_l + 0.15, ax.get_ylim()[0] + 0.005), fontsize=9, va="bottom",
                    bbox=dict(boxstyle="round", fc="white", ec="#c0392b", alpha=0.95))
    else:
        h1 = design["hypotheses"]["H1"]
        ax.annotate(f"H1 interval effect = {h1['effect_plugin']:+.4f}\nbootstrap q05 = {h1['bootstrap_q05']:+.4f}, "
                    f"q95 = {h1['bootstrap_q95']:+.4f}\nverdict: {design['verdict']} ({h1['reason']})",
                    xy=(b_l + 0.15, h1["bootstrap_q05"] - 0.005), fontsize=9, va="top",
                    bbox=dict(boxstyle="round", fc="white", ec="#c0392b", alpha=0.95))
    for b in (design["B_1"], design["B_2"]):
        ax.axvline(b, color="#777777", ls=":", lw=1)
    ax.set_xlim(b_l, b_h)
    ax.set_xlabel("compute budget B (ms per decision)")
    ax.set_ylabel("difference of budget-mixture values (success-rate points)")
    ax.set_title("Pairwise family differences on the budget interval"
                 + (" — dense families (exploratory)" if suffix else "") + "\n"
                 + ("dashed = interval-average plug-in effect" if suffix else
                    "dashed = interval-average plug-in effect; shaded = bootstrap 90 % band (10 000 paired replicates)"))
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(out / f"delta_curves{suffix}.png", dpi=170)
    fig.savefig(out / f"delta_curves{suffix}.pdf")
    plt.close(fig)


def fig_effect_summary(design: dict, out: pathlib.Path) -> None:
    names = ["H1", "H2", "S0_minus_T"]
    hyps = design["hypotheses"]
    per_task = design["per_task_descriptive"]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 5.2), gridspec_kw={"width_ratios": [2.2, 1]})
    ax.axhline(0.0, color="black", lw=1)
    rng = np.random.default_rng(0)
    for i, n in enumerate(names):
        h = hyps[n]
        ax.errorbar([i], [h["effect_plugin"]], yerr=[[h["effect_plugin"] - h["bootstrap_q05"]], [h["bootstrap_q95"] - h["effect_plugin"]]],
                    fmt="o", color="#c0392b" if n == "H1" else "#333333", capsize=6, ms=8, lw=2,
                    label="interval effect with bootstrap q05–q95" if i == 0 else None)
        ys = [per_task[t][n] for t in sorted(per_task, key=int)]
        ax.scatter(i + rng.uniform(-0.18, 0.18, len(ys)), ys, s=22, color="#888888", alpha=0.8, zorder=3,
                   label="per-task descriptive effect (10 tasks)" if i == 0 else None)
        ax.annotate(f"{h['effect_plugin']:+.3f}\n[{h['bootstrap_q05']:+.3f}, {h['bootstrap_q95']:+.3f}]",
                    (i, h["bootstrap_q95"]), xytext=(10, 6), textcoords="offset points", fontsize=8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([HYP_LABEL[n] for n in names], fontsize=8.5)
    ax.set_ylabel("interval-average value difference (SR points)")
    ax.set_title("Development-set effects (libero_10)\nH1 is the only gating hypothesis: pass needs q05 > 0")
    ax.grid(alpha=0.25, axis="y")
    ax.legend(fontsize=8, loc="upper left")
    reps = hyps["H1"]["audit"]["replicates"]
    diffs = np.array([r["sv"]["analytic"] - r["threshold"]["analytic"] for r in reps])
    ax2.hist(diffs, bins=18, color="#c0392b", alpha=0.75)
    ax2.axvline(0.0, color="black", lw=1)
    ax2.axvline(hyps["H1"]["effect_plugin"], color="#c0392b", ls="--", lw=1.2)
    ax2.set_xlabel("$V_{RIT(s,v)} - V_{GST}$ (audited replicates)")
    ax2.set_ylabel("count")
    ax2.set_title(f"H1 over {len(diffs)} audited replicates\n"
                  f"share ≤ 0: {(diffs <= 0).mean():.2f}", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "effect_summary.png", dpi=170)
    fig.savefig(out / "effect_summary.pdf")
    plt.close(fig)


def fig_tgrid_heatmap(design: dict, out: pathlib.Path) -> None:
    measured = design["families"]["threshold"]["measured_policies"]
    cells = {}
    for arm, m in measured.items():
        g = _TG.match(arm)
        if g:
            cells[(int(g.group(1)), int(g.group(2)))] = m
    fhs = sorted({k[0] for k in cells})
    wss = sorted({k[1] for k in cells})
    sr = np.full((len(fhs), len(wss)), np.nan)
    cost = np.full((len(fhs), len(wss)), np.nan)
    for (fh, ws), m in cells.items():
        sr[fhs.index(fh), wss.index(ws)] = m["sr"]
        cost[fhs.index(fh), wss.index(ws)] = m["cost"]
    active = set(design["families"]["threshold"]["active"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, mat, title, fmt, cmap in ((axes[0], sr, "success rate (300 episodes per cell)", "{:.2f}", "viridis"),
                                     (axes[1], cost, "analytic cost per decision (ms)", "{:.1f}", "magma_r")):
        im = ax.imshow(mat, cmap=cmap, aspect="auto")
        ax.set_xticks(range(len(wss)))
        ax.set_xticklabels([str(w) for w in wss])
        ax.set_yticks(range(len(fhs)))
        ax.set_yticklabels([str(f) for f in fhs])
        ax.set_xlabel("warm-start percentile ws")
        ax.set_ylabel("full-hit percentile fh")
        ax.set_title(title)
        for i, fh in enumerate(fhs):
            for j, ws in enumerate(wss):
                if np.isnan(mat[i, j]):
                    ax.text(j, i, "—", ha="center", va="center", color="#999999", fontsize=8)
                    continue
                arm = f"dsp_tg_fh{fh}_ws{ws}"
                ax.text(j, i, fmt.format(mat[i, j]), ha="center", va="center", fontsize=8,
                        color="white" if (mat[i, j] < np.nanmean(mat)) == (cmap == "viridis") else "black",
                        fontweight="bold" if arm in active else "normal")
                if arm in active:
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec="#c0392b", lw=2))
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.suptitle("Dense GST grid on the development set (29 cells; red frame = active on the budget interval)")
    fig.tight_layout()
    fig.savefig(out / "tgrid_heatmap.png", dpi=170)
    fig.savefig(out / "tgrid_heatmap.pdf")
    plt.close(fig)


def _pareto_staircase(pts: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Every non-dominated point (no other point with <= cost and >= sr), cost-ordered."""
    front = []
    best = -1.0
    for c, sr in sorted(pts):
        if sr > best:
            front.append((c, sr))
            best = sr
    return front


def fig_pareto_hull_percent(design: dict, anchor: dict, out: pathlib.Path, suffix: str = "",
                            crd: dict | None = None, sysgate: dict | None = None) -> None:
    """Every measured arm on one chart, cost as a percentage of always-full
    inference, with each family's realisable Pareto frontier drawn the Rev 1 /
    Phase 0 way (``frontier_hull.upper_concave_hull``: dominated points removed,
    upper concave envelope), not the budget-mixture value curve."""
    from exp.dispatch_surface.analysis.frontier_hull import upper_concave_hull

    full_cost = float(anchor["realized_cost_ms"])
    full_sr = float(anchor["sr"])
    pct = lambda c: 100.0 * c / full_cost  # noqa: E731
    b_l, b_h = design["interval"]
    fams = design["families"]
    fig, ax = plt.subplots(figsize=(10.5, 6.6))
    ax.axvspan(pct(b_l), pct(b_h), color="#bbbbbb", alpha=0.25, lw=0,
               label=f"budget interval [{b_l}, {b_h}] ms = [{pct(b_l):.1f}, {pct(b_h):.1f}] % of full inference")
    ax.axhline(full_sr, color="#444444", ls=":", lw=1)
    ax.scatter([100.0], [full_sr], marker="*", s=320, color="#333333", zorder=6,
               label=f"always full inference: 100 % cost ({full_cost:.1f} ms), SR {full_sr:.3f}")
    ax.annotate("full inference", (100.0, full_sr), xytext=(-70, 8), textcoords="offset points", fontsize=8, color="#333333")
    for fam in _present_families(fams):
        f = fams[fam]
        measured = f["measured_policies"]
        arms = sorted(measured)
        pts = [(measured[a]["cost"], measured[a]["sr"]) for a in arms]
        front = _pareto_staircase(pts)
        ax.plot([pct(c) for c, _ in front], [s_ for _, s_ in front], "-", color=FAMILY_COLOR[fam], lw=2.4,
                label=f"{FAMILY_LABEL[fam]} — Pareto frontier (every non-dominated arm, {len(front)} / {len(pts)})")
        hull = upper_concave_hull(pts)
        ax.plot([pct(c) for c, _ in hull], [s_ for _, s_ in hull], ":", color=FAMILY_COLOR[fam], lw=1.1, alpha=0.55, zorder=2)
        on_front = {(round(c, 9), round(s_, 9)) for c, s_ in front}
        face = [FAMILY_COLOR[fam] if (round(c, 9), round(s_, 9)) in on_front else "white" for c, s_ in pts]
        ax.scatter([pct(c) for c, _ in pts], [s_ for _, s_ in pts], marker=FAMILY_MARKER[fam], s=46,
                   facecolors=face, edgecolors=FAMILY_COLOR[fam], linewidths=1.2, zorder=4)
        for a, (c, s_) in zip(arms, pts):
            off = (3, 3) if fam == "threshold" else (-30, 7)
            ax.annotate(_short(a), (pct(c), s_), xytext=off, textcoords="offset points", fontsize=6.5,
                        color=FAMILY_COLOR[fam], alpha=0.9, fontweight="bold" if fam != "threshold" else "normal")
    if crd is not None:
        sweep, ablation = _crd_groups(crd)
        if sweep:
            arms_c = sorted(sweep)
            pts_c = [(sweep[a]["cost"], sweep[a]["sr"]) for a in arms_c]
            hull_c = upper_concave_hull(pts_c) if len(pts_c) >= 2 else list(pts_c)
            ax.plot([pct(c) for c, _ in hull_c], [s_ for _, s_ in hull_c], "--", color=CRD_COLOR, lw=2.4, zorder=5,
                    label=f"H-CRD on the (s,v) surface (γ=1, β=2δ, J_bad=3, L_max=6) — Pareto frontier "
                          f"({len(hull_c)} vertices / {len(pts_c)} arms, development points)")
            on_hull = {(round(c, 9), round(s_, 9)) for c, s_ in hull_c}
            face = [CRD_COLOR if (round(c, 9), round(s_, 9)) in on_hull else "white" for c, s_ in pts_c]
            ax.scatter([pct(c) for c, _ in pts_c], [s_ for _, s_ in pts_c], marker="^", s=70, facecolors=face,
                       edgecolors=CRD_COLOR, linewidths=1.4, zorder=7)
            for a, (c, s_) in zip(arms_c, pts_c):
                ax.annotate(_crd_short(a, sweep[a]["knobs"]), (pct(c), s_), xytext=(6, -11), textcoords="offset points",
                            fontsize=7, color=CRD_COLOR, fontweight="bold")
        if ablation:
            arms_a = sorted(ablation)
            pts_a = [(ablation[a]["cost"], ablation[a]["sr"]) for a in arms_a]
            ax.scatter([pct(c) for c, _ in pts_a], [s_ for _, s_ in pts_a], marker="v", s=64, facecolors="white",
                       edgecolors=CRD_ABL_COLOR, linewidths=1.4, zorder=7,
                       label="H-CRD ablations (pure CRD / fuse-only / other β) — no hull")
            for a, (c, s_) in zip(arms_a, pts_a):
                ax.annotate(_crd_short(a, ablation[a]["knobs"]), (pct(c), s_), xytext=(6, 4), textcoords="offset points",
                            fontsize=6.5, color=CRD_ABL_COLOR)
    if sysgate is not None:
        arms_g = sorted(sysgate["arms"], key=lambda a: sysgate["arms"][a]["cost"])
        pts_g = [(sysgate["arms"][a]["cost"], sysgate["arms"][a]["sr"]) for a in arms_g]
        front_g = _pareto_staircase(pts_g)
        ax.plot([pct(c) for c, _ in front_g], [s_ for _, s_ in front_g], "--", color=SYSGATE_COLOR, lw=2.4, zorder=5,
                label=f"RIT ladder (s-only) + production gate (score_hysteresis, j=3, probe=3, L=6) — Pareto frontier "
                      f"({len(front_g)} / {len(pts_g)} arms, development points)")
        on_g = {(round(c, 9), round(s_, 9)) for c, s_ in front_g}
        face_g = [SYSGATE_COLOR if (round(c, 9), round(s_, 9)) in on_g else "white" for c, s_ in pts_g]
        ax.scatter([pct(c) for c, _ in pts_g], [s_ for _, s_ in pts_g], marker="^", s=70, facecolors=face_g,
                   edgecolors=SYSGATE_COLOR, linewidths=1.4, zorder=7)
        for a, (c, s_) in zip(arms_g, pts_g):
            ax.annotate(_short(a), (pct(c), s_), xytext=(4, -12), textcoords="offset points",
                        fontsize=6.5, color=SYSGATE_COLOR, fontweight="bold")
    ax.scatter([], [], marker="o", facecolors="k", edgecolors="k", s=40, label="arm on its family's frontier (non-dominated)")
    ax.scatter([], [], marker="o", facecolors="white", edgecolors="k", s=40, label="arm dominated within its family")
    ax.plot([], [], ":", color="#666666", lw=1.1, label="two-arm mixture envelope (upper concave hull, Rev 1 rule)")
    ax.set_xlabel("analytic compute cost per decision, % of always-full inference")
    ax.set_ylabel("success rate on the development set (A′ 30 inits × 10 tasks)")
    ax.set_title("libero_10 development set: every measured arm — solid = per-family Pareto frontier (all non-dominated arms), dotted = mixture envelope"
                 + (" — with the RIT density sweep (exploratory)" if "dense" in suffix else "")
                 + (" — H-CRD development points overlaid (exploratory)" if crd is not None else "")
                 + (" — RIT(s-only) + production gate system points overlaid (exploratory)" if sysgate is not None else "") + "\n"
                 "GST labels = fh/ws percentiles; RIT labels = calibration quantile (sv = q0.90, sv_minus = q0.80, p85 = q0.85)")
    ax.set_xlim(40, 104)
    ax.set_ylim(0.1, 0.9)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=7.6, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(out / f"pareto_hull_percent{suffix}.png", dpi=170)
    fig.savefig(out / f"pareto_hull_percent{suffix}.pdf")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outcome-design", required=True)
    ap.add_argument("--budget-cost-map", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--sgrid-summary", default="",
                    help="sgrid_sweep summarize output; adds the surface density sweep as *_dense variants")
    ap.add_argument("--phase0-summary", default="",
                    help="phase0_summary.json; supplies the always-full-inference anchor for the percent-cost chart")
    ap.add_argument("--crd-summary", default="",
                    help="sgrid_sweep summarize output of H-CRD arms; overlays them on pareto_hull_percent(_dense)_crd")
    ap.add_argument("--sysgate-summary", default="",
                    help="sgrid_sweep summarize output of gated s0 arms; overlays them on pareto_hull_percent_dense_sysgate")
    args = ap.parse_args()
    if args.crd_summary and not args.phase0_summary:
        raise SystemExit("--crd-summary needs --phase0-summary (the percent-cost anchor)")
    design = json.loads(pathlib.Path(args.outcome_design).read_text())
    if design["budget_cost_map_sha256"] != cost_map_sha(args.budget_cost_map):
        raise SystemExit("outcome design does not bind the given budget cost map")
    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # Latest-only policy (owner, 2026-08-30): when the dense sweep summary is
    # given, the frozen-basis versions of fig 1 / 2 / pareto are superseded and
    # are NOT regenerated; only the *_dense variants are written.
    if not args.sgrid_summary:
        fig_family_frontiers(design, out)
        fig_delta_curves(design, out)
    fig_effect_summary(design, out)
    fig_tgrid_heatmap(design, out)
    fig_baseline_density(design, out)
    if args.phase0_summary:
        summary = json.loads(pathlib.Path(args.phase0_summary).read_text())
        anchor_arm = summary["arms"]["always_full_inference"]
        anchor = {"realized_cost_ms": anchor_arm["realized_cost_ms"], "sr": anchor_arm["sr_recorded_not_judged"]}
        if not args.sgrid_summary:
            fig_pareto_hull_percent(design, anchor, out)
        if args.crd_summary:
            crd = json.loads(pathlib.Path(args.crd_summary).read_text())
            if crd.get("suite") != design.get("suite"):
                raise SystemExit("crd summary suite != outcome design suite")
            fig_pareto_hull_percent(design, anchor, out, suffix="_crd", crd=crd)
    if args.sgrid_summary:
        dense = merge_sgrid(design, json.loads(pathlib.Path(args.sgrid_summary).read_text()))
        fig_family_frontiers(dense, out, suffix="_dense")
        fig_delta_curves(dense, out, suffix="_dense")
        if args.phase0_summary:
            fig_pareto_hull_percent(dense, anchor, out, suffix="_dense")
            if args.crd_summary:
                fig_pareto_hull_percent(dense, anchor, out, suffix="_dense_crd", crd=crd)
            if args.sysgate_summary:
                sg = json.loads(pathlib.Path(args.sysgate_summary).read_text())
                if sg.get("suite") != design.get("suite"):
                    raise SystemExit("sysgate summary suite != outcome design suite")
                fig_pareto_hull_percent(dense, anchor, out, suffix="_dense_sysgate", sysgate=sg)
    print(f"figures written to {out}")


def cost_map_sha(path: str) -> str:
    import hashlib
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()



def fig_baseline_density(design: dict, out: pathlib.Path, seed: int = 20260830, n_draw: int = 3000) -> None:
    """Exploratory: plug-in H1 as a function of how many dense-grid cells the
    threshold baseline may tune over (random subsets), against the full-family
    value. Shows how the Phase 0 gap (3 Rev 1 cells) closes with baseline density."""
    import itertools

    from exp.dispatch_surface.analysis.budget_mixture import auc_with_support

    b_l, b_h = design["interval"]
    thr = design["families"]["threshold"]["measured_policies"]
    sv = design["families"]["sv"]["measured_policies"]
    st_thr, st_sv = _stats(thr), _stats(sv)
    sv_arms = sorted(sv)
    tg = sorted(a for a in thr if _TG.match(a))
    rev1 = sorted(a for a in thr if _REV1_T.match(a))
    rng = np.random.default_rng(seed)
    ks = [3, 5, 8, 12, 16, 20, 24, len(tg)]
    rows = []
    for k in ks:
        if k == 3:
            subsets = list(itertools.combinations(tg, 3))
        elif k == len(tg):
            subsets = [tuple(tg)]
        else:
            subsets = [tuple(sorted(rng.choice(tg, k, replace=False))) for _ in range(n_draw)]
        vals = [v for v, miss in (auc_with_support(sv_arms, st_sv, list(s), st_thr, b_l, b_h) for s in subsets) if not miss]
        vals = np.asarray(vals)
        rows.append((k, np.median(vals), np.quantile(vals, 0.05), np.quantile(vals, 0.95), len(vals), len(subsets)))
    h1 = design["hypotheses"]["H1"]
    phase0 = auc_with_support(sv_arms, st_sv, rev1, st_thr, b_l, b_h)[0]
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    k_arr = np.array([r[0] for r in rows])
    ax.fill_between(k_arr, [r[2] for r in rows], [r[3] for r in rows], color="#e07b1a", alpha=0.18, lw=0,
                    label="q05–q95 over random cell subsets of the dense grid")
    ax.plot(k_arr, [r[1] for r in rows], "-o", color="#e07b1a", lw=2.2, ms=5,
            label="median plug-in H1 = $V_{RIT(s,v)}$ − $V_{GST}$(k cells)")
    ax.axhline(h1["effect_plugin"], color="#c0392b", ls="--", lw=1.4, label=f"full 29-cell grid: {h1['effect_plugin']:+.4f}")
    ax.axhspan(h1["bootstrap_q05"], h1["bootstrap_q95"], color="#c0392b", alpha=0.10, lw=0,
               label="episode-bootstrap 90 % band of the full-grid H1")
    ax.scatter([3], [phase0], marker="*", s=260, color="#1f5fbf", zorder=5,
               label=f"Phase 0 baseline (Rev 1's 3 cells): {phase0:+.4f}")
    ax.annotate("3 Rev 1 cells", (3, phase0), xytext=(8, 6), textcoords="offset points", fontsize=8, color="#1f5fbf")
    ax.axhline(0.0, color="black", lw=1)
    ax.set_xlabel("number of dense-grid cells the GST baseline may tune over, k")
    ax.set_ylabel("interval-average $V_{RIT(s,v)} - V_{GST}$ (SR points)")
    ax.set_title("How RIT's margin shrinks with GST's tuning freedom\n"
                 "(plug-in, development set; RIT(s,v) itself has only 3 measured arms)")
    ax.set_xticks(k_arr)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "baseline_density.png", dpi=170)
    fig.savefig(out / "baseline_density.pdf")
    plt.close(fig)
    for r in rows:
        print(f"k={r[0]:2d} median {r[1]:+.4f} q05 {r[2]:+.4f} q95 {r[3]:+.4f} feasible {r[4]}/{r[5]}")


if __name__ == "__main__":
    main()
