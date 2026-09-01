"""Per-episode oracle headroom above the measured families (exploratory).

Every arm of the Rev 1 primary layer, the Phase 0 exploratory layer and the
dense GST grid was rolled out on the SAME 300 A' episodes. With
hindsight, an oracle that may pick a different arm for every episode bounds
what ANY episode-adaptive dispatcher built from these arms could achieve:

    max_{a_1..a_n}  (1/n) sum_i S[i, a_i]
    s.t.            sum_i T[i, a_i] <= B * sum_i D[i, a_i]     (ratio-of-sums cost)

Solved by the Lagrangian sweep a_i(lambda) = argmax_a S[i,a] - lambda (T[i,a] - B D[i,a])
(integer solution; the fractional hull would be marginally higher). Compared
against each family's budget-mixture value V_F(B) (one arm or a two-arm
episode-level mixture, the same for every episode). The gap oracle - V_F is
the headroom that episode-level difficulty information could unlock; the
gap between the oracle and always-full inference is what NO reassignment of
the existing arms can recover.

Usage:
  python -m exp.dispatch_surface.analysis.oracle_headroom \
      --source tgrid:<arm_matrix>:<journal>:<per_step> \
      --source phase0:<arm_matrix>:<journal>:<per_step> \
      --source rev1:<arm_matrix>:<journal>:<per_step> \
      --split-manifest ... --outcome-design ... --phase0-summary ... --out-json ... --out-fig ...
"""
from __future__ import annotations

import argparse
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from exp.dispatch_surface.analysis.budget_mixture import ArmStats, value_at  # noqa: E402
from exp.dispatch_surface.analysis.precheck_io import load_accepted_cells_costonly, load_cost_cells_costonly  # noqa: E402
from exp.dispatch_surface.run_precheck import NUM_TASKS, official_test_inits  # noqa: E402

FAMILY_OF_PREFIX = (("dsp_tg_", "threshold"), ("dsp_t_", "threshold"), ("dsp_sv", "sv"), ("dsp_s0", "s0"),
                    ("always_full_inference", "anchor"))


def family_of(arm: str) -> str:
    for prefix, fam in FAMILY_OF_PREFIX:
        if arm.startswith(prefix):
            return fam
    raise SystemExit(f"unknown family for arm {arm}")


def load_source(spec: str, officials, trials: int):
    tag, matrix_path, journal, per_step = spec.split(":", 3)
    matrix = json.loads(pathlib.Path(matrix_path).read_text())
    arms = sorted(matrix["arms"])
    grid = {(t, i) for t in range(NUM_TASKS) for i in range(trials)}
    accepted = load_accepted_cells_costonly(journal, arms, grid)
    success = {a: {} for a in arms}
    for line in open(journal):
        row = json.loads(line)
        arm = row.get("yaml_id")
        if arm in success and row.get("accepted") is True:
            uid = row["task_uid"]
            _, task, subset = uid.split(":")[0], int(uid.split(":")[2]), int(uid.split(":")[3])
            success[arm][(task, subset)] = bool(row.get("success"))
    cells, _summary = load_cost_cells_costonly(per_step, arms, accepted, officials)
    out = {}
    for arm in arms:
        if len(cells[arm]) != len(grid):
            raise SystemExit(f"{tag}:{arm}: {len(cells[arm])} cells, expected {len(grid)}")
        out[arm] = {cell: (cells[arm][cell][0], cells[arm][cell][1], success[arm][cell]) for cell in grid}
    return tag, out


def oracle_curve(T, D, S, budgets, lambdas):
    """Integer Lagrangian oracle: best feasible mean success per budget."""
    n = T.shape[0]
    out = []
    for B in budgets:
        g = T - B * D  # (n, arms) budget slack per episode
        best = -1.0
        for lam in lambdas:
            obj = S - lam * g
            pick = obj.argmax(axis=1)
            idx = np.arange(n)
            if g[idx, pick].sum() <= 1e-9:
                best = max(best, S[idx, pick].mean())
        out.append(best if best >= 0 else np.nan)
    return np.asarray(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", action="append", required=True, help="tag:arm_matrix:journal:per_step")
    ap.add_argument("--split-manifest", required=True)
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--outcome-design", required=True)
    ap.add_argument("--phase0-summary", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-fig", required=True)
    args = ap.parse_args()

    officials = official_test_inits(args.split_manifest, args.trials)
    per_arm = {}
    for spec in args.source:
        tag, data = load_source(spec, officials, args.trials)
        for arm, cells in data.items():
            if arm in per_arm:
                raise SystemExit(f"arm {arm} appears in two sources")
            per_arm[arm] = cells
    arms = sorted(per_arm)
    cells = sorted(next(iter(per_arm.values())))
    T = np.array([[per_arm[a][c][0] for a in arms] for c in cells], dtype=float)
    D = np.array([[per_arm[a][c][1] for a in arms] for c in cells], dtype=float)
    S = np.array([[1.0 if per_arm[a][c][2] else 0.0 for a in arms] for c in cells], dtype=float)
    fams = np.array([family_of(a) for a in arms])
    design = json.loads(pathlib.Path(args.outcome_design).read_text())
    summary = json.loads(pathlib.Path(args.phase0_summary).read_text())
    anchor = summary["arms"]["always_full_inference"]
    full_cost, full_sr = float(anchor["realized_cost_ms"]), float(anchor["sr_recorded_not_judged"])
    b_l, b_h = design["interval"]

    budgets = np.linspace(30.0, full_cost, 150)
    lambdas = np.concatenate([[0.0], np.logspace(-5, 0, 200)])
    subsets = {
        "all arms": np.ones(len(arms), bool),
        "GST arms only": fams == "threshold",
        "RIT arms only (sv + s0)": np.isin(fams, ["sv", "s0"]),
    }
    curves = {name: oracle_curve(T[:, m], D[:, m], S[:, m], budgets, lambdas) for name, m in subsets.items()}
    # single-episode diagnostic: cheapest successful arm per episode
    any_success = (S.max(axis=1) > 0).mean()
    cheapest = []
    for i in range(len(cells)):
        ok = np.where(S[i] > 0)[0]
        cheapest.append((T[i, ok] / D[i, ok]).min() if len(ok) else np.nan)
    cheapest = np.asarray(cheapest)

    family_values = {}
    for fam, f in design["families"].items():
        m = f["measured_policies"]
        st = {a: ArmStats(T=v["t"], D=v["d"], S=v["sr"], E=1.0) for a, v in m.items()}
        family_values[fam] = np.array([np.nan if (v := value_at(sorted(m), st, float(b))[0]) is None else v for b in budgets])

    fig, ax = plt.subplots(figsize=(10, 6.2))
    ax.axvspan(b_l, b_h, color="#bbbbbb", alpha=0.25, lw=0, label=f"budget interval [{b_l}, {b_h}] ms")
    ax.axhline(full_sr, color="#444444", ls=":", lw=1)
    ax.scatter([full_cost], [full_sr], marker="*", s=300, color="#333333", zorder=6, label=f"always full inference ({full_sr:.3f})")
    styles = {"all arms": ("#c0392b", "-"), "GST arms only": ("#e07b1a", "--"), "RIT arms only (sv + s0)": ("#1f5fbf", "--")}
    for name, ys in curves.items():
        c, ls = styles[name]
        ax.plot(budgets, ys, ls, color=c, lw=2.2, label=f"per-episode oracle, {name} ({int(subsets[name].sum())} arms)")
    fam_style = {"threshold": ("#e07b1a", "GST — budget-mixture value"), "sv": ("#1f5fbf", "RIT (s,v) — budget-mixture value"),
                 "s0": ("#2a9d3f", "RIT (s-only) — budget-mixture value")}
    for fam, ys in family_values.items():
        c, lab = fam_style[fam]
        ax.plot(budgets, ys, "-", color=c, lw=1.6, alpha=0.85, label=lab)
    ax.set_xlabel("compute budget B (ms per decision)")
    ax.set_ylabel("success rate on the development set (A′ 300 episodes)")
    ax.set_title("Per-episode hindsight oracle over the same 300 episodes (NOT attainable: outcomes are deterministic\n"
                 f"but decision-sensitive, so selecting among {len(arms)} decision sequences per episode succeeds {any_success:.3f} of the time)")
    ax.set_ylim(0.1, 1.03)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="lower right", framealpha=0.95)
    secax = ax.secondary_xaxis("top", functions=(lambda b: 100.0 * b / full_cost, lambda p: p * full_cost / 100.0))
    secax.set_xlabel("% of always-full inference cost")
    fig.tight_layout()
    fig.savefig(args.out_fig, dpi=170)
    fig.savefig(str(pathlib.Path(args.out_fig).with_suffix(".pdf")))

    probe = [b_l, design["B_1"], design["B_2"], b_h, 50.0, 55.0, 60.0]
    def at(ys, b):
        return float(np.interp(b, budgets, ys))
    table = {f"{b:g}": {"oracle_all": at(curves["all arms"], b), "oracle_threshold_only": at(curves["GST arms only"], b),
                        "oracle_surface_only": at(curves["RIT arms only (sv + s0)"], b),
                        **{f"V_{fam}": at(ys, b) for fam, ys in family_values.items()}} for b in probe}
    out = {
        "protocol": "dispatch_surface_rev2_oracle_headroom", "posthoc_exploratory": True, "n_arms": len(arms),
        "arms": arms, "n_episodes": len(cells), "any_arm_success_rate": float(any_success),
        "cheapest_successful_cost_ms": {"median": float(np.nanmedian(cheapest)), "q25": float(np.nanquantile(cheapest, 0.25)),
                                        "q75": float(np.nanquantile(cheapest, 0.75)), "n_never_succeeds": int(np.isnan(cheapest).sum())},
        "budget_probe": table, "full_inference": {"cost_ms": full_cost, "sr": full_sr},
    }
    pathlib.Path(args.out_json).write_text(json.dumps(out, indent=2, sort_keys=True))
    for b, row in table.items():
        print(f"B={b:>5}: oracle_all {row['oracle_all']:.3f}  oracle_thr {row['oracle_threshold_only']:.3f}  "
              f"oracle_surf {row['oracle_surface_only']:.3f}  V_thr {row['V_threshold']:.3f}  V_sv {row['V_sv']:.3f}  V_s0 {row['V_s0']:.3f}")
    print(f"any-arm success {any_success:.3f}; cheapest successful cost median {np.nanmedian(cheapest):.1f} ms; "
          f"never succeeds: {int(np.isnan(cheapest).sum())}/{len(cells)}")


if __name__ == "__main__":
    main()
