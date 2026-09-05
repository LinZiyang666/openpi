"""Cross-line comparison of CP2 arms against the RIT-Pareto reference frontier.

For a 50-trajectory-library group: every CP2 arm of ``--cp2-run-dir`` is
compared with the ``exp/rit_pareto`` K=2 no-gate reference run of the same
suite (same pruned-500 pool, same library content) by plan §3.11 — the
reference arms are priced with the CP1 tier costs (``rit_pareto.tier_cost``),
the CP2 arm with the CP2 costs, both on the ``% of always-full`` axis; the
two-sided stratified bootstrap of ``stats.bootstrap_frontier_delta`` gives the
ΔSR point / interval / support-miss / three-way decision.

Usage:
  uv run python -m exp.actioncache_baseline.compare_to_reference \\
      --cp2-run-dir <run dir> --ref-run-dir exp/rit_pareto/data/runs/libero_spatial_ng \\
      --export-record <export_record.json> \\
      --out <comparison.json> [--B 2000] [--seed 0]
"""

from __future__ import annotations

import argparse
import json
import pathlib

from exp.actioncache_baseline import libs, stats
from exp.actioncache_baseline.aggregate import aggregate as aggregate_cp2_run
from exp.rit_pareto.aggregate_rit import decision_cost as cp1_decision_cost


def compare(cp2_run_dir: str, ref_run_dir: str, *, export_record: dict,
            expect_episodes: int, allow_partial: bool, B: int, seed: int,
            arms: list[str] | None = None) -> dict:
    # A comparison is a publishable result, so it must not provide a side door
    # around the plan §3.11 completeness, provenance, and tier-purity gates.
    checked = aggregate_cp2_run(
        cp2_run_dir,
        expect_episodes=expect_episodes,
        allow_partial=allow_partial,
        export_record=export_record,
    )
    cp2 = checked["ledger"]
    ref = stats.load_episode_ledger(ref_run_dir, cp1_decision_cost)
    miss_ms = libs.miss_cost("cuda_graph")
    out: dict[str, dict] = {}
    for arm, eps in cp2.items():
        if arms and arm not in arms:
            continue
        out[arm] = stats.bootstrap_frontier_delta(eps, ref, miss_ms=miss_ms, B=B, seed=seed)
    return {"protocol": libs.PROTOCOL, "cp2_run_dir": str(pathlib.Path(cp2_run_dir).resolve()),
            "ref_run_dir": str(pathlib.Path(ref_run_dir).resolve()), "B": B, "seed": seed,
            "reference_arms": sorted(ref), "cp2_audit": checked["audit"], "comparisons": out}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cp2-run-dir", required=True)
    ap.add_argument("--ref-run-dir", required=True)
    ap.add_argument("--export-record", required=True,
                    help="CP2 export_record.json; binds the run's arms and library digest")
    ap.add_argument("--expect-episodes", type=int, default=500)
    ap.add_argument("--allow-partial", action="store_true",
                    help="relax only episode count; all identity/provenance gates remain active")
    ap.add_argument("--arms", default="", help="comma list of CP2 arms; empty = all")
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    arms = [a for a in args.arms.split(",") if a.strip()] or None
    export_record = json.loads(pathlib.Path(args.export_record).read_text(encoding="utf-8"))
    res = compare(
        args.cp2_run_dir,
        args.ref_run_dir,
        export_record=export_record,
        expect_episodes=args.expect_episodes,
        allow_partial=args.allow_partial,
        B=args.B,
        seed=args.seed,
        arms=arms,
    )
    res["export_record"] = str(pathlib.Path(args.export_record).resolve())
    libs.dump_json(args.out, res)
    for arm, r in res["comparisons"].items():
        d = r["delta_sr"]
        print(f"{arm}: IR={r['ir_cp2']:.2f} SR={r['sr_cp2']:.3f} ref={r['sr_reference_at_ir']} "
              f"dSR={None if d is None else round(d, 4)} ci={r['delta_ci95']} miss={r['support_miss_frac']:.3f} -> {r['decision']}")


if __name__ == "__main__":
    main()
