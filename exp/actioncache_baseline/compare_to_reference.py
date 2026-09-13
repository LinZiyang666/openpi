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
from exp.actioncache_baseline.aggregate import aggregate as aggregate_cp2_run, pricing_for


def cp1_reference_cost(record: libs.CostRecord):
    """Per-decision cost of a CP1-line reference verdict under the same teacher table.

    A CP1 FULL_HIT skips stage 2 and 3 (``s1``), a WARM_START resumes the loop
    (``P + L * fraction``), a MISS is the teacher (``M``). No encoder term: the
    reference line never runs the CP2 key encoder. Shares the record -- and
    therefore the denominator ``M`` -- with the CP2 arm being compared.
    """
    def cost(hit_type: str, start_t) -> float:
        if hit_type == "FULL_HIT":
            return record.stage1_ms
        if hit_type == "MISS":
            return record.teacher_forward_ms
        if hit_type == "WARM_START":
            return record.prefix_ms + record.stage3_full_loop_ms * record.stage3_fraction(start_t)
        raise ValueError(f"unknown hit_type {hit_type!r}")
    return cost


GROOT_REFERENCE_PROTOCOL = "libero_groot_rit_arms_v1"


def check_groot_reference(ref_record: dict | None, *, suite: str, ref_arms: list[str]) -> dict:
    """The GR00T reference line's consumer contract (plan §3.10, G2-B4).

    The reference run (``exp/libero_groot`` RIT arms) must come with its arm
    record: same suite, the k=8 teacher, and every arm found in the reference
    ledger declared there. Its own interim pricing is *not* used -- both lines
    are re-priced from raw verdicts under the CP2 line's measured table, each
    along its own execution path -- so that record is reported, never summed.
    """
    if not ref_record:
        raise SystemExit("a GR00T comparison needs --ref-record (the reference line's arm record)")
    problems = []
    if ref_record.get("protocol") != GROOT_REFERENCE_PROTOCOL:
        problems.append(f"protocol {ref_record.get('protocol')!r} != {GROOT_REFERENCE_PROTOCOL!r}")
    if ref_record.get("suite") != suite:
        problems.append(f"suite {ref_record.get('suite')!r} != {suite!r}")
    if int(ref_record.get("denoising_steps") or 0) != 8:
        problems.append(f"denoising_steps {ref_record.get('denoising_steps')!r} != 8 (the CP2 line's teacher)")
    declared = set(ref_record.get("arms") or {})
    undeclared = sorted(a for a in ref_arms if a not in declared)
    if undeclared:
        problems.append(f"reference ledger arms not in the arm record: {undeclared[:5]}")
    if problems:
        raise SystemExit("reference record rejected: " + "; ".join(problems))
    return {"protocol": ref_record["protocol"], "suite": ref_record["suite"],
            "denoising_steps": int(ref_record["denoising_steps"]),
            "interim_cost_not_used": ref_record.get("cost"), "n_arms": len(declared)}


def compare(cp2_run_dir: str, ref_run_dir: str, *, export_record: dict,
            expect_episodes: int, allow_partial: bool, B: int, seed: int,
            arms: list[str] | None = None, ref_record: dict | None = None) -> dict:
    """ΔSR of every CP2 arm against the reference frontier at the arm's realised IR.

    Both ledgers are priced with the CP2 line's cost record: the CP2 arms
    through ``cp2_verdict_cost`` (with ``E``), the reference through
    ``cp1_reference_cost`` (no ``E``), over the same teacher denominator.
    """
    # A comparison is a publishable result, so it must not provide a side door
    # around the plan §3.11 completeness, provenance, and tier-purity gates.
    checked = aggregate_cp2_run(
        cp2_run_dir,
        expect_episodes=expect_episodes,
        allow_partial=allow_partial,
        export_record=export_record,
    )
    cp2 = checked["ledger"]
    prof, pricing = pricing_for(export_record, suite=checked["suite"])
    record = pricing[prof.default_cost_table]
    reference_binding = None
    if prof.name == libs.PI05.name:
        # The Pi0.5 reference (exp/rit_pareto) keeps its own frozen cost authority.
        from exp.rit_pareto.aggregate_rit import decision_cost as ref_cost
    else:
        priced = cp1_reference_cost(record)

        def ref_cost(hit_type, start_t, _priced=priced):
            try:
                return _priced(hit_type, start_t)
            except ValueError as exc:
                raise SystemExit(f"{ref_run_dir}: reference verdict {hit_type}@{start_t} is not priceable under "
                                 f"{record.schedule_id}: {exc}") from exc
    ref = stats.load_episode_ledger(ref_run_dir, ref_cost)
    if prof.name != libs.PI05.name:
        reference_binding = check_groot_reference(ref_record, suite=checked["suite"], ref_arms=sorted(ref))
    miss_ms = libs.teacher_forward_cost(record)
    out: dict[str, dict] = {}
    for arm, eps in cp2.items():
        if arms and arm not in arms:
            continue
        out[arm] = stats.bootstrap_frontier_delta(eps, ref, miss_ms=miss_ms, B=B, seed=seed)
    return {"protocol": libs.PROTOCOL, "cp2_run_dir": str(pathlib.Path(cp2_run_dir).resolve()),
            "ref_run_dir": str(pathlib.Path(ref_run_dir).resolve()), "B": B, "seed": seed,
            "teacher": prof.name, "suite": checked["suite"], "cost": libs.cost_record_summary(record, prof),
            "reference_binding": reference_binding,
            "reference_arms": sorted(ref), "cp2_audit": checked["audit"], "comparisons": out}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cp2-run-dir", required=True)
    ap.add_argument("--ref-run-dir", required=True)
    ap.add_argument("--export-record", required=True,
                    help="CP2 export_record.json; binds the run's arms and library digest")
    ap.add_argument("--ref-record", default="",
                    help="GR00T: the reference line's arm record (exp/libero_groot config/rit/<suite>/arm_record.json)")
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
    ref_record = json.loads(pathlib.Path(args.ref_record).read_text(encoding="utf-8")) if args.ref_record else None
    res = compare(
        args.cp2_run_dir,
        args.ref_run_dir,
        export_record=export_record,
        expect_episodes=args.expect_episodes,
        allow_partial=args.allow_partial,
        B=args.B,
        seed=args.seed,
        arms=arms,
        ref_record=ref_record,
    )
    res["export_record"] = str(pathlib.Path(args.export_record).resolve())
    res["ref_record"] = str(pathlib.Path(args.ref_record).resolve()) if args.ref_record else None
    libs.dump_json(args.out, res)
    for arm, r in res["comparisons"].items():
        d = r["delta_sr"]
        print(f"{arm}: IR={r['ir_cp2']:.2f} SR={r['sr_cp2']:.3f} ref={r['sr_reference_at_ir']} "
              f"dSR={None if d is None else round(d, 4)} ci={r['delta_ci95']} miss={r['support_miss_frac']:.3f} -> {r['decision']}")


if __name__ == "__main__":
    main()
