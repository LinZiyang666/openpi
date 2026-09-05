"""Aggregate one CP2 arm group: per-arm SR / IR, purity and completeness gates.

Reads the conductor ``journal.jsonl`` + ``per_step.jsonl`` of a run directory,
prices every accepted decision with the CP2 tier costs (both cost tables) and
applies the fail-closed gates of plan §3.5 / §3.11:

- completeness (``stats.audit_run``): terminal rows == unique uid (0 dup),
  per_step ``(uid, attempt)`` == journal ``(uid, attempt)``, no truncated /
  short ``failed`` episode, arm set == export record, every verdict row's
  ``library_sha256`` == the export record's, every arm at
  ``--expect-episodes`` accepted episodes (``--allow-partial`` relaxes only
  this last count);
- every priced row must carry ``checkpoint == "CP2"`` (older servers or CP1
  arms are rejected instead of mis-priced);
- tier purity: an ``n0`` arm has no WARM_START row, an ``n1`` arm has no
  FULL_HIT row.

Writes ``aggregate.json`` (per arm: n_ep, success_rate, wilson95, decisions,
counts, ir_percent, ir_percent_eager, tier, target label, the audit summary)
and ``episodes.jsonl`` (the per-episode ledger the bootstrap reads).

Usage:
  uv run python -m exp.actioncache_baseline.aggregate --run-dir <dir> \\
      --export-record <export_record.json> --out <aggregate.json> [--suite <suite>]
"""

from __future__ import annotations

import argparse
import json
import pathlib

from exp.actioncache_baseline import libs, stats


def _suite_of_run(arms: list[str], suite: str | None) -> str:
    if suite:
        return suite
    tags = {p["suite_tag"] for p in (libs.parse_arm(a) for a in arms) if p}
    if len(tags) != 1:
        raise SystemExit(f"cannot infer the suite from arm ids {arms}; pass --suite")
    return libs.suite_from_tag(tags.pop())


def aggregate(run_dir: str | pathlib.Path, *, expect_episodes: int = 500,
              allow_partial: bool = False, export_record: dict | None = None,
              suite: str | None = None) -> dict:
    cg = lambda h, t: libs.cp2_tier_cost(h, t, "cuda_graph")  # noqa: E731
    eg = lambda h, t: libs.cp2_tier_cost(h, t, "eager")  # noqa: E731
    ledger = stats.load_episode_ledger(run_dir, cg, require_checkpoint="CP2")
    ledger_eager = stats.load_episode_ledger(run_dir, eg, require_checkpoint="CP2")
    arms_rec = (export_record or {}).get("arms", {})
    suite = _suite_of_run(sorted(ledger), suite)
    audit = stats.audit_run(
        run_dir, step_cap=libs.STEP_CAP[suite], min_hit_rows=libs.MIN_HIT_ROWS[suite],
        expect_episodes=expect_episodes, allow_partial=allow_partial,
        expected_arms=sorted(arms_rec) if export_record is not None else None,
        expected_library_sha256=(export_record or {}).get("library_sha256"),
    )
    problems = list(audit["problems"])
    out: dict[str, dict] = {}
    for yaml_id, eps in ledger.items():
        parsed = libs.parse_arm(yaml_id)
        tier = parsed["tier"] if parsed else None
        counts = {"FULL_HIT": 0, "WARM_START": 0, "MISS": 0}
        for e in eps:
            for k, v in e["counts"].items():
                counts[k.split("@")[0]] += v
                if k.startswith("WARM_START@"):
                    counts[k] = counts.get(k, 0) + v
        if tier == "n0" and counts["WARM_START"]:
            problems.append(f"{yaml_id}: n0 arm has {counts['WARM_START']} WARM_START rows (tier purity)")
        if tier == "n1" and counts["FULL_HIT"]:
            problems.append(f"{yaml_id}: n1 arm has {counts['FULL_HIT']} FULL_HIT rows (tier purity)")
        s = stats.summarize(eps, libs.miss_cost("cuda_graph"))
        s_e = stats.summarize(ledger_eager[yaml_id], libs.miss_cost("eager"))
        rec = arms_rec.get(yaml_id, {})
        out[yaml_id] = {
            **s, "ir_percent_eager": s_e["ir_percent"], "counts": counts, "tier": tier,
            "target": parsed["target"] if parsed else None,
            "target_ir": rec.get("target_ir"), "predicted_ir": rec.get("predicted_ir"),
            "theta_raw": rec.get("theta_raw"), "theta_norm": rec.get("theta_norm"),
            "ir_gap_realized": (s["ir_percent"] - rec["predicted_ir"]) if rec.get("predicted_ir") is not None else None,
            "label": f"IR={int(round(rec['target_ir']))}" if rec.get("target_ir") is not None else (parsed["target"] if parsed else yaml_id),
        }
    if problems:
        raise SystemExit("aggregate gates failed:\n  " + "\n  ".join(problems))
    return {"arms": out, "ledger": ledger, "audit": audit, "suite": suite}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--export-record", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--suite", default="", choices=["", *sorted(libs.SUITE_TAGS)])
    ap.add_argument("--expect-episodes", type=int, default=500)
    ap.add_argument("--allow-partial", action="store_true",
                    help="relax only the per-arm episode count; identity gates stay fail-closed")
    args = ap.parse_args()
    rec = json.loads(pathlib.Path(args.export_record).read_text(encoding="utf-8")) if args.export_record else None
    res = aggregate(args.run_dir, expect_episodes=args.expect_episodes,
                    allow_partial=args.allow_partial, export_record=rec, suite=args.suite or None)
    out = pathlib.Path(args.out)
    libs.dump_json(out, {"protocol": libs.PROTOCOL, "run_dir": str(pathlib.Path(args.run_dir).resolve()),
                         "suite": res["suite"], "cost_tables": libs.COST_TABLES, "arms": res["arms"],
                         "audit": res["audit"], "export_record": args.export_record or None})
    with out.with_name("episodes.jsonl").open("w", encoding="utf-8") as fh:
        for yaml_id, eps in res["ledger"].items():
            for e in eps:
                fh.write(json.dumps({"yaml_id": yaml_id, **{k: v for k, v in e.items() if k != "counts"},
                                     "counts": dict(e["counts"])}) + "\n")
    for arm, r in res["arms"].items():
        print(f"{arm}: n={r['n_ep']} SR={r['success_rate']:.3f} [{r['wilson95'][0]:.3f},{r['wilson95'][1]:.3f}] "
              f"IR={r['ir_percent']:.2f}% (eager {r['ir_percent_eager']:.2f}%) counts={r['counts']}")


if __name__ == "__main__":
    main()
