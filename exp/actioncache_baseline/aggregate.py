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
  FULL_HIT row, and every WARM_START resumes at the teacher's N_hit=1
  snapshot (Pi0.5 0.1 / GR00T 0.875).

Pricing follows the export record (plan §3.5): a Pi0.5 record (or a legacy
record without ``teacher``) is priced with both frozen tables; a GR00T record
carries the bound cost summary (teacher table + this suite's encoder cost
``E``) and is priced with that alone -- never with a Pi0.5 table and never
with ``E = 0``. IR = sum(verdict cost) / (N * teacher forward cost).

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
    tags = {p["suite_tag"] for p in (libs.parse_arm(a) for a in arms) if p}
    if not tags and suite in libs.SUITE_TAGS:
        return suite
    if len(tags) != 1:
        raise SystemExit(f"cannot infer one suite from arm ids {arms}")
    inferred = libs.suite_from_tag(tags.pop())
    if suite is not None and suite != inferred:
        raise SystemExit(f"explicit suite {suite!r} differs from arm suite {inferred!r}")
    return inferred


def pricing_for(export_record: dict | None, *, suite: str | None = None) -> tuple[libs.TeacherProfile, dict[str, libs.CostRecord]]:
    """(profile, {table: CostRecord}) the run is priced with, from its export record.

    For GR00T the record's bound cost summary is re-built and, when the run's
    ``suite`` is known, must name that suite (the encoder cost ``E`` is
    measured per suite) and the record's own ``suite`` must agree (G2-B4).
    """
    rec = export_record or {}
    teacher = rec.get("teacher", libs.PI05.name)
    prof = libs.profile(teacher)
    if prof.name == libs.PI05.name:
        return prof, {t: libs.pi05_cost_record(t) for t in prof.cost_tables}
    if "cost" not in rec:
        raise SystemExit(f"a {prof.name} export record must carry its bound cost summary")
    cost = libs.cost_record_from_summary(rec["cost"])
    if cost.teacher != prof.name or cost.encoder_ms <= 0.0:
        raise SystemExit(f"{prof.name} cost summary is not bound (teacher={cost.teacher}, E={cost.encoder_ms})")
    if rec.get("suite") != cost.provenance["suite"]:
        raise SystemExit("export record suite differs from the bound encoder cost")
    if (rec.get("library_model") or {}).get("weights_digest") != cost.provenance["model"]["weights_digest"]:
        raise SystemExit("encoder cost model differs from the export library model")
    if (rec.get("projection") or {}).get("layout") != cost.provenance["layout"]:
        raise SystemExit("encoder cost layout differs from the export library projection")
    if suite is not None:
        if rec.get("suite") != suite:
            raise SystemExit(f"export record suite {rec.get('suite')!r} != run suite {suite!r}")
        if cost.provenance.get("suite") != suite:
            raise SystemExit(f"cost summary was bound for suite {cost.provenance.get('suite')!r}, the run is {suite!r}")
    if cost.schedule_id != prof.denoise_schedule or rec.get("schedule_id") not in (None, prof.denoise_schedule):
        raise SystemExit(f"cost summary / export record schedule is not {prof.denoise_schedule!r}")
    return prof, {cost.table: cost}


def aggregate(run_dir: str | pathlib.Path, *, expect_episodes: int = 500,
              allow_partial: bool = False, export_record: dict | None = None,
              suite: str | None = None) -> dict:
    """Per-arm SR / IR of one run directory behind the completeness, provenance and tier-purity gates.

    ``export_record`` (the emitter's ``export_record.json``) fixes the arm set,
    the library digest and -- for GR00T -- the bound pricing; ``suite`` is
    inferred from the arm ids unless given.
    """
    prof, _ = pricing_for(export_record)
    main_table = prof.default_cost_table

    def _priced(rec: libs.CostRecord):
        def cost_fn(hit_type, start_t):
            try:
                return libs.cp2_verdict_cost(rec, hit_type, start_t)
            except ValueError as exc:
                # e.g. a WARM_START at a timestep the teacher's loop cannot resume from.
                raise SystemExit(f"{run_dir}: unpriceable verdict {hit_type}@{start_t} under {prof.name}: {exc}") from exc
        return cost_fn

    # The suite is read off the arm ids first so the GR00T pricing can be
    # bound to it before any decision is priced.
    probe = stats.load_episode_ledger(run_dir, lambda h, t: 0.0, require_checkpoint="CP2")
    suite = _suite_of_run(sorted(probe), suite)
    prof, pricing = pricing_for(export_record, suite=suite)
    ledgers = {
        table: stats.load_episode_ledger(run_dir, _priced(rec), require_checkpoint="CP2")
        for table, rec in pricing.items()
    }
    ledger = ledgers[main_table]
    arms_rec = (export_record or {}).get("arms", {})
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
        warm_keys = [k for k in counts if k.startswith("WARM_START@")]
        expected_warm = f"WARM_START@{prof.warm_start_t:g}"
        if any(k != expected_warm for k in warm_keys):
            problems.append(f"{yaml_id}: WARM_START rows at {warm_keys} but {prof.name} N_hit=1 is {expected_warm}")
        s = stats.summarize(eps, libs.teacher_forward_cost(pricing[main_table]))
        extra = {
            f"ir_percent_{table}": stats.summarize(ledgers[table][yaml_id], libs.teacher_forward_cost(rec_t))["ir_percent"]
            for table, rec_t in pricing.items() if table != main_table
        }
        rec = arms_rec.get(yaml_id, {})
        out[yaml_id] = {
            **s, **extra, "counts": counts, "tier": tier,
            "target": parsed["target"] if parsed else None,
            "target_ir": rec.get("target_ir"), "predicted_ir": rec.get("predicted_ir"),
            "theta_raw": rec.get("theta_raw"), "theta_norm": rec.get("theta_norm"),
            "ir_gap_realized": (s["ir_percent"] - rec["predicted_ir"]) if rec.get("predicted_ir") is not None else None,
            "label": f"IR={int(round(rec['target_ir']))}" if rec.get("target_ir") is not None else (parsed["target"] if parsed else yaml_id),
        }
    if problems:
        raise SystemExit("aggregate gates failed:\n  " + "\n  ".join(problems))
    return {"arms": out, "ledger": ledger, "audit": audit, "suite": suite, "teacher": prof.name,
            "schedule_id": prof.denoise_schedule,
            "cost": {t: libs.cost_record_summary(r, prof) for t, r in pricing.items()}}


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
                         "suite": res["suite"], "teacher": res["teacher"], "cost": res["cost"],
                         "arms": res["arms"], "audit": res["audit"],
                         "export_record": args.export_record or None})
    with out.with_name("episodes.jsonl").open("w", encoding="utf-8") as fh:
        for yaml_id, eps in res["ledger"].items():
            for e in eps:
                fh.write(json.dumps({"yaml_id": yaml_id, **{k: v for k, v in e.items() if k != "counts"},
                                     "counts": dict(e["counts"])}) + "\n")
    for arm, r in res["arms"].items():
        eager = f" (eager {r['ir_percent_eager']:.2f}%)" if "ir_percent_eager" in r else ""
        print(f"{arm}: n={r['n_ep']} SR={r['success_rate']:.3f} [{r['wilson95'][0]:.3f},{r['wilson95'][1]:.3f}] "
              f"IR={r['ir_percent']:.2f}%{eager} counts={r['counts']}")


if __name__ == "__main__":
    main()
