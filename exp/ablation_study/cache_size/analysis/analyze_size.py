"""End-to-end analysis for the cache-size ablation (plan §8).

Consumes the **conductor Journal contract** -- ``task_uid`` / ``yaml_id`` /
``phase`` / ``status`` / ``success``, where ``task_uid`` encodes
``<yaml_id>:<phase>:<task_id>:<episode_idx>`` -- through the shared,
accepted-attempt-aware reader in ``exp.common.conductor_journal``. Episode
identity and the "both terminal statuses stay in the denominator" rule are
centralized there and must not fork.

The completeness gate runs *before* any statistic. Task-level aggregation hides
missing episodes: a tier that lost 40 rollouts still yields a per-task success
rate, just a quieter one. So every arm, plus the teacher anchor, must present
the identical set of 500 ``(task_id, episode_idx)`` keys; anything missing,
duplicated or mismatched is fatal.

Reporting contract:

*   every cell reports **both** axes -- adequacy (Q, the main question) and
    direction (D) -- because either half alone is misleading;
*   cell assignment comes from Holm-adjusted rejections, never a raw CI. Where a
    CI would imply a different cell, the disagreement is printed, not resolved;
*   the plateau axis is labelled descriptive wherever it appears.
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
from dataclasses import asdict
from typing import Iterable, Mapping

from exp.ablation_study.cache_size.full_hit import (
    assert_full_hit_per_episode,
    load_per_episode_hits,
)
from exp.common.conductor_journal import load_accepted, success_map
from exp.ablation_study.cache_size.analysis.cache_size_decision import (
    DELTA,
    classify_d,
    classify_m,
    classify_p,
    classify_q,
    decide,
)
from exp.ablation_study.cache_size.analysis.cache_size_stats import (
    signflip_test,
    cluster_bootstrap_ci,
    holm,
    not_evaluable,
)

logger = logging.getLogger(__name__)

TIERS = ("S1", "S2", "S3", "S4", "S5", "S6")
EXPECTED_TASKS = 10
EXPECTED_EPISODES_PER_TASK = 50
EXPECTED_KEYS = EXPECTED_TASKS * EXPECTED_EPISODES_PER_TASK

Ledger = Mapping[tuple[int, int], bool]


def assert_complete_ledger(name: str, ledger: Ledger) -> None:
    """One arm's episode ledger must be exactly the frozen A-pool grid."""
    if len(ledger) != EXPECTED_KEYS:
        raise SystemExit(
            f"{name}: {len(ledger)} unique (task_id, episode_idx) keys, expected {EXPECTED_KEYS}. "
            "Task-level aggregation would hide the gap behind a plausible success rate."
        )
    tasks = sorted({t for t, _ in ledger})
    if tasks != list(range(EXPECTED_TASKS)):
        raise SystemExit(f"{name}: task ids {tasks}, expected 0..{EXPECTED_TASKS - 1}")
    for t in tasks:
        eps = sorted(e for tt, e in ledger if tt == t)
        if eps != list(range(EXPECTED_EPISODES_PER_TASK)):
            raise SystemExit(
                f"{name}: task {t} has episode indices {eps[:5]}... "
                f"({len(eps)} of {EXPECTED_EPISODES_PER_TASK})"
            )


def assert_keys_match(reference_name: str, reference: Ledger, name: str, other: Ledger) -> None:
    """Paired analysis needs identical identity sets, not merely equal counts."""
    a, b = set(reference), set(other)
    if a != b:
        missing, extra = sorted(a - b)[:5], sorted(b - a)[:5]
        raise SystemExit(
            f"{name} is not key-identical to {reference_name}: "
            f"{len(a - b)} missing (e.g. {missing}), {len(b - a)} extra (e.g. {extra}). "
            "Paired statistics require the same episodes on both sides."
        )


def task_rates(ledger: Ledger) -> dict[int, float]:
    by_task: dict[int, list[bool]] = {}
    for (task_id, _), ok in ledger.items():
        by_task.setdefault(task_id, []).append(ok)
    return {t: sum(v) / len(v) for t, v in sorted(by_task.items())}


def paired_diff(a: dict[int, float], b: dict[int, float]) -> list[float]:
    """d_t over tasks. Both sides must already be complete and key-identical."""
    if set(a) != set(b):
        raise SystemExit(f"task sets differ: {sorted(set(a) ^ set(b))}")
    return [a[t] - b[t] for t in sorted(a)]


def load_teacher_anchor(anchor_dir: str | pathlib.Path) -> dict[tuple[int, int], bool]:
    """Teacher anchor rows -> the same (task_id, episode_idx) key space.

    The anchor records ``init_state_idx``; under the official A-pool protocol the
    episode runner walks those 50 inits in order, so ``episode_idx`` and
    ``init_state_idx`` denote the same slot. The completeness gate below would
    catch any drift in that correspondence.
    """
    rows: list[dict] = []
    for path in sorted(pathlib.Path(anchor_dir).glob("*.json")):
        rows.extend(json.loads(path.read_text()))
    ledger: dict[tuple[int, int], bool] = {}
    for r in rows:
        key = (int(r["task_id"]), int(r["init_state_idx"]))
        if key in ledger:
            raise SystemExit(f"teacher anchor has duplicate entry for {key}")
        ledger[key] = bool(r["success"])
    return ledger


def verify_launch_binding(
    launch_path: str | pathlib.Path,
    *,
    suite: str,
    arms: set[str],
    apool_digest_expected: str,
    trials_per_task: int = EXPECTED_EPISODES_PER_TASK,
) -> dict:
    """Check the run was launched against the frozen A-pool, and carry it forward.

    ``apool_digest_expected`` is required, not optional. Without it the only
    thing checkable is that the launch record contains *a* digest -- which it
    wrote itself, so it attests nothing. The expected value has to come from
    outside the run: the frozen A-pool record under version control.

    ``smoke`` and ``trials_per_task`` are checked for the same reason the arm set
    is checked by equality rather than containment: a subset run, a 5-trial
    shakedown and the reported 8-arm x 500-episode run all produce well-formed
    launch records, and only these fields tell them apart.
    """
    record = json.loads(pathlib.Path(launch_path).read_text())
    if record.get("suite") != suite:
        raise SystemExit(
            f"launch record is for suite {record.get('suite')!r}, analysis for {suite!r}"
        )
    if record.get("smoke"):
        raise SystemExit(
            "launch record is from a --smoke run; smoke runs relax the A-pool binding "
            "and the completeness gates, so their episodes cannot back a reported result"
        )
    launched = set(record.get("arms", []))
    if launched != arms:
        raise SystemExit(
            f"launched arms != analysed arms: only launched {sorted(launched - arms)}, "
            f"only analysed {sorted(arms - launched)}. The formal matrix is the 8 "
            "pre-registered arms per suite."
        )
    if record.get("trials_per_task") != trials_per_task:
        raise SystemExit(
            f"launch record declares {record.get('trials_per_task')!r} trials per task, "
            f"expected {trials_per_task} (the official A-pool protocol)"
        )
    apool = record.get("apool")
    if not apool or not apool.get("rollup_sha256"):
        raise SystemExit(
            "launch record carries no A-pool digest; the run was not bound to the "
            "frozen evaluation pool and its 500 episodes cannot be attested"
        )
    if apool["rollup_sha256"] != apool_digest_expected:
        raise SystemExit(
            f"A-pool digest mismatch: run used {apool['rollup_sha256']}, "
            f"expected {apool_digest_expected}"
        )
    return record


# The sensitivity arms' equivalence margin. Deliberately separate from DELTA:
# this asks "did the normalizer choice move the answer?", not "is replay good
# enough?", and it is descriptive -- it never enters the Holm family.
RECAL_EQUIV_MARGIN = 0.03


def analyze_sensitivity(
    main_rates: dict[int, float],
    recal_rates: dict[int, float],
    *,
    tier: str,
    b: int = 10_000,
    seed: int = 0,
) -> dict:
    """Compare a tier's arm against its recalibrated twin (descriptive).

    Branch N of the decision tree tells the reader to adjudicate the
    "normalization mismatch" explanation here, so the arms need a consumer that
    states the equivalence claim explicitly rather than reading it off a CI that
    happens to contain zero.
    """
    d = paired_diff(recal_rates, main_rates)
    point = sum(d) / len(d)
    ci = cluster_bootstrap_ci(d, b=b, seed=seed)
    # Two one-sided tests at +/- margin: equivalence needs BOTH rejected.
    lower = signflip_test(d, h0_center=-RECAL_EQUIV_MARGIN, side="greater",
                          name=f"recal_{tier}_lower")
    upper = signflip_test(d, h0_center=RECAL_EQUIV_MARGIN, side="less",
                          name=f"recal_{tier}_upper")
    equivalent = (lower.p <= 0.05 and upper.p <= 0.05)
    return {
        "tier": tier,
        "delta_sr": point,
        "ci": asdict(ci),
        "tost_lower_p": lower.p,
        "tost_upper_p": upper.p,
        "margin": RECAL_EQUIV_MARGIN,
        "equivalent": equivalent,
        "interpretation": (
            "normalizer choice does not move the answer at this tier "
            f"(|delta| within +/-{RECAL_EQUIV_MARGIN:.0%})"
            if equivalent else
            "normalizer choice is NOT shown equivalent here; the main curve's wording "
            "must be qualified as holding under the production calibration"
        ),
        "note": "descriptive: not part of the pre-registered Holm family",
    }


def analyze_suite(
    *,
    tier_rates: dict[str, dict[int, float]],
    teacher_rates: dict[int, float],
    degenerate_pairs: Iterable[str] = (),
    b: int = 10_000,
    seed: int = 0,
) -> dict:
    """Run the eight-test family and resolve the tree for one suite."""
    degenerate = set(degenerate_pairs)
    tests = []
    adjacent_points: dict[str, float] = {}

    for i, (prev, cur) in enumerate(zip(TIERS, TIERS[1:]), start=1):
        name = f"t{i}_{prev}_{cur}"
        d = paired_diff(tier_rates[cur], tier_rates[prev])
        adjacent_points[name] = sum(d) / len(d)
        if f"{prev}-{cur}" in degenerate:
            tests.append(not_evaluable(
                name, "two-sided", 0.0, len(d),
                f"tiers {prev} and {cur} selected identical trajectories",
            ))
        else:
            tests.append(signflip_test(d, h0_center=0.0, side="two-sided",
                                       name=name))

    gap = paired_diff(teacher_rates, tier_rates["S6"])
    gap_point = sum(gap) / len(gap)
    # The three gap tests share a reference distribution *identically*, with no
    # seed to coordinate: the pattern set is enumerated, and h0 cancels inside
    # t* = (mean(h0 + r*w) - h0) / (sd(h0 + r*w)/sqrt(n)) = mean(r*w)/(sd(r*w)/sqrt(n)).
    # So the t* array is bit-identical across h0 values and the nesting between
    # the two-sided and one-sided tails is exact by construction. Under the
    # sampled bootstrap this had to be bought with a shared seed, and even then
    # p7 <= p6 was violated ~2 in 3000 -- both times inside the Holm-critical band.
    tests.append(signflip_test(gap, h0_center=0.0, side="two-sided", name="t6_direction"))
    tests.append(signflip_test(gap, h0_center=DELTA, side="less", name="t7_noninferior"))
    tests.append(signflip_test(gap, h0_center=DELTA, side="greater", name="t8_inferior"))

    hr = holm(tests, alpha=0.05)

    slopes = {}
    for prev, cur in zip(TIERS, TIERS[1:]):
        d = paired_diff(tier_rates[cur], tier_rates[prev])
        slopes[f"{prev}-{cur}"] = cluster_bootstrap_ci(d, b=b, seed=seed)
    gap_ci = cluster_bootstrap_ci(gap, b=b, seed=seed)

    d_axis = classify_d(test6_rejected=hr.rejected["t6_direction"], gap_point=gap_point)
    q_axis = classify_q(
        test7_rejected=hr.rejected["t7_noninferior"],
        test8_rejected=hr.rejected["t8_inferior"],
    )
    # A slope CI that collapsed for a degeneracy reason carries no information
    # about the plateau; treat it as inconclusive rather than as a tight zero.
    plateau_degenerate = any(
        "denominator" in slopes[k].reason for k in ("S4-S5", "S5-S6")
    )
    p_axis = classify_p(
        slope5_ci_hi=slopes["S4-S5"].hi,
        slope6_ci_lo=slopes["S5-S6"].lo,
        slope6_ci_hi=slopes["S5-S6"].hi,
        degenerate=plateau_degenerate,
    )
    m_yes = classify_m(
        {t.name: hr.rejected[t.name] for t in tests[:5]},
        adjacent_points,
    )
    verdict = decide(
        d=d_axis, q=q_axis, p=p_axis, m_yes=m_yes,
        equivalence_note=(q_axis == "Q-pass" and gap_ci.lo > -DELTA),
    )

    disagreements = []
    if (gap_ci.hi < DELTA) != hr.rejected["t7_noninferior"]:
        disagreements.append(
            f"gap CI upper bound {gap_ci.hi:+.4f} vs delta {DELTA} disagrees with the "
            f"Holm-adjusted non-inferiority decision (rejected={hr.rejected['t7_noninferior']}); "
            "the family gate governs"
        )
    if (gap_ci.lo > 0 or gap_ci.hi < 0) != hr.rejected["t6_direction"]:
        disagreements.append(
            f"gap CI [{gap_ci.lo:+.4f}, {gap_ci.hi:+.4f}] vs the Holm-adjusted direction "
            f"decision (rejected={hr.rejected['t6_direction']}); the family gate governs"
        )

    # Per-tier overall SR (task-equal-weight; every task has the same 50
    # episodes here, so this equals the episode-weighted rate) plus its CI --
    # the plotting step's inputs, so it does not have to reshape by hand.
    tier_sr = {t: sum(r.values()) / len(r) for t, r in tier_rates.items()}
    tier_ci = {
        t: [cluster_bootstrap_ci(list(r.values()), b=b, seed=seed).lo,
            cluster_bootstrap_ci(list(r.values()), b=b, seed=seed).hi]
        for t, r in tier_rates.items()
    }

    return {
        "family_size": len(tests),
        "tier_sr": tier_sr,
        "tier_ci": tier_ci,
        "teacher_sr": sum(teacher_rates.values()) / len(teacher_rates),
        "tests": [asdict(t) for t in tests],
        "holm_adjusted": hr.adjusted,
        "holm_rejected": hr.rejected,
        "gap_point": gap_point,
        "gap_ci": asdict(gap_ci),
        "slopes": {k: asdict(v) for k, v in slopes.items()},
        "axes": {"D": d_axis, "Q": q_axis, "P": p_axis, "M_yes": m_yes},
        "plateau_degenerate": plateau_degenerate,
        "degenerate_resample_fractions": {
            t.name: t.degenerate_resample_fraction for t in tests
        },
        "verdict": asdict(verdict),
        "ci_gate_disagreements": disagreements,
        "not_evaluable": [t.name for t in tests if not t.evaluable],
    }


def render(result: dict, suite: str) -> str:
    v = result["verdict"]
    lines = [
        f"# cache-size ablation — {suite}",
        "",
        f"**Branch {v['branch']}** — {v['headline']}",
        "",
        f"- adequacy (Q, main axis): `{v['q']}`",
        f"- direction (D): `{v['d']}`",
        f"- plateau (P, **descriptive**): `{v['p']}`",
        f"- monotonicity violated (M): `{v['m_yes']}`",
        "",
        f"gap = teacher − S6 = {result['gap_point']:+.4f} "
        f"(95% {result['gap_ci']['method']} CI "
        f"[{result['gap_ci']['lo']:+.4f}, {result['gap_ci']['hi']:+.4f}])",
        "",
        f"## Pre-registered family ({result['family_size']} tests, Holm)",
        "",
        "| test | p | Holm p | rejected |",
        "|---|---|---|---|",
    ]
    for t in result["tests"]:
        name = t["name"]
        flag = "" if t["evaluable"] else " *(not evaluable)*"
        lines.append(
            f"| `{name}`{flag} | {t['p']:.4f} | {result['holm_adjusted'][name]:.4f} | "
            f"{'yes' if result['holm_rejected'][name] else 'no'} |"
        )
    binding = result.get("launch_binding") or {}
    lines += ["", "## Provenance", "",
              f"- A-pool rollup: `{binding.get('apool_rollup_sha256') or 'UNBOUND'}`",
              f"- A-pool dir: `{binding.get('apool_dir') or 'n/a'}`"]
    if binding.get("smoke"):
        lines.append("- ⚠ **smoke mode** -- completeness, binding and sensitivity "
                     "requirements were relaxed; not a reportable run")
    witness = result.get("full_hit_witness") or {}
    if witness:
        lines += ["", "## FULL_HIT witness (per accepted episode)", ""]
        lines += [f"- `{a}`: {v['episodes']} episodes / {v['steps']} steps, all FULL_HIT"
                  for a, v in sorted(witness.items())]
    if result.get("sensitivity"):
        lines += ["", "## Normalizer sensitivity (descriptive)", "",
                  "| tier | ΔSR (recal − main) | 95% CI | equivalent at ±3pp |",
                  "|---|---|---|---|"]
        for s_ in result["sensitivity"]:
            lines.append(
                f"| {s_['tier']} | {s_['delta_sr']:+.4f} | "
                f"[{s_['ci']['lo']:+.4f}, {s_['ci']['hi']:+.4f}] | "
                f"{'yes' if s_['equivalent'] else 'NO'} |"
            )
        for s_ in result["sensitivity"]:
            if not s_["equivalent"]:
                lines += ["", f"- ⚠ {s_['tier']}: {s_['interpretation']}"]

    lines += ["", "## Caveats", ""]
    lines += [f"- {c}" for c in v["caveats"]]
    if result["ci_gate_disagreements"]:
        lines += ["", "## CI / family-gate disagreements", ""]
        lines += [f"- {d}" for d in result["ci_gate_disagreements"]]
    if result["not_evaluable"]:
        lines += ["", "## Not evaluable", ""]
        lines += [f"- `{n}` (kept in the family at p=1; NOT the same as 'no difference')"
                  for n in result["not_evaluable"]]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--journal", required=True, action="append",
                    help="conductor journal jsonl (repeatable)")
    ap.add_argument("--teacher-anchor", required=True)
    ap.add_argument("--suite", required=True)
    ap.add_argument("--per-step", required=True,
                    help="per-step jsonl; the per-episode FULL_HIT witness")
    ap.add_argument("--launch-record", required=True,
                    help="<per-step>.launch.json written by run_size_eval")
    ap.add_argument("--apool-digest", default=None,
                    help="expected A-pool rollup sha256, from the frozen record under "
                         "version control. Required unless --smoke: a launch record "
                         "cannot attest its own digest.")
    ap.add_argument("--grid", default=None)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--smoke", action="store_true",
                    help="subset/unbound mode: relaxes the formal completeness and "
                         "sensitivity-arm requirements. NEVER for the reported run.")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO)

    arms = load_accepted(args.journal)

    tier_arm_names = {t: f"cache_size_{args.suite}_{t}" for t in TIERS}
    recal_arm_names = {t: f"cache_size_{args.suite}_{t}_recal" for t in ("S1", "S6")}

    tier_ledgers: dict[str, dict[tuple[int, int], bool]] = {}
    # uid -> the attempt the scheduler accepted. The per-step join needs both
    # halves of that key; a uid-only join lets a stale attempt's rows witness an
    # accepted episode that produced none.
    accepted_attempts: dict[str, dict[str, int | None]] = {}
    for tier, arm in tier_arm_names.items():
        if arm not in arms:
            raise SystemExit(f"journal has no accepted rows for arm {arm!r}; "
                             f"present: {sorted(arms)}")
        ledger = success_map(arms[arm])
        if not args.smoke:
            assert_complete_ledger(arm, ledger)
        tier_ledgers[tier] = ledger
        accepted_attempts[arm] = {uid: r.attempt for uid, r in arms[arm].items()}

    reference_arm = tier_arm_names["S1"]
    if not args.smoke:
        for tier, ledger in tier_ledgers.items():
            assert_keys_match(reference_arm, tier_ledgers["S1"],
                              tier_arm_names[tier], ledger)

    teacher = load_teacher_anchor(args.teacher_anchor)
    if not args.smoke:
        assert_complete_ledger("teacher anchor", teacher)
        assert_keys_match(reference_arm, tier_ledgers["S1"], "teacher anchor", teacher)

    # Sensitivity arms are part of the formal matrix, not an optional extra:
    # branch N cannot adjudicate the normalization explanation without them.
    sensitivity_ledgers: dict[str, dict[tuple[int, int], bool]] = {}
    for tier, arm in recal_arm_names.items():
        if arm not in arms:
            if args.smoke:
                logger.warning("smoke mode: sensitivity arm %s absent", arm)
                continue
            raise SystemExit(
                f"journal has no accepted rows for sensitivity arm {arm!r}. The formal "
                "matrix is 8 arms per suite; without both recal arms branch N cannot "
                "adjudicate the normalization-mismatch explanation. Use --smoke only "
                "for subset runs that are never reported."
            )
        ledger = success_map(arms[arm])
        if not args.smoke:
            assert_complete_ledger(arm, ledger)
            assert_keys_match(reference_arm, tier_ledgers["S1"], arm, ledger)
        sensitivity_ledgers[tier] = ledger
        accepted_attempts[arm] = {uid: r.attempt for uid, r in arms[arm].items()}

    # Bind the analysis to the launch record, and thereby to the frozen A-pool.
    expected_arms = set(tier_arm_names.values()) | set(accepted_attempts)
    launch = None
    if args.smoke:
        logger.warning("smoke mode: launch/A-pool binding not enforced")
    else:
        if not args.apool_digest:
            raise SystemExit(
                "--apool-digest is required for a formal analysis: the launch record "
                "writes its own digest, so checking it against itself attests nothing. "
                "Pass the rollup from the frozen A-pool record under version control."
            )
        launch = verify_launch_binding(
            args.launch_record, suite=args.suite,
            arms=expected_arms, apool_digest_expected=args.apool_digest,
        )

    # Per-episode FULL_HIT witness, joined on (task_uid, accepted attempt).
    per_ep, _row_counts = load_per_episode_hits(args.per_step, require_attempt=not args.smoke)
    hit_summary = {}
    for arm, attempts in accepted_attempts.items():
        arm_hits = per_ep.get(arm, {})
        if args.smoke:
            arm_hits = {k: h for k, h in arm_hits.items() if k[0] in attempts}
            attempts = {uid: att for (uid, att) in arm_hits}
        hit_summary[arm] = assert_full_hit_per_episode(arm, attempts, arm_hits)

    degenerate = []
    if args.grid:
        import yaml

        degenerate = yaml.safe_load(pathlib.Path(args.grid).read_text()).get(
            "degenerate_pairs", []
        )

    tier_rate_map = {t: task_rates(l) for t, l in tier_ledgers.items()}
    result = analyze_suite(
        tier_rates=tier_rate_map,
        teacher_rates=task_rates(teacher),
        degenerate_pairs=degenerate,
    )
    result["full_hit_witness"] = hit_summary
    result["launch_binding"] = {
        "apool_rollup_sha256": (launch or {}).get("apool", {}).get("rollup_sha256"),
        "apool_dir": (launch or {}).get("apool", {}).get("apool_dir"),
        "smoke": bool(args.smoke),
    }

    sensitivity = [
        analyze_sensitivity(tier_rate_map[tier], task_rates(ledger), tier=tier)
        for tier, ledger in sorted(sensitivity_ledgers.items())
    ]
    result["sensitivity"] = sensitivity

    pathlib.Path(args.out_json).write_text(json.dumps(result, indent=2))
    md = render(result, args.suite)
    pathlib.Path(args.out_md).write_text(md)
    print(md)


if __name__ == "__main__":
    main()
