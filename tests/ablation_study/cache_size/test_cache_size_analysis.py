"""Analysis wiring against the REAL conductor journal schema (plan §8.5, §10).

Every journal fixture here is written in the production contract -- ``task_uid``
(``<yaml_id>:<phase>:<task_id>:<episode_idx>``), ``yaml_id``, ``phase``,
``status``, ``success`` -- because an earlier version of these tests invented an
``arm``/``task_id`` schema and therefore exercised a reader that could never
consume a real run.
"""

from __future__ import annotations

import json

import pytest

from exp.ablation_study.cache_size.analysis.analyze_size import (
    EXPECTED_EPISODES_PER_TASK,
    EXPECTED_TASKS,
    TIERS,
    analyze_suite,
    assert_complete_ledger,
    assert_full_hit_per_episode,
    assert_keys_match,
    load_per_episode_hits,
    load_teacher_anchor,
    paired_diff,
    render,
    task_rates,
    verify_launch_binding,
)
from exp.common.conductor_journal import load_accepted, success_map

SUITE = "libero_spatial"


def _journal_rows(arm: str, *, success_of, tasks=EXPECTED_TASKS,
                  episodes=EXPECTED_EPISODES_PER_TASK, phase="eval",
                  attempt=None, accepted=None):
    for t in range(tasks):
        for e in range(episodes):
            row = {
                "task_uid": f"{arm}:{phase}:{t}:{e}",
                "yaml_id": arm,
                "phase": phase,
                "status": "done",
                "success": bool(success_of(t, e)),
            }
            if attempt is not None:
                row["attempt"] = attempt
            if accepted is not None:
                row["accepted"] = accepted
            yield row


def _write_journal(path, arms_spec):
    with open(path, "w") as f:
        for arm, fn in arms_spec.items():
            for row in _journal_rows(arm, success_of=fn):
                f.write(json.dumps(row) + "\n")
    return str(path)


def _ladder_fn(rate):
    """Deterministic per-(task, episode) success at approximately `rate`."""
    def fn(t, e):
        return ((t * 7 + e * 3) % 100) < int(round(rate * 100))
    return fn


def test_reads_the_real_journal_schema(tmp_path):
    arm = f"cache_size_{SUITE}_S1"
    p = _write_journal(tmp_path / "j.jsonl", {arm: _ladder_fn(0.30)})

    arms = load_accepted([p])
    assert arm in arms, "must key by yaml_id, not an invented 'arm' field"
    ledger = success_map(arms[arm])
    assert len(ledger) == EXPECTED_TASKS * EXPECTED_EPISODES_PER_TASK
    assert (0, 0) in ledger, "keys come from parse_task_uid, not a raw task_id column"


def test_stale_attempt_cannot_overwrite_the_accepted_result(tmp_path):
    """Journal.record journals a superseded dispatch exactly like the live one.

    A last-wins read lets a late stale row rewrite a success rate; only
    ``accepted`` distinguishes them.
    """
    arm = f"cache_size_{SUITE}_S1"
    uid = f"{arm}:eval:0:0"
    j = tmp_path / "j.jsonl"
    with j.open("w") as f:
        f.write(json.dumps({"task_uid": uid, "yaml_id": arm, "phase": "eval",
                            "status": "done", "success": True,
                            "attempt": 2, "accepted": True}) + "\n")
        # Arrives later, but belongs to a superseded dispatch.
        f.write(json.dumps({"task_uid": uid, "yaml_id": arm, "phase": "eval",
                            "status": "failed", "success": False,
                            "attempt": 1, "accepted": False}) + "\n")

    ledger = success_map(load_accepted([str(j)])[arm])
    assert ledger[(0, 0)] is True, "the accepted attempt must win regardless of order"


def test_conflicting_accepted_rows_are_fatal(tmp_path):
    arm = f"cache_size_{SUITE}_S1"
    uid = f"{arm}:eval:0:0"
    j = tmp_path / "j.jsonl"
    with j.open("w") as f:
        for ok in (True, False):
            f.write(json.dumps({"task_uid": uid, "yaml_id": arm, "phase": "eval",
                                "status": "done", "success": ok,
                                "attempt": 1, "accepted": True}) + "\n")

    with pytest.raises(ValueError, match="journal is inconsistent"):
        load_accepted([str(j)])


def test_complete_ledger_gate_catches_missing_episodes(tmp_path):
    arm = f"cache_size_{SUITE}_S1"
    with open(tmp_path / "j.jsonl", "w") as f:
        for row in _journal_rows(arm, success_of=_ladder_fn(0.5), episodes=49):
            f.write(json.dumps(row) + "\n")
    ledger = success_map(load_accepted([str(tmp_path / "j.jsonl")])[arm])

    with pytest.raises(SystemExit, match="expected 500"):
        assert_complete_ledger(arm, ledger)


def test_complete_ledger_gate_catches_missing_task(tmp_path):
    arm = f"cache_size_{SUITE}_S1"
    with open(tmp_path / "j.jsonl", "w") as f:
        for row in _journal_rows(arm, success_of=_ladder_fn(0.5), tasks=9):
            f.write(json.dumps(row) + "\n")
    ledger = success_map(load_accepted([str(tmp_path / "j.jsonl")])[arm])

    with pytest.raises(SystemExit, match="expected 500"):
        assert_complete_ledger(arm, ledger)


def test_key_mismatch_between_arms_is_fatal():
    a = {(t, e): True for t in range(10) for e in range(50)}
    b = dict(a)
    del b[(3, 7)]
    b[(3, 99)] = True  # same count, different identity
    with pytest.raises(SystemExit, match="not key-identical"):
        assert_keys_match("armA", a, "armB", b)


def test_teacher_anchor_missing_one_row_is_fatal(tmp_path):
    d = tmp_path / "anchor"
    d.mkdir()
    rows = [
        {"task_id": t, "init_state_idx": e, "success": True}
        for t in range(10) for e in range(50)
    ][:-1]  # drop exactly one
    (d / "a.json").write_text(json.dumps(rows))

    ledger = load_teacher_anchor(d)
    with pytest.raises(SystemExit, match="expected 500"):
        assert_complete_ledger("teacher anchor", ledger)


def test_teacher_anchor_duplicate_row_is_fatal(tmp_path):
    d = tmp_path / "anchor"
    d.mkdir()
    rows = [{"task_id": 0, "init_state_idx": 0, "success": True}] * 2
    (d / "a.json").write_text(json.dumps(rows))
    with pytest.raises(SystemExit, match="duplicate entry"):
        load_teacher_anchor(d)


def _write_per_step(path, arm, uids, hits_per_uid=3, bad_uid=None, attempt=1,
                    extra_rows=()):
    """Per-step rows shaped exactly like ``ConductorDriver.handle_result`` writes.

    That includes the ``attempt`` and ``step_idx`` stamps: fixtures that omit
    them cannot exercise the join the gate is actually made of, and this suite
    has been bitten before by fixtures that invented a shape production never
    produces.
    """
    with open(path, "w") as f:
        for uid in uids:
            for k in range(hits_per_uid):
                hit = "MISS" if (bad_uid == uid and k == 0) else "FULL_HIT"
                f.write(json.dumps({"task_uid": uid, "yaml_id": arm, "attempt": attempt,
                                    "step_idx": k, "hit_type": hit}) + "\n")
        for row in extra_rows:
            f.write(json.dumps(row) + "\n")
    return path


def _accepted(arm, uids, attempt=1):
    return {uid: attempt for uid in uids}


def test_per_episode_witness_requires_a_trace_for_every_episode(tmp_path):
    """One FULL_HIT row would satisfy an arm-level ratio; it must not satisfy this."""
    arm = "cache_size_x_S1"
    uids = {f"{arm}:eval:0:{e}" for e in range(5)}
    p = _write_per_step(tmp_path / "ps.jsonl", arm, sorted(uids)[:1])  # only one episode

    hits, _ = load_per_episode_hits(p)
    with pytest.raises(SystemExit, match="have no inference rows"):
        assert_full_hit_per_episode(arm, _accepted(arm, uids), hits[arm])


def test_per_episode_witness_flags_a_single_non_full_hit(tmp_path):
    arm = "cache_size_x_S1"
    uids = {f"{arm}:eval:0:{e}" for e in range(5)}
    bad = f"{arm}:eval:0:2"
    p = _write_per_step(tmp_path / "ps.jsonl", arm, sorted(uids), bad_uid=bad)

    hits, _ = load_per_episode_hits(p)
    with pytest.raises(SystemExit, match="non-FULL_HIT"):
        assert_full_hit_per_episode(arm, _accepted(arm, uids), hits[arm])


def test_per_episode_witness_rejects_evidence_outside_the_ledger(tmp_path):
    arm = "cache_size_x_S1"
    uids = {f"{arm}:eval:0:{e}" for e in range(3)}
    p = _write_per_step(tmp_path / "ps.jsonl", arm,
                        sorted(uids) + [f"{arm}:eval:9:9"])

    hits, _ = load_per_episode_hits(p)
    with pytest.raises(SystemExit, match="absent from the accepted ledger"):
        assert_full_hit_per_episode(arm, _accepted(arm, uids), hits[arm])


def test_per_episode_witness_passes_a_clean_arm(tmp_path):
    arm = "cache_size_x_S1"
    uids = {f"{arm}:eval:0:{e}" for e in range(4)}
    p = _write_per_step(tmp_path / "ps.jsonl", arm, sorted(uids))

    hits, _ = load_per_episode_hits(p)
    summary = assert_full_hit_per_episode(arm, _accepted(arm, uids), hits[arm])
    assert summary == {"episodes": 4, "steps": 12, "stale_rows_ignored": 0}


# ---------------------------------------------------------------------------
# The attempt half of the join key (G2-R5 B2)
# ---------------------------------------------------------------------------


def test_stale_attempt_rows_cannot_witness_an_accepted_episode(tmp_path):
    """The failure this join exists to stop.

    ``ConductorDriver.handle_result`` forwards per-step rows for a stale attempt
    exactly as it does for the live one. Here attempt 1 was superseded; the
    accepted attempt 2 produced no rows at all. A uid-only join sees a complete
    FULL_HIT trace and passes.
    """
    arm = "cache_size_x_S1"
    uids = [f"{arm}:eval:0:{e}" for e in range(3)]
    p = _write_per_step(tmp_path / "ps.jsonl", arm, uids, attempt=1)

    hits, _ = load_per_episode_hits(p)
    with pytest.raises(SystemExit, match="no inference rows at their accepted attempt"):
        assert_full_hit_per_episode(arm, _accepted(arm, uids, attempt=2), hits[arm])


def test_stale_attempt_miss_does_not_condemn_a_clean_accepted_run(tmp_path):
    """And the mirror image: a stale MISS must not fail an all-FULL_HIT rerun."""
    arm = "cache_size_x_S1"
    uids = [f"{arm}:eval:0:{e}" for e in range(2)]
    stale = [{"task_uid": uids[0], "yaml_id": arm, "attempt": 1, "step_idx": 0,
              "hit_type": "MISS"}]
    p = _write_per_step(tmp_path / "ps.jsonl", arm, uids, attempt=2, extra_rows=stale)

    hits, _ = load_per_episode_hits(p)
    summary = assert_full_hit_per_episode(arm, _accepted(arm, uids, attempt=2), hits[arm])
    assert summary["episodes"] == 2
    assert summary["stale_rows_ignored"] == 1


def test_duplicate_row_for_one_step_is_fatal(tmp_path):
    """The canonical merge de-duplicates, so a survivor means two rows disagree."""
    arm = "cache_size_x_S1"
    uid = f"{arm}:eval:0:0"
    dup = [{"task_uid": uid, "yaml_id": arm, "attempt": 1, "step_idx": 0,
            "hit_type": "MISS"}]
    p = _write_per_step(tmp_path / "ps.jsonl", arm, [uid], hits_per_uid=2,
                        extra_rows=dup)
    with pytest.raises(SystemExit, match="duplicate per-step row"):
        load_per_episode_hits(p)


def test_unstamped_row_is_fatal_in_formal_mode(tmp_path):
    """An unstamped row cannot be matched to an attempt, so it may be stale."""
    arm = "cache_size_x_S1"
    uid = f"{arm}:eval:0:0"
    p = tmp_path / "ps.jsonl"
    p.write_text(json.dumps({"task_uid": uid, "yaml_id": arm, "step_idx": 0,
                             "hit_type": "FULL_HIT"}) + "\n")
    with pytest.raises(SystemExit, match="carries no 'attempt'"):
        load_per_episode_hits(p)


# ---------------------------------------------------------------------------
# Launch binding
# ---------------------------------------------------------------------------


def _launch(tmp_path, **over):
    rec = {
        "suite": SUITE,
        "arms": [f"cache_size_{SUITE}_S1"],
        "trials_per_task": 50,
        "smoke": False,
        "apool": {"rollup_sha256": "abc123", "apool_dir": "/frozen/apool"},
    }
    rec.update(over)
    p = tmp_path / "launch.json"
    p.write_text(json.dumps(rec))
    return str(p)


def test_launch_binding_requires_a_digest(tmp_path):
    p = _launch(tmp_path, apool=None)
    with pytest.raises(SystemExit, match="not bound to the frozen evaluation pool"):
        verify_launch_binding(p, suite=SUITE, arms={f"cache_size_{SUITE}_S1"},
                              apool_digest_expected="abc123")


def test_launch_binding_rejects_a_smoke_run(tmp_path):
    """Smoke runs relax the binding and the completeness gates; their episodes
    cannot back a reported result, however well-formed the record looks."""
    p = _launch(tmp_path, smoke=True)
    with pytest.raises(SystemExit, match="--smoke run"):
        verify_launch_binding(p, suite=SUITE, arms={f"cache_size_{SUITE}_S1"},
                              apool_digest_expected="abc123")


def test_launch_binding_rejects_a_short_trial_count(tmp_path):
    """A 5-trial shakedown produces a perfectly well-formed launch record."""
    p = _launch(tmp_path, trials_per_task=5)
    with pytest.raises(SystemExit, match="trials per task"):
        verify_launch_binding(p, suite=SUITE, arms={f"cache_size_{SUITE}_S1"},
                              apool_digest_expected="abc123")


def test_launch_binding_rejects_extra_launched_arms(tmp_path):
    """Equality, not containment: analysing 8 of 12 launched arms is a subset run."""
    p = _launch(tmp_path, arms=[f"cache_size_{SUITE}_S1", f"cache_size_{SUITE}_S2"])
    with pytest.raises(SystemExit, match="only launched"):
        verify_launch_binding(p, suite=SUITE, arms={f"cache_size_{SUITE}_S1"},
                              apool_digest_expected="abc123")


def test_launch_binding_detects_digest_mismatch(tmp_path):
    p = _launch(tmp_path)
    with pytest.raises(SystemExit, match="digest mismatch"):
        verify_launch_binding(p, suite=SUITE, arms={f"cache_size_{SUITE}_S1"},
                              apool_digest_expected="different")


def test_launch_binding_detects_suite_and_arm_drift(tmp_path):
    with pytest.raises(SystemExit, match="for suite"):
        verify_launch_binding(_launch(tmp_path, suite="libero_10"),
                              suite=SUITE, arms={f"cache_size_{SUITE}_S1"},
                              apool_digest_expected="abc123")
    with pytest.raises(SystemExit, match="only analysed"):
        verify_launch_binding(_launch(tmp_path), suite=SUITE,
                              arms={f"cache_size_{SUITE}_S1", f"cache_size_{SUITE}_S6"},
                              apool_digest_expected="abc123")


def test_launch_binding_accepts_a_matching_record(tmp_path):
    rec = verify_launch_binding(_launch(tmp_path), suite=SUITE,
                                arms={f"cache_size_{SUITE}_S1"},
                                apool_digest_expected="abc123")
    assert rec["apool"]["apool_dir"] == "/frozen/apool"


def test_paired_diff_requires_identical_task_sets():
    with pytest.raises(SystemExit, match="task sets differ"):
        paired_diff({0: 1.0}, {1: 1.0})


# ---------------------------------------------------------------------------
# Statistics wiring (task-level, driven by complete ledgers)
# ---------------------------------------------------------------------------


def _rising_suite(top=0.90, teacher=0.95, jitter=0.01):
    tier_rates = {}
    ladder = [0.30, 0.50, 0.70, 0.82, top - 0.005, top]
    for k, (tier, base) in enumerate(zip(TIERS, ladder)):
        tier_rates[tier] = {i: base + ((i * 7 + k * 3) % 5) * jitter for i in range(10)}
    teacher_rates = {i: teacher + ((i * 3 + 1) % 4) * jitter for i in range(10)}
    return tier_rates, teacher_rates


def test_family_is_eight_tests_and_reports_both_axes():
    tier_rates, teacher_rates = _rising_suite()
    res = analyze_suite(tier_rates=tier_rates, teacher_rates=teacher_rates, b=800)

    assert res["family_size"] == 8
    assert set(res["axes"]) == {"D", "Q", "P", "M_yes"}

    md = render(res, SUITE)
    assert "adequacy (Q, main axis)" in md
    assert "direction (D)" in md
    assert "**descriptive**" in md
    assert "Provenance" in md


def test_task_rates_from_ledger():
    ledger = {(0, e): e < 25 for e in range(50)}
    assert task_rates(ledger) == {0: 0.5}


def test_degenerate_tiers_stay_in_family_as_not_evaluable():
    tier_rates, teacher_rates = _rising_suite()
    tier_rates["S6"] = dict(tier_rates["S5"])
    res = analyze_suite(tier_rates=tier_rates, teacher_rates=teacher_rates,
                        degenerate_pairs=["S5-S6"], b=800)
    assert res["family_size"] == 8
    assert "t5_S5_S6" in res["not_evaluable"]
    assert res["holm_rejected"]["t5_S5_S6"] is False
    assert "NOT the same as 'no difference'" in render(res, SUITE)


def test_large_gap_drives_q_fail():
    tier_rates, teacher_rates = _rising_suite(top=0.55, teacher=0.95)
    res = analyze_suite(tier_rates=tier_rates, teacher_rates=teacher_rates, b=1500)
    assert res["axes"]["Q"] == "Q-fail"
    assert res["axes"]["D"] == "D-teacher"


def test_small_gap_never_silently_claims_equivalence():
    tier_rates, teacher_rates = _rising_suite(top=0.94, teacher=0.95)
    res = analyze_suite(tier_rates=tier_rates, teacher_rates=teacher_rates, b=1500)
    assert res["axes"]["Q"] in ("Q-pass", "Q-inconc")
    if res["axes"]["Q"] == "Q-inconc":
        assert any("NOT evidence of equivalence" in c for c in res["verdict"]["caveats"])


def test_ci_gate_disagreements_are_disclosed_not_resolved():
    tier_rates, teacher_rates = _rising_suite()
    res = analyze_suite(tier_rates=tier_rates, teacher_rates=teacher_rates, b=800)
    for msg in res["ci_gate_disagreements"]:
        assert "family gate governs" in msg


def test_monotonic_drop_triggers_branch_n():
    tier_rates, teacher_rates = _rising_suite()
    # A regression with realistic task-to-task spread; a constant offset would
    # give every d_t the same value, i.e. zero between-cluster variance, which
    # the degeneracy guard correctly refuses to evaluate.
    tier_rates["S4"] = {t: v - 0.30 - (t % 4) * 0.01 for t, v in tier_rates["S3"].items()}
    res = analyze_suite(tier_rates=tier_rates, teacher_rates=teacher_rates, b=1500)
    assert res["axes"]["M_yes"] is True
    assert res["verdict"]["branch"] == "N"


# ---------------------------------------------------------------------------
# Sensitivity arms + plotting inputs
# ---------------------------------------------------------------------------


def test_analyze_suite_exports_plotting_inputs():
    """plot_size needs per-tier SR and CI; the analyzer must supply them."""
    tier_rates, teacher_rates = _rising_suite()
    res = analyze_suite(tier_rates=tier_rates, teacher_rates=teacher_rates, b=600)

    assert set(res["tier_sr"]) == set(TIERS)
    assert set(res["tier_ci"]) == set(TIERS)
    for t in TIERS:
        lo, hi = res["tier_ci"][t]
        assert lo <= res["tier_sr"][t] <= hi
    assert 0.0 <= res["teacher_sr"] <= 1.0


def test_sensitivity_equivalent_when_recal_barely_moves():
    from exp.ablation_study.cache_size.analysis.analyze_size import analyze_sensitivity

    main = {i: 0.80 + (i % 3) * 0.01 for i in range(10)}
    recal = {i: v + 0.002 * (1 if i % 2 else -1) for i, v in main.items()}

    out = analyze_sensitivity(main, recal, tier="S6", b=1500)
    assert out["equivalent"] is True
    assert "does not move the answer" in out["interpretation"]
    assert out["note"].startswith("descriptive")


def test_sensitivity_not_equivalent_when_recal_shifts_the_curve():
    from exp.ablation_study.cache_size.analysis.analyze_size import analyze_sensitivity

    main = {i: 0.80 + (i % 3) * 0.01 for i in range(10)}
    recal = {i: v + 0.12 for i, v in main.items()}  # far beyond the 3pp margin

    out = analyze_sensitivity(main, recal, tier="S1", b=1500)
    assert out["equivalent"] is False
    assert "must be qualified" in out["interpretation"]


def test_sensitivity_renders_into_the_report():
    from exp.ablation_study.cache_size.analysis.analyze_size import analyze_sensitivity

    tier_rates, teacher_rates = _rising_suite()
    res = analyze_suite(tier_rates=tier_rates, teacher_rates=teacher_rates, b=600)
    main = tier_rates["S6"]
    res["sensitivity"] = [
        analyze_sensitivity(main, {i: v + 0.15 for i, v in main.items()}, tier="S6", b=600)
    ]

    md = render(res, SUITE)
    assert "Normalizer sensitivity (descriptive)" in md
    assert "must be qualified" in md


# --- outcome-filter wiring (plan §3.1b ruling 1 / §12.2) -----------------------
#
# The two library families are evaluated side by side, so the arm id carries the
# filter (``cache_size_<suite>_<filter>_<tier>``). The analyzer used to rebuild
# that id without the filter, which made every arm lookup miss on a real run.
# These tests drive ``main`` end to end because that is where the id was built.

def _full_run(tmp_path, *, outcome_filter, with_recal):
    """A complete, formally valid single-family run on disk."""
    from exp.ablation_study.cache_size.emit_size_yamls import arm_name

    tmp_path.mkdir(parents=True, exist_ok=True)
    # A recal arm carries its base tier's rate, so the sensitivity comparison is
    # the near-equivalence the real arms show rather than an artefact of ordering.
    rate_of = {arm_name(SUITE, tier, outcome_filter=outcome_filter): 0.50 + 0.05 * i
               for i, tier in enumerate(TIERS)}
    names = list(rate_of)
    if with_recal:
        for tier in ("S1", "S6"):
            recal = arm_name(SUITE, tier, recal=True, outcome_filter=outcome_filter)
            rate_of[recal] = rate_of[arm_name(SUITE, tier, outcome_filter=outcome_filter)]
            names.append(recal)

    journal = tmp_path / "journal.jsonl"
    per_step = tmp_path / "per_step.jsonl"
    with open(journal, "w") as jf, open(per_step, "w") as pf:
        for arm in names:
            rate = rate_of[arm]
            for row in _journal_rows(arm, success_of=_ladder_fn(rate),
                                     attempt=1, accepted=True):
                jf.write(json.dumps(row) + "\n")
                for step in range(2):
                    pf.write(json.dumps({
                        "task_uid": row["task_uid"], "yaml_id": arm,
                        "attempt": 1, "step_idx": step, "hit_type": "FULL_HIT",
                    }) + "\n")

    anchor_dir = tmp_path / "anchor"
    anchor_dir.mkdir()
    teacher = _ladder_fn(0.95)
    (anchor_dir / "t.json").write_text(json.dumps([
        {"task_id": t, "init_state_idx": e, "success": bool(teacher(t, e))}
        for t in range(EXPECTED_TASKS) for e in range(EXPECTED_EPISODES_PER_TASK)
    ]))

    launch = tmp_path / "launch.json"
    launch.write_text(json.dumps({
        "suite": SUITE, "arms": names, "trials_per_task": 50, "smoke": False,
        "apool": {"rollup_sha256": "deadbeef", "apool_dir": "/frozen/apool"},
    }))
    return names, journal, per_step, anchor_dir, launch


def _run_main(monkeypatch, tmp_path, fixture, *, flag):
    _names, journal, per_step, anchor_dir, launch = fixture
    out_json = tmp_path / "out.json"
    argv = ["analyze_size.py",
            "--journal", str(journal), "--per-step", str(per_step),
            "--launch-record", str(launch), "--teacher-anchor", str(anchor_dir),
            "--suite", SUITE, "--apool-digest", "deadbeef",
            "--out-json", str(out_json), "--out-md", str(tmp_path / "out.md")]
    if flag is not None:
        argv += ["--outcome-filter", flag]
    monkeypatch.setattr("sys.argv", argv)
    from exp.ablation_study.cache_size.analysis.analyze_size import main
    main()
    return json.loads(out_json.read_text())


def test_primary_family_is_found_through_the_filtered_arm_ids(monkeypatch, tmp_path):
    fx = _full_run(tmp_path, outcome_filter="all", with_recal=True)
    result = _run_main(monkeypatch, tmp_path, fx, flag="all")
    assert result["family_role"] == "primary"
    assert result["outcome_filter"] == "all"
    assert result["family_size"] == 8
    # both sensitivity arms consumed
    assert {s["tier"] for s in result["sensitivity"]} == {"S1", "S6"}


def test_omitting_the_filter_misses_every_arm(monkeypatch, tmp_path):
    """The original defect: names rebuilt without the filter match nothing."""
    fx = _full_run(tmp_path, outcome_filter="all", with_recal=True)
    with pytest.raises(SystemExit, match=r"no accepted rows for arm"):
        _run_main(monkeypatch, tmp_path, fx, flag=None)


def test_secondary_family_needs_no_sensitivity_arms(monkeypatch, tmp_path):
    fx = _full_run(tmp_path, outcome_filter="success", with_recal=False)
    result = _run_main(monkeypatch, tmp_path, fx, flag="success")
    assert result["family_role"] == "secondary-descriptive"
    assert result["sensitivity"] == []
    assert result["family_size"] == 8   # the test family itself is unchanged


def test_stray_recal_arm_in_the_secondary_family_is_fatal(monkeypatch, tmp_path):
    fx = _full_run(tmp_path, outcome_filter="success", with_recal=True)
    with pytest.raises(SystemExit, match="carries no sensitivity arms by design"):
        _run_main(monkeypatch, tmp_path, fx, flag="success")


def test_secondary_report_announces_itself_before_any_verdict(monkeypatch, tmp_path):
    fx = _full_run(tmp_path, outcome_filter="success", with_recal=False)
    result = _run_main(monkeypatch, tmp_path, fx, flag="success")
    md = render(result, SUITE)
    assert "Secondary, descriptive read" in md
    assert "Pre-registered family" not in md
    assert "**descriptive**" in md
    # and the primary keeps its wording
    fx2 = _full_run(tmp_path / "p", outcome_filter="all", with_recal=True)
    primary = _run_main(monkeypatch, tmp_path / "p", fx2, flag="all")
    assert "Pre-registered family" in render(primary, SUITE)
    assert "Secondary, descriptive read" not in render(primary, SUITE)
