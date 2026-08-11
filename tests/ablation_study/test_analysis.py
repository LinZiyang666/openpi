"""Analyzer contract tests: journal schema, coverage, Holm step-down, preflight."""

from __future__ import annotations

import json

import pytest

from exp.ablation_study.analysis.analyze_ablation import check_paired_coverage
from exp.ablation_study.analysis.analyze_ablation import holm_adjust
from exp.ablation_study.analysis.analyze_ablation import load_journal
from exp.ablation_study.analysis.analyze_ablation import paired_compare
from exp.ablation_study.analysis.analyze_ablation import parse_task_uid
from exp.ablation_study.analysis.analyze_ablation import preflight


def _journal(tmp_path, records):
    p = tmp_path / "journal.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    return str(p)


def _rec(arm, task, ep, status, success):
    from openpi.conductor.task import make_task_uid

    return {"task_uid": make_task_uid(arm, "eval", task, ep), "yaml_id": arm,
            "phase": "eval", "status": status, "success": success, "ts": 0.0}


def test_parse_task_uid_roundtrips_canonical_format():
    from openpi.conductor.task import make_task_uid

    assert parse_task_uid(make_task_uid("hit_smolvla", "eval", 3, 17)) == (3, 17)
    with pytest.raises(ValueError):
        parse_task_uid("weird-uid")


def test_journal_boundary_roundtrip(tmp_path):
    # Through the REAL writer: Journal.record + strategy-format uid.
    from openpi.conductor.journal import Journal
    from openpi.conductor.task import make_task_uid

    jp = tmp_path / "j.jsonl"
    j = Journal(str(jp))
    j.record(task_uid=make_task_uid("armA", "eval", 2, 7), yaml_id="armA",
             phase="eval", status="done", success=True)
    j.record(task_uid=make_task_uid("armA", "eval", 2, 8), yaml_id="armA",
             phase="eval", status="failed", success=False)
    arms = load_journal([str(jp)])
    assert arms == {"armA": {(2, 7): True, (2, 8): False}}


def test_load_journal_real_contract_keeps_failed_episodes(tmp_path):
    # Conductor journals unsuccessful rollouts as status="failed", success=False;
    # dropping them would inflate SR.
    path = _journal(tmp_path, [
        _rec("a", 0, 0, "done", True),
        _rec("a", 0, 1, "failed", False),
        _rec("a", 0, 1, "done", True),   # retry: last record wins
        _rec("a", 0, 2, "failed", False),
        _rec("a", 0, 3, "running", False),  # non-terminal: ignored
    ])
    arms = load_journal([path])
    assert arms["a"] == {(0, 0): True, (0, 1): True, (0, 2): False}


def test_coverage_flags_missing_identities(tmp_path):
    path = _journal(tmp_path, [
        _rec("a", 0, 0, "done", True), _rec("a", 0, 1, "done", True),
        _rec("b", 0, 0, "done", False),
    ])
    cov = check_paired_coverage(load_journal([path]))
    assert cov["a"]["missing"] == 0
    assert cov["b"]["missing"] == 1


def test_paired_compare_counts_discordance(tmp_path):
    path = _journal(tmp_path, [
        _rec("a", 0, i, "done", s) for i, s in enumerate([True, True, False, True])
    ] + [
        _rec("b", 0, i, "done", s) for i, s in enumerate([True, False, False, False])
    ])
    arms = load_journal([path])
    r = paired_compare(arms["a"], arms["b"])
    assert r["n_pairs"] == 4
    assert r["n10"] == 2 and r["n01"] == 0  # a wins twice, b never


def test_holm_step_down_is_monotone():
    # Reviewer probe: naive (m-rank)*p gives 0.06, 0.04 and falsely rejects the
    # second hypothesis after retaining the first. Correct Holm must carry the
    # cumulative max and stop rejecting at the first retention.
    out = holm_adjust([("h1", 0.03), ("h2", 0.04)])
    assert out["h1"]["holm_adjusted_p"] == pytest.approx(0.06)
    assert out["h2"]["holm_adjusted_p"] == pytest.approx(0.06)  # cummax, not 0.04
    assert not out["h1"]["significant"] and not out["h2"]["significant"]


def test_holm_rejects_full_chain_when_all_below_alpha():
    # Every adjusted p stays below alpha along the step-down ordering, so the
    # whole chain is rejected: 3*0.001=0.003, 2*0.01=0.02, 1*0.04=0.04.
    out = holm_adjust([("h1", 0.001), ("h2", 0.04), ("h3", 0.01)])
    assert out["h1"]["significant"] is True
    assert out["h3"]["significant"] is True
    assert out["h2"]["significant"] is True
    assert out["h2"]["holm_adjusted_p"] == pytest.approx(0.04)


def test_holm_stops_after_first_retention():
    # Ordering: h1 (2*0.04=0.08 >= alpha, retained) blocks h2 even though its
    # raw (m-i)*p would be 0.045 < alpha.
    out = holm_adjust([("h1", 0.04), ("h2", 0.045)])
    assert out["h1"]["significant"] is False
    assert out["h2"]["significant"] is False
    assert out["h2"]["holm_adjusted_p"] == pytest.approx(0.08)  # cummax


def test_preflight_is_a_power_gate_not_halfwidth():
    # Reviewer probe: q=0.15, n=500 has only ~7% TOST power at true diff 0 —
    # the gate must block, not pass on the half-width heuristic.
    bad = preflight(500, 0.15)
    assert bad["tost_power_at_zero_diff"] == pytest.approx(0.0696, abs=0.005)
    assert bad["decidable"] is False
    ok = preflight(500, 0.04)
    assert ok["tost_power_at_zero_diff"] > 0.8 and ok["decidable"] is True
    assert bad["max_decidable_discordance"] == pytest.approx(0.0525, abs=0.002)


def test_snapshot_merge_canonical_dedup(tmp_path):
    import json as _json

    from exp.ablation_study.run_ablation_eval import merge_snapshot

    main = tmp_path / "rows.jsonl"
    snap = tmp_path / "rows.snapshot.jsonl"
    # Same object, different key order: must NOT duplicate on merge.
    main.write_text(_json.dumps({"task_uid": "u1", "yaml_id": "a", "step_idx": 0}) + "\n")
    snap.write_text(
        _json.dumps({"yaml_id": "a", "task_uid": "u1", "step_idx": 0}) + "\n"
        + _json.dumps({"task_uid": "u2", "yaml_id": "a", "step_idx": 0}) + "\n"
    )
    assert merge_snapshot(main, snap) == 1
    assert not snap.exists()
    lines = [line for line in main.read_text().splitlines() if line]
    assert len(lines) == 2
    # Second window: a fresh stale snapshot with only known rows merges zero.
    snap.write_text(_json.dumps({"step_idx": 0, "yaml_id": "a", "task_uid": "u2"}) + "\n")
    assert merge_snapshot(main, snap) == 0
    assert len([line for line in main.read_text().splitlines() if line]) == 2


def test_approval_file_validation(tmp_path):
    import yaml as _yaml

    from exp.ablation_study.run_ablation_eval import load_approval

    bad = tmp_path / "bad.yaml"
    bad.write_text("just: nonsense\n")
    with pytest.raises(SystemExit, match="approval file"):
        load_approval(str(bad))
    good = tmp_path / "ok.yaml"
    good.write_text(_yaml.safe_dump({
        "approved_by": "Ziyang Lin", "date": "2026-08-11",
        "decision": "approved_delta", "delta": 0.05,
    }))
    doc = load_approval(str(good))
    assert doc["delta"] == 0.05 and "sha256" in doc


def test_preflight_delta_threading():
    # A larger approved margin flips decidability at the same n/q.
    assert preflight(500, 0.15, delta=0.03)["decidable"] is False
    assert preflight(500, 0.15, delta=0.08)["decidable"] is True


def test_approval_normalizes_yaml_date_and_rejects_bad_decision(tmp_path):
    from exp.ablation_study.run_ablation_eval import load_approval

    f = tmp_path / "a.yaml"
    # Unquoted YAML date parses as datetime.date; must normalise to str and
    # survive json.dumps downstream.
    f.write_text("approved_by: Ziyang Lin\ndate: 2026-08-11\n"
                 "decision: approved_delta\ndelta: 0.05\n")
    doc = load_approval(str(f))
    assert isinstance(doc["date"], str)
    json.dumps(doc)  # must not raise
    f.write_text("approved_by: x\ndate: 2026-08-11\ndecision: garbage\ndelta: 0.08\n")
    with pytest.raises(SystemExit, match="decision"):
        load_approval(str(f))
    f.write_text("approved_by: x\ndate: 2026-08-11\ndecision: approved_delta\ndelta: 0.5\n")
    with pytest.raises(SystemExit, match="range"):
        load_approval(str(f))


def test_snapshot_writer_merge_lifecycle_race(tmp_path):
    # Real writer thread: stop + JOIN must precede the final fold so a stale
    # snapshot can never be re-created after the merge.
    import threading
    import time as _time
    from types import SimpleNamespace

    from exp.ablation_study.run_ablation_eval import merge_snapshot

    per_step = tmp_path / "rows.jsonl"
    snap = per_step.with_suffix(".snapshot.jsonl")
    rows = [{"task_uid": "u1", "step_idx": 0}]
    driver = SimpleNamespace(per_step_rows=rows)
    stop = threading.Event()

    def _loop():
        import json as _json
        import os

        while not stop.wait(0.02):
            tmpf = snap.with_suffix(".tmp")
            with tmpf.open("w", encoding="utf-8") as f:
                for row in list(driver.per_step_rows):
                    f.write(_json.dumps(row) + "\n")
            os.replace(tmpf, snap)

    t = threading.Thread(target=_loop, daemon=True)
    t.start()
    _time.sleep(0.1)
    stop.set()
    t.join(timeout=5)  # the fix under test: join BEFORE folding
    merge_snapshot(per_step, snap)
    _time.sleep(0.1)
    assert not snap.exists()  # no post-merge re-creation
    assert len(per_step.read_text().splitlines()) == 1


def test_snapshot_internal_duplicates_deduped(tmp_path):
    from exp.ablation_study.run_ablation_eval import merge_snapshot

    per_step = tmp_path / "rows.jsonl"
    snap = per_step.with_suffix(".snapshot.jsonl")
    snap.write_text('{"a": 1, "b": 2}\n{"b": 2, "a": 1}\n')  # same object twice
    assert merge_snapshot(per_step, snap) == 1
