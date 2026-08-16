"""Runner gates for the cache-size ablation (plan §7, §12-P6/P7).

The shared executor-substitution runner rejects every arm here (it requires a
``routing`` section on any non-baseline arm), so this experiment ships its own.
These tests hold the dedicated runner to the three gates that make the pure-cache
premise checkable.
"""

from __future__ import annotations

import json

import pytest
import yaml

from exp.ablation_study.cache_size.emit_size_yamls import make_main_arm
from exp.ablation_study.cache_size.run_size_eval import (
    full_hit_rates,
    load_apool_digest,
    validate_pure_cache_arms,
)

BASELINE_PATH = "exp/ablation_study/config/common/libero_spatial_baseline.yaml"


@pytest.fixture()
def baseline():
    with open(BASELINE_PATH) as f:
        return yaml.safe_load(f)


def _write_arm(tmp_path, name, cfg):
    p = tmp_path / f"{name}.yaml"
    p.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return {"arm": name, "yaml": str(p), "sidecar": None}


def test_pure_cache_arms_are_accepted(baseline, tmp_path):
    """The exact shape the shared runner refuses must be accepted here."""
    lib = tmp_path / "S3.pkl"
    lib.write_bytes(b"")
    row = _write_arm(tmp_path, "cache_size_libero_spatial_S3",
                     make_main_arm(baseline, str(lib)))

    paths = validate_pure_cache_arms([row])
    assert set(paths) == {"cache_size_libero_spatial_S3"}


def test_shared_runner_would_have_rejected_the_same_arm(baseline, tmp_path):
    """Pin why a dedicated runner exists, so nobody 'simplifies' it away later."""
    from exp.ablation_study.run_ablation_eval import _validate_arms

    lib = tmp_path / "S3.pkl"
    lib.write_bytes(b"")
    row = _write_arm(tmp_path, "cache_size_libero_spatial_S3",
                     make_main_arm(baseline, str(lib)))

    with pytest.raises(SystemExit, match="expected a routing section"):
        _validate_arms([row])


def test_threshold_verdict_is_rejected(baseline, tmp_path):
    """A threshold arm would smuggle the calibration degree of freedom back in."""
    lib = tmp_path / "S3.pkl"
    lib.write_bytes(b"")
    cfg = make_main_arm(baseline, str(lib))
    cfg["checkpoints"]["cp1"]["judge"] = {"type": "threshold", "threshold": 0.98}
    row = _write_arm(tmp_path, "cache_size_libero_spatial_S3", cfg)

    with pytest.raises(SystemExit, match="expected 'always_hit'"):
        validate_pure_cache_arms([row])


def test_non_always_search_gate_is_rejected(baseline, tmp_path):
    lib = tmp_path / "S3.pkl"
    lib.write_bytes(b"")
    cfg = make_main_arm(baseline, str(lib))
    cfg["checkpoints"]["cp1"]["gate"] = {"type": "always_skip"}
    row = _write_arm(tmp_path, "cache_size_libero_spatial_S3", cfg)

    with pytest.raises(SystemExit, match="expected 'always_search'"):
        validate_pure_cache_arms([row])


def test_sidecar_declaration_is_rejected(baseline, tmp_path):
    lib = tmp_path / "S3.pkl"
    lib.write_bytes(b"")
    row = _write_arm(tmp_path, "cache_size_libero_spatial_S3",
                     make_main_arm(baseline, str(lib)))
    row["sidecar"] = "act"

    with pytest.raises(SystemExit, match="must not declare a sidecar"):
        validate_pure_cache_arms([row])


# ---------------------------------------------------------------------------
# A-pool binding
# ---------------------------------------------------------------------------


def _write_pool(tmp_path, *, n_files=10, states_per_task=50):
    """Real ``.init`` files in the shape ``materialize_apool.py`` writes."""
    import torch

    pool = tmp_path / "apool"
    pool.mkdir(exist_ok=True)
    for i in range(n_files):
        torch.save([[float(i), float(k)] for k in range(states_per_task)],
                   pool / f"task_{i}.init")
    return pool


def _apool_record(tmp_path, *, n_files=10, total=500, digests=10, with_dir=True,
                  states_per_task=50, rollup=None):
    """A record in the shape verify_apool.py writes, backed by real .init files.

    The digests are the *real* ones, computed from the files just written. A
    fixture with invented digests could never fail the rehash, so it could never
    witness the check this record exists to support.
    """
    from exp.ablation_study.cache_size.verify_apool import digest_init_file, rollup_digest

    pool = _write_pool(tmp_path, n_files=n_files, states_per_task=states_per_task)
    real = {f.stem: digest_init_file(f) for f in sorted(pool.glob("*.init"))}
    listed = {k: real[k] for k in sorted(real)[:digests]}
    rec = {
        "suite": "libero_spatial",
        "total_inits": total,
        "rollup_sha256": rollup if rollup is not None else rollup_digest(real),
        "per_task_digests": listed,
    }
    if with_dir:
        rec["apool_dir"] = str(pool)
    p = tmp_path / "apool.yaml"
    p.write_text(yaml.safe_dump(rec))
    return str(p)


def test_apool_record_binds_directory_and_digest(tmp_path):
    """The record must name the pool, not just hash one: workers load that dir."""
    loaded = load_apool_digest(_apool_record(tmp_path))
    assert loaded["apool_dir"].endswith("apool")
    assert len(loaded["per_task_digests"]) == 10


def test_apool_contents_swapped_after_the_record_is_fatal(tmp_path):
    """The whole point of re-hashing at launch.

    Reading the record's digests back and declaring them consistent proves only
    that the file agrees with itself. Substitute one ``.init`` afterwards and the
    workers load the substituted pool while the run attests the old digest.
    """
    import torch

    record = _apool_record(tmp_path)
    torch.save([[9.9, 9.9]] * 50, tmp_path / "apool" / "task_3.init")
    with pytest.raises(SystemExit, match="contents changed since the record"):
        load_apool_digest(record)


def test_apool_rollup_tampering_is_fatal(tmp_path):
    """Per-task digests can agree while the rollup does not; both are checked."""
    record = _apool_record(tmp_path, rollup="0" * 64)
    with pytest.raises(SystemExit, match="rollup mismatch"):
        load_apool_digest(record)


def test_apool_short_task_is_fatal(tmp_path):
    """49 inits in one task silently shrinks that task's success-rate denominator."""
    record = _apool_record(tmp_path, states_per_task=49)
    with pytest.raises(SystemExit, match="wrong init count"):
        load_apool_digest(record)


def test_apool_rehash_is_skipped_only_under_smoke(tmp_path):
    """Smoke may run against an unfrozen pool; a formal run may not."""
    import torch

    record = _apool_record(tmp_path)
    torch.save([[9.9, 9.9]] * 50, tmp_path / "apool" / "task_3.init")
    loaded = load_apool_digest(record, verify_contents=False)
    assert loaded["apool_dir"].endswith("apool")


def test_apool_record_without_directory_is_fatal(tmp_path):
    """A digest with no directory cannot attest what the workers actually loaded."""
    with pytest.raises(SystemExit, match="apool_dir"):
        load_apool_digest(_apool_record(tmp_path, with_dir=False))


def test_apool_record_with_wrong_count_is_fatal(tmp_path):
    with pytest.raises(SystemExit, match="expected 500"):
        load_apool_digest(_apool_record(tmp_path, total=450))


def test_apool_record_with_wrong_file_count_is_fatal(tmp_path):
    with pytest.raises(SystemExit, match="expected 10"):
        load_apool_digest(_apool_record(tmp_path, n_files=9))


def test_apool_record_is_required_for_a_formal_run(tmp_path):
    """Omitting it would let the workers fall back to their environment's pool."""
    with pytest.raises(SystemExit, match="--apool-record is required"):
        load_apool_digest(None, required=True)


def test_apool_record_optional_only_under_smoke(tmp_path):
    assert load_apool_digest(None, required=False) is None


# ---------------------------------------------------------------------------
# FULL_HIT witness
# ---------------------------------------------------------------------------


def test_full_hit_rate_detects_teacher_fallback(tmp_path):
    """A tier missing a task falls back to the teacher; that must be visible."""
    p = tmp_path / "ps.jsonl"
    with p.open("w") as f:
        for i in range(100):
            hit = "FULL_HIT" if i < 95 else "MISS"
            f.write(json.dumps({"task_uid": "cache_size_libero_spatial_S1:eval:3:0",
                                "hit_type": hit}) + "\n")

    rates = full_hit_rates(p)
    assert rates["cache_size_libero_spatial_S1"] == pytest.approx(0.95)
    assert rates["cache_size_libero_spatial_S1"] < 1.0, "gate must flag this arm"


def test_full_hit_rate_clean_arm_is_one(tmp_path):
    p = tmp_path / "ps.jsonl"
    with p.open("w") as f:
        for _ in range(50):
            f.write(json.dumps({"task_uid": "cache_size_libero_spatial_S6:eval:0:0",
                                "hit_type": "FULL_HIT"}) + "\n")
    assert full_hit_rates(p)["cache_size_libero_spatial_S6"] == 1.0


def test_full_hit_rate_missing_file_is_empty(tmp_path):
    assert full_hit_rates(tmp_path / "nope.jsonl") == {}


# ---------------------------------------------------------------------------
# The runner's exit gate is the analyzer's gate, not a weaker cousin
# ---------------------------------------------------------------------------


def _journal_and_steps(tmp_path, *, arm, n_ep, accepted_attempt=1, rows_attempt=1,
                       stale_rows=False):
    journal = tmp_path / "journal.jsonl"
    steps = tmp_path / "ps.jsonl"
    with journal.open("w") as j, steps.open("w") as st:
        for e in range(n_ep):
            uid = f"{arm}:eval:0:{e}"
            if stale_rows:
                j.write(json.dumps({"task_uid": uid, "yaml_id": arm, "phase": "eval",
                                    "status": "failed", "success": False,
                                    "attempt": 1, "accepted": False}) + "\n")
                st.write(json.dumps({"task_uid": uid, "yaml_id": arm, "attempt": 1,
                                     "step_idx": 0, "hit_type": "MISS"}) + "\n")
            j.write(json.dumps({"task_uid": uid, "yaml_id": arm, "phase": "eval",
                                "status": "done", "success": True,
                                "attempt": accepted_attempt, "accepted": True}) + "\n")
            for k in range(3):
                st.write(json.dumps({"task_uid": uid, "yaml_id": arm,
                                     "attempt": rows_attempt, "step_idx": k,
                                     "hit_type": "FULL_HIT"}) + "\n")
    return journal, steps


def test_exit_gate_passes_a_clean_run(tmp_path):
    from exp.ablation_study.cache_size.run_size_eval import assert_accepted_full_hit

    arm = "cache_size_libero_spatial_S1"
    journal, steps = _journal_and_steps(tmp_path, arm=arm, n_ep=4)
    summary = assert_accepted_full_hit(journal, steps, {arm})
    assert summary[arm]["episodes"] == 4


def test_exit_gate_ignores_stale_rows_but_still_needs_the_accepted_ones(tmp_path):
    from exp.ablation_study.cache_size.run_size_eval import assert_accepted_full_hit

    arm = "cache_size_libero_spatial_S1"
    journal, steps = _journal_and_steps(tmp_path, arm=arm, n_ep=3, accepted_attempt=2,
                                        rows_attempt=2, stale_rows=True)
    summary = assert_accepted_full_hit(journal, steps, {arm})
    assert summary[arm]["episodes"] == 3
    assert summary[arm]["stale_rows_ignored"] == 3


def test_formal_main_does_not_apply_arm_level_floor_before_accepted_gate():
    """The formal call order must not let a stale MISS reject a clean retry."""
    import ast
    import inspect

    from exp.ablation_study.cache_size import run_size_eval

    tree = ast.parse(inspect.getsource(run_size_eval.main))
    smoke_if = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.If) and ast.unparse(node.test) == "args.smoke"
        and node.orelse
        and any(
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "assert_accepted_full_hit"
            for stmt in node.orelse for n in ast.walk(stmt)
        )
    )

    def direct_calls(statements):
        return {
            n.func.id
            for stmt in statements for n in ast.walk(stmt)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        }

    assert direct_calls(smoke_if.body) >= {"assert_full_hit"}
    assert "assert_accepted_full_hit" not in direct_calls(smoke_if.body)
    assert direct_calls(smoke_if.orelse) >= {"assert_accepted_full_hit"}
    assert "assert_full_hit" not in direct_calls(smoke_if.orelse)


def test_exit_gate_rejects_a_run_witnessed_only_by_stale_rows(tmp_path):
    """The arm-level ratio is 1.0 here; the per-episode join is what catches it."""
    from exp.ablation_study.cache_size.run_size_eval import (
        assert_accepted_full_hit,
        full_hit_rates,
    )

    arm = "cache_size_libero_spatial_S1"
    journal, steps = _journal_and_steps(tmp_path, arm=arm, n_ep=3, accepted_attempt=2,
                                        rows_attempt=1)
    assert full_hit_rates(steps)[arm] == 1.0, "the weaker gate is happy -- that is the point"
    with pytest.raises(SystemExit, match="no inference rows at their accepted attempt"):
        assert_accepted_full_hit(journal, steps, {arm})


def test_exit_gate_rejects_an_arm_with_one_witnessed_episode(tmp_path):
    """One FULL_HIT row gives a perfect arm-level rate and must not pass."""
    from exp.ablation_study.cache_size.run_size_eval import assert_accepted_full_hit

    arm = "cache_size_libero_spatial_S1"
    journal, steps = _journal_and_steps(tmp_path, arm=arm, n_ep=5)
    kept = [line for line in steps.read_text().splitlines()
            if f"{arm}:eval:0:0" in line]
    steps.write_text("\n".join(kept) + "\n")
    with pytest.raises(SystemExit, match="no inference rows at their accepted attempt"):
        assert_accepted_full_hit(journal, steps, {arm})


def test_exit_gate_fails_an_arm_absent_from_the_journal(tmp_path):
    from exp.ablation_study.cache_size.run_size_eval import assert_accepted_full_hit

    arm = "cache_size_libero_spatial_S1"
    journal, steps = _journal_and_steps(tmp_path, arm=arm, n_ep=2)
    with pytest.raises(SystemExit, match="no accepted episodes for arm"):
        assert_accepted_full_hit(journal, steps, {arm, "cache_size_libero_spatial_S6"})


# ---------------------------------------------------------------------------
# Absence must fail the gate, not skip it
# ---------------------------------------------------------------------------


def test_missing_arm_evidence_fails_the_gate():
    """An arm with no per-step rows is what a crashed arm looks like."""
    from exp.ablation_study.cache_size.run_size_eval import assert_full_hit

    rates = {"cache_size_x_S1": 1.0}
    with pytest.raises(SystemExit, match="no per-step evidence"):
        assert_full_hit(rates, {"cache_size_x_S1", "cache_size_x_S2"}, 1.0)


def test_present_but_low_rate_fails_the_gate():
    from exp.ablation_study.cache_size.run_size_eval import assert_full_hit

    rates = {"cache_size_x_S1": 1.0, "cache_size_x_S2": 0.93}
    with pytest.raises(SystemExit, match="pure-cache premise violated"):
        assert_full_hit(rates, set(rates), 1.0)


def test_all_arms_at_floor_passes():
    from exp.ablation_study.cache_size.run_size_eval import assert_full_hit

    rates = {"cache_size_x_S1": 1.0, "cache_size_x_S2": 1.0}
    assert_full_hit(rates, set(rates), 1.0)


# ---------------------------------------------------------------------------
# Crash-safe per-step snapshot
# ---------------------------------------------------------------------------


def test_snapshot_merge_recovers_rows_lost_to_a_crash(tmp_path):
    from exp.ablation_study.cache_size.run_size_eval import merge_snapshot

    sink = tmp_path / "ps.jsonl"
    sink.write_text(json.dumps({"task_uid": "a:eval:0:0", "hit_type": "FULL_HIT"}) + "\n")
    snap = sink.with_suffix(".snapshot.jsonl")
    snap.write_text(
        json.dumps({"task_uid": "a:eval:0:0", "hit_type": "FULL_HIT"}) + "\n"
        + json.dumps({"task_uid": "a:eval:0:1", "hit_type": "FULL_HIT"}) + "\n"
    )

    added = merge_snapshot(sink, snap)
    assert added == 1, "only the row absent from the sink should be folded in"
    assert len(sink.read_text().strip().splitlines()) == 2
    assert not snap.exists(), "the snapshot must be retired once folded in"


def test_snapshot_merge_is_a_noop_without_snapshot(tmp_path):
    from exp.ablation_study.cache_size.run_size_eval import merge_snapshot

    sink = tmp_path / "ps.jsonl"
    sink.write_text("")
    assert merge_snapshot(sink, sink.with_suffix(".snapshot.jsonl")) == 0


def test_snapshot_merge_dedups_on_canonical_identity(tmp_path):
    """Key order must not create a duplicate, and the snapshot self-dedups.

    The snapshot writer and the sink writer serialize independently, so the same
    logical row can differ byte-wise. A byte-wise check folds it in twice and the
    per-episode FULL_HIT join then sees phantom inference rows.
    """
    from exp.ablation_study.cache_size.run_size_eval import merge_snapshot

    sink = tmp_path / "ps.jsonl"
    snap = sink.with_suffix(".snapshot.jsonl")
    sink.write_text(json.dumps({"task_uid": "a:eval:0:0", "step_idx": 0, "hit_type": "FULL_HIT"}) + "\n")
    snap.write_text(
        # same row, different key order -> must not duplicate
        json.dumps({"hit_type": "FULL_HIT", "step_idx": 0, "task_uid": "a:eval:0:0"}) + "\n"
        # new row, written twice in different key orders -> counted once
        + json.dumps({"task_uid": "a:eval:0:1", "step_idx": 0, "hit_type": "FULL_HIT"}) + "\n"
        + json.dumps({"step_idx": 0, "hit_type": "FULL_HIT", "task_uid": "a:eval:0:1"}) + "\n"
    )

    assert merge_snapshot(sink, snap) == 1
    assert len(sink.read_text().strip().splitlines()) == 2
    assert not snap.exists()


def test_stale_snapshot_cannot_reach_a_later_run(tmp_path):
    """Retiring the snapshot at exit is what isolates a reused output path."""
    from exp.ablation_study.cache_size.run_size_eval import merge_snapshot

    sink = tmp_path / "ps.jsonl"
    snap = sink.with_suffix(".snapshot.jsonl")
    sink.write_text("")
    snap.write_text(json.dumps({"task_uid": "old:eval:0:0", "hit_type": "FULL_HIT"}) + "\n")
    assert merge_snapshot(sink, snap) == 1
    assert not snap.exists()

    # A later run truncates the sink and finds no snapshot to fold back in.
    sink.write_text("")
    assert merge_snapshot(sink, snap) == 0
    assert sink.read_text() == ""


def test_snapshot_writer_is_joined_before_the_final_merge(tmp_path):
    """The lifecycle race: a live writer must not re-create the retired snapshot.

    Mirrors the equivalent guard on the executor-substitution runner -- stop the
    loop and join it *before* folding, or the merged snapshot reappears on disk
    and the next run inherits it.
    """
    import threading
    import time as _time
    from types import SimpleNamespace

    from exp.ablation_study.cache_size.run_size_eval import _snapshot_loop, merge_snapshot

    sink = tmp_path / "ps.jsonl"
    sink.write_text("")
    snap = sink.with_suffix(".snapshot.jsonl")
    driver = SimpleNamespace(per_step_rows=[{"task_uid": "a:eval:0:0", "hit_type": "FULL_HIT"}])
    stop = threading.Event()
    t = threading.Thread(target=_snapshot_loop, args=(driver, sink, stop), kwargs={"interval_s": 0.02},
                         daemon=True)
    t.start()
    _time.sleep(0.1)

    stop.set()
    t.join(timeout=5)
    assert not t.is_alive()

    merge_snapshot(sink, snap)
    _time.sleep(0.1)
    assert not snap.exists(), "a joined writer must not re-create the snapshot"
    assert len(sink.read_text().strip().splitlines()) == 1


# ---------------------------------------------------------------------------
# Journal read-back: retry-exhausted episodes are never journaled
# ---------------------------------------------------------------------------


def test_journal_shortfall_detects_unjournaled_episodes(tmp_path):
    from exp.ablation_study.cache_size.run_size_eval import journal_shortfall

    j = tmp_path / "j.jsonl"
    with j.open("w") as f:
        for e in range(498):
            f.write(json.dumps({"task_uid": f"armA:eval:0:{e}", "yaml_id": "armA",
                                "status": "done", "success": True}) + "\n")

    short = journal_shortfall(j, {"armA"}, 500)
    assert short == {"armA": 498}


def test_journal_shortfall_empty_when_complete(tmp_path):
    from exp.ablation_study.cache_size.run_size_eval import journal_shortfall

    j = tmp_path / "j.jsonl"
    with j.open("w") as f:
        for e in range(500):
            f.write(json.dumps({"task_uid": f"armA:eval:0:{e}", "yaml_id": "armA",
                                "status": "failed", "success": False}) + "\n")
    assert journal_shortfall(j, {"armA"}, 500) == {}
