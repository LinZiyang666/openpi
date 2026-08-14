"""Tests for the journal -> per-arm conversion.

The failure this guards against is silent and expensive: the two init pools use
overlapping index ranges, so merging them on a bare ``(task_id, init_idx)`` key
would collapse 950 paired episodes into 500 and pair every arm against the
wrong episode. Everything else here is bookkeeping around that.
"""

from __future__ import annotations

import json

import pytest

from exp.markov_sufficiency import journal_to_arms as j2a


def _row(yaml_id, task, init, success, status="done", ts=1.0):
    return {
        "task_uid": f"{yaml_id}:eval:{task}:{init}",
        "yaml_id": yaml_id,
        "phase": "eval",
        "status": status,
        "success": success,
        "ts": ts,
    }


def _journal(tmp_path, name, rows):
    p = tmp_path / name
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return str(p)


# ------------------------------------------------------------------
# task_uid parsing
# ------------------------------------------------------------------


def test_parse_task_uid_splits_from_the_right():
    assert j2a.parse_task_uid("e4_spatial__A0:eval:7:42") == ("e4_spatial__A0", "eval", 7, 42)


def test_parse_task_uid_rejects_a_malformed_id():
    with pytest.raises(ValueError, match="unrecognised task_uid"):
        j2a.parse_task_uid("only:three:parts")


# ------------------------------------------------------------------
# Pool separation -- the point of the module
# ------------------------------------------------------------------


def test_the_two_pools_do_not_collide_on_the_same_index(tmp_path):
    """Index 0 of each pool is a different initial state and must stay distinct."""
    a = _journal(tmp_path, "official.jsonl", [_row("A0", 0, 0, True)])
    b = _journal(tmp_path, "db.jsonl", [_row("A0", 0, 0, False)])
    arms = j2a.build_arms([(a, "official"), (b, "db_init")])
    table = arms["A0"]
    assert len(table) == 2, "the two pools collapsed into one episode"
    assert table[(0, 0)] is True
    assert table[(0, j2a.POOL_OFFSET["db_init"])] is False


def test_pool_offset_keeps_full_pools_disjoint(tmp_path):
    """50 official + 45 db_init inits per task must yield 95 distinct episodes."""
    off = [_row("A0", 0, i, i % 2 == 0) for i in range(50)]
    db = [_row("A0", 0, i, True) for i in range(45)]
    arms = j2a.build_arms([
        (_journal(tmp_path, "o.jsonl", off), "official"),
        (_journal(tmp_path, "d.jsonl", db), "db_init"),
    ])
    assert len(arms["A0"]) == 95


def test_duplicate_key_within_one_pool_raises(tmp_path):
    """A repeat across batches of the same pool is a bug, not a data condition."""
    p = _journal(tmp_path, "a.jsonl", [_row("A0", 0, 0, True)])
    q = _journal(tmp_path, "b.jsonl", [_row("A0", 0, 0, False)])
    with pytest.raises(ValueError, match="pool offset"):
        j2a.build_arms([(p, "official"), (q, "official")])


def test_unknown_pool_is_rejected(tmp_path):
    p = _journal(tmp_path, "a.jsonl", [_row("A0", 0, 0, True)])
    with pytest.raises(ValueError, match="unknown pool"):
        j2a.load_journal(p, "made_up")


# ------------------------------------------------------------------
# Terminal states and de-duplication
# ------------------------------------------------------------------


def test_failed_episodes_are_kept_as_outcomes(tmp_path):
    """Failure is a terminal state: dropping it inflates SR."""
    rows = [_row("A0", 0, 0, True), _row("A0", 0, 1, False, status="failed")]
    table = j2a.load_journal(_journal(tmp_path, "j.jsonl", rows), "official")["A0"]
    assert table == {(0, 0): True, (0, 1): False}


def test_non_terminal_rows_are_ignored(tmp_path):
    rows = [_row("A0", 0, 0, False, status="running"), _row("A0", 0, 1, True)]
    table = j2a.load_journal(_journal(tmp_path, "j.jsonl", rows), "official")["A0"]
    assert table == {(0, 1): True}


def test_the_latest_terminal_row_wins(tmp_path):
    rows = [_row("A0", 0, 0, False, ts=1.0), _row("A0", 0, 0, True, ts=2.0)]
    table = j2a.load_journal(_journal(tmp_path, "j.jsonl", rows), "official")["A0"]
    assert table == {(0, 0): True}


def test_arms_are_split_by_yaml_id(tmp_path):
    rows = [_row("A0", 0, 0, True), _row("A1", 0, 0, False), _row("A2", 1, 3, True)]
    arms = j2a.load_journal(_journal(tmp_path, "j.jsonl", rows), "official")
    assert set(arms) == {"A0", "A1", "A2"}


# ------------------------------------------------------------------
# Output shape
# ------------------------------------------------------------------


def test_records_match_what_load_arm_reads(tmp_path):
    from exp.markov_sufficiency import e45_rollout_analysis as e45

    rows = [_row("A0", 0, 0, True), _row("A0", 1, 2, False)]
    arms = j2a.build_arms([(_journal(tmp_path, "j.jsonl", rows), "official")])
    p = tmp_path / "A0.json"
    p.write_text(json.dumps(j2a.to_records(arms["A0"])))
    assert e45.load_arm(p) == {(0, 0): True, (1, 2): False}


def test_cli_writes_one_file_per_arm_plus_a_summary(tmp_path):
    rows = [_row("A0", 0, i, i < 3) for i in range(4)] + [_row("A1", 0, 0, True)]
    j = _journal(tmp_path, "j.jsonl", rows)
    out = tmp_path / "arms"
    j2a.main(["--batch", f"official={j}", "--out-dir", str(out)])
    assert (out / "A0.json").exists() and (out / "A1.json").exists()
    summary = json.loads((out / "arms_summary.json").read_text())
    assert summary["A0"]["n_episodes"] == 4
    assert summary["A0"]["n_success"] == 3
    assert summary["A0"]["sr"] == pytest.approx(0.75)


def test_cli_rejects_a_malformed_batch(tmp_path):
    with pytest.raises(SystemExit):
        j2a.main(["--batch", "nonsense", "--out-dir", str(tmp_path / "x")])
