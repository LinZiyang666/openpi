"""Journal append/replay tests (M2)."""

from __future__ import annotations

from openpi.conductor.journal import Journal


def test_record_and_replay(tmp_path):
    j = Journal(tmp_path / "j.jsonl")
    j.record(task_uid="u1", yaml_id="y", phase="eval", status="done", success=True)
    j.record(task_uid="u2", yaml_id="y", phase="eval", status="failed", success=False)
    assert j.replay_done_uids() == {"u1", "u2"}


def test_failed_status_is_terminal(tmp_path):
    j = Journal(tmp_path / "j.jsonl")
    j.record(task_uid="u1", yaml_id="y", phase="eval", status="done", success=True)
    j.record(task_uid="u1", yaml_id="y", phase="eval", status="failed", success=False)
    # done OR failed are both terminal: a non-retriable failed record keeps the
    # uid in the resume-skip set so it is not re-run (anti-livelock fix).
    assert j.replay_done_uids() == {"u1"}


def test_replay_missing_file_is_empty(tmp_path):
    j = Journal(tmp_path / "nope.jsonl")
    assert j.replay_done_uids() == set()


def test_torn_last_line_tolerated(tmp_path):
    p = tmp_path / "j.jsonl"
    j = Journal(p)
    j.record(task_uid="u1", yaml_id="y", phase="eval", status="done", success=True)
    with p.open("a", encoding="utf-8") as fh:
        fh.write('{"task_uid": "u2", "status": "do')  # torn write from a hard crash
    assert j.replay_done_uids() == {"u1"}  # torn line ignored, u1 survives


def test_record_is_appendonly_across_instances(tmp_path):
    p = tmp_path / "j.jsonl"
    Journal(p).record(task_uid="u1", yaml_id="y", phase="eval", status="done", success=True)
    Journal(p).record(task_uid="u2", yaml_id="y", phase="eval", status="done", success=True)
    assert Journal(p).replay_done_uids() == {"u1", "u2"}
