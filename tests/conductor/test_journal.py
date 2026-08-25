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


def test_duration_is_persisted_and_omitted_when_absent(tmp_path):
    """Utilisation has to be answerable from the ledger, which is the only
    per-episode artifact a phase leaves behind -- but a caller that does not
    supply a duration must still write a byte-identical line."""
    import json

    path = tmp_path / "j.jsonl"
    j = Journal(path)
    j.record(task_uid="u1", yaml_id="y", phase="eval", status="done", success=True)
    j.record(
        task_uid="u2",
        yaml_id="y",
        phase="eval",
        status="done",
        success=True,
        duration_s=12.3456,
    )
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    assert "duration_s" not in rows[0]
    assert rows[1]["duration_s"] == 12.346


def test_fenced_stale_record_does_not_count_as_done(tmp_path):
    """A rejected result is not completed work.

    The driver journals a superseded dispatch's late result too, marked
    ``accepted: false``. A crash between that fence and the live retry would
    otherwise leave the episode permanently skipped on resume.
    """
    path = tmp_path / "j.jsonl"
    j = Journal(path)
    j.record(task_uid="u1", yaml_id="y", phase="eval", status="failed",
             success=False, attempt=1, accepted=False)
    assert j.replay_done_uids() == set()
    j.record(task_uid="u1", yaml_id="y", phase="eval", status="done",
             success=True, attempt=2, accepted=True)
    assert j.replay_done_uids() == {"u1"}


def test_a_rejected_line_cannot_cancel_an_accepted_one(tmp_path):
    """Order on disk must not matter: the fence can land after the real result."""
    path = tmp_path / "j.jsonl"
    j = Journal(path)
    j.record(task_uid="u1", yaml_id="y", phase="eval", status="done",
             success=True, attempt=2, accepted=True)
    j.record(task_uid="u1", yaml_id="y", phase="eval", status="failed",
             success=False, attempt=1, accepted=False)
    assert j.replay_done_uids() == {"u1"}


def test_records_predating_the_accepted_field_still_replay(tmp_path):
    """Absent means unknown, not rejected -- older ledgers must be unaffected."""
    path = tmp_path / "j.jsonl"
    j = Journal(path)
    j.record(task_uid="u1", yaml_id="y", phase="eval", status="done", success=True)
    assert j.replay_done_uids() == {"u1"}
