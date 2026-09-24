"""CPU tests for the init_probe resume/merge helper (synthetic segments, no models)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from exp.step_diag.ops import resume_init_probe as R


def _journal(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _term(uid: str, attempt: int = 1, ok: bool = True) -> dict:
    return {"task_uid": uid, "attempt": attempt, "accepted": True, "status": "done" if ok else "failed", "success": ok}


def _segment(root: Path, name: str, probes: list[tuple[str, int, int, bytes]], sha: str = "c" * 64) -> None:
    p = root / name / "groot"
    (p / "probe").mkdir(parents=True)
    (p / "server").mkdir(parents=True)
    rows = []
    for uid, attempt, idx, payload in probes:
        arr = f"arrays_{hashlib.sha256(f'{uid}:{attempt}:{idx}'.encode()).hexdigest()[:24]}.npz"
        (p / "probe" / arr).write_bytes(payload)
        rows.append({"task_uid": uid, "attempt": attempt, "decision_idx": idx, "arrays": arr,
                     "arrays_sha256": hashlib.sha256(payload).hexdigest()})
    _journal(p / "probe/probe.jsonl", rows)
    _journal(p / "server/rows_x.jsonl", [{"task_uid": u, "attempt": a, "status": "finalize"} for u, a, _, _ in probes])
    (p / "server/manifest_shadow.json").write_text(json.dumps({"config_sha": sha}))
    (p / "probe/probe_contract.json").write_text(json.dumps({"decisions": [0, 5], "argv": []}))


def test_merge_keeps_only_accepted_segments_and_copies_arrays_unchanged(tmp_path):
    pilot = tmp_path / "pilot"
    # pilot: e0 completed; e1 was interrupted mid-episode (attempt 1, no terminal)
    _segment(tmp_path, "pilot", [("e0", 1, 0, b"A"), ("e0", 1, 5, b"B"), ("e1", 1, 0, b"STALE")])
    _journal(pilot / "groot/client_stopped/journal_x.jsonl", [_term("e0")])
    # resume: e1 reruns with the SAME (uid, attempt) key -> same array name as the stale one
    _segment(tmp_path, "resume_r1", [("e1", 1, 0, b"FRESH")])
    final = tmp_path / "final_client"
    _journal(final / "journal_x.jsonl", [_term("e0"), _term("e1", ok=False)])
    (final / "launch_a.json").write_text("{}")
    res = R.merge("groot", "resume_r1", final, out_root=tmp_path / "merged", pilot=pilot, seg_root=tmp_path / "resume_r1")
    assert res["pilot_episodes"] == 1 and res["resume_episodes"] == 1 and res["probe_rows"] == 3
    assert res["excluded_rows"] == {"pilot": 1, "resume_r1": 0}
    rows = [json.loads(line) for line in (tmp_path / "merged/groot/probe/probe.jsonl").read_text().splitlines()]
    fresh = next(r for r in rows if r["task_uid"] == "e1")
    assert fresh["arrays"].startswith("resume_r1/")
    assert (tmp_path / "merged/groot/probe" / fresh["arrays"]).read_bytes() == b"FRESH"
    assert all(hashlib.sha256((tmp_path / "merged/groot/probe" / r["arrays"]).read_bytes()).hexdigest() == r["arrays_sha256"]
               for r in rows)
    server = [json.loads(line) for f in (tmp_path / "merged/groot/server").glob("rows_*.jsonl")
              for line in f.read_text().splitlines()]
    assert sorted((r["task_uid"], r["attempt"]) for r in server) == [("e0", 1), ("e0", 1), ("e1", 1)]
    with pytest.raises(FileExistsError):
        R.merge("groot", "resume_r1", final, out_root=tmp_path / "merged", pilot=pilot, seg_root=tmp_path / "resume_r1")


def test_merge_refuses_a_changed_snapshot_terminal_and_a_config_change(tmp_path):
    pilot = tmp_path / "pilot"
    _segment(tmp_path, "pilot", [("e0", 1, 0, b"A")])
    _journal(pilot / "groot/client_stopped/journal_x.jsonl", [_term("e0")])
    _segment(tmp_path, "resume_r1", [], sha="d" * 64)
    final = tmp_path / "final_client"
    _journal(final / "journal_x.jsonl", [_term("e0", ok=False)])
    with pytest.raises(ValueError, match="changed or vanished"):
        R.merge("groot", "resume_r1", final, out_root=tmp_path / "m1", pilot=pilot, seg_root=tmp_path / "resume_r1")
    _journal(final / "journal_x.jsonl", [_term("e0")])
    with pytest.raises(ValueError, match="config_sha"):
        R.merge("groot", "resume_r1", final, out_root=tmp_path / "m2", pilot=pilot, seg_root=tmp_path / "resume_r1")


def test_resume_argv_replaces_only_the_evidence_location_and_launch_id():
    contract = {"argv": ["--mode", "shadow", "--diag-out", "old/server", "--launch-id", "pilot23158", "--", "--port", "23158"]}
    argv = R.resume_argv(contract, Path("new/groot"), "resume_r1_23158")
    assert argv == ["--mode", "shadow", "--diag-out", "new/groot/server", "--launch-id", "resume_r1_23158", "--", "--port", "23158"]
