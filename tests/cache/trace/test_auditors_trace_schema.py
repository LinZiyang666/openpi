"""Auditor branches for trace-mode files (plan §7.3, §12-H).

``verify_collection_artifacts._check_h5_schema`` and
``verify_shadow_h5.trace_problems``: a file without ``trace_schema_version``
keeps the legacy rules; version 1 must be committed (closed_ok, terminal, no
write errors) and carry the admitted identity; a diagnostic file (no loop
inputs) is fine for diagnostics and rejected for a warm-start corpus; an
unknown version is rejected; ``.h5.tmp`` / ``.h5.failed`` leftovers are
reported by the audit sweep and never admitted.
"""

from __future__ import annotations

import pathlib

import h5py
import numpy as np
import pytest

from exp.libero_groot.verify_shadow_h5 import trace_problems
from exp.robocasa365.verify_collection_artifacts import _check_h5_schema, audit
from openpi.cache.types import groot_n15_schedule
from tests.robocasa365.test_collection_artifacts import _journal_row, _run_plan, _write_h5

SCHEDULE = groot_n15_schedule(4)


def _stamp(path: pathlib.Path, *, version=1, closed=True, terminal=True, errors=0,
           noise=True, uid="OpenCabinet:0", attempt=1, snapshots=True) -> None:
    with h5py.File(path, "a") as f:
        f.attrs["denoise_schedule_id"] = SCHEDULE.schedule_id
        f.attrs["denoising_num_steps"] = SCHEDULE.num_steps
        f.attrs["trace_schema_version"] = version
        f.attrs["trace_closed_ok"] = closed
        f.attrs["trace_terminal"] = terminal
        f.attrs["trace_write_errors"] = errors
        f.attrs["trace_noise_actions_recorded"] = noise
        f.attrs["trace_task_uid"] = uid
        f.attrs["trace_attempt"] = attempt
        if snapshots:
            for name in f:
                if name.startswith("step_"):
                    g = f[name]
                    for i in range(SCHEDULE.num_steps):
                        g.create_dataset(f"noise_action_{i}", data=np.zeros((50, 32), np.float32))


def _file(tmp_path, **kw):
    path = _write_h5(tmp_path, "groot/OpenCabinet/episode_0000", 1, task="OpenCabinet", success=True, steps=1)
    _stamp(path, **kw)
    return path


def test_legacy_file_without_version_keeps_the_old_rules(tmp_path):
    path = _write_h5(tmp_path, "groot/OpenCabinet/episode_0000", 1, task="OpenCabinet", success=True)
    assert _check_h5_schema(path, "OpenCabinet") == []
    assert _check_h5_schema(path, "OpenCabinet", expected_identity=("x", 9)) == []
    assert trace_problems(None) == []


def test_committed_build_file_passes_with_identity(tmp_path):
    path = _file(tmp_path)
    assert _check_h5_schema(path, "OpenCabinet", require_schedule=True,
                            expected_identity=("OpenCabinet:0", 1)) == []


@pytest.mark.parametrize(
    "kw,needle",
    [
        (dict(closed=False), "trace_closed_ok"),
        (dict(terminal=False), "trace_terminal"),
        (dict(errors=2), "trace_write_errors=2"),
        (dict(version=7), "unknown trace_schema_version 7"),
        (dict(uid="Other:1"), "trace identity"),
        (dict(attempt=2), "trace identity"),
    ],
)
def test_uncommitted_or_misidentified_trace_files_are_rejected(tmp_path, kw, needle):
    path = _file(tmp_path, **kw)
    problems = _check_h5_schema(path, "OpenCabinet", require_schedule=True,
                                expected_identity=("OpenCabinet:0", 1))
    assert any(needle in p for p in problems), problems


def test_diagnostic_file_is_fine_for_diagnostics_and_refused_for_warm_start(tmp_path):
    path = _file(tmp_path, noise=False, snapshots=False)
    assert _check_h5_schema(path, "OpenCabinet", expected_identity=("OpenCabinet:0", 1)) == []
    problems = _check_h5_schema(path, "OpenCabinet", require_schedule=True,
                                expected_identity=("OpenCabinet:0", 1))
    assert any("recorded no noise_action" in p for p in problems), problems


def test_build_file_missing_a_snapshot_is_rejected(tmp_path):
    path = _file(tmp_path)
    with h5py.File(path, "a") as f:
        del f["step_0000"]["noise_action_2"]
    problems = _check_h5_schema(path, "OpenCabinet", require_schedule=True)
    assert any("noise_action indices" in p for p in problems), problems


def test_shadow_auditor_trace_branch():
    ok = {"version": 1, "closed_ok": True, "terminal": True, "write_errors": 0, "noise_recorded": True}
    assert trace_problems(ok) == []
    assert trace_problems({**ok, "version": 3}) == ["unknown trace_schema_version 3"]
    bad = trace_problems({**ok, "closed_ok": False, "terminal": False, "write_errors": 1, "noise_recorded": False})
    assert len(bad) == 4


def test_audit_reports_tmp_and_failed_leftovers_and_checks_identity(tmp_path):
    plan = _run_plan(tmp_path, tasks=[("OpenCabinet", 1)])
    from exp.robocasa365.verify_collection_artifacts import merge_run_plans

    expected, prefixes, _batches, _hashes = merge_run_plans([plan])
    (uid,) = sorted(expected)
    root = tmp_path / "data"
    path = _write_h5(root, prefixes[uid], 1, task="OpenCabinet", success=True, steps=1)
    _stamp(path, uid=uid, attempt=1)
    leftover = root / f"{prefixes[uid]}_a02.h5.tmp"
    leftover.write_bytes(b"")
    failed = root / f"{prefixes[uid]}_a03.h5.failed"
    failed.write_bytes(b"")
    report = audit(root=root, journal_records=[_journal_row(uid, attempt=1)], plans=[plan], target=1,
                   require_denoise_schedule=True)
    assert report["ok"], report
    assert set(report["unfinished_files"]) == {
        str(leftover.relative_to(root)), str(failed.relative_to(root))
    }
    # A file whose embedded identity names another attempt is not admitted.
    _stamp(path, uid=uid, attempt=2, snapshots=False)
    report = audit(root=root, journal_records=[_journal_row(uid, attempt=1)], plans=[plan], target=1,
                   require_denoise_schedule=True)
    assert not report["ok"] and uid in report["schema_errors"]


@pytest.mark.parametrize("trace", [True, False])
def test_failed_audit_manifest_is_whole_batch_for_trace_and_per_episode_for_legacy(tmp_path, trace):
    """Plan §7.3: a trace batch is admitted whole or not at all; a legacy-schema
    batch keeps its per-episode rejection (the admitted episodes are listed)."""
    from exp.robocasa365.verify_collection_artifacts import build_manifest, merge_run_plans

    plan = _run_plan(tmp_path, tasks=[("OpenCabinet", 2)])
    expected, prefixes, _batches, _hashes = merge_run_plans([plan])
    good, bad = sorted(expected)
    root = tmp_path / "data"
    for uid in (good, bad):
        # The bad episode's file claims a different task: rejected under either schema.
        path = _write_h5(root, prefixes[uid], 1, task="OpenCabinet" if uid == good else "CloseDrawer",
                         success=True, steps=1)
        if trace:
            _stamp(path, uid=uid, attempt=1)
    rows = [_journal_row(good, attempt=1), _journal_row(bad, attempt=1)]
    report = audit(root=root, journal_records=rows, plans=[plan], target=1, require_denoise_schedule=trace)
    assert not report["ok"] and bad in report["schema_errors"] and good in report["admitted"]
    assert report["trace_batch"] is trace
    if trace:
        with pytest.raises(ValueError, match="failed audit of a trace batch"):
            build_manifest(report, root=root, target=1)
    else:
        manifest = build_manifest(report, root=root, target=1)
        assert [row["task_uid"] for rows in manifest["tasks"].values() for row in rows] == [good]
    # A report that does not declare its batch schema is treated as a trace batch.
    undeclared = {k: v for k, v in report.items() if k != "trace_batch"}
    with pytest.raises(ValueError, match="failed audit"):
        build_manifest(undeclared, root=root, target=1)


def test_batch_classification_reads_every_file_and_legacy_tmp_headers(tmp_path):
    """Plan §7.3: whether a batch is trace is decided from every file on disk,
    admitted or not, and a legacy collector's ``.h5.tmp`` is judged by its header."""
    from exp.robocasa365.verify_collection_artifacts import merge_run_plans

    plan = _run_plan(tmp_path, tasks=[("OpenCabinet", 2)])
    expected, prefixes, _batches, _hashes = merge_run_plans([plan])
    good, unjournaled = sorted(expected)
    root = tmp_path / "data"
    _write_h5(root, prefixes[good], 1, task="OpenCabinet", success=True, steps=1)
    rows = [_journal_row(good, attempt=1)]

    # A trace file whose uid has no journal row never reaches admission, yet
    # it makes the batch a trace batch.
    trace_file = _write_h5(root, prefixes[unjournaled], 1, task="OpenCabinet", success=True, steps=1)
    _stamp(trace_file, uid=unjournaled, attempt=1)
    report = audit(root=root, journal_records=rows, plans=[plan], target=1)
    assert unjournaled in report["missing_terminal"] and report["trace_batch"] is True
    trace_file.unlink()

    # A readable legacy temporary file keeps the legacy semantics ...
    legacy = _write_h5(root, prefixes[unjournaled], 2, task="OpenCabinet", success=True, steps=1)
    legacy_tmp = legacy.rename(legacy.with_suffix(".h5.tmp"))
    report = audit(root=root, journal_records=rows, plans=[plan], target=1)
    assert report["unfinished_files"] and report["trace_batch"] is False
    # ... an unreadable one is not evidence of a legacy batch.
    legacy_tmp.write_bytes(b"")
    report = audit(root=root, journal_records=rows, plans=[plan], target=1)
    assert report["trace_batch"] is True
