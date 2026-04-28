"""Unit tests for ``exp.verdict_factor_judge.per_step_log_writer``.

These tests pin the writer/pool contract that the LIBERO B1 client wiring
depends on (plan §7):

* Append-only line-buffered semantics so a worker crash mid-episode keeps
  every previously written row.
* Per-yaml merge sorted by ``(task_id, subset_init_state_idx, episode_id,
  step_idx)``.
* Idempotent finalize (eval_libero may call it from a ``finally`` block in
  addition to a normal return path).
* Stale temp file truncation so repeated runs of the same yaml do not double
  rows.
* Explicit ``install_atexit`` opt-in so pytest does not accumulate callbacks.
"""

from __future__ import annotations

import atexit
import json
import pathlib

import pytest

from exp.verdict_factor_judge.per_step_log_writer import (
    PerStepWriter,
    PerStepWriterPool,
    install_atexit,
)


def _make_row(
    task_id: int,
    subset_init_state_idx: int,
    episode_id: int,
    step_idx: int,
    *,
    yaml_id: str = "yaml_x",
    phase: str = "eval",
    hit_type: str = "WARM_START",
) -> dict:
    """Build a row matching the plan §7.2 schema."""
    return {
        "yaml_id": yaml_id,
        "task_id": int(task_id),
        "subset_init_state_idx": int(subset_init_state_idx),
        "orig_init_state_idx": int(subset_init_state_idx),
        "episode_id": int(episode_id),
        "step_idx": int(step_idx),
        "phase": phase,
        "hit_type": hit_type,
        "start_t": 0.7 if hit_type == "WARM_START" else None,
        "winner_id": f"ep_{episode_id}:{step_idx}" if hit_type != "MISS" else None,
        "cp1_score": 0.42 if hit_type != "MISS" else None,
    }


def test_writer_appends_lines(tmp_path: pathlib.Path) -> None:
    """Three rows in → exactly three JSON lines on disk → all parseable."""
    path = tmp_path / "x.jsonl"
    writer = PerStepWriter(path)
    rows = [_make_row(0, 0, 0, i) for i in range(3)]
    for row in rows:
        writer.write_row(row)
    writer.close()

    text = path.read_text(encoding="utf-8")
    lines = [ln for ln in text.split("\n") if ln]
    assert len(lines) == 3
    parsed = [json.loads(ln) for ln in lines]
    assert parsed == rows


def test_pool_finalize_merges_workers_sorted(tmp_path: pathlib.Path) -> None:
    """3 workers writing out-of-order rows merge into one sorted file."""
    pool = PerStepWriterPool(tmp_path, yaml_id="yaml_a", num_workers=3)
    w0 = pool.writer_for(0)
    w1 = pool.writer_for(1)
    w2 = pool.writer_for(2)

    # Deliberately scrambled across the 4-tuple sort key.
    w1.write_row(_make_row(task_id=2, subset_init_state_idx=0, episode_id=10, step_idx=3))
    w0.write_row(_make_row(task_id=0, subset_init_state_idx=1, episode_id=1, step_idx=0))
    w2.write_row(_make_row(task_id=0, subset_init_state_idx=0, episode_id=0, step_idx=2))
    w0.write_row(_make_row(task_id=0, subset_init_state_idx=0, episode_id=0, step_idx=0))
    w1.write_row(_make_row(task_id=0, subset_init_state_idx=0, episode_id=0, step_idx=1))
    w2.write_row(_make_row(task_id=2, subset_init_state_idx=0, episode_id=10, step_idx=0))

    merged = pool.finalize()
    assert merged == tmp_path / "yaml_a.jsonl"
    assert merged.exists()

    rows = [json.loads(ln) for ln in merged.read_text(encoding="utf-8").splitlines() if ln]
    sort_keys = [
        (r["task_id"], r["subset_init_state_idx"], r["episode_id"], r["step_idx"])
        for r in rows
    ]
    assert sort_keys == sorted(sort_keys)
    assert len(rows) == 6

    # Temp files must be cleaned up so reruns do not double rows.
    leftover = list(tmp_path.glob("_yaml_a.worker_*.jsonl"))
    assert leftover == []


def test_pool_finalize_idempotent(tmp_path: pathlib.Path) -> None:
    """finalize() twice → same path, no exception, content unchanged."""
    pool = PerStepWriterPool(tmp_path, yaml_id="yaml_b", num_workers=2)
    pool.writer_for(0).write_row(_make_row(0, 0, 0, 0))
    pool.writer_for(1).write_row(_make_row(0, 0, 1, 0))

    first = pool.finalize()
    snapshot = first.read_text(encoding="utf-8")
    second = pool.finalize()

    assert first == second
    assert second.read_text(encoding="utf-8") == snapshot


def test_pool_truncates_stale_temp_files(tmp_path: pathlib.Path) -> None:
    """Pre-existing temp file from a prior crashed run must be truncated."""
    stale = tmp_path / "_yaml_c.worker_0.jsonl"
    stale.write_text("garbage from previous run\n", encoding="utf-8")

    pool = PerStepWriterPool(tmp_path, yaml_id="yaml_c", num_workers=1)
    # Pool ctor truncates lazily-allocated temp files; after construction the
    # file should already be empty even before we open a writer.
    assert stale.read_text(encoding="utf-8") == ""

    pool.writer_for(0).write_row(_make_row(0, 0, 0, 0))
    merged = pool.finalize()

    rows = [json.loads(ln) for ln in merged.read_text(encoding="utf-8").splitlines() if ln]
    assert len(rows) == 1
    assert "garbage" not in merged.read_text(encoding="utf-8")


def test_pool_handles_zero_rows(tmp_path: pathlib.Path) -> None:
    """No writes → finalize still produces an empty merged file."""
    pool = PerStepWriterPool(tmp_path, yaml_id="yaml_d", num_workers=2)
    merged = pool.finalize()
    assert merged.exists()
    assert merged.read_text(encoding="utf-8") == ""


def test_writer_buffering_flushes_per_line(tmp_path: pathlib.Path) -> None:
    """Each row hits disk before close() — proves buffering=1 line-buffering."""
    path = tmp_path / "live.jsonl"
    writer = PerStepWriter(path)
    row = _make_row(0, 0, 0, 0)
    writer.write_row(row)

    # Open via a separate handle without closing the writer first.
    with open(path, "r", encoding="utf-8") as fh:
        content = fh.read()
    assert content.endswith("\n")
    assert json.loads(content.strip()) == row

    writer.close()


def test_install_atexit_registers_callback(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """install_atexit forwards the pool's finalize bound method to atexit."""
    pool = PerStepWriterPool(tmp_path, yaml_id="yaml_e", num_workers=1)
    captured: list = []

    def fake_register(func, *args, **kwargs):
        captured.append((func, args, kwargs))
        return func

    monkeypatch.setattr(atexit, "register", fake_register)
    install_atexit(pool)

    assert len(captured) == 1
    func, args, kwargs = captured[0]
    assert func == pool.finalize
    assert args == ()
    assert kwargs == {}
