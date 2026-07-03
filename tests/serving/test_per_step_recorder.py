"""Tests for the generalized per-step recorder (two-mode)."""

import json
import pathlib

from openpi.serving.per_step_recorder import PerStepWriter
from openpi.serving.per_step_recorder import PerStepWriterPool
from openpi.serving.per_step_recorder import filter_searched
from openpi.serving.per_step_recorder import summarize_gate_log


def _read(path) -> list[dict]:
    text = pathlib.Path(path).read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_summarize_gate_log_excludes_episode_summary_row(tmp_path):
    """n_eval_verdicts must count only real verdict rows; the per-episode
    ``episode_summary`` row (phase==eval but no hit_type) must NOT inflate the
    inference-ratio denominator (G2 R9)."""
    _write_jsonl(tmp_path / "y.jsonl", [
        {"phase": "eval", "hit_type": "MISS", "step_idx": 0},
        {"phase": "eval", "hit_type": "FULL_HIT", "step_idx": 1},
        {"phase": "eval", "hit_type": "WARM_START", "step_idx": 2},
        # Provenance row: phase==eval, _kind set, NO hit_type -> excluded.
        {"_kind": "episode_summary", "phase": "eval", "success": True, "num_steps": 3},
        # Warmup rows are ignored regardless of hit_type.
        {"phase": "warmup", "hit_type": "MISS", "step_idx": 0},
    ])
    counts = summarize_gate_log(str(tmp_path), "y")
    assert counts["n_eval_verdicts"] == 3  # not 4 (summary excluded), not 5 (warmup excluded)
    assert counts["n_full_hit"] == 1
    assert counts["n_warm_start"] == 1
    assert counts["n_miss"] == 1
    # Invariant the downstream inference ratio relies on.
    assert counts["n_eval_verdicts"] == (
        counts["n_full_hit"] + counts["n_warm_start"] + counts["n_miss"]
    )


def test_summarize_gate_log_missing_dir_is_zero(tmp_path):
    """Best-effort: absent dir/file yields zero counts (collection disabled)."""
    assert summarize_gate_log("", "y")["n_eval_verdicts"] == 0
    assert summarize_gate_log(str(tmp_path), "nope")["n_eval_verdicts"] == 0


def test_shim_mode_never_injects_success(tmp_path):
    """stamp_success=False keeps rows byte-identical (verdict_factor compat)."""
    p = tmp_path / "w.jsonl"
    w = PerStepWriter(p, stamp_success=False)
    rows = [{"task_id": 1, "step_idx": 0, "hit_type": "MISS"}]
    for r in rows:
        w.write_row(dict(r))
    n = w.flush_episode()  # no-arg
    w.close()
    assert n == 1
    assert _read(p) == rows  # no "success" key injected


def test_gate_mode_stamps_success(tmp_path):
    p = tmp_path / "w.jsonl"
    w = PerStepWriter(p, stamp_success=True)
    w.begin_episode()
    w.write_row({"step_idx": 0})
    w.write_row({"step_idx": 1})
    w.flush_episode(success=True)
    w.close()
    out = _read(p)
    assert len(out) == 2
    assert all(r["success"] is True for r in out)


def test_gate_mode_tail_flush_null_on_crash(tmp_path):
    """close() on an in-flight episode stamps success=None, never False."""
    p = tmp_path / "w.jsonl"
    w = PerStepWriter(p, stamp_success=True)
    w.begin_episode()
    w.write_row({"step_idx": 0})
    w.close()  # no flush_episode -> crash between episodes
    out = _read(p)
    assert len(out) == 1
    assert out[0]["success"] is None


def test_pool_finalize_merges_sorted(tmp_path):
    pool = PerStepWriterPool(tmp_path, "yaml_x", num_workers=2, stamp_success=True)
    w0 = pool.writer_for(0)
    w1 = pool.writer_for(1)
    w1.begin_episode()
    w1.write_row({"task_id": 1, "subset_init_state_idx": 0, "episode_id": 0, "step_idx": 1})
    w1.flush_episode(success=True)
    w0.begin_episode()
    w0.write_row({"task_id": 1, "subset_init_state_idx": 0, "episode_id": 0, "step_idx": 0})
    w0.flush_episode(success=False)
    merged = pool.finalize()
    out = _read(merged)
    assert [r["step_idx"] for r in out] == [0, 1]  # sorted by _DEFAULT_SORT_KEYS


def test_filter_searched():
    rows = [{"searched": True, "x": 1}, {"searched": False, "x": 2}, {"x": 3}]
    assert [r["x"] for r in filter_searched(rows)] == [1, 3]


def test_pool_gate_mode_finalize_drops_searched_false(tmp_path):
    """filter_searched=True wires the C5 filter into finalize (not just a helper)."""
    pool = PerStepWriterPool(
        tmp_path, "y", num_workers=1, stamp_success=True, filter_searched=True
    )
    w = pool.writer_for(0)
    w.begin_episode()
    w.write_row({"task_id": 1, "subset_init_state_idx": 0, "episode_id": 0, "step_idx": 0, "searched": True})
    w.write_row({"task_id": 1, "subset_init_state_idx": 0, "episode_id": 0, "step_idx": 1, "searched": False})
    w.flush_episode(success=True)
    out = _read(pool.finalize())
    assert [r["step_idx"] for r in out] == [0]  # gate-skip row dropped at merge


def test_pool_default_keeps_all_rows(tmp_path):
    """Without filter_searched the merge keeps everything (shim/verdict_factor)."""
    pool = PerStepWriterPool(tmp_path, "y", num_workers=1)  # filter_searched=False
    w = pool.writer_for(0)
    w.write_row({"task_id": 1, "subset_init_state_idx": 0, "episode_id": 0, "step_idx": 0, "searched": False})
    w.flush_episode()
    assert len(_read(pool.finalize())) == 1
