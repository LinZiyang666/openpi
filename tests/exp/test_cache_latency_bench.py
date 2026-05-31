"""Tests for the cache latency replay benchmark (``exp/cache_latency_bench``).

All synthetic, CPU-only, non-manual: no model, no GPU, no network. Mini HDF5
trajectories + mini InMemoryBackend pkl artifacts + minimal cache YAMLs drive a
real ``CacheOrchestrator`` through ``ReplayHarness``.
"""

from __future__ import annotations

import pickle

import h5py
import numpy as np
import pytest
import torch
import yaml

from openpi.cache.config import ConfigValidationError
from openpi.cache.storage_types import CacheEntry, CachePayload
from openpi.cache.types import CheckpointID
from openpi.serving import monitor

from exp.cache_latency_bench.h5_episode import H5EpisodeSource
from exp.cache_latency_bench.replay import ReplayHarness
from exp.cache_latency_bench.summarize import summarize

# Mini-data dimensions (kept small for fast CI; vision must be 256 tokens so the
# fake-Stage1 token layout is valid, but emb_dim can be tiny).
_EMB_DIM = 8
_VISION_TOKENS = 256
_PROMPT_TOKENS = 4
_STATE_DIM = 4
_ACTION_H = 4
_ACTION_D = 4


# ----------------------------------------------------------------------
# Mini-data builders
# ----------------------------------------------------------------------


def _write_mini_h5(path, *, n_steps, task, episode_id, success, state_fn) -> None:
    with h5py.File(path, "w") as f:
        f.attrs["task"] = task
        f.attrs["episode_id"] = episode_id
        f.attrs["success"] = success
        f.attrs["num_steps"] = n_steps
        for s in range(n_steps):
            g = f.create_group(f"step_{s:04d}")
            zeros = np.zeros((_VISION_TOKENS, _EMB_DIM), dtype=np.float16)
            g.create_dataset("vision_0", data=zeros)
            g.create_dataset("vision_1", data=zeros)
            g.create_dataset(
                "prompt_emb",
                data=np.zeros((_PROMPT_TOKENS, _EMB_DIM), dtype=np.float16),
            )
            g.create_dataset(
                "robot_state", data=np.asarray(state_fn(s), dtype=np.float32)
            )
            g.create_dataset(
                "clean_action",
                data=np.full((_ACTION_H, _ACTION_D), 0.1 * s + 0.1, dtype=np.float32),
            )


def _entry(
    eid,
    state_vec,
    *,
    action_val,
    factors=None,
    intermediates=None,
    denoise=None,
    prev_ids=None,
    next_ids=None,
    traj=None,
    task_key="taskA",
) -> CacheEntry:
    payload = CachePayload(
        action_chunk=torch.full(
            (_ACTION_H, _ACTION_D), float(action_val), dtype=torch.float32
        ),
        intermediates=intermediates,
        denoising_num_steps=denoise,
        factors=factors,
        task_key=task_key,
    )
    return CacheEntry(
        id=eid,
        checkpoint_id=CheckpointID.CP1,
        query_keys={
            "robot_state": torch.as_tensor(state_vec, dtype=torch.float32).contiguous()
        },
        payload=payload,
        prev_ids=prev_ids or [],
        next_ids=next_ids or [],
        trajectory_id=traj,
    )


def _write_pkl(path, entries, *, dim=_STATE_DIM, library_stats=None) -> None:
    data = {
        "key_builder_type": "placeholder",
        "checkpoint_id": "CP1",
        "vector_dims": {"robot_state": dim},
        "entries": entries,
    }
    if library_stats is not None:
        data["library_stats"] = library_stats
    with open(path, "wb") as f:
        pickle.dump(data, f)


def _onehot(i, dim=_STATE_DIM):
    v = np.zeros(dim, dtype=np.float32)
    v[i % dim] = 1.0
    return v


def _simple_library(pkl_path, *, n=3):
    """A small library of distinct one-hot robot_state entries (no factors)."""
    entries = [_entry(f"e{i}", _onehot(i), action_val=0.1 * (i + 1)) for i in range(n)]
    _write_pkl(pkl_path, entries)


# ----------------------------------------------------------------------
# Cache YAML builders
# ----------------------------------------------------------------------


def _keys_block():
    return {
        "vision_0": {"enabled": False, "weight": 0.0},
        "vision_1": {"enabled": False, "weight": 0.0},
        "vision_2": {"enabled": False, "weight": 0.0},
        "prompt_emb": {"enabled": False, "weight": 0.0},
        "robot_state": {"enabled": True, "weight": 1.0},
    }


def _backend_block(pkl_path, dim=_STATE_DIM):
    return {
        "type": "in_memory",
        "vector_dims": {"robot_state": dim},
        "in_memory": {"preload_path": str(pkl_path), "index_type": "brute_force"},
    }


def _rrf_strategy():
    return {
        "type": "weighted_rrf_knn",
        "top_k": 1,
        "step_filter": "all",
        "rrf_k": 60,
        "trajectory_depth": 1,
        "field_similarity": {"robot_state": {"type": "cosine"}},
    }


def _score_sum_strategy():
    return {
        "type": "weighted_score_sum_knn",
        "top_k": 1,
        "step_filter": "all",
        "trajectory_depth": 1,
        "field_similarity": {"robot_state": {"type": "cosine"}},
        "score_normalization": {
            "type": "per_field",
            "fields": {
                "robot_state": {
                    "method": "affine_clip",
                    "params": {"lo": 0.0, "hi": 1.0},
                }
            },
        },
    }


def _write_yaml(
    path,
    *,
    pkl_path,
    gate="always_search",
    judge=None,
    strategy=None,
    write_policy="never",
    dim=_STATE_DIM,
):
    cfg = {
        "enabled": True,
        "timer": {"enabled": False, "buffer_size": 10000, "output_csv_dir": None},
        "keys": _keys_block(),
        "key_builder": {"type": "placeholder"},
        "checkpoints": {
            "cp1": {
                "enabled": True,
                "gate": {"type": gate},
                "judge": judge or {"type": "always_hit"},
                "search_strategy": strategy or _rrf_strategy(),
            }
        },
        "backend": _backend_block(pkl_path, dim),
        "write_policy": {"type": write_policy},
    }
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f)
    return str(path)


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _restore_monitor_level():
    """Snapshot + restore the process-wide monitor level around each test.

    ``ReplayHarness`` auto-elevates the level; restore it so tests don't leak.
    """
    saved = monitor.get_monitor_level()
    try:
        yield
    finally:
        monitor.set_monitor_level(saved)


# ----------------------------------------------------------------------
# 1. H5 parsing
# ----------------------------------------------------------------------


def test_h5_episode_source_parses_meta_and_shapes(tmp_path):
    h5 = tmp_path / "episode_0000.h5"
    _write_mini_h5(
        h5, n_steps=3, task="taskA", episode_id=7, success=True, state_fn=_onehot
    )
    src = H5EpisodeSource(str(tmp_path))
    assert len(src) == 1
    (ep,) = list(src.iter_episodes())
    assert ep.task_key == "taskA"
    assert ep.episode_id == 7
    assert ep.success is True
    assert ep.num_steps == 3
    steps = list(ep.steps())
    assert [s.step_idx for s in steps] == [0, 1, 2]
    s0 = steps[0]
    assert s0.fake_stage1.state.shape == (1, _STATE_DIM)
    assert s0.fake_stage1.prefix_embs.shape[0] == 1
    assert s0.clean_action.shape == (_ACTION_H, _ACTION_D)


def test_h5_episode_source_empty_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        H5EpisodeSource(str(tmp_path))


# ----------------------------------------------------------------------
# 2. YAML assembly + write_policy fail-fast contract
# ----------------------------------------------------------------------


def test_assembly_with_never_policy_succeeds(tmp_path):
    pkl = tmp_path / "lib.pkl"
    _simple_library(pkl)
    yml = _write_yaml(tmp_path / "c.yaml", pkl_path=pkl, write_policy="never")
    harness = ReplayHarness(yml)
    assert harness._orchestrator is not None
    assert harness._timing_ready is True


def test_assembly_with_non_never_policy_fails_fast(tmp_path):
    pkl = tmp_path / "lib.pkl"
    _simple_library(pkl)
    yml = _write_yaml(tmp_path / "c.yaml", pkl_path=pkl, write_policy="on_any_miss")
    with pytest.raises(ConfigValidationError):
        ReplayHarness(yml)


# ----------------------------------------------------------------------
# 3. Lifecycle order (spy)
# ----------------------------------------------------------------------


def test_lifecycle_order(tmp_path, monkeypatch):
    pkl = tmp_path / "lib.pkl"
    _simple_library(pkl)
    h5 = tmp_path / "episode_0000.h5"
    _write_mini_h5(
        h5, n_steps=2, task="taskA", episode_id=0, success=True, state_fn=_onehot
    )
    yml = _write_yaml(tmp_path / "c.yaml", pkl_path=pkl)
    harness = ReplayHarness(yml)

    calls = []
    orch = harness._orchestrator
    for name in (
        "on_task_begin",
        "on_episode_start",
        "check",
        "broadcast_action",
        "buffer_for_write",
        "clear",
        "on_episode_end",
        "on_task_end",
    ):
        orig = getattr(orch, name)

        def spy(*a, _n=name, _o=orig, **k):
            calls.append(_n)
            return _o(*a, **k)

        monkeypatch.setattr(orch, name, spy)

    harness.run(
        H5EpisodeSource(str(tmp_path)), repeats=1, out_csv=str(tmp_path / "out.csv")
    )

    assert calls[0] == "on_task_begin"
    assert calls[1] == "on_episode_start"
    assert calls[-1] == "on_task_end"
    assert calls[-2] == "on_episode_end"
    # Each step is check -> broadcast_action -> buffer_for_write -> clear, in order.
    step_seq = [
        c
        for c in calls
        if c in ("check", "broadcast_action", "buffer_for_write", "clear")
    ]
    assert step_seq == ["check", "broadcast_action", "buffer_for_write", "clear"] * 2


# ----------------------------------------------------------------------
# 4. Per-step segment slicing (SEARCH = 6 segs, gate-skip = 3 segs)
# ----------------------------------------------------------------------


def _run_and_read_rows(tmp_path, yml, *, n_steps=2, task="taskA", state_fn=_onehot):
    h5 = tmp_path / "episode_0000.h5"
    _write_mini_h5(
        h5, n_steps=n_steps, task=task, episode_id=0, success=True, state_fn=state_fn
    )
    out_csv = tmp_path / "out.csv"
    summary = ReplayHarness(yml).run(
        H5EpisodeSource(str(tmp_path)), repeats=1, out_csv=str(out_csv)
    )
    import csv

    with open(out_csv, newline="") as fh:
        rows = list(csv.DictReader(fh))
    return rows, summary


def test_per_step_search_has_six_segments(tmp_path):
    pkl = tmp_path / "lib.pkl"
    _simple_library(pkl)
    yml = _write_yaml(
        tmp_path / "c.yaml",
        pkl_path=pkl,
        gate="always_search",
        judge={"type": "always_hit"},
    )
    rows, _ = _run_and_read_rows(tmp_path, yml)
    for row in rows:
        assert row["hit_type"] == "FULL_HIT"
        for seg in (
            "cp1_collect_ms",
            "cp1_gate_ms",
            "cp1_build_ms",
            "cp1_search_ms",
            "cp1_judge_ms",
            "cp1_fetch_ms",
        ):
            assert row[seg] != "", f"{seg} should be recorded on a FULL_HIT step"
        total = float(row["cp1_total_ms"])
        seg_sum = sum(
            float(row[s])
            for s in (
                "cp1_collect_ms",
                "cp1_gate_ms",
                "cp1_build_ms",
                "cp1_search_ms",
                "cp1_judge_ms",
                "cp1_fetch_ms",
            )
        )
        assert total >= seg_sum - 1e-3


def test_per_step_gate_skip_has_three_segments(tmp_path):
    pkl = tmp_path / "lib.pkl"
    _simple_library(pkl)
    yml = _write_yaml(tmp_path / "c.yaml", pkl_path=pkl, gate="always_skip")
    rows, _ = _run_and_read_rows(tmp_path, yml)
    for row in rows:
        assert row["hit_type"] == "MISS"
        for seg in ("cp1_collect_ms", "cp1_gate_ms", "cp1_build_ms"):
            assert row[seg] != ""
        for seg in ("cp1_search_ms", "cp1_judge_ms", "cp1_fetch_ms"):
            assert row[seg] == "", f"{seg} should be empty on a gate-skip step"


# ----------------------------------------------------------------------
# 5. Library frozen (count unchanged after a full run)
# ----------------------------------------------------------------------


def test_library_frozen_after_run(tmp_path):
    pkl = tmp_path / "lib.pkl"
    _simple_library(pkl, n=3)
    yml = _write_yaml(tmp_path / "c.yaml", pkl_path=pkl)
    harness = ReplayHarness(yml)
    before = harness._orchestrator._storage.count()
    h5 = tmp_path / "episode_0000.h5"
    _write_mini_h5(
        h5, n_steps=3, task="taskA", episode_id=0, success=True, state_fn=_onehot
    )
    harness.run(
        H5EpisodeSource(str(tmp_path)), repeats=1, out_csv=str(tmp_path / "out.csv")
    )
    assert harness._orchestrator._storage.count() == before == 3


# ----------------------------------------------------------------------
# 6. in-DB FULL_HIT (+fetch) vs out-of-DB MISS (no fetch)
# ----------------------------------------------------------------------


def test_in_db_vs_out_of_db_fetch_divergence(tmp_path):
    pkl = tmp_path / "lib.pkl"
    _simple_library(pkl, n=3)
    yml = _write_yaml(
        tmp_path / "c.yaml",
        pkl_path=pkl,
        judge={"type": "threshold", "threshold": 0.5},
        strategy=_score_sum_strategy(),
    )

    def state_fn(s):
        # step0: exact match of e0 (cosine 1 -> FULL_HIT); step1: far vector
        # (cosine <= 0 after affine_clip -> below threshold -> MISS).
        return _onehot(0) if s == 0 else np.full(_STATE_DIM, -1.0, dtype=np.float32)

    rows, _ = _run_and_read_rows(tmp_path, yml, n_steps=2, state_fn=state_fn)
    # search runs on both steps; only the fetch segment diverges with hit/miss.
    assert rows[0]["cp1_search_ms"] != "" and rows[1]["cp1_search_ms"] != ""
    assert rows[0]["hit_type"] == "FULL_HIT"
    assert rows[0]["cp1_fetch_ms"] != ""  # fetch fires on a hit
    assert rows[1]["hit_type"] == "MISS"
    assert rows[1]["cp1_fetch_ms"] == ""  # no fetch on a miss


# ----------------------------------------------------------------------
# 7. monitor level auto-elevate (guarantees timing data)
# ----------------------------------------------------------------------


def test_monitor_level_auto_elevate(tmp_path):
    pkl = tmp_path / "lib.pkl"
    _simple_library(pkl)
    yml = _write_yaml(tmp_path / "c.yaml", pkl_path=pkl)
    monitor.set_monitor_level(monitor.MonitorLevel.OFF)
    harness = ReplayHarness(yml)
    assert monitor.get_monitor_level() >= monitor.MonitorLevel.BASIC
    assert harness._timing_ready is True


# ----------------------------------------------------------------------
# 8. repeats comparable (same segment structure across repeats)
# ----------------------------------------------------------------------


def test_repeats_same_structure(tmp_path):
    pkl = tmp_path / "lib.pkl"
    _simple_library(pkl)
    yml = _write_yaml(tmp_path / "c.yaml", pkl_path=pkl)
    rows, summary = _run_and_read_rows_repeats(tmp_path, yml, repeats=2, n_steps=2)
    assert summary["n_steps"] == 4
    by_repeat = {"0": [], "1": []}
    for row in rows:
        by_repeat[row["repeat"]].append(row["hit_type"])
    assert by_repeat["0"] == by_repeat["1"]


def _run_and_read_rows_repeats(tmp_path, yml, *, repeats, n_steps):
    h5 = tmp_path / "episode_0000.h5"
    _write_mini_h5(
        h5, n_steps=n_steps, task="taskA", episode_id=0, success=True, state_fn=_onehot
    )
    out_csv = tmp_path / "out.csv"
    summary = ReplayHarness(yml).run(
        H5EpisodeSource(str(tmp_path)), repeats=repeats, out_csv=str(out_csv)
    )
    import csv

    with open(out_csv, newline="") as fh:
        rows = list(csv.DictReader(fh))
    return rows, summary


# ----------------------------------------------------------------------
# 10. on_episode_end no-arg contract
# ----------------------------------------------------------------------


def test_on_episode_end_called_without_success(tmp_path, monkeypatch):
    pkl = tmp_path / "lib.pkl"
    _simple_library(pkl)
    h5 = tmp_path / "episode_0000.h5"
    _write_mini_h5(
        h5, n_steps=1, task="taskA", episode_id=0, success=True, state_fn=_onehot
    )
    yml = _write_yaml(tmp_path / "c.yaml", pkl_path=pkl)
    harness = ReplayHarness(yml)
    seen = []
    orig = harness._orchestrator.on_episode_end

    def spy(*a, **k):
        seen.append((a, k))
        return orig(*a, **k)

    monkeypatch.setattr(harness._orchestrator, "on_episode_end", spy)
    harness.run(
        H5EpisodeSource(str(tmp_path)), repeats=1, out_csv=str(tmp_path / "out.csv")
    )
    assert seen == [((), {})]  # no positional / keyword args (no success)


# ----------------------------------------------------------------------
# 11. composite verdict support range (harness self-validation)
# ----------------------------------------------------------------------


_OFFLINE_KEY = "jerk_offline_action__p1_f1"
_ONLINE_KEY = "jerk_online_action__p1_f1"


def _composite_judge_block(calib_path):
    return {
        "type": "composite",
        "normalization": {
            "type": "zscore",
            "stats_source": {"type": "offline", "offline": {}},
        },
        "factors": [
            {
                "type": "jerk_online_action",
                "params": {"windows": [{"past": 1, "future": 1}]},
            },
            {
                "type": "jerk_offline_action",
                "params": {"windows": [{"past": 1, "future": 1}]},
            },
        ],
        "calibration": {
            "type": "percentile_rolling",
            "params": {"window_size": 1},
            "samples_source": {
                "type": "offline",
                "offline": {"path": str(calib_path), "format": "jsonl"},
            },
        },
        "composer": {
            "type": "weighted_sum_zero_nan",
            "weights": {_ONLINE_KEY: 1.0, _OFFLINE_KEY: 0.0},
            "tier_thresholds": {"full_hit": 1.01, "warm_start": 0.0},
            "warm_start_t": 0.5,
        },
    }


def _write_calib(path):
    import json

    with open(path, "w") as f:
        f.write(
            json.dumps({"factor_raw": {_ONLINE_KEY: 0.5, _OFFLINE_KEY: 0.5}}) + "\n"
        )


def _library_stats():
    from openpi.cache.components.factors.base import LibraryStats

    return LibraryStats(
        action_sigma=torch.ones(_ACTION_D, dtype=torch.float32),
        action_active_mask=torch.ones(_ACTION_D, dtype=torch.bool),
        state_sigma=torch.ones(_STATE_DIM, dtype=torch.float32),
        state_active_mask=torch.ones(_STATE_DIM, dtype=torch.bool),
    )


def _enriched_chain_library(pkl_path):
    """Two-entry chain; winner e0 carries factors + intermediates for WARM_START."""
    factors = {_OFFLINE_KEY: 0.5}
    e0 = _entry(
        "e0",
        _onehot(0),
        action_val=0.0,
        factors=factors,
        intermediates={0.5: torch.zeros(_ACTION_H, _ACTION_D, dtype=torch.float32)},
        denoise=10,
        next_ids=["e1"],
        traj="t0",
    )
    e1 = _entry(
        "e1", _onehot(1), action_val=1.0, factors=factors, prev_ids=["e0"], traj="t0"
    )
    _write_pkl(pkl_path, [e0, e1], library_stats=_library_stats())


def test_composite_warm_start_flags(tmp_path):
    pkl = tmp_path / "lib.pkl"
    _enriched_chain_library(pkl)
    calib = tmp_path / "calib.jsonl"
    _write_calib(calib)
    yml = _write_yaml(
        tmp_path / "c.yaml", pkl_path=pkl, judge=_composite_judge_block(calib)
    )
    # winner e0 carries intermediates[0.5]; composer full_hit=1.01 is unreachable, so
    # every matching step lands in the warm tier -> WARM_START.
    rows, summary = _run_and_read_rows(
        tmp_path, yml, n_steps=2, state_fn=lambda s: _onehot(0)
    )
    assert summary["judge_consumes_action_history"] is True
    assert all(r["hit_type"] == "WARM_START" for r in rows)
    assert all(r["warm_start_action_approx"] == "1" for r in rows)
    # cumulative flag: false on the first step, true once a prior non-FULL_HIT exists.
    assert rows[0]["action_history_approx_active"] == "0"
    assert rows[1]["action_history_approx_active"] == "1"


def test_warm_start_without_action_history_flags_false(tmp_path):
    # always_warm_start does NOT read action-history -> all three flags stay false
    # even though every step is WARM_START (control for test_composite_warm_start_flags).
    pkl = tmp_path / "lib.pkl"
    _enriched_chain_library(pkl)
    yml = _write_yaml(
        tmp_path / "c.yaml",
        pkl_path=pkl,
        judge={"type": "always_warm_start", "start_t": 0.5},
    )
    rows, summary = _run_and_read_rows(
        tmp_path, yml, n_steps=2, state_fn=lambda s: _onehot(0)
    )
    assert summary["judge_consumes_action_history"] is False
    assert all(r["hit_type"] == "WARM_START" for r in rows)
    assert all(r["warm_start_action_approx"] == "0" for r in rows)
    assert all(r["action_history_approx_active"] == "0" for r in rows)


def test_unenriched_artifact_fails_loud(tmp_path):
    # e1 is missing the offline factor key -> harness must catch it (full traversal).
    factors = {_OFFLINE_KEY: 0.5}
    e0 = _entry(
        "e0", _onehot(0), action_val=0.0, factors=factors, next_ids=["e1"], traj="t0"
    )
    e1 = _entry(
        "e1", _onehot(1), action_val=1.0, factors=None, prev_ids=["e0"], traj="t0"
    )
    pkl = tmp_path / "lib.pkl"
    _write_pkl(pkl, [e0, e1], library_stats=_library_stats())
    calib = tmp_path / "calib.jsonl"
    _write_calib(calib)
    yml = _write_yaml(
        tmp_path / "c.yaml", pkl_path=pkl, judge=_composite_judge_block(calib)
    )
    with pytest.raises(RuntimeError, match="not enriched"):
        ReplayHarness(yml)


def test_unenriched_artifact_all_missing_fails_loud(tmp_path):
    # every entry missing the offline factor key (not just a partial subset).
    e0 = _entry(
        "e0", _onehot(0), action_val=0.0, factors=None, next_ids=["e1"], traj="t0"
    )
    e1 = _entry(
        "e1", _onehot(1), action_val=1.0, factors=None, prev_ids=["e0"], traj="t0"
    )
    pkl = tmp_path / "lib.pkl"
    _write_pkl(pkl, [e0, e1], library_stats=_library_stats())
    _write_calib(tmp_path / "calib.jsonl")
    yml = _write_yaml(
        tmp_path / "c.yaml",
        pkl_path=pkl,
        judge=_composite_judge_block(tmp_path / "calib.jsonl"),
    )
    with pytest.raises(RuntimeError, match="not enriched"):
        ReplayHarness(yml)


def test_warmup_calibration_rejected(tmp_path):
    pkl = tmp_path / "lib.pkl"
    _enriched_chain_library(pkl)
    judge = _composite_judge_block(tmp_path / "calib.jsonl")
    judge["calibration"]["samples_source"] = {"type": "warmup"}
    yml = _write_yaml(tmp_path / "c.yaml", pkl_path=pkl, judge=judge)
    # match the harness's own message ("Use samples_source.type='offline'"), not the
    # generic downstream config error, to prove _reject_warmup_calibration fired.
    with pytest.raises(RuntimeError, match="offline"):
        ReplayHarness(yml)


def test_llm_layer_extract_keybuilder_rejected():
    # cp1_llm_layer_extract needs a real model -> harness must fail-fast (plan R7).
    from types import SimpleNamespace

    from exp.cache_latency_bench.replay import _reject_unsupported_keybuilder

    with pytest.raises(NotImplementedError, match="cp1_llm_layer_extract"):
        _reject_unsupported_keybuilder(
            SimpleNamespace(key_builder=SimpleNamespace(type="cp1_llm_layer_extract"))
        )
    # a supported keybuilder passes through untouched.
    _reject_unsupported_keybuilder(
        SimpleNamespace(key_builder=SimpleNamespace(type="placeholder"))
    )


# ----------------------------------------------------------------------
# summarize
# ----------------------------------------------------------------------


def test_summarize_produces_segment_stats(tmp_path):
    pkl = tmp_path / "lib.pkl"
    _simple_library(pkl)
    yml = _write_yaml(tmp_path / "c.yaml", pkl_path=pkl, judge={"type": "always_hit"})
    h5 = tmp_path / "episode_0000.h5"
    _write_mini_h5(
        h5, n_steps=2, task="taskA", episode_id=0, success=True, state_fn=_onehot
    )
    out_csv = tmp_path / "out.csv"
    ReplayHarness(yml).run(
        H5EpisodeSource(str(tmp_path)), repeats=1, out_csv=str(out_csv)
    )
    agg = summarize(str(out_csv))
    assert agg["n_steps"] == 2
    assert "cp1_search_ms" in agg["probes"]
    assert agg["probes"]["cp1_search_ms"]["ALL"]["count"] == 2
