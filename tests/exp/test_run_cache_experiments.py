from __future__ import annotations

import json

import pytest

from pathlib import Path

from exp.cache_experiment.run_cache_experiments import (
    RunState,
    _aggregate_episode_results,
    _compute_aggregate_success_rate,
    _episode_results_path_for,
    _infer_config_id_from_source,
    _parse_runs_filter,
    _parse_task_result,
    _remaining_tasks,
)


def test_parse_task_result_prefers_summary_line():
    stdout = "...\nTotal success rate: 60.0% (3/5)\nSuccess: True\nSuccess: True\n"
    assert _parse_task_result(stdout) == (3, 5)


def test_parse_task_result_falls_back_to_success_lines():
    stdout = "Success: True\nSuccess: False\nSuccess: TRUE\n"
    assert _parse_task_result(stdout) == (2, 3)


def test_parse_runs_filter_supports_ranges_and_singletons():
    assert _parse_runs_filter(None, 4) == {0, 1, 2, 3}
    assert _parse_runs_filter("1-3,5", 10) == {0, 1, 2, 4}


def test_remaining_tasks_returns_non_done_only():
    state = RunState(
        yaml_path="x.yaml",
        run_id="run_x",
        task_progress={"0": "done", "1": "failed", "2": "pending"},
    )
    assert _remaining_tasks(state) == [1, 2]


def test_compute_aggregate_success_rate_sums_completed_tasks_only():
    """§5e69373 changed the aggregation contract from "all task_results"
    to "only task_progress[tid] == 'done'". The in-flight / pending /
    failed task counts must NOT leak into the running success_rate —
    otherwise resume decisions key off a misleading mid-run value.

    Lock: task "2" is pending with a large fake [9, 99] counts; it must
    be filtered out by the ``status == "done"`` guard, so the result is
    3/5 from tasks "0" and "1" only, not 12/104.
    """
    state = RunState(
        yaml_path="x.yaml",
        run_id="run_x",
        task_progress={"0": "done", "1": "done", "2": "pending"},
        task_results={"0": [2, 3], "1": [1, 2], "2": [9, 99]},
    )
    assert _compute_aggregate_success_rate(state) == pytest.approx(3 / 5)


# ---------------------------------------------------------------------------
# §9.0 改动 2 + §19.B5: per-episode aggregation / dedup
# ---------------------------------------------------------------------------


def test_episode_results_path_uses_plain_name_for_single_batch(tmp_path: Path) -> None:
    """Single-batch runs must keep the plain ``{run_id}.episode_results.json``
    name so the aggregator's config_id inference lands on the bare YAML stem."""
    log_path = tmp_path / "e1_clip_w7_d4.log"
    out = _episode_results_path_for(log_path, batch_idx=0, num_batches=1)
    assert out == tmp_path / "e1_clip_w7_d4.episode_results.json"


def test_episode_results_path_tags_multi_batch_runs(tmp_path: Path) -> None:
    """Multi-batch runs must embed ``batchKofN`` so one ``main.py`` call
    does not overwrite another's output — otherwise `_aggregate_episode_results`
    would silently lose half the episodes on 2+ batch runs."""
    log_path = tmp_path / "e1_clip_w7_d4.log"
    out0 = _episode_results_path_for(log_path, batch_idx=0, num_batches=2)
    out1 = _episode_results_path_for(log_path, batch_idx=1, num_batches=2)
    assert out0 != out1
    assert out0.name == "e1_clip_w7_d4.batch0of2.episode_results.json"
    assert out1.name == "e1_clip_w7_d4.batch1of2.episode_results.json"


def test_episode_results_path_preserves_retry_marker(tmp_path: Path) -> None:
    """Retry log paths carry ``.retryN`` before ``.log``; that marker must
    survive onto the episode_results filename so the aggregator can infer
    both the config_id and the attempt correctly."""
    log_path = tmp_path / "retry" / "e1_clip_w7_d4.retry1.log"
    out = _episode_results_path_for(log_path, batch_idx=0, num_batches=1)
    assert out.name == "e1_clip_w7_d4.retry1.episode_results.json"


def test_infer_config_id_strips_batch_and_retry_markers() -> None:
    # Plain first-pass path.
    assert _infer_config_id_from_source("e1_clip_w7_d4.episode_results.json") == "e1_clip_w7_d4"
    # Multi-batch first-pass.
    assert (
        _infer_config_id_from_source("e1_clip_w7_d4.batch0of2.episode_results.json")
        == "e1_clip_w7_d4"
    )
    # Retry pass (lives under retry/ subdir — prefix stripped by Path.name).
    assert (
        _infer_config_id_from_source("retry/e1_clip_w7_d4.retry1.episode_results.json")
        == "e1_clip_w7_d4"
    )
    # Multi-batch retry.
    assert (
        _infer_config_id_from_source("retry/e1_clip_w7_d4.batch1of2.retry2.episode_results.json")
        == "e1_clip_w7_d4"
    )


def test_aggregate_dedups_retry_success_over_first_pass_failure(tmp_path: Path) -> None:
    """Critical §19.B5 invariant: when an episode fails on the first pass
    but a retry run rescues it, the aggregated ``cache_eval_results.json``
    must carry ``success=True``. The downstream ``dump_step1a_failed_inits.py``
    depends on this — without dedup it would treat retry-recovered episodes
    as failures and dump them into the Step 1b subset, wasting GT capture
    time on already-successful inits."""
    (tmp_path / "e1_clip_w7_d4.episode_results.json").write_text(
        json.dumps([
            {"task_id": 3, "init_state_idx": 2, "orig_init_state_idx": 2,
             "episode_id": 152, "seed": 7, "success": False},
            {"task_id": 3, "init_state_idx": 5, "orig_init_state_idx": 5,
             "episode_id": 155, "seed": 7, "success": True},
        ])
    )
    (tmp_path / "retry").mkdir()
    (tmp_path / "retry" / "e1_clip_w7_d4.retry1.episode_results.json").write_text(
        json.dumps([
            {"task_id": 3, "init_state_idx": 2, "orig_init_state_idx": 2,
             "episode_id": 152, "seed": 7, "success": True},
        ])
    )

    out = tmp_path / "cache_eval_results.json"
    _aggregate_episode_results(tmp_path, out)
    rows = json.loads(out.read_text())

    by_key = {(r["task_id"], r["init_state_idx"]): r for r in rows}
    assert by_key[(3, 2)]["success"] is True, "retry must override failed first-pass"
    assert by_key[(3, 2)]["attempt"] == 1
    assert by_key[(3, 2)]["config_id"] == "e1_clip_w7_d4"
    assert by_key[(3, 5)]["success"] is True
    assert by_key[(3, 5)]["attempt"] == 0
    assert len(rows) == 2, "dedup must collapse the two (3, 2) records"


def test_aggregate_is_noop_on_missing_run_root(tmp_path: Path) -> None:
    """Aggregator must not crash when called before any main.py ever ran —
    important for --resume runs that produced no new per-episode files."""
    ghost = tmp_path / "does_not_exist"
    _aggregate_episode_results(ghost, tmp_path / "out.json")
    assert not (tmp_path / "out.json").exists()


def test_load_or_init_fresh_builds_states_from_yaml(tmp_path):
    """cleanup/04 behavior lock: the non-resume path simply builds one
    ``RunState`` per YAML in ``yaml_files`` order and does not read the
    state file at all."""
    from types import SimpleNamespace
    from exp.cache_experiment.run_cache_experiments import _load_or_init_states

    yaml_dir = tmp_path / "yamls"
    yaml_dir.mkdir()
    (yaml_dir / "a.yaml").write_text("x")
    (yaml_dir / "b.yaml").write_text("x")
    args = SimpleNamespace(
        resume=False, episodes_per_run=10, task_suite="libero_spatial",
    )

    states = _load_or_init_states(
        args, sorted(yaml_dir.glob("*.yaml")), tmp_path / "state.json"
    )

    assert states is not None
    assert [s.run_id for s in states] == ["a", "b"]
    assert all(s.task_suite == "libero_spatial" for s in states)
    assert all(s.episodes_per_task == 10 for s in states)


def test_load_or_init_resume_rejects_task_suite_mismatch(tmp_path):
    """cleanup/04 behavior lock: resume with a saved state whose
    ``task_suite`` disagrees with the current CLI returns ``None``
    (signalling ``main`` to bail)."""
    from types import SimpleNamespace
    from exp.cache_experiment.run_cache_experiments import _load_or_init_states

    yaml_dir = tmp_path / "yamls"
    yaml_dir.mkdir()
    (yaml_dir / "run_001.yaml").write_text("x")
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps([{
        "yaml_path": str(yaml_dir / "run_001.yaml"),
        "run_id": "run_001",
        "status": "pending",
        "episodes_per_task": 10,
        "task_suite": "libero_object",
    }]))

    args = SimpleNamespace(
        resume=True, episodes_per_run=10, task_suite="libero_spatial",
    )

    states = _load_or_init_states(
        args, sorted(yaml_dir.glob("*.yaml")), state_path
    )

    assert states is None


def test_resume_with_task_ids_subset_does_not_expand_task_progress(tmp_path):
    """cleanup/04 F3 lock: when ``--task-ids`` on resume specifies a
    subset that differs from the saved ``task_progress`` keys, the runner
    does **not** retroactively add/remove tasks from existing progress.
    This documents the pre-cleanup behavior (``_init_task_progress`` only
    fires when ``task_progress`` is empty) so future F3 work can change
    it deliberately, not by accident."""
    from exp.cache_experiment.run_cache_experiments import _init_task_progress, RunState

    state = RunState(
        yaml_path="x.yaml", run_id="run_x",
        # Saved progress: original run used 4 tasks.
        task_progress={"0": "done", "1": "done", "2": "pending", "3": "pending"},
        task_results={"0": [5, 5], "1": [4, 5]},
    )

    # CLI reruns with --task-ids=0,2 (subset). _init_task_progress must
    # be a no-op — existing keys survive unchanged.
    _init_task_progress(state, [0, 2])

    assert state.task_progress == {
        "0": "done", "1": "done", "2": "pending", "3": "pending",
    }


def test_run_stats_agg_postfix_str():
    from exp.cache_experiment.run_cache_experiments import _RunStats

    s = _RunStats(total_episodes_all=0, total_successes_all=0, completed=0, failed=0)
    assert "agg=N/A" in s.agg_postfix_str()
    assert "done=0" in s.agg_postfix_str()

    s = _RunStats(total_episodes_all=10, total_successes_all=3, completed=1, failed=2)
    out = s.agg_postfix_str()
    assert "agg=30.0%" in out
    assert "(3/10)" in out
    assert "done=1" in out
    assert "fail=2" in out


def test_resume_validation_requires_task_suite_for_incomplete_state(tmp_path):
    state_path = tmp_path / "state.json"
    yaml_dir = tmp_path / "yamls"
    yaml_dir.mkdir()
    (yaml_dir / "run_001.yaml").write_text("enabled: true\n")
    state_path.write_text(json.dumps([
        {
            "yaml_path": str(yaml_dir / "run_001.yaml"),
            "run_id": "run_001",
            "status": "failed",
            "episodes_per_task": 10,
            "task_suite": "",
        }
    ]))

    import sys
    import exp.cache_experiment.run_cache_experiments as runner

    argv = sys.argv[:]
    try:
        sys.argv = [
            "run_cache_experiments.py",
            "--yaml-dir", str(yaml_dir),
            "--state-path", str(state_path),
            "--task-suite", "libero_spatial",
            "--episodes-per-run", "10",
            "--resume",
        ]
        with pytest.raises(SystemExit) as exc:
            raise SystemExit(runner.main())
        assert exc.value.code in (None, 0)
    finally:
        sys.argv = argv
