from __future__ import annotations

import json

import pytest

from exp.run_cache_experiments import (
    RunState,
    _compute_aggregate_success_rate,
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


def test_compute_aggregate_success_rate_sums_all_tasks():
    state = RunState(
        yaml_path="x.yaml",
        run_id="run_x",
        task_results={"0": [2, 3], "1": [1, 2]},
    )
    assert _compute_aggregate_success_rate(state) == pytest.approx(3 / 5)


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
    import exp.run_cache_experiments as runner

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
