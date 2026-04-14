"""Unit tests for ``exp/trajectory_deviation/run_step1b_gt.py`` (plan §9.2 + §18.B4 + §19.B6).

We cannot drive a real ``main.py`` subprocess in CI (no libero_sim / GPU /
server), so these tests exercise:

- Pure helpers (``build_unit_key`` / ``parse_unit_key`` / ``load_filter_entries``
  / ``subset_size_per_task`` / ``_read_unit_result``) which require no side
  effects.
- ``_build_subprocess_cmd`` environment sanitising (conda_env branch mirrors
  ``run_cache_experiments.py``).
- ``Step1bRunner.execute_unit`` against a stubbed ``subprocess.run`` — this
  locks the main_args shape, the per-unit filter/results path layout, and
  the result dict returned to ``BaseRunState``.

No test invokes ``send_load_cache_config`` — the config switch is asserted
indirectly by the fact that ``Step1bRunner`` takes no YAML argument (it
is the ``main()`` driver's responsibility, not the runner's).
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from exp.trajectory_deviation.run_step1b_gt import (
    DEFAULT_INFERENCE_YAML,
    FilterEntry,
    Step1bRunner,
    _build_subprocess_cmd,
    _read_unit_result,
    build_unit_key,
    load_filter_entries,
    parse_unit_key,
    subset_size_per_task,
)


# ---------------------------------------------------------------------------
# Unit key round-trip
# ---------------------------------------------------------------------------


def test_unit_key_roundtrip() -> None:
    """Key format is load-bearing: any state-file schema drift would break
    `--resume` across runs."""
    assert build_unit_key(3, 15) == "3:15"
    assert parse_unit_key("3:15") == (3, 15)
    for task_id, orig in [(0, 0), (9, 49), (7, 28)]:
        assert parse_unit_key(build_unit_key(task_id, orig)) == (task_id, orig)


# ---------------------------------------------------------------------------
# Filter loading + subset sizing
# ---------------------------------------------------------------------------


def test_load_filter_entries_and_subset_size(tmp_path: Path) -> None:
    filter_path = tmp_path / "step1b_filter.json"
    filter_path.write_text(json.dumps([
        {"task_id": 3, "orig_init_state_idx": 15, "subset_init_state_idx": 0},
        {"task_id": 3, "orig_init_state_idx": 28, "subset_init_state_idx": 1},
        {"task_id": 5, "orig_init_state_idx": 2,  "subset_init_state_idx": 0},
    ]))
    entries = load_filter_entries(filter_path)
    assert entries[0] == FilterEntry(task_id=3, orig_init_state_idx=15, subset_init_state_idx=0)
    assert len(entries) == 3

    # subset_size_per_task: must yield max subset position + 1 per task so
    # main.py's --num-trials-per-task reaches every subset_idx in the filter.
    sizes = subset_size_per_task(entries)
    assert sizes == {3: 2, 5: 1}


# ---------------------------------------------------------------------------
# Default YAML lock (plan §18.B4.3)
# ---------------------------------------------------------------------------


def test_default_inference_yaml_points_to_clip() -> None:
    """§18.B4.3 explicitly fixes the Step 1b config to the clip bundle.
    If this path is ever changed to a different bundle, the three Step 2
    configs would no longer share the same GT — breaking the §18.B4 shared
    GT invariant. This test locks the default value."""
    assert DEFAULT_INFERENCE_YAML.endswith("inference_clip_w7_d4.yaml")


# ---------------------------------------------------------------------------
# _build_subprocess_cmd: conda_env branch
# ---------------------------------------------------------------------------


def test_build_subprocess_cmd_uv_branch_leaves_env_default() -> None:
    cmd, env = _build_subprocess_cmd(["examples/libero/main.py", "--host", "h"], conda_env=None)
    assert cmd[:2] == ["uv", "run"]
    assert env is None


def test_build_subprocess_cmd_conda_branch_sanitises_virtualenv(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without this sanitisation the conda subprocess inherits uv's
    ``VIRTUAL_ENV`` and loads the wrong Python — the same regression
    ``run_cache_experiments.py`` guards against. Lock it here too so a
    future refactor doesn't accidentally diverge the two runners."""
    monkeypatch.setenv("VIRTUAL_ENV", "/tmp/fake_venv")
    monkeypatch.setenv("PYTHONPATH", "/tmp/fake_pythonpath")
    monkeypatch.setenv("PATH", "/tmp/fake_venv/bin:/usr/bin")

    cmd, env = _build_subprocess_cmd(["examples/libero/main.py"], conda_env="libero_sim")
    assert cmd[:6] == ["conda", "run", "--no-capture-output", "-n", "libero_sim", "python"]
    assert env is not None
    assert "VIRTUAL_ENV" not in env
    assert "PYTHONPATH" not in env
    assert "/tmp/fake_venv/bin" not in env["PATH"].split(":")
    # MUJOCO_GL=egl is required for headless rendering on the cluster.
    assert env["MUJOCO_GL"] == "egl"


# ---------------------------------------------------------------------------
# _read_unit_result: the three result shapes (success / failure / no file)
# ---------------------------------------------------------------------------


def test_read_unit_result_success(tmp_path: Path) -> None:
    results_path = tmp_path / "unit_3_15.episode_results.json"
    results_path.write_text(json.dumps([
        {"task_id": 3, "init_state_idx": 0, "orig_init_state_idx": 15,
         "episode_id": 150, "seed": 7, "success": True},
    ]))
    r = _read_unit_result(
        results_path=results_path,
        out_dir=tmp_path / "gt", task_id=3, subset_idx=0, orig_idx=15,
    )
    assert r["success"] is True
    assert r["inference_failed"] is False
    assert r["hdf5_path"] == "task_3/episode_0.h5"
    assert r["orig_init_state_idx"] == 15
    assert r["seed"] == 7


def test_read_unit_result_episode_failure_is_not_inference_failure(tmp_path: Path) -> None:
    """An episode that ran but didn't reach the goal must report
    ``success=False`` with ``inference_failed=False`` — Step 2 still needs
    the GT trajectory even when the inference failed to solve the task
    (that's the whole point of "correct trajectory under AlwaysSkip")."""
    results_path = tmp_path / "unit_3_15.episode_results.json"
    results_path.write_text(json.dumps([
        {"task_id": 3, "init_state_idx": 0, "orig_init_state_idx": 15,
         "episode_id": 150, "seed": 7, "success": False},
    ]))
    r = _read_unit_result(
        results_path=results_path,
        out_dir=tmp_path / "gt", task_id=3, subset_idx=0, orig_idx=15,
    )
    assert r["success"] is False
    assert r["inference_failed"] is False
    assert r["hdf5_path"] == "task_3/episode_0.h5"


def test_read_unit_result_missing_file_flags_inference_failed(tmp_path: Path) -> None:
    r = _read_unit_result(
        results_path=tmp_path / "nope.json",
        out_dir=tmp_path / "gt", task_id=3, subset_idx=0, orig_idx=15,
    )
    assert r == {
        "success": False,
        "inference_failed": True,
        "reason": "no episode_results.json emitted",
    }


def test_read_unit_result_row_for_wrong_subset_flags_inference_failed(tmp_path: Path) -> None:
    """main.py writes one row per executed episode. If the filter/subset
    position decode is broken we would see rows for the wrong subset_idx —
    that's worse than no row at all, so surface it explicitly."""
    results_path = tmp_path / "unit_3_15.episode_results.json"
    results_path.write_text(json.dumps([
        {"task_id": 3, "init_state_idx": 99, "orig_init_state_idx": 15,
         "episode_id": 150, "seed": 7, "success": True},
    ]))
    r = _read_unit_result(
        results_path=results_path,
        out_dir=tmp_path / "gt", task_id=3, subset_idx=0, orig_idx=15,
    )
    assert r["success"] is False
    assert r["inference_failed"] is True
    assert "no row for subset_idx=0" in r["reason"]


# ---------------------------------------------------------------------------
# Step1bRunner.execute_unit: main_args shape lock
# ---------------------------------------------------------------------------


def _make_runner(tmp_path: Path, *, entries) -> Step1bRunner:
    return Step1bRunner(
        state_path=tmp_path / "step1b_state.json",
        inits_dir=tmp_path / "inits",
        filter_entries=entries,
        out_dir=tmp_path / "gt",
        per_unit_results_dir=tmp_path / "per_unit_results",
        per_unit_filter_dir=tmp_path / "per_unit_filters",
        task_suite_name="libero_spatial",
        host="127.0.0.1",
        port=9000,
        seed=7,
        conda_env=None,
        max_retries=0,
    )


def test_build_units_covers_every_filter_entry(tmp_path: Path) -> None:
    entries = [
        FilterEntry(task_id=3, orig_init_state_idx=15, subset_init_state_idx=0),
        FilterEntry(task_id=3, orig_init_state_idx=28, subset_init_state_idx=1),
        FilterEntry(task_id=5, orig_init_state_idx=2,  subset_init_state_idx=0),
    ]
    runner = _make_runner(tmp_path, entries=entries)
    keys = [u.unit_key for u in runner.build_units()]
    assert set(keys) == {"3:15", "3:28", "5:2"}


def test_execute_unit_dispatches_expected_main_args(tmp_path: Path) -> None:
    """End-to-end main_args contract: one subprocess per unit, carrying the
    single-entry filter, the right --num-trials-per-task for the subset,
    --save-trajectory, and --save-episode-results. This is the exact
    command line the server-side Step 1b GT collection relies on; any drift
    here will silently produce wrong HDF5 layouts."""
    entries = [
        FilterEntry(task_id=3, orig_init_state_idx=15, subset_init_state_idx=0),
        FilterEntry(task_id=3, orig_init_state_idx=28, subset_init_state_idx=1),
    ]
    runner = _make_runner(tmp_path, entries=entries)

    captured: dict = {}

    def fake_run(cmd, env=None, capture_output=False, text=False, timeout=None):
        captured["cmd"] = cmd
        captured["env"] = env
        # Simulate main.py writing the per-episode results file.
        # We need to find --episode-results-path in cmd.
        rp_idx = cmd.index("--episode-results-path") + 1
        Path(cmd[rp_idx]).parent.mkdir(parents=True, exist_ok=True)
        Path(cmd[rp_idx]).write_text(json.dumps([{
            "task_id": 3, "init_state_idx": 1, "orig_init_state_idx": 28,
            "episode_id": 151, "seed": 7, "success": True,
        }]))
        mock = MagicMock()
        mock.returncode = 0
        mock.stdout = "ok"
        return mock

    with patch.object(subprocess, "run", side_effect=fake_run):
        result = runner.execute_unit(runner.build_units()[1])  # unit 3:28

    cmd = captured["cmd"]
    assert cmd[:2] == ["uv", "run"]
    assert "examples/libero/main.py" in cmd
    # Task + subset contract: --num-trials-per-task must cover max subset (=2).
    assert cmd[cmd.index("--task-ids") + 1] == "3"
    assert cmd[cmd.index("--num-trials-per-task") + 1] == "2"
    # Trajectory + results flags present.
    assert "--save-trajectory" in cmd
    assert "--save-trajectory-dir" in cmd
    assert "--save-episode-results" in cmd
    assert "--init-states-dir" in cmd
    # Filter file exists and contains exactly this unit.
    filter_idx = cmd.index("--episode-filter") + 1
    filter_data = json.loads(Path(cmd[filter_idx]).read_text())
    assert filter_data == [{
        "task_id": 3, "orig_init_state_idx": 28, "subset_init_state_idx": 1,
    }]

    # Result dict contract into BaseRunState.
    assert result == {
        "success": True, "inference_failed": False,
        "hdf5_path": "task_3/episode_1.h5",
        "orig_init_state_idx": 28, "seed": 7,
    }


def test_execute_unit_raises_on_nonzero_exit(tmp_path: Path) -> None:
    """BaseRunState converts raised exceptions into ``status='failed'``.
    We rely on that path so a crashed main.py is retry-queued cleanly —
    lock the raise here."""
    entries = [FilterEntry(task_id=3, orig_init_state_idx=15, subset_init_state_idx=0)]
    runner = _make_runner(tmp_path, entries=entries)

    def fake_run(cmd, env=None, capture_output=False, text=False, timeout=None):
        mock = MagicMock()
        mock.returncode = 137
        mock.stdout = "boom\n" * 3
        return mock

    with patch.object(subprocess, "run", side_effect=fake_run):
        with pytest.raises(RuntimeError, match="main.py exit 137"):
            runner.execute_unit(runner.build_units()[0])
