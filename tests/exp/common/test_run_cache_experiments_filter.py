"""Unit tests for `--episode-filter` CLI passthrough in run_cache_experiments.

Covers:
  - When --episode-filter is given, _execute_tasks builds main.py command
    with `--episode-filter PATH`.
  - When omitted, main.py command does NOT include `--episode-filter`
    (legacy backward compat).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def captured_main_args():
    """Patch _execute_tasks's subprocess to capture command-line args without running."""
    captured: list[list[str]] = []

    def _fake_popen(cmd, *args, **kwargs):
        captured.append(list(cmd))
        proc = MagicMock()
        proc.wait.return_value = 0
        proc.poll.return_value = 0
        # Drain stdout pipe so the runner's own logic exits cleanly
        proc.stdout = MagicMock()
        proc.stdout.readline.side_effect = ["", ""]
        return proc

    return captured, _fake_popen


def _call_execute_tasks(tmp_path: Path, episode_filter: str | None):
    """Invoke _execute_tasks with a filter (or None) and return the captured cmd."""
    from exp.common import run_cache_experiments as rce

    log_path = tmp_path / "out.log"
    with patch.object(rce.subprocess, "Popen") as popen_mock:
        proc = MagicMock()
        proc.wait.return_value = 0
        proc.stdout = MagicMock()
        # Yield empty line then EOF so the readline loop terminates.
        proc.stdout.readline.side_effect = [""]
        proc.returncode = 0
        popen_mock.return_value = proc

        rce._execute_tasks(
            task_ids=[0],
            episodes_per_task=1,
            num_workers=1,
            host="localhost",
            port=8000,
            task_suite="libero_spatial",
            log_path=log_path,
            seed=7,
            episode_filter=episode_filter,
        )
        # First positional arg of Popen is the command list.
        return list(popen_mock.call_args.args[0])


def test_episode_filter_forwarded_to_main_py(tmp_path):
    cmd = _call_execute_tasks(tmp_path, episode_filter="/tmp/init_subset.json")
    assert "--episode-filter" in cmd
    idx = cmd.index("--episode-filter")
    assert cmd[idx + 1] == "/tmp/init_subset.json"


def test_episode_filter_omitted_when_none(tmp_path):
    cmd = _call_execute_tasks(tmp_path, episode_filter=None)
    assert "--episode-filter" not in cmd


def test_episode_filter_omitted_when_empty(tmp_path):
    cmd = _call_execute_tasks(tmp_path, episode_filter="")
    assert "--episode-filter" not in cmd
