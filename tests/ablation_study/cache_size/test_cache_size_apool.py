"""A-pool freeze + anti-pollution assertions (plan §7)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from exp.ablation_study.cache_size.verify_apool import (
    assert_disjoint,
    digest_init_file,
    states_to_rows,
)


def _write_pool(d, rows_per_task, n_tasks=2, offset=0.0):
    d.mkdir(parents=True, exist_ok=True)
    for t in range(n_tasks):
        states = np.arange(rows_per_task * 4, dtype=np.float64).reshape(rows_per_task, 4) + offset
        torch.save(states, d / f"task_{t}.init")


def test_disjoint_pools_pass(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    _write_pool(a, 5, offset=0.0)
    _write_pool(b, 5, offset=1000.0)

    report = assert_disjoint(a, b)
    assert all(v["shared"] == 0 for v in report.values())
    assert report["task_0"]["a_count"] == 5


def test_any_shared_init_is_fatal(tmp_path):
    """One overlapping row must stop the run -- this is the leakage red line."""
    a, b = tmp_path / "a", tmp_path / "b"
    _write_pool(a, 5, offset=0.0)
    _write_pool(b, 5, offset=1000.0)

    # Contaminate: copy one A row into B.
    a_rows = torch.load(a / "task_0.init", weights_only=False)
    b_rows = torch.load(b / "task_0.init", weights_only=False)
    b_rows[2] = a_rows[1]
    torch.save(b_rows, b / "task_0.init")

    with pytest.raises(SystemExit, match="appear in BOTH"):
        assert_disjoint(a, b)


def test_missing_counterpart_is_loud(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    _write_pool(a, 5, n_tasks=2, offset=0.0)
    _write_pool(b, 5, n_tasks=1, offset=1000.0)  # disjoint, but missing task_1
    with pytest.raises(FileNotFoundError, match="no counterpart"):
        assert_disjoint(a, b)


def test_row_canonicalization_is_order_insensitive():
    """Set arithmetic must not depend on how each pool happens to be ordered."""
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    y = x[::-1].copy()
    assert states_to_rows(x) == states_to_rows(y)


def test_digest_is_stable_and_content_sensitive(tmp_path):
    d = tmp_path / "p"
    _write_pool(d, 3, n_tasks=1)
    f = d / "task_0.init"
    first = digest_init_file(f)
    assert first == digest_init_file(f)

    rows = torch.load(f, weights_only=False)
    rows[0][0] += 1.0
    torch.save(rows, f)
    assert digest_init_file(f) != first
