"""Unit tests for ``exp/trajectory_deviation/dump_step1a_failed_inits.py`` (plan §9.1 + §19.B6).

We can't import the LIBERO benchmark here (no ``libero_sim`` in the plain
venv) so every test either exercises the pure helpers (``_group_failed_by_task``,
``_build_filter_entries``, ``_write_artifacts``) or drives the ``_run`` entry
point in ``--no-torch`` mode, which stubs the benchmark and skips the
``torch.save`` write. The end-to-end CLI with real tensors is covered by the
Step 1a smoke run in the implementation log (§21 T14).
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "exp" / "trajectory_deviation"
# Load the module by file path — loading by path avoids requiring the
# full ``exp.trajectory_deviation`` package (which pulls LIBERO in ``__init__``).
_SPEC = importlib.util.spec_from_file_location(
    "dump_step1a_failed_inits",
    _SCRIPTS_DIR / "dump_step1a_failed_inits.py",
)
assert _SPEC and _SPEC.loader
_MOD = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MOD
_SPEC.loader.exec_module(_MOD)


# ---------------------------------------------------------------------------
# _group_failed_by_task
# ---------------------------------------------------------------------------


def test_group_failed_by_task_filters_success_and_sorts() -> None:
    rows = [
        {"task_id": 3, "init_state_idx": 28, "success": False},
        {"task_id": 3, "init_state_idx": 15, "success": False},
        {"task_id": 3, "init_state_idx": 7,  "success": True},   # dropped
        {"task_id": 5, "init_state_idx": 2,  "success": False},
        {"task_id": 1, "init_state_idx": 0,  "success": True},   # dropped
    ]
    out = _MOD._group_failed_by_task(rows)
    assert out == {3: [15, 28], 5: [2]}


def test_group_failed_by_task_prefers_orig_when_present() -> None:
    """Step 1a normally has no filter so init_state_idx == orig_init_state_idx.
    But if the aggregator is ever fed a filtered run, the per-task subset
    file must reference the original benchmark indices, not the subset
    positions — so ``orig_init_state_idx`` takes precedence (§19.B6)."""
    rows = [
        {"task_id": 3, "init_state_idx": 1, "orig_init_state_idx": 28, "success": False},
        {"task_id": 3, "init_state_idx": 0, "orig_init_state_idx": 15, "success": False},
    ]
    out = _MOD._group_failed_by_task(rows)
    assert out == {3: [15, 28]}


def test_group_failed_by_task_dedups_repeats() -> None:
    """A single (task, init) must only appear once in the subset file."""
    rows = [
        {"task_id": 3, "init_state_idx": 15, "success": False},
        {"task_id": 3, "init_state_idx": 15, "success": False},   # dup
    ]
    assert _MOD._group_failed_by_task(rows) == {3: [15]}


# ---------------------------------------------------------------------------
# _build_filter_entries: plan §19.B6 schema lock
# ---------------------------------------------------------------------------


def test_build_filter_entries_matches_19_b6_schema() -> None:
    failed = {3: [15, 28], 5: [2]}
    entries = _MOD._build_filter_entries(failed)

    # Exact match against the schema fixed in §19.B6 (test_libero_main.py
    # `test_load_episode_filter_parses_step1b_schema` locks the consumer
    # side — this locks the producer side).
    assert entries == [
        {"task_id": 3, "orig_init_state_idx": 15, "subset_init_state_idx": 0},
        {"task_id": 3, "orig_init_state_idx": 28, "subset_init_state_idx": 1},
        {"task_id": 5, "orig_init_state_idx": 2,  "subset_init_state_idx": 0},
    ]


# ---------------------------------------------------------------------------
# _write_artifacts (torch skipped)
# ---------------------------------------------------------------------------


def test_write_artifacts_emits_init_map_and_filter_without_torch(tmp_path: Path) -> None:
    failed = {3: [15, 28], 5: [2]}
    task_names = {3: "pick_apple", 5: "open_drawer"}
    written, filter_path = _MOD._write_artifacts(
        tmp_path, failed, task_names, init_subsets=None
    )

    # No .init writes when init_subsets is None (no torch).
    assert written == []

    assert (tmp_path / "pick_apple.init_map.json").read_text() == "[15, 28]"
    assert (tmp_path / "open_drawer.init_map.json").read_text() == "[2]"

    filter_entries = json.loads(filter_path.read_text())
    assert filter_entries[0] == {
        "task_id": 3, "orig_init_state_idx": 15, "subset_init_state_idx": 0,
    }
    assert len(filter_entries) == 3


# ---------------------------------------------------------------------------
# _run end-to-end in --no-torch mode
# ---------------------------------------------------------------------------


def test_run_skip_torch_produces_all_artifacts_from_json(tmp_path: Path) -> None:
    step1a = tmp_path / "cache_eval_results.json"
    step1a.write_text(json.dumps([
        {"task_id": 0, "init_state_idx": 4,  "seed": 7, "success": True},   # pass
        {"task_id": 3, "init_state_idx": 15, "seed": 7, "success": False},
        {"task_id": 3, "init_state_idx": 28, "seed": 7, "success": False},
        {"task_id": 5, "init_state_idx": 2,  "seed": 7, "success": False},
    ]))

    out_dir = tmp_path / "out"
    _MOD._run(
        step1a_results=step1a,
        task_suite_name="libero_spatial",
        out_dir=out_dir,
        skip_torch=True,
    )

    # Per-task init_map files.
    assert (out_dir / "task_3.init_map.json").read_text() == "[15, 28]"
    assert (out_dir / "task_5.init_map.json").read_text() == "[2]"
    # No .init tensors in no-torch mode.
    assert not any(p.suffix == ".init" for p in out_dir.iterdir())

    filter_entries = json.loads((out_dir / "step1b_filter.json").read_text())
    assert len(filter_entries) == 3
    # Subset positions must restart at 0 per task_id.
    assert {e["subset_init_state_idx"] for e in filter_entries if e["task_id"] == 3} == {0, 1}
    assert {e["subset_init_state_idx"] for e in filter_entries if e["task_id"] == 5} == {0}


def test_run_with_no_failures_noop(tmp_path: Path, capsys) -> None:
    """All-success Step 1a → nothing to do. Must not write a malformed
    ``step1b_filter.json`` (empty list is still technically fine but we
    explicitly skip writing so downstream runners can early-exit on
    file-not-found)."""
    step1a = tmp_path / "cache_eval_results.json"
    step1a.write_text(json.dumps([
        {"task_id": 0, "init_state_idx": 0, "success": True},
    ]))
    out_dir = tmp_path / "out"
    _MOD._run(
        step1a_results=step1a,
        task_suite_name="libero_spatial",
        out_dir=out_dir,
        skip_torch=True,
    )
    captured = capsys.readouterr()
    assert "No failed episodes" in captured.out
    assert not (out_dir / "step1b_filter.json").exists()
