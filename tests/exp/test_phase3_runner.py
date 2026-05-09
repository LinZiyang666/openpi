"""Tests for exp/verdict_factor_judge/run_phase3.py runner helpers.

Covers G2 R1 B2: solver fail-fast _NA propagation must reach
``per_yaml_summary.jsonl`` so downstream aggregation does not silently
lose 16 cells per failed recipe.
"""

from __future__ import annotations

import json
from pathlib import Path

from exp.verdict_factor_judge.phase3_spec import GRID
from exp.verdict_factor_judge.run_phase3 import (
    _recipe_end_cleanup,
    _write_na_summary_rows,
)


# ----------------------------------------------------------------------
# G2 R1 B2: fail-fast _NA propagates 16 rows per recipe to summary
# ----------------------------------------------------------------------


def test_write_na_summary_rows_writes_one_row_per_cell(tmp_path: Path) -> None:
    summary_path = tmp_path / "summary.jsonl"
    summary_path.write_text("")    # like main() does on non-resume start

    rows = _write_na_summary_rows(
        summary_path,
        recipe_id="g2_f1b_t_w_long_risk_d_jerk",
        warmup_yaml_id="spatial16_w8_d4_phase3_g2_f1b_t_w_long_risk_d_jerk__warmup",
        warmup_eval_yaml_id="spatial16_w8_d4_phase3_g2_f1b_t_w_long_risk_d_jerk",
        error="Calibration key 'jerk_offline_state__p7_f7': only 12 non-NaN samples, "
              "need >= window_size=50",
        grid=list(GRID),
    )

    # 16 rows in memory + 16 lines on disk
    assert len(rows) == 16
    on_disk = [json.loads(line) for line in summary_path.read_text().splitlines() if line]
    assert len(on_disk) == 16

    # All rows carry recipe-level provenance + the error string
    for row in on_disk:
        assert row["recipe_id"] == "g2_f1b_t_w_long_risk_d_jerk"
        assert "error" in row
        assert "non-NaN samples" in row["error"]
        assert row["fh_thr"] is None
        assert row["ws_thr"] is None
        assert row["success_rate"] is None
        assert row["n_eval_verdicts"] == 0
        # Verdict counts present (downstream aggregator can sum them safely)
        assert {"n_full_hit", "n_warm_start", "n_miss"} <= set(row)

    # Cells span the full 4x4 grid
    seen = {(row["fh_ratio"], row["ws_ratio"]) for row in on_disk}
    assert seen == set(GRID)


def test_write_na_summary_rows_appends_idempotent(tmp_path: Path) -> None:
    """Re-running the helper appends; the runner's main() handles dedup
    via _load_done_yaml_ids on resume, not via overwrite here.
    """
    summary_path = tmp_path / "summary.jsonl"
    _write_na_summary_rows(
        summary_path,
        recipe_id="g4_f1b_t_w_short_d_jerk",
        warmup_yaml_id="spatial16_w8_d4_phase3_g4_f1b_t_w_short_d_jerk__warmup",
        warmup_eval_yaml_id="spatial16_w8_d4_phase3_g4_f1b_t_w_short_d_jerk",
        error="degenerate",
        grid=[(0.2, 0.2), (0.3, 0.3)],
    )
    _write_na_summary_rows(
        summary_path,
        recipe_id="g4_f1b_t_w_short_d_jerk",
        warmup_yaml_id="spatial16_w8_d4_phase3_g4_f1b_t_w_short_d_jerk__warmup",
        warmup_eval_yaml_id="spatial16_w8_d4_phase3_g4_f1b_t_w_short_d_jerk",
        error="degenerate",
        grid=[(0.4, 0.4)],
    )
    on_disk = [
        json.loads(line) for line in summary_path.read_text().splitlines() if line
    ]
    assert len(on_disk) == 3


def test_write_na_summary_rows_yaml_id_marks_NA(tmp_path: Path) -> None:
    summary_path = tmp_path / "summary.jsonl"
    rows = _write_na_summary_rows(
        summary_path,
        recipe_id="g4_f1b_t_w_short_d_jerk",
        warmup_yaml_id="spatial16_w8_d4_phase3_g4_f1b_t_w_short_d_jerk__warmup",
        warmup_eval_yaml_id="spatial16_w8_d4_phase3_g4_f1b_t_w_short_d_jerk",
        error="x",
        grid=[(0.2, 0.3)],
    )
    assert rows[0]["yaml_id"] == "spatial16_w8_d4_phase3_g4_f1b_t_w_short_d_jerk__fh0.2_ws0.3__NA"


def _make_warmup_yaml(tmp_path: Path) -> Path:
    p = tmp_path / "warmup.yaml"
    p.write_text("enabled: true\n")    # content irrelevant — read as text only
    return p


class _RecordingCtl:
    """Mock WebsocketClientPolicy that records (method, kwargs) call order."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def load_cache_config(self, *, yaml_content: str, yaml_id: str) -> dict:
        self.calls.append(("load_cache_config", {"yaml_id": yaml_id}))
        return {"__ack__": "load_cache_config"}

    def unload_warmup_buffer(self, eval_yaml_id: str) -> dict:
        self.calls.append(("unload_warmup_buffer", {"eval_yaml_id": eval_yaml_id}))
        return {"__ack__": "unload_warmup_buffer"}


# ----------------------------------------------------------------------
# G2 R1 (post-deployment): bundle race fix — recipe-end cleanup ordering
# ----------------------------------------------------------------------


def test_recipe_end_cleanup_resets_bundle_first_then_unloads(tmp_path: Path) -> None:
    """ORDER MATTERS: load_cache_config(warmup_yaml) MUST run before any
    unload_warmup_buffer call. Otherwise the server's _current_bundle
    stays at the previous cell's eval yaml (samples_source=warmup), and
    the next recipe's first connect-time build_per_connection_components
    fails with "WarmupPool has no entry for yaml_id=...".
    """
    warmup_yaml_path = _make_warmup_yaml(tmp_path)
    ctl = _RecordingCtl()
    grid = [(0.2, 0.2), (0.3, 0.3), (0.5, 0.5)]
    _recipe_end_cleanup(
        ctl,
        recipe_id="g4_test",
        warmup_yaml_path=warmup_yaml_path,
        warmup_yaml_id="spatial16_phase3_g4_test__warmup",
        warmup_eval_yaml_id="spatial16_phase3_g4_test",
        grid=grid,
    )
    # First call MUST be the bundle-reset load_cache_config.
    assert ctl.calls[0] == (
        "load_cache_config",
        {"yaml_id": "spatial16_phase3_g4_test__warmup"},
    )
    # Then 3 cell unloads.
    assert ctl.calls[1] == (
        "unload_warmup_buffer",
        {"eval_yaml_id": "spatial16_phase3_g4_test__fh0.2_ws0.2"},
    )
    assert ctl.calls[2] == (
        "unload_warmup_buffer",
        {"eval_yaml_id": "spatial16_phase3_g4_test__fh0.3_ws0.3"},
    )
    assert ctl.calls[3] == (
        "unload_warmup_buffer",
        {"eval_yaml_id": "spatial16_phase3_g4_test__fh0.5_ws0.5"},
    )
    # Last call: warmup_eval_yaml_id unload (server-side deletes disk dump).
    assert ctl.calls[4] == (
        "unload_warmup_buffer",
        {"eval_yaml_id": "spatial16_phase3_g4_test"},
    )


def test_recipe_end_cleanup_full_grid_yields_18_calls(tmp_path: Path) -> None:
    """Full 16-cell grid: 1 load + 16 cell unloads + 1 warmup unload = 18 RPCs."""
    warmup_yaml_path = _make_warmup_yaml(tmp_path)
    ctl = _RecordingCtl()
    _recipe_end_cleanup(
        ctl,
        recipe_id="g1_test",
        warmup_yaml_path=warmup_yaml_path,
        warmup_yaml_id="spatial16_phase3_g1_test__warmup",
        warmup_eval_yaml_id="spatial16_phase3_g1_test",
        grid=list(GRID),
    )
    assert len(ctl.calls) == 1 + 16 + 1
    # Method tally
    method_counts = {"load_cache_config": 0, "unload_warmup_buffer": 0}
    for m, _ in ctl.calls:
        method_counts[m] += 1
    assert method_counts == {"load_cache_config": 1, "unload_warmup_buffer": 17}


def test_recipe_end_cleanup_tolerates_unload_errors_continues(tmp_path: Path) -> None:
    """A single cell unload raising must not abort the cleanup — the
    remaining unloads (and the final warmup_eval unload) must still run.
    Otherwise a server-side glitch on cell 5 would leak entries 6-15.
    """
    warmup_yaml_path = _make_warmup_yaml(tmp_path)

    class FlakyCtl(_RecordingCtl):
        def unload_warmup_buffer(self, eval_yaml_id: str) -> dict:
            self.calls.append(("unload_warmup_buffer", {"eval_yaml_id": eval_yaml_id}))
            if "fh0.3_ws0.3" in eval_yaml_id:
                raise RuntimeError("simulated wire glitch")
            return {"__ack__": "unload_warmup_buffer"}

    ctl = FlakyCtl()
    _recipe_end_cleanup(
        ctl,
        recipe_id="g4_test",
        warmup_yaml_path=warmup_yaml_path,
        warmup_yaml_id="spatial16_phase3_g4_test__warmup",
        warmup_eval_yaml_id="spatial16_phase3_g4_test",
        grid=[(0.2, 0.2), (0.3, 0.3), (0.5, 0.5)],
    )
    # All 3 cell unloads attempted + 1 final warmup_eval unload + 1 load.
    assert len(ctl.calls) == 1 + 3 + 1
    # Final unload reached.
    assert ctl.calls[-1] == (
        "unload_warmup_buffer",
        {"eval_yaml_id": "spatial16_phase3_g4_test"},
    )


def test_no_intercell_unload_in_cell_loop_source() -> None:
    """Source-level invariant: the cell loop in _run_one_recipe must NOT
    contain a per-cell ``ctl.unload_warmup_buffer(eval_yaml_id)`` call.

    Why: a per-cell unload pops WarmupPool[eval_yaml_id] while server
    ``_current_bundle`` still references that yaml, so the next cell's
    connect-time build_per_connection_components fails. Keep entries
    until recipe end.
    """
    import inspect
    from exp.verdict_factor_judge.run_phase3 import _run_one_recipe
    src = inspect.getsource(_run_one_recipe)
    # Locate the cell loop body — between "for fh_ratio, ws_ratio in GRID:"
    # and "Recipe-end cleanup".
    cell_loop_marker = "for fh_ratio, ws_ratio in GRID:"
    recipe_end_marker = "Recipe-end cleanup"
    assert cell_loop_marker in src and recipe_end_marker in src
    cell_loop_body = src.split(cell_loop_marker, 1)[1].split(recipe_end_marker, 1)[0]
    # The cell loop body must not contain any unload_warmup_buffer call.
    assert "unload_warmup_buffer" not in cell_loop_body, (
        "_run_one_recipe cell loop must not call unload_warmup_buffer per cell; "
        "it was the source of the inter-cell connect-time race. Move it to "
        "_recipe_end_cleanup."
    )


def test_cell_ids_substring_filter_in_phase3_runner_source() -> None:
    """Stage 5 6-server sharding requires phase3 runner to skip cells
    whose ``eval_yaml_id`` does not match any of ``args.cell_ids``
    substrings (mirrors phase4 commit fbadea3's --cell-ids filter).

    Source-level invariant: the cell loop body must contain a substring
    allowlist check that uses ``args.cell_ids`` and ``eval_yaml_id``.
    """
    import inspect
    from exp.verdict_factor_judge.run_phase3 import _run_one_recipe, Args
    src = inspect.getsource(_run_one_recipe)
    cell_loop_marker = "for fh_ratio, ws_ratio in GRID:"
    recipe_end_marker = "Recipe-end cleanup"
    cell_loop_body = src.split(cell_loop_marker, 1)[1].split(recipe_end_marker, 1)[0]
    assert "args.cell_ids" in cell_loop_body, (
        "phase3 runner cell loop must consult args.cell_ids substring allowlist"
    )
    assert "eval_yaml_id" in cell_loop_body
    # Args dataclass must declare cell_ids tuple field.
    fields = {f.name: f.type for f in __import__("dataclasses").fields(Args)}
    assert "cell_ids" in fields


def test_write_na_summary_rows_summary_path_none_returns_rows_no_io(tmp_path: Path) -> None:
    """Allows main() to call with summary_path=None when --summary-out is empty;
    rows still returned for in-memory aggregation by the caller.
    """
    rows = _write_na_summary_rows(
        None,
        recipe_id="g4_f1b_t_w_short_d_jerk",
        warmup_yaml_id="w",
        warmup_eval_yaml_id="e",
        error="y",
        grid=[(0.2, 0.2)],
    )
    assert len(rows) == 1
    # No file should have been created
    assert not (tmp_path / "summary.jsonl").exists()
