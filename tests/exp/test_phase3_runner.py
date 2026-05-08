"""Tests for exp/verdict_factor_judge/run_phase3.py runner helpers.

Covers G2 R1 B2: solver fail-fast _NA propagation must reach
``per_yaml_summary.jsonl`` so downstream aggregation does not silently
lose 16 cells per failed recipe.
"""

from __future__ import annotations

import json
from pathlib import Path

from exp.verdict_factor_judge.phase3_spec import GRID
from exp.verdict_factor_judge.run_phase3 import _write_na_summary_rows


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
