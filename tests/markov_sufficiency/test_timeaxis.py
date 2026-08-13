"""Tests for the environment-step to inference-cycle gate.

The gate exists because per-step logs count environment steps (0, 5, 10, ...)
while cache entries count inference cycles (0, 1, 2, ...). Comparing them
un-converted mislabels the back half of every episode.
"""

from __future__ import annotations

from exp.markov_sufficiency import _timeaxis


def _row(step, yaml_id="y", ep=0, **extra):
    return {
        "yaml_id": yaml_id,
        "task_id": 0,
        "subset_init_state_idx": 0,
        "episode_id": ep,
        "attempt": 0,
        "step_idx": step,
        **extra,
    }


def test_converts_env_steps_to_cycles():
    rows = [_row(0), _row(5), _row(10)]
    kept, report = _timeaxis.to_cycles(rows, replan_steps=5)
    assert [r["cycle"] for r in kept] == [0, 1, 2]
    assert report.kept_rows == 3
    assert not report.quarantined_yamls


def test_episode_summary_excluded_by_schema_not_counted_as_anomaly():
    rows = [_row(0), {"_kind": "episode_summary", "yaml_id": "y", "episode_id": 0}, _row(5)]
    kept, report = _timeaxis.to_cycles(rows, replan_steps=5)
    assert report.episode_summary == 1
    # It carries no step_idx, but that is the schema, not a missing field.
    assert report.missing_step_idx == 0
    assert not report.quarantined_yamls
    assert len(kept) == 2


def test_non_divisible_step_quarantines_the_yaml():
    rows = [_row(0), _row(3), _row(10)]
    kept, report = _timeaxis.to_cycles(rows, replan_steps=5)
    assert report.non_divisible == 1
    assert report.quarantined_yamls == {"y"}
    assert kept == []


def test_irregular_spacing_quarantines_the_yaml():
    rows = [_row(0), _row(5), _row(20)]
    kept, report = _timeaxis.to_cycles(rows, replan_steps=5)
    assert "y" in report.bad_spacing
    assert kept == []


def test_non_contiguous_cycles_quarantine_the_yaml():
    rows = [_row(5), _row(10)]  # starts at cycle 1, not 0
    kept, report = _timeaxis.to_cycles(rows, replan_steps=5)
    assert "y" in report.non_contiguous
    assert kept == []


def test_missing_step_idx_on_a_step_row_is_an_anomaly():
    rows = [_row(0), {"yaml_id": "y", "episode_id": 0}]
    _, report = _timeaxis.to_cycles(rows, replan_steps=5)
    assert report.missing_step_idx == 1
    assert report.quarantined_yamls == {"y"}


def test_quarantine_is_per_yaml_not_per_episode():
    good = [_row(0, yaml_id="ok"), _row(5, yaml_id="ok")]
    bad = [_row(0, yaml_id="bad", ep=1), _row(7, yaml_id="bad", ep=1)]
    kept, report = _timeaxis.to_cycles(good + bad, replan_steps=5)
    assert {r["yaml_id"] for r in kept} == {"ok"}
    assert report.quarantined_yamls == {"bad"}


def test_report_serialises_for_manifest():
    _, report = _timeaxis.to_cycles([_row(0)], replan_steps=5)
    payload = report.as_dict()
    assert isinstance(payload["quarantined_yamls"], list)
    assert payload["kept_rows"] == 1
