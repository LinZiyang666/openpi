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


def test_sidecar_rows_are_detected_by_schema_not_by_tag_value():
    """The two writers use different `_kind` values; both must be excluded.

    gate_research per-step logs tag their per-episode record "episode_summary",
    the conductor's per-step writer tags its timing record "client_timing".
    Matching only the first turned the second into a "missing step_idx" anomaly
    and quarantined the entire yaml -- a whole batch read as zero usable rows.
    """
    rows = [
        {"yaml_id": "y", "task_id": 0, "subset_init_state_idx": 0, "episode_id": 0,
         "attempt": 1, "step_idx": s}
        for s in (0, 5, 10)
    ]
    rows.append({"_kind": "client_timing", "yaml_id": "y", "task_id": 0,
                 "subset_init_state_idx": 0, "attempt": 1, "infer_ms": 12.3})
    rows.append({"_kind": "episode_summary", "yaml_id": "y", "task_id": 0,
                 "subset_init_state_idx": 0, "episode_id": 0, "attempt": 1})
    kept, report = _timeaxis.to_cycles(rows, 5)
    assert report.episode_summary == 2, "both sidecar tags must be excluded by schema"
    assert report.missing_step_idx == 0
    assert report.quarantined_yamls == set()
    assert [r["cycle"] for r in kept] == [0, 1, 2]


def test_a_genuinely_malformed_row_still_quarantines():
    """Widening the sidecar rule must not weaken the missing-step_idx gate."""
    rows = [
        {"yaml_id": "y", "task_id": 0, "subset_init_state_idx": 0, "episode_id": 0,
         "attempt": 1, "step_idx": 0},
        {"yaml_id": "y", "task_id": 0, "subset_init_state_idx": 0, "episode_id": 0,
         "attempt": 1},  # no _kind, no step_idx -> real anomaly
    ]
    kept, report = _timeaxis.to_cycles(rows, 5)
    assert report.missing_step_idx == 1
    assert "y" in report.quarantined_yamls
    assert kept == []
