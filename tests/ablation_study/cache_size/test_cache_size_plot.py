"""Plot provenance: the figure may not disagree with the analysis (plan §8.5)."""
from __future__ import annotations

import pytest

from exp.ablation_study.cache_size.analysis.plot_size import TIERS, authoritative_series


def _result(**over):
    r = {
        "tier_sr": {t: 0.5 + i * 0.05 for i, t in enumerate(TIERS)},
        "tier_ci": {t: [0.4 + i * 0.05, 0.6 + i * 0.05] for i, t in enumerate(TIERS)},
        "teacher_sr": 0.95,
    }
    r.update(over)
    return r


def test_reads_all_three_series_from_the_json():
    sr, ci, teacher = authoritative_series(_result())
    assert set(sr) == set(TIERS)
    assert teacher == 0.95
    assert ci["S1"] == (0.4, 0.6)


@pytest.mark.parametrize("missing", ["tier_sr", "tier_ci", "teacher_sr"])
def test_missing_field_is_fatal(missing):
    r = _result()
    del r[missing]
    with pytest.raises(SystemExit, match=missing):
        authoritative_series(r)


def test_partial_tiers_are_fatal():
    r = _result()
    del r["tier_sr"]["S4"]
    with pytest.raises(SystemExit, match="missing tiers"):
        authoritative_series(r)


def test_numbers_cannot_be_overridden_by_the_caller():
    """plot() takes no SR/CI/teacher parameters -- provenance is not optional."""
    import inspect

    from exp.ablation_study.cache_size.analysis.plot_size import plot

    params = set(inspect.signature(plot).parameters)
    assert not (params & {"tier_sr", "tier_ci", "teacher_sr"}), (
        "the figure must not be able to disagree with the analysis JSON"
    )
