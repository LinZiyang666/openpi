"""Phase utilisation: the quantity a capacity change is actually judged on."""

from __future__ import annotations

import json

import pytest

from exp.libero_groot.analysis.gate_pareto.phase_utilisation import load_terminal_records
from exp.libero_groot.analysis.gate_pareto.phase_utilisation import utilisation


def _rec(uid, ts, duration=None, arm="a", status="done"):
    row = {"task_uid": uid, "yaml_id": arm, "phase": "eval", "status": status,
           "success": True, "ts": ts}
    if duration is not None:
        row["duration_s"] = duration
    return row


def test_one_worker_of_six_doing_everything_reads_near_one_sixth():
    """The shape the single-arm phases had: six slots, one busy."""
    # 10 back-to-back episodes of 10s on a 100s span, six workers available.
    records = [_rec(f"u{i}", 10.0 * (i + 1), 10.0) for i in range(10)]
    point = utilisation(records, workers=6)
    # span is 90s (first completion to last), busy is 100s of worker time.
    assert point["phase_wall_clock_s"] == 90.0
    assert point["busy_worker_s"] == 100.0
    assert point["utilisation"] == pytest.approx(100.0 / (6 * 90.0), rel=1e-6)
    assert point["utilisation"] < 0.2


def test_a_saturated_fleet_reads_one_against_a_measured_wall_clock():
    """Six workers busy for the whole 90 s phase."""
    records = [_rec(f"u{i}", 90.0, 90.0) for i in range(6)]
    records[0]["ts"] = 1.0
    point = utilisation(records, workers=6, phase_wall_clock_s=90.0)
    assert point["utilisation"] == pytest.approx(1.0, rel=1e-9)
    assert point["bound"] == "lower"


def test_a_reported_lower_bound_can_never_exceed_one():
    """The earlier version of this test accepted 1.011, which is impossible.

    It used the span between first and last *completion* as the denominator --
    which starts after the opening batch has already run, so capacity came out
    short and the ratio came out above unity. A measured wall clock now yields
    exactly 1.0 for the same data, and an over-unity result with a measured
    clock is raised rather than reported.
    """
    records = [_rec(f"u{i}", 90.0, 90.0) for i in range(6)]
    records[0]["ts"] = 1.0
    with pytest.raises(ValueError, match="exceeds 1.0"):
        utilisation(records, workers=6, phase_wall_clock_s=89.0)


def test_without_a_measured_clock_the_direction_is_declared_unknown():
    """The fallback biases up while the numerator biases down: no bound."""
    records = [_rec(f"u{i}", 90.0, 90.0) for i in range(6)]
    records[0]["ts"] = 1.0
    point = utilisation(records, workers=6)
    assert point["bound"] == "unknown"
    assert point["wall_clock_source"] == "terminal-record span"
    # Over-unity is tolerated here precisely because it is not claimed as a bound.
    assert point["utilisation"] > 1.0


def test_untimed_records_are_reported_not_silently_dropped():
    """Excluding them biases utilisation down, so the count has to be visible."""
    records = [_rec("u0", 10.0, 10.0), _rec("u1", 20.0)]
    point = utilisation(records, workers=1)
    assert point["episodes"] == 2
    assert point["episodes_timed"] == 1
    assert point["episodes_untimed"] == 1


def test_empty_and_single_record_do_not_divide_by_zero():
    assert utilisation([], workers=4)["utilisation"] is None
    assert utilisation([_rec("u0", 5.0, 1.0)], workers=4)["utilisation"] is None


def test_workers_must_be_positive():
    with pytest.raises(ValueError, match="workers must be"):
        utilisation([_rec("u0", 1.0, 1.0)], workers=0)


def test_only_terminal_records_are_loaded(tmp_path):
    path = tmp_path / "j.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(x)
            for x in [
                {"task_uid": "a", "status": "dispatched", "ts": 1.0},
                _rec("b", 2.0, 1.0),
                _rec("c", 3.0, 1.0, status="failed"),
                "not json",
            ]
        ).replace('"not json"', "not json"),
        encoding="utf-8",
    )
    rows, fenced = load_terminal_records(path)
    assert {r["task_uid"] for r in rows} == {"b", "c"}
    assert fenced == 0


def test_fenced_attempts_are_excluded_and_counted(tmp_path):
    """Rejected work must not be credited, and the reader must be told.

    A fenced attempt was real worker time, but the run discarded its result;
    counting it would inflate occupancy with work that bought nothing. The
    count is surfaced because it is the only signal for how loose the lower
    bound is.
    """
    path = tmp_path / "j.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(x)
            for x in [
                dict(_rec("a", 10.0, 5.0), accepted=True),
                dict(_rec("a", 12.0, 9.0), accepted=False),
                _rec("b", 20.0, 5.0),
            ]
        ),
        encoding="utf-8",
    )
    rows, fenced = load_terminal_records(path)
    assert {r["task_uid"] for r in rows} == {"a", "b"}
    assert fenced == 1
    # The 9.0 s fenced attempt must not reach the numerator.
    assert sum(r["duration_s"] for r in rows) == 10.0


def test_unknown_estimate_is_not_also_described_as_a_lower_bound(capsys, tmp_path, monkeypatch):
    """One condition must drive both the structured field and the prose.

    An operator reading "lower bound" under an unknown-direction estimate would
    reasonably gate on it, which is exactly what the unknown case forbids.
    """
    import sys

    from exp.libero_groot.analysis.gate_pareto import phase_utilisation as pu

    path = tmp_path / "j.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(x)
            for x in [
                dict(_rec("a", 10.0, 5.0), accepted=True),
                dict(_rec("b", 20.0, 5.0), accepted=False),
                dict(_rec("c", 30.0, 5.0), accepted=True),
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(sys, "argv", ["phase_utilisation.py", str(path), "--workers", "2"])
    pu.main()
    out = capsys.readouterr().out
    assert '"bound": "unknown"' in out
    assert "lower bound" not in out
