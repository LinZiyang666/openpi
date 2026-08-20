"""P3's verdict tool: the pairing, the retry rule, and the exact test.

Every case here is a way the comparison has gone wrong before, or a way it
could: an unpaired comparison quoted as paired, a retried episode counted
twice, a one-sided reading of a two-sided test.
"""
import json
import math
import pathlib

import pytest

from exp.rl_router.analysis.paired_mcnemar import (
    compare,
    exact_mcnemar,
    terminal_outcomes,
)


def _journal(tmp_path: pathlib.Path, name: str, rows: list[dict]) -> str:
    path = tmp_path / name
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return str(path)


def test_terminal_outcome_is_the_highest_attempt_not_the_last_line(tmp_path) -> None:
    """The driver replays failed slots, so one uid can appear several times. File
    order depends on worker interleaving; the attempt number does not."""
    path = _journal(tmp_path, "j.jsonl", [
        {"task_uid": "u1", "attempt": 2, "success": True},
        {"task_uid": "u1", "attempt": 1, "success": False},   # earlier attempt, later line
        {"task_uid": "u2", "attempt": 1, "success": False},
    ])
    assert terminal_outcomes(path) == {"u1": True, "u2": False}


def test_missing_attempt_field_defaults_to_one(tmp_path) -> None:
    """Journals written before the retry path existed carry no attempt field;
    they must still load rather than raise."""
    path = _journal(tmp_path, "j.jsonl", [{"task_uid": "u1", "success": True}])
    assert terminal_outcomes(path) == {"u1": True}


def test_comparison_pairs_only_shared_slots_and_reports_the_rest(tmp_path) -> None:
    """Unshared slots are dropped and COUNTED. A silent drop is how an unpaired
    comparison gets quoted as a paired one."""
    a = {"u1": True, "u2": False, "u3": True}
    b = {"u1": False, "u2": False, "u9": True}
    r = compare(a, b)
    assert r["n_paired"] == 2
    assert r["n_only_a"] == 1 and r["n_only_b"] == 1
    assert r["table"] == {"both": 0, "only_a": 1, "only_b": 0, "neither": 1}


def test_disjoint_slot_sets_are_refused_not_averaged() -> None:
    """Two arms that ran different slots have no paired comparison at all; the
    tool must fail loudly instead of returning a marginal difference."""
    with pytest.raises(SystemExit, match="no shared task_uids"):
        compare({"u1": True}, {"u2": True})


def test_exact_mcnemar_matches_the_binomial_by_hand() -> None:
    """Exact, not chi-square: the pre-registered detectable effect sits where
    the approximation's tail is least trustworthy."""
    # m=10 discordant, all one way -> 2 * (1/2)^10
    assert exact_mcnemar(10, 0) == pytest.approx(2 * (1 / 2) ** 10)
    # symmetric discordance -> no evidence at all
    assert exact_mcnemar(7, 7) == 1.0
    # m=5, k=1: 2 * (C(5,0)+C(5,1)) / 2^5
    assert exact_mcnemar(4, 1) == pytest.approx(2 * (1 + 5) / 32)
    # no discordant pairs at all -> nothing to test
    assert exact_mcnemar(0, 0) == 1.0


def test_identical_arms_give_p_one_and_zero_difference() -> None:
    """The null case has to read as null: same outcomes on every slot means no
    discordant pairs, difference exactly zero, p = 1."""
    outcomes = {f"u{i}": i % 3 == 0 for i in range(60)}
    r = compare(dict(outcomes), dict(outcomes))
    assert r["paired_diff"] == 0.0
    assert r["discordant"] == 0
    assert r["p_value"] == 1.0
    assert r["significant_at_05"] is False


def test_paired_se_uses_discordant_pairs_not_the_unpaired_formula() -> None:
    """sqrt(m)/n, not the unpaired sqrt(pA(1-pA)/n + pB(1-pB)/n).

    Built at P3's actual shape: n=1,000 slots, both arms at SR 0.70, outcomes
    strongly correlated because the (task, init) difficulty is shared. That
    correlation is exactly what pairing exploits -- here it nearly halves the
    standard error, which is the whole reason a ~0.02 effect is detectable at
    this budget instead of needing four times the episodes.
    """
    a, b = {}, {}
    for i in range(640):                     # both succeed
        a[f"u{i}"] = b[f"u{i}"] = True
    for i in range(640, 880):                # both fail
        a[f"u{i}"] = b[f"u{i}"] = False
    for i in range(880, 940):                # only A
        a[f"u{i}"], b[f"u{i}"] = True, False
    for i in range(940, 1000):               # only B
        a[f"u{i}"], b[f"u{i}"] = False, True

    r = compare(a, b)
    assert r["n_paired"] == 1000
    assert r["sr_a"] == pytest.approx(0.70) and r["sr_b"] == pytest.approx(0.70)
    assert r["discordant"] == 120
    assert r["paired_se"] == pytest.approx(math.sqrt(120) / 1000)

    unpaired = math.sqrt(2 * 0.70 * 0.30 / 1000)      # SE of the DIFFERENCE, unpaired
    assert r["paired_se"] < unpaired / 1.5


def test_a_two_sided_verdict_does_not_depend_on_which_arm_is_first() -> None:
    """Swapping the arms flips the sign and leaves the p-value alone. A test
    whose significance depended on argument order would be one-sided by
    accident."""
    a = {f"u{i}": i >= 8 for i in range(40)}
    b = {f"u{i}": i >= 15 for i in range(40)}
    fwd, rev = compare(a, b), compare(b, a)
    assert fwd["p_value"] == rev["p_value"]
    assert fwd["paired_diff"] == pytest.approx(-rev["paired_diff"])
    assert fwd["significant_at_05"] == rev["significant_at_05"]
