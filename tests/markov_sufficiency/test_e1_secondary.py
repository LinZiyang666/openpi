"""Tests for the E1 secondary read-outs.

The point of E1-O is that the baseline moves with the contrast: an
oracle-aligned B has to be compared against an oracle-aligned A, because the
candidate-pool restriction changes the residual on its own. Comparing it
against the global A would credit the restriction to history. These tests pin
that, the fixed decision rule, and the exclusion of oracle rows from the
exploratory grid.
"""

from __future__ import annotations

import json

import pytest

from exp.markov_sufficiency import analyze_e1_secondary as sec


def _rows():
    """Two episodes where the oracle-aligned A is much better than the global A.

    If the oracle B were scored against the global A it would look like a large
    history gain; against its own aligned baseline it is exactly zero.
    """
    out = []
    for ep in ("ep0", "ep1"):
        base = {"suite": "s", "trajectory_id": ep, "k": 1, "padding": False, "step_idx": 0}
        out += [
            {**base, "group": "A", "residual": 1.0},
            {**base, "group": "B-d3", "residual": 0.9},
            {**base, "group": "O-A-e0.05", "residual": 0.5},
            {**base, "group": "O-B-d3-e0.05", "residual": 0.5},
        ]
    return out


def test_oracle_baseline_is_the_aligned_a_not_the_global_a():
    cells = sec.oracle_cells(_rows())
    assert len(cells) == 1
    cell = cells[0]
    # Aligned A == aligned B -> zero gain. Against the global A (1.0) the same
    # rows would have shown a 50% "gain" that belongs to the alignment.
    assert cell["hodges_lehmann"] == pytest.approx(0.0)
    assert cell["relative_delta"] == pytest.approx(0.0)
    assert cell["median_residual_A"] == pytest.approx(0.5)


def test_oracle_cell_carries_its_eps_and_depth():
    cell = sec.oracle_cells(_rows())[0]
    assert (cell["eps"], cell["depth"], cell["suite"]) == (0.05, 3, "s")


@pytest.mark.parametrize(
    ("delta", "lo", "expected"),
    [
        (0.08, 0.01, "h_b_supported"),
        (0.03, 0.01, "positive_but_below_floor"),
        (0.08, -0.01, "not_supported"),
        (float("nan"), float("nan"), "no_data"),
    ],
)
def test_oracle_verdict_needs_both_the_floor_and_a_positive_bound(delta, lo, expected):
    assert sec._oracle_verdict({"relative_delta": delta, "hl_ci": [lo, 1.0]}) == expected


def test_effect_floor_matches_the_registered_five_percent():
    assert sec.ORACLE_EFFECT_FLOOR == 0.05


def test_exploratory_grid_excludes_oracle_groups_and_the_baseline():
    cells = sec.exploratory_cells(_rows(), ks=(1,))
    groups = {c["group"] for c in cells}
    assert groups == {"B-d3"}, "oracle groups and A must not appear as exploratory cells"
    assert all(c["exploratory"] is True for c in cells)


def test_oracle_pairs_are_discovered_not_hardcoded():
    rows = _rows()
    for ep in ("ep0", "ep1"):
        base = {"suite": "s", "trajectory_id": ep, "k": 1, "padding": False, "step_idx": 0}
        rows += [
            {**base, "group": "O-A-e0.1", "residual": 0.6},
            {**base, "group": "O-B-d5-e0.1", "residual": 0.4},
        ]
    pairs = sec._oracle_pairs(rows)
    assert ("0.05", 3) in pairs and ("0.1", 5) in pairs


def test_analyse_reads_jsonl_and_reports_no_family_verdict(tmp_path):
    p = tmp_path / "rows.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in _rows()))
    out = sec.analyse([str(p)])
    assert out["n_rows"] == len(_rows())
    assert "verdict" not in out, "only family_analysis may emit a family verdict"
    assert "not adjusted" in out["multiplicity"]
    assert out["oracle"] and out["exploratory"]


def test_analyse_fails_loudly_on_empty_input(tmp_path):
    p = tmp_path / "empty.jsonl"
    p.write_text("")
    with pytest.raises(SystemExit, match="no residual rows"):
        sec.analyse([str(p)])


# ------------------------------------------------------------------
# Degeneracy guard
# ------------------------------------------------------------------


def test_pool_diagnostic_flags_a_collapsed_pool(monkeypatch):
    """A pool of one makes the contrast zero by construction -- it must be flagged."""
    import numpy as np

    from exp.markov_sufficiency import _library
    from exp.markov_sufficiency import e1_loeo_residual as e1

    class _P:
        task_key = "t"

    class _E:
        payload = _P()

        def __init__(self, traj, step):
            self.trajectory_id, self.step_idx = traj, step

    entries = [_E(f"ep{e}", s) for e in range(3) for s in range(4)]
    by_traj = {}
    for x in entries:
        by_traj.setdefault(x.trajectory_id, []).append(x)
    lib = _library.Library(entries=entries, by_id={}, by_traj=by_traj,
                           vector_dims={}, key_builder_type="t", meta={})
    monkeypatch.setattr(_library, "load_library", lambda p: lib)
    # Progress is unique per (episode, step), so a zero eps leaves no candidate
    # from any *other* episode -- exactly the collapse the guard exists to catch.
    monkeypatch.setattr(e1, "_progress", lambda lib, e: float(e.step_idx) + 0.1 * int(e.trajectory_id[-1]))

    rows = sec.oracle_pool_sizes("ignored", eps_values=(0.0,))
    assert rows[0]["degenerate"] is True
    assert rows[0]["share_degenerate"] > 0.05

    wide = sec.oracle_pool_sizes("ignored", eps_values=(10.0,))
    assert wide[0]["degenerate"] is False
    assert wide[0]["aligned_median"] == pytest.approx(wide[0]["unaligned_median"])
    assert np.isfinite(wide[0]["aligned_min"])


def test_cli_rejects_a_malformed_pool_diagnostic(tmp_path):
    p = tmp_path / "rows.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in _rows()))
    with pytest.raises(SystemExit):
        sec.main(["--rows", str(p), "--pool-diagnostic", "no-equals", "--out", str(tmp_path / "o.json")])
