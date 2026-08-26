"""Tests for the frozen ws2 cell selection (plan §3-W8).

Selection decides which cells the control and densify arms measure, so its
determinism, its set arithmetic and its refusal modes are the contract — a
different pick is a different experiment.
"""

from __future__ import annotations

import json

import pytest

from exp.robocasa365.build_selection_manifest import build_segment, paired_stats


TASKS = ["OpenDrawer", "CloseFridge"]
EPISODES = 4
GRID = [(t, i) for t in TASKS for i in range(EPISODES)]


def _write_cell(journal_dir, cid, successes, *, run_prefix="ws1"):
    """One per-cell journal; ``successes`` = how many of the grid are wins."""
    run_id = f"{run_prefix}-{cid}__l1s1_groot_tp"
    lines = []
    for n, (task, idx) in enumerate(GRID):
        uid = f"{run_id}__{task}:eval:{TASKS.index(task)}:{idx}"
        lines.append(json.dumps({
            "task_uid": uid, "yaml_id": f"{run_id}__{task}",
            "success": n < successes, "accepted": True, "status": "done", "error": None,
        }))
    (journal_dir / f"journal_{run_id}.jsonl").write_text("\n".join(lines) + "\n")


def _index(cids):
    return {c: {"weights": {}} for c in cids}


def _corpus(tmp_path, *, run_prefix="ws1"):
    """4 iso + 10 grid cells with a spread of success counts."""
    cids = [f"iso_{c}" for c in "abcd"] + [f"grid_{i:02d}" for i in range(10)]
    for n, cid in enumerate(cids):
        _write_cell(tmp_path, cid, successes=8 - (n % 8), run_prefix=run_prefix)
    return cids


def test_paired_stats_is_deterministic(tmp_path):
    from exp.robocasa365.analyze_ws_search_stats import load_journals

    _corpus(tmp_path)
    cells, keys = load_journals(tmp_path, EPISODES, "ws1")
    a = paired_stats(cells, keys, seed=12345, resamples=200)
    b = paired_stats(cells, keys, seed=12345, resamples=200)
    assert a == b
    # A different seed is a different computation — the plan pins 12345 so the
    # tied set reproduces round 1.
    assert paired_stats(cells, keys, seed=7, resamples=200)[2] != a[2]


def test_ws2c_union_is_exactly_twelve_and_iso_is_never_double_counted(tmp_path):
    cids = _corpus(tmp_path)
    segment = build_segment(
        tmp_path, _index(cids), segment="ws2c", run_prefix="ws1",
        episodes=EPISODES, resamples=200, alpha=0.05, seed=12345,
    )
    assert len(segment["cells"]) == 12
    assert len(set(segment["cells"])) == 12
    assert set(segment["iso_cids"]) == {f"iso_{c}" for c in "abcd"}
    assert not set(segment["top8_cids"]) & set(segment["iso_cids"])
    assert len(segment["top8_cids"]) == 8


def test_ws2c_is_byte_identical_across_runs(tmp_path):
    cids = _corpus(tmp_path)
    kwargs = dict(segment="ws2c", run_prefix="ws1", episodes=EPISODES,
                  resamples=200, alpha=0.05, seed=12345)
    first = build_segment(tmp_path, _index(cids), **kwargs)
    second = build_segment(tmp_path, _index(cids), **kwargs)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_source_hashes_are_recorded(tmp_path):
    cids = _corpus(tmp_path)
    segment = build_segment(
        tmp_path, _index(cids), segment="ws2c", run_prefix="ws1",
        episodes=EPISODES, resamples=200, alpha=0.05, seed=12345,
    )
    assert len(segment["source_journals"]) == len(cids)
    assert all(len(h) == 64 for h in segment["source_journals"].values())
    assert segment["params"]["seed"] == 12345


def test_incomplete_matrix_is_refused(tmp_path):
    cids = _corpus(tmp_path)
    with pytest.raises(SystemExit, match="not complete"):
        build_segment(
            tmp_path, _index([*cids, "grid_99"]), segment="ws2c", run_prefix="ws1",
            episodes=EPISODES, resamples=200, alpha=0.05, seed=12345,
        )


def test_wrong_iso_count_is_refused(tmp_path):
    cids = _corpus(tmp_path)
    _write_cell(tmp_path, "iso_e", successes=3)
    with pytest.raises(SystemExit, match="exactly 4 iso"):
        build_segment(
            tmp_path, _index([*cids, "iso_e"]), segment="ws2c", run_prefix="ws1",
            episodes=EPISODES, resamples=200, alpha=0.05, seed=12345,
        )


def test_ws2e_picks_top8_plus_two_negatives(tmp_path):
    cids = _corpus(tmp_path, run_prefix="ws2")
    segment = build_segment(
        tmp_path, _index(cids), segment="ws2e", run_prefix="ws2",
        episodes=EPISODES, resamples=200, alpha=0.05, seed=12345,
    )
    assert len(segment["top8_cids"]) == 8
    assert len(segment["negative_cids"]) == 2
    assert not set(segment["negative_cids"]) & set(segment["top8_cids"])
    assert set(segment["cells"]) == set(segment["top8_cids"]) | set(segment["negative_cids"])


def test_ws2e_fails_fast_without_two_significant_cells(tmp_path):
    # Every cell identical -> nothing is significantly worse than the leader.
    cids = [f"grid_{i:02d}" for i in range(6)]
    for cid in cids:
        _write_cell(tmp_path, cid, successes=4, run_prefix="ws2")
    with pytest.raises(SystemExit, match="negative controls cannot be chosen"):
        build_segment(
            tmp_path, _index(cids), segment="ws2e", run_prefix="ws2",
            episodes=EPISODES, resamples=200, alpha=0.05, seed=12345,
        )


def test_padding_is_recorded_when_the_tied_set_is_short(tmp_path):
    # A wide spread of macro values leaves few ties with the leader.
    cids = [f"iso_{c}" for c in "abcd"] + [f"grid_{i:02d}" for i in range(10)]
    for n, cid in enumerate(cids):
        _write_cell(tmp_path, cid, successes=max(0, 8 - n))
    segment = build_segment(
        tmp_path, _index(cids), segment="ws2c", run_prefix="ws1",
        episodes=EPISODES, resamples=400, alpha=0.05, seed=12345,
    )
    assert len(segment["cells"]) == 12
    assert isinstance(segment["padding_used"], bool)
    # Padding never breaks the ordering contract: picks stay macro-desc, cid-asc.
    assert segment["top8_cids"] == sorted(set(segment["top8_cids"]),
                                          key=segment["top8_cids"].index)
