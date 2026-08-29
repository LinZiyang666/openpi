"""Tests for block_variance_probe: verdict cost model, blocks, paired SD, R.

The probe supplies the sigma that decides the cost bench's block count R, so
its arithmetic has to stay pinned: a silent change here would move a
pre-registered quantity.
"""

from __future__ import annotations

import json
import math
import statistics

import pytest

from exp.dispatch_surface.analysis.block_variance_probe import (
    COMPUTE_MARGIN,
    LATENCY_EFFECT,
    STAGE1_MS,
    STAGE2_MS,
    STAGE3_MS,
    Z_GATE1,
    Panel,
    decision_compute_ms,
    load_episodes,
    needed_r,
)

FULL_MS = STAGE1_MS + STAGE2_MS + STAGE3_MS


def _per_step(tmp_path, rows):
    path = tmp_path / "per_step.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def _episode(arm, task, init, verdicts, infer_ms, infers):
    """Verdict rows plus the client_timing row for one episode."""
    uid = f"{arm}:eval:{task}:{init}"
    out = [
        {"yaml_id": arm, "task_uid": uid, "step_idx": i * 5,
         "hit_type": h, "start_t": st}
        for i, (h, st) in enumerate(verdicts)
    ]
    out.append({"yaml_id": arm, "_kind": "client_timing", "task_uid": uid,
                "task_id": task, "infer_ms": infer_ms, "infers": infers,
                "steps": 5 * len(verdicts), "success": True})
    return out


# --- verdict cost model ----------------------------------------------------

def test_full_hit_pays_stage1_only():
    assert decision_compute_ms("FULL_HIT", None) == STAGE1_MS


def test_miss_and_unprobed_pay_everything():
    assert decision_compute_ms("MISS", None) == FULL_MS
    # An un-probed step has no verdict at all and runs the full policy.
    assert decision_compute_ms(None, None) == FULL_MS


@pytest.mark.parametrize("start_t", [0.0, 0.3, 0.5, 0.9])
def test_warm_start_runs_start_t_of_stage3(start_t):
    """start_t is what REMAINS to run, not what is skipped.

    run_stage3_from steps from start_t down to 0 in round(start_t * num_steps)
    steps, so start_t=0.3 runs 3 of 10 and saves 70% (pi0_pytorch.py:691).
    Getting this backwards inverts the cost of every warm decision.
    """
    got = decision_compute_ms("WARM_START", start_t)
    assert got == pytest.approx(STAGE1_MS + STAGE2_MS + start_t * STAGE3_MS)


def test_warm_start_is_bounded_by_the_two_extremes():
    # Resuming at t=0 means stage3 is already done; t=1 is a full stage3.
    assert decision_compute_ms("WARM_START", 0.0) == pytest.approx(
        STAGE1_MS + STAGE2_MS
    )
    assert decision_compute_ms("WARM_START", 1.0) == pytest.approx(FULL_MS)
    assert STAGE1_MS < decision_compute_ms("WARM_START", 0.3) < FULL_MS


def test_pinned_warm_tier_saves_seventy_percent_of_stage3():
    """The deployed tier is start_t=0.3; it must cost stage1+stage2+0.3*stage3."""
    got = decision_compute_ms("WARM_START", 0.3)
    assert got == pytest.approx(STAGE1_MS + STAGE2_MS + 0.3 * STAGE3_MS)
    saved = (FULL_MS - got) / STAGE3_MS
    assert saved == pytest.approx(0.7)


# --- episode reduction -----------------------------------------------------

def test_load_episodes_averages_per_decision_and_joins_timing(tmp_path):
    rows = _episode("a", 0, 7, [("FULL_HIT", None), ("MISS", None)],
                    infer_ms=400.0, infers=2)
    episodes = load_episodes(_per_step(tmp_path, rows))
    assert len(episodes) == 1
    e = episodes[0]
    assert (e["arm"], e["task_id"], e["init_idx"]) == ("a", 0, 7)
    assert e["compute_ms"] == pytest.approx((STAGE1_MS + FULL_MS) / 2)
    # The client-side figure is carried as a contrast, never as the cost axis.
    assert e["measured_client_ms"] == pytest.approx(200.0)


def test_episode_without_a_timing_row_is_dropped(tmp_path):
    rows = [r for r in _episode("a", 0, 7, [("MISS", None)], 100.0, 1)
            if r.get("_kind") != "client_timing"]
    assert load_episodes(_per_step(tmp_path, rows)) == []


def test_zero_infers_is_dropped_rather_than_dividing_by_zero(tmp_path):
    rows = _episode("a", 0, 7, [("MISS", None)], infer_ms=0.0, infers=0)
    assert load_episodes(_per_step(tmp_path, rows)) == []


# --- panel / blocks --------------------------------------------------------

def _panel(arms=("a", "b"), tasks=3, inits=4, compute=None, latency=None):
    eps = []
    for arm in arms:
        for t in range(tasks):
            for i in range(inits):
                eps.append({
                    "arm": arm, "task_id": t, "init_idx": i,
                    "compute_ms": (compute or (lambda a, t, i: 100.0))(arm, t, i),
                    "measured_client_ms": (latency or (lambda a, t, i: 200.0))(arm, t, i),
                    "success": True,
                })
    return Panel(eps)


def test_blocks_are_task_stratified_one_init_per_task():
    panel = _panel(tasks=3, inits=4)
    blocks = panel.blocks()
    assert len(blocks) == 4                      # one block per init index
    for block in blocks:
        assert len(block) == 3                   # one cell per task
        assert sorted(t for t, _ in block) == [0, 1, 2]
    # Blocks must be disjoint: no (task, init) cell may appear twice.
    flat = [c for b in blocks for c in b]
    assert len(flat) == len(set(flat))


def test_larger_blocks_take_several_inits_per_task():
    panel = _panel(tasks=3, inits=4)
    blocks = panel.blocks(inits_per_task=2)
    assert len(blocks) == 2
    assert all(len(b) == 6 for b in blocks)
    flat = [c for b in blocks for c in b]
    assert len(flat) == len(set(flat))


def test_blocks_do_not_run_past_the_available_inits():
    panel = _panel(tasks=2, inits=3)
    # 3 inits cannot make two blocks of 2 inits each without reuse.
    assert len(panel.blocks(inits_per_task=2)) == 1


def test_panel_uses_only_cells_every_arm_shares():
    eps = [
        {"arm": "a", "task_id": 0, "init_idx": 0, "compute_ms": 1.0,
         "measured_client_ms": 1.0, "success": True},
        {"arm": "a", "task_id": 0, "init_idx": 1, "compute_ms": 1.0,
         "measured_client_ms": 1.0, "success": True},
        {"arm": "b", "task_id": 0, "init_idx": 0, "compute_ms": 1.0,
         "measured_client_ms": 1.0, "success": True},
    ]
    # init 1 is missing for arm b, so it cannot be part of a paired block.
    assert Panel(eps).cells == [(0, 0)]


# --- paired relative differences -------------------------------------------

def test_identical_arms_have_zero_relative_difference():
    panel = _panel()
    diffs = panel.rel_diffs("a", "b", panel.blocks(), "compute_ms")
    assert diffs == pytest.approx([0.0] * len(diffs))


def test_a_constant_ratio_gives_a_constant_relative_difference():
    # b costs 10% more than a on every cell, however the cells themselves vary.
    def compute(arm, t, i):
        base = 100.0 + 13 * t + 7 * i
        return base * (1.1 if arm == "b" else 1.0)

    panel = _panel(compute=compute)
    diffs = panel.rel_diffs("b", "a", panel.blocks(), "compute_ms")
    assert diffs == pytest.approx([0.1] * len(diffs))
    assert statistics.stdev(diffs) == pytest.approx(0.0, abs=1e-12)


def test_pairing_cancels_variation_shared_by_both_arms():
    """The whole point of a paired block: common init difficulty drops out."""
    def compute(arm, t, i):
        shared = 100.0 + 50 * i          # init difficulty, identical per arm
        return shared + (5.0 if arm == "b" else 0.0)

    panel = _panel(compute=compute)
    blocks = panel.blocks()
    diffs = panel.rel_diffs("b", "a", blocks, "compute_ms")
    a_means = [panel.mean("a", block, "compute_ms") for block in blocks]
    unpaired_cv = statistics.stdev(a_means) / statistics.fmean(a_means)
    # One arm's block means swing by tens of percent; the paired difference
    # keeps only the 5 ms offset, so its spread is an order smaller.
    assert unpaired_cv > 0.25
    assert statistics.stdev(diffs) < unpaired_cv / 10
    assert all(d > 0 for d in diffs)


# --- required R ------------------------------------------------------------

def test_needed_r_matches_the_closed_form():
    sigma = 0.05
    expected = math.ceil(((Z_GATE1 + 0.8416) * sigma / LATENCY_EFFECT) ** 2)
    assert needed_r(sigma, LATENCY_EFFECT, Z_GATE1) == expected


def test_needed_r_grows_with_the_square_of_sigma():
    # Doubling sigma quadruples the blocks needed. Values large enough that
    # the ceiling does not dominate the ratio.
    small = needed_r(0.10, LATENCY_EFFECT, Z_GATE1)
    large = needed_r(0.20, LATENCY_EFFECT, Z_GATE1)
    assert large == pytest.approx(4 * small, rel=0.02)


def test_needed_r_shrinks_when_the_effect_is_easier_to_detect():
    tight = needed_r(0.05, 0.02, Z_GATE1)   # latency gate: 2% margin
    loose = needed_r(0.05, 0.05, Z_GATE1)   # compute gate: 5% margin
    assert loose < tight


def test_ten_episode_block_is_too_small_for_the_frozen_r():
    """The finding this probe exists to establish, pinned as a regression.

    At the measured 10-episode block sigma the GPU-inference-ratio gate needs
    an R above the frozen candidates {5, 10, 15}; enlarging the block is what
    brings it back in range. If a future change to the cost model made the
    10-episode block comfortably sufficient, that conclusion needs re-deriving
    rather than silently flipping.
    """
    assert needed_r(0.110, COMPUTE_MARGIN, Z_GATE1) > 15
    # Roughly halving sigma (what a 40-episode block measures) fixes it.
    assert needed_r(0.051, COMPUTE_MARGIN, Z_GATE1) <= 15
