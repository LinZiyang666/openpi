"""Unit tests for Stage 2c danger-step offline join (all non-manual).

Covers the plan's locked contracts:
- deviate_score>=5 -> danger labels keyed by (task_id, orig_init, step_idx=5*t);
- gate_signals derive prev_score / prev_is_MISS with first-step None;
- inner join + per-signal AUC (separable -> ~1.0), missing keys dropped, early
  phase slice restricts steps.
"""

from __future__ import annotations

import json
import math

from exp.gate_research import stage2c_danger_join as C


def _write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def _gr(task_id, orig, step, ht, cp1, attempt=1):
    uid = f"y:eval:{task_id}:{orig}"
    return {"task_uid": uid, "task_id": task_id, "orig_init_state_idx": orig,
            "step_idx": step, "hit_type": ht, "cp1_score": cp1, "attempt": attempt,
            "yaml_id": "y"}


# ---------------------------------------------------------------- gt index + danger labels
def test_gt_index_from_attrs():
    idx = C.gt_index_from_attrs([(0, 1, 7), (2, 0, 3)])
    assert idx == {"task_0/episode_1": (0, 7), "task_2/episode_0": (2, 3)}


def test_danger_labels_threshold_and_step_key():
    deviate = {"task_0/episode_1": {"deviate_score": [0.5, 6.0, 2.0]}}
    gt_index = {"task_0/episode_1": (0, 7)}
    lab = C.danger_labels(deviate, gt_index, threshold=5.0)
    assert lab == {(0, 7, 0): False, (0, 7, 5): True, (0, 7, 10): False}


def test_danger_labels_skips_unindexed_episode():
    deviate = {"task_9/episode_9": {"deviate_score": [9.0]}}
    assert C.danger_labels(deviate, {}) == {}


# ---------------------------------------------------------------- gate signals
def test_gate_signals_prev_and_orientation(tmp_path):
    rows = [
        _gr(0, 7, 0, "FULL_HIT", 0.9),
        _gr(0, 7, 5, "MISS", 0.5),
        _gr(0, 7, 10, "MISS", 0.2),
    ]
    _write_jsonl(tmp_path / "gate_rows.jsonl", rows)
    sig = C.gate_signals(tmp_path / "gate_rows.jsonl", "y")
    assert sig[(0, 7, 0)]["neg_prev_score"] is None          # first step: no prev
    assert sig[(0, 7, 0)]["prev_is_MISS"] is None
    assert math.isclose(sig[(0, 7, 0)]["neg_cp1_score"], -0.9)
    assert math.isclose(sig[(0, 7, 5)]["neg_prev_score"], -0.9)
    assert sig[(0, 7, 5)]["prev_is_MISS"] == 0.0             # step 0 was FULL_HIT
    assert sig[(0, 7, 10)]["prev_is_MISS"] == 1.0            # step 5 was MISS
    assert math.isclose(sig[(0, 7, 10)]["neg_prev_score"], -0.5)


# ---------------------------------------------------------------- join + auc
def test_join_auc_separable_and_inner_join():
    labels = {(0, 7, 0): False, (0, 7, 5): True, (0, 7, 10): False, (0, 7, 15): True,
              (9, 9, 0): True}  # (9,9,0) absent from signals -> dropped by inner join
    signals = {
        (0, 7, 0): {"neg_prev_score": 0.1, "prev_is_MISS": None, "neg_cp1_score": None, "step_idx": 0.0},
        (0, 7, 5): {"neg_prev_score": 0.9, "prev_is_MISS": None, "neg_cp1_score": None, "step_idx": 5.0},
        (0, 7, 10): {"neg_prev_score": 0.2, "prev_is_MISS": None, "neg_cp1_score": None, "step_idx": 10.0},
        (0, 7, 15): {"neg_prev_score": 0.95, "prev_is_MISS": None, "neg_cp1_score": None, "step_idx": 15.0},
    }
    res = C.join_auc(labels, signals)
    assert res["n_joined"] == 4 and res["n_danger"] == 2
    assert math.isclose(res["auc"]["neg_prev_score"], 1.0)   # danger steps carry the higher score
    assert math.isnan(res["auc"]["prev_is_MISS"])            # all None -> undefined


def test_join_auc_early_phase_slice():
    labels = {(0, 7, 0): False, (0, 7, 5): True, (0, 7, 10): False, (0, 7, 15): True}
    signals = {k: {"neg_prev_score": 0.5, "prev_is_MISS": None, "neg_cp1_score": None,
                   "step_idx": float(k[2])} for k in labels}
    early = C.join_auc(labels, signals, early_frac=0.5)      # max_step 15 -> keep <= 7.5
    assert early["n_joined"] == 2
