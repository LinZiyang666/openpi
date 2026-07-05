"""Unit tests for the Stage 2 shared analysis foundation (all non-manual).

Covers the plan's locked contracts for ``exp/gate_research/stage2_common.py``:
- ``auc`` on separable / reversed / all-tied / single-class inputs;
- ``mcnemar_exact_p`` deterministic values (b=0,c=5 -> 0.0625; b=c=3 -> 1.0;
  b=1,c=8 -> 0.0390625; b=c=0 -> 1.0);
- ``action_source_seq`` maps skip AND searched-MISS to NEW_INFER, FULL_HIT to
  CACHE_FH, WARM_START to its own class;
- ``cache_run_lengths`` under both include_ws polarities;
- ``reconstruct_searched`` periodic ordinal (re-exported);
- ``load_run_episodes`` / ``load_stage0_episodes`` build aligned EpisodeRecs.
"""

from __future__ import annotations

import json
import math

from exp.gate_research import stage2_common as S


def _write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def _row(uid, step, ht, searched, attempt=1, cp1=0.9, start_t=None, success=True):
    return {"task_uid": uid, "step_idx": step, "hit_type": ht, "searched": searched,
            "attempt": attempt, "cp1_score": cp1, "start_t": start_t, "success": success,
            "yaml_id": uid.rsplit(":", 3)[0]}


# ---------------------------------------------------------------- auc
def test_auc_separable():
    assert S.auc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == 1.0


def test_auc_reversed():
    # positives carry the LOWER scores -> AUC 0.0
    assert S.auc([0.1, 0.2, 0.8, 0.9], [1, 1, 0, 0]) == 0.0


def test_auc_all_tied():
    # every score equal -> averaged ranks -> exactly 0.5
    assert S.auc([0.5, 0.5, 0.5, 0.5], [1, 0, 1, 0]) == 0.5


def test_auc_single_class_is_nan():
    assert math.isnan(S.auc([0.1, 0.2, 0.3], [0, 0, 0]))
    assert math.isnan(S.auc([0.1, 0.2, 0.3], [1, 1, 1]))


# ---------------------------------------------------------------- mcnemar_exact_p
def test_mcnemar_exact_p_known_values():
    assert S.mcnemar_exact_p(0, 5) == 0.0625
    assert S.mcnemar_exact_p(3, 3) == 1.0
    assert math.isclose(S.mcnemar_exact_p(1, 8), 20 / 512)
    assert S.mcnemar_exact_p(0, 0) == 1.0


def test_mcnemar_exact_p_symmetric():
    assert S.mcnemar_exact_p(2, 7) == S.mcnemar_exact_p(7, 2)


# ---------------------------------------------------------------- action_source_seq
def test_action_source_seq_labels():
    hit = ["FULL_HIT", "WARM_START", "MISS", "MISS"]
    searched = [True, True, True, False]
    assert S.action_source_seq(hit, searched) == [
        S.CACHE_FH, S.WARM_START, S.NEW_INFER, S.NEW_INFER]


def test_action_source_seq_skip_over_placeholder_hit():
    # a skipped step stamps a placeholder MISS but must read as NEW_INFER via searched=False
    assert S.action_source_seq(["FULL_HIT"], [False]) == [S.NEW_INFER]


# ---------------------------------------------------------------- cache_run_lengths
def test_cache_run_lengths_exclude_ws():
    src = [S.CACHE_FH, S.CACHE_FH, S.NEW_INFER, S.CACHE_FH, S.WARM_START, S.CACHE_FH, S.NEW_INFER]
    assert S.cache_run_lengths(src, include_ws=False) == [2, 1, 1]


def test_cache_run_lengths_include_ws():
    src = [S.CACHE_FH, S.CACHE_FH, S.NEW_INFER, S.CACHE_FH, S.WARM_START, S.CACHE_FH, S.NEW_INFER]
    assert S.cache_run_lengths(src, include_ws=True) == [2, 3]


def test_cache_run_lengths_trailing_run():
    assert S.cache_run_lengths([S.NEW_INFER, S.CACHE_FH, S.CACHE_FH], include_ws=False) == [2]
    assert S.cache_run_lengths([S.NEW_INFER, S.NEW_INFER], include_ws=True) == []


# ---------------------------------------------------------------- reconstruct_searched (re-exported)
def test_reconstruct_searched_reexport():
    assert S.reconstruct_searched(4, 1, 1) == [True, False, True, False]
    assert S.reconstruct_searched(6, 2, 1) == [True, True, False, True, True, False]


# ---------------------------------------------------------------- load_run_episodes
def test_load_run_episodes_client_controlled(tmp_path):
    rows = [
        _row("y:eval:0:0", 0, "FULL_HIT", True), _row("y:eval:0:0", 5, "MISS", False),
        _row("y:eval:1:2", 0, "FULL_HIT", True), _row("y:eval:1:2", 5, "WARM_START", True, start_t=0.5),
    ]
    _write_jsonl(tmp_path / "rows.jsonl", rows)
    _write_jsonl(tmp_path / "journal.jsonl", [
        {"task_uid": "y:eval:0:0", "yaml_id": "y", "success": True},
        {"task_uid": "y:eval:1:2", "yaml_id": "y", "success": False},
    ])
    manifest = {"per_step_out_path": str(tmp_path / "rows.jsonl"),
                "journal_path": str(tmp_path / "journal.jsonl"),
                "yaml_id": "y", "gate_type": "client_controlled", "replan_steps": 5}
    (tmp_path / "m.json").write_text(json.dumps(manifest))

    eps = {e.unit: e for e in S.load_run_episodes(tmp_path / "m.json")}
    assert set(eps) == {(0, 0), (1, 2)}
    e0 = eps[(0, 0)]
    assert e0.searched_seq == [True, False]
    assert e0.hit_type_seq == ["FULL_HIT", "MISS"]
    assert e0.success is True
    assert S.action_source_seq(e0.hit_type_seq, e0.searched_seq) == [S.CACHE_FH, S.NEW_INFER]
    assert eps[(1, 2)].success is False


def test_load_run_episodes_periodic_reconstructs_searched(tmp_path):
    # periodic rows carry NO searched field; the ordinal (cache_len=1, inference_len=1)
    # gives [True, False]
    rows = [
        {"task_uid": "y:eval:0:0", "step_idx": 0, "hit_type": "FULL_HIT", "attempt": 1,
         "cp1_score": 0.9, "start_t": None, "success": True, "yaml_id": "y"},
        {"task_uid": "y:eval:0:0", "step_idx": 5, "hit_type": "MISS", "attempt": 1,
         "cp1_score": 0.2, "start_t": None, "success": True, "yaml_id": "y"},
    ]
    _write_jsonl(tmp_path / "rows.jsonl", rows)
    _write_jsonl(tmp_path / "journal.jsonl",
                 [{"task_uid": "y:eval:0:0", "yaml_id": "y", "success": True}])
    manifest = {"per_step_out_path": str(tmp_path / "rows.jsonl"),
                "journal_path": str(tmp_path / "journal.jsonl"), "yaml_id": "y",
                "gate_type": "periodic", "cache_len": 1, "inference_len": 1, "replan_steps": 5}
    (tmp_path / "m.json").write_text(json.dumps(manifest))
    e = S.load_run_episodes(tmp_path / "m.json")[0]
    assert e.searched_seq == [True, False]


# ---------------------------------------------------------------- load_stage0_episodes
def test_load_stage0_episodes_per_step_success(tmp_path):
    rows = [
        _row("y:eval:0:0", 0, "FULL_HIT", True, success=True),
        _row("y:eval:0:0", 5, "MISS", True, success=True),
        _row("y:eval:3:7", 0, "MISS", True, success=False),
        _row("y:eval:3:7", 5, "FULL_HIT", True, success=False),
    ]
    _write_jsonl(tmp_path / "gate_rows.jsonl", rows)
    eps = {e.unit: e for e in S.load_stage0_episodes(tmp_path / "gate_rows.jsonl", "y", replan_steps=5)}
    assert set(eps) == {(0, 0), (3, 7)}
    assert eps[(0, 0)].searched_seq == [True, True]
    assert eps[(0, 0)].success is True
    assert eps[(3, 7)].success is False
