"""Unit tests for the Stage 1b N1 live analyzer (all non-manual)."""

from __future__ import annotations

import json

import pytest

from exp.gate_research import analyze_n1_live as A


def _write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def _row(uid, step, ht, searched, attempt, cp1=0.9, start_t=None, success=True):
    """A live per-step row (client_controlled: carries authoritative searched)."""
    return {"task_uid": uid, "step_idx": step, "hit_type": ht, "searched": searched,
            "attempt": attempt, "cp1_score": cp1, "start_t": start_t, "success": success,
            "yaml_id": uid.rsplit(":", 3)[0]}


def _gr(uid, step, ht, attempt=1, cp1=0.9, start_t=None, searched=True):
    """A Stage-0 always-search gate_row (every step searched)."""
    return {"task_uid": uid, "step_idx": step, "hit_type": ht, "attempt": attempt,
            "cp1_score": cp1, "start_t": start_t, "searched": searched,
            "yaml_id": uid.rsplit(":", 3)[0]}


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def test_parse_task_uid():
    assert A.parse_task_uid("some_yaml_id:eval:3:28") == (3, 28)


def test_inf_value():
    assert A.inf_value("FULL_HIT", None) == 0.0
    assert A.inf_value("MISS", None) == 1.0
    assert A.inf_value("WARM_START", 0.5) == 0.75


def test_reconstruct_searched_periodic():
    assert A.reconstruct_searched(4, 1, 1) == [True, False, True, False]
    assert A.reconstruct_searched(6, 2, 1) == [True, True, False, True, True, False]


# ----------------------------------------------------------------------
# dedup: episode-global max attempt (NOT per-(uid,step))
# ----------------------------------------------------------------------
def test_dedup_global_max_attempt_shorter_higher():
    uid = "y:eval:0:0"
    rows = [
        _row(uid, 0, "MISS", True, 1), _row(uid, 1, "MISS", True, 1),
        _row(uid, 2, "MISS", True, 1), _row(uid, 3, "MISS", True, 1),
        _row(uid, 0, "FULL_HIT", True, 2), _row(uid, 1, "FULL_HIT", True, 2),
        {"_kind": "episode_summary", "task_uid": uid, "attempt": 2},
    ]
    ep = A.dedup_episodes(rows)[uid]
    assert len(ep) == 2  # only attempt-2 steps, not a 4-step franken-episode
    assert all(r["attempt"] == 2 and r["hit_type"] == "FULL_HIT" for r in ep)


# ----------------------------------------------------------------------
# journal
# ----------------------------------------------------------------------
def test_journal_success_and_filter():
    rows = [
        {"task_uid": "a:eval:0:0", "yaml_id": "a", "success": True},
        {"task_uid": "a:eval:0:1", "yaml_id": "a", "success": False},
        {"task_uid": "b:eval:0:0", "yaml_id": "b", "success": True},
    ]
    assert A.journal_success(rows, yaml_id="a") == {"a:eval:0:0": True, "a:eval:0:1": False}


def test_journal_duplicate_fail_fast():
    rows = [{"task_uid": "a:eval:0:0", "yaml_id": "a", "success": True},
            {"task_uid": "a:eval:0:0", "yaml_id": "a", "success": False}]
    with pytest.raises(ValueError, match="duplicate terminal task_uid"):
        A.journal_success(rows)


# ----------------------------------------------------------------------
# baseline_inf_ratio (dedup + strict searched)
# ----------------------------------------------------------------------
def test_baseline_inf_ratio(tmp_path):
    p = tmp_path / "gr.jsonl"
    _write_jsonl(p, [_gr("base:eval:0:0", 0, "FULL_HIT"), _gr("base:eval:0:0", 1, "MISS"),
                     _gr("other:eval:0:0", 0, "MISS")])
    assert A.baseline_inf_ratio(p, "base", 1) == pytest.approx(0.5)  # (0 + 1) / 2


def test_baseline_inf_ratio_dedups_max_attempt(tmp_path):
    p = tmp_path / "gr.jsonl"
    # attempt 1: two MISS (inf 1,1); attempt 2 (accepted): two FULL_HIT (inf 0,0)
    _write_jsonl(p, [_gr("base:eval:0:0", 0, "MISS", attempt=1), _gr("base:eval:0:0", 1, "MISS", attempt=1),
                     _gr("base:eval:0:0", 0, "FULL_HIT", attempt=2), _gr("base:eval:0:0", 1, "FULL_HIT", attempt=2)])
    assert A.baseline_inf_ratio(p, "base", 1) == pytest.approx(0.0)  # only attempt-2 kept


def test_baseline_inf_ratio_missing_config(tmp_path):
    p = tmp_path / "gr.jsonl"
    _write_jsonl(p, [_gr("other:eval:0:0", 0, "MISS")])
    with pytest.raises(ValueError):
        A.baseline_inf_ratio(p, "base", 1)


def test_baseline_inf_ratio_rejects_non_searched(tmp_path):
    p = tmp_path / "gr.jsonl"
    _write_jsonl(p, [_gr("base:eval:0:0", 0, "MISS", searched=False)])
    with pytest.raises(ValueError):
        A.baseline_inf_ratio(p, "base", 1)


# ----------------------------------------------------------------------
# run_metrics: client_controlled vs periodic + data integrity
# ----------------------------------------------------------------------
def _manifest(**kw):
    base = dict(run_id="test", gate_type="client_controlled", yaml_id="y",
                n_episodes=1, replan_steps=1)
    base.update(kw)
    return base


def test_run_metrics_client_controlled(tmp_path):
    uid = "y:eval:0:0"
    per = tmp_path / "rows.jsonl"
    _write_jsonl(per, [_row(uid, 0, "FULL_HIT", True, 1), _row(uid, 1, "MISS", True, 1),
                       _row(uid, 2, "MISS", False, 1), _row(uid, 3, "WARM_START", True, 1, start_t=0.5)])
    jr = tmp_path / "j.jsonl"
    _write_jsonl(jr, [{"task_uid": uid, "yaml_id": "y", "success": True}])
    met = A.run_metrics(_manifest(per_step_out_path=str(per), journal_path=str(jr)))
    assert met["skip_pct"] == pytest.approx(0.25)
    assert met["live_inf_ratio"] == pytest.approx(2.75 / 4)  # 0 + 1 + 1(skip) + 0.75
    assert met["verdict_mix"] == {"FULL_HIT": 1, "MISS": 1, "WARM_START": 1}
    assert met["sr"] == pytest.approx(1.0)


def test_run_metrics_missing_searched_raises(tmp_path):
    uid = "y:eval:0:0"
    per = tmp_path / "rows.jsonl"
    row = _row(uid, 0, "MISS", True, 1)
    del row["searched"]  # data-integrity break: missing searched must NOT become skip
    _write_jsonl(per, [row])
    jr = tmp_path / "j.jsonl"
    _write_jsonl(jr, [{"task_uid": uid, "yaml_id": "y", "success": True}])
    with pytest.raises(ValueError, match="searched"):
        A.run_metrics(_manifest(per_step_out_path=str(per), journal_path=str(jr)))


def test_run_metrics_episode_count_mismatch_raises(tmp_path):
    uid = "y:eval:0:0"
    per = tmp_path / "rows.jsonl"
    _write_jsonl(per, [_row(uid, 0, "MISS", True, 1)])
    jr = tmp_path / "j.jsonl"
    _write_jsonl(jr, [{"task_uid": uid, "yaml_id": "y", "success": True}])
    with pytest.raises(ValueError, match="expects 5"):
        A.run_metrics(_manifest(per_step_out_path=str(per), journal_path=str(jr), n_episodes=5))


def test_check_complete_decisions_ok_real_spacing():
    # real data step_idx = ordinal * replan_steps, trusted spacing 5 -> 0,5,10 valid.
    A.check_complete_decisions({"y:eval:0:0": [{"step_idx": 0}, {"step_idx": 5}, {"step_idx": 10}]}, 5)


def test_check_complete_decisions_gap_raises():
    eps = {"y:eval:0:0": [{"step_idx": 0}, {"step_idx": 1}, {"step_idx": 2}],
           "y:eval:0:1": [{"step_idx": 0}, {"step_idx": 2}]}  # dropped ordinal 1
    with pytest.raises(ValueError, match="expected multiples"):
        A.check_complete_decisions(eps, 1)


def test_check_complete_decisions_uniform_drop_caught():
    # every episode dropped every other decision: 0,5,10 -> 0,10,20. Inferring the
    # spacing from data would say 10 and pass; the TRUSTED spacing 5 catches it.
    eps = {"y:eval:0:0": [{"step_idx": 0}, {"step_idx": 10}, {"step_idx": 20}],
           "y:eval:0:1": [{"step_idx": 0}, {"step_idx": 10}, {"step_idx": 20}]}
    with pytest.raises(ValueError, match="expected multiples"):
        A.check_complete_decisions(eps, 5)


def test_check_complete_decisions_bad_spacing():
    with pytest.raises(ValueError, match="trusted replan spacing"):
        A.check_complete_decisions({"y:eval:0:0": [{"step_idx": 0}]}, 0)


def test_run_metrics_unit_mismatch_raises(tmp_path):
    per = tmp_path / "rows.jsonl"
    _write_jsonl(per, [_row("y:eval:0:0", 0, "MISS", True, 1)])
    jr = tmp_path / "j.jsonl"
    _write_jsonl(jr, [{"task_uid": "y:eval:0:0", "yaml_id": "y", "success": True},
                      {"task_uid": "y:eval:0:1", "yaml_id": "y", "success": True}])
    with pytest.raises(ValueError, match="per-step episode set"):
        A.run_metrics(_manifest(per_step_out_path=str(per), journal_path=str(jr), n_episodes=None))


def test_run_metrics_periodic_reconstruct(tmp_path):
    uid = "p:eval:0:0"
    per = tmp_path / "rows.jsonl"
    _write_jsonl(per, [{"task_uid": uid, "step_idx": i, "hit_type": ht, "start_t": None,
                        "attempt": 1, "success": True, "yaml_id": "p"}
                       for i, ht in enumerate(["FULL_HIT", "MISS", "FULL_HIT", "MISS"])])
    jr = tmp_path / "j.jsonl"
    _write_jsonl(jr, [{"task_uid": uid, "yaml_id": "p", "success": True}])
    met = A.run_metrics(_manifest(gate_type="periodic", yaml_id="p", cache_len=1,
                                  inference_len=1, per_step_out_path=str(per), journal_path=str(jr)))
    assert met["skip_pct"] == pytest.approx(0.5)  # T,F,T,F
    assert met["live_inf_ratio"] == pytest.approx(0.5)


# ----------------------------------------------------------------------
# pairing / mcnemar / net / match_periodic
# ----------------------------------------------------------------------
def test_mcnemar_pairing():
    n1 = {(0, 0): True, (0, 1): False, (0, 2): True}
    base = {(0, 0): True, (0, 1): True, (0, 2): False}
    r = A.mcnemar(n1, base)
    assert r["n_paired"] == 3 and r["b"] == 1 and r["c"] == 1


def test_mcnemar_requires_equal_units():
    n1 = {(0, 0): True, (0, 1): True}
    base = {(0, 0): True}  # missing a unit
    with pytest.raises(ValueError, match="unit sets differ"):
        A.mcnemar(n1, base)


def test_net_row():
    assert A.net_row(0.2, 0.01)["stock_2.6k"] == pytest.approx(3.8)  # 0.2*34 - 0.01*300


def _n1(run_id="n1a", skip=0.15, sr=None):
    return {"run_id": run_id, "suite": "s", "config": "c", "skip_pct": skip,
            "sr_by_unit": sr or {(0, 0): True, (0, 1): True}}


def _per(run_id="per_a", matched="n1a", skip=0.16, sr=None):
    return {"run_id": run_id, "suite": "s", "config": "c", "matched_to": matched,
            "skip_pct": skip, "sr_by_unit": sr or {(0, 0): True, (0, 1): False}}


def test_match_periodic_verdict():
    v = A.match_periodic(_n1(), [_per()])
    assert v["periodic_run_id"] == "per_a"
    assert v["skip_match_ok"] and v["skip_delta_pp"] == pytest.approx(1.0)
    assert v["n1_ge_periodic"] and v["periodic_pass"] and v["n1_sr"] == pytest.approx(1.0)


def test_match_periodic_none_when_unmatched():
    assert A.match_periodic(_n1(), [_per(matched="other")]) is None


def test_match_periodic_oob_budget_fails():
    # N1 SR >= periodic SR, but |Δskip| = 5pp > 2pp -> periodic_pass must be FALSE.
    v = A.match_periodic(_n1(skip=0.15), [_per(skip=0.20)])
    assert v["n1_ge_periodic"] and not v["skip_match_ok"] and not v["periodic_pass"]


def test_match_periodic_duplicate_raises():
    with pytest.raises(ValueError, match="matched periodic"):
        A.match_periodic(_n1(), [_per(run_id="p1"), _per(run_id="p2")])


def test_match_periodic_suite_config_mismatch_raises():
    bad = _per()
    bad["config"] = "other"
    with pytest.raises(ValueError, match="suite/config"):
        A.match_periodic(_n1(), [bad])


# ----------------------------------------------------------------------
# offline-replay diagnostic (shared N1GateState)
# ----------------------------------------------------------------------
def test_replay_offline(tmp_path):
    gr = tmp_path / "gr.jsonl"
    _write_jsonl(gr, [_gr("base:eval:0:0", i, "FULL_HIT", cp1=0.99) for i in range(5)])
    out = A.replay_offline(gr, "base", theta_low=0.5, theta_high=0.5, j=2, M=3, replan_steps=1)
    assert out["skip_pct"] == pytest.approx(0.0)  # all-high -> never skips
    assert out["verdict_mix"] == {"FULL_HIT": 5}
