"""Unit tests for the trajectory step-weight screening emitter + analysis.

Covers the plan's locked contracts:
- exact search-matrix counts (|S1|=24/depth, |union| d3=52 / d4=60 / d5=59, total 171);
- winner modality weights resolved exactly from grid3 (sum==1.0, full key names);
- derived config differs from its rebuilt base ONLY in trajectory_weights;
- emit stale-guard refuses out-of-tree paths;
- mutually-exclusive shape classifier + orthogonal current_dominant tag;
- journal latest-ts dedup before pairing + McNemar + acceptance gate.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from openpi.cache.config import load_cache_config

from exp.weighted_sum.emit_yamls import build_eval_config
from exp.weighted_sum import emit_traj_weight_alloc as E
from exp.weighted_sum.analysis import analyze_stepweight as A

REPO = Path(__file__).resolve().parents[2]
CALIB = REPO / "exp/weighted_sum/data/libero_spatial/phase1/calibration_normalizers.json"


# ---------------------------------------------------------------- search matrix
def test_union_counts_locked():
    assert {d: len(E.build_depth_configs(d)) for d in (3, 4, 5)} == {3: 52, 4: 60, 5: 59}
    assert sum(len(E.build_depth_configs(d)) for d in E.DEFAULT_DEPTHS) == 171


def test_s1_count_and_incumbent_present():
    for d in (3, 4, 5):
        labels = [lab for _c, lab, _w in E.build_depth_configs(d)]
        assert sum(1 for lab in labels if lab.startswith("s1_")) == 24
        assert labels.count("incumbent") == 1


@pytest.mark.parametrize("d", [3, 4, 5])
def test_vectors_valid_and_unique(d):
    cfgs = E.build_depth_configs(d)
    canon = [c for c, _lab, _w in cfgs]
    assert len(set(canon)) == len(cfgs)  # dedup: no repeats
    for _c, _lab, w in cfgs:
        assert len(w) == d
        assert all(x > 0 for x in w)
        # 6-decimal canonicalization over up to 5 components -> |sum-1| can reach ~2.5e-6.
        assert abs(sum(w) - 1.0) < 1e-5


def test_boundary_and_uniform_membership():
    # S1 reaches the near-d1 boundary (current weight 0.9) at every depth.
    for d in (3, 4, 5):
        w0s = [w[0] for _c, _lab, w in E.build_depth_configs(d)]
        assert max(w0s) == pytest.approx(0.9)
    vecs = {d: {c for c, _lab, _w in E.build_depth_configs(d)} for d in (3, 4, 5)}
    assert tuple(round(x, 6) for x in [0.25] * 4) in vecs[4]        # d4 uniform present
    assert tuple(round(x, 6) for x in [0.2] * 5) in vecs[5]         # d5 uniform present
    assert tuple(round(x, 6) for x in [1 / 3] * 3) not in vecs[3]   # d3 has no exact uniform


def test_expected_ids_no_d1_and_unique():
    ids = E.expected_ids()
    assert len(ids) == 171
    assert not any("__d1__" in i for i in ids)
    assert all(i.startswith("cp1_spatial_pool_16__grid3_") for i in ids)


# ---------------------------------------------------------------- winner weights
def test_winner_modality_weights_exact():
    mod = E._modality_weights()
    assert set(mod) == {3, 4, 5}
    for d, w in mod.items():
        assert set(w) == {"vision_0", "vision_1", "robot_state"}   # full key names
        assert abs(sum(w.values()) - 1.0) < 1e-9                    # sum to 1.0 exactly-ish
    assert mod[3]["vision_0"] == pytest.approx(0.3125)
    assert mod[4]["vision_0"] == pytest.approx(0.5625)
    assert mod[5]["robot_state"] == pytest.approx(0.625)


def test_derived_config_only_tw_differs(tmp_path):
    # calibration is the emitter source-of-truth; a hard failure (not skip) if missing.
    assert CALIB.exists(), "tracked calibration JSON missing — reproducibility broken"
    entry = json.loads(CALIB.read_text())["cp1_spatial_pool_16"]
    mod = E._modality_weights()
    d = 4
    kw = dict(builder_type=entry["builder_type"], vector_dims=entry["vector_dims"],
              preload_path="exp/common/data/cache_artifacts/libero_spatial/cp1_spatial_pool_16.pkl",
              weights=mod[d], fields_calib=entry["fields"], trajectory_depth=d)
    base = build_eval_config(trajectory_weights=[0.4, 0.3, 0.2, 0.1], **kw)
    swept = build_eval_config(trajectory_weights=[0.7, 0.1, 0.1, 0.1], **kw)

    ss_b = base["checkpoints"]["cp1"]["search_strategy"]
    ss_s = swept["checkpoints"]["cp1"]["search_strategy"]
    assert ss_b["trajectory_depth"] == d and ss_s["trajectory_depth"] == d
    assert ss_b["trajectory_weights"] != ss_s["trajectory_weights"]
    # Everything except trajectory_weights is identical.
    ss_b.pop("trajectory_weights"), ss_s.pop("trajectory_weights")
    assert base == swept

    # Both configs pass the real schema validator.
    for cfg, name in ((base, "base.yaml"), (swept, "swept.yaml")):
        cfg["checkpoints"]["cp1"]["search_strategy"]["trajectory_weights"] = [0.4, 0.3, 0.2, 0.1]
        p = tmp_path / name
        p.write_text(yaml.safe_dump(cfg, sort_keys=False))
        load_cache_config(p)


def test_emit_refuses_out_of_tree(tmp_path):
    with pytest.raises(SystemExit):
        E.emit(tmp_path / "eval", CALIB, "exp/common/data/cache_artifacts/libero_spatial")


# ---------------------------------------------------------------- shape classifier
@pytest.mark.parametrize("w,expected", [
    ([0.25, 0.25, 0.25, 0.25], "uniform"),
    ([0.2, 0.2, 0.2, 0.2, 0.2], "uniform"),
    ([0.34, 0.33, 0.33], "uniform"),           # within tol
    ([0.5, 0.3, 0.2], "decreasing"),
    ([0.9, 0.05, 0.05], "decreasing"),
    ([0.1, 0.3, 0.6], "increasing"),
    ([0.2, 0.6, 0.2], "peak"),
    ([0.4, 0.2, 0.4], "trough"),
    ([0.4, 0.1, 0.4, 0.1], "other"),           # sawtooth
    ([0.1, 0.4, 0.4, 0.1], "other"),           # plateau peak -> NOT peak (non-unique extremum)
    ([0.4, 0.1, 0.1, 0.4], "other"),           # plateau trough -> NOT trough
    ([0.1, 0.3, 0.5, 0.3, 0.1], "peak"),       # genuine strict interior peak
])
def test_classify_shape(w, expected):
    assert A.classify_shape(w, tol=0.02) == expected


def test_current_dominant_orthogonal():
    assert A.is_current_dominant([0.9, 0.05, 0.05])
    assert A.is_current_dominant([0.6, 0.2, 0.2])
    assert not A.is_current_dominant([0.5, 0.3, 0.2])
    # A monotone-decreasing shape can be either current-dominant or not (orthogonal).
    assert A.classify_shape([0.9, 0.05, 0.05]) == "decreasing"


# ---------------------------------------------------------------- journal / pairing
def _rec(yaml_id, task, ep, success, ts, phase="eval", status=None):
    status = status or ("done" if success else "failed")
    return {"task_uid": f"{yaml_id}:eval:{task}:{ep}", "yaml_id": yaml_id,
            "phase": phase, "status": status, "success": success, "ts": ts}


def test_dedup_latest_keeps_final_and_filters_nonterminal():
    recs = [
        _rec("Y", 0, 0, False, ts=1.0),          # early failed
        _rec("Y", 0, 0, True, ts=2.0),           # retried done (latest wins)
        _rec("Y", 0, 1, True, ts=1.0),
        {"task_uid": "Y:warmup:0:0", "phase": "warmup", "status": "done", "success": True, "ts": 5.0},
    ]
    final = A.dedup_latest(recs)
    assert set(final) == {"Y:eval:0:0", "Y:eval:0:1"}
    assert final["Y:eval:0:0"]["success"] is True   # latest ts kept


def test_paired_and_mcnemar():
    recs = [_rec("A", 0, 0, True, 1), _rec("A", 0, 1, False, 1), _rec("A", 0, 2, True, 1),
            _rec("B", 0, 0, False, 1), _rec("B", 0, 1, False, 1), _rec("B", 0, 2, True, 1)]
    paired = A.paired_by_yaml(recs)
    n10, n01, npair, p = A.mcnemar(paired["A"], paired["B"])
    assert (n10, n01, npair) == (1, 0, 3)     # A wins the (0,0) discordant pair
    assert p == pytest.approx(1.0)
    # all-discordant extreme
    a = {(0, i): True for i in range(3)}
    b = {(0, i): False for i in range(3)}
    assert A.mcnemar(a, b) == (3, 0, 3, pytest.approx(0.25))


def test_acceptance_gate():
    tasks, trials = [0, 1], 2
    exp = {"cfg0", "cfg1"}
    good = {"cfg0": {(t, e): True for t in tasks for e in range(trials)},
            "cfg1": {(t, e): False for t in tasks for e in range(trials)}}
    assert A.acceptance_check(good, exp, exp, tasks, trials) == []
    # missing one episode -> error
    bad = {"cfg0": {(0, 0): True, (0, 1): True, (1, 0): True}, "cfg1": good["cfg1"]}
    assert A.acceptance_check(bad, exp, exp, tasks, trials)
    # truncated batch (journal + yaml both cover only 1 of 2 expected) -> error
    assert A.acceptance_check({"cfg0": good["cfg0"]}, {"cfg0"}, exp, tasks, trials)
    # wrong episode index (episode_idx=5 outside 0..trials-1) -> error
    wrong_ep = {"cfg0": {(0, 0): True, (0, 5): True, (1, 0): True, (1, 1): True}, "cfg1": good["cfg1"]}
    assert A.acceptance_check(wrong_ep, exp, exp, tasks, trials)


def test_bootstrap_ci_reproducible_and_bracketing():
    a = {(0, i): (i % 2 == 0) for i in range(20)}
    b = {(0, i): (i % 3 == 0) for i in range(20)}
    r1 = A.paired_bootstrap_ci(a, b, n_boot=500, seed=7)
    r2 = A.paired_bootstrap_ci(a, b, n_boot=500, seed=7)
    assert r1 == r2                       # fixed seed -> reproducible
    assert r1[1] <= r1[0] <= r1[2]        # lo <= point <= hi
    assert A.paired_bootstrap_ci({}, {}, n_boot=10) == (0.0, 0.0, 0.0)


def _stub_yaml(path, w):
    path.write_text(yaml.safe_dump(
        {"checkpoints": {"cp1": {"search_strategy": {"trajectory_weights": list(w)}}}}))


def test_main_rejects_truncated_batch(tmp_path, monkeypatch):
    ids = sorted(E.expected_ids())[:2]     # 2 of the locked 171
    ydir = tmp_path / "yamls"
    ydir.mkdir()
    jpath = tmp_path / "journal.jsonl"
    with jpath.open("w") as f:
        for yid in ids:
            depth = int(yid.split("__d")[1][0])
            _stub_yaml(ydir / f"{yid}.yaml", [1.0 / depth] * depth)
            for t in range(10):
                for e in range(10):
                    f.write(json.dumps({"task_uid": f"{yid}:eval:{t}:{e}", "phase": "eval",
                                        "status": "done", "success": True, "ts": 1.0}) + "\n")
    monkeypatch.setattr("sys.argv", [
        "prog", "--journal", str(jpath), "--yaml-dir", str(ydir),
        "--out-dir", str(tmp_path / "out"), "--decision-out", str(tmp_path / "data" / "decision.json")])
    with pytest.raises(SystemExit):   # 2 yamls != locked 171 -> acceptance fails
        A.main()


def test_main_rejects_analysis_decision_path(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.argv", [
        "prog", "--journal", "x", "--yaml-dir", "y",
        "--out-dir", str(tmp_path), "--decision-out", str(tmp_path / "analysis" / "decision.json")])
    with pytest.raises(SystemExit):   # JSON under analysis/ -> refused before any file read
        A.main()
