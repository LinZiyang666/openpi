"""Round-5 look-cost stack tests: kwargs validation, wrist-look metric swap + corrector skip on a recorded query,
gated-follow extension on recorded blind steps, and the batch spec.  Artifact tests skip when the frozen fits or the
store are absent."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from exp.offline_search.closed_loop.blind import BlindQueryView, BlindResult, LookReason
from exp.offline_search.rounds.r09.explore_fable.round5.tools import build_r5 as b
from exp.offline_search.rounds.r09.explore_fable.round5.tools import methods as m5

R5 = b.R5
STORE = Path(b.STORE)
WRIST_ART = R5 / "fits" / "r9f5_pi05_l10_50_esc_weasy.pkl"
FOLLOW_ART = {"pi05": R5 / "fits" / "r9f5_pi05_l10_50_esc_fg.pkl", "groot": R5 / "fits" / "r9f5_groot_l10_50_np_fg.pkl"}


def _load(path):
    with open(path, "rb") as f:
        return pickle.load(f)["method"]


def _episode(cell, k=0):
    from exp.offline_search.harness import api, store
    qc = store.QueryCell(STORE, cell)
    A = api.QueryArrays(qc)
    eps = [e for e in qc.episodes if e["num_steps"] >= 8]
    e = eps[k]
    ev = api.EpisodeView(e["uid"], e["task"], e["task_id"], e["init"], qc.episodes.index(e), 7)
    return qc, A, e, ev


def _q(A, e, ev, step):
    from exp.offline_search.harness import api
    start = int(e["start"])
    return api.QueryView(A, start + step, start, step, int(e["task_id"]), ev, None)


def _bq(A, e, ev, step, prev_hit, blind_age, has_vision):
    start = int(e["start"])
    row = start + step
    return BlindQueryView(step=step, task_id=int(e["task_id"]), episode=ev, rs=np.asarray(A.rs[row], np.float32),
                          raw_state=np.asarray(A.raw_state[row], np.float32), prev_hit=prev_hit,
                          prev_a_exec=np.asarray(A.a_exec[row - 1], np.float32), hist_a_exec=np.asarray(A.a_exec[start:row], np.float32),
                          hist_hit=np.asarray(A.exec_hit[start:row]).astype(int), hist_rs=np.asarray(A.rs[start:row], np.float32),
                          hist_has_vision=np.asarray(has_vision, bool), blind_age=blind_age)


def test_kwargs_validation():
    with pytest.raises(ValueError):
        m5.LookCostEsc(wrist_gate="bogus")
    with pytest.raises(ValueError):
        m5.LookCostEsc(wrist_gate="all")                      # wrist looks need the wrist artifact
    with pytest.raises(ValueError):
        m5.LookCostEsc(follow_blocks=2)
    with pytest.raises(ValueError):
        m5.LookCostGroot(wrist_gate="easy", wrist_fit="x")    # no one-camera GR00T path
    m = m5.LookCostEsc(wrist_gate="pace", pace_lag=1, wrist_fit="x", follow_blocks=1, follow_stage_gate=False, max_calls=0)
    assert m.lc_wrist_gate == "pace" and m.lc_follow_blocks == 1 and m.next_camera_mode == "full" and m.gm_max_calls == 0
    assert set(m5._LookCost._LC_OWN) <= set(vars(m))          # every snapshot attribute exists before fit


@pytest.mark.skipif(not (WRIST_ART.exists() and STORE.exists()), reason="round-5 artifacts / store not available")
def test_wrist_look_swaps_metric_and_skips_corrector():
    m = _load(WRIST_ART)
    assert type(m.base).__name__ == "CorrectedCacheJW" and m.lc_wrist is not None and int(m.burst) == 0
    qc, A, e, ev = _episode("pi05_l10_cache")
    m.reset(ev)
    before = {k: getattr(m.base, k) for k in m5.METRIC_FIELDS}
    full = m.query(_q(A, e, ev, 0))
    a = m.base._anchor
    plain = np.tensordot(a["weights"], np.asarray(m.base.act)[a["rows"], :, :7], 1)
    assert not np.allclose(full.action[:10, :6], plain[:10, :6])          # correction active on a full look
    assert full.extras["os_lc_wrist_look"] == 0.0 and "os_lc_cam" in full.extras and "os_lc_reason" in full.extras
    with pytest.raises(ValueError):
        m.set_camera_mode("sideways")
    m.set_camera_mode("wrist_only")
    with pytest.raises(Exception):
        m.query(_q(A, e, ev, 0))                                          # episode start must encode all cameras
    m.reset(ev)
    m.query(_q(A, e, ev, 0))
    m.set_camera_mode("wrist_only")
    res = m.query(_q(A, e, ev, 2))
    a = m.base._anchor
    plain = np.tensordot(a["weights"], np.asarray(m.base.act)[a["rows"], :, :7], 1)
    assert np.allclose(res.action[:, :7], plain)                          # wrist look: kernel chunk, no correction
    assert res.extras["os_lc_wrist_look"] == 1.0 and len(res.topk) and np.isfinite(res.scores).all()
    assert all(getattr(m.base, k) is before[k] for k in m5.METRIC_FIELDS)  # metric restored after the look
    assert m.base.wrist_pass is False
    assert m.lc_wrist.B1T.shape[0] + m.lc_state_width == 72


@pytest.mark.skipif(not (all(p.exists() for p in FOLLOW_ART.values()) and STORE.exists()), reason="round-5 artifacts / store not available")
@pytest.mark.parametrize("model", ["pi05", "groot"])
def test_gated_follow_serves_one_successor_block_then_looks(model):
    m = _load(FOLLOW_ART[model])
    assert m.lc_follow is not None and m.lc_wrist is None and m.lc_reference_blocks == 2
    qc, A, e, ev = _episode(f"{model}_l10_cache")
    served = looked = vetoed = 0
    for k in range(min(12, len([x for x in qc.episodes if x["num_steps"] >= 8]))):
        qc, A, e, ev = _episode(f"{model}_l10_cache", k)
        m.reset(ev)
        for s in (0, 2):
            m.query(_q(A, e, ev, s))
        a = m.base._anchor
        assert a["step"] == 2 and m.lc_follow_plan is not None
        hv = [True, False, True]
        r1 = m.blind_step(_bq(A, e, ev, 3, True, 0, hv))
        if isinstance(r1, LookReason):
            vetoed += 1                                                      # the no-progress blind veto came first
            assert r1.code == 8
            continue
        assert isinstance(r1, BlindResult) and "os_sf_source" not in r1.extras   # the ordinary blind block
        r2 = m.blind_step(_bq(A, e, ev, 4, True, 1, hv + [False]))
        if isinstance(r2, BlindResult):
            served += 1
            assert r2.extras["os_sf_source"] > 0 and m.lc_follow_plan.blocks == 1
            assert a["last_step"] == 4 and r2.action.shape[0] == m.base.H and np.isfinite(r2.action).all()
            r3 = m.blind_step(_bq(A, e, ev, 5, True, 2, hv + [False, False]))
            assert isinstance(r3, LookReason) and r3.code == 1              # capped at one extra block
        else:
            looked += 1
            assert isinstance(r2, LookReason) and r2.code in (1, 8, 11)
    assert served >= 1 and served + looked + vetoed >= 3


def test_batch_spec_controls_and_variants(tmp_path):
    if not (b.R3C / "arms.json").exists():
        pytest.skip("r3c arms not available")
    arms, prefit, prov = b.build(tmp_path)
    names = [a["name"] for a in arms]
    assert len(arms) == 8 and len(prefit) == 6 and len(set(names)) == 8
    for a in arms:
        if a["name"] in ("r9f5_pi05_l10_50_esc", "r9f5_groot_l10_50_np"):
            assert "/r09_fable_r3c/fits/" in a["plugin_args"][-1] and "--os-request-cameras" not in a["plugin_args"]
            assert a["kwargs"]["max_calls"] == 0 and a["kwargs"]["force_trigger_at"] == []
        else:
            kw = a["kwargs"]
            wrist = kw["wrist_gate"] != "off"
            assert ("--os-request-cameras" in a["plugin_args"]) == wrist and ("--os-tokens" in a["plugin_args"]) == wrist
            assert (kw["wrist_fit"] != "") == wrist and a["method"].endswith(b.CLASS[a["model"]])
            assert a["model"] == "pi05" or not wrist
            assert kw["max_calls"] == 0 and kw["hold_decisions"] == 2 and kw["burst"] == 2
            assert a["plugin_args"][-1] == str(Path(tmp_path) / "fits" / f"{a['name']}.pkl")
    text = json.dumps(arms)
    for forbidden in ("task_budget", "per_task", "task_ids", "task_switch", "hard_tasks"):
        assert forbidden not in text
    manifest = json.loads(b.MANIFEST_SRC.read_text())
    assert {(int(t), int(i)) for t, i in manifest} == {(t, i) for t in range(10) for i in range(20, 30)}
