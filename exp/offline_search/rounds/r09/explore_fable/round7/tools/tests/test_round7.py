"""Round-7 tests: the delay-free GR00T follow class on recorded store queries, and the batch spec."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from exp.offline_search.closed_loop.blind import BlindQueryView, BlindResult, LookReason
from exp.offline_search.rounds.r09.explore_fable.round5.tools.tests.test_round5 import _bq, _episode, _q
from exp.offline_search.rounds.r09.explore_fable.round7.tools import build_r7 as b7
from exp.offline_search.rounds.r09.explore_fable.round7.tools import methods as m7

ART = b7.R7 / "fits" / "r9f7_groot_l10_50_np_fgp.pkl"
STORE = Path(b7.b5.STORE)


def test_kwargs_validation():
    with pytest.raises(ValueError):
        m7.LookCostGrootPace(follow_blocks=0, max_calls=0)
    with pytest.raises(ValueError):
        m7.LookCostGrootPace(follow_blocks=1, lag_jump=0, max_calls=0)
    with pytest.raises(ValueError):
        m7.LookCostGrootPace(follow_blocks=1, wrist_gate="pace", wrist_fit="x", max_calls=0)   # no GR00T wrist path
    m = m7.LookCostGrootPace(follow_blocks=1, max_calls=0)
    assert m.lc7_lag_jump == 2.0 and m.lc_pace_lag == 1.0 and m.lc7_pending is None and m.lc_follow_blocks == 1


@pytest.mark.skipif(not (ART.exists() and STORE.exists()), reason="round-7 artifact / store not available")
def test_pace_gate_and_post_follow_check_on_recorded_episodes():
    with open(ART, "rb") as f:
        m = pickle.load(f)["method"]
    assert type(m).__name__ == "LookCostGrootPace" and m.lc_follow is not None and m.lc_wrist is None and int(m.burst) == 0
    qc, A, e, ev = _episode("groot_l10_cache")
    served = revoked = post = forced = vetoed = 0
    for k in range(min(16, len([x for x in qc.episodes if x["num_steps"] >= 8]))):
        qc, A, e, ev = _episode("groot_l10_cache", k)
        m.reset(ev)
        r0 = m.query(_q(A, e, ev, 0))
        assert r0.extras["os_lc7_post"] == 0.0 and "os_lc7_revoked" in r0.extras
        r2 = m.query(_q(A, e, ev, 2))
        plan = m.lc_follow_plan
        lag, span = m.lc_last_lag, float(getattr(m, "_noprog_span", 0) or 0)
        if r2.extras["os_lc7_revoked"]:
            revoked += 1
            assert plan is not None and plan.blocks == 0 and r2.extras["os_lc_fgrant"] == 0.0
            assert (not np.isfinite(lag)) or lag > m.lc_pace_lag or span > 0            # revoked only for the stated reasons
        elif plan is not None and plan.blocks:
            assert np.isfinite(lag) and lag <= m.lc_pace_lag and span == 0
        hv = [True, False, True]
        r3 = m.blind_step(_bq(A, e, ev, 3, True, 0, hv))
        if isinstance(r3, LookReason):
            vetoed += 1
            continue
        r4 = m.blind_step(_bq(A, e, ev, 4, True, 1, hv + [False]))
        if isinstance(r4, BlindResult) and r4.extras.get("os_sf_source", 0) > 0:
            served += 1
            assert m.lc7_pending is not None and m.lc7_pending["step"] == 4
            r5 = m.query(_q(A, e, ev, 5))                                             # the look after the follow block
            post += 1
            assert r5.extras["os_lc7_post"] == 1.0 and m.lc7_pending is None
            jump = r5.extras["os_lc7_jump"]
            if r5.extras["os_lc7_forced"]:
                forced += 1
                assert jump >= m.lc7_lag_jump and r5.extras["os_force_miss"] == 1.0 and r5.extras["os_reason"] == m7.POSTFOLLOW_REASON
                assert int(r5.extras["os_flags"]) & m7.POSTFOLLOW_BIT
            else:
                assert (not np.isfinite(jump)) or jump < m.lc7_lag_jump or r5.extras.get("os_force_miss", 0) == 1.0
        else:
            assert isinstance(r4, LookReason) or r4.extras.get("os_sf_source", 0) == 0
            assert m.lc7_pending is None
    assert served >= 1 and post == served and served + revoked + vetoed >= 3


def test_batch_spec(tmp_path):
    if not ((b7.b5.R3C / "arms.json").exists() and (b7.R4 / "arms.json").exists()):
        pytest.skip("r3c / r4 arms not available")
    arms, prefit, prov = b7.build(tmp_path)
    names = [a["name"] for a in arms]
    assert names == ["r9f7_groot_l10_50_np", "r9f7_groot_l10_50_np_fgp", "r9f7_pi05_sp_50_esc", "r9f7_pi05_sp_50_esc_wpace_fg"] and len(prefit) == 2
    g_ctrl, g_var, p_ctrl, p_var = arms
    assert "/r09_fable_r3c/fits/r9f3c_groot_l10_50_np_corr05.pkl" in g_ctrl["plugin_args"][-1]
    assert "/r09_fable_r4/fits/r9f4_pi05_spatial_50_np_corr05_esc.pkl" in p_ctrl["plugin_args"][-1]
    assert g_var["method"].endswith(":LookCostGrootPace") and g_var["kwargs"]["lag_jump"] == 2.0 and g_var["kwargs"]["pace_lag"] == 1
    assert g_var["kwargs"]["follow_blocks"] == 1 and g_var["kwargs"]["wrist_gate"] == "off" and "--os-request-cameras" not in g_var["plugin_args"]
    assert p_var["method"].endswith(":LookCostEsc") and p_var["kwargs"]["wrist_gate"] == "pace" and p_var["kwargs"]["follow_blocks"] == 1
    assert p_var["kwargs"]["stage_fit"].endswith("stages_pi05_sp_50.pkl") and p_var["kwargs"]["wrist_fit"].endswith("wrist_pi05_sp_50.pkl")
    assert "--os-request-cameras" in p_var["plugin_args"] and p_var["suite"] == "spatial" and p_var["kwargs"]["lag_threshold"] == 12
    for a in arms:
        assert a["kwargs"]["max_calls"] == 0 and a["kwargs"]["force_trigger_at"] == []
    text = json.dumps(arms)
    for forbidden in ("task_budget", "per_task", "task_ids", "task_switch", "hard_tasks", "guard_call", "opus"):
        assert forbidden not in text
