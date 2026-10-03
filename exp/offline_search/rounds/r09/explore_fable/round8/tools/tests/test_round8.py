"""Round-8 tests: PaceWrist camera plan on recorded queries and the batch spec."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from exp.offline_search.rounds.r09.explore_fable.round5.tools.tests.test_round5 import _episode, _q
from exp.offline_search.rounds.r09.explore_fable.round8.tools import build_r8 as b8
from exp.offline_search.rounds.r09.explore_fable.round8.tools import methods as m8

ART = b8.R8 / "fits" / "r9f8_pi05_l10_500_wpace_fg.pkl"
STORE = Path(b8.STORE)


def test_kwargs_validation():
    with pytest.raises(ValueError):
        m8.PaceWrist(pace_lag=-1)
    m = m8.PaceWrist(pace_lag=1, base_spec=m8.FOLLOW_BASE, base_kwargs={}, base_fit="", wrist_fit="", stage_fit="")
    assert m.pace_lag == 1.0 and m.next_camera_mode == "full" and m.name.startswith("PW_L1")


@pytest.mark.skipif(not (ART.exists() and STORE.exists()), reason="round-8 artifact / store not available")
def test_pace_plan_and_wrist_query_on_recorded_episodes():
    with open(ART, "rb") as f:
        m = pickle.load(f)["method"]
    assert type(m).__name__ == "PaceWrist" and type(m.base).__name__ == "StageFollow" and m.base.follow_extend_blocks == 1
    qc, A, e, ev = _episode("pi05_l10_cache")
    wrist_planned = full_planned = 0
    for k in range(6):
        qc, A, e, ev = _episode("pi05_l10_cache", k)
        m.reset(ev)
        res = m.query(_q(A, e, ev, 0))
        ex = res.extras
        assert "os_pw_lag" in ex and "os_sw_reason" in ex and "os_sf_granted" in ex
        reason, lag = int(ex["os_sw_reason"]), float(ex["os_pw_lag"])
        expect = reason in (0, 2, 4) and lag <= m.pace_lag
        assert (m.next_camera_mode == "wrist_only") == expect and ex["os_sw_next_camera"] == float(expect)
        wrist_planned += expect
        full_planned += not expect
        if expect:
            m.set_camera_mode("wrist_only")
            before = {key: getattr(m.base, key) for key in ("B0T", "B1T", "tasks")}
            r2 = m.query(_q(A, e, ev, 2))
            assert len(r2.topk) and np.isfinite(r2.scores).all() and r2.action is not None
            assert all(getattr(m.base, key) is before[key] for key in before)      # metric restored after the wrist look
            m.set_camera_mode("full")
    assert wrist_planned + full_planned == 6


def test_batch_spec(tmp_path):
    if not b8.R8_MAIN.exists():
        pytest.skip("R8 arms not available")
    arms, prefit, prov = b8.build(tmp_path)
    assert len(arms) == 8 and len(prefit) == 2 and len(prov) == 6
    rows = b8._rows()
    for a in arms:
        assert "--os-judge" not in a["plugin_args"] and "--os-policy-tail" not in a["plugin_args"] and "full_model" not in a
        if a["name"].endswith("_A"):
            src = rows[prov[a["name"]]["source"]]
            assert (a["method"], a["kwargs"], a["plugin_args"], a["client_overrides"]) == (src["method"], src["kwargs"], src["plugin_args"], src["client_overrides"])
        elif a["model"] == "groot":
            src = rows[prov[a["name"]]["source"]]
            assert src["arm"].endswith("_SF1") and a["method"].endswith(":StageFollow") and a["kwargs"]["extend_blocks"] == 1
            assert a["plugin_args"] == src["plugin_args"] and "--os-request-cameras" not in a["plugin_args"]
        else:
            kw = a["kwargs"]
            assert a["method"].endswith(":PaceWrist") and kw["pace_lag"] == 1 and kw["base_spec"] == b8.FOLLOW_BASE
            assert kw["base_kwargs"]["extend_blocks"] == 1 and kw["base_kwargs"]["stage_gate"] is True and kw["base_kwargs"]["lib"] == "big"
            assert kw["wrist_fit"].endswith(f"wrist_pi05_{'l10' if a['suite'] == 'l10' else 'sp'}_500.pkl")
            assert "--os-request-cameras" in a["plugin_args"] and "--os-tokens" in a["plugin_args"]
            assert a["plugin_args"][-1] == str(Path(tmp_path) / "fits" / f"{a['name']}.pkl")
    text = json.dumps(arms)
    for forbidden in ("task_budget", "per_task", "task_ids", "task_switch", "hard_tasks", "onlynp", "Escalate", "opus"):
        assert forbidden not in text
