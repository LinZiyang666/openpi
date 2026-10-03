"""Round-6 batch spec: one combined pace-wrist + gated-follow arm plus the plain stack control."""
import json

import pytest

from exp.offline_search.rounds.r09.explore_fable.round5.tools import build_r5 as b5
from exp.offline_search.rounds.r09.explore_fable.round6.tools import build_r6 as b6


def test_spec(tmp_path):
    if not (b5.R3C / "arms.json").exists():
        pytest.skip("r3c arms not available")
    arms, prefit, prov = b6.build(tmp_path)
    assert [a["name"] for a in arms] == ["r9f6_pi05_l10_50_esc", "r9f6_pi05_l10_50_esc_wpace_fg"] and len(prefit) == 1
    ctrl, arm = arms
    assert "/r09_fable_r3c/fits/r9f3c_pi05_l10_50_np_corr05_esc.pkl" in ctrl["plugin_args"][-1] and "--os-request-cameras" not in ctrl["plugin_args"]
    kw = arm["kwargs"]
    assert kw["wrist_gate"] == "pace" and kw["pace_lag"] == 1 and kw["follow_blocks"] == 1 and kw["follow_stage_gate"] is True
    assert kw["max_calls"] == 0 and kw["lag_threshold"] == 12 and kw["deadline"] == 80          # stack untouched
    assert kw["wrist_fit"].endswith("wrist_pi05_l10_50.pkl") and kw["stage_fit"].endswith("stages_pi05_l10_50.pkl")
    assert "--os-request-cameras" in arm["plugin_args"] and "--os-tokens" in arm["plugin_args"] and arm["method"].endswith(":LookCostEsc")
    text = json.dumps(arms)
    for forbidden in ("task_budget", "per_task", "task_ids", "task_switch", "hard_tasks", "guard_call", "opus"):
        assert forbidden not in text
