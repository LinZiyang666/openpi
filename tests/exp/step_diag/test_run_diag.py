"""CPU tests for exp/step_diag/run_diag.py: task ids are roster positions (never renumbered by the
subset), lane membership is enforced, the arm is folded into the identity / extras, the expected
identity set matches the planned graph one to one, and per-step rows land in per-yaml files."""

import json

import pytest

from exp.step_diag import envs as E
from exp.step_diag import run_diag as RD
from openpi.conductor import ServerEndpoint


def _strategy(tasks, lane="main", arm="plain_k2", base_seed=E.RC_FORMAL_BASE_SEED, **kw):
    return RD.StepDiagStrategy(arm_id=arm, experiment_id="sdiag_rc_v1", lane=lane, teacher="pi05", layout=1, style=1,
                               base_seed=base_seed, replan_steps=5, tasks=tasks, **kw)


def test_roster_task_ids_and_identity():
    s = _strategy([("OpenCabinet", 3), ("CloseFridge", 2)])
    assert s.run_id.startswith("sdiag-plain_k2-main-") and s.run_id.endswith("__l1s1_pi05")
    assignment = {yid: ServerEndpoint("h100", 23150) for yid in s.yaml_ids}
    graph = s.plan(sorted(s.yaml_ids), assignment)
    eps = [e for st in graph.stages.values() for e in st.episodes]
    assert len(eps) == 5
    by_task = {}
    for e in eps:
        by_task.setdefault(e.extra["task_name"], set()).add(e.task_id)
    # roster positions, not subset positions: the subset order (OpenCabinet first) must not renumber
    assert by_task == {"OpenCabinet": {RD.ROSTER.index("OpenCabinet")}, "CloseFridge": {RD.ROSTER.index("CloseFridge")}}
    assert RD.ROSTER.index("CloseFridge") < RD.ROSTER.index("OpenCabinet")
    for e in eps:
        assert e.extra["arm_id"] == "plain_k2" and e.extra["experiment_id"] == "sdiag_rc_v1" and e.extra["lane"] == "main"
        assert e.orig_init_state_idx == e.episode_idx and "pin_id" not in e.extra
    expected = s.expected_identities()
    assert {x["task_uid"] for x in expected} == {e.task_uid for e in eps}
    assert all(x["env_seed"] == E.RC_FORMAL_BASE_SEED + x["init_idx"] for x in expected)
    assert all(x["task_id"] == RD.ROSTER.index(x["task"]) for x in expected)


def test_lane_and_roster_checks():
    with pytest.raises(ValueError, match="roster"):
        _strategy([("NotATask", 1)])
    with pytest.raises(ValueError, match="lane"):
        _strategy([("PickPlaceCounterToStove", 1)], lane="main")
    with pytest.raises(ValueError, match="lane"):
        _strategy([("CloseFridge", 1)], lane="pnp")
    with pytest.raises(ValueError):  # pin table required together with pin_id
        _strategy([("PickPlaceCounterToStove", 1)], lane="pnp", pin_id="p", pinned_objects=None)


def test_pnp_lane_carries_pins_and_episode_lo_offsets():
    pins = {"PickPlaceCounterToStove": {"obj": "apple", "container": "pan"}}
    s = _strategy([("PickPlaceCounterToStove", 2)], lane="pnp", pin_id="pin-1", pinned_objects=pins,
                  episode_lo={"PickPlaceCounterToStove": 50})
    graph = s.plan(sorted(s.yaml_ids), {yid: ServerEndpoint("h", 1) for yid in s.yaml_ids})
    eps = [e for st in graph.stages.values() for e in st.episodes]
    assert [e.episode_idx for e in eps] == [50, 51]
    assert all(e.extra["pin_id"] == "pin-1" and e.extra["pinned_objects"] == pins["PickPlaceCounterToStove"] for e in eps)
    ident = s.expected_identities()
    assert [x["init_idx"] for x in ident] == [50, 51] and ident[0]["pin_id"] == "pin-1" and ident[0]["lane"] == "pnp"


def test_build_tasks_with_map_and_per_step_writer(tmp_path):
    import argparse
    args = argparse.Namespace(tasks="CloseFridge,OpenDrawer", episodes=50, episodes_map='{"OpenDrawer": 100}')
    assert RD.build_tasks(args) == [("CloseFridge", 50), ("OpenDrawer", 100)]
    write = RD.per_step_writer_for(tmp_path)
    write("y1", [{"a": 1}, {"a": 2}])
    write("y2", [{"b": 1}])
    write("y1", [{"a": 3}])
    rows = [json.loads(line) for line in (tmp_path / "per_step_y1.jsonl").read_text().splitlines()]
    assert rows == [{"a": 1}, {"a": 2}, {"a": 3}] and (tmp_path / "per_step_y2.jsonl").exists()


def test_qb_episode_counts():
    assert E.qb_episode_count("pi05", "OpenDrawer", "plain_k2") == 100
    assert E.qb_episode_count("pi05", "OpenDrawer", "warm_t0.2") == 100
    assert E.qb_episode_count("pi05", "OpenDrawer", "full") == 50
    assert E.qb_episode_count("pi05", "CloseFridge", "plain_k2") == 50
    assert E.qb_episode_count("groot", "PickPlaceCounterToStove", "warm_t0.75") == 100
    assert E.qb_episode_count("groot", "PickPlaceCounterToStove", "plain_k2") == 50
