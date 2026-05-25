"""Schema/semantics tests for conductor core data structures (M1)."""

from __future__ import annotations

import pytest

from openpi.conductor import task as T  # noqa: N812


def _ep(yaml_id="y", phase="eval", task_id=0, episode_idx=0):
    return T.EpisodeTask(
        task_uid=T.make_task_uid(yaml_id, phase, task_id, episode_idx),
        yaml_id=yaml_id,
        phase=phase,
        experiment="libero_spatial",
        task_id=task_id,
        episode_idx=episode_idx,
        orig_init_state_idx=episode_idx,
        server_host="h",
        server_port=8000,
        bundle_id="b",
    )


# ------------------------------------------------------------------
# identity
# ------------------------------------------------------------------


def test_make_task_uid_deterministic():
    a = T.make_task_uid("y", "eval", 3, 7)
    b = T.make_task_uid("y", "eval", 3, 7)
    assert a == b == "y:eval:3:7"
    assert T.make_task_uid("y", "warmup", 3, 7) != a


def test_episode_task_server_property():
    e = _ep()
    assert e.server == T.ServerEndpoint("h", 8000)
    assert e.server.key == "h:8000"


# ------------------------------------------------------------------
# CalibrationArtifact validation (G1R2 Item 1 — cleanup_id shape)
# ------------------------------------------------------------------


def test_calibration_warmup_requires_cleanup_id():
    with pytest.raises(ValueError, match="requires"):
        T.CalibrationArtifact(calib_id="c", source="warmup_stage", warmup_stage_id="s")
    with pytest.raises(ValueError, match="requires"):
        T.CalibrationArtifact(calib_id="c", source="warmup_stage", cleanup_id="x")


def test_calibration_historical_requires_path():
    with pytest.raises(ValueError, match="requires"):
        T.CalibrationArtifact(calib_id="c", source="historical_file")
    art = T.CalibrationArtifact(calib_id="c", source="historical_file", historical_path="/p.jsonl")
    assert art.dump_name is None  # no warmup dump to clean


def test_calibration_dump_name_shape():
    art = T.CalibrationArtifact(calib_id="c", source="warmup_stage", warmup_stage_id="s", cleanup_id="myeval")
    assert art.dump_name == "myeval__warmup"


# ------------------------------------------------------------------
# TaskGraph construction + validation
# ------------------------------------------------------------------


def _stage(sid, phase="eval", **kw):
    return T.Stage(stage_id=sid, yaml_id="y", phase=phase, server=T.ServerEndpoint("h", 8000), **kw)


def test_duplicate_stage_rejected():
    g = T.TaskGraph()
    g.add_stage(_stage("s1"))
    with pytest.raises(ValueError, match="duplicate"):
        g.add_stage(_stage("s1"))


def test_dependency_unknown_stage_rejected():
    g = T.TaskGraph()
    g.add_stage(_stage("s1"))
    with pytest.raises(ValueError, match="unknown stage"):
        g.add_dependency("s1", "missing")


def test_upstreams_downstreams():
    g = T.TaskGraph()
    g.add_stage(_stage("w", phase="warmup"))
    g.add_stage(_stage("e"))
    g.add_dependency("w", "e")
    assert g.upstreams("e") == ["w"]
    assert g.downstreams("w") == ["e"]


def test_validate_dangling_calib_rejected():
    g = T.TaskGraph()
    g.add_stage(_stage("e", consumes_calib_id="nope"))
    with pytest.raises(ValueError, match="unknown calib_id"):
        g.validate()


def test_validate_cycle_detected():
    g = T.TaskGraph()
    g.add_stage(_stage("a"))
    g.add_stage(_stage("b"))
    g.add_dependency("a", "b")
    g.add_dependency("b", "a")
    with pytest.raises(ValueError, match="cycle"):
        g.validate()


def test_validate_ok_warmup_eval_chain():
    g = T.TaskGraph()
    g.add_calibration(T.CalibrationArtifact(calib_id="c", source="warmup_stage", warmup_stage_id="w", cleanup_id="y"))
    g.add_stage(_stage("w", phase="warmup", produces_calib_id="c"))
    g.add_stage(_stage("e", consumes_calib_id="c"))
    g.add_dependency("w", "e")
    g.validate()  # no raise
