"""``executable_warm_tiers`` vs ``required_warm_timesteps`` (plan §4.3, §12-C).

Six judge types: the executable set is what the trace computes as warm
variants, the required set is what the library must carry. They differ for
online_rit (successor snapshots are required, never executed), for the
dispatch surface / CRD (the tier lives in the artifact, not the yaml) and for
the routers (no warm tier at all).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from openpi.cache.config import (
    CacheConfig,
    CheckpointConfig,
    ComposerConfig,
    JudgeConfig,
    required_warm_timesteps,
)
from openpi.cache.trace.runtime import executable_warm_tiers
from openpi.cache.types import PI05_V1, CheckpointID, groot_n15_schedule


def _cfg(judge: JudgeConfig, schedule_id=None) -> CacheConfig:
    return CacheConfig(
        checkpoints={"cp1": CheckpointConfig(judge=judge)},
        denoise_schedule=schedule_id,
    )


def test_threshold_tiers_match_yaml():
    j = JudgeConfig(type="threshold", warm_tiers=[{"threshold": 0.9, "start_t": 0.7}, {"threshold": 0.8, "start_t": 0.3}])
    idx = executable_warm_tiers(_cfg(j), checkpoint=CheckpointID.CP1, assembled_judge=None, schedule=PI05_V1)
    assert idx == (PI05_V1.snapshot_index(0.7), PI05_V1.snapshot_index(0.3)) == (3, 7)
    assert required_warm_timesteps(_cfg(j)) == frozenset({0.7, 0.3})


def test_always_warm_start_and_composite():
    j = JudgeConfig(type="always_warm_start", start_t=0.5)
    assert executable_warm_tiers(_cfg(j), checkpoint=CheckpointID.CP1, assembled_judge=None, schedule=PI05_V1) == (5,)
    j = JudgeConfig(type="composite", composer=ComposerConfig(warm_start_t=0.7, warm_fallback_start_t=0.3))
    assert executable_warm_tiers(_cfg(j), checkpoint=CheckpointID.CP1, assembled_judge=None, schedule=PI05_V1) == (3, 7)


def test_surface_and_crd_read_the_assembled_artifact():
    j = JudgeConfig(type="dispatch_surface", surface_artifact_path="x.npz")
    judge = SimpleNamespace(artifact=SimpleNamespace(start_t_ws=0.3))
    assert executable_warm_tiers(_cfg(j), checkpoint=CheckpointID.CP1, assembled_judge=judge, schedule=PI05_V1) == (7,)
    # A dump wrapper is unwrapped through ``_inner``.
    wrapped = SimpleNamespace(_inner=judge)
    assert executable_warm_tiers(_cfg(j), checkpoint=CheckpointID.CP1, assembled_judge=wrapped, schedule=PI05_V1) == (7,)
    # required_warm_timesteps deliberately skips the surface (library check uses the artifact contract).
    assert required_warm_timesteps(_cfg(j)) == frozenset()
    with pytest.raises(ValueError, match="start_t_ws"):
        executable_warm_tiers(_cfg(j), checkpoint=CheckpointID.CP1, assembled_judge=SimpleNamespace(), schedule=PI05_V1)


def test_online_rit_executes_tiers_but_requires_successors():
    sched = groot_n15_schedule(8)
    j = JudgeConfig(type="online_rit", tiers=[0.875, 0.75, 0.5])
    cfg = _cfg(j, schedule_id=sched.schedule_id)
    execu = executable_warm_tiers(cfg, checkpoint=CheckpointID.CP1, assembled_judge=None, schedule=sched)
    assert execu == (sched.snapshot_index(0.5), sched.snapshot_index(0.75), sched.snapshot_index(0.875))
    req = required_warm_timesteps(cfg)
    assert {0.875, 0.75, 0.5} <= req
    assert 0.625 in req  # successor of 0.5 is required, never executed
    assert sched.snapshot_index(0.625) not in execu


def test_routers_have_no_warm_tiers_and_unknown_judge_raises():
    for jtype in ("mlp_router", "risk_router"):
        j = JudgeConfig(type=jtype)
        assert executable_warm_tiers(_cfg(j), checkpoint=CheckpointID.CP1, assembled_judge=None, schedule=PI05_V1) == ()
    j = JudgeConfig(type="something_new")
    with pytest.raises(ValueError, match="does not know"):
        executable_warm_tiers(_cfg(j), checkpoint=CheckpointID.CP1, assembled_judge=None, schedule=PI05_V1)


def test_no_config_or_checkpoint_is_empty():
    assert executable_warm_tiers(None, checkpoint=CheckpointID.CP1, assembled_judge=None, schedule=PI05_V1) == ()
    j = JudgeConfig(type="threshold", warm_tiers=[{"threshold": 0.9, "start_t": 0.7}])
    assert executable_warm_tiers(_cfg(j), checkpoint=None, assembled_judge=None, schedule=PI05_V1) == ()
    assert executable_warm_tiers(_cfg(j), checkpoint=CheckpointID.CP3, assembled_judge=None, schedule=PI05_V1) == ()


def test_tier_outside_schedule_raises():
    j = JudgeConfig(type="threshold", warm_tiers=[{"threshold": 0.9, "start_t": 0.55}])
    with pytest.raises(ValueError, match="not a snapshot"):
        executable_warm_tiers(_cfg(j), checkpoint=CheckpointID.CP1, assembled_judge=None, schedule=PI05_V1)
