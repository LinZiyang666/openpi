"""Small exact physical tapes; no simulator/model import or global random state."""

from copy import deepcopy
import numpy as np
import pytest

from exp.offline_search.debug.tools.physical.common import Episode


@pytest.fixture
def physical_episode():
    n = 82
    obj = np.zeros((n, 2, 3))
    obj[:, 0, 2] = 0.1
    obj[0, 0, 2] = 0.2  # spawned above the table; settles by the last wait
    obj[:, 1] = [0.5, 0, 0.1]
    eef = np.tile([0.0, 0.0, 0.12], (n, 1))
    eef[2:4] = [0.3, 0, 0.2]
    obj[8:20, 0, 2] = 0.16
    eef[8:20, 2] = 0.18
    action = np.zeros((n, 7))
    action[:, 6] = -1.0
    action[4:, 6] = 1.0
    seq = np.r_[[-1, -1], np.repeat(np.arange(16), 5)].astype(np.int32)
    controls = dict(
        control_idx=np.arange(n, dtype=np.int32),
        decision_seq=seq,
        chunk_offset=np.r_[[-1, -1], np.tile(np.arange(5), 16)].astype(np.int32),
        is_settle=np.arange(n) < 2,
        action=action,
        eef_pos=eef,
        eef_quat=np.tile([0.0, 0.0, 0.0, 1.0], (n, 1)),
        obj_pos=obj,
        obj_quat=np.tile([0.0, 0.0, 0.0, 1.0], (n, 2, 1)),
        qpos=np.column_stack([eef, np.zeros((n, 9))]),
        qvel=np.zeros((n, 12)),
        predicates=np.zeros((n, 1)),
        gripper_qpos=np.tile([0.01, -0.01], (n, 1)),
        contact_off=np.arange(n + 1, dtype=np.int32),
        contact_geom=np.tile([[0, 1]], (n, 1)),
    )
    entities = dict(
        movable=[
            dict(name="object", body_id=1),
            dict(name="target", role="fixture", body_id=2),
        ],
        predicates=[["in", "object", "target"]],
        body=[
            dict(id=0, name="table", role="table"),
            dict(id=1, name="object"),
            dict(id=2, name="target"),
        ],
        geom=[
            dict(id=0, name="table_collision", body_id=0),
            dict(id=1, name="object_collision", body_id=1),
        ],
    )
    meta = dict(
        arm="A",
        episode_key="tape_a1",
        task_id=0,
        init=0,
        attempt=1,
        suite="libero_10",
        env_seed=7,
        n_controls=n,
        success=False,
        entities=entities,
        control_dt=0.05,
        reset_state_sha256="fixed-reset",
    )
    decisions = [
        dict(
            decision_seq=i,
            decision_id="tape_a1:1:%d" % i,
            control_idx_start=2 + 5 * i,
            n_applied=5,
            src="cache" if i % 2 == 0 else "cache_tail",
            vision=i % 2 == 0,
            anchor_decision_id="tape_a1:1:%d" % (i - i % 2),
            rows=[0],
            weights=[1.0],
            chunk_offset=5 * (i % 2),
            blind_age_controls=5 * (i % 2),
        )
        for i in range(16)
    ]
    return Episode(meta, controls, decisions, dict(model="groot", exec_steps=5))


@pytest.fixture
def clone():
    return deepcopy
