"""Unit tests for ``examples/libero/main.py`` Layer C additions (plan §4 +
§19.B6 + §20.R1).

We cannot import ``main`` directly in CI because it depends on the LIBERO sim
package (MuJoCo, bddl files) which is not available in the test venv. We
stub the four ``libero.*`` submodules before import so the helper functions
under test (``_load_episode_filter`` / ``_flush_trajectory_h5`` /
``_write_episode_results_json`` / ``_run_episode``) can be exercised in
isolation from the sim stack.

Focus areas locked by this suite:
- §19.B6 filter JSON → ``(filter_pairs, orig_map)`` lookup.
- §20.R1.2 HDF5 schema (subdir path, attrs, per-cycle ``executed_action_count``).
- §20.R1.1 mid-chunk finalise: given a mock env that fires ``done=True`` on
  the 3rd step of a chunk, the written ``executed_actions`` row must have
  ``shape[0] == 3`` (this is the exact assertion required by plan §21.T4 /
  §24.T4 for Layer C).
- per-episode results JSON default-path convention.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import h5py
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Stub out libero-related imports so ``main.py`` loads cleanly.
# ---------------------------------------------------------------------------


def _stub_libero_modules() -> None:
    for name in (
        "libero",
        "libero.libero",
        "libero.libero.benchmark",
        "libero.libero.envs",
    ):
        sys.modules.setdefault(name, MagicMock())


_stub_libero_modules()
# Ensure ``examples/libero`` is importable as a top-level ``main`` module.
_LIBERO_DIR = Path(__file__).resolve().parents[2] / "examples" / "libero"
if str(_LIBERO_DIR) not in sys.path:
    sys.path.insert(0, str(_LIBERO_DIR))

libero_main = importlib.import_module("main")  # noqa: E402


# ---------------------------------------------------------------------------
# _compute_global_episode_id: serial / concurrent must agree (plan §4 / §19.B6)
# ---------------------------------------------------------------------------


def test_compute_global_episode_id_is_deterministic_from_task_and_subset() -> None:
    """Plan §4 / §19.B6: the ID must depend only on ``(task_id, episode_idx,
    num_trials_per_task)`` — no execution-order leak. ``_eval_serial`` used
    to increment a counter, which broke this under ``--task_ids`` or
    ``--episode_filter`` subsets because the counter drifted away from the
    (task, idx) formula that ``_eval_concurrent`` already used."""
    fn = libero_main._compute_global_episode_id
    # Reference formula: task_id * N + idx.
    for task_id in (0, 3, 9):
        for idx in (0, 1, 49):
            assert fn(task_id, idx, 50) == task_id * 50 + idx
    # Subset reproduces same IDs as full sweep (no counter drift).
    full = [fn(t, i, 50) for t in (0, 1, 2, 3) for i in range(5)]
    subset = [fn(t, i, 50) for t in (1, 3) for i in (0, 2, 4)]
    # Every subset id must appear in the full list with the same value.
    for t, i in ((1, 0), (1, 2), (1, 4), (3, 0), (3, 2), (3, 4)):
        assert fn(t, i, 50) == t * 50 + i


def test_eval_paths_use_shared_episode_id_helper_source() -> None:
    """Source-level lock (plan §22-style): both evaluation paths must call
    ``_compute_global_episode_id`` rather than inline the formula. A past
    regression (serial path using a monotonic counter) is exactly what this
    lock prevents."""
    src = Path(libero_main.__file__).read_text()
    # Exactly two call sites expected: one in ``_eval_serial``, one in
    # ``_eval_concurrent``. A drift back to ``global_episode_id += 1`` or an
    # inline ``task_id * args.num_trials_per_task + episode_idx`` would break
    # this assertion.
    assert src.count("_compute_global_episode_id(") >= 3  # def + 2 callsites
    assert "global_episode_id += 1" not in src, (
        "serial path must NOT increment a monotonic counter (plan §4 / §19.B6)"
    )


# ---------------------------------------------------------------------------
# _load_episode_filter
# ---------------------------------------------------------------------------


def test_load_episode_filter_empty_path_is_noop() -> None:
    pairs, orig_map = libero_main._load_episode_filter("")
    assert pairs is None
    assert orig_map == {}


def test_load_episode_filter_parses_step1b_schema(tmp_path: Path) -> None:
    filter_path = tmp_path / "step1b_filter.json"
    filter_path.write_text(
        json.dumps(
            [
                {"task_id": 3, "orig_init_state_idx": 15, "subset_init_state_idx": 1},
                {"task_id": 3, "orig_init_state_idx": 28, "subset_init_state_idx": 2},
                {"task_id": 5, "orig_init_state_idx": 2, "subset_init_state_idx": 0},
            ]
        )
    )

    pairs, orig_map = libero_main._load_episode_filter(str(filter_path))

    assert pairs == {(3, 1), (3, 2), (5, 0)}
    assert orig_map[(3, 1)] == 15
    assert orig_map[(3, 2)] == 28
    assert orig_map[(5, 0)] == 2


# ---------------------------------------------------------------------------
# _count_filtered_episodes (G2 deferred should-fix: concurrent progress bar
# total must respect episode_filter so tqdm does not promise more episodes
# than will actually run).
# ---------------------------------------------------------------------------


def test_count_filtered_episodes_no_filter_returns_full_product() -> None:
    assert libero_main._count_filtered_episodes(None, [0, 1, 2], 5) == 15
    assert libero_main._count_filtered_episodes(None, [], 5) == 0


def test_count_filtered_episodes_counts_only_pairs_inside_task_list() -> None:
    # filter includes task_id=7 which is NOT in task_id_list → must be
    # excluded (mirrors the worker's ``continue`` guard predicate).
    pairs = {(3, 0), (3, 1), (5, 0), (7, 0)}
    assert libero_main._count_filtered_episodes(pairs, [3, 5], 50) == 3
    assert libero_main._count_filtered_episodes(pairs, [3], 50) == 2
    assert libero_main._count_filtered_episodes(pairs, [], 50) == 0


def test_count_filtered_episodes_ignores_num_trials_when_filter_active() -> None:
    # With a filter, num_trials_per_task is irrelevant — each filter entry
    # maps to exactly one episode run.
    pairs = {(0, 0), (0, 1)}
    assert libero_main._count_filtered_episodes(pairs, [0], 1) == 2
    assert libero_main._count_filtered_episodes(pairs, [0], 1000) == 2


# ---------------------------------------------------------------------------
# _flush_trajectory_h5 (plan §20.R1.2 schema)
# ---------------------------------------------------------------------------


def _make_step(executed_k: int = 5) -> dict:
    return {
        "sim_state": np.zeros(71, dtype=np.float64),
        "env_timestep": 12,
        "env_cur_time": 0.25,
        "agentview_image": np.zeros((224, 224, 3), dtype=np.uint8),
        "eye_in_hand_image": np.zeros((224, 224, 3), dtype=np.uint8),
        "robot_state": np.zeros(8, dtype=np.float64),
        "env_action_chunk": np.zeros((50, 7), dtype=np.float32),
        "executed_actions": np.zeros((executed_k, 7), dtype=np.float32),
    }


def test_flush_trajectory_empty_buffer_returns_none(tmp_path: Path) -> None:
    out = libero_main._flush_trajectory_h5(
        str(tmp_path),
        task_id=0,
        subset_init_state_idx=0,
        task_name="noop",
        orig_init_state_idx=0,
        episode_id=0,
        seed=7,
        success=False,
        replan_steps=5,
        num_steps_wait=10,
        final_env_timestep=0,
        traj_buffer=[],
    )
    assert out is None
    assert list(tmp_path.iterdir()) == []  # Nothing written.


def test_flush_trajectory_writes_subdir_schema(tmp_path: Path) -> None:
    traj = [_make_step(5), _make_step(3)]  # K varies between cycles
    out_path = libero_main._flush_trajectory_h5(
        str(tmp_path),
        task_id=3,
        subset_init_state_idx=1,
        task_name="pick apple",
        orig_init_state_idx=15,
        episode_id=42,
        seed=7,
        success=True,
        replan_steps=5,
        num_steps_wait=10,
        final_env_timestep=37,
        traj_buffer=traj,
    )
    expected = tmp_path / "task_3" / "episode_1.h5"
    assert out_path == expected
    assert expected.exists()

    with h5py.File(expected, "r") as f:
        # Plan §20.R1.2 attrs lock.
        assert f.attrs["task_name"] == "pick apple"
        assert int(f.attrs["task_id"]) == 3
        assert int(f.attrs["init_state_idx"]) == 1
        assert int(f.attrs["orig_init_state_idx"]) == 15
        assert int(f.attrs["episode_id"]) == 42
        assert int(f.attrs["seed"]) == 7
        assert int(f.attrs["num_cycles"]) == 2
        assert int(f.attrs["num_steps"]) == 8  # 5 + 3
        assert bool(f.attrs["success"]) is True
        assert int(f.attrs["replan_steps"]) == 5
        # G2 §26 MF-3: env-timestep budget anchor attrs (Layer F spawn needs
        # these to compute ``budget = final - anchor`` on the env-step axis).
        assert int(f.attrs["num_steps_wait"]) == 10
        assert int(f.attrs["final_env_timestep"]) == 37
        # executed_action_count locked per-cycle (plan §20.R1.2).
        assert int(f["step_0000"].attrs["executed_action_count"]) == 5
        assert int(f["step_0001"].attrs["executed_action_count"]) == 3
        # Schema coverage: every required dataset present.
        for group_name in ("step_0000", "step_0001"):
            g = f[group_name]
            for key in (
                "sim_state",
                "env_timestep",
                "env_cur_time",
                "agentview_image",
                "eye_in_hand_image",
                "robot_state",
                "env_action_chunk",
                "executed_actions",
            ):
                assert key in g, f"{group_name}/{key} missing"


# ---------------------------------------------------------------------------
# _write_episode_results_json
# ---------------------------------------------------------------------------


def test_write_episode_results_default_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    rows = [{"task_id": 0, "init_state_idx": 0, "success": True}]
    out = libero_main._write_episode_results_json("", "libero_spatial", rows)
    assert out == Path("libero_spatial_episode_results.json")
    assert json.loads(out.read_text()) == rows


def test_write_episode_results_explicit_path_creates_parent(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "out.json"
    rows = [{"task_id": 3, "init_state_idx": 1, "success": False}]
    out = libero_main._write_episode_results_json(str(target), "libero_spatial", rows)
    assert out == target
    assert target.exists()
    assert json.loads(target.read_text()) == rows


# ---------------------------------------------------------------------------
# _run_episode with mock env/client
# ---------------------------------------------------------------------------


class _InnerEnv:
    """Mimics ``OffScreenRenderEnv.env`` (robosuite env) — holds
    ``timestep`` and ``cur_time`` because ``OffScreenRenderEnv`` itself
    does not expose them; verified against libero_sim conda env."""

    timestep: int = 0
    cur_time: float = 0.0


class _FakeEnv:
    """Smallest stand-in for ``OffScreenRenderEnv`` covering ``_run_episode``.

    The scripted ``done_at_step`` fires ``done=True`` on the n-th ``env.step``
    call after ``num_steps_wait``, so we can verify §20.R1.1 mid-chunk finalise
    semantics without touching MuJoCo.
    """

    def __init__(self, *, done_at_step: int | None = None) -> None:
        self._done_at_step = done_at_step
        self._step_count = 0  # counts post-wait env.step() calls
        self.env = _InnerEnv()

    def _obs(self) -> dict:
        return {
            "agentview_image": np.zeros((256, 256, 3), dtype=np.uint8),
            "robot0_eye_in_hand_image": np.zeros((256, 256, 3), dtype=np.uint8),
            "robot0_eef_pos": np.zeros(3, dtype=np.float32),
            "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            "robot0_gripper_qpos": np.zeros(2, dtype=np.float32),
        }

    def reset(self) -> dict:
        self._step_count = 0
        self.env.timestep = 0
        self.env.cur_time = 0.0
        return self._obs()

    def set_init_state(self, initial_state):  # noqa: ARG002
        return self._obs()

    def step(self, action):  # noqa: ARG002
        self.env.timestep += 1
        self.env.cur_time += 0.02
        # Decide whether this step fires done. ``done_at_step`` counts the
        # post-``num_steps_wait`` step index (0-indexed).
        is_dummy = False
        done = False
        # Track dummy wait vs. real step by looking at the action length? No —
        # the test harness picks done_at_step conservatively so we simply
        # start counting once we've seen at least one post-wait step.
        self._step_count += 1
        if self._done_at_step is not None and self._step_count == self._done_at_step:
            done = True
        return self._obs(), 0.0, done, {}

    def get_sim_state(self) -> np.ndarray:
        return np.zeros(71, dtype=np.float64)

    def close(self) -> None:
        pass


class _FakeClient:
    """Mock client whose ``infer`` returns a canned action chunk."""

    def __init__(self, chunk: np.ndarray) -> None:
        self._chunk = chunk
        self.infer_calls = 0

    def infer(self, element: dict) -> dict:  # noqa: ARG002
        self.infer_calls += 1
        return {"actions": self._chunk.copy()}


class _Args:
    """Minimal ``Args`` shim for ``_run_episode``; only used attributes."""

    num_steps_wait = 0  # skip the warm-up window in tests
    replan_steps = 5
    resize_size = 224
    display = False
    save_trajectory = True


def test_run_episode_records_cycles_on_every_infer() -> None:
    """Normal path: every replan boundary must push a cycle with full chunk
    and the previous cycle's executed_actions must equal ``replan_steps``
    (5 by default)."""
    # 10 post-wait steps, no done → 2 full replan cycles expected.
    env = _FakeEnv(done_at_step=None)
    chunk = np.ones((50, 7), dtype=np.float32)
    client = _FakeClient(chunk)
    args = _Args()

    done, _images, _ts, traj, final_env_timestep = libero_main._run_episode(
        env, client, initial_state=None,
        task_description="pick apple",
        args=args,
        max_steps=10,
        record_video=False,
    )

    assert done is False
    assert traj is not None
    # 10 steps / replan_steps=5 ⇒ two completed cycles.
    assert len(traj) == 2
    for cycle in traj:
        assert cycle["executed_actions"].shape == (5, 7)
    # env_action_chunk mirrors full policy output per cycle.
    for cycle in traj:
        assert cycle["env_action_chunk"].shape == (50, 7)
    # G2 §26 MF-3: final_env_timestep must equal the inner env's timestep
    # counter after the loop — here, num_steps_wait=0 + 10 real steps = 10.
    assert final_env_timestep == 10


def test_run_episode_mid_chunk_done_finalises_partial_executed(
    tmp_path: Path,
) -> None:
    """§20.R1 must-have: when ``done=True`` fires on the 3rd step of a chunk,
    the last cycle's ``executed_actions`` is finalised with ``shape[0] == 3``
    and HDF5 flush succeeds. This is the exact assertion in plan §21.T4.
    """
    # done on the 3rd post-wait step.
    env = _FakeEnv(done_at_step=3)
    chunk = np.full((50, 7), 0.5, dtype=np.float32)
    client = _FakeClient(chunk)
    args = _Args()

    done, _, _, traj, final_env_timestep = libero_main._run_episode(
        env, client, initial_state=None,
        task_description="open drawer",
        args=args,
        max_steps=20,
        record_video=False,
    )

    assert done is True
    assert traj is not None
    assert len(traj) == 1  # only one cycle started before done
    assert traj[0]["executed_actions"].shape == (3, 7), (
        f"mid-chunk finalise must yield shape[0]==3, got {traj[0]['executed_actions'].shape}"
    )

    # And flushing to HDF5 must succeed (no zero-shape rejections).
    out = libero_main._flush_trajectory_h5(
        str(tmp_path),
        task_id=0,
        subset_init_state_idx=0,
        task_name="open drawer",
        orig_init_state_idx=0,
        episode_id=0,
        seed=7,
        success=True,
        replan_steps=5,
        num_steps_wait=0,
        final_env_timestep=int(final_env_timestep),
        traj_buffer=traj,
    )
    assert out is not None and out.exists()
    with h5py.File(out, "r") as f:
        assert int(f["step_0000"].attrs["executed_action_count"]) == 3
        assert int(f.attrs["num_steps"]) == 3
        assert int(f.attrs["num_cycles"]) == 1


def test_run_episode_no_save_trajectory_returns_none() -> None:
    """When ``save_trajectory`` is False the traj slot must be None (the
    Layer-B client naming fallback also depends on this — plan §4 改动 4)."""
    env = _FakeEnv(done_at_step=None)
    client = _FakeClient(np.zeros((50, 7), dtype=np.float32))

    class _ArgsNoSave(_Args):
        save_trajectory = False

    _done, _, _, traj, _final = libero_main._run_episode(
        env, client, initial_state=None,
        task_description="pick apple",
        args=_ArgsNoSave(),
        max_steps=5,
        record_video=False,
    )
    assert traj is None
