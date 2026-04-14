"""Unit tests for ``exp/run_spawn_experiment.py`` (plan §11 + §18.A3.4 +
§19.4 + §20.R4).

A real spawn run needs a LIBERO env + a websocket server + a collected
HDF5 with server-side ``clean_action`` — none of that is reproducible
in CI. These tests therefore lock everything that can be verified without
side effects:

- Pure helpers: unit-key round-trip, top-k / random-k / equidistant-k
  selection, trajectory-depth parsing from config ids, and
  ``_quat2axisangle`` against hand-computed values.
- ``_SpawnCommon`` test seams (``client_factory`` / ``env_factory``).
- ``SpawnRunner.build_units`` + ``BaselineRunner.build_units``.
- ``_execute_spawn_unit`` end-to-end with tiny HDF5 fixtures and a fake
  client + env, so the teleport → prefill → rollout contract is locked.
- ``aggregate_spawn_results`` scanning a stub state JSON.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

import h5py
import numpy as np
import pytest

import exp.run_spawn_experiment as spawn


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_unit_key_roundtrip() -> None:
    """State-file keys must round-trip so ``--resume`` survives refactors.
    Using task/episode with slash inside ``ep`` locks the ':' split."""
    key = spawn.build_unit_key("clip_w7_d4", "task_3/episode_7", 12, 5, 2)
    assert key == "clip_w7_d4:task_3/episode_7:s12:n5:k2"
    assert spawn.parse_unit_key(key) == ("clip_w7_d4", "task_3/episode_7", 12, 5, 2)


def test_parse_unit_key_rejects_malformed() -> None:
    with pytest.raises(ValueError):
        spawn.parse_unit_key("not:valid")


def test_top_k_points_orders_by_descending_score() -> None:
    scores = [0.1, 0.9, 0.3, 0.95, 0.05]
    assert spawn._top_k_points(scores, 3) == [3, 1, 2]
    # Over-request is clamped to len(scores) so build_units can iterate safely.
    assert len(spawn._top_k_points(scores, 99)) == len(scores)


def test_random_k_points_deterministic_under_same_seed() -> None:
    a = spawn._random_k_points(T=100, k=5, seed=42)
    b = spawn._random_k_points(T=100, k=5, seed=42)
    assert a == b
    # Different seed → different draw (overwhelmingly likely for T=100).
    c = spawn._random_k_points(T=100, k=5, seed=43)
    assert c != a


def test_equidistant_k_points_spans_range() -> None:
    # T=10, k=5 → 0, 2, 4, 6, 8
    assert spawn._equidistant_k_points(10, 5) == [0, 2, 4, 6, 8]
    # k=1 → just start
    assert spawn._equidistant_k_points(10, 1) == [0]
    # k=0 → empty (no crash)
    assert spawn._equidistant_k_points(10, 0) == []


def test_infer_d_from_cfg() -> None:
    assert spawn.infer_d_from_cfg("clip_w7_d4") == 4
    assert spawn.infer_d_from_cfg("spatial16_w8_d4") == 4
    assert spawn.infer_d_from_cfg("max_pool_w3_d5") == 5
    with pytest.raises(ValueError):
        spawn.infer_d_from_cfg("no_depth_here")


def test_quat2axisangle_matches_reference_values() -> None:
    """Identity quaternion → zero rotation; 180° about x → π-axisangle.
    These two cases exercise both branches of the arccos/den switch."""
    assert np.allclose(spawn._quat2axisangle(np.array([0.0, 0.0, 0.0, 1.0])), np.zeros(3))
    # 180° rotation around x-axis: quat = (1, 0, 0, 0) → axis-angle = (π, 0, 0).
    out = spawn._quat2axisangle(np.array([1.0, 0.0, 0.0, 0.0]))
    assert np.allclose(out, np.array([math.pi, 0.0, 0.0]), atol=1e-6)


# ---------------------------------------------------------------------------
# SpawnRunner.build_units / BaselineRunner.build_units
# ---------------------------------------------------------------------------


def _make_common(tmp_path: Path, scores: Dict[str, Any], *, k_grid=(1, 3)) -> spawn._SpawnCommon:
    return spawn._SpawnCommon(
        cfg="cfgA",
        gt_dir=tmp_path / "gt",
        collected_dir=tmp_path / "collected",
        task_suite_name="libero_spatial",
        D=4,
        out_dir=tmp_path / "out",
        host="127.0.0.1",
        port=9000,
        n_grid=(1, 3),
        k_grid=k_grid,
        scores=scores,
        client_factory=lambda h, p: MagicMock(),
        env_factory=lambda tid, ts: MagicMock(),
    )


def test_spawn_runner_build_units_covers_top_k_x_n_grid(tmp_path: Path) -> None:
    """Lock Strategy A unit enumeration: for each episode, units are the
    cross product (top-k_max cycles × n_grid). A smaller k_max inside
    k_grid is a prefix, so distinct k_max values collapse correctly."""
    scores = {
        "task_0/episode_0": {"deviate_score": [0.1, 0.9, 0.3, 0.95, 0.05]},
    }
    common = _make_common(tmp_path, scores, k_grid=(1, 3))
    runner = spawn.SpawnRunner(state_path=tmp_path / "s.json", common=common)
    keys = {u.unit_key for u in runner.build_units()}

    # Top-3 cycles are [3, 1, 2]; k_grid includes 1 and 3. The k=1 row is
    # a prefix of k=3's list, so total unique units = 3 cycles × 2 (n_grid).
    expected = {
        f"cfgA:task_0/episode_0:s{s}:n{n}:k{k_idx}"
        for n in (1, 3)
        for k_idx, s in enumerate([3, 1, 2])
    }
    assert keys == expected


def test_spawn_runner_skips_episodes_without_scores(tmp_path: Path) -> None:
    scores = {
        "task_0/episode_0": {"deviate_score": []},  # empty — must skip
        "task_0/episode_1": {"deviate_score": [0.9, 0.1]},
    }
    common = _make_common(tmp_path, scores, k_grid=(1,))
    runner = spawn.SpawnRunner(state_path=tmp_path / "s.json", common=common)
    keys = {u.unit_key for u in runner.build_units()}
    # Only episode_1 contributes; k=1 × n_grid(1,3) → 2 units.
    assert keys == {
        "cfgA:task_0/episode_1:s0:n1:k0",
        "cfgA:task_0/episode_1:s0:n3:k0",
    }


def test_baseline_runner_random_strategy_builds_reproducibly(tmp_path: Path) -> None:
    scores = {"task_0/episode_0": {"deviate_score": [0.1] * 20}}
    common = _make_common(tmp_path, scores, k_grid=(3,))
    r1 = spawn.BaselineRunner(state_path=tmp_path / "r1.json", common=common, strategy="random")
    r2 = spawn.BaselineRunner(state_path=tmp_path / "r2.json", common=common, strategy="random")
    # Same cfg+ep+random_seed → same draw (deterministic resume).
    assert {u.unit_key for u in r1.build_units()} == {u.unit_key for u in r2.build_units()}


def test_baseline_random_seed_is_cross_process_stable(tmp_path: Path) -> None:
    """G2 §26 should-fix: the seed used for random baselines must be a
    stable hash (blake2b) rather than Python's ``hash()``, which is
    per-interpreter-randomised (``PYTHONHASHSEED``). Two runners that
    differ only in what Python's hash() would return must still produce
    the same draw — we simulate that by locking the expected seed bytes
    and checking the derived points are the blake2b-seeded output, not
    the hash()-seeded output.
    """
    import hashlib

    cfg, ep, random_seed = "clip_w7_d4", "task_3/episode_7", 0
    digest = hashlib.blake2b(
        f"{cfg}|{ep}|{random_seed}".encode(), digest_size=8
    ).digest()
    expected_seed = int.from_bytes(digest, "big")
    expected_points = spawn._random_k_points(T=20, k=3, seed=expected_seed)

    scores = {ep: {"deviate_score": [0.1] * 20}}
    common = spawn._SpawnCommon(
        cfg=cfg,
        gt_dir=tmp_path / "gt",
        collected_dir=tmp_path / "collected",
        task_suite_name="libero_spatial",
        D=4,
        out_dir=tmp_path / "out",
        host="127.0.0.1",
        port=9000,
        n_grid=(1,),
        k_grid=(3,),
        scores=scores,
        random_seed=random_seed,
        client_factory=lambda h, p: MagicMock(),
        env_factory=lambda tid, ts: MagicMock(),
    )
    runner = spawn.BaselineRunner(
        state_path=tmp_path / "s.json", common=common, strategy="random"
    )
    # Re-derive what unit_keys the runner would emit for the expected points.
    expected_keys = {
        spawn.build_unit_key(cfg, ep, int(s), 1, int(k_idx))
        for k_idx, s in enumerate(expected_points)
    }
    assert {u.unit_key for u in runner.build_units()} == expected_keys


def test_baseline_runner_equidistant_strategy(tmp_path: Path) -> None:
    scores = {"task_0/episode_0": {"deviate_score": [0.1] * 10}}
    common = _make_common(tmp_path, scores, k_grid=(5,))
    runner = spawn.BaselineRunner(
        state_path=tmp_path / "s.json", common=common, strategy="equidistant"
    )
    keys = {u.unit_key for u in runner.build_units()}
    # 5 equidistant cycles in T=10 → [0, 2, 4, 6, 8].
    expected = {
        f"cfgA:task_0/episode_0:s{s}:n{n}:k{k_idx}"
        for n in (1, 3)
        for k_idx, s in enumerate([0, 2, 4, 6, 8])
    }
    assert keys == expected


def test_baseline_runner_rejects_unknown_strategy(tmp_path: Path) -> None:
    common = _make_common(tmp_path, {}, k_grid=(1,))
    with pytest.raises(ValueError):
        spawn.BaselineRunner(
            state_path=tmp_path / "s.json", common=common, strategy="gibberish"
        )


# ---------------------------------------------------------------------------
# _execute_spawn_unit end-to-end against tiny HDF5 + fake client/env
# ---------------------------------------------------------------------------


def _write_minimal_gt(
    path: Path,
    *,
    num_cycles: int,
    num_steps: int,
    success: bool = False,
    num_steps_wait: int = 10,
    final_env_timestep: int | None = None,
) -> None:
    """Write a GT HDF5 small enough for tests but shaped exactly like
    main.py:_flush_trajectory_h5. Everything the spawn runner reads must
    be present (sim_state / env_timestep / env_cur_time / images / state).

    ``final_env_timestep`` defaults to ``num_steps_wait + num_steps`` so the
    happy-path tests don't have to think about it; pass ``None`` explicitly
    (and set ``omit_final_env_timestep=True`` via the write helper below)
    to exercise the §26 MF-3 fallback path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    if final_env_timestep is None:
        final_env_timestep = num_steps_wait + num_steps
    with h5py.File(path, "w") as f:
        f.attrs["task_name"] = "pick the thing"
        f.attrs["task_id"] = 3
        f.attrs["init_state_idx"] = 0
        f.attrs["orig_init_state_idx"] = 15
        f.attrs["episode_id"] = 150
        f.attrs["seed"] = 7
        f.attrs["num_cycles"] = num_cycles
        f.attrs["num_steps"] = num_steps
        f.attrs["success"] = bool(success)
        f.attrs["replan_steps"] = 5
        # G2 §26 MF-3: env-timestep budget anchor attrs.
        f.attrs["num_steps_wait"] = int(num_steps_wait)
        f.attrs["final_env_timestep"] = int(final_env_timestep)
        for i in range(num_cycles):
            g = f.create_group(f"step_{i:04d}")
            g.create_dataset("sim_state", data=rng.standard_normal(32).astype(np.float64))
            g.create_dataset("env_timestep", data=np.int64(i * 5))
            g.create_dataset("env_cur_time", data=np.float64(i * 0.1))
            g.create_dataset("agentview_image", data=np.zeros((8, 8, 3), dtype=np.uint8))
            g.create_dataset("eye_in_hand_image", data=np.zeros((8, 8, 3), dtype=np.uint8))
            g.create_dataset("robot_state", data=np.zeros(9, dtype=np.float64))
            g.create_dataset("env_action_chunk", data=np.zeros((10, 7), dtype=np.float32))
            g.create_dataset("executed_actions", data=np.zeros((5, 7), dtype=np.float32))
            g.attrs["executed_action_count"] = 5


def _write_minimal_collected(path: Path, *, num_cycles: int, H: int, model_Ad: int) -> None:
    """Server-side --collect HDF5 stub. Only ``clean_action`` is needed
    for prefill — everything else mirrors the production layout but can
    stay empty."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)
    with h5py.File(path, "w") as f:
        for i in range(num_cycles):
            g = f.create_group(f"step_{i:04d}")
            g.create_dataset(
                "clean_action",
                data=rng.standard_normal((H, model_Ad)).astype(np.float32),
            )


class _FakeClient:
    """Stand-in for WebsocketClientPolicy.

    ``infer`` returns a deterministic action chunk; ``prefill_trajectory``
    records its call shape so tests can assert the model-space contract.
    """

    def __init__(self, action_chunk: np.ndarray) -> None:
        self._chunk = action_chunk
        self.episode_start_kwargs: Dict[str, Any] | None = None
        self.episode_end_kwargs: Dict[str, Any] | None = None
        self.prefill_calls: List[Dict[str, Any]] = []
        self.infer_calls = 0
        self.close_calls = 0
        # If set, ``infer`` raises ``_infer_raise`` on that call index (1-based)
        # so tests can simulate mid-rollout server crashes.
        self._infer_raise: Exception | None = None
        self._infer_raise_after: int = 0
        # Flip this to make env.step report success immediately.
        self._fake_success_after = 999999

    def close(self) -> None:
        self.close_calls += 1

    def set_infer_raise(self, exc: Exception, *, after: int = 1) -> None:
        """Raise ``exc`` on the ``after``-th infer call (1-based).

        Used to verify MF-2's finally-block closes client+env even when
        rollout aborts mid-loop.
        """
        self._infer_raise = exc
        self._infer_raise_after = int(after)

    def set_success_after(self, n: int) -> None:
        self._fake_success_after = n

    def episode_start(self, *, experiment, task, episode_id, episode_name) -> Dict[str, Any]:
        self.episode_start_kwargs = dict(
            experiment=experiment, task=task,
            episode_id=episode_id, episode_name=episode_name,
        )
        return {}

    def episode_end(self, success: bool = False) -> Dict[str, Any]:
        self.episode_end_kwargs = {"success": success}
        return {}

    def prefill_trajectory(self, *, observations, actions, record=False, on_miss="error"):
        self.prefill_calls.append({
            "obs_len": len(observations),
            "actions_len": len(actions),
            "record": record,
            "on_miss": on_miss,
            "action_shapes": [np.asarray(a).shape for a in actions],
        })
        return {}

    def infer(self, obs: Dict[str, Any]) -> Dict[str, np.ndarray]:  # noqa: ARG002
        self.infer_calls += 1
        if self._infer_raise is not None and self.infer_calls >= self._infer_raise_after:
            raise self._infer_raise
        return {"actions": self._chunk.copy()}


class _FakeEnv:
    """Tiny env stand-in that tracks env.step calls and optionally flips
    ``info["success"]=True`` after a fixed number of steps.
    """

    def __init__(self, *, success_after: int | None = None) -> None:
        self.reset_calls = 0
        self.set_init_state_calls: List[np.ndarray] = []
        self.step_calls: List[np.ndarray] = []
        self.close_calls = 0
        self.timestep = 0
        self.cur_time = 0.0
        self.done = False
        self._success_after = success_after

    # OffScreenRenderEnv exposes an inner .env with timestep/cur_time; spawn
    # runner uses getattr(env, "env", env) so serving them off self works.

    def close(self) -> None:
        self.close_calls += 1

    def reset(self) -> Dict[str, Any]:
        self.reset_calls += 1
        return self._obs()

    def set_init_state(self, sim_state: np.ndarray) -> Dict[str, Any]:
        self.set_init_state_calls.append(np.asarray(sim_state).copy())
        return self._obs()

    def step(self, action) -> tuple:
        self.step_calls.append(np.asarray(action).copy())
        success = (self._success_after is not None
                   and len(self.step_calls) >= self._success_after)
        return self._obs(), 0.0, False, {"success": success}

    def _obs(self) -> Dict[str, Any]:
        return {
            "agentview_image": np.zeros((8, 8, 3), dtype=np.uint8),
            "robot0_eye_in_hand_image": np.zeros((8, 8, 3), dtype=np.uint8),
            "robot0_eef_pos": np.zeros(3, dtype=np.float64),
            "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
            "robot0_gripper_qpos": np.zeros(2, dtype=np.float64),
        }


def _make_common_with_hdf5(
    tmp_path: Path,
    *,
    ep: str = "task_3/episode_0",
    num_cycles: int = 8,
    client: _FakeClient,
    env: _FakeEnv,
) -> spawn._SpawnCommon:
    gt_path = tmp_path / "gt" / f"{ep}.h5"
    collected_path = tmp_path / "collected" / "libero_spatial" / f"{ep}.h5"
    _write_minimal_gt(gt_path, num_cycles=num_cycles, num_steps=num_cycles * 5)
    _write_minimal_collected(collected_path, num_cycles=num_cycles, H=10, model_Ad=32)
    return spawn._SpawnCommon(
        cfg="clip_w7_d4",
        gt_dir=tmp_path / "gt",
        collected_dir=tmp_path / "collected",
        task_suite_name="libero_spatial",
        D=4,
        out_dir=tmp_path / "out",
        host="127.0.0.1",
        port=9000,
        n_grid=(1, 3),
        k_grid=(3,),
        scores={ep: {"deviate_score": [0.1] * num_cycles}},
        max_spawn_env_steps=10,
        client_factory=lambda h, p: client,
        env_factory=lambda tid, ts: env,
    )


def test_execute_spawn_unit_full_contract(tmp_path: Path) -> None:
    """End-to-end through _execute_spawn_unit. This is the single most
    important test of the file — it locks the full teleport → prefill →
    rollout flow. If this passes, the individual helper mocks above cover
    the remaining edge cases."""
    chunk = np.full((10, 7), 0.25, dtype=np.float32)
    client = _FakeClient(chunk)
    env = _FakeEnv(success_after=4)  # success after 4 env steps
    common = _make_common_with_hdf5(tmp_path, client=client, env=env, num_cycles=8)

    # Unit: anchor_cycle = s + n = 5, so D-1 = 3 prefill cycles from [2, 3, 4].
    result = spawn._execute_spawn_unit(common=common, ep="task_3/episode_0", s=2, n=3, k_idx=0)

    # Teleport happened exactly once and used the HDF5 sim_state.
    assert len(env.set_init_state_calls) == 1
    # env.timestep was restored from HDF5 attrs step_{5}/env_timestep = 5*5=25.
    # (We wrote env_timestep as i*5 in the fixture.)
    assert env.timestep == 25
    assert env.cur_time == pytest.approx(0.5, rel=1e-6)

    # episode_start / episode_end both fired with the right contract.
    assert client.episode_start_kwargs is not None
    assert client.episode_start_kwargs["experiment"] == "spawn_clip_w7_d4"
    assert client.episode_start_kwargs["episode_name"] == ""   # no HDF5 capture
    assert client.episode_end_kwargs == {"success": True}

    # prefill_trajectory was called exactly once with MODEL-space actions
    # (H=10, model_Ad=32 per the collected-HDF5 fixture).
    assert len(client.prefill_calls) == 1
    pf = client.prefill_calls[0]
    assert pf["obs_len"] == 3 and pf["actions_len"] == 3
    assert all(shape == (10, 32) for shape in pf["action_shapes"])
    assert pf["on_miss"] == "error" and pf["record"] is False

    # Rollout ran 4 env.step calls and then success=True broke the loop.
    assert len(env.step_calls) == 4
    assert result["success"] is True
    assert result["env_steps_executed"] == 4
    assert result["episode"] == "task_3/episode_0"
    assert (result["s"], result["n"], result["k_idx"]) == (2, 3, 0)


def test_execute_spawn_unit_missing_collected_raises(tmp_path: Path) -> None:
    """§19.4 regression guard: if the server-side collected HDF5 is
    missing, we must fail loudly rather than silently fall back to
    env-space actions — which would break CP3 cache keying."""
    chunk = np.full((10, 7), 0.25, dtype=np.float32)
    client = _FakeClient(chunk)
    env = _FakeEnv()
    common = _make_common_with_hdf5(tmp_path, client=client, env=env)
    # Delete the collected HDF5.
    (tmp_path / "collected" / "libero_spatial" / "task_3/episode_0.h5").unlink()

    with pytest.raises(FileNotFoundError, match="collected HDF5 not found"):
        spawn._execute_spawn_unit(common=common, ep="task_3/episode_0", s=2, n=3, k_idx=0)


def test_execute_spawn_unit_beyond_num_cycles_returns_skip(tmp_path: Path) -> None:
    """If deviate_score references a point beyond the GT trajectory we
    don't want BaseRunState to retry forever — surface the skip as a
    non-error result."""
    chunk = np.full((10, 7), 0.25, dtype=np.float32)
    client = _FakeClient(chunk)
    env = _FakeEnv()
    common = _make_common_with_hdf5(tmp_path, client=client, env=env, num_cycles=5)

    # anchor = s + n = 10 > num_cycles = 5.
    result = spawn._execute_spawn_unit(common=common, ep="task_3/episode_0", s=7, n=3, k_idx=0)
    assert result["success"] is False
    assert result["env_steps_executed"] == 0
    assert "anchor_cycle" in result["reason"]
    # Must have neither teleported nor called the client.
    assert len(env.set_init_state_calls) == 0
    assert client.episode_start_kwargs is None


# ---------------------------------------------------------------------------
# G2 §26 MF-2: cleanup must run on every exit path
# ---------------------------------------------------------------------------


def test_execute_spawn_unit_closes_client_and_env_on_success(tmp_path: Path) -> None:
    """Happy path: client.close() and env.close() each fire exactly once,
    in addition to episode_end. Long parallel sweeps cannot leak MuJoCo/
    EGL or websocket resources (MF-2)."""
    chunk = np.full((10, 7), 0.25, dtype=np.float32)
    client = _FakeClient(chunk)
    env = _FakeEnv(success_after=2)
    common = _make_common_with_hdf5(tmp_path, client=client, env=env)

    result = spawn._execute_spawn_unit(common=common, ep="task_3/episode_0", s=2, n=3, k_idx=0)

    assert result["success"] is True
    assert client.close_calls == 1, "client.close must run on the success path"
    assert env.close_calls == 1, "env.close must run on the success path"
    assert client.episode_end_kwargs == {"success": True}


def test_execute_spawn_unit_closes_resources_when_infer_raises(tmp_path: Path) -> None:
    """If ``client.infer`` raises mid-rollout (simulated server crash), the
    original exception must propagate BUT both client.close() and env.close()
    must still run. Otherwise a parallel sweep that hits 100 infer crashes
    leaks 100 websockets + 100 MuJoCo handles."""
    chunk = np.full((10, 7), 0.25, dtype=np.float32)
    client = _FakeClient(chunk)
    client.set_infer_raise(RuntimeError("server died"), after=1)
    env = _FakeEnv()
    common = _make_common_with_hdf5(tmp_path, client=client, env=env)

    with pytest.raises(RuntimeError, match="server died"):
        spawn._execute_spawn_unit(common=common, ep="task_3/episode_0", s=2, n=3, k_idx=0)

    # Both resources released even though rollout aborted.
    assert client.close_calls == 1
    assert env.close_calls == 1
    # episode_end must fire with the last-seen success value (False here,
    # since the crash happened before any successful env.step).
    assert client.episode_end_kwargs == {"success": False}


# ---------------------------------------------------------------------------
# cleanup/05 G2 boundaries: make_client / env.close failure modes
# ---------------------------------------------------------------------------


def test_execute_spawn_unit_closes_env_when_make_client_raises(tmp_path: Path) -> None:
    """``common.make_client()`` is called *after* ``env`` is constructed.
    If client construction fails, the already-built env MUST still be
    closed — otherwise a flaky server would leak MuJoCo handles in
    parallel sweeps (cleanup/05 boundary 1)."""
    chunk = np.full((10, 7), 0.25, dtype=np.float32)
    env = _FakeEnv()

    def _raising_client_factory(host: str, port: int):
        raise RuntimeError("ws connect refused")

    common = _make_common_with_hdf5(tmp_path, client=_FakeClient(chunk), env=env)
    # Swap the factory to the raising one; the instance in common isn't
    # used because make_client calls client_factory first.
    common.client_factory = _raising_client_factory

    with pytest.raises(RuntimeError, match="ws connect refused"):
        spawn._execute_spawn_unit(common=common, ep="task_3/episode_0", s=2, n=3, k_idx=0)

    # env was built, then client_factory raised — but env must still close.
    assert len(env.set_init_state_calls) == 1
    assert env.close_calls == 1


def test_execute_spawn_unit_rollout_error_not_masked_by_env_close_error(
    tmp_path: Path,
) -> None:
    """If ``env.close()`` raises during cleanup, the runner must log the
    env-close failure but still propagate the *original* rollout exception
    (cleanup/05 boundary 2 — defense against finally-block masking)."""
    chunk = np.full((10, 7), 0.25, dtype=np.float32)
    client = _FakeClient(chunk)
    client.set_infer_raise(RuntimeError("server died"), after=1)
    env = _FakeEnv()

    original_close = env.close

    def _close_and_raise() -> None:
        original_close()
        raise RuntimeError("env teardown exploded")

    env.close = _close_and_raise  # type: ignore[method-assign]

    common = _make_common_with_hdf5(tmp_path, client=client, env=env)

    # Python rule: when the try-block raises and finally-block also
    # raises, the finally exception replaces the try exception UNLESS the
    # finally explicitly catches. Our _cleanup_spawn_resources catches
    # env.close exceptions and just logs them, so the original RuntimeError
    # from infer must be the one that propagates.
    with pytest.raises(RuntimeError, match="server died"):
        spawn._execute_spawn_unit(common=common, ep="task_3/episode_0", s=2, n=3, k_idx=0)

    # env.close was still invoked (once) even though it threw.
    assert env.close_calls == 1
    assert client.close_calls == 1


def test_run_rollout_budget_exceeded_returns_false(tmp_path: Path) -> None:
    """cleanup/05 boundary 3: when neither ``info['success']`` nor
    ``done`` triggers within ``budget`` steps, ``_run_rollout`` returns
    ``(False, budget)``. Budget-exceeded must surface as
    ``success=False`` + ``env_steps_executed==budget`` — not as an
    error (BaseRunState would otherwise retry the unit forever)."""
    chunk = np.full((10, 7), 0.25, dtype=np.float32)
    client = _FakeClient(chunk)
    env = _FakeEnv()  # never sets success, never sets done
    common = _make_common_with_hdf5(tmp_path, client=client, env=env)
    common.max_spawn_env_steps = 3  # tight budget

    result = spawn._execute_spawn_unit(
        common=common, ep="task_3/episode_0", s=2, n=3, k_idx=0
    )

    assert result["success"] is False
    assert result["env_steps_executed"] == 3
    assert len(env.step_calls) == 3
    # episode_end records the final success value (False here).
    assert client.episode_end_kwargs == {"success": False}


# ---------------------------------------------------------------------------
# G2 §26 MF-3: env-timestep budget axis + fallback warn
# ---------------------------------------------------------------------------


def _strip_attr(path: Path, name: str) -> None:
    """Delete an attr from an HDF5 file so we can simulate older GT data."""
    with h5py.File(path, "r+") as f:
        if name in f.attrs:
            del f.attrs[name]


def test_execute_spawn_unit_budget_uses_final_env_timestep_attr(tmp_path: Path) -> None:
    """The rollout budget must be ``final_env_timestep - anchor_env_timestep``
    (clamped by ``max_spawn_env_steps``), NOT the old ``num_steps - env_timestep``
    approximation that mixed env-step and policy-step axes.

    Construct a GT where the two would differ: ``num_steps_wait=20`` bakes
    a large warm-up offset into the env-timestep axis, so if the runner
    still used ``num_steps`` it would under-budget by exactly 20 steps.
    """
    chunk = np.full((10, 7), 0.25, dtype=np.float32)
    client = _FakeClient(chunk)
    env = _FakeEnv()  # no success_after → runs to budget

    ep = "task_3/episode_0"
    gt_path = tmp_path / "gt" / f"{ep}.h5"
    collected_path = tmp_path / "collected" / "libero_spatial" / f"{ep}.h5"
    # num_cycles=3, num_steps=15. Anchor cycle 1 → env_timestep = 5 (from
    # fixture's i*5). num_steps_wait=20, final_env_timestep=35
    # (= num_steps_wait + num_steps, realistic Layer C output).
    _write_minimal_gt(
        gt_path,
        num_cycles=3,
        num_steps=15,
        num_steps_wait=20,
        final_env_timestep=35,
    )
    _write_minimal_collected(collected_path, num_cycles=3, H=10, model_Ad=32)
    common = spawn._SpawnCommon(
        cfg="clip_w7_d4",
        gt_dir=tmp_path / "gt",
        collected_dir=tmp_path / "collected",
        task_suite_name="libero_spatial",
        D=2,  # small D so prefill is short
        out_dir=tmp_path / "out",
        host="127.0.0.1",
        port=9000,
        n_grid=(1,),
        k_grid=(1,),
        scores={ep: {"deviate_score": [0.1] * 3}},
        # Large cap so budget is limited by GT remaining, not the safety clamp.
        max_spawn_env_steps=100,
        client_factory=lambda h, p: client,
        env_factory=lambda tid, ts: env,
    )

    spawn._execute_spawn_unit(common=common, ep=ep, s=0, n=1, k_idx=0)

    # Anchor env_timestep = 5, final_env_timestep = 35 ⇒ budget should be 30.
    # The OLD broken formula (num_steps - env_timestep = 15 - 5 = 10) would
    # stop after 10 steps; MF-3 must give the full 30.
    assert len(env.step_calls) == 30, (
        f"Expected 30 env.step calls (final=35 minus anchor=5), got "
        f"{len(env.step_calls)} — MF-3 axis regression?"
    )


def test_execute_spawn_unit_falls_back_to_num_steps_when_final_missing(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    """Older GT HDF5 predating MF-3 lack ``final_env_timestep``; the runner
    must fall back to ``num_steps_wait + num_steps`` AND warn loudly so the
    operator knows the budget is only approximate for that episode."""
    chunk = np.full((10, 7), 0.25, dtype=np.float32)
    client = _FakeClient(chunk)
    env = _FakeEnv()

    ep = "task_3/episode_0"
    gt_path = tmp_path / "gt" / f"{ep}.h5"
    collected_path = tmp_path / "collected" / "libero_spatial" / f"{ep}.h5"
    _write_minimal_gt(
        gt_path,
        num_cycles=3,
        num_steps=15,
        num_steps_wait=20,
        final_env_timestep=999,  # written, then stripped below
    )
    _write_minimal_collected(collected_path, num_cycles=3, H=10, model_Ad=32)
    # Simulate "old GT" by deleting the attr.
    _strip_attr(gt_path, "final_env_timestep")

    common = spawn._SpawnCommon(
        cfg="clip_w7_d4",
        gt_dir=tmp_path / "gt",
        collected_dir=tmp_path / "collected",
        task_suite_name="libero_spatial",
        D=2,
        out_dir=tmp_path / "out",
        host="127.0.0.1",
        port=9000,
        n_grid=(1,),
        k_grid=(1,),
        scores={ep: {"deviate_score": [0.1] * 3}},
        max_spawn_env_steps=100,
        client_factory=lambda h, p: client,
        env_factory=lambda tid, ts: env,
    )

    with caplog.at_level("WARNING", logger="exp.run_spawn_experiment"):
        spawn._execute_spawn_unit(common=common, ep=ep, s=0, n=1, k_idx=0)

    # Fallback: num_steps_wait(20) + num_steps(15) = 35. Anchor=5 ⇒ 30 steps.
    assert len(env.step_calls) == 30
    # Warning must fire so the operator can spot stale GT.
    assert any("final_env_timestep" in rec.message for rec in caplog.records), (
        "Fallback path must warn operator about missing MF-3 attr"
    )


def test_execute_spawn_unit_no_prefill_when_anchor_zero(tmp_path: Path) -> None:
    """When anchor_cycle = s + n = 0, prefill_obs is empty, so
    ``client.prefill_trajectory`` must NOT fire — calling it with an
    empty observation list would put the server in an inconsistent state."""
    chunk = np.full((10, 7), 0.25, dtype=np.float32)
    client = _FakeClient(chunk)
    env = _FakeEnv()
    common = _make_common_with_hdf5(tmp_path, client=client, env=env)

    spawn._execute_spawn_unit(common=common, ep="task_3/episode_0", s=0, n=0, k_idx=0)
    assert client.prefill_calls == []


# ---------------------------------------------------------------------------
# SpawnRunner via BaseRunState glue — parallel_run presence already locked
# by test_compute_deviate_scores. Here we verify the runner ties to
# _execute_spawn_unit through execute_unit() (no re-implementation drift).
# ---------------------------------------------------------------------------


def test_spawn_runner_execute_unit_delegates_to_execute_spawn_unit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    chunk = np.full((10, 7), 0.25, dtype=np.float32)
    client = _FakeClient(chunk)
    env = _FakeEnv(success_after=1)
    common = _make_common_with_hdf5(tmp_path, client=client, env=env)

    runner = spawn.SpawnRunner(state_path=tmp_path / "st.json", common=common)
    # One unit taken from the top-k enumeration — must call _execute_spawn_unit.
    called: Dict[str, Any] = {}

    def fake_execute(**kwargs):
        called.update(kwargs)
        return {"success": True, "inference_failed": False,
                "s": kwargs["s"], "n": kwargs["n"], "k_idx": kwargs["k_idx"],
                "episode": kwargs["ep"], "env_steps_executed": 1}

    monkeypatch.setattr(spawn, "_execute_spawn_unit", fake_execute)
    unit = runner.build_units()[0]
    result = runner.execute_unit(unit)
    # The runner must forward cfg through common, not through the key — the
    # key mismatch guard in execute_unit would catch a refactor that swapped
    # ordering.
    assert called["common"] is common
    assert result["success"] is True


def test_spawn_runner_execute_unit_rejects_cross_cfg_key(tmp_path: Path) -> None:
    """Defensive check: sharing a state file across cfgs is a bug and
    must raise rather than silently running with the wrong bundle."""
    chunk = np.full((10, 7), 0.25, dtype=np.float32)
    client = _FakeClient(chunk)
    env = _FakeEnv()
    common = _make_common_with_hdf5(tmp_path, client=client, env=env)

    runner = spawn.SpawnRunner(state_path=tmp_path / "st.json", common=common)
    bad = spawn.UnitState(unit_key="OTHERCFG:task_3/episode_0:s0:n0:k0")
    with pytest.raises(ValueError, match="cfg mismatch"):
        runner.execute_unit(bad)


# ---------------------------------------------------------------------------
# aggregate_spawn_results
# ---------------------------------------------------------------------------


def test_aggregate_spawn_results_emits_csv_with_done_units_only(tmp_path: Path) -> None:
    state = {
        "clip_w7_d4:task_3/episode_0:s2:n3:k0": {
            "unit_key": "clip_w7_d4:task_3/episode_0:s2:n3:k0",
            "status": "done",
            "result": {
                "success": True, "s": 2, "n": 3, "k_idx": 0,
                "episode": "task_3/episode_0", "env_steps_executed": 4,
            },
        },
        "clip_w7_d4:task_3/episode_0:s2:n5:k0": {
            "unit_key": "clip_w7_d4:task_3/episode_0:s2:n5:k0",
            "status": "failed",  # must not appear in CSV
            "result": {"error": "boom"},
        },
    }
    (tmp_path / "spawn_state_clip_w7_d4.json").write_text(json.dumps(state))
    rows = spawn.aggregate_spawn_results(
        out_dir=tmp_path, configs=["clip_w7_d4"],
    )
    assert len(rows) == 1
    assert rows[0]["strategy"] == "top_k"
    assert rows[0]["success"] is True
    assert rows[0]["env_steps_executed"] == 4
    # CSV file exists and contains exactly one data row.
    csv_text = (tmp_path / "spawn_aggregate.csv").read_text().strip().splitlines()
    assert len(csv_text) == 2  # header + 1 row


# ---------------------------------------------------------------------------
# cleanup/08 P1-5: _BaseSpawnRunner is abstract
# ---------------------------------------------------------------------------


def test_base_spawn_runner_cannot_be_instantiated_without_iter_targets(tmp_path):
    """P1-5 lock: ``_BaseSpawnRunner`` must refuse direct instantiation so
    future subclasses cannot accidentally inherit a runner that silently
    produces zero units. The ``@abstractmethod`` decorator converts the
    previously-silent ``NotImplementedError`` path into a clean TypeError
    at construction time."""
    from exp.run_spawn_experiment import _BaseSpawnRunner, _SpawnCommon

    common = _SpawnCommon(
        cfg="cfgA",
        gt_dir=tmp_path / "gt",
        collected_dir=tmp_path / "coll",
        task_suite_name="libero_spatial",
        D=4,
        out_dir=tmp_path / "out",
        host="127.0.0.1",
        port=0,
    )
    with pytest.raises(TypeError, match="_iter_targets"):
        _BaseSpawnRunner(
            state_path=tmp_path / "s.json", common=common, strategy="top_k",
        )


# ---------------------------------------------------------------------------
# cleanup/09 P1-4: spawn_state metadata sidecar
# ---------------------------------------------------------------------------


def _build_fake_common_for_sidecar(tmp_path: Path, cfg: str = "clip_w7_d4") -> "spawn._SpawnCommon":
    return spawn._SpawnCommon(
        cfg=cfg,
        gt_dir=tmp_path / "gt",
        collected_dir=tmp_path / "coll",
        task_suite_name="libero_spatial",
        D=4,
        out_dir=tmp_path / "out",
        host="127.0.0.1",
        port=0,
    )


def test_spawn_runner_writes_meta_sidecar_on_construction(tmp_path: Path) -> None:
    """P1-4 lock: ``SpawnRunner(...)`` emits ``{state}.meta.json`` with
    ``strategy=top_k`` and the exact ``cfg`` from ``_SpawnCommon``."""
    common = _build_fake_common_for_sidecar(tmp_path)
    state_path = tmp_path / "spawn_state_clip_w7_d4.json"
    spawn.SpawnRunner(state_path=state_path, common=common)

    meta_path = state_path.with_suffix(".meta.json")
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text())
    assert meta == {"strategy": "top_k", "config": "clip_w7_d4"}


def test_baseline_runner_writes_meta_sidecar_with_baseline_strategy(tmp_path: Path) -> None:
    """Baselines bake ``strategy`` into both the state filename AND the
    sidecar; the sidecar is the authoritative field aggregate reads."""
    common = _build_fake_common_for_sidecar(tmp_path)
    state_path = tmp_path / "spawn_state_random_clip_w7_d4.json"
    spawn.BaselineRunner(state_path=state_path, common=common, strategy="random")

    meta = json.loads(state_path.with_suffix(".meta.json").read_text())
    assert meta == {"strategy": "random", "config": "clip_w7_d4"}


def test_aggregate_prefers_sidecar_over_filename(tmp_path: Path) -> None:
    """When a sidecar exists, aggregate must use its ``(strategy, config)``
    instead of re-parsing the filename — this closes the pre-cleanup
    ambiguity where ``spawn_state_random_clip_w7_d4.json`` could have been
    interpreted as cfg=``random_clip_w7_d4`` by a naive filename parse."""
    state_path = tmp_path / "spawn_state_random_clip_w7_d4.json"
    state_path.write_text(json.dumps({
        "clip_w7_d4:task_3/episode_0:s2:n3:k0": {
            "unit_key": "clip_w7_d4:task_3/episode_0:s2:n3:k0",
            "status": "done",
            "result": {
                "success": True, "s": 2, "n": 3, "k_idx": 0,
                "episode": "task_3/episode_0", "env_steps_executed": 11,
            },
        },
    }))
    state_path.with_suffix(".meta.json").write_text(json.dumps({
        "strategy": "random", "config": "clip_w7_d4",
    }))

    rows = spawn.aggregate_spawn_results(
        out_dir=tmp_path, configs=[],
        extra_state_files=[state_path],
    )
    assert len(rows) == 1
    assert rows[0]["config"] == "clip_w7_d4"
    assert rows[0]["strategy"] == "random"


def test_aggregate_falls_back_with_warning_when_sidecar_missing(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Legacy state file lacks a sidecar — aggregate must warn AND still
    recover ``(strategy, config)`` from the filename so pre-cleanup runs
    remain aggregatable during the migration window."""
    state_path = tmp_path / "spawn_state_clip_w7_d4.json"
    state_path.write_text(json.dumps({
        "clip_w7_d4:task_3/episode_0:s2:n3:k0": {
            "status": "done",
            "result": {
                "success": True, "s": 2, "n": 3, "k_idx": 0,
                "episode": "task_3/episode_0", "env_steps_executed": 7,
            },
        },
    }))
    # No meta.json sidecar on purpose.

    with caplog.at_level("WARNING"):
        rows = spawn.aggregate_spawn_results(
            out_dir=tmp_path, configs=["clip_w7_d4"],
        )
    assert len(rows) == 1
    assert rows[0]["strategy"] == "top_k"
    assert rows[0]["config"] == "clip_w7_d4"
    assert any("No .meta.json sidecar" in r.message for r in caplog.records)


def test_meta_sidecar_idempotent_when_content_matches(tmp_path: Path) -> None:
    """P1-4 lock: reconstructing the same SpawnRunner over an existing,
    matching sidecar must be a no-op — resume flows build a fresh runner
    object but the state+meta on disk are already correct."""
    common = _build_fake_common_for_sidecar(tmp_path)
    state_path = tmp_path / "spawn_state_clip_w7_d4.json"
    spawn.SpawnRunner(state_path=state_path, common=common)
    meta_path = state_path.with_suffix(".meta.json")
    first_mtime = meta_path.stat().st_mtime_ns
    first_content = meta_path.read_text()

    # Second construction must not raise and must not rewrite the file.
    spawn.SpawnRunner(state_path=state_path, common=common)
    assert meta_path.read_text() == first_content
    assert meta_path.stat().st_mtime_ns == first_mtime


def test_meta_sidecar_raises_on_strategy_mismatch(tmp_path: Path) -> None:
    """P1-4 lock: if the existing sidecar says strategy=top_k but a
    BaselineRunner (strategy=random) is constructed over the same state
    path, the runner MUST refuse rather than silently overwriting."""
    common = _build_fake_common_for_sidecar(tmp_path)
    state_path = tmp_path / "spawn_state_clip_w7_d4.json"
    state_path.with_suffix(".meta.json").write_text(
        json.dumps({"strategy": "top_k", "config": "clip_w7_d4"})
    )

    with pytest.raises(ValueError, match="sidecar mismatch"):
        spawn.BaselineRunner(
            state_path=state_path, common=common, strategy="random",
        )


def test_meta_sidecar_raises_on_config_mismatch(tmp_path: Path) -> None:
    """P1-4 lock: same strategy, different cfg must also raise. Catches the
    case where a state file was copied from another run and the operator
    forgot to rename the sidecar."""
    common = _build_fake_common_for_sidecar(tmp_path, cfg="clip_w7_d4")
    state_path = tmp_path / "spawn_state_clip_w7_d4.json"
    state_path.with_suffix(".meta.json").write_text(
        json.dumps({"strategy": "top_k", "config": "OTHER"})
    )

    with pytest.raises(ValueError, match="sidecar mismatch"):
        spawn.SpawnRunner(state_path=state_path, common=common)


def test_meta_sidecar_raises_on_corrupt_existing_sidecar(tmp_path: Path) -> None:
    """Unreadable JSON in the sidecar is as dangerous as a mismatched one —
    we cannot verify content equality, so we must refuse rather than
    overwriting and losing whatever invariant it encoded."""
    common = _build_fake_common_for_sidecar(tmp_path)
    state_path = tmp_path / "spawn_state_clip_w7_d4.json"
    state_path.with_suffix(".meta.json").write_text("not-json{{")

    with pytest.raises(RuntimeError, match="unreadable"):
        spawn.SpawnRunner(state_path=state_path, common=common)


def test_aggregate_falls_back_when_sidecar_is_corrupted(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Corrupt sidecar (unreadable JSON or missing keys) must not crash
    aggregate — it should warn, treat the sidecar as missing, and fall
    back to the filename parse."""
    state_path = tmp_path / "spawn_state_clip_w7_d4.json"
    state_path.write_text(json.dumps({
        "clip_w7_d4:task_3/episode_0:s2:n3:k0": {
            "status": "done",
            "result": {
                "success": True, "s": 2, "n": 3, "k_idx": 0,
                "episode": "task_3/episode_0", "env_steps_executed": 2,
            },
        },
    }))
    state_path.with_suffix(".meta.json").write_text("not json at all")

    with caplog.at_level("WARNING"):
        rows = spawn.aggregate_spawn_results(
            out_dir=tmp_path, configs=["clip_w7_d4"],
        )
    assert len(rows) == 1
    assert rows[0]["config"] == "clip_w7_d4"
    assert any("unreadable" in r.message for r in caplog.records)
