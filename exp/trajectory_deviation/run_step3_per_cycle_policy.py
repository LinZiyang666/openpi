"""Step 3 per-cycle policy runner (plan: trajectory_deviation_step3_redesign).

For one cache config (``--cfg``), this driver replays every Step-2-covered
episode inside a real LIBERO env and, on each inference cycle, tells the
server whether to use cache or real inference based on the pre-computed
Step 2 ``deviate_score`` plus the ``(tau, n)`` sweep grid.

The episode list is driven directly by ``deviate_score_{cfg}.json`` keys:
Step 2 is computed only on the per-cfg failure subset from Step 1a, so its
keys are the authoritative Step 3 input (``task_X/episode_Y`` where ``Y`` is
the Step 1b ``subset_init_state_idx``). This avoids mixing the Step 1a
original-LIBERO index space with the Step 1b subset index space.

Signal channel: the runner injects ``"__gate_decision__": "skip" | "search"``
into the obs dict passed to ``WebsocketClientPolicy.infer``. The server's
``InferenceInterceptor`` pops that field before the input transform and
forwards it through ``CacheOrchestrator`` to ``ClientControlledGate``. The
underlying Step 3 YAML sets ``gate.type: client_controlled`` at CP1.

Parallelism model mirrors ``examples/libero/main.py::_run_concurrent``:
up to 5 workers (MuJoCo EGL limit), each owns its own
``WebsocketClientPolicy`` connection and its own LIBERO ``OffScreenRenderEnv``.
An ``init_lock`` serialises env / websocket construction; per-unit state is
persisted via ``BaseRunState`` with atomic save.

Reference CLI (one of three cfgs; the other two are launched against their
own server):

    uv run exp/trajectory_deviation/run_step3_per_cycle_policy.py \\
        --cfg clip_w7_d4 \\
        --host host_a --port 8001 \\
        --yaml configs/cache_runs/deviate_exp/step3_clip_w7_d4.yaml \\
        --deviate-score-json data/deviation_experiment/step2_deviate_scores/deviate_score_clip_w7_d4.json \\
        --init-states-dir data/deviation_experiment/step1b_inits \\
        --out-dir data/deviation_experiment/step3/clip_w7_d4 \\
        --num-workers 5
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Path bootstrap so ``python exp/.../run_step3_per_cycle_policy.py`` works
# without installing the package. Mirrors other runners in this directory.
if __package__ in {None, ""}:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from exp.common._cache_config_rpc import send_load_cache_config
from exp.common._run_state_base import BaseRunState, UnitState
from exp.common._unit_key import Step3PerCycleKey
from exp.trajectory_deviation._libero_env import build_libero_env

logger = logging.getLogger(__name__)

# Constants match examples/libero/main.py so policy / cache artifacts stay
# byte-equivalent to the Step 1a / Step 2 reference pipelines.
_LIBERO_DUMMY_ACTION: List[float] = [0.0] * 6 + [-1.0]
_LIBERO_ENV_RESOLUTION = 256
_DEFAULT_RESIZE = 224
_DEFAULT_REPLAN_STEPS = 5
_DEFAULT_NUM_STEPS_WAIT = 10
# Max env steps per suite. Copied from ``examples/libero/main.py::_get_max_steps``.
_MAX_STEPS_BY_SUITE: Dict[str, int] = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}
# MuJoCo EGL concurrency ceiling; enforced by examples/libero/main.py too.
_MAX_WORKERS_CAP = 5

# Single lock guarding JSONL append across all worker threads.
_JSONL_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Observation / rotation helpers (copied from examples/libero/main.py and
# exp/trajectory_deviation/run_spawn_experiment.py to keep Step 3 decoupled
# from the about-to-be-archived spawn runner).
# ---------------------------------------------------------------------------


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64).copy()
    if q[3] > 1.0:
        q[3] = 1.0
    elif q[3] < -1.0:
        q[3] = -1.0
    den = math.sqrt(1.0 - q[3] * q[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (q[:3] * 2.0 * math.acos(q[3])) / den


def _obs_env_to_policy(
    env_obs: Dict[str, Any],
    task_description: str,
    *,
    resize: int = _DEFAULT_RESIZE,
) -> Dict[str, Any]:
    """Convert LIBERO env obs to the dict consumed by the policy server."""
    from openpi_client import image_tools  # type: ignore[import]

    img = np.ascontiguousarray(env_obs["agentview_image"][::-1, ::-1])
    wrist_img = np.ascontiguousarray(env_obs["robot0_eye_in_hand_image"][::-1, ::-1])
    img = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(img, resize, resize)
    )
    wrist_img = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(wrist_img, resize, resize)
    )
    state = np.concatenate(
        (
            np.asarray(env_obs["robot0_eef_pos"], dtype=np.float64),
            _quat2axisangle(env_obs["robot0_eef_quat"]),
            np.asarray(env_obs["robot0_gripper_qpos"], dtype=np.float64),
        )
    )
    return {
        "observation/image": img,
        "observation/wrist_image": wrist_img,
        "observation/state": state,
        "prompt": str(task_description),
    }


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


_EP_KEY_RE = re.compile(r"^task_(\d+)/episode_(\d+)$")


def _parse_ep_key(ep: str) -> Tuple[int, int]:
    """Return ``(task_id, subset_init_state_idx)`` for an ``ep`` string.

    Step 1b writes GT HDF5 files at
    ``gt_trajectories/task_{task_id}/episode_{subset_init_state_idx}.h5`` and
    Step 2 carries the same ``task_X/episode_Y`` layout forward as keys in
    ``deviate_score_{cfg}.json``.
    """
    m = _EP_KEY_RE.match(ep)
    if not m:
        raise ValueError(f"Unrecognised episode key {ep!r}; expected 'task_X/episode_Y'")
    return int(m.group(1)), int(m.group(2))


def _load_deviate_scores(score_path: Path) -> Dict[str, np.ndarray]:
    """Return per-episode ``deviate_score`` arrays keyed by ``task_X/episode_Y``."""
    raw = json.loads(score_path.read_text())
    out: Dict[str, np.ndarray] = {}
    for ep, payload in raw.items():
        score_list = payload.get("deviate_score")
        if score_list is None:
            raise ValueError(
                f"{score_path}: episode {ep!r} missing 'deviate_score' field"
            )
        out[ep] = np.asarray(score_list, dtype=np.float64)
    return out


def _load_pruned_inits(init_states_dir: Path, task_name: str) -> np.ndarray:
    """Load the Step 1b pruned init-state tensor for ``task_name``.

    The returned array is indexed by ``subset_init_state_idx`` (the Y in
    ``task_X/episode_Y``).
    """
    import torch  # type: ignore[import]

    pruned = init_states_dir / f"{task_name}.pruned_init"
    full = init_states_dir / f"{task_name}.init"
    for path in (pruned, full):
        if path.exists():
            return torch.load(path, weights_only=False)
    raise FileNotFoundError(
        f"No init states for task {task_name!r} under {init_states_dir}. "
        f"Run Step 1b first (scripts/dump_step1a_failed_inits.py)."
    )


def _resolve_task(task_suite_name: str, task_id: int):
    """Return the LIBERO task object (for ``name``, ``language``)."""
    from libero.libero.benchmark import get_benchmark_dict  # type: ignore[import]

    suite = get_benchmark_dict()[task_suite_name]()
    return suite.get_task(int(task_id))


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


@dataclass
class _RunnerConfig:
    cfg: str
    host: str
    port: int
    task_suite_name: str
    init_states_dir: Path
    deviate_scores: Dict[str, np.ndarray]
    tau_grid: Sequence[int]
    n_grid: Sequence[int]
    max_cycles_safety: int  # extra cycles beyond ceil(max_env_steps/replan_steps)
    replan_steps: int
    num_steps_wait: int
    max_env_steps: int
    resize_size: int
    results_jsonl: Path
    experiment_tag: str


class Step3PerCycleRunner(BaseRunState):
    """``BaseRunState`` subclass for Step 3 per-cycle policy sweep.

    Each unit is a ``(cfg, ep, tau, n)`` quadruple. Workers are thread-local
    holders for one ``WebsocketClientPolicy`` plus a per-task LIBERO env cache
    so repeated units targeting the same task do not pay env construction
    cost.
    """

    def __init__(
        self,
        *,
        state_path: Path,
        runner_cfg: _RunnerConfig,
        episodes: Sequence[str],
        max_retries: int = 2,
    ) -> None:
        super().__init__(state_path=state_path, max_retries=max_retries)
        self.runner_cfg = runner_cfg
        self.episodes = list(episodes)
        self._init_lock = threading.Lock()
        self._tls = threading.local()
        # Keep a handle to every opened worker resource so the main thread
        # can close them deterministically at the end of ``run``.
        self._all_clients: List[Any] = []
        self._all_envs: List[Any] = []
        self._resources_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Unit build
    # ------------------------------------------------------------------

    def build_units(self) -> List[UnitState]:
        cfg = self.runner_cfg.cfg
        return [
            UnitState(
                unit_key=Step3PerCycleKey(cfg=cfg, ep=ep, tau=int(tau), n=int(n)).encode()
            )
            for ep in self.episodes
            for tau in self.runner_cfg.tau_grid
            for n in self.runner_cfg.n_grid
        ]

    # ------------------------------------------------------------------
    # Worker-local fixtures
    # ------------------------------------------------------------------

    def _get_client(self):
        client = getattr(self._tls, "client", None)
        if client is not None:
            return client
        from openpi_client.websocket_client_policy import (  # type: ignore[import]
            WebsocketClientPolicy,
        )

        with self._init_lock:
            client = WebsocketClientPolicy(
                host=self.runner_cfg.host, port=self.runner_cfg.port
            )
        self._tls.client = client
        with self._resources_lock:
            self._all_clients.append(client)
        return client

    def _get_env(self, task_id: int, task) -> Any:
        """Return the worker-local env for ``task_id``; rebuild on task switch."""
        cached_task_id = getattr(self._tls, "task_id", None)
        env = getattr(self._tls, "env", None)
        if env is not None and cached_task_id == task_id:
            return env
        # Different task for this worker → close the old env before building a
        # new one. MuJoCo EGL contexts are scarce; never leak.
        if env is not None:
            _close_quietly(getattr(env, "close", None))
            with self._resources_lock:
                try:
                    self._all_envs.remove(env)
                except ValueError:
                    pass
        with self._init_lock:
            new_env = build_libero_env(
                self.runner_cfg.task_suite_name,
                task_id,
                resolution=_LIBERO_ENV_RESOLUTION,
                seed=None,
            )
        self._tls.env = new_env
        self._tls.task_id = task_id
        with self._resources_lock:
            self._all_envs.append(new_env)
        return new_env

    # ------------------------------------------------------------------
    # Unit execution
    # ------------------------------------------------------------------

    def execute_unit(self, unit: UnitState) -> dict:
        key = Step3PerCycleKey.decode(unit.unit_key)
        if key.cfg != self.runner_cfg.cfg:
            raise ValueError(
                f"Unit {unit.unit_key!r} has cfg={key.cfg!r} but runner is "
                f"configured for cfg={self.runner_cfg.cfg!r}"
            )
        task_id, subset_idx = _parse_ep_key(key.ep)

        score_arr = self.runner_cfg.deviate_scores.get(key.ep)
        if score_arr is None:
            raise KeyError(
                f"deviate_score JSON has no entry for {key.ep!r}; "
                "regenerate Step 2 output first."
            )
        is_deviate = score_arr > float(key.tau)
        t_gt_cycles = int(score_arr.shape[0])

        task = _resolve_task(self.runner_cfg.task_suite_name, task_id)
        initial_states = _load_pruned_inits(
            self.runner_cfg.init_states_dir, task.name
        )
        if subset_idx >= len(initial_states):
            raise IndexError(
                f"subset_init_state_idx={subset_idx} out of range for "
                f"{task.name!r} ({len(initial_states)} init states)"
            )
        initial_state = initial_states[subset_idx]
        task_description = str(task.language)

        env = self._get_env(task_id, task)
        client = self._get_client()

        record = self._run_one_unit(
            env=env,
            client=client,
            key=key,
            initial_state=initial_state,
            task_description=task_description,
            is_deviate=is_deviate,
            t_gt_cycles=t_gt_cycles,
            subset_idx=subset_idx,
            task_id=task_id,
        )
        _jsonl_append(self.runner_cfg.results_jsonl, record)
        return record

    # ------------------------------------------------------------------
    # Rollout loop
    # ------------------------------------------------------------------

    def _run_one_unit(
        self,
        *,
        env: Any,
        client: Any,
        key: Step3PerCycleKey,
        initial_state: Any,
        task_description: str,
        is_deviate: np.ndarray,
        t_gt_cycles: int,
        subset_idx: int,
        task_id: int,
    ) -> dict:
        cfg_local = self.runner_cfg
        max_cycles_guard = (
            math.ceil(cfg_local.max_env_steps / max(1, cfg_local.replan_steps))
            + cfg_local.max_cycles_safety
        )

        env.reset()
        obs = env.set_init_state(initial_state)

        # Warm-up: num_steps_wait dummy actions so objects settle, matching
        # examples/libero/main.py. These env steps are NOT counted as cycles.
        done = False
        for _ in range(cfg_local.num_steps_wait):
            obs, _reward, done, _info = env.step(_LIBERO_DUMMY_ACTION)
            if done:
                break

        # Unique episode id per unit so the server's per-connection facade
        # starts from a clean trajectory-history state.
        episode_id = (
            (task_id * 100_000 + subset_idx) * 1000 + int(key.tau) * 10 + int(key.n)
        ) & 0x7FFFFFFF
        episode_started = False
        client.episode_start(
            experiment=cfg_local.experiment_tag,
            task=task_description,
            episode_id=episode_id,
            episode_name="",
        )
        episode_started = True

        burst_remaining = 0
        burst_count = 0
        num_inference = 0
        cycle = 0
        success = False
        # Initialised here so the success-calc after the loop has a defined
        # ``info`` even when the rollout loop never runs (e.g. warmup already
        # returned done=True).
        info: Dict[str, Any] = {}

        try:
            while cycle < max_cycles_guard and not done:
                if (
                    burst_remaining == 0
                    and cycle < t_gt_cycles
                    and bool(is_deviate[cycle])
                ):
                    burst_remaining = int(key.n)
                    burst_count += 1

                if burst_remaining > 0:
                    decision = "skip"
                    burst_remaining -= 1
                    num_inference += 1
                else:
                    decision = "search"

                policy_obs = _obs_env_to_policy(
                    obs, task_description, resize=cfg_local.resize_size
                )
                policy_obs["__gate_decision__"] = decision
                resp = client.infer(policy_obs)
                action_chunk = np.asarray(resp["actions"], dtype=np.float32)
                if action_chunk.shape[0] < cfg_local.replan_steps:
                    raise RuntimeError(
                        f"Policy returned {action_chunk.shape[0]} actions; "
                        f"replan_steps={cfg_local.replan_steps} required."
                    )

                for env_action in action_chunk[: cfg_local.replan_steps]:
                    obs, _reward, done, info = env.step(env_action.tolist())
                    if done:
                        break
                cycle += 1
            # LIBERO semantics (examples/libero/main.py::_run_episode): done
            # fires only on task completion — timeout is enforced by the outer
            # max_cycles_guard, never by the env. ``info['success']`` is kept
            # as a belt-and-suspenders signal in case a future env backend
            # sets it independently.
            success = bool(done) or bool(info.get("success", False))
        finally:
            if episode_started:
                try:
                    client.episode_end(success=success)
                except Exception as e:  # pragma: no cover - defensive
                    logger.warning(
                        "episode_end raised for %s: %s", key.encode(), e
                    )

        total_cycles = cycle
        return {
            "cfg": key.cfg,
            "ep": key.ep,
            "tau": int(key.tau),
            "n": int(key.n),
            "success": bool(success),
            "total_cycles": int(total_cycles),
            "num_inference_cycles": int(num_inference),
            "inference_ratio": (
                float(num_inference) / float(total_cycles)
                if total_cycles > 0 else 0.0
            ),
            "T_gt_cycles": int(t_gt_cycles),
            "burst_count": int(burst_count),
        }

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close_all(self) -> None:
        """Close every env / client opened by any worker."""
        with self._resources_lock:
            envs = list(self._all_envs)
            clients = list(self._all_clients)
            self._all_envs.clear()
            self._all_clients.clear()
        for env in envs:
            _close_quietly(getattr(env, "close", None))
        for client in clients:
            _close_quietly(getattr(client, "close", None))


def _close_quietly(close_fn) -> None:
    if callable(close_fn):
        try:
            close_fn()
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("close() raised: %s", e)


def _jsonl_append(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _JSONL_LOCK:
        with path.open("a") as fh:
            fh.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_grid(text: str) -> List[int]:
    return [int(tok) for tok in text.split(",") if tok.strip()]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 3 per-cycle policy runner.")
    parser.add_argument("--cfg", required=True, help="Cache config id, e.g. clip_w7_d4")
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument(
        "--yaml",
        required=True,
        type=Path,
        help="Path to the Step 3 cache YAML (gate.type=client_controlled).",
    )
    parser.add_argument(
        "--deviate-score-json",
        required=True,
        type=Path,
        help=(
            "deviate_score_{cfg}.json produced by compute_deviate_scores.py. "
            "Its keys (subset-space task_X/episode_Y) drive the Step 3 "
            "episode list — they are exactly the per-cfg Step 1a failure set."
        ),
    )
    parser.add_argument(
        "--init-states-dir",
        required=True,
        type=Path,
        help="Step 1b pruned init-states directory.",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        type=Path,
        help="Output root for run_state.json + results.jsonl.",
    )
    parser.add_argument(
        "--task-suite-name", default="libero_spatial",
        choices=sorted(_MAX_STEPS_BY_SUITE.keys()),
    )
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--tau-grid", default="3,5,7,10")
    parser.add_argument("--n-grid", default="1,2,3,5,10")
    parser.add_argument("--replan-steps", type=int, default=_DEFAULT_REPLAN_STEPS)
    parser.add_argument("--num-steps-wait", type=int, default=_DEFAULT_NUM_STEPS_WAIT)
    parser.add_argument("--resize-size", type=int, default=_DEFAULT_RESIZE)
    parser.add_argument(
        "--max-cycles-safety", type=int, default=5,
        help="Extra cycles added to ceil(max_env_steps/replan_steps) runaway guard.",
    )
    parser.add_argument(
        "--experiment-tag", default="trajectory_deviation_step3",
        help="Forwarded to client.episode_start(experiment=...).",
    )
    parser.add_argument(
        "--skip-load-cache-config", action="store_true",
        help="Skip the load_cache_config RPC (useful if the server already "
             "has the bundle loaded, or for dry runs).",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--log-level", default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.num_workers > _MAX_WORKERS_CAP:
        raise SystemExit(
            f"--num-workers={args.num_workers} exceeds MuJoCo EGL cap "
            f"of {_MAX_WORKERS_CAP}; reduce or add a second server."
        )

    tau_grid = _parse_grid(args.tau_grid)
    n_grid = _parse_grid(args.n_grid)
    if not tau_grid or not n_grid:
        raise SystemExit("--tau-grid and --n-grid must each be non-empty")

    deviate_scores = _load_deviate_scores(args.deviate_score_json)
    # Step 2 keys ARE the Step 3 episode list (subset-space, per-cfg failures).
    episodes = sorted(deviate_scores.keys())
    if not episodes:
        raise SystemExit(
            f"{args.deviate_score_json} has no episodes; rerun Step 2 first."
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    state_path = args.out_dir / "run_state.json"
    results_jsonl = args.out_dir / "results.jsonl"

    if not args.skip_load_cache_config:
        server_url = f"ws://{args.host}:{args.port}"
        send_load_cache_config(server_url, args.yaml)

    runner_cfg = _RunnerConfig(
        cfg=args.cfg,
        host=args.host,
        port=args.port,
        task_suite_name=args.task_suite_name,
        init_states_dir=args.init_states_dir,
        deviate_scores=deviate_scores,
        tau_grid=tau_grid,
        n_grid=n_grid,
        max_cycles_safety=int(args.max_cycles_safety),
        replan_steps=int(args.replan_steps),
        num_steps_wait=int(args.num_steps_wait),
        max_env_steps=_MAX_STEPS_BY_SUITE[args.task_suite_name],
        resize_size=int(args.resize_size),
        results_jsonl=results_jsonl,
        experiment_tag=args.experiment_tag,
    )

    runner = Step3PerCycleRunner(
        state_path=state_path,
        runner_cfg=runner_cfg,
        episodes=episodes,
    )
    start_time = time.monotonic()
    try:
        runner.parallel_run(num_workers=max(1, args.num_workers), resume=args.resume)
    finally:
        runner.close_all()
    logger.info(
        "Step 3 cfg=%s complete in %.1fs (%d episodes × %d tau × %d n = %d units)",
        args.cfg, time.monotonic() - start_time, len(episodes),
        len(tau_grid), len(n_grid), len(episodes) * len(tau_grid) * len(n_grid),
    )


if __name__ == "__main__":
    main()
