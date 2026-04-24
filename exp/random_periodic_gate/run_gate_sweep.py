"""Random & Periodic gate sweep runner (batch-dir driven).

Iterates every rendered YAML under ``--batch-dir``, switches the server's
cache bundle via ``send_load_cache_config`` before each grid point, and
runs the full LIBERO suite (500 eps for libero_spatial) with up to 5
workers (MuJoCo EGL cap). A separate ``BaseRunState`` instance is used
per YAML so resume is at ``(yaml_basename, task_id, init_idx)`` granularity.

Inference cost metric is computed entirely on the runner side (no
framework plumbing was added for this experiment):

  - PeriodicGate: ``num_inference_cycles`` is derived from
    ``(cache_len, inference_len, total_cycles)`` via the closed-form
    formula; JSONL field ``inference_ratio_source = "derived"``.
  - RandomGate: ``num_inference_cycles_estimated = round(total_cycles *
    p_inference)`` is recorded as an expected value; JSONL field
    ``inference_ratio_source = "expected"``.

Reference CLI::

    uv run python -m exp.random_periodic_gate.run_gate_sweep \\
        --batch-dir exp/random_periodic_gate/config/batch1 \\
        --host host_1 --port 8001 \\
        --out-root exp/random_periodic_gate/data \\
        --num-workers 5

Plan: ``logs/random_periodic_gate_plan.log.md`` §5.6.
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
from typing import Any, Iterable, Sequence

import numpy as np

if __package__ in {None, ""}:   # pragma: no cover - CLI bootstrap
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from exp.common._cache_config_rpc import send_load_cache_config
from exp.common._run_state_base import BaseRunState, UnitState
from exp.trajectory_deviation._libero_env import build_libero_env

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants — copied from examples/libero/main.py to stay byte-equivalent
# with Step 1a / Step 1b baseline rollouts.
# ---------------------------------------------------------------------------

_LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
_LIBERO_ENV_RESOLUTION = 256
_DEFAULT_RESIZE = 224
_DEFAULT_REPLAN_STEPS = 5
_DEFAULT_NUM_STEPS_WAIT = 10

_MAX_STEPS_BY_SUITE = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}
_MAX_WORKERS_CAP = 5

_JSONL_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Slug parsing (mirrors generate_batches.py naming).
# ---------------------------------------------------------------------------


_PERIODIC_SLUG_RE = re.compile(r"^periodic_k(?P<k>\d+)_n(?P<n>\d+)$")
_RANDOM_SLUG_RE = re.compile(r"^random_p(?P<p>\d+p\d+)_s(?P<s>\d+)$")


@dataclass(frozen=True)
class GatePointMeta:
    """Static metadata about the gate configuration baked into one YAML."""

    slug: str
    gate_type: str        # "periodic" | "random"
    gate_params: dict[str, Any]
    seed: int | None      # None for periodic, int for random


def parse_slug(slug: str) -> GatePointMeta:
    mp = _PERIODIC_SLUG_RE.match(slug)
    if mp:
        k = int(mp.group("k"))
        n = int(mp.group("n"))
        return GatePointMeta(
            slug=slug, gate_type="periodic",
            gate_params={"cache_len": k, "inference_len": n},
            seed=None,
        )
    mr = _RANDOM_SLUG_RE.match(slug)
    if mr:
        # p-field encoding: "0p30" -> 0.30
        p_str = mr.group("p").replace("p", ".", 1)
        p = float(p_str)
        s = int(mr.group("s"))
        return GatePointMeta(
            slug=slug, gate_type="random",
            gate_params={"p_inference": p},
            seed=s,
        )
    raise ValueError(f"unrecognized slug: {slug!r}")


# ---------------------------------------------------------------------------
# Episode enumeration (plan §5.6 canonical mapping).
# ---------------------------------------------------------------------------


def enumerate_full_suite(task_suite_name: str) -> list[tuple[int, int]]:
    """Return ``[(task_id, init_idx)]`` for every LIBERO default init state.

    ``init_idx`` is the index into ``task_suite.get_task_init_states(task_id)``,
    matching the ``orig_init_state_idx`` column in
    ``exp/trajectory_deviation/data/cache_eval_results.json`` (the
    baseline join key).
    """
    from libero.libero.benchmark import get_benchmark_dict  # type: ignore[import]

    suite = get_benchmark_dict()[task_suite_name]()
    out: list[tuple[int, int]] = []
    for task_id in range(suite.n_tasks):
        init_states = suite.get_task_init_states(task_id)
        for init_idx in range(len(init_states)):
            out.append((task_id, init_idx))
    return out


def _resolve_task(task_suite_name: str, task_id: int):
    from libero.libero.benchmark import get_benchmark_dict  # type: ignore[import]

    return get_benchmark_dict()[task_suite_name]().get_task(int(task_id))


# ---------------------------------------------------------------------------
# Observation helpers (bit-for-bit with Step 3 runner).
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
    env_obs: dict[str, Any], task_description: str, *, resize: int = _DEFAULT_RESIZE,
) -> dict[str, Any]:
    from openpi_client import image_tools  # type: ignore[import]

    img = np.ascontiguousarray(env_obs["agentview_image"][::-1, ::-1])
    wrist_img = np.ascontiguousarray(env_obs["robot0_eye_in_hand_image"][::-1, ::-1])
    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, resize, resize))
    wrist_img = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(wrist_img, resize, resize)
    )
    state = np.concatenate(
        (
            env_obs["robot0_eef_pos"],
            _quat2axisangle(env_obs["robot0_eef_quat"]),
            env_obs["robot0_gripper_qpos"],
        )
    )
    return {
        "observation/image": img,
        "observation/wrist_image": wrist_img,
        "observation/state": state,
        "prompt": str(task_description),
    }


# ---------------------------------------------------------------------------
# Cost metric derivation (runner-side only).
# ---------------------------------------------------------------------------


def derive_periodic_num_inference(total_cycles: int, cache_len: int, inference_len: int) -> int:
    """Closed-form formula: decision at cycle c is search iff
    ``c % (k+n) < k``. Skips are the complement.
    """
    period = cache_len + inference_len
    return sum(1 for i in range(int(total_cycles)) if i % period >= cache_len)


def estimated_random_num_inference(total_cycles: int, p_inference: float) -> int:
    return int(round(total_cycles * float(p_inference)))


# ---------------------------------------------------------------------------
# Unit-key encode/decode (3-tuple; no extension to exp/common/_unit_key.py).
# ---------------------------------------------------------------------------


def _encode_unit(yaml_basename: str, task_id: int, init_idx: int) -> str:
    return f"{yaml_basename}:{int(task_id)}:{int(init_idx)}"


_UNIT_KEY_RE = re.compile(r"^(?P<name>[^:]+):(?P<t>\d+):(?P<i>\d+)$")


def _decode_unit(key: str) -> tuple[str, int, int]:
    m = _UNIT_KEY_RE.match(key)
    if not m:
        raise ValueError(f"Malformed unit_key: {key!r}")
    return m.group("name"), int(m.group("t")), int(m.group("i"))


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


@dataclass
class _YamlRunConfig:
    yaml_path: Path
    gate_meta: GatePointMeta
    cfg: str                 # e.g. "clip_w7_d4" — parsed from batch-dir cfg
    host: str
    port: int
    task_suite_name: str
    episodes: Sequence[tuple[int, int]]
    replan_steps: int
    num_steps_wait: int
    max_env_steps: int
    max_cycles_safety: int
    resize_size: int
    results_jsonl: Path
    experiment_tag: str


def _jsonl_append(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _JSONL_LOCK:
        with path.open("a") as fh:
            fh.write(json.dumps(record) + "\n")


class GateSweepRunner(BaseRunState):
    """``BaseRunState`` subclass for a single YAML (one (cfg, gate) combo).

    Units are ``(yaml_basename, task_id, init_idx)`` triples. Workers hold
    a thread-local ``WebsocketClientPolicy`` plus a per-task env cache so
    same-task units reuse the env across rollouts.
    """

    def __init__(
        self, *,
        state_path: Path,
        yaml_cfg: _YamlRunConfig,
        max_retries: int = 2,
    ) -> None:
        super().__init__(state_path=state_path, max_retries=max_retries)
        self.yaml_cfg = yaml_cfg
        self._init_lock = threading.Lock()
        self._tls = threading.local()
        self._all_clients: list[Any] = []
        self._all_envs: list[Any] = []
        self._resources_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Unit build
    # ------------------------------------------------------------------

    def build_units(self) -> list[UnitState]:
        yaml_base = self.yaml_cfg.yaml_path.name
        return [
            UnitState(unit_key=_encode_unit(yaml_base, tid, iid))
            for (tid, iid) in self.yaml_cfg.episodes
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
            t0 = time.monotonic()
            client = WebsocketClientPolicy(
                host=self.yaml_cfg.host, port=self.yaml_cfg.port
            )
            logger.info(
                "[%s] connected WebsocketClientPolicy in %.2fs",
                threading.current_thread().name, time.monotonic() - t0,
            )
        self._tls.client = client
        with self._resources_lock:
            self._all_clients.append(client)
        return client

    def _get_env(self, task_id: int) -> Any:
        cached_task_id = getattr(self._tls, "task_id", None)
        env = getattr(self._tls, "env", None)
        if env is not None and cached_task_id == task_id:
            return env
        if env is not None:
            _close_quietly(getattr(env, "close", None))
            with self._resources_lock:
                try:
                    self._all_envs.remove(env)
                except ValueError:
                    pass
        with self._init_lock:
            t0 = time.monotonic()
            new_env = build_libero_env(
                self.yaml_cfg.task_suite_name,
                int(task_id),
                resolution=_LIBERO_ENV_RESOLUTION,
                seed=None,
            )
            logger.info(
                "[%s] built env (task_id=%d) in %.2fs",
                threading.current_thread().name, task_id, time.monotonic() - t0,
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
        yaml_base, task_id, init_idx = _decode_unit(unit.unit_key)
        if yaml_base != self.yaml_cfg.yaml_path.name:
            raise ValueError(
                f"Unit {unit.unit_key!r} yaml={yaml_base!r} does not match "
                f"runner yaml={self.yaml_cfg.yaml_path.name!r}"
            )
        task = _resolve_task(self.yaml_cfg.task_suite_name, task_id)
        initial_states = task_init_states(self.yaml_cfg.task_suite_name, task_id)
        if init_idx >= len(initial_states):
            raise IndexError(
                f"init_idx={init_idx} out of range for task_id={task_id} "
                f"({len(initial_states)} LIBERO default init states)"
            )
        initial_state = initial_states[init_idx]
        task_description = str(task.language)

        env = self._get_env(task_id)
        client = self._get_client()

        record = self._run_one_unit(
            env=env,
            client=client,
            task_id=task_id,
            init_idx=init_idx,
            initial_state=initial_state,
            task_description=task_description,
        )
        _jsonl_append(self.yaml_cfg.results_jsonl, record)
        return record

    # ------------------------------------------------------------------
    # Rollout loop — standard LIBERO rollout; gate decision is server-side.
    # ------------------------------------------------------------------

    def _run_one_unit(
        self, *,
        env: Any, client: Any,
        task_id: int, init_idx: int,
        initial_state: Any, task_description: str,
    ) -> dict:
        cfg_local = self.yaml_cfg
        max_cycles_guard = (
            math.ceil(cfg_local.max_env_steps / max(1, cfg_local.replan_steps))
            + cfg_local.max_cycles_safety
        )

        env.reset()
        obs = env.set_init_state(initial_state)

        done = False
        for _ in range(cfg_local.num_steps_wait):
            obs, _reward, done, _info = env.step(_LIBERO_DUMMY_ACTION)
            if done:
                break

        # Stable per-unit episode_id so the server's per-connection facade
        # gets a clean trajectory-history boundary. 5 digits init_idx + 3
        # digits task_id fits a signed 32-bit int comfortably.
        assert 0 <= init_idx < 100_000, f"init_idx={init_idx} overflows episode_id layout"
        assert 0 <= task_id < 1_000, f"task_id={task_id} overflows episode_id layout"
        episode_id = (task_id * 100_000 + init_idx) & 0x7FFFFFFF
        ep_key = f"task_{task_id}/init_{init_idx}"

        client.episode_start(
            experiment=cfg_local.experiment_tag,
            task=task_description,
            episode_id=episode_id,
            episode_name=ep_key,
        )
        episode_started = True

        cycle = 0
        success = False
        info: dict[str, Any] = {}

        try:
            while cycle < max_cycles_guard and not done:
                policy_obs = _obs_env_to_policy(
                    obs, task_description, resize=cfg_local.resize_size
                )
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
            # Mirror examples/libero/main.py::_run_episode success semantics.
            success = bool(done) or bool(info.get("success", False))
        finally:
            if episode_started:
                try:
                    client.episode_end(success=success)
                except Exception as e:  # pragma: no cover - defensive
                    logger.warning("episode_end raised for %s: %s", ep_key, e)

        total_cycles = cycle
        gate_meta = cfg_local.gate_meta
        base = {
            "cfg": cfg_local.cfg,
            "gate_type": gate_meta.gate_type,
            "gate_params": dict(gate_meta.gate_params),
            "seed": gate_meta.seed,
            "task_id": int(task_id),
            "init_idx": int(init_idx),
            "ep_key": ep_key,
            "success": bool(success),
            "total_cycles": int(total_cycles),
        }
        if gate_meta.gate_type == "periodic":
            num_inf = derive_periodic_num_inference(
                total_cycles,
                int(gate_meta.gate_params["cache_len"]),
                int(gate_meta.gate_params["inference_len"]),
            )
            base["num_inference_cycles"] = int(num_inf)
            base["inference_ratio"] = (
                float(num_inf) / float(total_cycles) if total_cycles > 0 else 0.0
            )
            base["inference_ratio_source"] = "derived"
        else:  # random
            p = float(gate_meta.gate_params["p_inference"])
            num_inf = estimated_random_num_inference(total_cycles, p)
            base["num_inference_cycles_estimated"] = int(num_inf)
            base["inference_ratio"] = p
            base["inference_ratio_source"] = "expected"
        return base

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close_all(self) -> None:
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


# ---------------------------------------------------------------------------
# cfg inference from base YAML + caching init-state lookup.
# ---------------------------------------------------------------------------


_INIT_STATE_CACHE: dict[tuple[str, int], Any] = {}
_INIT_STATE_LOCK = threading.Lock()


def task_init_states(task_suite_name: str, task_id: int):
    """Return (and memoize) LIBERO default init states for one task."""
    key = (task_suite_name, int(task_id))
    with _INIT_STATE_LOCK:
        cached = _INIT_STATE_CACHE.get(key)
        if cached is not None:
            return cached
    from libero.libero.benchmark import get_benchmark_dict  # type: ignore[import]

    states = get_benchmark_dict()[task_suite_name]().get_task_init_states(int(task_id))
    with _INIT_STATE_LOCK:
        _INIT_STATE_CACHE[key] = states
    return states


def _infer_cfg_from_batch(batch_dir: Path) -> str:
    """Map batch directory name (batch1..batch6) to its keybuilder cfg.

    batch1/2 -> clip_w7_d4; batch3/4 -> spatial16_w8_d4; batch5/6 -> max_pool_w3_d5.
    """
    name = batch_dir.name
    m = re.match(r"^batch(\d+)$", name)
    if not m:
        raise ValueError(f"batch dir {batch_dir!r} does not match batch<N>")
    n = int(m.group(1))
    if n in (1, 2):
        return "clip_w7_d4"
    if n in (3, 4):
        return "spatial16_w8_d4"
    if n in (5, 6):
        return "max_pool_w3_d5"
    raise ValueError(f"unexpected batch index: {n}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Random & Periodic gate sweep runner")
    p.add_argument("--batch-dir", required=True, type=Path,
                   help="Directory containing the 19 rendered YAMLs for one batch.")
    p.add_argument("--host", required=True)
    p.add_argument("--port", type=int, required=True)
    p.add_argument(
        "--out-root", type=Path, default=Path("exp/random_periodic_gate/data"),
        help="Root for per-batch run_state.json / results.jsonl",
    )
    p.add_argument("--task-suite-name", default="libero_spatial",
                   choices=sorted(_MAX_STEPS_BY_SUITE.keys()))
    p.add_argument("--num-workers", type=int, default=1)
    p.add_argument("--replan-steps", type=int, default=_DEFAULT_REPLAN_STEPS)
    p.add_argument("--num-steps-wait", type=int, default=_DEFAULT_NUM_STEPS_WAIT)
    p.add_argument("--resize-size", type=int, default=_DEFAULT_RESIZE)
    p.add_argument("--max-cycles-safety", type=int, default=5)
    p.add_argument("--experiment-tag", default="random_periodic_gate_sweep")
    p.add_argument("--skip-load-cache-config", action="store_true",
                   help="Skip the load_cache_config RPC (dry-run / resume).")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--log-level", default="INFO",
                   choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return p.parse_args(argv if argv is None else list(argv))


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.num_workers > _MAX_WORKERS_CAP:
        raise SystemExit(
            f"--num-workers={args.num_workers} exceeds MuJoCo EGL cap of {_MAX_WORKERS_CAP}"
        )

    batch_dir: Path = args.batch_dir.resolve()
    if not batch_dir.is_dir():
        raise SystemExit(f"--batch-dir not found or not a directory: {batch_dir}")
    yaml_paths = sorted(batch_dir.glob("*.yaml"))
    if not yaml_paths:
        raise SystemExit(f"no YAML files under {batch_dir}")

    cfg = _infer_cfg_from_batch(batch_dir)
    out_dir = args.out_root / batch_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    episodes = enumerate_full_suite(args.task_suite_name)
    server_url = f"ws://{args.host}:{args.port}"

    prev_version: int | None = None
    for yaml_path in yaml_paths:
        slug = yaml_path.stem
        gate_meta = parse_slug(slug)

        if not args.skip_load_cache_config:
            new_version = send_load_cache_config(server_url, yaml_path)
            if prev_version is not None and new_version <= prev_version:
                raise RuntimeError(
                    f"cache bundle version did not increase after loading "
                    f"{yaml_path}: prev={prev_version} new={new_version}"
                )
            prev_version = new_version

        state_path = out_dir / f"run_state_{slug}.json"
        results_jsonl = out_dir / "results.jsonl"

        yaml_cfg = _YamlRunConfig(
            yaml_path=yaml_path,
            gate_meta=gate_meta,
            cfg=cfg,
            host=args.host,
            port=int(args.port),
            task_suite_name=args.task_suite_name,
            episodes=episodes,
            replan_steps=int(args.replan_steps),
            num_steps_wait=int(args.num_steps_wait),
            max_env_steps=_MAX_STEPS_BY_SUITE[args.task_suite_name],
            max_cycles_safety=int(args.max_cycles_safety),
            resize_size=int(args.resize_size),
            results_jsonl=results_jsonl,
            experiment_tag=args.experiment_tag,
        )
        runner = GateSweepRunner(state_path=state_path, yaml_cfg=yaml_cfg)
        try:
            if args.num_workers > 1:
                runner.parallel_run(
                    num_workers=int(args.num_workers),
                    resume=bool(args.resume),
                )
            else:
                runner.run(resume=bool(args.resume))
        finally:
            runner.close_all()
        logger.info("completed %s", yaml_path.name)


if __name__ == "__main__":
    main()
