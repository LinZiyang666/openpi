"""Step 3 Spawn runner (plan §11 + §18.A3.4 + §19.4 + §20.R4).

For each deviate-score cycle flagged as "off manifold", we ask:

    If we teleport the env to the GT ``sim_state`` at cycle ``s + n`` and
    let a pure cache-driven rollout continue from there, does the policy
    recover and still reach the goal?

Units are indexed by ``(cfg, ep, s, n, k_idx)``:

- ``cfg``   — cache-config id (e.g. ``clip_w7_d4``).
- ``ep``    — GT episode (e.g. ``task_3/episode_7``) — same name Step 1b
  stored the client-side HDF5 under AND the server stored the ``--collect``
  HDF5 under (§20.R4.1 contract: ``collected/{task_suite}/{ep}.h5``).
- ``s``     — intervention cycle index (the deviate-score cycle we target).
- ``n``     — how many cycles after ``s`` we let the GT play out before
  teleporting (``n ∈ {1, 3, 5, 10, 20}``).
- ``k_idx`` — which of the top-k highest-scoring cycles this unit
  represents (strategy A per §11.1: one independent spawn per point).

Contract (the core "why each step exists"):

- **Env teleport** uses ``env.set_init_state(sim_state)`` + restoring
  ``timestep`` / ``cur_time``. §11.0 / §19.8 Phase-0 smoke scripts must
  have validated the round-trip before this runner is allowed to run.
- **Prefill** (plan §19.4): the D-1 prior cycles' MODEL-space clean
  action chunks are fetched from the server's ``--collect`` HDF5
  (``step_{t:04d}/clean_action``). Env-space 7D actions are explicitly
  NOT accepted by ``prefill_trajectory`` (plan §7 docstring). Prefill
  observations come from the client-side GT HDF5 (env-space post-transform
  obs), exactly as ``compute_deviate_scores.load_gt_episode`` reads them.
- **Pure cache rollout** drives the env with ``env.step(action[0])`` per
  ``client.infer`` call. Loop exits on ``info["success"]`` (goal reached),
  ``done``, or a per-unit ``max_spawn_env_steps`` budget.
- **No HDF5 capture** for spawn runs — the server-side ``CollectionPolicy``
  is driven by ``episode_start(experiment=…, episode_name="")``; the empty
  name tells the collector not to pin a filename. We only care about the
  success flag.

Test hooks are kept first-class: ``_SpawnCommon`` accepts a
``client_factory`` and ``env_factory`` so unit tests can exercise the
runner without a real LIBERO env or websocket server. Nothing about the
runner depends on actually connecting to a server at import time (``libero``
/ ``h5py`` / ``torch`` are imported lazily inside the helpers that need
them).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from exp._cache_config_rpc import send_load_cache_config
from exp._run_state_base import BaseRunState, UnitState

logger = logging.getLogger(__name__)

# --- Tunable defaults (kept in one place so CLI surface is obvious) -------

DEFAULT_N_GRID: Tuple[int, ...] = (1, 3, 5, 10, 20)
DEFAULT_K_GRID: Tuple[int, ...] = (1, 3, 5)
# Budget for post-teleport env.step calls. Set generously — the point is to
# see if the policy recovers, not to bound experiment time per unit. Actual
# GT episodes rarely exceed a few hundred env steps; 300 is ~2x margin.
DEFAULT_MAX_SPAWN_ENV_STEPS: int = 300
# Default resize matches examples/libero/main.py args.resize_size.
_DEFAULT_RESIZE: int = 224


# --- Unit-key helpers (public so tests can lock the exact format) --------


_UNIT_KEY_RE = re.compile(r"^(?P<cfg>[^:]+):(?P<ep>[^:]+):s(?P<s>\d+):n(?P<n>\d+):k(?P<k>\d+)$")


def build_unit_key(cfg: str, ep: str, s: int, n: int, k_idx: int) -> str:
    """Canonical unit-key. Stable against reorderings of the k-list.

    ``ep`` carries a ``/`` separator (``task_X/episode_Y``) so the five
    fields remain parseable with a strict colon split.
    """
    return f"{cfg}:{ep}:s{int(s)}:n{int(n)}:k{int(k_idx)}"


def parse_unit_key(key: str) -> Tuple[str, str, int, int, int]:
    m = _UNIT_KEY_RE.match(key)
    if not m:
        raise ValueError(f"Malformed spawn unit_key: {key!r}")
    return (
        m.group("cfg"),
        m.group("ep"),
        int(m.group("s")),
        int(m.group("n")),
        int(m.group("k")),
    )


# --- Point-selection strategies (SpawnRunner = top-k; Baselines = others)


def _top_k_points(scores: Sequence[float], k: int) -> List[int]:
    """Indices of the ``k`` highest scores, descending.

    We allow ``k`` to exceed ``len(scores)`` — callers (e.g. build_units) iterate
    ``k_idx in range(min(k, len(scores)))`` so the truncation is harmless.
    """
    k = min(int(k), len(scores))
    # np.argsort is stable in descending order via negation.
    order = np.argsort(-np.asarray(scores))
    return [int(i) for i in order[:k]]


def _random_k_points(T: int, k: int, *, seed: int) -> List[int]:
    """Deterministic baseline: `k` cycle indices drawn uniformly without
    replacement. ``seed`` threads through so every BaselineRunner unit is
    reproducible across resumes."""
    rng = np.random.default_rng(int(seed))
    k = min(int(k), int(T))
    return rng.choice(int(T), size=k, replace=False).tolist()


def _equidistant_k_points(T: int, k: int) -> List[int]:
    """Deterministic baseline: evenly spaced cycle indices in ``[0, T)``."""
    T = int(T)
    k = min(int(k), T)
    if k <= 0:
        return []
    # ``linspace`` with endpoint=False gives k points at 0, T/k, 2T/k, …
    return [int(x) for x in np.floor(np.linspace(0, T, num=k, endpoint=False))]


# --- Obs transform for env→policy (mirrors examples/libero/main.py:155-185)


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """Direct copy of ``examples/libero/main.py:_quat2axisangle``.

    Kept here (not imported) so ``exp/`` has no dependency on
    ``examples/``. The function is tiny and stable; if the original ever
    diverges we add a smoke test to catch the drift.
    """
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
    """Convert raw LIBERO env obs → ``client.infer`` input dict.

    Mirrors ``examples/libero/main.py`` exactly (flip images + resize_with_pad
    + concat state). ``openpi_client.image_tools`` is imported lazily so
    pure-math unit tests don't have to install that package.
    """
    from openpi_client import image_tools  # type: ignore[import]

    img = np.ascontiguousarray(env_obs["agentview_image"][::-1, ::-1])
    wrist_img = np.ascontiguousarray(env_obs["robot0_eye_in_hand_image"][::-1, ::-1])
    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, resize, resize))
    wrist_img = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist_img, resize, resize))
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


# --- Config-id parsing (D = trajectory depth from the bundle name) -------


_D_RE = re.compile(r"_d(?P<d>\d+)(?:$|[_.])")


def infer_d_from_cfg(cfg: str) -> int:
    """Parse trajectory depth ``D`` from a config id like ``clip_w7_d4``.

    We could read it from the YAML directly, but that forces
    ``SpawnRunner`` to know how to parse every cache config schema. Keeping
    it to a naming convention (already enforced by §18.B4.1) lets Step 3
    stay decoupled from the server-side cache module.
    """
    m = _D_RE.search(cfg)
    if not m:
        raise ValueError(
            f"Cannot infer D from cfg={cfg!r}: expected a '_dN' suffix "
            f"(e.g. 'clip_w7_d4'). Pass --D explicitly to override."
        )
    return int(m.group("d"))


# --- Shared common block (keeps SpawnRunner / BaselineRunner ctors tiny)


@dataclass
class _SpawnCommon:
    """Config shared by SpawnRunner and BaselineRunner.

    ``client_factory`` and ``env_factory`` are the test seams. In production
    both are ``None`` and we fall back to real ``WebsocketClientPolicy`` /
    LIBERO ``OffScreenRenderEnv`` constructors.
    """

    cfg: str
    gt_dir: Path
    collected_dir: Path
    task_suite_name: str
    D: int
    out_dir: Path
    host: str
    port: int
    n_grid: Tuple[int, ...] = DEFAULT_N_GRID
    k_grid: Tuple[int, ...] = DEFAULT_K_GRID
    max_spawn_env_steps: int = DEFAULT_MAX_SPAWN_ENV_STEPS
    # Per-cfg scores dict (episode → {deviate_score, cache_l2, background_l2})
    # loaded from ``deviate_score_{cfg}.json``.
    scores: Dict[str, Dict[str, List[float]]] = field(default_factory=dict)
    client_factory: Optional[Callable[[str, int], Any]] = None
    env_factory: Optional[Callable[[int, str], Any]] = None
    # Seed used by BaselineRunner's random strategy (SpawnRunner ignores it).
    random_seed: int = 0

    def make_client(self) -> Any:
        if self.client_factory is not None:
            return self.client_factory(self.host, self.port)
        from openpi_client.websocket_client_policy import WebsocketClientPolicy  # type: ignore[import]

        return WebsocketClientPolicy(host=self.host, port=self.port)

    def make_env(self, task_id: int, task_suite: str) -> Any:
        if self.env_factory is not None:
            return self.env_factory(task_id, task_suite)
        # Lazy import: unit tests don't need LIBERO installed.
        from libero.libero.benchmark import get_benchmark_dict  # type: ignore[import]
        from libero.libero.envs import OffScreenRenderEnv  # type: ignore[import]

        suite = get_benchmark_dict()[task_suite]()
        task = suite.get_task(int(task_id))
        # 256×256 matches examples/libero/main.py LIBERO_ENV_RESOLUTION, so the
        # teleport-then-resize pipeline produces byte-identical obs to GT.
        return OffScreenRenderEnv(
            bddl_file_name=task.bddl_file,
            camera_heights=256,
            camera_widths=256,
        )


# --- Runner base: shared teleport + rollout logic ------------------------


class _BaseSpawnRunner(BaseRunState):
    """Factored base with the common ``execute_unit`` internals.

    SpawnRunner and BaselineRunner differ only in how they pick cycle
    indices; the teleport / prefill / rollout mechanics are identical.
    """

    def __init__(
        self,
        *,
        state_path: Path,
        common: _SpawnCommon,
        max_retries: int = 2,
    ) -> None:
        super().__init__(state_path=state_path, max_retries=max_retries)
        self.common = common

    # ---- Subclass hook: enumerate (episode, n, k_idx) triples ----

    def _iter_targets(self) -> List[Tuple[str, int, int, int]]:
        """Yield ``(episode, n, k_idx, cycle_s)`` for every unit this
        runner should dispatch.

        ``cycle_s`` is the cycle index the runner will eventually teleport
        ``n`` cycles *after* (the teleport target is ``s + n`` per §11.1).
        """
        raise NotImplementedError

    # ---- BaseRunState hooks ----

    def build_units(self) -> List[UnitState]:
        return [
            UnitState(unit_key=build_unit_key(self.common.cfg, ep, s, n, k_idx))
            for ep, n, k_idx, s in self._iter_targets()
        ]

    def execute_unit(self, unit: UnitState) -> dict:
        cfg, ep, s, n, k_idx = parse_unit_key(unit.unit_key)
        if cfg != self.common.cfg:
            # Defensive: two SpawnRunners sharing a state file would be a bug.
            raise ValueError(
                f"cfg mismatch: unit={cfg!r} runner={self.common.cfg!r}"
            )
        return _execute_spawn_unit(
            common=self.common, ep=ep, s=s, n=n, k_idx=k_idx,
        )


class SpawnRunner(_BaseSpawnRunner):
    """Strategy A (§11.1): spawn at the top-k highest-scoring cycles.

    ``k_idx`` is a positional index into the top-k list (0 = highest
    score). Running with ``k=5`` creates 5 units per episode — each one
    independently teleports to a distinct point; we never stack two
    teleports into one run.
    """

    def _iter_targets(self) -> List[Tuple[str, int, int, int]]:
        out: List[Tuple[str, int, int, int]] = []
        for ep, ep_scores in self.common.scores.items():
            scores = ep_scores.get("deviate_score") or []
            if not scores:
                logger.warning("SpawnRunner: %s has no deviate_score — skipping", ep)
                continue
            top = _top_k_points(scores, max(self.common.k_grid))
            for k_max in self.common.k_grid:
                for k_idx, s in enumerate(top[: int(k_max)]):
                    for n in self.common.n_grid:
                        out.append((ep, int(n), int(k_idx), int(s)))
        # De-duplicate: distinct k_max values collapse to the same
        # (ep, s, n, k_idx) tuple whenever the smaller k_max is a prefix.
        # The state file only needs one row per unique point.
        seen: Dict[str, Tuple[str, int, int, int]] = {}
        for row in out:
            ep, n, k_idx, s = row
            key = build_unit_key(self.common.cfg, ep, s, n, k_idx)
            seen.setdefault(key, row)
        return list(seen.values())


class BaselineRunner(_BaseSpawnRunner):
    """Reference baselines (§6.3 / §11.2): random or equidistant cycles.

    ``strategy`` ∈ {"random", "equidistant"}. The index meaning flips:
    for baselines ``k_idx`` identifies the baseline-list position (the
    whole list is drawn anew per episode); not a top-k ranking.
    """

    def __init__(
        self,
        *,
        state_path: Path,
        common: _SpawnCommon,
        strategy: str,
        max_retries: int = 2,
    ) -> None:
        if strategy not in ("random", "equidistant"):
            raise ValueError(f"Unknown baseline strategy: {strategy!r}")
        super().__init__(state_path=state_path, common=common, max_retries=max_retries)
        self.strategy = strategy

    def _iter_targets(self) -> List[Tuple[str, int, int, int]]:
        out: List[Tuple[str, int, int, int]] = []
        for ep, ep_scores in self.common.scores.items():
            T = len(ep_scores.get("deviate_score") or [])
            if T == 0:
                continue
            k_max = max(self.common.k_grid)
            if self.strategy == "random":
                # Seed with (cfg, ep, random_seed) so two independent reruns
                # get the same baseline draw — resume + cross-process
                # reproducibility both matter (G2 §26 should-fix). Python's
                # built-in ``hash()`` is per-interpreter-randomised, so it
                # cannot be used for a seed that must survive process
                # boundaries; blake2b gives a stable 8-byte digest.
                digest = hashlib.blake2b(
                    f"{self.common.cfg}|{ep}|{self.common.random_seed}".encode(),
                    digest_size=8,
                ).digest()
                seed = int.from_bytes(digest, "big")
                points = _random_k_points(T, k_max, seed=seed)
            else:
                points = _equidistant_k_points(T, k_max)
            for k_idx, s in enumerate(points):
                for n in self.common.n_grid:
                    out.append((ep, int(n), int(k_idx), int(s)))
        return out


# --- Core: teleport + prefill + rollout for one unit ---------------------


def _execute_spawn_unit(
    *, common: _SpawnCommon, ep: str, s: int, n: int, k_idx: int
) -> dict:
    """One teleported rollout. Shared by SpawnRunner and BaselineRunner so
    any fix to the prefill/rollout contract lands in a single place.
    """
    import h5py  # type: ignore[import]

    gt_path = Path(common.gt_dir) / f"{ep}.h5"
    collected_path = Path(common.collected_dir) / common.task_suite_name / f"{ep}.h5"

    # --- 1. Read GT HDF5: anchor state + D-1 prefill obs. -----------------
    with h5py.File(gt_path, "r") as f:
        num_cycles = int(f.attrs["num_cycles"])
        num_env_steps = int(f.attrs["num_steps"])
        task_name = str(f.attrs["task_name"])
        task_id = int(f.attrs["task_id"])
        # G2 §26 MF-3: rollout budget axis = env-timestep (matches Layer C's
        # env.env.timestep counter that restore_init_state writes back into).
        # ``final_env_timestep`` is the authoritative GT end-of-episode anchor
        # written by _flush_trajectory_h5. Older HDF5 predate the field; we
        # fall back to ``num_steps_wait + num_steps`` (env-step axis, same
        # units) and warn loudly because the approximation loses any warm-up
        # divergence and is only tight when warm-up ran clean.
        if "final_env_timestep" in f.attrs:
            final_env_timestep = int(f.attrs["final_env_timestep"])
        else:
            num_steps_wait_attr = int(f.attrs.get("num_steps_wait", 0))
            final_env_timestep = num_steps_wait_attr + num_env_steps
            logger.warning(
                "GT %s missing 'final_env_timestep' attr — falling back to "
                "num_steps_wait (%d) + num_steps (%d) = %d. Rerun Step 1b to "
                "emit the authoritative env-timestep (see MF-3).",
                gt_path, num_steps_wait_attr, num_env_steps, final_env_timestep,
            )
        anchor_cycle = s + n
        if anchor_cycle >= num_cycles:
            # No cycle at s+n — this unit targets a point beyond the GT
            # trajectory. Record it as skipped rather than failing the unit
            # (would otherwise flag as retry-eligible in BaseRunState).
            return {
                "success": False,
                "inference_failed": False,
                "reason": f"anchor_cycle={anchor_cycle} >= num_cycles={num_cycles}",
                "s": s, "n": n, "k_idx": k_idx, "episode": ep,
                "env_steps_executed": 0,
            }
        g_anchor = f[f"step_{anchor_cycle:04d}"]
        sim_state = np.asarray(g_anchor["sim_state"][...]).copy()
        env_timestep = int(np.asarray(g_anchor["env_timestep"][...]).item())
        env_cur_time = float(np.asarray(g_anchor["env_cur_time"][...]).item())

        # D-1 prior cycles' env-space post-transform observations (prompt
        # included, matches exactly the obs shape that drove the original
        # inference — plan §18.A3.3).
        prefill_start = max(0, anchor_cycle - common.D + 1)
        prefill_obs: List[Dict[str, Any]] = []
        for t in range(prefill_start, anchor_cycle):
            g = f[f"step_{t:04d}"]
            prefill_obs.append({
                "observation/image": np.asarray(g["agentview_image"][...], dtype=np.uint8),
                "observation/wrist_image": np.asarray(g["eye_in_hand_image"][...], dtype=np.uint8),
                "observation/state": np.asarray(g["robot_state"][...], dtype=np.float64),
                "prompt": task_name,
            })

    # --- 2. MODEL-space prefill actions come from the server --collect HDF5.
    # (§19.4 regression fix: env-space chunks cannot satisfy the per-connection
    # facade's CP3 keying.)
    prefill_actions: List[np.ndarray] = []
    if prefill_obs:
        if not collected_path.exists():
            # Fail loudly — Step 3 must not silently fall back to env-space.
            raise FileNotFoundError(
                f"Server-side collected HDF5 not found at {collected_path}. "
                f"Start the server with --collect --collect-dir and rerun "
                f"Step 1b so the clean_action records exist for this episode."
            )
        with h5py.File(collected_path, "r") as sf:
            for t in range(prefill_start, anchor_cycle):
                key = f"step_{t:04d}/clean_action"
                if key not in sf:
                    raise RuntimeError(
                        f"{collected_path} missing {key}; did the server "
                        f"run the same bundle that generated the GT?"
                    )
                prefill_actions.append(
                    np.asarray(sf[key][...], dtype=np.float32)
                )

    # --- 3. Env teleport. -------------------------------------------------
    # Both env and client live for the duration of this unit and MUST be
    # released on every exit path — long-running parallel sweeps cannot
    # leak MuJoCo/EGL handles or websocket file descriptors (G2 §26 MF-2).
    env = common.make_env(task_id=task_id, task_suite=common.task_suite_name)
    client: Any = None
    success = False
    env_steps_executed = 0
    episode_started = False
    try:
        env.reset()
        raw_obs = env.set_init_state(sim_state)
        # OffScreenRenderEnv forwards timestep/cur_time through env.env; setting
        # them back matches the Phase-0 smoke script semantics (§19.8).
        inner = getattr(env, "env", env)
        try:
            inner.timestep = env_timestep
            inner.cur_time = env_cur_time
        except AttributeError:
            # Fake envs in tests may not expose these attrs — ignore.
            pass
        # Some wrappers stash a ``done`` flag across set_init_state; clear it so
        # the rollout loop doesn't exit on step 0.
        if hasattr(inner, "done"):
            try:
                inner.done = False
            except AttributeError:
                pass

        # --- 4. Open WS + prefill. -------------------------------------------
        client = common.make_client()
        client.episode_start(
            experiment=f"spawn_{common.cfg}",
            task=task_name,
            # episode_id just needs to be unique on this connection for telemetry.
            episode_id=(hash((ep, s, n, k_idx)) & 0x7FFFFFFF),
            # Empty name → no server-side HDF5 capture (§20.R4, spawn is not
            # driving a collection run; we only care about the success flag).
            episode_name="",
        )
        episode_started = True

        if prefill_actions:
            client.prefill_trajectory(
                observations=prefill_obs,
                actions=prefill_actions,
                record=False,
                on_miss="error",
            )

        # --- 5. Pure cache rollout from the teleport point. --------------
        obs = _obs_env_to_policy(raw_obs, task_name)
        # G2 §26 MF-3: budget is the GT's remaining env-timestep ceiling,
        # clamped by the CLI-level max_spawn_env_steps safety cap. Both
        # operands live on the *env* axis now (previously the code mixed
        # policy-step ``num_steps`` with env-step ``env_timestep``, which
        # systematically under-estimated the budget by exactly the warm-up
        # window). Always give the rollout at least one step so GT that
        # already consumed all its budget still surfaces a result row.
        remaining = max(0, final_env_timestep - env_timestep)
        budget = max(1, min(int(common.max_spawn_env_steps), int(remaining)))

        for _ in range(budget):
            resp = client.infer(obs)
            action_chunk = np.asarray(resp["actions"], dtype=np.float32)
            # §11.1 / §19.5 note: we take the first action of the chunk
            # per env.step to match the cache's CP3 key granularity (each
            # infer = one cycle). A future G2 revision may want to step
            # the full replan_steps; that'd be a one-line change here.
            action = action_chunk[0]
            raw_obs, _reward, done, info = env.step(action.tolist())
            env_steps_executed += 1
            if bool(info.get("success", False)):
                success = True
                break
            if bool(done):
                break
            obs = _obs_env_to_policy(raw_obs, task_name)
    finally:
        # Unconditional cleanup order (G2 §26 MF-2):
        #   episode_end → client close → env close.
        # Each step is independently guarded so a failure in one does not
        # mask the original exception or skip later cleanup. We also
        # tolerate clients/envs that lack ``close`` (FakeClient/FakeEnv in
        # tests) — those are resource-free stand-ins.
        if client is not None and episode_started:
            try:
                client.episode_end(success=success)
            except Exception as e:  # pragma: no cover - defensive
                logger.warning("episode_end raised for %s/%s: %s", common.cfg, ep, e)
        if client is not None:
            close = getattr(client, "close", None)
            if callable(close):
                try:
                    close()
                except Exception as e:  # pragma: no cover - defensive
                    logger.warning("client.close raised for %s/%s: %s", common.cfg, ep, e)
        close_env = getattr(env, "close", None)
        if callable(close_env):
            try:
                close_env()
            except Exception as e:  # pragma: no cover - defensive
                logger.warning("env.close raised for %s/%s: %s", common.cfg, ep, e)

    return {
        "success": bool(success),
        "inference_failed": False,
        "s": int(s),
        "n": int(n),
        "k_idx": int(k_idx),
        "episode": ep,
        "env_steps_executed": int(env_steps_executed),
    }


# --- Aggregate → CSV -----------------------------------------------------


def aggregate_spawn_results(
    out_dir: Path,
    configs: Sequence[str],
    *,
    out_csv: Optional[Path] = None,
    extra_state_files: Optional[Sequence[Path]] = None,
) -> List[Dict[str, Any]]:
    """Scan every ``spawn_state_{cfg}.json`` and emit a long-form CSV row
    per *done* unit: ``config, strategy, episode, s, n, k_idx, success,
    env_steps_executed``.

    ``extra_state_files`` lets ``main()`` thread in BaselineRunner states
    (which have ``{strategy}`` baked into the filename) so one CSV
    captures every spawn variant. Unknown / pending / failed units are
    skipped silently — the status column lives in the source JSON.

    The return value mirrors what we write to CSV so callers (e.g.
    analysis notebooks) can skip the file round-trip.
    """
    out_dir = Path(out_dir)
    rows: List[Dict[str, Any]] = []

    state_files: List[Tuple[Path, str]] = []
    for cfg in configs:
        state_files.append((out_dir / f"spawn_state_{cfg}.json", "top_k"))
    for p in extra_state_files or []:
        # Extract strategy from filename pattern spawn_state_{strategy}_{cfg}.json
        name = Path(p).stem  # e.g. "spawn_state_random_clip_w7_d4"
        strategy = name.split("_")[2] if name.count("_") >= 2 else "unknown"
        state_files.append((Path(p), strategy))

    for path, strategy in state_files:
        if not path.exists():
            continue
        state = json.loads(path.read_text())
        for _key, u in state.items():
            if u.get("status") != "done":
                continue
            r = u.get("result") or {}
            if "episode" not in r:
                # Older/partial result — skip rather than crash the aggregate.
                continue
            rows.append({
                "config": path.stem.replace("spawn_state_", "").replace(f"{strategy}_", "", 1)
                          if strategy != "top_k" else path.stem.replace("spawn_state_", ""),
                "strategy": strategy,
                "episode": r.get("episode", ""),
                "s": int(r.get("s", -1)),
                "n": int(r.get("n", -1)),
                "k_idx": int(r.get("k_idx", -1)),
                "success": bool(r.get("success", False)),
                "env_steps_executed": int(r.get("env_steps_executed", 0)),
            })

    csv_path = Path(out_csv) if out_csv else out_dir / "spawn_aggregate.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["config", "strategy", "episode", "s", "n", "k_idx", "success", "env_steps_executed"]
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    logger.info("Spawn aggregate: wrote %d rows to %s", len(rows), csv_path)
    return rows


# --- CLI -----------------------------------------------------------------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--gt-dir", required=True, help="Client-side GT HDF5 root (Step 1b --save-trajectory-dir)")
    ap.add_argument("--collected-dir", required=True,
                    help="Server-side --collect-dir (root, WITHOUT task_suite; see §20.R4.1)")
    ap.add_argument("--task-suite-name", required=True,
                    help="Must match main.py --task-suite-name / client.episode_start(experiment=…)")
    ap.add_argument("--deviate-score-dir", required=True,
                    help="Directory containing deviate_score_{cfg}.json (Step 2 output)")
    ap.add_argument("--out-dir", required=True, help="Spawn state/artifact root")
    ap.add_argument("--configs", nargs="+", required=True,
                    help="Cache config IDs to sweep (e.g. clip_w7_d4 spatial16_w8_d4 max_pool_w3_d5)")
    ap.add_argument("--D", type=int, default=None,
                    help="Override trajectory depth (default: parse from cfg name '_dN')")
    ap.add_argument("--n-grid", type=int, nargs="+", default=list(DEFAULT_N_GRID))
    ap.add_argument("--k-grid", type=int, nargs="+", default=list(DEFAULT_K_GRID))
    ap.add_argument("--max-spawn-env-steps", type=int, default=DEFAULT_MAX_SPAWN_ENV_STEPS)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--skip-config-switch", action="store_true",
                    help="Skip load_cache_config (caller already switched)")
    ap.add_argument("--config-yaml-dir", default="configs/cache_runs/deviate_exp")
    ap.add_argument("--baselines", nargs="*", default=[], choices=["random", "equidistant"],
                    help="Additional baseline runners to execute after SpawnRunner")
    ap.add_argument("--random-seed", type=int, default=0,
                    help="Seed for BaselineRunner's random-k draw")
    return ap.parse_args(argv)


def _load_scores(scores_path: Path) -> Dict[str, Dict[str, List[float]]]:
    return json.loads(scores_path.read_text())


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    args = parse_args()

    gt_dir = Path(args.gt_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    deviate_dir = Path(args.deviate_score_dir)
    server_url = f"ws://{args.host}:{args.port}"

    extra_state_files: List[Path] = []

    for cfg in args.configs:
        scores_path = deviate_dir / f"deviate_score_{cfg}.json"
        if not scores_path.exists():
            raise FileNotFoundError(
                f"{scores_path} not found — run Step 2 "
                f"(exp/compute_deviate_scores.py) before Step 3."
            )
        D = args.D if args.D is not None else infer_d_from_cfg(cfg)
        scores = _load_scores(scores_path)

        common = _SpawnCommon(
            cfg=cfg,
            gt_dir=gt_dir,
            collected_dir=Path(args.collected_dir),
            task_suite_name=args.task_suite_name,
            D=D,
            out_dir=out_dir,
            host=args.host,
            port=args.port,
            n_grid=tuple(args.n_grid),
            k_grid=tuple(args.k_grid),
            max_spawn_env_steps=args.max_spawn_env_steps,
            scores=scores,
            random_seed=args.random_seed,
        )

        # Switch server to this config's cache bundle (real cache, not inference_).
        if not args.skip_config_switch:
            yaml_path = Path(args.config_yaml_dir) / f"{cfg}.yaml"
            version = send_load_cache_config(server_url, str(yaml_path))
            logger.info("Server switched to cfg=%s v%s for Step 3", cfg, version)

        runner = SpawnRunner(
            state_path=out_dir / f"spawn_state_{cfg}.json",
            common=common,
        )
        runner.parallel_run(num_workers=args.num_workers, resume=args.resume)

        for strategy in args.baselines:
            state_path = out_dir / f"spawn_state_{strategy}_{cfg}.json"
            extra_state_files.append(state_path)
            BaselineRunner(
                state_path=state_path,
                common=common,
                strategy=strategy,
            ).parallel_run(num_workers=args.num_workers, resume=args.resume)

    aggregate_spawn_results(
        out_dir=out_dir,
        configs=args.configs,
        extra_state_files=extra_state_files,
    )


if __name__ == "__main__":
    main()
