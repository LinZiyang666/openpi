"""Step 2: deviate-score computation (plan §10 + §18.A1.3 + §18.A3 + §18.B3).

For each (cache-config, GT episode) we ask:

- Phase 1 — "background L2": with ``gate.type == always_skip`` on both CP1
  and CP3, replay the GT obs sequence *M* times through the server and
  collect the policy action chunks. The pairwise L2 spread between those
  M samples (per cycle) is the floor on stochastic variation of the
  inference path — the "background noise" of the model.

- Phase 2 — "cache L2": with the real cache-enabled YAML loaded, replay
  the same GT obs sequence once and compute the per-cycle L2 between the
  cache-driven action chunk and the GT action chunk. Cache is
  deterministic given a fixed replay, so a single sample is enough
  (§18.B4.2).

- Phase 3 — aggregate: ``deviate_score[t] = cache_l2[t] / max(bg_l2[t], floor)``.
  A score near 1 means cache looks indistinguishable from natural model
  variation; scores far above 1 flag cycles where the cache has pulled
  inference off the ground-truth manifold.

This module exposes its helpers (``load_gt_episode`` / ``aggregate``) as
importable functions so ``exp/trajectory_deviation/run_spawn_experiment.py`` (Step 3) can
reuse them and unit tests can exercise the math without needing a server.
"""
from __future__ import annotations

import argparse
import json
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from exp.common._cache_config_rpc import send_load_cache_config
from exp.common._run_state_base import BaseRunState, UnitState

logger = logging.getLogger(__name__)

# Default floor for deviate_score denominator (plan §10.2). Tiny bg_l2
# values would otherwise explode the ratio; 0.1 was chosen empirically in
# the plan and is exposed as a CLI flag so a future sweep can retune it
# without editing the runner.
_DEFAULT_FLOOR = 0.1


# ---------------------------------------------------------------------------
# GT HDF5 loading (plan §18.A3.3)
# ---------------------------------------------------------------------------


def load_gt_episode(gt_dir: Path, ep_name: str) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    """Read a GT episode HDF5 into ``(obs_seq, gt_first_actions)``.

    ``ep_name`` is the *relative* path under ``gt_dir`` without extension
    (e.g. ``"task_3/episode_0"``) — matches the layout
    ``_flush_trajectory_h5`` writes in ``examples/libero/main.py``.

    Each entry of ``obs_seq`` is a dict shaped exactly as
    ``client.infer`` expects (plan §18.A3.1 locks this format). The
    ``prompt`` comes from the HDF5 ``task_name`` attr because that's where
    the Layer-C writer stashes it.

    ``gt_first_actions`` has shape ``(num_cycles, action_dim)`` — each
    cycle's ``executed_actions[0]``, i.e. the *single* env-space action
    that was actually ``env.step``ed at the start of that replan cycle.
    G2 §26 reverted this module's deviation metric to "first-action L2"
    (§10 original design): comparing the full chunk conflated off-manifold
    divergence with tail horizon slack the env never executes.

    Caller note (§21 review): iterate over ``num_cycles`` (HDF5 attr), not
    ``num_steps``. ``num_steps`` counts **env** steps (sum of per-cycle
    ``executed_action_count``); groups are keyed by cycle.
    """
    # h5py is imported lazily so test stubs / non-libero environments can
    # still import this module for pure-math unit tests.
    import h5py  # type: ignore[import]

    path = Path(gt_dir) / f"{ep_name}.h5"
    with h5py.File(path, "r") as f:
        prompt = f.attrs.get("task_name") or f.attrs.get("prompt") or ""
        num_cycles = int(f.attrs["num_cycles"])
        obs_seq: List[Dict[str, Any]] = []
        gt_first: List[np.ndarray] = []
        for i in range(num_cycles):
            g = f[f"step_{i:04d}"]
            obs_seq.append({
                "observation/image": np.asarray(g["agentview_image"][...], dtype=np.uint8),
                "observation/wrist_image": np.asarray(g["eye_in_hand_image"][...], dtype=np.uint8),
                "observation/state": np.asarray(g["robot_state"][...], dtype=np.float64),
                "prompt": str(prompt),
            })
            # Each cycle's executed_actions is guaranteed shape (K>=1, Ad) by
            # the traj_buffer contract in main.py:_run_episode — no empty
            # stacks ever reach disk (§20.R1.1 drops zero-length tails).
            gt_first.append(np.asarray(g["executed_actions"][0], dtype=np.float32))
    return obs_seq, np.stack(gt_first)


# ---------------------------------------------------------------------------
# Aggregate math (plan §10.2 + §18.B3)
# ---------------------------------------------------------------------------


def _pairwise_l2_mean(vectors: np.ndarray) -> float:
    """Mean pairwise L2 distance over the first axis of ``vectors``.

    ``vectors`` is ``(M, D)``. Uses ``scipy.spatial.distance.pdist`` per
    §18.B3 to avoid the ``(M, M, D)`` temporary tensor that the naive
    broadcasting version would allocate.
    """
    if vectors.shape[0] < 2:
        # With M<2 the pairwise mean is undefined; return 0.0 so the
        # caller's deviate_score uses the floor in the denominator and
        # yields a sane magnitude-only signal (not a NaN).
        return 0.0
    from scipy.spatial.distance import pdist  # type: ignore[import]

    return float(pdist(vectors, metric="euclidean").mean())


def compute_deviate_score(
    bg_chunks: np.ndarray,
    cache_chunks: np.ndarray,
    gt_first_actions: np.ndarray,
    *,
    floor: float = _DEFAULT_FLOOR,
) -> Dict[str, List[float]]:
    """Pure-math core of Phase 3 — first-action L2 metric (G2 §26).

    Inputs:
      ``bg_chunks``        : ``(M, T, H, Ad)`` — Phase 1 samples, full chunks.
      ``cache_chunks``     : ``(T, H, Ad)``    — Phase 2 single sample, full chunks.
      ``gt_first_actions`` : ``(T, Ad)``       — GT ``executed_actions[0]`` per cycle.

    We reduce ``bg_chunks`` and ``cache_chunks`` to first-action internally
    (``[..., 0, :]``) so the jsonl records can keep the full chunks for
    future re-analysis without forcing every reader to be first-action
    aware. The metric itself only uses the single action that actually
    executed — comparing the full chunk conflated off-manifold divergence
    with tail horizon slack the env never steps through (G2 §26 MF-1).

    Outputs have length ``T`` (same as the number of replan cycles).
    """
    if bg_chunks.ndim != 4:
        raise ValueError(f"bg_chunks must be (M, T, H, Ad), got {bg_chunks.shape}")
    if cache_chunks.ndim != 3:
        raise ValueError(f"cache_chunks must be (T, H, Ad), got {cache_chunks.shape}")
    if gt_first_actions.ndim != 2:
        raise ValueError(
            f"gt_first_actions must be (T, Ad), got {gt_first_actions.shape}"
        )
    T, H, Ad = cache_chunks.shape
    if bg_chunks.shape[1:] != (T, H, Ad):
        raise ValueError(
            f"bg per-cycle shape {bg_chunks.shape[1:]} must match cache {(T, H, Ad)}"
        )
    if gt_first_actions.shape != (T, Ad):
        raise ValueError(
            f"gt_first_actions shape {gt_first_actions.shape} must be (T={T}, Ad={Ad})"
        )

    # Reduce both bg and cache to first-action (shape (Ad,) per cycle).
    bg_first = bg_chunks[:, :, 0, :]       # (M, T, Ad)
    cache_first = cache_chunks[:, 0, :]    # (T, Ad)

    bg_l2: List[float] = []
    for t in range(T):
        bg_l2.append(_pairwise_l2_mean(bg_first[:, t, :]))

    # Cache[0] vs GT[0] L2 per cycle.
    diff = cache_first - gt_first_actions  # (T, Ad)
    cache_l2 = np.linalg.norm(diff, axis=1).astype(float).tolist()

    dev = [float(cl / max(bg, floor)) for cl, bg in zip(cache_l2, bg_l2)]
    return {
        "background_l2": bg_l2,
        "cache_l2": cache_l2,
        "deviate_score": dev,
    }


def aggregate(
    bg_jsonl: Path,
    cache_jsonl: Path,
    gt_dir: Path,
    out_path: Path,
    *,
    episodes: Optional[List[str]] = None,
    floor: float = _DEFAULT_FLOOR,
) -> Dict[str, Dict[str, List[float]]]:
    """Read Phase 1/2 jsonl dumps + GT HDF5 → per-episode deviate scores.

    Each jsonl line carries:
      Phase 1 — ``{"config": cfg, "episode": ep, "sample_idx": s, "chunks": [[...]]}``
      Phase 2 — ``{"config": cfg, "episode": ep, "chunks": [[...]]}``

    Writing one line per (episode, sample) keeps the file append-safe for
    the parallel runners — workers never need to read-modify-write.
    """
    bg_per_ep: Dict[str, Dict[int, np.ndarray]] = {}
    for line in bg_jsonl.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        ep = str(d["episode"])
        bg_per_ep.setdefault(ep, {})[int(d["sample_idx"])] = np.asarray(
            d["chunks"], dtype=np.float32
        )

    cache_per_ep: Dict[str, np.ndarray] = {}
    for line in cache_jsonl.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        cache_per_ep[str(d["episode"])] = np.asarray(d["chunks"], dtype=np.float32)

    allowed = set(episodes) if episodes is not None else None
    out: Dict[str, Dict[str, List[float]]] = {}
    for ep, samples in bg_per_ep.items():
        if allowed is not None and ep not in allowed:
            continue
        if ep not in cache_per_ep:
            logger.warning("Episode %s has bg samples but no cache sample — skipping", ep)
            continue
        _obs, gt_first = load_gt_episode(gt_dir, ep)
        # Dense (M, T, H, Ad) stack; require samples indexed 0..M-1.
        sample_ids = sorted(samples)
        bg_stack = np.stack([samples[s] for s in sample_ids])
        out[ep] = compute_deviate_score(
            bg_stack, cache_per_ep[ep], gt_first, floor=floor
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    return out


# ---------------------------------------------------------------------------
# Phase 1 / Phase 2 runners
# ---------------------------------------------------------------------------


# Serialises parallel writes to the per-config jsonl files. Each worker
# appends a single whole line so we only need coarse-grained locking.
_JSONL_WRITE_LOCK = threading.Lock()


def _append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _JSONL_WRITE_LOCK:
        with path.open("a") as fh:
            fh.write(json.dumps(record) + "\n")


def _roll_out_episode(
    client, obs_seq: List[Dict[str, Any]]
) -> np.ndarray:
    """Replay a GT obs sequence one cycle at a time and collect action chunks.

    Each ``client.infer(obs)`` returns ``{"actions": (H, Ad), ...}`` per
    main.py:186. The AlwaysSkipGate + inference YAML guarantees that
    Phase 1 never hits cache regardless of the server's gate state.
    """
    chunks: List[np.ndarray] = []
    for obs in obs_seq:
        resp = client.infer(obs)
        chunks.append(np.asarray(resp["actions"], dtype=np.float32))
    return np.stack(chunks) if chunks else np.zeros((0, 0, 0), dtype=np.float32)


@dataclass
class _PhaseCommon:
    """Shared config for both Phase runners — keeps the subclass ctors tiny."""

    config_id: str
    gt_dir: Path
    episodes: List[str]
    out_dir: Path
    host: str
    port: int
    # Test hook: inject a fake WebsocketClientPolicy factory.
    client_factory: Optional[Any] = None

    def make_client(self):
        if self.client_factory is not None:
            return self.client_factory(self.host, self.port)
        # Lazy import so unit tests without openpi_client still work.
        from openpi_client.websocket_client_policy import WebsocketClientPolicy  # type: ignore[import]

        return WebsocketClientPolicy(host=self.host, port=self.port)


class _PhaseRunner(BaseRunState):
    """Shared scaffolding for Phase 1 / Phase 2 runners.

    Subclasses declare four class-level attributes that differentiate the
    two phases (filename prefix, experiment tag, whether the unit key
    carries ``sample_idx``, and whether each episode fans out into M
    samples). The actual body — GT load, episode bookkeeping, jsonl
    append — lives here so ``execute_unit`` stays honest to the cleanup/07
    contract: structure only, no lifecycle changes (F2 follow-up).
    """

    # Subclass overrides. ``USES_SAMPLE_IDX`` is the single switch that
    # controls DeviateKey shape, task label, episode_id hashing, and the
    # extra ``sample_idx`` field in the jsonl record.
    FILE_PREFIX: str = ""
    EXPERIMENT: str = ""
    USES_SAMPLE_IDX: bool = False

    def __init__(
        self,
        *,
        state_path: Path,
        common: _PhaseCommon,
        max_retries: int = 2,
    ) -> None:
        super().__init__(state_path=state_path, max_retries=max_retries)
        self.common = common

    # Subclasses yield the sample_idx values to fan out per episode.
    # Phase 1: range(M); Phase 2: a single None.
    def _sample_idxs(self) -> List[Optional[int]]:
        raise NotImplementedError

    def build_units(self) -> List[UnitState]:
        from exp.common._unit_key import DeviateKey

        return [
            UnitState(
                unit_key=DeviateKey(
                    cfg=self.common.config_id, ep=ep, sample_idx=s
                ).encode()
            )
            for ep in self.common.episodes
            for s in self._sample_idxs()
        ]

    def execute_unit(self, unit: UnitState) -> dict:
        from exp.common._unit_key import DeviateKey

        k = DeviateKey.decode(unit.unit_key)
        cfg, ep, s = k.cfg, k.ep, k.sample_idx
        # Explicit ValueError (not ``assert``) so the shape guard survives
        # ``python -O`` — a corrupted unit key mixing Phase 1 / Phase 2
        # schemas must always fail loudly, regardless of optimisation level.
        if self.USES_SAMPLE_IDX and s is None:
            raise ValueError(
                f"{type(self).__name__} unit_key {unit.unit_key!r} must carry sample_idx"
            )
        if not self.USES_SAMPLE_IDX and s is not None:
            raise ValueError(
                f"{type(self).__name__} unit_key {unit.unit_key!r} must not carry sample_idx"
            )
        if self.USES_SAMPLE_IDX:
            task_label = f"{cfg}_ep{ep}_s{s}"
            ep_id = hash((ep, s)) & 0x7FFFFFFF
        else:
            task_label = f"{cfg}_ep{ep}"
            ep_id = hash(ep) & 0x7FFFFFFF

        obs_seq, _ = load_gt_episode(self.common.gt_dir, ep)
        client = self.common.make_client()
        # Unique episode_id per (episode[,sample]) so the server-side
        # per-connection facade starts from a clean trajectory-history
        # state (plan §18.A1.3 note: CP3 is also always_skip so no
        # cross-sample contamination, but the episode_id still needs to be
        # distinct for log/telemetry clarity).
        client.episode_start(
            experiment=self.EXPERIMENT,
            task=task_label,
            episode_id=ep_id,
            episode_name="",
        )
        try:
            chunks = _roll_out_episode(client, obs_seq)
        finally:
            client.episode_end(success=True)

        record: dict = {"config": cfg, "episode": ep, "chunks": chunks.tolist()}
        if self.USES_SAMPLE_IDX:
            record["sample_idx"] = s
        _append_jsonl(self.common.out_dir / f"{self.FILE_PREFIX}{cfg}.jsonl", record)
        return {"T": int(chunks.shape[0])}


class Phase1Runner(_PhaseRunner):
    """Phase 1 worker: M stochastic replays per episode under AlwaysSkipGate."""

    FILE_PREFIX = "bg_"
    EXPERIMENT = "deviate_score_phase1"
    USES_SAMPLE_IDX = True

    def __init__(
        self,
        *,
        state_path: Path,
        common: _PhaseCommon,
        M: int,
        max_retries: int = 2,
    ) -> None:
        super().__init__(state_path=state_path, common=common, max_retries=max_retries)
        self.M = M

    def _sample_idxs(self) -> List[Optional[int]]:
        return list(range(self.M))


class Phase2Runner(_PhaseRunner):
    """Phase 2 worker: one cache-driven replay per episode."""

    FILE_PREFIX = "cache_"
    EXPERIMENT = "deviate_score_phase2"
    USES_SAMPLE_IDX = False

    def _sample_idxs(self) -> List[Optional[int]]:
        return [None]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def discover_episodes(
    gt_dir: Path,
    *,
    include_failed: bool = False,
    include_unknown: bool = False,
) -> List[str]:
    """Enumerate every ``task_X/episode_Y.h5`` under ``gt_dir`` as a list
    of relative episode names (without extension).

    G2 §26 MF-4: by default, episodes with ``success=False`` or missing
    ``success`` attr are dropped — a failed GT rollout doesn't reach the
    goal, so its deviate-score carries no recovery signal for Step 3.

    ``include_failed``   : keep explicit ``success=False`` episodes.
    ``include_unknown``  : keep legacy HDF5 without a ``success`` attr
                           (logs a per-file warning so the operator can
                           re-collect GT for those episodes).
    """
    import h5py  # type: ignore[import]

    gt_dir = Path(gt_dir)
    out: List[str] = []
    for p in sorted(gt_dir.rglob("episode_*.h5")):
        rel = p.relative_to(gt_dir).with_suffix("")
        ep_name = rel.as_posix()
        try:
            with h5py.File(p, "r") as f:
                if "success" in f.attrs:
                    if bool(f.attrs["success"]):
                        out.append(ep_name)
                    elif include_failed:
                        out.append(ep_name)
                    else:
                        logger.info(
                            "discover_episodes: skipping %s (success=False; "
                            "pass --include-failed-gt to keep)", ep_name,
                        )
                else:
                    # Legacy HDF5 predating the success attr (Layer C always
                    # writes it now, but archived data might not).
                    logger.warning(
                        "discover_episodes: %s missing 'success' attr — "
                        "%s. Rerun Step 1b to regenerate.",
                        ep_name,
                        "including via --include-unknown-gt" if include_unknown else "skipping",
                    )
                    if include_unknown:
                        out.append(ep_name)
        except (OSError, KeyError) as e:
            # Corrupted / unreadable HDF5: surface as warning + skip.
            logger.warning("discover_episodes: could not read %s (%s) — skipping", p, e)
    return out


FailedUnit = Tuple[int, int]


def load_failed_units_by_config(results_path: Path) -> Dict[str, Set[FailedUnit]]:
    """Load Step-1a failed units keyed by cache config.

    The input can be either the already-filtered
    ``cache_eval_results_cache_fail.json`` or the full
    ``cache_eval_results.json``; rows with ``success=True`` are ignored.
    Units are identified in the original LIBERO init-state namespace:
    ``(task_id, orig_init_state_idx)``.
    """
    rows = json.loads(Path(results_path).read_text())
    if not isinstance(rows, list):
        raise ValueError(f"{results_path} must contain a JSON list of episode rows")

    out: Dict[str, Set[FailedUnit]] = {}
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"{results_path}: row {i} is not an object")
        if bool(row.get("success", False)):
            continue
        cfg = row.get("config_id") or row.get("config")
        if cfg is None:
            raise ValueError(f"{results_path}: failed row {i} missing config_id")
        if "task_id" not in row:
            raise ValueError(f"{results_path}: failed row {i} missing task_id")
        init_key = "orig_init_state_idx" if "orig_init_state_idx" in row else "init_state_idx"
        if init_key not in row:
            raise ValueError(
                f"{results_path}: failed row {i} missing orig_init_state_idx/init_state_idx"
            )
        out.setdefault(str(cfg), set()).add((int(row["task_id"]), int(row[init_key])))
    return out


def _gt_episode_failed_unit(gt_dir: Path, ep_name: str) -> FailedUnit:
    """Read ``(task_id, orig_init_state_idx)`` attrs from one GT HDF5."""
    import h5py  # type: ignore[import]

    with h5py.File(Path(gt_dir) / f"{ep_name}.h5", "r") as f:
        if "task_id" not in f.attrs:
            raise KeyError("task_id")
        if "orig_init_state_idx" in f.attrs:
            init_idx = f.attrs["orig_init_state_idx"]
        elif "init_state_idx" in f.attrs:
            init_idx = f.attrs["init_state_idx"]
        else:
            raise KeyError("orig_init_state_idx/init_state_idx")
        return int(f.attrs["task_id"]), int(init_idx)


def filter_episodes_by_failed_units(
    gt_dir: Path,
    episodes: List[str],
    failed_units: Set[FailedUnit],
) -> List[str]:
    """Keep only GT episodes whose original init failed for the active cfg."""
    out: List[str] = []
    for ep in episodes:
        try:
            unit = _gt_episode_failed_unit(gt_dir, ep)
        except (OSError, KeyError, ValueError) as e:
            logger.warning(
                "filter_episodes_by_failed_units: could not read identity for %s (%s) — skipping",
                ep,
                e,
            )
            continue
        if unit in failed_units:
            out.append(ep)
    return out


def _episode_unit_filter(episodes: List[str]) -> Callable[[UnitState], bool]:
    """Scope resumed state files to the current episode set."""
    from exp.common._unit_key import DeviateKey

    allowed = set(episodes)

    def _in_scope(unit: UnitState) -> bool:
        return DeviateKey.decode(unit.unit_key).ep in allowed

    return _in_scope


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt-dir", required=True, help="Step 1b GT trajectory root")
    ap.add_argument("--out-dir", required=True, help="Where bg/cache jsonl + deviate_score_*.json land")
    ap.add_argument("--configs", nargs="+", required=True,
                    help="Cache config IDs to sweep (e.g. clip_w7_d4 spatial16_w8_d4 max_pool_w3_d5)")
    ap.add_argument("--M", type=int, default=20, help="Phase 1 samples per episode")
    ap.add_argument("--num-workers", type=int, default=4,
                    help="Concurrent websocket connections per phase")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--floor", type=float, default=_DEFAULT_FLOOR)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--config-yaml-dir", default="configs/cache_runs/deviate_exp",
                    help="Directory holding inference_*.yaml + cache_*.yaml")
    ap.add_argument("--config-fail-results", default=None,
                    help="Optional Step-1a results JSON. When set, each cfg only runs GT "
                         "episodes whose (task_id, orig_init_state_idx) failed for that cfg. "
                         "Use data/deviation_experiment/cache_eval_results_cache_fail.json "
                         "to filter out inits that already succeeded for the active cfg.")
    ap.add_argument("--skip-config-switch", action="store_true",
                    help="Do not call load_cache_config — assume the server is already correctly configured "
                    "(useful when sharding this driver across GPUs).")
    # G2 §26 MF-4: failed GT rollouts don't reach the goal so their deviate
    # scores are uninformative for Step 3. Default: skip. Flags below are
    # escape hatches when the operator deliberately wants the noise.
    ap.add_argument("--include-failed-gt", action="store_true",
                    help="Include GT episodes whose 'success' attr is False "
                         "(default: skip them — they carry no recovery signal).")
    ap.add_argument("--include-unknown-gt", action="store_true",
                    help="Include legacy GT HDF5 without a 'success' attr "
                         "(default: warn + skip).")
    return ap.parse_args(argv)


def _yaml_path_for(config_yaml_dir: Path, prefix: str, cfg: str) -> Path:
    """Resolve a cache-bundle YAML for a given phase + config id."""
    return Path(config_yaml_dir) / f"{prefix}{cfg}.yaml"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    args = parse_args()

    gt_dir = Path(args.gt_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    episodes = discover_episodes(
        gt_dir,
        include_failed=args.include_failed_gt,
        include_unknown=args.include_unknown_gt,
    )
    if not episodes:
        raise RuntimeError(f"No GT episodes found under {gt_dir}")

    failed_by_config: Optional[Dict[str, Set[FailedUnit]]] = None
    if args.config_fail_results:
        fail_results_path = Path(args.config_fail_results)
        failed_by_config = load_failed_units_by_config(fail_results_path)
        logger.info(
            "Loaded per-config failed-init filter from %s (%d configs)",
            fail_results_path,
            len(failed_by_config),
        )

    server_url = f"ws://{args.host}:{args.port}"

    for cfg in args.configs:
        cfg_episodes = episodes
        if failed_by_config is not None:
            if cfg not in failed_by_config:
                raise RuntimeError(
                    f"No failed Step-1a rows found for config {cfg!r} in "
                    f"{args.config_fail_results}"
                )
            cfg_episodes = filter_episodes_by_failed_units(
                gt_dir, episodes, failed_by_config[cfg]
            )
            logger.info(
                "Config %s: kept %d/%d GT episodes after per-config failure filtering",
                cfg,
                len(cfg_episodes),
                len(episodes),
            )
            if not cfg_episodes:
                logger.warning("Config %s has no matching GT episodes; writing empty score file", cfg)
                (out_dir / f"deviate_score_{cfg}.json").write_text("{}")
                continue

        common = _PhaseCommon(
            config_id=cfg, gt_dir=gt_dir, episodes=cfg_episodes, out_dir=out_dir,
            host=args.host, port=args.port,
        )

        # Phase 1: AlwaysSkip on CP1 + CP3 (§18.A1.3).
        if not args.skip_config_switch:
            send_load_cache_config(
                server_url,
                str(_yaml_path_for(args.config_yaml_dir, "inference_", cfg)),
            )
        Phase1Runner(
            state_path=out_dir / f"phase1_state_{cfg}.json",
            common=common,
            M=args.M,
        ).parallel_run(
            num_workers=args.num_workers,
            resume=args.resume,
            unit_filter=_episode_unit_filter(cfg_episodes),
        )

        # Phase 2: real cache bundle.
        if not args.skip_config_switch:
            send_load_cache_config(
                server_url,
                str(_yaml_path_for(args.config_yaml_dir, "", cfg)),
            )
        Phase2Runner(
            state_path=out_dir / f"phase2_state_{cfg}.json",
            common=common,
        ).parallel_run(
            num_workers=args.num_workers,
            resume=args.resume,
            unit_filter=_episode_unit_filter(cfg_episodes),
        )

        # Phase 3: offline aggregate.
        aggregate(
            bg_jsonl=out_dir / f"bg_{cfg}.jsonl",
            cache_jsonl=out_dir / f"cache_{cfg}.jsonl",
            gt_dir=gt_dir,
            out_path=out_dir / f"deviate_score_{cfg}.json",
            episodes=cfg_episodes,
            floor=args.floor,
        )
        logger.info("Wrote %s", out_dir / f"deviate_score_{cfg}.json")


if __name__ == "__main__":
    main()
