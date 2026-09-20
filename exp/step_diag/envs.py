"""Frozen environment identity table of the step-vs-warm-start line (plan §3.0) and manifests.

An *environment* is one ``(policy, benchmark)`` pair. Its identity fixes the action horizon, the
executed-dim mask, the full step count and denoise schedule, the warm-start resume timesteps
(with the Euler steps each executes), the reduced step counts of the shadow (``k_set``) and the
serving topology. The table below is the plan's; ``resolve_env`` returns a copy the serve scripts
extend with the host-resolved artefacts (checkpoint path/sha, library/config sha) into the run
manifest that every evidence row references by ``config_sha``.

Seeds: RoboCasa formal runs use ``base_seed = 2_000_000`` (Q-B idx 0..49, shadow idx 0..9);
smoke runs use ``3_000_000`` and a separate experiment id; LIBERO keeps its frozen init pools
(shadow idx 0..9 of the pool).
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import functools
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from openpi.cache.types import DenoiseSchedule

RC_FORMAL_BASE_SEED = 2_000_000
RC_SMOKE_BASE_SEED = 3_000_000
H_EXEC = 5
REPLAN_STEPS = 5


@dataclass(frozen=True)
class EnvSpec:
    """One row of the plan §3.0 table."""

    env_id: str
    policy: str  # pi05 | groot
    benchmark: str  # robocasa365 | libero_spatial | libero_10 | libero_object | libero_goal
    action_horizon: int
    action_dim: int
    n_executed: int
    k_full: int
    schedule_id: str
    warm_ts: Tuple[float, ...]
    k_set: Tuple[int, ...]
    server_host: str
    worker_host: str

    @property
    def schedule(self) -> DenoiseSchedule:
        # Simulator workers use this table without the serving dependency island.
        from openpi.cache.types import PI05_V1, groot_n15_schedule

        return PI05_V1 if self.policy == "pi05" else groot_n15_schedule(self.k_full)

    def remaining_steps(self, t: float) -> int:
        return self.schedule.remaining_steps(float(t))

    def to_json(self) -> dict:
        d = asdict(self)
        d["warm_ts"] = list(self.warm_ts)
        d["k_set"] = list(self.k_set)
        d["warm_remaining_steps"] = {f"{t:.4f}": self.remaining_steps(t) for t in self.warm_ts}
        return d


ENVS: Dict[str, EnvSpec] = {
    "pi05_rc": EnvSpec("pi05_rc", "pi05", "robocasa365", 50, 32, 12, 10, "pi05_v1", (0.1, 0.2, 0.3), (1, 2, 3, 5),
                       "h100", "timan108"),
    "pi05_libero_spatial": EnvSpec("pi05_libero_spatial", "pi05", "libero_spatial", 10, 32, 7, 10, "pi05_v1",
                                   (0.1, 0.2, 0.3), (1, 2, 3, 5), "weilandserver", "timan107"),
    "pi05_libero_10": EnvSpec("pi05_libero_10", "pi05", "libero_10", 10, 32, 7, 10, "pi05_v1", (0.1, 0.2, 0.3),
                              (1, 2, 3, 5), "weilandserver", "timan107"),
    "groot_rc": EnvSpec("groot_rc", "groot", "robocasa365", 16, 32, 12, 4, "groot_n15_k4_v1", (0.75, 0.5), (1, 2, 3),
                        "h100", "timan108"),
    "groot_libero_spatial": EnvSpec("groot_libero_spatial", "groot", "libero_spatial", 16, 32, 7, 8,
                                    "groot_n15_k8_v1", (0.875, 0.75, 0.5), (1, 2, 4, 6), "weilandserver", "timan107"),
    "groot_libero_10": EnvSpec("groot_libero_10", "groot", "libero_10", 16, 32, 7, 8, "groot_n15_k8_v1",
                               (0.875, 0.75, 0.5), (1, 2, 4, 6), "weilandserver", "timan107"),
}

# Q-B fixed groups (plan §2 Q-B): frozen before any new closed-loop result is read.
QB_MAIN_M = {"pi05": 2, "groot": 1}
QB_MAIN_T = {"pi05": 0.2, "groot": 0.75}
QB_PLAIN_KS = {"pi05": (1, 2, 3), "groot": (1, 2)}
QB_WARM_TS = {"pi05": (0.1, 0.2, 0.3), "groot": (0.75, 0.5)}
QB_CLIFF = {
    "pi05": ("CloseFridge", "OpenCabinet", "PickPlaceToasterToCounter", "PickPlaceDrawerToCounter"),
    "groot": ("PickPlaceDrawerToCounter", "SlideDishwasherRack", "TurnOnSinkFaucet", "OpenCabinet"),
}
QB_FLAT = {
    "pi05": ("PickPlaceSinkToCounter", "OpenDrawer", "PickPlaceCounterToStove"),
    "groot": ("PickPlaceCounterToStove",),
}
# historical gaps at the main m (selection evidence only; never used as an estimate)
QB_HISTORICAL_GAP = {
    "pi05": {"CloseFridge": 0.50, "OpenCabinet": 0.18, "PickPlaceToasterToCounter": 0.16,
             "PickPlaceDrawerToCounter": 0.18},
    "groot": {"PickPlaceDrawerToCounter": 0.30, "SlideDishwasherRack": 0.24, "TurnOnSinkFaucet": 0.22,
              "OpenCabinet": 0.18},
}
RC_MAIN_LANE = ("CloseFridge", "OpenCabinet", "OpenDrawer", "SlideDishwasherRack", "TurnOnSinkFaucet",
                "CloseBlenderLid", "CoffeeSetupMug", "OpenStandMixerHead")
RC_PNP_LANE = ("PickPlaceCounterToCabinet", "PickPlaceCounterToStove", "PickPlaceDrawerToCounter",
               "PickPlaceSinkToCounter", "PickPlaceToasterToCounter")
QB_EPISODES = 50
QB_FLAT_EPISODES = 100  # v3.1: the two arms defining the flat Delta run 100 episodes
SHADOW_EPISODES = 10


def lane_of(task: str) -> str:
    if task in RC_MAIN_LANE:
        return "main"
    if task in RC_PNP_LANE:
        return "pnp"
    raise KeyError(f"unknown RoboCasa365 task {task!r}")


@functools.lru_cache(maxsize=1)
def canonical_pin_id() -> str:
    from exp.robocasa365.pinned_objects import load_pin_manifest

    path = pathlib.Path(__file__).resolve().parents[1] / "robocasa365/config/pnp_pinned_objects.json"
    return load_pin_manifest(path)[0]


def qb_tasks(policy: str) -> Tuple[str, ...]:
    return tuple(QB_CLIFF[policy]) + tuple(QB_FLAT[policy])


def qb_episode_count(policy: str, task: str, arm_id: str) -> int:
    """50 episodes, or 100 for a flat task on the two arms that define its Delta (plan v3.1)."""
    main_arms = {f"plain_k{QB_MAIN_M[policy]}", f"warm_t{QB_MAIN_T[policy]:g}"}
    if task in QB_FLAT[policy] and arm_id in main_arms:
        return QB_FLAT_EPISODES
    return QB_EPISODES


def resolve_env(env_id: str) -> EnvSpec:
    try:
        return ENVS[env_id]
    except KeyError as exc:
        raise KeyError(f"unknown env_id {env_id!r}; known: {sorted(ENVS)}") from exc


def sha256_file(path: str | pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode("utf-8")).hexdigest()


@dataclass
class RunManifest:
    """What one served arm binds to (written next to the evidence rows as ``manifest_<arm>.json``)."""

    experiment_id: str
    env: dict
    arm_id: str
    mode: str
    exec_steps: Optional[int]
    checkpoint: str
    checkpoint_sha256: Optional[str]
    cache_config: Optional[str]
    cache_config_sha256: Optional[str]
    library: Optional[str]
    library_sha256: Optional[str]
    code_commit: str
    extras: dict = field(default_factory=dict)

    # Fields that identify the served CONTRACT. ``code_commit`` and ``extras.runtime`` are recorded
    # next to it but not hashed: a server restarted on the same contract after a commit keeps its
    # config_sha, so the driver can resume the same cell (run_id / task_uid embed this digest).
    CONTRACT_FIELDS = ("experiment_id", "env", "arm_id", "mode", "exec_steps", "checkpoint", "checkpoint_sha256",
                       "cache_config", "cache_config_sha256", "library", "library_sha256")

    @property
    def config_sha(self) -> str:
        d = asdict(self)
        contract = {k: d[k] for k in self.CONTRACT_FIELDS}
        contract["extras"] = {k: v for k, v in d["extras"].items() if k != "runtime"}
        return sha256_json(contract)

    def write(self, path: str | pathlib.Path) -> str:
        payload = asdict(self) | {"config_sha": self.config_sha}
        text = json.dumps(payload, indent=1, sort_keys=True, default=str) + "\n"
        pathlib.Path(path).with_name(f"manifest_{self.config_sha}.json").write_text(text)
        pathlib.Path(path).write_text(text)
        return payload["config_sha"]


def git_commit(repo: str | pathlib.Path = ".") -> str:
    import subprocess

    try:
        return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001 - outside a checkout the manifest says so
        return "unknown"


def sha256_tree(path: str | pathlib.Path) -> str:
    """Full content identity of a file or directory (offline use: pools, small asset trees)."""
    if not path:
        raise ValueError("artifact path is required")
    root = pathlib.Path(path)
    if root.is_file():
        return sha256_file(root)
    if not root.is_dir():
        raise FileNotFoundError(f"artifact does not exist: {root}")
    files = sorted(p for p in root.rglob("*") if p.is_file() and ".cache" not in p.relative_to(root).parts)
    if not files:
        raise ValueError(f"empty artifact directory: {root}")
    return sha256_json({str(p.relative_to(root)): sha256_file(p) for p in files})


SMALL_FILE_BYTES = 8 << 20


def checkpoint_digest(path: str | pathlib.Path) -> str:
    """Cheap checkpoint identity: the file listing with sizes, plus the content of every file
    under ``SMALL_FILE_BYTES`` (configs, norm stats, tokenizer assets). Multi-GB weight shards are
    identified by path + size; the digest costs no weight reads, so it can run at every server
    start (owner ruling 2026-09-12: identity is recorded, it must not delay starts)."""
    if not path:
        raise ValueError("checkpoint path is required")
    root = pathlib.Path(path)
    if root.is_file():
        return sha256_json({root.name: [root.stat().st_size, sha256_file(root) if root.stat().st_size <= SMALL_FILE_BYTES else None]})
    if not root.is_dir():
        raise FileNotFoundError(f"checkpoint does not exist: {root}")
    listing = {}
    for p in sorted(x for x in root.rglob("*") if x.is_file() and ".cache" not in x.relative_to(root).parts):
        size = p.stat().st_size
        listing[str(p.relative_to(root))] = [size, sha256_file(p) if size <= SMALL_FILE_BYTES else None]
    if not listing:
        raise ValueError(f"empty checkpoint directory: {root}")
    return sha256_json(listing)


def library_digest(path: str | pathlib.Path) -> str:
    """sha256 of a library pickle, cached in a ``<path>.sha256`` sidecar (``sha256sum`` format).

    The sidecar is written once (ops Step 0 / ``freeze_weights`` / first server start) and reused
    while the file's size and mtime are unchanged, so a 28 GB library is hashed once, not at every
    server start."""
    p = pathlib.Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"library does not exist: {p}")
    side = p.with_name(p.name + ".sha256")
    stat = p.stat()
    stamp = f"{stat.st_size}:{int(stat.st_mtime)}"
    if side.is_file():
        parts = side.read_text().split()
        if len(parts) >= 3 and parts[2] == stamp and len(parts[0]) == 64:
            return parts[0]
    digest = sha256_file(p)
    try:
        side.write_text(f"{digest}  {p.name} {stamp}\n")
    except OSError:
        pass  # read-only location: the digest is still returned and recorded
    return digest


def resource_identity(env_id: str, checkpoint: str, cache_config: str | None) -> dict:
    """Identity of what the serving process loads, recorded in the manifest.

    Cheap by construction (checkpoint listing + small files, sidecar-cached library digest, source
    listing): it binds the arm's evidence to concrete assets without re-reading multi-GB weights or
    unpickling the library at start. The library's payload contract is checked offline by
    ``emit_arms --check-libraries`` (ops Step 0), not here.
    """
    import sys
    import socket
    import torch
    import yaml

    from exp.step_diag.emit_arms import load_base, verify_cell

    ckpt_sha = checkpoint_digest(checkpoint)
    library = library_sha = None
    if cache_config:
        cfg = yaml.safe_load(pathlib.Path(cache_config).read_text())
        base, _ = load_base(env_id)
        verify_cell(env_id, cfg, base)
        library = cfg["backend"]["in_memory"]["preload_path"]
        library_sha = library_digest(library)
    repo = pathlib.Path(__file__).resolve().parents[2]
    sources = {}
    for directory in ("exp/step_diag", "src/openpi/cache", "exp/robocasa365", "exp/libero_groot", "src/openpi/policies",
                      "examples/libero", "src/openpi/models_pytorch"):
        for p in sorted((repo / directory).rglob("*.py")):
            sources[str(p.relative_to(repo))] = sha256_file(p)
    return {"checkpoint_sha256": ckpt_sha, "library": library, "library_sha256": library_sha,
            "runtime": {"python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda,
                        "host": socket.gethostname(),
                        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                        "source_sha256": sha256_json(sources)}}


def validate_arm(env_id: str, mode: str, arm_id: str, exec_steps: int | None, cache_config: str | None) -> None:
    import yaml

    env = resolve_env(env_id)
    expected_arm = f"plain_k{exec_steps}" if mode == "plain" else mode
    if mode != "warm" and arm_id != expected_arm:
        raise ValueError(f"mode/step identity requires arm_id={expected_arm}")
    if mode in ("shadow", "warm"):
        cfg = yaml.safe_load(pathlib.Path(cache_config).read_text())
        judge = cfg["checkpoints"]["cp1"]["judge"]
        if mode == "warm":
            if not arm_id.startswith("warm_t"):
                raise ValueError("warm arm must name its start_t")
            t = float(arm_id.removeprefix("warm_t"))
            if t not in QB_WARM_TS[env.policy] or env.benchmark != "robocasa365":
                raise ValueError("warm start_t is outside the frozen arm set")
            expected = {"type": "always_warm_start", "start_t": t}
        else:
            expected = {"type": "threshold", "threshold": 99.0} if env.policy == "pi05" else {"type": "always_hit"}
        if judge != expected:
            raise ValueError(f"judge differs from {arm_id}: expected {expected}")
