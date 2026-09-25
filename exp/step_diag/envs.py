"""Frozen environment identity table of the step-vs-warm-start line (plan §3.0) and manifests.

An *environment* is one ``(policy, benchmark)`` pair. Its identity fixes the action horizon, the
executed-dim mask, the full step count and denoise schedule, the warm-start resume timesteps
(with the Euler steps each executes), the reduced step counts of the shadow (``k_set``) and the
serving topology. The table below is the plan's; ``resolve_env`` returns a copy the serve scripts
extend with the host-resolved artefacts (checkpoint path/sha, library/config sha) into the run
manifest that every evidence row references by ``config_sha``.

Seeds: RoboCasa formal runs use ``base_seed = 2_000_000`` (Q-B idx 0..49, shadow idx 0..9);
smoke runs use ``3_000_000`` and a separate experiment id; LIBERO keeps its frozen init pools
(shadow idx 0..9 of the pool, the self-start round idx 0..49) and env seed 7.
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
# Warm-start continuation variants (2026-09-21 reviewer question): served mode -> variant of
# ``exp.step_diag.pi05.warm_variant_stage3``. Arm ids are ``<mode>_t<start_t>`` (e.g. ``warmreset_t0.2``);
# they use the ``warm_t<start_t>.yaml`` cache config and run the same number of Euler steps as
# ``warm_t<start_t>`` but with ``dt = -1/remaining_steps``. Pi0.5 RoboCasa only.
WARM_VARIANT_MODES = {"warmreset": "reset_t", "warmshoot": "overshoot", "resetfinal": "reset_final",
                      "midfinal": "mid_final", "midfinal50": "mid_final50",
                      "midreset": "mid_snap", "midreset50": "mid_snap50"}
# ``midfinal`` (owner 2026-09-22): the ``resetfinal`` start (the cache's final action chunk, fed as-is, no
# noise added) entered at the intermediate noise level MID_ENTRY_T instead of pure noise, then n Euler
# steps down to the clean end with dt = -MID_ENTRY_T/n. Written in pi0.5 flow time (1 = noise, 0 = clean);
# GR00T's ascending loop enters at 1 - MID_ENTRY_T. The entry is ONE full-schedule grid step below pure
# noise for each policy (pi0.5 K=10 -> 0.9, GR00T K=4 -> 0.75). The arm's ``start_t`` only sets the budget n.
MID_ENTRY_T = {"pi05": 0.9, "groot": 0.75}
# ``midfinal50`` (owner 2026-09-23): the same final-chunk start entered at flow time 0.5 for both policies.
# ``midreset`` (owner 2026-09-23): the ``warmreset`` start (the cached snapshot at the arm's start_t) entered at
# the same one-grid-step-below-noise level as ``midfinal`` instead of t = 1.
MID_ENTRY_T_BY_VARIANT = {"mid_final": MID_ENTRY_T, "mid_final50": {"pi05": 0.5, "groot": 0.5}, "mid_snap": MID_ENTRY_T,
                          "mid_snap50": {"pi05": 0.5, "groot": 0.5}}  # midreset50: snapshot start fed at 0.5
MID_VARIANTS = tuple(MID_ENTRY_T_BY_VARIANT)
# Self-start ablation (owner 2026-09-24): the same warm-reset variants, but the start is NOT the cache. Each
# decision first runs the policy's own full inference on the current observation (private noise), then feeds
# that run's snapshot at the arm's start_t (T = start_t in pi0.5 time) or its final action (T = 0) exactly as
# the cache arm feeds the retrieved one. Arm ids are ``self`` + the cache arm id (``selfwarmreset_t0.2``); the
# served yaml is the cache arm's, whose retrieval still runs so the serving path is unchanged, but its payload
# is never used. The exact-resume arm (``warm``) has no self counterpart: resuming a run's own snapshot on its
# own grid reproduces that run. No overshoot counterpart either.
SELF_VARIANT_MODES = {f"self{mode}": variant for mode, variant in WARM_VARIANT_MODES.items() if mode != "warmshoot"}
# Shoot ablation (owner 2026-09-24, GR00T only): no reset. The snapshot stays at its own flow time (T; GR00T native
# 1 - T) and takes the arm's N steps with the step size of a warm-reset variant, dt = SHOOT_ENTRY_T / N in pi0.5 time:
# ``warmshoot`` borrows warmreset's (entry 1), ``midshoot`` midreset's (0.75), ``midshoot50`` midreset50's (0.5), so
# the flow time runs past the clean end. Cache and self starts, like the reset variants; RoboCasa ``sdiag_self13``.
SHOOT_ENTRY_T = {"overshoot": 1.0, "mid_shoot": 0.75, "mid_shoot50": 0.5}
GROOT_SHOOT_MODES = {"warmshoot": "overshoot", "midshoot": "mid_shoot", "midshoot50": "mid_shoot50"}
SELF_SHOOT_MODES = {f"self{mode}": variant for mode, variant in GROOT_SHOOT_MODES.items()}
WARM_ARM_PREFIXES = ("warm", *WARM_VARIANT_MODES, *SELF_VARIANT_MODES, *GROOT_SHOOT_MODES, *SELF_SHOOT_MODES)


def is_self_mode(mode: str) -> bool:
    """A self-start served mode: the start comes from a direct inference on the decision, not the cache."""
    return mode in SELF_VARIANT_MODES or mode in SELF_SHOOT_MODES


def split_warm_steps(arm_id: str) -> Tuple[str, Optional[int]]:
    """``(arm id without the step suffix, N)`` of a warm-family arm id carrying an explicit Euler-step count
    (``midreset_t0.75_n1`` -> ``("midreset_t0.75", 1)``); ``(arm_id, None)`` when it carries none."""
    head, sep, n = arm_id.rpartition("_n")
    if sep and n.isdigit() and "_t" in head:
        return head, int(n)
    return arm_id, None


def warm_steps_of(arm_id: str) -> Optional[int]:
    """Explicit Euler-step count N of a warm-reset arm id (GR00T LIBERO, ``..._n<N>``), or None: N then follows
    the schedule, ``remaining_steps(start_t)`` (every RoboCasa arm and every exact-resume arm)."""
    return split_warm_steps(arm_id)[1]


def warm_t_of(arm_id: str) -> float:
    """start_t named by a warm-family arm id (``warm_t0.2``, ``warmreset_t0.2``, ``midreset_t0.75_n1``)."""
    base = split_warm_steps(arm_id)[0]
    for prefix in WARM_ARM_PREFIXES:
        head = f"{prefix}_t"
        if base.startswith(head):
            return float(base[len(head):])
    raise ValueError(f"{arm_id!r} is not a warm-family arm id")


def warm_mode_of(arm_id: str) -> str:
    for prefix in sorted(WARM_ARM_PREFIXES, key=len, reverse=True):
        if arm_id.startswith(f"{prefix}_t"):
            return prefix
    raise ValueError(f"{arm_id!r} is not a warm-family arm id")


def executed_steps_of(env_id: str, arm_id: str) -> int:
    """Euler steps every decision of ``arm_id`` executes under ``env_id`` (its equal-NFE budget m)."""
    env = resolve_env(env_id)
    if arm_id == "full":
        return env.k_full
    if arm_id.startswith("plain_k"):
        return int(arm_id[len("plain_k"):])
    n = warm_steps_of(arm_id)
    return int(n) if n is not None else env.remaining_steps(warm_t_of(arm_id))


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

# 2026-09-21 follow-up, second round (owner-approved): the warm-start continuation variants and their
# two equal-NFE references re-run at 500 episodes per task on the two diagnostic tasks, under a
# separate experiment id and a separate out root so nothing mixes with the 50/100-episode Q-B cells.
# Seeds stay RC_FORMAL_BASE_SEED + idx (idx 0..499), so the first 50/100 identities coincide with Q-B.
VAR500_EXPERIMENT_ID = "sdiag_var500"
VAR500_EPISODES = 500
VAR500_ARMS = ("plain_k2", "warm_t0.2", "warmreset_t0.2", "warmshoot_t0.2")
VAR500_TASKS = ("CloseFridge", "PickPlaceCounterToStove")

# Third round (owner 2026-09-21 23:50): the same four arms x two tasks x 50 episodes on the historical
# evaluation segment (seed 1,000,000+idx, the segment of the nfe_baseline ladders and ws_search), as a
# cross-check that the 2,000,000 segment of this line carries no seed-segment effect. Own experiment id
# and out root; the driver refuses this seed under any other experiment id.
RC_XCHECK_BASE_SEED = 1_000_000
XSEED_EXPERIMENT_ID = "sdiag_xseed1m"
XSEED_EPISODES = 50

# Fourth round (owner 2026-09-22 00:00): the macro view — the variant arms on the whole 13-task RoboCasa
# roster and the three Q-B references on the six tasks Q-B did not cover, 50 episodes each, seed 2M.
# Own experiment id and out root; the analysis merges this root with the Q-B root (arms_root list).
MACRO13_EXPERIMENT_ID = "sdiag_macro13"
MACRO13_EPISODES = 50
MACRO13_ARMS = ("full", "plain_k2", "warm_t0.2", "warmreset_t0.2", "warmshoot_t0.2",
                "resetfinal_t0.1", "resetfinal_t0.2", "resetfinal_t0.3",  # resetfinal added 2026-09-22 (owner)
                "midfinal_t0.2")  # midfinal 13-task round queued 2026-09-22 (owner)
# GR00T mirror (owner 2026-09-22: symmetric to pi0.5, no 500-episode round, no warmshoot). K = 4, ascending:
# warm_t0.75 resumes 1 step, warm_t0.5 resumes 2 steps.
MACRO13_ARMS_BY_POLICY = {
    "pi05": MACRO13_ARMS,
    "groot": ("full", "plain_k1", "plain_k2", "warm_t0.75", "warm_t0.5", "warmreset_t0.75", "warmreset_t0.5",
              "resetfinal_t0.75", "resetfinal_t0.5", "midfinal_t0.75", "midfinal_t0.5",
              "midfinal50_t0.75", "midfinal50_t0.5", "midreset_t0.75", "midreset_t0.5",
              "midreset50_t0.75", "midreset50_t0.5"),
}
# Self-start ablation round (owner 2026-09-24): every warm-reset configuration shown on the macro-13 page, with
# the self start, on the 13-task roster, 50 episodes per task, seed 2M (pairs with the macro-13 cache arms).
SELF13_EXPERIMENT_ID = "sdiag_self13"
SELF13_EPISODES = 50
SELF13_ARMS_BY_POLICY = {
    "pi05": ("selfwarmreset_t0.2", "selfresetfinal_t0.2", "selfmidfinal_t0.2"),
    "groot": ("selfwarmreset_t0.75", "selfmidreset_t0.75", "selfmidreset50_t0.75",
              "selfresetfinal_t0.75", "selfmidfinal_t0.75", "selfmidfinal50_t0.75",
              "selfwarmreset_t0.5", "selfmidreset_t0.5", "selfresetfinal_t0.5", "selfmidfinal_t0.5", "selfmidfinal50_t0.5",
              # shoot ablation (owner 2026-09-24): no reset, warm-reset step size, cache and self starts
              "warmshoot_t0.75", "midshoot_t0.75", "midshoot50_t0.75", "warmshoot_t0.5", "midshoot_t0.5",
              "selfwarmshoot_t0.75", "selfmidshoot_t0.75", "selfmidshoot50_t0.75", "selfwarmshoot_t0.5",
              "selfmidshoot_t0.5"),
}
# LIBERO self-start round (owner 2026-09-24, logs/step_diag_libero_selfstart_plan.log.md): the macro-13 page's
# configurations on LIBERO spatial and libero_10, on the frozen pruned A pool (10 tasks x init_idx 0..49 = 500
# episodes per arm and suite, LIBERO env seed 7, as the LIBERO shadow). Own experiment id and out roots.
LIBERO_SELF_EXPERIMENT_ID = "sdiag_libero_self"
LIBERO_SELF_EPISODES = 50
LIBERO_SELF_SUITES = ("libero_spatial", "libero_10")
LIBERO_N_TASKS = 10
LIBERO_ENV_SEED = 7
# GR00T K = 8 (owner 2026-09-24): every warm-reset arm keeps RoboCasa's K = 4 (T, N, t) tuple, so the Euler-step
# count N is decoupled from the snapshot's start_t and named in the arm id, ``<mode>_t<native start_t>_n<N>``:
# N = 1 starts from the T = 0.25 snapshot (native 0.75), N = 2 from the T = 0.5 snapshot (native 0.5). A
# final-action arm takes the final chunk of the entry retrieved at that same start_t and shares its yaml. The
# exact resume (ours) stays on the native 8-step grid: warm_t0.875 (N = 1) and warm_t0.75 (N = 2).
GROOT_LIBERO_START_T_BY_N = {1: 0.75, 2: 0.5}
GROOT_LIBERO_RESET_MODES = ("warmreset", "midreset", "midreset50", "resetfinal", "midfinal", "midfinal50")
_GROOT_LIBERO_CACHE_ARMS = tuple(f"{mode}_t{t:g}_n{n}" for n, t in GROOT_LIBERO_START_T_BY_N.items()
                                 for mode in GROOT_LIBERO_RESET_MODES)
LIBERO_SELF_ARMS_BY_POLICY = {
    "pi05": ("full", "plain_k2", "warm_t0.2", "warmreset_t0.2", "selfwarmreset_t0.2", "resetfinal_t0.2",
             "selfresetfinal_t0.2", "midfinal_t0.2", "selfmidfinal_t0.2"),
    "groot": ("full", "plain_k1", "plain_k2", "warm_t0.875", "warm_t0.75", *_GROOT_LIBERO_CACHE_ARMS,
              *(f"self{arm}" for arm in _GROOT_LIBERO_CACHE_ARMS)),
}
# seed-segment cross-check arms / tasks per policy (pi0.5 = the VAR500 set; GR00T = one cliff + one flat Q-B task)
XSEED_ARMS_BY_POLICY = {"pi05": VAR500_ARMS, "groot": ("plain_k1", "warm_t0.75", "warmreset_t0.75", "resetfinal_t0.75")}
XSEED_TASKS_BY_POLICY = {"pi05": VAR500_TASKS, "groot": ("TurnOnSinkFaucet", "PickPlaceCounterToStove")}
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
    warm_family = (mode == "warm" or mode in WARM_VARIANT_MODES or mode in SELF_VARIANT_MODES
                   or mode in GROOT_SHOOT_MODES or mode in SELF_SHOOT_MODES)
    expected_arm = f"plain_k{exec_steps}" if mode == "plain" else mode
    if not warm_family and arm_id != expected_arm:
        raise ValueError(f"mode/step identity requires arm_id={expected_arm}")
    if mode == "shadow" or warm_family:
        cfg = yaml.safe_load(pathlib.Path(cache_config).read_text())
        judge = cfg["checkpoints"]["cp1"]["judge"]
        if warm_family:
            base, n_steps = split_warm_steps(arm_id)
            if not base.startswith(f"{mode}_t"):
                raise ValueError(f"{mode} arm must be named {mode}_t<start_t>")
            t = float(base[len(mode) + 2:])
            if env.benchmark == "robocasa365":
                if t not in QB_WARM_TS[env.policy] or n_steps is not None:
                    raise ValueError("warm start_t is outside the frozen arm set")
            else:
                if t not in env.warm_ts:
                    raise ValueError("warm start_t is outside the frozen arm set")
                # GR00T LIBERO warm-reset arms name their step count (K=8 keeps RoboCasa's K=4 tuples);
                # the exact resume and every pi0.5 arm run remaining_steps(start_t)
                named = env.policy == "groot" and mode != "warm"
                if named != (n_steps is not None) or (n_steps is not None and not 1 <= n_steps <= env.k_full):
                    raise ValueError(f"{arm_id}: GR00T LIBERO warm-reset arms are named <mode>_t<start_t>_n<N> "
                                     "(1 <= N <= K); no other warm arm carries a step count")
            if mode in ("midshoot", "midshoot50", *SELF_SHOOT_MODES) and env.policy != "groot":
                raise ValueError("the midshoot / self shoot variants are GR00T only (pi0.5 ran warmshoot)")
            if (mode in GROOT_SHOOT_MODES or mode in SELF_SHOOT_MODES) and env.benchmark != "robocasa365":
                raise ValueError("the shoot variants are a RoboCasa365 ablation only")
            expected = {"type": "always_warm_start", "start_t": t}
        else:
            expected = {"type": "threshold", "threshold": 99.0} if env.policy == "pi05" else {"type": "always_hit"}
        if judge != expected:
            raise ValueError(f"judge differs from {arm_id}: expected {expected}")
