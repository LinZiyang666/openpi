import dataclasses
import enum
import logging
import os
from pathlib import Path
import socket
from typing import TYPE_CHECKING

import tyro

from openpi.models_pytorch.stage_device_placement import (
    StageDeviceConfig,
    relocate_model_stages,
)
from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config

if TYPE_CHECKING:
    from openpi.cache.config import CacheConfig


# ------------------------------------------------------------------
# Runtime write-frozen contract (C2) — M4.5
# ------------------------------------------------------------------

def _enforce_runtime_write_policy(cache_config: "CacheConfig") -> "CacheConfig":
    """Server runtime contract (C2): backend is read-only; write_policy must be 'never'.

    Existing yaml configs default to ``on_any_miss`` so sweep workflows can
    keep using their unmodified yamls. At server runtime, however, the
    backend is frozen (no insert/batch_insert/delete) and any write attempt
    raises ``BackendFrozenError``. We therefore auto-override the policy to
    ``"never"`` with a single warning per server start, rather than
    fail-fast on the first MISS episode. Offline tools (artifact build /
    enrich) do not go through this entry point and keep their configured
    write policy.
    """
    from openpi.cache.config import WritePolicyConfig

    if cache_config.write_policy.type == "never":
        return cache_config

    original_type = cache_config.write_policy.type
    overridden = dataclasses.replace(
        cache_config,
        write_policy=WritePolicyConfig(type="never"),
    )
    logging.warning(
        "write_policy %r auto-overridden to 'never' for runtime serving (hard "
        "constraint C2: runtime backend is write-frozen). To write cache "
        "artifacts, use offline tools (e.g. exp/common/factor_postprocess.py). "
        "This override is applied per server start; the source yaml is unchanged.",
        original_type,
    )
    return overridden


class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi0_aloha_sim").
    config: str
    # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
    dir: str


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # Environment to serve the policy for. This is only used when serving default policies.
    env: EnvMode = EnvMode.ALOHA_SIM

    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    default_prompt: str | None = None

    # Port to serve the policy on.
    port: int = 8000
    # Record the policy's behavior for debugging.
    record: bool = False
    # Enable per-episode embedding collection to HDF5.
    collect: bool = False
    # Root directory for collected episode files.
    collect_dir: str = "./data"

    # Pass model-input images (post-transform, mask-filtered) through cache
    # check() kwargs so that KeyBuilder implementations can optionally use them.
    # Only takes effect when --cache or --cache_config is also set.
    collect_images: bool = False

    # Enable the staged inference cache system.
    # When True, inference is routed through InferenceInterceptor (run_stage1/2/3).
    # External behavior (actions, timing fields) is identical to the default path.
    cache: bool = False

    # Directory to write per-task timing CSV files.  Only takes effect when
    # --cache is also set (requires InferenceInterceptor / SystemTimer).
    # Each client connection produces one file named
    # timing_task_NNNN_YYYYMMDD_HHMMSS.csv in the specified directory.
    # When None (default), timing data is printed to the terminal at task end
    # but not written to disk.
    # Example: --timing_csv_dir /tmp/timing
    timing_csv_dir: str | None = None

    # Path to a YAML cache config file. When set, loads full cache components
    # (backend, key_builder, gate, judge, search_strategy) from the YAML file
    # and assembles a CacheOrchestrator. Overrides --cache and --timing_csv_dir.
    # Example: --cache_config cache.yaml
    cache_config: str | None = None

    # Enable concurrent multi-client mode (default).  When True, each WebSocket
    # connection gets its own InferenceInterceptor / CacheOrchestrator / Timer
    # wrapper stack while sharing the same base policy (GPU model).  Timing
    # summary prints and orchestrator info logs are suppressed to avoid
    # interleaved output.
    #
    # Phase 5 (M6) flipped this default to True so sweep workflows naturally
    # run as 1-server × N-bundle. Pass ``--non-concurrent`` (mapped below) or
    # ``--no-concurrent`` (tyro's negation) to opt back into the single-
    # connection baseline path. Hard constraint C1 guarantees the single-
    # connection path stays bit-identical to pre-Phase-5 behavior.
    concurrent: bool = True

    # Opt-out flag for the single-connection (a.k.a. baseline / C1) path.
    # When True, overrides ``concurrent=True`` to ``False``. Provided as a
    # discoverable user-facing CLI argument since flipping ``concurrent``'s
    # default would otherwise hide the opt-out behind tyro's ``--no-`` prefix.
    non_concurrent: bool = False

    # Per-stage device placement. Default: None (no override, legacy behavior).
    # Set all three to enable split-device or meta placement.
    # Example: --stage1_device cuda:0 --stage2_device meta --stage3_device meta
    stage1_device: str | None = None
    stage2_device: str | None = None
    stage3_device: str | None = None

    # verdict_factor_judge B2: filesystem root for warmup dump JSONL files.
    # When set, the WebSocket server creates the directory with mode 0o700 +
    # ownership self-check at startup, registers it with
    # ``websocket_policy_server.set_warmup_dump_root``, and accepts the
    # ``fetch_dump`` / ``preload_normalizer_buffer`` / ``unload_warmup_buffer``
    # ctrl messages plus ``dump.deferred=true`` resolution in
    # ``load_cache_config``. Leave None on production / non-warmup runs.
    warmup_dump_root: str | None = None

    # Specifies how to load the policy. If not provided, the default policy for the environment will be used.
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)


# Default checkpoints that should be used for each environment.
DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint(
        config="pi05_aloha",
        dir="gs://openpi-assets/checkpoints/pi05_base",
    ),
    EnvMode.ALOHA_SIM: Checkpoint(
        config="pi05_aloha_sim",
        dir="gs://openpi-assets/checkpoints/pi05_base",
    ),
    EnvMode.DROID: Checkpoint(
        config="pi05_droid",
        dir="gs://openpi-assets/checkpoints/pi05_droid",
    ),
    EnvMode.LIBERO: Checkpoint(
        config="pi05_libero",
        dir="gs://openpi-assets/checkpoints/pi05_libero",
    ),
}


def create_default_policy(
    env: EnvMode,
    *,
    default_prompt: str | None = None,
    pytorch_device: str | None = None,
) -> _policy.Policy:
    """Create a default policy for the given environment."""
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        return _policy_config.create_trained_policy(
            _config.get_config(checkpoint.config),
            checkpoint.dir,
            default_prompt=default_prompt,
            pytorch_device=pytorch_device,
        )
    raise ValueError(f"Unsupported environment mode: {env}")


def _get_stage_device_config(args: Args) -> StageDeviceConfig:
    """Build and validate StageDeviceConfig from CLI args."""
    return StageDeviceConfig.create(
        args.stage1_device, args.stage2_device, args.stage3_device,
    )


def create_policy(args: Args) -> _policy.Policy:
    """Create a policy with optional per-stage device placement.

    Three-layer loading path:
      1. Legacy Default: all stage devices None -> existing auto-select.
      2. All Same Device: direct load to that device.
      3. Split/Meta: load to CPU, then relocate per-stage.
    """
    stage_config = _get_stage_device_config(args)

    # Startup guard: split/meta requires cache/interceptor path
    if stage_config.needs_relocation and not (args.cache or args.cache_config):
        raise ValueError(
            "Split device placement requires --cache or --cache_config. "
            "Without cache, Policy.infer() uses single-device staged path."
        )
    # Meta stages require --cache_config (not just --cache).
    # --cache creates InferenceInterceptor without orchestrator, so CP1 never
    # hits and stage2/3 always execute — meta stages would fail.
    if stage_config.has_meta_stage and not args.cache_config:
        raise ValueError(
            "Meta stage placement requires --cache_config (not just --cache). "
            "--cache creates an interceptor without orchestrator, so stage2/3 "
            "always execute. Use --cache_config with always_hit judge."
        )

    # Determine pytorch_device for create_trained_policy()
    if stage_config.is_all_same_device:
        pytorch_device = stage_config.stage1
    elif stage_config.needs_relocation:
        pytorch_device = "cpu"
    else:
        pytorch_device = None  # legacy default

    match args.policy:
        case Checkpoint():
            policy = _policy_config.create_trained_policy(
                _config.get_config(args.policy.config),
                args.policy.dir,
                default_prompt=args.default_prompt,
                pytorch_device=pytorch_device,
            )
        case Default():
            policy = create_default_policy(
                args.env,
                default_prompt=args.default_prompt,
                pytorch_device=pytorch_device,
            )

    if stage_config.needs_relocation:
        relocate_model_stages(policy._model, stage_config)  # noqa: SLF001
        policy._pytorch_device = stage_config.primary_device  # noqa: SLF001

    return policy


def _configure_torchinductor_cache_dir() -> None:
    """Set a stable torch.compile cache path under the current working directory.

    If TORCHINDUCTOR_CACHE_DIR is already set by the caller, keep it unchanged.
    """
    if os.environ.get("TORCHINDUCTOR_CACHE_DIR"):
        return

    cache_dir = Path.cwd() / ".cache" / "torch" / "inductor"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache_dir)
    logging.info("TORCHINDUCTOR_CACHE_DIR=%s", cache_dir)


def _wrap_policy(
    base_policy,
    args: Args,
    *,
    quiet: bool = False,
    eager: bool = False,
    shared_cache=None,
    stage_config: StageDeviceConfig | None = None,
    bundle_id: str = "default",
):
    """Build the wrapper chain around a base policy.

    Wrapper ordering matters:
      1. InferenceInterceptor (innermost -- needs direct Policy access)
      2. PolicyRecorder (records interceptor's output)
      3. CollectionPolicy (outermost -- hooks into model internals via _model)
    DO NOT reorder without verifying CollectionPolicy._model lookup.

    Args:
        base_policy: The unwrapped policy (shared GPU model).
        args: CLI arguments.
        quiet: When True, suppress timing prints and orchestrator info logs
               (used in concurrent mode).
        shared_cache: Pre-built cache components from build_cache_components().
               When provided, only storage is reused (thread-safe);
               key_builder/timer/gates/judges/strategies are created fresh per call.
               In concurrent mode this dict also carries the shared
               ``BatchingCoordinator`` instance under ``"coordinator"`` so
               every connection's InferenceInterceptor batches stage1/2/3
               forwards across connections.
        bundle_id: The bundle identifier this connection's wrapper stack is
               bound to (M2 BundleDispatcher). Selects which loaded bundle
               feeds the cache wrapper and is forwarded to the
               InferenceInterceptor so the coordinator can keep bundle-level
               request bookkeeping.
    """
    policy = base_policy
    coordinator = (shared_cache or {}).get("coordinator")

    # Dynamic bundle from load_cache_config control message (highest priority).
    # Must be checked before args.cache_config so that server started without
    # --cache_config can still pick up bundles injected at runtime.
    #
    # M2 (Phase 3): when a concrete ``bundle_id`` is selected (via
    # ``__ctrl__: select_bundle`` or the first ``episode_start`` carrying a
    # bundle id) the WebsocketPolicyServer factory passes it in here so we
    # bind to that specific bundle instead of the legacy single-latest
    # pointer.
    from openpi.serving.websocket_policy_server import get_current_cache_bundle

    bundle = get_current_cache_bundle(bundle_id) or get_current_cache_bundle()
    if bundle is not None:
        from openpi.cache.config import build_per_connection_components
        from openpi.cache.interceptor import InferenceInterceptor
        from openpi.cache.orchestrator import CacheOrchestrator

        components = build_per_connection_components(
            bundle.cache_config,
            bundle.shared_storage,
            yaml_id=bundle.yaml_id,
            quiet=True,
        )
        need_images = args.collect_images or bundle.cache_config.key_builder.type == "clip"
        orchestrator = CacheOrchestrator(
            storage=components["storage"],
            key_builder=components["key_builder"],
            gates=components["gates"],
            judges=components["judges"],
            search_strategies=components["search_strategies"],
            timer=components["timer"],
            write_policy=components.get("write_policy"),
            offline_writers=components.get("offline_writers", ()),
            library_stats=components.get("library_stats"),
        )
        policy = InferenceInterceptor(
            policy,
            timer=components["timer"],
            orchestrator=orchestrator,
            eager=eager,
            collect_images=need_images,
            stage_config=stage_config,
            coordinator=coordinator,
            bundle_id=bundle_id,
        )
    elif args.cache_config is not None:
        from openpi.cache.config import (
            build_cache_components,
            build_per_connection_components,
            load_cache_config,
        )
        from openpi.cache.interceptor import InferenceInterceptor
        from openpi.cache.orchestrator import CacheOrchestrator

        if args.cache:
            logging.warning("--cache_config overrides --cache. Ignoring --cache flag.")

        if shared_cache is not None:
            cache_config = _enforce_runtime_write_policy(
                load_cache_config(args.cache_config)
            )
            components = build_per_connection_components(
                cache_config,
                shared_cache["storage"],
                quiet=True,
            )
        else:
            cache_config = _enforce_runtime_write_policy(
                load_cache_config(args.cache_config)
            )
            components = build_cache_components(cache_config)
            if quiet:
                components["timer"]._quiet = True
                logging.getLogger("openpi.cache.orchestrator").setLevel(logging.WARNING)

        # CLIP key builder requires input images for online encoding.
        need_images = args.collect_images or cache_config.key_builder.type == "clip"

        orchestrator = CacheOrchestrator(
            storage=components["storage"],
            key_builder=components["key_builder"],
            gates=components["gates"],
            judges=components["judges"],
            search_strategies=components["search_strategies"],
            timer=components["timer"],
            write_policy=components.get("write_policy"),
            offline_writers=components.get("offline_writers", ()),
            library_stats=components.get("library_stats"),
        )
        policy = InferenceInterceptor(
            policy,
            timer=components["timer"],
            orchestrator=orchestrator,
            eager=eager,
            collect_images=need_images,
            stage_config=stage_config,
            coordinator=coordinator,
            bundle_id=bundle_id,
        )
    elif args.cache:
        from openpi.cache.interceptor import InferenceInterceptor
        from openpi.cache.timing import SystemTimer
        timer = SystemTimer(enabled=True, output_csv_dir=args.timing_csv_dir, quiet=quiet)
        if args.timing_csv_dir:
            logging.info("Timing CSV output enabled: writing to %s", args.timing_csv_dir)
        policy = InferenceInterceptor(
            policy, timer=timer, eager=eager,
            collect_images=args.collect_images, stage_config=stage_config,
            coordinator=coordinator,
            bundle_id=bundle_id,
        )

    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    if args.collect:
        from openpi.collect.collection_policy import CollectionPolicy
        from openpi.collect.data_collector import EpisodeDataCollector
        collector = EpisodeDataCollector(base_dir=args.collect_dir)
        policy = CollectionPolicy(policy, collector)
        logging.info("Data collection enabled -> %s", args.collect_dir)

    return policy


def _setup_warmup_dump_root(raw_path: str | None) -> None:
    """Create the warmup dump directory + register it with the WS module.

    No-op when ``raw_path`` is None. Enforces mode 0o700 + same-uid
    ownership so a multi-user host cannot drop a symlink into the directory
    after server start; the resolved path is stored on the WS module so the
    ``fetch_dump`` / ``unload_warmup_buffer`` ctrl handlers can range-check
    against it.
    """
    if not raw_path:
        return
    root = Path(raw_path).expanduser().resolve()
    root.mkdir(mode=0o700, exist_ok=True)
    st = root.stat()
    if st.st_uid != os.getuid():
        raise RuntimeError(
            f"warmup_dump_root {root} is not owned by uid={os.getuid()}: refusing"
        )
    if st.st_mode & 0o077:
        raise RuntimeError(
            f"warmup_dump_root {root} is group/world accessible (mode={oct(st.st_mode)}): refusing"
        )
    websocket_policy_server.set_warmup_dump_root(root)
    logging.info("verdict_factor_judge warmup dump root configured: %s", root)


def main(args: Args) -> None:
    # M6 (Phase 5): ``--non-concurrent`` overrides the new default concurrent
    # mode to provide the single-connection baseline path (hard constraint C1).
    if args.non_concurrent:
        args = dataclasses.replace(args, concurrent=False, non_concurrent=False)
    _configure_torchinductor_cache_dir()
    _setup_warmup_dump_root(args.warmup_dump_root)
    stage_config = _get_stage_device_config(args)
    base_policy = create_policy(args)
    policy_metadata = base_policy.metadata

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    if args.concurrent:
        # Concurrent mode: the base policy (GPU model) is shared.
        # Each connection gets its own wrapper stack via the factory.
        # Only storage is shared (Qdrant client is thread-safe);
        # key_builder/timer/gates/judges/strategies are per-connection (have mutable state).
        shared_cache: dict | None = None
        if args.cache_config is not None:
            from openpi.cache.config import build_shared_storage, load_cache_config
            cache_config = _enforce_runtime_write_policy(
                load_cache_config(args.cache_config)
            )
            shared_cache = {"storage": build_shared_storage(cache_config)}

        # Phase 4 M1: spawn the BatchingCoordinator once per server process
        # and share across all per-connection InferenceInterceptor instances.
        # The coordinator owns three background threads (one per stage); the
        # connection-side InferenceInterceptor enqueues per-request payloads
        # and blocks on the reply event. Hard constraint C1 ensures
        # ``--non-concurrent`` skips this entire wiring.
        from openpi.serving.batching_coordinator import BatchingCoordinator

        _coordinator = BatchingCoordinator(base_policy._model)
        _coordinator.start()
        if shared_cache is None:
            shared_cache = {}
        shared_cache["coordinator"] = _coordinator
        logging.info("BatchingCoordinator started for concurrent serving.")

        def _connection_policy_factory(shared_base_policy, bundle_id="default"):
            return _wrap_policy(
                shared_base_policy, args, quiet=True, eager=True,
                shared_cache=shared_cache, stage_config=stage_config,
                bundle_id=bundle_id,
            )

        policy_metadata = {**policy_metadata, "concurrent": True}
        logging.info("Concurrent mode enabled.")

        server = websocket_policy_server.WebsocketPolicyServer(
            policy=base_policy,
            host="0.0.0.0",
            port=args.port,
            metadata=policy_metadata,
            concurrent=True,
            connection_policy_factory=_connection_policy_factory,
        )
    else:
        # Single-connection mode: wrap once at startup.
        policy = _wrap_policy(base_policy, args, quiet=False, stage_config=stage_config)
        if args.cache_config:
            logging.info("Cache mode enabled via config: %s", args.cache_config)
        elif args.cache:
            logging.info("Cache mode enabled (simple, no config).")

        server = websocket_policy_server.WebsocketPolicyServer(
            policy=policy,
            host="0.0.0.0",
            port=args.port,
            metadata=policy_metadata,
        )

    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
