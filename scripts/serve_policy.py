import dataclasses
import enum
import logging
import os
from pathlib import Path
import socket
from typing import TYPE_CHECKING

import tyro

from openpi.models_pytorch.stage_device_placement import StageDeviceConfig
from openpi.models_pytorch.stage_device_placement import relocate_model_stages
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
    """Server runtime contract (C2): backend is read-only; write_policy MUST be 'never'.

    At server runtime the backend is frozen (no insert/batch_insert/delete) and
    any write attempt raises ``BackendFrozenError``. Rather than silently
    neutralising a write-enabled config, this entry point fails fast: any
    non-``"never"`` write_policy raises ``ConfigValidationError`` at server
    start (and on ``load_cache_config``). Set ``write_policy: {type: never}``
    explicitly in the cache config, and build / enrich cache artifacts with
    offline tools (e.g. ``exp/common/factor_postprocess.py``) — those do not go
    through this entry point and keep their configured write policy.
    """
    from openpi.cache.config import ConfigValidationError

    if cache_config.write_policy.type == "never":
        return cache_config

    raise ConfigValidationError(
        f"write_policy.type {cache_config.write_policy.type!r} is not allowed at "
        "server runtime (hard constraint C2: the runtime backend is write-frozen). "
        "Set write_policy: {type: never} in the cache config and build cache "
        "artifacts offline (e.g. exp/common/factor_postprocess.py)."
    )


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

    # Number of serving replicas (separate processes) to run behind a single
    # public ``port``. ``1`` (default) is the ordinary single-process server.
    # ``>1`` runs a supervisor: it spawns R child servers on internal loopback
    # ports ``port+1 .. port+R`` and fronts them with an in-process
    # connection-sticky router (``openpi.serving.replica_proxy``). Scale-out is
    # process-level because the single-process ceiling is the GIL-serialized
    # CUDA kernel-launch path; each child has its own interpreter. Requires
    # concurrent mode (incompatible with --non-concurrent).
    replicas: int = 1

    # Replica spawn batching (only used when replicas>1). Spawn this many child
    # processes concurrently, wait for that batch to finish loading + binding,
    # then spawn the next batch. ``0`` (default) spawns all at once. Use a small
    # value (e.g. 2) on hosts where N simultaneous model loads would spike host
    # RAM past a cgroup limit (jupyter: 32GB) — stagger so peak load RAM stays
    # bounded.
    replica_spawn_batch: int = 0

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

    # Gate-research per-step collection. Tri-state override of the YAML
    # ``collection`` block: None => use YAML; otherwise CLI wins. Distinct from
    # the legacy ``--collect`` (forward-hook HDF5) path above.
    export_collect_meta: bool | None = None
    # Comma-separated field list; overrides ``collection.collect_fields`` when set.
    collect_fields: str | None = None

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
    # connection baseline path. Hard constraint C1 keeps that path structurally
    # the raw single-connection Policy (no coordinator / bundle / interceptor);
    # its numerics match the current model, which runs sdpa attention (set once
    # in PI0Pytorch.__init__) — NOT the pre-Phase-5 eager baseline. See
    # logs/full_repo_audit_2026-05-26.log.md §3.2.
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


def _disable_compile_for_serving(train_config: "_config.TrainConfig") -> "_config.TrainConfig":
    """Force eager execution for the serving runtime.

    Serving runs the model through the batching coordinator's worker threads
    (and, with a cache config, the interceptor's own staged compile). The
    model-level ``torch.compile`` of ``sample_actions`` is unused by that path
    and only costs multi-minute autotune compilation per replica at startup, so
    the serving entrypoint loads the model eager (``pytorch_compile_mode=None``).
    """
    model = train_config.model
    if hasattr(model, "pytorch_compile_mode"):
        model = dataclasses.replace(model, pytorch_compile_mode=None)
    return dataclasses.replace(train_config, model=model)


def create_default_policy(
    env: EnvMode,
    *,
    default_prompt: str | None = None,
    pytorch_device: str | None = None,
) -> _policy.Policy:
    """Create a default policy for the given environment."""
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        return _policy_config.create_trained_policy(
            _disable_compile_for_serving(_config.get_config(checkpoint.config)),
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
                _disable_compile_for_serving(_config.get_config(args.policy.config)),
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


def _configure_server_gpu_memory_lock() -> None:
    """Keep CUDA allocator reservations owned by the server process.

    PyTorch can free tensors internally without returning its cached CUDA blocks
    to the driver. That is the behaviour we want for long-running experiment
    servers: the process may reuse memory normally, but from ``nvidia-smi`` the
    server's VRAM footprint should not drop and then let another process take
    the gap. Existing cleanup paths still run ``gc.collect()`` and delete wrapper
    state; this only suppresses ``torch.cuda.empty_cache()``, which is the call
    that hands unused cached blocks back to the system.

    Set ``OPENPI_SERVER_GPU_MEMORY_LOCK=0`` to restore the old release-to-driver
    behaviour for debugging or one-off local runs.
    """
    raw = os.environ.get("OPENPI_SERVER_GPU_MEMORY_LOCK", "1").strip().lower()
    if raw in {"0", "false", "no", "off"}:
        logging.info("OPENPI_SERVER_GPU_MEMORY_LOCK=0; torch.cuda.empty_cache() unchanged")
        return

    try:
        import torch
    except Exception as exc:  # pragma: no cover - torch is always present here.
        logging.warning("GPU memory lock requested but torch import failed: %s", exc)
        return

    cuda = getattr(torch, "cuda", None)
    if cuda is None or not hasattr(cuda, "empty_cache"):
        logging.info("GPU memory lock requested but torch.cuda.empty_cache is unavailable")
        return

    current = cuda.empty_cache
    if getattr(current, "_openpi_gpu_memory_lock_noop", False):
        return

    def _locked_empty_cache(*_args, **_kwargs) -> None:
        logging.debug(
            "Suppressed torch.cuda.empty_cache(); server GPU memory lock keeps "
            "cached CUDA blocks reserved by this process."
        )

    _locked_empty_cache._openpi_gpu_memory_lock_noop = True  # type: ignore[attr-defined]
    _locked_empty_cache._openpi_original_empty_cache = current  # type: ignore[attr-defined]
    cuda.empty_cache = _locked_empty_cache
    logging.info(
        "Server GPU memory lock enabled: torch.cuda.empty_cache() is suppressed; "
        "unused CUDA blocks remain reserved for this process. Set "
        "OPENPI_SERVER_GPU_MEMORY_LOCK=0 to opt out."
    )


def _resolve_collection(cache_config, args) -> tuple[bool, tuple[str, ...]]:
    """Effective (export_collect_meta, collect_fields) from YAML + CLI tri-state.

    CLI wins only when explicitly set (not None), so ``--export-collect-meta``
    unset leaves the YAML ``collection`` block authoritative.
    """
    from openpi.cache.config import validate_effective_collection

    coll = cache_config.collection
    export = coll.export_collect_meta if args.export_collect_meta is None else bool(args.export_collect_meta)
    if args.collect_fields is None:
        fields = tuple(coll.collect_fields)
    else:
        fields = tuple(f.strip() for f in args.collect_fields.split(",") if f.strip())
    # Legacy --collect (forward-hook HDF5) and gate collection are mutually
    # exclusive. Check the EFFECTIVE export (YAML / bundle-enabled too), not just
    # the raw CLI flag that _validate_collect_isolation already covers.
    if export and getattr(args, "collect", False):
        raise ValueError(
            "--collect (legacy forward-hook HDF5) and gate-research collection "
            "(collection.export_collect_meta / --export-collect-meta) are "
            "mutually exclusive; enable only one."
        )
    # Re-validate the EFFECTIVE config: load_cache_config() validated the YAML
    # before these CLI overrides, so this is the only place a CLI-enabled
    # collection gets the C5 always-search / field / frame-cap hard gate.
    validate_effective_collection(cache_config, export_collect_meta=export, collect_fields=fields)
    return export, fields


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
        _cm_export, _cm_fields = _resolve_collection(bundle.cache_config, args)
        policy = InferenceInterceptor(
            policy,
            timer=components["timer"],
            orchestrator=orchestrator,
            eager=eager,
            collect_images=need_images,
            stage_config=stage_config,
            coordinator=coordinator,
            bundle_id=bundle_id,
            export_collect_meta=_cm_export,
            collect_fields=_cm_fields,
            collect_kb_id=bundle.cache_config.key_builder.type,
        )
    elif args.cache_config is not None:
        from openpi.cache.config import build_cache_components
        from openpi.cache.config import build_per_connection_components
        from openpi.cache.config import load_cache_config
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
        _cm_export, _cm_fields = _resolve_collection(cache_config, args)
        policy = InferenceInterceptor(
            policy,
            timer=components["timer"],
            orchestrator=orchestrator,
            eager=eager,
            collect_images=need_images,
            stage_config=stage_config,
            coordinator=coordinator,
            bundle_id=bundle_id,
            export_collect_meta=_cm_export,
            collect_fields=_cm_fields,
            collect_kb_id=cache_config.key_builder.type,
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


def _configure_monitor_level() -> None:
    """Resolve OPENPI_MONITOR_LEVEL and prime PyTorch memory history if asked.

    Reads the env var once at startup and (for HISTORY level) flips on
    ``torch.cuda.memory._record_memory_history`` *before* any model weight
    is loaded so the recorded trace covers the entire process lifecycle.
    Cheaper levels (OFF / BASIC / SNAPSHOT) just log the choice.
    """
    from openpi.serving import monitor as _monitor

    level = _monitor.get_monitor_level()
    logging.info("OPENPI_MONITOR_LEVEL=%s", level.name)
    if level >= _monitor.MonitorLevel.HISTORY:
        result = _monitor.enable_memory_history()
        if result.get("ok"):
            logging.info("torch memory history recording enabled at startup")
        else:
            logging.warning(
                "torch memory history recording requested but failed: %s",
                result.get("reason", "unknown"),
            )


def _validate_collect_isolation(args: Args) -> None:
    """Reject --collect combined with concurrent or multi-replica serving.

    Embedding collection (``openpi.collect.CollectionPolicy``) attaches forward
    hooks to the shared base model on every ``infer()`` call. Those hooks are
    module-global, so under concurrent connections — or the batching
    coordinator's worker threads — one connection's forward fires another
    connection's hook and the captured tensors cross-contaminate, leaving the
    recorded HDF5 silently corrupt or empty. Multiple replicas would
    additionally race on identical ``collect_dir`` filenames across processes.
    Collection is only correct on the single-connection C1 path with one
    replica, so fail fast here instead of writing bad data.
    """
    if not args.collect:
        return
    # The legacy forward-hook HDF5 collector (--collect) and the gate-research
    # per-step collector (--export-collect-meta / collection block) are distinct,
    # mutually-exclusive subsystems (docs/data_collection/guide.md). Running both
    # at once is a configuration error.
    if args.export_collect_meta:
        raise ValueError(
            "--collect (legacy forward-hook HDF5) and --export-collect-meta "
            "(gate-research per-step collection) are mutually exclusive; enable "
            "only one."
        )
    if args.replicas > 1:
        raise ValueError(
            f"--collect requires a single replica (got --replicas {args.replicas}). "
            "Embedding hooks cannot be isolated across replica processes; "
            "use --replicas 1 --non-concurrent."
        )
    if args.concurrent and not args.non_concurrent:
        raise ValueError(
            "--collect requires the non-concurrent single-connection path; "
            "concurrent forward hooks cross-contaminate captured embeddings. "
            "Pass --non-concurrent (concurrent mode is the default)."
        )


def main(args: Args) -> None:
    _validate_collect_isolation(args)
    if args.replicas > 1:
        _run_supervisor(args)
    else:
        _serve_single(args)


def _serve_replica(args: "Args", ready_event) -> None:
    """Child entry point for one replica (spawned by the supervisor).

    Binds loopback-only: children must be reachable ONLY through the router on
    the public port, never directly — a direct connection to an internal port
    would bypass sticky routing, control-plane broadcast, and metric aggregation.
    """
    logging.basicConfig(level=logging.INFO, force=True)
    _serve_single(args, ready_callback=ready_event.set, bind_host="127.0.0.1")


def _run_supervisor(args: Args) -> None:
    """Spawn R replica processes on internal ports + front them with the router.

    Children bind ``port+1 .. port+R`` on loopback (not exposed); only the
    public ``port`` is served, by the in-process connection-sticky router.
    """
    import multiprocessing as mp
    import threading

    if args.non_concurrent:
        raise ValueError("--replicas>1 requires concurrent mode (remove --non-concurrent)")

    # CUDA requires the 'spawn' start method (fork after CUDA init is unsafe).
    ctx = mp.get_context("spawn")

    internal_ports = [args.port + 1 + i for i in range(args.replicas)]
    batch = args.replica_spawn_batch if args.replica_spawn_batch > 0 else args.replicas
    procs: list = []
    stop_watchdog = threading.Event()

    def _terminate_all() -> None:
        for p in procs:
            if p.is_alive():
                p.terminate()
        for p in procs:
            p.join(timeout=10)

    try:
        # Spawn in batches: start `batch` children, wait for that batch to load
        # + bind, then start the next. Bounds peak host-RAM from concurrent
        # model loads on memory-constrained hosts.
        for start in range(0, args.replicas, batch):
            group = []
            for ip in internal_ports[start:start + batch]:
                ev = ctx.Event()
                child_args = dataclasses.replace(args, port=ip, replicas=1)
                p = ctx.Process(target=_serve_replica, args=(child_args, ev), daemon=False)
                p.start()
                procs.append(p)
                group.append((ip, ev, p))
                logging.info("Spawned replica pid=%d on internal port %d", p.pid, ip)
            for ip, ev, p in group:
                # Fail fast: poll readiness in 5s ticks (180 × 5s = 900s budget)
                # and abort immediately if the child exits during load instead
                # of blocking the full timeout.
                ready = False
                for _ in range(180):
                    if ev.wait(timeout=5.0):
                        ready = True
                        break
                    if not p.is_alive():
                        raise RuntimeError(
                            f"replica on port {ip} (pid {p.pid}) exited during load "
                            f"(exitcode={p.exitcode}) — aborting supervisor"
                        )
                if not ready:
                    raise RuntimeError(f"replica on port {ip} did not become ready within 900s")
                logging.info("Replica on internal port %d READY", ip)

        # Liveness watchdog: run_proxy() below blocks forever, so a replica that
        # crashes AFTER becoming ready (e.g. CUDA OOM mid-run) would otherwise go
        # unnoticed — the router would keep routing sticky connections to a dead
        # loopback port, and the crashed child's GPU memory would never be
        # reclaimed. The watchdog tears the whole supervisor down (terminate
        # siblings, exit non-zero) the moment any replica dies.
        def _watchdog() -> None:
            while not stop_watchdog.wait(5.0):
                dead = [(ip, p) for ip, p in zip(internal_ports, procs) if not p.is_alive()]
                if dead:
                    logging.error(
                        "replica(s) died: ports=%s exitcodes=%s — shutting down supervisor",
                        [ip for ip, _ in dead], [p.exitcode for _, p in dead],
                    )
                    _terminate_all()
                    os._exit(1)

        threading.Thread(target=_watchdog, name="replica-watchdog", daemon=True).start()

        from openpi.serving.replica_proxy import run_proxy

        logging.info("All %d replicas ready; starting router on public port %d -> %s",
                     args.replicas, args.port, internal_ports)
        run_proxy(public_port=args.port, backend_ports=internal_ports)
    finally:
        stop_watchdog.set()
        _terminate_all()


def _serve_single(args: Args, ready_callback=None, bind_host: str = "0.0.0.0") -> None:
    # M6 (Phase 5): ``--non-concurrent`` overrides the new default concurrent
    # mode to provide the single-connection baseline path (hard constraint C1).
    if args.non_concurrent:
        args = dataclasses.replace(args, concurrent=False, non_concurrent=False)
    _configure_server_gpu_memory_lock()
    _configure_monitor_level()
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
            from openpi.cache.config import build_shared_storage
            from openpi.cache.config import load_cache_config
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
        # Allow runtime sweep of coordinator params via env vars without
        # touching the yaml — useful for finding the right window for the
        # current worker count (M7 sweep). Defaults match the plan defaults.
        import os as _os_bc

        from openpi.serving.batching_coordinator import BatchingCoordinator
        _max_wait_ms = float(_os_bc.environ.get("BATCHING_MAX_WAIT_MS", "10"))
        _max_batch_size = int(_os_bc.environ.get("BATCHING_MAX_BATCH_SIZE", "8"))
        _coordinator = BatchingCoordinator(
            base_policy._model,
            max_batch_size=_max_batch_size,
            max_wait_ms=_max_wait_ms,
        )
        _coordinator.start()
        logging.info(
            "BatchingCoordinator params: max_batch_size=%d max_wait_ms=%.1f",
            _max_batch_size, _max_wait_ms,
        )
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
            host=bind_host,
            port=args.port,
            metadata=policy_metadata,
            concurrent=True,
            connection_policy_factory=_connection_policy_factory,
            ready_callback=ready_callback,
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
            host=bind_host,
            port=args.port,
            metadata=policy_metadata,
            ready_callback=ready_callback,
        )

    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
