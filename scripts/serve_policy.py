import dataclasses
import enum
import logging
import os
from pathlib import Path
import socket

import tyro

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


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

    # Enable concurrent multi-client mode.  When True, each WebSocket
    # connection gets its own InferenceInterceptor / CacheOrchestrator / Timer
    # wrapper stack while sharing the same base policy (GPU model).  Timing
    # summary prints and orchestrator info logs are suppressed to avoid
    # interleaved output.
    concurrent: bool = False

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


def create_default_policy(env: EnvMode, *, default_prompt: str | None = None) -> _policy.Policy:
    """Create a default policy for the given environment."""
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        return _policy_config.create_trained_policy(
            _config.get_config(checkpoint.config), checkpoint.dir, default_prompt=default_prompt
        )
    raise ValueError(f"Unsupported environment mode: {env}")


def create_policy(args: Args) -> _policy.Policy:
    """Create a policy from the given arguments."""
    match args.policy:
        case Checkpoint():
            return _policy_config.create_trained_policy(
                _config.get_config(args.policy.config), args.policy.dir, default_prompt=args.default_prompt
            )
        case Default():
            return create_default_policy(args.env, default_prompt=args.default_prompt)


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


def _wrap_policy(base_policy, args: Args, *, quiet: bool = False, eager: bool = False, shared_cache=None):
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
    """
    policy = base_policy

    # Dynamic bundle from load_cache_config control message (highest priority).
    # Must be checked before args.cache_config so that server started without
    # --cache_config can still pick up bundles injected at runtime.
    from openpi.serving.websocket_policy_server import get_current_cache_bundle

    bundle = get_current_cache_bundle()
    if bundle is not None:
        from openpi.cache.config import build_per_connection_components
        from openpi.cache.interceptor import InferenceInterceptor
        from openpi.cache.orchestrator import CacheOrchestrator

        components = build_per_connection_components(
            bundle.cache_config,
            bundle.shared_storage,
            quiet=True,
        )
        orchestrator = CacheOrchestrator(
            storage=components["storage"],
            key_builder=components["key_builder"],
            gates=components["gates"],
            judges=components["judges"],
            search_strategies=components["search_strategies"],
            timer=components["timer"],
        )
        policy = InferenceInterceptor(
            policy,
            timer=components["timer"],
            orchestrator=orchestrator,
            eager=eager,
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
            cache_config = load_cache_config(args.cache_config)
            components = build_per_connection_components(
                cache_config,
                shared_cache["storage"],
                quiet=True,
            )
        else:
            cache_config = load_cache_config(args.cache_config)
            components = build_cache_components(cache_config)
            if quiet:
                components["timer"]._quiet = True
                logging.getLogger("openpi.cache.orchestrator").setLevel(logging.WARNING)

        orchestrator = CacheOrchestrator(
            storage=components["storage"],
            key_builder=components["key_builder"],
            gates=components["gates"],
            judges=components["judges"],
            search_strategies=components["search_strategies"],
            timer=components["timer"],
        )
        policy = InferenceInterceptor(
            policy,
            timer=components["timer"],
            orchestrator=orchestrator,
            eager=eager,
        )
    elif args.cache:
        from openpi.cache.interceptor import InferenceInterceptor
        from openpi.cache.timing import SystemTimer
        timer = SystemTimer(enabled=True, output_csv_dir=args.timing_csv_dir, quiet=quiet)
        if args.timing_csv_dir:
            logging.info("Timing CSV output enabled: writing to %s", args.timing_csv_dir)
        policy = InferenceInterceptor(policy, timer=timer, eager=eager)

    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    if args.collect:
        from openpi.collect.collection_policy import CollectionPolicy
        from openpi.collect.data_collector import EpisodeDataCollector
        collector = EpisodeDataCollector(base_dir=args.collect_dir)
        policy = CollectionPolicy(policy, collector)
        logging.info("Data collection enabled -> %s", args.collect_dir)

    return policy


def main(args: Args) -> None:
    _configure_torchinductor_cache_dir()
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
        shared_cache = None
        if args.cache_config is not None:
            from openpi.cache.config import build_shared_storage, load_cache_config
            cache_config = load_cache_config(args.cache_config)
            shared_cache = {"storage": build_shared_storage(cache_config)}

        def _connection_policy_factory(shared_base_policy):
            return _wrap_policy(
                shared_base_policy, args, quiet=True, eager=True,
                shared_cache=shared_cache,
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
        policy = _wrap_policy(base_policy, args, quiet=False)
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
