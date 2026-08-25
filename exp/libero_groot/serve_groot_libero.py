"""Serve GR00T N1.5 on the LIBERO wire protocol, optionally collecting embeddings.

Runs in the GR00T island (``/home/weiland/gr00t_n15_venv/.venv``) and speaks
``examples/libero/main.py``'s wire format, so the existing LIBERO client drives
it unchanged. Two modes:

  * plain teacher -- what the collection campaign and the anchor arm use;
  * ``--collect-hdf5 DIR`` -- additionally records per-episode CP1 embeddings
    for offline library building, via the same ``GrootCacheCollector`` the
    RoboCasa365 line uses.

``--concurrent`` serves many simultaneous connections from one loaded model
(ported from ``exp/robocasa365/serve_groot_n15.py``, which the RoboCasa365
search proved out): only the GPU policy and the read-only storage backend are
shared, everything mutable is rebuilt per connection. It is refused together
with ``--collect-hdf5`` -- collection hangs per-episode state off one runner and
is single-connection by construction.

Yaml hot-swap is off by default: the served configuration is then carried by
the process, which is what makes a cell scheduler's results impossible to
attribute to another cell's weights -- it restarts the server per cell and pays
for that property. ``--allow-dynamic-bundles`` trades it away deliberately, for
a driver that owns the swap schedule (the conductor sends ``load_cache_config``
once per stage and names the bundle on every episode's connection); the GR00T
guards then re-run per bundle rather than once at startup. Either way
``select_bundle("default")`` stays an idempotent no-op, which is what the
conductor's ``LiberoEpisodeRunner`` issues on every connection.

⚠ Data config is ``examples/Libero/custom_data_config.py:LiberoDataConfig``
(two cameras, seven scalar action keys). The key builder must therefore be a
``cp1_groot_libero_*`` type: the three-camera RoboCasa builders assert three
image-token runs and would reject every LIBERO observation.

Example::

    PYTHONPATH=/home/weiland/gr00t_n15:/home/weiland/gr00t_n15/examples/Libero:\\
    /home/weiland/openpi:/home/weiland/openpi/src \\
    /home/weiland/gr00t_n15_venv/.venv/bin/python \\
      exp/libero_groot/serve_groot_libero.py \\
      --checkpoint /home/weiland/ckpt_n15_libero_spatial --port 8030 \\
      --collect-hdf5 /data/libero_cache/build_spatial
"""

from __future__ import annotations

import argparse
import os
import pathlib
import threading
import uuid
from typing import Any

DEFAULT_CHECKPOINT = "/home/weiland/ckpt_n15_libero_spatial"
DEFAULT_PORT = 8030
EMBODIMENT_TAG = "new_embodiment"
# The published LIBERO numbers were produced with ``--denoising-steps 8``
# (examples/Libero/README.md), not with the value baked into the checkpoint
# config. ``Gr00tPolicy`` writes it onto ``action_head.num_inference_timesteps``,
# which the staged runner reads too, so the cache split stays in lockstep.
DEFAULT_DENOISING_STEPS = 8


class _InferLockedPolicy:
    """Serialize ``infer`` across connections; everything else passes through.

    One shared lock strictly serializes the GPU-touching ``infer`` path while
    the other connections' sim stepping and socket I/O overlap. Lifecycle hooks
    stay outside the lock: they run per-connection-serial in the server handler
    and their only shared touch point (the storage backend) is read-only by
    construction (``write_policy=never`` is enforced at config load).

    ``__getattr__`` delegation keeps the ``hasattr`` surface identical to the
    wrapped stack -- the WebSocket server feature-detects every lifecycle hook,
    so a wrapper that hid one would silently disable episode resets, which on
    the collection path is exactly the failure that produced zero HDF5.
    """

    def __init__(self, inner: Any, lock: threading.Lock) -> None:
        self._inner = inner
        self._lock = lock

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            return self._inner.infer(obs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def _require_default_bundle(bundle_id: str) -> None:
    """Fail fast instead of serving the CLI yaml under another name.

    This factory only knows the configuration it was started with, so acking
    any other bundle id would be a silent provenance mismatch.
    """
    if bundle_id != "default":
        raise ValueError(
            f"GR00T LIBERO concurrent server serves only bundle_id='default' "
            f"(its CLI config); got {bundle_id!r}. Restart the server with the "
            f"desired --cache-config instead of select_bundle."
        )


def _resolve_bundle(
    bundle_id: str,
    *,
    cli_config: Any,
    cli_storage: Any,
    allow_dynamic: bool,
) -> tuple[Any, Any]:
    """Return the ``(config, shared_storage)`` this connection is served under.

    With hot-swap disabled this is the CLI configuration and nothing else --
    ``_require_default_bundle`` rejects any other id rather than acking it.

    With ``--allow-dynamic-bundles`` the conductor drives configuration: each
    stage sends ``load_cache_config`` and every episode of that stage then opens
    a connection naming the stage's ``bundle_id``. Two things matter here.

    First, the *guards must re-run on the loaded config*. ``load_cache_config``
    runs only the generic validator, so a hot-swapped yaml would otherwise reach
    serving with an unsatisfiable WARM_START silently downgraded to MISS, a CP3
    checkpoint built and never consulted, or a three-camera RoboCasa builder
    that rejects every LIBERO observation. The startup checks protect the CLI
    config; nothing protected the loaded one.

    Second, the storage is *read, never rebuilt*. The server's
    ``load_cache_config`` handler already called ``build_shared_storage`` and
    hung the result off the bundle; building it again here would mean one
    gigabyte-scale artifact load per connection per arm.

    A missing bundle under the default id is not an error: on a server started
    with ``--cache-config`` that slot means the startup configuration, and on a
    teacher-only server it means no cache at all (``None``) -- which is what the
    runner's opening ``select_bundle("default")`` is asking for in both cases.
    """
    if not allow_dynamic:
        _require_default_bundle(bundle_id)
        return cli_config, cli_storage

    from openpi.serving.websocket_policy_server import get_current_cache_bundle

    bundle = get_current_cache_bundle(bundle_id)
    if bundle is None:
        if bundle_id == "default":
            return cli_config, cli_storage
        raise ValueError(
            f"no cache bundle is registered under bundle_id={bundle_id!r}; "
            "load_cache_config must precede the first connection that names it, "
            "otherwise this connection would silently be served the startup "
            "configuration under another id."
        )

    from openpi.cache.groot.load_guard import (
        validate_artifact_identity,
        validate_groot_cache_config,
    )

    config = bundle.cache_config
    validate_groot_cache_config(config, allow_hysteresis_gate=True)
    _check_libero_builder(config.key_builder.type, lambda m: (_ for _ in ()).throw(ValueError(m)))
    validate_artifact_identity(bundle.shared_storage, config)
    return config, bundle.shared_storage


def _check_libero_builder(builder_type: str, fail) -> None:
    """The three-camera RoboCasa builders reject every LIBERO observation."""
    if not builder_type.startswith("cp1_groot_libero"):
        fail(
            f"key_builder.type {builder_type!r} is not a LIBERO builder; the "
            "three-camera RoboCasa builders assert three image-token runs and "
            "reject every LIBERO observation."
        )


def _build_concurrent_factory(policy: Any, args: Any) -> tuple[Any, str]:
    """Per-connection policy factory for concurrent serving.

    Shares exactly two things across connections: the GPU policy (guarded by
    the infer lock) and the storage backend (via per-connection facades). Every
    mutable component -- key_builder, gates, judges, strategies, timer,
    orchestrator, staged runner, adapter -- is built fresh per connection.
    """
    # Imported here, not at module scope: this module stays importable (and its
    # translation layer unit-testable) in the main venv, where ``gr00t`` and the
    # island's torch are absent.
    from exp.libero_groot.policy_adapter import GrootLiberoPolicyAdapter

    lock = threading.Lock()
    allow_dynamic = bool(getattr(args, "allow_dynamic_bundles", False))

    if not args.cache_config and not allow_dynamic:

        def teacher_factory(shared_base_policy: Any, bundle_id: str = "default") -> Any:
            _require_default_bundle(bundle_id)
            return _InferLockedPolicy(GrootLiberoPolicyAdapter(shared_base_policy), lock)

        return teacher_factory, "concurrent teacher-only (no cache)"

    from openpi.cache.config import (
        build_per_connection_components,
        build_shared_storage,
        load_cache_config,
        validate_cache_config,
    )
    from openpi.cache.groot.interceptor import GrootCacheInterceptor
    from openpi.cache.groot.load_guard import (
        validate_artifact_identity,
        validate_groot_cache_config,
    )
    from openpi.cache.groot.staged import GrootStagedRunner
    from openpi.cache.orchestrator import CacheOrchestrator

    # With --allow-dynamic-bundles and no --cache-config the server starts with
    # no configuration at all and receives every arm over the wire; the guards
    # then run per bundle in ``_resolve_bundle`` instead of here.
    config = None
    shared_storage = None
    if args.cache_config:
        config = load_cache_config(args.cache_config)
        validate_cache_config(config)
    # The generic validator permits recipes the two-stage split cannot honour
    # and that fail *silently*: an unsatisfiable WARM_START is downgraded to
    # MISS (a GR00T library never carries intermediates), a CP3 checkpoint is
    # built and never consulted, a non-``always_search`` gate changes what
    # ``searched`` means downstream.
    #
    # ``allow_hysteresis_gate`` is this entry point's explicit opt-in for the
    # gate-threshold Pareto experiment. It is claimed here rather than relaxed
    # in the guard because the sibling RoboCasa365 server shares that guard and
    # its analysis still assumes every step searched. The claim is only valid
    # while this line's analysis reads ``searched`` -- which it does: gate-skip
    # steps are counted as teacher calls in the Pareto's x-axis.
    if config is not None:
        validate_groot_cache_config(config, allow_hysteresis_gate=True)
        _check_libero_builder(
            config.key_builder.type, lambda m: (_ for _ in ()).throw(ValueError(m))
        )
        shared_storage = build_shared_storage(config)
        # ``load_artifact`` only compares ``vector_dims``, and mean-pool and
        # max-pool libraries are dimensionally identical -- nothing else would ever
        # notice a swapped artifact.
        validate_artifact_identity(shared_storage, config)

    def cache_factory(shared_base_policy: Any, bundle_id: str = "default") -> Any:
        conn_config, conn_storage = _resolve_bundle(
            bundle_id,
            cli_config=config,
            cli_storage=shared_storage,
            allow_dynamic=allow_dynamic,
        )
        if conn_config is None:
            # Dynamic bundles enabled but nothing loaded yet: serve the teacher.
            # Refusing instead would break the runner's opening handshake, which
            # selects "default" before the first stage has been sent.
            return _InferLockedPolicy(GrootLiberoPolicyAdapter(shared_base_policy), lock)
        components = build_per_connection_components(conn_config, conn_storage, quiet=True)
        timer = components["timer"]
        if conn_config.timer.output_csv_dir:
            # Per-connection subdirectory: the per-task CSV name is only
            # (task ordinal, second) and every connection counts from task 0,
            # so two connections writing one directory would silently
            # overwrite each other's latency evidence.
            conn_dir = os.path.join(
                conn_config.timer.output_csv_dir, f"conn_{uuid.uuid4().hex[:8]}"
            )
            os.makedirs(conn_dir, exist_ok=True)
            timer.enable_csv(conn_dir)
        orchestrator = CacheOrchestrator(
            storage=components["storage"],
            key_builder=components["key_builder"],
            gates=components["gates"],
            judges=components["judges"],
            search_strategies=components["search_strategies"],
            timer=timer,
            write_policy=components["write_policy"],
            offline_writers=components["offline_writers"],
            library_stats=components["library_stats"],
        )
        runner = GrootStagedRunner(shared_base_policy.model, timer=timer)
        interceptor = GrootCacheInterceptor(
            shared_base_policy, runner, orchestrator=orchestrator, timer=timer
        )
        return _InferLockedPolicy(GrootLiberoPolicyAdapter(interceptor), lock)

    if config is None:
        return cache_factory, "concurrent cache -> dynamic bundles (no startup yaml)"
    return cache_factory, (
        f"concurrent cache -> {args.cache_config} ({config.key_builder.type})"
        + (" + dynamic bundles" if allow_dynamic else "")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--denoising-steps",
        type=int,
        default=DEFAULT_DENOISING_STEPS,
        help="Flow-matching inference steps. Default reproduces the official "
        "evaluation recipe; the checkpoint config's own value is lower.",
    )
    parser.add_argument(
        "--cache-config",
        default=None,
        help="YAML cache config; routes inference through the CP1 cache.",
    )
    parser.add_argument(
        "--collect-hdf5",
        default=None,
        help="Directory for per-episode HDF5 embeddings. Mutually exclusive "
        "with --cache-config: a library must be collected from the teacher's "
        "own actions, and with the cache active some recorded actions would be "
        "replayed library entries.",
    )
    parser.add_argument(
        "--experiment",
        default="groot_libero",
        help="Collection subdirectory name under --collect-hdf5.",
    )
    parser.add_argument(
        "--concurrent",
        action="store_true",
        help="Serve simultaneous connections from one loaded model via a "
        "per-connection policy factory. Default OFF: without it the server is "
        "one connection at a time, which is what collection requires.",
    )
    parser.add_argument(
        "--allow-dynamic-bundles",
        action="store_true",
        help="Accept load_cache_config over the wire, so one process can serve "
        "successive cache configurations addressed by bundle_id. Default OFF: "
        "configuration identity is otherwise carried by the process, which is "
        "what makes a cell's results impossible to attribute to another cell's "
        "weights. Turn it on only for a driver that owns the swap schedule "
        "(the conductor); the guards then re-run per bundle.",
    )
    args = parser.parse_args()

    if args.cache_config and args.collect_hdf5:
        parser.error("--cache-config and --collect-hdf5 are mutually exclusive")
    if args.concurrent and args.collect_hdf5:
        parser.error(
            "--concurrent cannot be combined with --collect-hdf5: the collector "
            "hangs per-episode state off one runner and one HDF5 writer, so two "
            "connections would interleave into the same episode buffer."
        )
    if args.allow_dynamic_bundles and not args.concurrent:
        parser.error(
            "--allow-dynamic-bundles requires --concurrent: the server only "
            "consults a bundle when building a per-connection policy, so "
            "without the factory a loaded yaml would be acked and never served."
        )
    if args.allow_dynamic_bundles and args.collect_hdf5:
        parser.error(
            "--allow-dynamic-bundles cannot be combined with --collect-hdf5: "
            "collection writes one HDF5 file per run and its provenance is the "
            "process's single configuration, so swapping the library underneath "
            "it would put entries from two configurations in one artifact."
        )

    from gr00t.model.policy import Gr00tPolicy
    from openpi.serving import websocket_policy_server

    from custom_data_config import LiberoDataConfig  # examples/Libero on PYTHONPATH

    from exp.libero_groot.policy_adapter import GrootLiberoPolicyAdapter

    checkpoint = pathlib.Path(args.checkpoint)
    data_config = LiberoDataConfig()

    print(f"loading policy from {checkpoint}", flush=True)
    policy: Any = Gr00tPolicy(
        model_path=str(checkpoint),
        embodiment_tag=EMBODIMENT_TAG,
        modality_config=data_config.modality_config(),
        modality_transform=data_config.transform(),
        denoising_steps=args.denoising_steps,
        device="cuda",
    )
    print(f"denoising steps: {policy.denoising_steps}", flush=True)

    served: Any = None
    stack = ""
    if args.concurrent:
        # The factory owns the whole per-connection stack; building the
        # single-connection one too would load the artifact a second time.
        factory, stack = _build_concurrent_factory(policy, args)
    elif args.collect_hdf5:
        from openpi.cache.groot.staged import GrootStagedRunner

        from openpi.cache.types import VISION_0, VISION_1

        from exp.robocasa365.groot_cache_collector import GrootCacheCollector

        runner = GrootStagedRunner(policy.model)
        served = GrootLiberoPolicyAdapter(
            GrootCacheCollector(
                policy,
                runner,
                out_dir=args.collect_hdf5,
                experiment=args.experiment,
                # LIBERO feeds two cameras; the slicer's three-run default
                # would reject every observation.
                vision_fields=(VISION_0, VISION_1),
            )
        )
        stack = f"collector -> {args.collect_hdf5}/{args.experiment}"
    elif args.cache_config:
        from openpi.cache.config import (
            build_cache_components,
            load_cache_config,
            validate_cache_config,
        )
        from openpi.cache.groot.interceptor import GrootCacheInterceptor
        from openpi.cache.groot.load_guard import validate_groot_cache_config
        from openpi.cache.groot.staged import GrootStagedRunner

        config = load_cache_config(args.cache_config)
        validate_cache_config(config)
        # Recipes the two-stage split cannot honour fail *silently* otherwise:
        # an unsatisfiable WARM_START is downgraded to MISS, a CP3 checkpoint is
        # built and never consulted, a non-always_search gate changes what
        # ``searched`` means downstream. ``allow_hysteresis_gate`` is this entry
        # point's explicit opt-in (see the concurrent path above for why it is
        # claimed per entry point rather than relaxed in the shared guard).
        validate_groot_cache_config(config, allow_hysteresis_gate=True)
        _check_libero_builder(config.key_builder.type, parser.error)
        components = build_cache_components(config)
        runner = GrootStagedRunner(policy.model)
        served = GrootLiberoPolicyAdapter(
            GrootCacheInterceptor(policy, runner, **components)
        )
        stack = f"cache -> {args.cache_config} ({config.key_builder.type})"
    else:
        served = GrootLiberoPolicyAdapter(policy)
        stack = "teacher-only (no cache, no collection)"

    print(f"serving stack: {stack}", flush=True)
    if args.concurrent:
        server = websocket_policy_server.WebsocketPolicyServer(
            policy,
            host="0.0.0.0",
            port=args.port,
            metadata={"concurrent": True, "denoising_steps": args.denoising_steps},
            concurrent=True,
            connection_policy_factory=factory,
            # Off by default: the factory then only ever builds the CLI config,
            # so an ack for a loaded bundle would be a silent provenance
            # mismatch. ``select_bundle("default")`` stays an idempotent no-op
            # either way, which is what the conductor's LiberoEpisodeRunner
            # issues on every connection.
            allow_dynamic_bundles=args.allow_dynamic_bundles,
        )
    else:
        server = websocket_policy_server.WebsocketPolicyServer(
            policy=served, host="0.0.0.0", port=args.port
        )
    print(f"SERVER-LISTENING on 0.0.0.0:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
