"""Serve the official RoboCasa365 GR00T N1.5 teacher over a websocket.

This is the second-teacher counterpart to the pi0.5 server used earlier in the
cross-scene cache experiment.  It loads the released checkpoint as-is: there is
no training and no fine-tuning anywhere in this pipeline.

Where everything lives (all paths verified on the host named below)
------------------------------------------------------------------
host            weilandserver, single RTX 4090 (49140 MiB), shared with other
                sessions.  A ``serve_policy.py`` owned by someone else occupies
                port 8000 and ~8.8 GB and must never be shut down; only ever act
                on your own PID / tmux session, never a broad ``pkill``.
port            8020  (8000 = foreign server, 8010 = the pi0.5 teacher)
checkpoint      /home/weiland/ckpt_n15_robocasa/gr00t_n1-5/multitask_learning/checkpoint-120000
                7.2 GB, two safetensors shards, embodiment tag "new_embodiment".
this venv       /home/weiland/gr00t_n15_venv/.venv
                py3.11.15, numpy 1.26.4, transformers 4.51.3, torch 2.5.1+cu124.
gr00t source    NOT installed in that venv -- imported from the n1.5-release
                git worktree at /home/weiland/gr00t_n15, so PYTHONPATH must
                include it.
extra runtime   ``decord`` is required: the parent DataConfig's transform chain
                loads it through a transformers dynamic module, and without it
                construction fails with an ImportError.
extra deps      ``uv pip install -e /home/weiland/openpi/packages/openpi-client``
                (its numpy<2.0.0 pin is satisfied here, so no --no-deps needed --
                unlike the simulation island, where --no-deps is mandatory).
PYTHONPATH      /home/weiland/gr00t_n15:/home/weiland/openpi/src:/home/weiland/openpi
                the middle entry pulls in the websocket server only;
                that module imports nothing heavier than openpi_client and
                websockets, and openpi/__init__.py is empty, so no jax or torch
                gets dragged in from this repo.
sim client      runs in a *different* island (py3.12 / numpy 2.2.5), which is why
                this is a server rather than a single in-process script.

Launch::

    tmux new-session -d -s grootsrv "export HOME=/home/weiland; \\
      PYTHONPATH=/home/weiland/gr00t_n15:/home/weiland/openpi/src:/home/weiland/openpi \\
      /home/weiland/gr00t_n15_venv/.venv/bin/python \\
      /home/weiland/openpi/exp/robocasa365/serve_groot_n15.py 2>&1 | tee /tmp/grootsrv.log"
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pathlib
import threading
import uuid
from typing import Any

import numpy as np

from exp.robocasa365 import groot_keys
from exp.robocasa365.groot_policy_adapter import GrootPolicyAdapter

logging.basicConfig(level=logging.INFO)

# The target-posttrained branch: the one the cross-scene experiment settled on
# after the admission gates, and the one every dimension in the cache plan was
# read from. The multitask checkpoint is kept only as an archived comparison
# and must be requested explicitly.
DEFAULT_CHECKPOINT = (
    "/home/weiland/ckpt_n15_robocasa_tp/gr00t_n1-5/foundation_model_learning/"
    "target_posttraining/atomic_seen/checkpoint-60000"
)
ARCHIVED_MULTITASK_CHECKPOINT = (
    "/home/weiland/ckpt_n15_robocasa/gr00t_n1-5/multitask_learning/checkpoint-120000"
)
DEFAULT_PORT = 8020
EMBODIMENT_TAG = "new_embodiment"


# ------------------------------------------------------------------
# Deterministic-seed wrapper (diagnostics only)
# ------------------------------------------------------------------


class _SeededPolicy:
    """Reset the global torch RNG immediately before each inference.

    The N1.5 action head is flow-matching: every ``get_action`` call starts from
    a fresh ``torch.randn`` sample, so repeated calls on identical input differ.
    Pinning the seed here -- inside the server process, adjacent to the call --
    is what makes the wire-parity check reproducible.  Seeding from the client
    would accomplish nothing: client and server are separate interpreters and do
    not share a global RNG.

    Diagnostics only.  Leaving this on during a real rollout would drive every
    step from the same noise sample.
    """

    def __init__(self, policy: Any, seed: int) -> None:
        self._policy = policy
        self._seed = seed

    def get_action(self, observations: dict[str, Any]) -> dict[str, Any]:
        import torch

        torch.manual_seed(self._seed)
        return self._policy.get_action(observations)

    def get_modality_config(self) -> dict[str, Any]:
        return self._policy.get_modality_config()


# ------------------------------------------------------------------
# Start-up handshake
# ------------------------------------------------------------------


def _dummy_observation(checkpoint: pathlib.Path) -> dict[str, Any]:
    """Build a *legal* observation for the start-up self-check.

    Two of the state fields are quaternions.  Zero-filling them -- the obvious
    thing to do for a smoke input -- does not describe a rotation, and the
    quaternion -> matrix -> rotation_6d path turns it into non-finite values, so
    the probe would either mask a real fault or reject a healthy server.  They
    are set to the identity quaternion instead, and the remaining state fields
    are taken from the checkpoint's own per-key means so every value sits inside
    the domain the model was normalized against.
    """
    stats = json.loads((checkpoint / "experiment_cfg" / "metadata.json").read_text())
    state_stats = stats[EMBODIMENT_TAG]["statistics"]["state"]

    obs: dict[str, Any] = {}
    for key in groot_keys.VIDEO_KEYS:
        resolution = groot_keys.MODEL_IMAGE_RESOLUTION
        obs[key] = np.zeros((resolution, resolution, 3), dtype=np.uint8)

    for key in groot_keys.STATE_KEYS:
        vector = np.asarray(
            state_stats[key.removeprefix("state.")]["mean"], dtype=np.float64
        )
        if key in groot_keys.QUATERNION_STATE_KEYS:
            # The mean of a set of quaternions is not itself unit-norm, and a
            # non-unit quaternion is not a rotation.  Renormalising keeps the
            # probe both in-distribution and geometrically valid; falling back
            # to a fixed identity would instead feed two of the five state
            # fields values the model never saw in training.
            norm = float(np.linalg.norm(vector))
            vector = (
                np.asarray(groot_keys.IDENTITY_QUATERNION_WXYZ, dtype=np.float64)
                if norm < 1e-8
                else vector / norm
            )
        obs[key] = vector

    for key in groot_keys.LANGUAGE_KEYS:
        obs[key] = "pick up the object"
    return obs


def _handshake(
    adapter: GrootPolicyAdapter, policy: Any, checkpoint: pathlib.Path
) -> None:
    """Fail loudly at start-up rather than serving a mis-wired policy."""
    modality = policy.get_modality_config()
    for name, expected in (
        ("video", groot_keys.VIDEO_KEYS),
        ("state", groot_keys.STATE_KEYS),
        ("action", groot_keys.ACTION_KEYS),
        ("language", groot_keys.LANGUAGE_KEYS),
    ):
        actual = list(modality[name].modality_keys)
        print(f"  {name:9s} {actual}", flush=True)
        if actual != expected:
            raise RuntimeError(
                f"{name} modality keys/order mismatch: {actual} != {expected}"
            )

    horizon = len(modality["action"].delta_indices)
    if horizon != groot_keys.ACTION_HORIZON:
        raise RuntimeError(f"action horizon {horizon} != {groot_keys.ACTION_HORIZON}")

    result = adapter.infer(_dummy_observation(checkpoint))
    for key, value in result["actions"].items():
        print(f"  {key:38s} shape={value.shape} dtype={value.dtype}", flush=True)
        if not np.isfinite(value).all():
            raise RuntimeError(
                f"handshake inference produced non-finite values for {key}"
            )

    print(f"  checkpoint: {checkpoint}", flush=True)


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------


def _build_served_policy(policy: Any, args: Any) -> tuple[Any, str]:
    """Pick exactly one of the three serving stacks and return it with a label.

    Plain teacher / cache-aware / collecting are mutually exclusive by
    construction: whichever is chosen is the single object handed to the
    adapter, so there is never more than one wrapper in the chain.
    """
    if not args.cache_config and not args.collect_hdf5:
        return policy, "teacher-only (no cache, no collection)"

    from openpi.cache.groot.staged import GrootStagedRunner

    if args.collect_hdf5:
        from exp.robocasa365.groot_cache_collector import GrootCacheCollector

        runner = GrootStagedRunner(policy.model)
        return (
            GrootCacheCollector(policy, runner, out_dir=args.collect_hdf5),
            f"collector -> {args.collect_hdf5}",
        )

    from openpi.cache.config import build_cache_components, load_cache_config, validate_cache_config
    from openpi.cache.groot.interceptor import GrootCacheInterceptor
    from openpi.cache.groot.load_guard import (
        validate_artifact_identity,
        validate_groot_cache_config,
    )
    from openpi.cache.orchestrator import CacheOrchestrator

    config = load_cache_config(args.cache_config)
    validate_cache_config(config)
    validate_groot_cache_config(config)

    components = build_cache_components(config)
    validate_artifact_identity(components["storage"], config)

    timer = components["timer"]
    if config.timer.output_csv_dir:
        # Mirror scripts/serve_policy.py's --timing_csv_dir path: the yaml
        # field alone sets the directory but leaves the legacy auto-flush off,
        # so on_task_end would never write the per-task CSV the G0-E probe
        # counts are read from. First real closed-loop run caught this.
        timer.enable_csv(config.timer.output_csv_dir)
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
    runner = GrootStagedRunner(
        policy.model, timer=timer, compile_vision=args.compile_stage1
    )
    return (
        GrootCacheInterceptor(policy, runner, orchestrator=orchestrator, timer=timer),
        f"cache -> {args.cache_config} ({config.key_builder.type})",
    )


class _InferLockedPolicy:
    """Serialize ``infer`` across connections; everything else passes through.

    v1 concurrency model (plan D4): one shared lock strictly serializes the
    GPU-touching ``infer`` path while sim stepping and socket I/O of the other
    connections overlap. Lifecycle hooks are NOT taken under the lock — they
    run per-connection-serial in the server's handler and their only shared
    touch point (the storage backend) is read-only by construction
    (``write_policy=never`` is enforced at config load).

    ``__getattr__`` delegation keeps the ``hasattr`` surface identical to the
    wrapped stack — the WebSocket server feature-detects every lifecycle hook
    (``on_task_begin`` / ``on_episode_start`` / ``prefill_trajectory`` / ...),
    so a wrapper that faked or hid attributes would silently disable CSV
    flushing and episode resets.
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
    """Fail fast instead of silently serving the CLI yaml under another name.

    In concurrent mode the server exposes ``load_cache_config`` /
    ``select_bundle``; this factory only knows the configuration it was
    started with, so acking any other bundle id would be a silent provenance
    mismatch. The raise surfaces as an error ack on the requesting client.
    Yaml hot-swap on the GR00T server is future work, not this plan.
    """
    if bundle_id != "default":
        raise ValueError(
            f"GR00T concurrent server serves only bundle_id='default' (its CLI "
            f"config); got {bundle_id!r}. Restart the server with the desired "
            f"--cache-config instead of select_bundle."
        )


def _resolve_bundle(
    bundle_id: str,
    *,
    cli_config: Any,
    cli_storage: Any,
    allow_dynamic: bool,
) -> tuple[Any, Any]:
    """Return the ``(config, shared_storage)`` this connection is served under.

    Mirrors ``exp/libero_groot/serve_groot_libero._resolve_bundle``. With
    hot-swap off this is the CLI configuration and any other id is refused.
    With ``--allow-dynamic-bundles`` the driver owns the swap schedule, so the
    GR00T guards re-run on the *loaded* config -- ``load_cache_config`` runs
    only the generic validator, and the recipes the two-stage split cannot
    honour fail silently (an unsatisfiable WARM_START downgrades to MISS, a CP3
    checkpoint is built and never consulted).

    The storage is read off the bundle, never rebuilt: the server's
    ``load_cache_config`` handler already paid for that artifact load, and
    repeating it per connection would repeat it per episode.
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
    validate_groot_cache_config(config)
    validate_artifact_identity(bundle.shared_storage, config)
    return config, bundle.shared_storage


def _build_concurrent_factory(policy: Any, args: Any) -> tuple[Any, str]:
    """Per-connection policy factory for concurrent serving (plan D2/D3).

    Shares exactly two things across connections: the GPU policy (guarded by
    the infer lock) and the storage backend (via per-connection facades).
    Every mutable component — key_builder, gates, judges, strategies, timer,
    orchestrator, staged runner, adapter — is built fresh per connection.
    """
    lock = threading.Lock()

    if not args.cache_config:

        def teacher_factory(shared_base_policy: Any, bundle_id: str = "default") -> Any:
            _require_default_bundle(bundle_id)
            return _InferLockedPolicy(GrootPolicyAdapter(shared_base_policy), lock)

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

    config = load_cache_config(args.cache_config)
    validate_cache_config(config)
    validate_groot_cache_config(config)
    shared_storage = build_shared_storage(config)
    validate_artifact_identity(shared_storage, config)

    allow_dynamic = bool(getattr(args, "allow_dynamic_bundles", False))

    def cache_factory(shared_base_policy: Any, bundle_id: str = "default") -> Any:
        conn_config, conn_storage = _resolve_bundle(
            bundle_id,
            cli_config=config,
            cli_storage=shared_storage,
            allow_dynamic=allow_dynamic,
        )
        components = build_per_connection_components(conn_config, conn_storage, quiet=True)
        timer = components["timer"]
        if conn_config.timer.output_csv_dir:
            # Per-connection subdirectory: the per-task CSV name is only
            # (task ordinal, second) and every connection counts from task 0,
            # so two concurrent connections writing one directory would
            # silently overwrite each other's latency evidence.
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
        runner = GrootStagedRunner(
            shared_base_policy.model, timer=timer, compile_vision=args.compile_stage1
        )
        interceptor = GrootCacheInterceptor(
            shared_base_policy, runner, orchestrator=orchestrator, timer=timer
        )
        return _InferLockedPolicy(GrootPolicyAdapter(interceptor), lock)

    return cache_factory, (
        f"concurrent cache -> {args.cache_config} ({config.key_builder.type})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--diagnostic-seed",
        type=int,
        default=None,
        help="Reset the torch RNG to this seed before every inference. For the "
        "wire-parity check only; must stay unset for real rollouts.",
    )
    parser.add_argument(
        "--cache-config",
        default=None,
        help="YAML cache config. Routes inference through the CP1 cache; "
        "without it the server is the plain teacher.",
    )
    parser.add_argument(
        "--collect-hdf5",
        default=None,
        help="Directory for per-episode HDF5 embeddings, for offline library "
        "building. Mutually exclusive with --cache-config.",
    )
    parser.add_argument(
        "--compile-stage1",
        action="store_true",
        help="torch.compile the vision tower (mode=reduce-overhead / CUDA "
        "graphs). Compilation is persisted via TORCHINDUCTOR_CACHE_DIR so "
        "later server starts reuse it; the first real inference double-runs "
        "eager vs compiled and refuses to serve on divergence. Eval paths "
        "only — collection stays eager (frozen byte-fidelity).",
    )
    parser.add_argument(
        "--concurrent",
        action="store_true",
        help="Serve multiple simultaneous connections via a per-connection "
        "policy factory (plan groot_concurrent_serving). Default OFF: without "
        "this flag every existing invocation keeps the approved "
        "single-connection semantics byte for byte.",
    )
    parser.add_argument(
        "--allow-dynamic-bundles",
        action="store_true",
        help="Accept load_cache_config over the wire, so one process can serve "
        "successive cache configurations addressed by bundle_id. Default OFF: "
        "configuration identity is otherwise carried by the process, which is "
        "what lets a run's results be attributed to one yaml. Turn it on only "
        "for a driver that owns the swap schedule; the GR00T guards then re-run "
        "per bundle instead of once at startup.",
    )
    args = parser.parse_args()

    if args.compile_stage1:
        if args.collect_hdf5:
            parser.error(
                "--compile-stage1 cannot be combined with --collect-hdf5: the "
                "collection path is frozen eager (byte-fidelity)."
            )
        # Persist inductor/triton artifacts so only the first server start on
        # a machine pays the compile; every later start reuses the cache.
        os.environ.setdefault(
            "TORCHINDUCTOR_CACHE_DIR",
            os.path.expanduser("~/.cache/openpi_inductor"),
        )

    if args.concurrent and args.collect_hdf5:
        parser.error(
            "--concurrent cannot be combined with --collect-hdf5: collection "
            "topology is frozen at one server process <-> one connection <-> "
            "one worker (plan D-L); horizontal scale = more server processes."
        )
    if args.cache_config and args.collect_hdf5:
        parser.error(
            "--cache-config and --collect-hdf5 are mutually exclusive. A library "
            "must be collected from the teacher's own actions; with the cache "
            "active some recorded actions would be replayed library entries."
        )
    if args.diagnostic_seed is not None and (args.cache_config or args.collect_hdf5):
        parser.error(
            "--diagnostic-seed pins the flow-matching noise and cannot be "
            "combined with --cache-config / --collect-hdf5: those paths drive "
            "the model through the staged runner, which bypasses the seeding "
            "wrapper, so the seed would appear to be set but would not be."
        )

    if args.allow_dynamic_bundles and not args.cache_config:
        # Unlike the LIBERO entry point, this one has no "start with nothing and
        # receive every configuration over the wire" mode: its teacher-only
        # factory refuses any non-default bundle id, so the server would ack
        # load_cache_config and then fail the first episode's select_bundle --
        # after the whole fleet is already up. Requiring a startup config keeps
        # the flag meaning what it says.
        parser.error(
            "--allow-dynamic-bundles requires --cache-config on this entry point: "
            "without one the factory is teacher-only and cannot serve a loaded bundle."
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
            "collection writes one artifact whose provenance is the process's "
            "single configuration, so swapping the library underneath it would "
            "put entries from two configurations in one file."
        )

    from gr00t.model.policy import Gr00tPolicy
    from openpi.serving import websocket_policy_server

    from exp.robocasa365.groot_data_config import RoboCasa365DataConfig

    checkpoint = pathlib.Path(args.checkpoint)
    data_config = RoboCasa365DataConfig()

    print(f"loading policy from {checkpoint}", flush=True)
    policy: Any = Gr00tPolicy(
        model_path=str(checkpoint),
        embodiment_tag=EMBODIMENT_TAG,
        modality_config=data_config.modality_config(),
        modality_transform=data_config.transform(),
        device="cuda",
    )
    if args.diagnostic_seed is not None:
        print(
            f"DIAGNOSTIC MODE: seeding torch with {args.diagnostic_seed} per call",
            flush=True,
        )
        policy = _SeededPolicy(policy, args.diagnostic_seed)

    # Handshake against the bare policy, before any cache or collector is
    # installed. It runs before a client ever connects, so on_task_begin /
    # on_episode_start have not fired: a probe inference through the cache
    # would advance the step counter, feed the gate and land in the timer,
    # polluting the first real episode's statistics.
    print("running start-up handshake", flush=True)
    _handshake(GrootPolicyAdapter(policy), policy, checkpoint)
    print("POLICY-READY", flush=True)

    # Advertise the diagnostic seed so a client can refuse to collect success
    # rates from a server whose sampling noise is pinned: that bias is invisible
    # in the resulting numbers and unrecoverable after the fact.
    metadata = {
        "checkpoint": str(checkpoint),
        "diagnostic_seed": args.diagnostic_seed,
        "action_horizon": groot_keys.ACTION_HORIZON,
    }
    if args.concurrent:
        factory, stack = _build_concurrent_factory(policy, args)
        print(f"serving stack: {stack}", flush=True)
        # Only the concurrent branch adds the key (pi05 convention): the
        # default-mode metadata bytes stay identical to the approved form so
        # provenance captures diff clean across batches.
        metadata["concurrent"] = True
        server = websocket_policy_server.WebsocketPolicyServer(
            policy,
            host="0.0.0.0",
            port=args.port,
            metadata=metadata,
            concurrent=True,
            connection_policy_factory=factory,
            # Off by default, which closes the whole hot-swap ctrl surface
            # (load_cache_config / select_bundle): the factory then only ever
            # builds the CLI configuration, so any success ack for a loaded
            # bundle would be a silent provenance mismatch (G2 R1 item 1).
            allow_dynamic_bundles=args.allow_dynamic_bundles,
        )
    else:
        served, stack = _build_served_policy(policy, args)
        adapter = GrootPolicyAdapter(served)
        print(f"serving stack: {stack}", flush=True)
        server = websocket_policy_server.WebsocketPolicyServer(
            adapter, host="0.0.0.0", port=args.port, metadata=metadata
        )
    print(f"SERVER-LISTENING on 0.0.0.0:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
