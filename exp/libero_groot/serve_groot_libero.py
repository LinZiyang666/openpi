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
    /home/weiland/projects/openpi:/home/weiland/projects/openpi/src \\
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
#: RIT ladder rungs, ladder order (cheapest first). Written as resume
#: timesteps, not as step counts, because that is what the judge and the
#: payload key on. Under the ascending k=8 schedule ``t`` is elapsed flow, so
#: ``remaining = 8 - 8t``: 0.75 leaves 2 steps, 0.5 leaves 4. Quoting the
#: fractions instead (0.25 / 0.5 of stage 3) is the portable reading -- the
#: RoboCasa ladder uses the same two fractions at k=4, where they are 1 and 2
#: steps. Never reuse Pi0.5's numbers here: that schedule descends, so its
#: ``start_t`` means the opposite fraction.
DEFAULT_RIT_WARM_TS = "0.75,0.5"
#: Matches the LIBERO client's ``--replan-steps 5``. A window wider than the
#: client's would average deviation over actions the robot never executes.
DEFAULT_RIT_H_EXEC = 5


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
    num_inference_timesteps: int | None = None,
) -> tuple[Any, Any]:
    """Return the ``(config, shared_storage)`` this connection is served under.

    With hot-swap disabled this is the CLI configuration and nothing else --
    ``_require_default_bundle`` rejects any other id rather than acking it.

    With ``--allow-dynamic-bundles`` the conductor drives configuration: each
    stage sends ``load_cache_config`` and every episode of that stage then opens
    a connection naming the stage's ``bundle_id``. Two things matter here.

    First, the *guards must re-run on the loaded config*. ``load_cache_config``
    runs only the generic validator, so a hot-swapped yaml would otherwise reach
    serving with an unsatisfiable WARM_START rejected on its first hit, a CP3
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
    validate_groot_cache_config(
        config,
        allow_hysteresis_gate=True,
        num_inference_timesteps=num_inference_timesteps,
    )
    _check_libero_builder(
        config.key_builder.type, lambda m: (_ for _ in ()).throw(ValueError(m))
    )
    validate_artifact_identity(bundle.shared_storage, config)
    return config, bundle.shared_storage


#: The CP2 (ActionCache-style) builder keys on the action head's encoded
#: conditioning, which is camera-count agnostic, so it is serviceable on LIBERO
#: alongside the two-camera CP1 builders.
_CP2_LIBERO_BUILDER = "cp2_groot_ternary"


def _check_libero_builder(builder_type: str, fail) -> None:
    """The three-camera RoboCasa builders reject every LIBERO observation."""
    if not (
        builder_type.startswith("cp1_groot_libero") or builder_type == _CP2_LIBERO_BUILDER
    ):
        fail(
            f"key_builder.type {builder_type!r} is not a LIBERO builder; the "
            "three-camera RoboCasa builders assert three image-token runs and "
            "reject every LIBERO observation."
        )


def _build_shadow_factory(
    args: Any, config: Any, shared_storage: Any, lock: Any
) -> tuple[Any, str]:
    """Per-connection factory for the RIT calibration pass.

    The shadow is the same object the RoboCasa line calibrates with
    (``exp.robocasa365.rit_shadow.GrootRitShadow``) -- one implementation, so
    the two teachers' ladders are fitted on columns produced by the same code.
    It satisfies ``get_action`` plus the episode hooks, which is exactly the
    surface ``GrootLiberoPolicyAdapter`` forwards, so it drops into the place
    the interceptor would otherwise take.

    Two things are per-connection rather than shared:

    *   **the output file.** ``on_episode_end`` appends a whole episode's rows
        under one ``open(..., "a")``; two connections flushing at once would
        interleave at line granularity and no column would reveal it. Each
        connection therefore writes ``<stem>.conn_<id>.jsonl`` and the fit
        reads the glob.
    *   **the orchestrator.** Retrieval state is per-episode and the search
        session identity is bound in ``on_episode_start``.

    The action weights are read once from the library the recipe names, not
    from a separate npz: a weights file that drifted from the deployed library
    would rescale every deviation without failing anything.
    """
    import numpy as _np
    import torch as _torch

    from openpi.cache.config import build_per_connection_components
    from openpi.cache.groot.staged import GrootStagedRunner
    from openpi.cache.orchestrator import CacheOrchestrator

    from exp.libero_groot.policy_adapter import GrootLiberoPolicyAdapter
    from exp.robocasa365.rit_shadow import GrootRitShadow, library_action_weights

    warm_ts = [float(x) for x in str(args.rit_warm_ts).split(",") if x.strip()]
    if not warm_ts:
        raise SystemExit("--rit-warm-ts is empty")
    preload = config.backend.in_memory.preload_path
    w, active_mask = library_action_weights(preload)
    w = _torch.as_tensor(_np.asarray(w), dtype=_torch.float32)
    active_mask = _torch.as_tensor(_np.asarray(active_mask), dtype=_torch.bool)
    out_stem = pathlib.Path(args.rit_shadow_out)
    out_stem.parent.mkdir(parents=True, exist_ok=True)

    def shadow_factory(shared_base_policy: Any, bundle_id: str = "default") -> Any:
        _require_default_bundle(bundle_id)
        components = build_per_connection_components(config, shared_storage, quiet=True)
        orchestrator = CacheOrchestrator(
            storage=components["storage"],
            key_builder=components["key_builder"],
            gates=components["gates"],
            judges=components["judges"],
            search_strategies=components["search_strategies"],
            timer=components["timer"],
            write_policy=components["write_policy"],
            offline_writers=components["offline_writers"],
            library_stats=components["library_stats"],
        )
        runner = GrootStagedRunner(
            shared_base_policy.model,
            timer=components["timer"],
            compile_vision=getattr(args, "compile_stage1", False),
        )
        conn_out = out_stem.with_name(
            f"{out_stem.stem}.conn_{uuid.uuid4().hex[:8]}{out_stem.suffix or '.jsonl'}"
        )
        shadow = GrootRitShadow(
            shared_base_policy,
            runner,
            orchestrator=orchestrator,
            out_path=str(conn_out),
            warm_ts=warm_ts,
            w=w,
            active_mask=active_mask,
            h_exec=int(args.rit_h_exec),
            experiment=args.experiment,
        )
        return _InferLockedPolicy(GrootLiberoPolicyAdapter(shadow), lock)

    return shadow_factory, (
        f"rit-shadow -> {out_stem}.conn_*{out_stem.suffix or '.jsonl'} "
        f"(ts={warm_ts}, h_exec={args.rit_h_exec}, library={pathlib.Path(preload).name})"
    )


def _build_loto_factory(
    args: Any, config: Any, shared_storage: Any, lock: Any
) -> tuple[Any, str]:
    """Per-connection factory for the LOTO closed-loop verification log.

    The served stack is the production one -- ``GrootCacheInterceptor`` over the
    CLI recipe -- wrapped by ``GrootLotoLogger`` (``exp.rit_loto.loto_logger``),
    which records every decision's stage-1 slices, wire hit meta and executed
    chunk without recomputing anything. Per-connection output directories keep
    simultaneous connections apart exactly as the shadow factory does, and the
    frozen identity (arm / library / checkpoint / pool manifest digests) is
    stamped on every episode so the merge step can refuse a mismatched run.
    """
    from openpi.cache.config import build_per_connection_components
    from openpi.cache.groot.load_guard import live_num_inference_timesteps
    from openpi.cache.groot.staged import GrootStagedRunner
    from openpi.cache.orchestrator import CacheOrchestrator
    from openpi.cache.types import groot_n15_schedule

    from exp.libero_groot.policy_adapter import GrootLiberoPolicyAdapter
    from exp.rit_loto.build_loto_table import checkpoint_identity, sha256_file
    from exp.rit_loto.emit_verify_arm import load_frozen_record
    from exp.rit_loto.loto_logger import GrootLotoLogger

    cp2 = config.checkpoints.get("cp2")
    if cp2 is not None and getattr(cp2, "enabled", False):
        raise SystemExit("--loto-log-out supports the CP1 recipe of this line only (cp2 is enabled in the yaml)")
    record, record_sha = load_frozen_record(args.loto_frozen_record)
    if bool(getattr(args, "compile_stage1", False)) != record["compile_stage1"]:
        raise SystemExit("LOTO run must use the frozen eager stage-1 path")
    if args.loto_run_tag not in set(record["run_tags"].values()):
        raise SystemExit(f"run tag {args.loto_run_tag!r} is not one of the frozen record's {record['run_tags']}")
    preload = config.backend.in_memory.preload_path
    ident = record["identity"]
    arm_sha = sha256_file(args.cache_config)
    library_sha = sha256_file(preload)
    ckpt = checkpoint_identity(args.checkpoint)
    problems = []
    if arm_sha != record["arm"]["yaml_sha256"]:
        problems.append(f"arm yaml sha {arm_sha[:12]} != frozen {record['arm']['yaml_sha256'][:12]}")
    if library_sha != ident["library_sha256"]:
        problems.append(f"library sha {library_sha[:12]} != frozen {ident['library_sha256'][:12]}")
    if ckpt["sha256"] != ident["checkpoint_identity_sha256"]:
        problems.append(f"checkpoint identity {ckpt['sha256'][:12]} != frozen {ident['checkpoint_identity_sha256'][:12]}")
    if int(args.rit_h_exec) != int(ident["h_exec"]):
        problems.append(f"--rit-h-exec {args.rit_h_exec} != frozen h_exec {ident['h_exec']}")
    if problems:
        raise SystemExit("frozen run record does not describe this server: " + "; ".join(problems))
    identity_attrs = {
        "loto_frozen_record_sha256": record_sha,
        "loto_arm_yaml_sha256": arm_sha,
        "loto_library_sha256": library_sha,
        "loto_checkpoint_identity_sha256": ckpt["sha256"],
        "loto_pool_manifest_sha256": record["pool_manifest_sha256"],
        "loto_fits_sha256": record["fits_sha256"],
    }
    frozen_schedule_id = ident["schedule_id"]
    out_dir = pathlib.Path(args.loto_log_out)
    out_dir.mkdir(parents=True, exist_ok=True)

    def loto_factory(shared_base_policy: Any, bundle_id: str = "default") -> Any:
        _require_default_bundle(bundle_id)
        live = groot_n15_schedule(live_num_inference_timesteps(shared_base_policy)).schedule_id
        if live != frozen_schedule_id:
            raise SystemExit(f"live schedule {live} != frozen {frozen_schedule_id}")
        components = build_per_connection_components(config, shared_storage, quiet=True)
        orchestrator = CacheOrchestrator(
            storage=components["storage"],
            key_builder=components["key_builder"],
            gates=components["gates"],
            judges=components["judges"],
            search_strategies=components["search_strategies"],
            timer=components["timer"],
            write_policy=components["write_policy"],
            offline_writers=components["offline_writers"],
            library_stats=components["library_stats"],
        )
        runner = GrootStagedRunner(
            shared_base_policy.model,
            timer=components["timer"],
            compile_vision=getattr(args, "compile_stage1", False),
        )
        logger_policy = GrootLotoLogger(
            shared_base_policy,
            runner,
            orchestrator=orchestrator,
            timer=components["timer"],
            out_dir=str(out_dir),
            experiment=args.experiment,
            run_tag=args.loto_run_tag,
            identity_attrs=identity_attrs,
            h_exec=int(args.rit_h_exec),
        )
        return _InferLockedPolicy(GrootLiberoPolicyAdapter(logger_policy), lock)

    return loto_factory, (
        f"loto-log -> {out_dir}/{args.loto_run_tag}/conn_*/ (arm={pathlib.Path(args.cache_config).name}, "
        f"library={pathlib.Path(preload).name}, h_exec={args.rit_h_exec})"
    )


def yaml_identity(bundle_id: str, cache_config: str | None) -> str:
    """The bundle identity an online judge learns under.

    A dynamic bundle is addressed by its yaml stem already; the startup
    ``--cache-config`` arm answers to the placeholder ``"default"``, so its
    identity is the startup yaml's stem instead. Two runs of one arm that must
    not pool their learning therefore need distinct yaml file names.
    """
    if bundle_id and bundle_id != "default":
        return str(bundle_id)
    if cache_config:
        return pathlib.Path(cache_config).stem
    return str(bundle_id or "default")


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
    # One registry per process: every connection serving the same bundle
    # attaches to the same online RIT curves (plan online_rit_groot §3.7).
    # Built here, not at import, and only read by an ``online_rit`` judge.
    from openpi.cache.online_state import CurveRegistry

    online_registry = CurveRegistry(state_log_root=getattr(args, "online_state_dir", None), require_persistence=True)

    if not args.cache_config and not allow_dynamic:

        def teacher_factory(shared_base_policy: Any, bundle_id: str = "default") -> Any:
            _require_default_bundle(bundle_id)
            return _InferLockedPolicy(
                GrootLiberoPolicyAdapter(shared_base_policy), lock
            )

        return teacher_factory, "concurrent teacher-only (no cache)"

    from openpi.cache.config import (
        build_per_connection_components,
        build_shared_storage,
        load_cache_config,
        validate_cache_config,
    )
    from openpi.cache.groot.interceptor import GrootCacheInterceptor
    from openpi.cache.groot.load_guard import (
        live_num_inference_timesteps,
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
    # The generic validator permits recipes this GR00T integration cannot honour:
    # a wrong-schedule WARM_START is rejected at runtime, a CP3 checkpoint is
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
        validate_groot_cache_config(
            config,
            allow_hysteresis_gate=True,
            num_inference_timesteps=live_num_inference_timesteps(policy),
        )
        _check_libero_builder(
            config.key_builder.type, lambda m: (_ for _ in ()).throw(ValueError(m))
        )
        shared_storage = build_shared_storage(config)
        # ``load_artifact`` only compares ``vector_dims``, and mean-pool and
        # max-pool libraries are dimensionally identical -- nothing else would ever
        # notice a swapped artifact.
        validate_artifact_identity(shared_storage, config)

    if getattr(args, "rit_shadow_out", None):
        return _build_shadow_factory(args, config, shared_storage, lock)
    if getattr(args, "loto_log_out", None):
        return _build_loto_factory(args, config, shared_storage, lock)

    def cache_factory(shared_base_policy: Any, bundle_id: str = "default") -> Any:
        conn_config, conn_storage = _resolve_bundle(
            bundle_id,
            cli_config=config,
            cli_storage=shared_storage,
            num_inference_timesteps=live_num_inference_timesteps(shared_base_policy),
            allow_dynamic=allow_dynamic,
        )
        if conn_config is None:
            # Dynamic bundles enabled but nothing loaded yet: serve the teacher.
            # Refusing instead would break the runner's opening handshake, which
            # selects "default" before the first stage has been sent.
            return _InferLockedPolicy(
                GrootLiberoPolicyAdapter(shared_base_policy), lock
            )
        components = build_per_connection_components(
            conn_config,
            conn_storage,
            quiet=True,
            yaml_id=yaml_identity(bundle_id, args.cache_config),
            online_registry=online_registry,
        )
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
        runner = GrootStagedRunner(
            shared_base_policy.model, timer=timer, compile_vision=getattr(args, "compile_stage1", False)
        )
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


def _unload_stages_2_and_3(model: Any) -> None:
    """Move every stage-2/3 module to the meta device; stage 1 keeps the GPU.

    Stage 1 (``GrootStagedRunner.run_stage1``) touches the vision tower, the
    ``mlp1`` projector and ``language_model.model.embed_tokens``; everything
    else (the LLM's transformer layers, its norm and lm_head, the backbone's
    post-LLM ``eagle_linear`` and the whole action head) only runs on MISS /
    WARM_START. Meta parameters keep their shapes and attributes (the schedule
    guards still read ``action_head.num_inference_timesteps``) but any forward
    through them raises, which is the intended fail-loud behaviour for a
    pure-cache replica.
    """
    import torch  # noqa: PLC0415

    eagle = model.backbone.eagle_model
    lm = eagle.language_model
    before = torch.cuda.memory_allocated() / 2**30
    for name, module in (
        ("language_model.model.layers", lm.model.layers),
        ("language_model.model.norm", lm.model.norm),
        ("language_model.lm_head", lm.lm_head),
        ("backbone.eagle_linear", model.backbone.eagle_linear),
        ("action_head", model.action_head),
    ):
        module.to(device="meta")
        print(f"stage1-only: {name} -> meta", flush=True)
    torch.cuda.empty_cache()
    after = torch.cuda.memory_allocated() / 2**30
    print(f"stage1-only: GPU allocated {before:.2f} GB -> {after:.2f} GB", flush=True)


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
        "--stage1-only",
        action="store_true",
        help="Pure-cache serving: after loading, move the language model's "
        "transformer layers, its lm_head, the backbone's post-LLM projection and "
        "the action head to the meta device, keeping only what stage 1 needs "
        "(vision tower, projector, token embeddings). Frees ~4 GB per replica. "
        "Any MISS or WARM_START then fails loudly, so use it only with "
        "always_hit recipes on libraries that cover every task.",
    )
    parser.add_argument(
        "--compile-stage1",
        action="store_true",
        help="torch.compile the vision tower (mode=reduce-overhead / CUDA graphs), "
        "as serve_groot_n15 --compile-stage1 does. Compilation is persisted via "
        "TORCHINDUCTOR_CACHE_DIR so later server starts reuse it; the first real "
        "inference double-runs eager vs compiled and refuses to serve on "
        "divergence. Eval paths only -- collection stays eager (byte-fidelity).",
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
    parser.add_argument(
        "--rit-shadow-out",
        default=None,
        help="JSONL sink for the RIT calibration pass. The served action stays "
        "the teacher's; the cache is queried in the shadow and each rung's "
        "deviation is labelled. One file per connection is written beside this "
        "path so simultaneous connections cannot interleave rows.",
    )
    parser.add_argument(
        "--rit-warm-ts",
        default=DEFAULT_RIT_WARM_TS,
        help="Resume timesteps to label, ladder order, comma separated. The "
        "default is the k=8 schedule's 2-steps-remaining and 4-steps-remaining "
        "rungs (t=0.75 / t=0.5) -- the same 25%% / 50%% fractions of stage 3 "
        "the RoboCasa ladder uses at k=4.",
    )
    parser.add_argument(
        "--rit-h-exec",
        type=int,
        default=DEFAULT_RIT_H_EXEC,
        help="Executed window the deviation is averaged over; must equal the "
        "client's --replan-steps, because steps past it are never executed.",
    )
    parser.add_argument(
        "--online-state-dir",
        default=None,
        help="Root directory for online RIT state snapshots and feedback logs "
        "(one subdirectory per bundle + library). Requires --concurrent; only an "
        "online_rit judge writes there.",
    )
    parser.add_argument(
        "--loto-log-out",
        default=None,
        help="Root directory for the LOTO closed-loop verification log: one HDF5 "
        "episode + one sidecar JSONL row per decision, written per connection "
        "under <root>/<run_tag>/conn_<id>/. Requires --cache-config and "
        "--concurrent; the served verdicts are the production ones.",
    )
    parser.add_argument(
        "--loto-run-tag",
        default=None,
        help="Run tag of the verification log (e.g. smoke / verify); smoke and "
        "formal runs must never share a tag or a directory.",
    )
    parser.add_argument(
        "--loto-frozen-record",
        default=None,
        help="frozen_run.json written by exp.rit_loto.emit_verify_arm: the server refuses "
        "to start unless the arm yaml, library, checkpoint content, H_exec and live "
        "schedule match it, and stamps its digest on every logged episode.",
    )
    args = parser.parse_args()

    if args.stage1_only and (
        args.collect_hdf5 or args.rit_shadow_out or getattr(args, "loto_log_out", None)
    ):
        parser.error(
            "--stage1-only serves the cache path alone; collection, the RIT shadow "
            "and the LOTO logger all run the teacher and cannot use it."
        )

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

    if args.loto_log_out or args.loto_run_tag or args.loto_frozen_record:
        if not (args.loto_log_out and args.loto_run_tag and args.loto_frozen_record):
            parser.error("--loto-log-out, --loto-run-tag and --loto-frozen-record must be given together")
        if not args.cache_config:
            parser.error("--loto-log-out requires --cache-config: it logs cache decisions")
        if not args.concurrent:
            parser.error("--loto-log-out requires --concurrent (per-connection log directories)")
        if args.collect_hdf5:
            parser.error("--loto-log-out and --collect-hdf5 are mutually exclusive")
        if args.rit_shadow_out:
            parser.error("--loto-log-out and --rit-shadow-out are mutually exclusive")
        if args.allow_dynamic_bundles:
            parser.error(
                "--loto-log-out cannot be combined with --allow-dynamic-bundles: "
                "every logged episode must describe one frozen arm and library"
            )

    if args.online_state_dir and not args.concurrent:
        parser.error("--online-state-dir requires --concurrent")
    if args.cache_config and args.collect_hdf5:
        parser.error("--cache-config and --collect-hdf5 are mutually exclusive")
    if args.rit_shadow_out:
        # The shadow labels rungs against one library with one retrieval stack.
        # Dynamic bundles would let the library change under a single output
        # file, and the fit has no column that would reveal the swap.
        if not args.cache_config:
            parser.error("--rit-shadow-out requires --cache-config")
        if not args.concurrent:
            # Only the concurrent factory is wired for the shadow. The pass is
            # 150 episodes per suite and labels every rung on every step, so a
            # one-connection server would spend hours where the fleet spends
            # minutes; there is no reason to carry a second code path for it.
            parser.error("--rit-shadow-out requires --concurrent")
        if args.allow_dynamic_bundles:
            parser.error(
                "--rit-shadow-out cannot be combined with --allow-dynamic-bundles: "
                "the calibration rows must all describe one library"
            )
        if args.collect_hdf5:
            parser.error("--rit-shadow-out and --collect-hdf5 are mutually exclusive")
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
    if args.stage1_only:
        _unload_stages_2_and_3(policy.model)

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
        from openpi.cache.groot.load_guard import (
            live_num_inference_timesteps,
            validate_groot_cache_config,
        )
        from openpi.cache.groot.staged import GrootStagedRunner

        config = load_cache_config(args.cache_config)
        validate_cache_config(config)
        # Reject incompatible recipes at startup: an unsatisfiable WARM_START
        # would fail on its first hit, a CP3 checkpoint is
        # built and never consulted, a non-always_search gate changes what
        # ``searched`` means downstream. ``allow_hysteresis_gate`` is this entry
        # point's explicit opt-in (see the concurrent path above for why it is
        # claimed per entry point rather than relaxed in the shared guard).
        validate_groot_cache_config(
            config,
            allow_hysteresis_gate=True,
            num_inference_timesteps=live_num_inference_timesteps(policy),
        )
        _check_libero_builder(config.key_builder.type, parser.error)
        components = build_cache_components(config)
        runner = GrootStagedRunner(policy.model, compile_vision=getattr(args, "compile_stage1", False))
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
