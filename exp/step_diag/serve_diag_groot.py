"""GR00T N1.5 server of the step-vs-warm-start line (RoboCasa365 via ``serve_groot_n15``, LIBERO via
``serve_groot_libero``): one arm per process, evidence for every decision.

The production servers are imported unchanged; only their served-object builder is rebound for
this process (the same technique as ``serve_groot_n15_ksweep``, which rebinds the policy class):

* ``--benchmark rc``    ``serve_groot_n15._build_served_policy`` -> ``GrootDiagPolicy`` (shadow /
                        plain / full; plain pins the head with the k-sweep constructor patch) or
                        ``GrootEvidencePolicy(GrootCacheInterceptor(...))`` (warm);
* ``--benchmark libero`` ``serve_groot_libero._build_shadow_factory`` -> a per-connection
                        ``GrootDiagPolicy`` behind the LIBERO adapter and infer lock (shadow only;
                        the LIBERO ladders keep the production ``--denoising-steps`` server).

The runner, orchestrator and cache-config validation are built exactly as the production
builder builds them (config loaded, validated against the live head, artifact identity checked).

usage::

    python -m exp.step_diag.serve_diag_groot --benchmark rc --mode shadow --env-id groot_rc \\
        --arm-id shadow --experiment-id sdiag_rc_v1 --diag-out /data/step_diag/groot_rc \\
        --cache-config <always_hit calibration yaml> -- --checkpoint <ckpt> --port 23230
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import uuid

from exp.step_diag import envs as _envs
from exp.step_diag.recorder import DiagRecorder, DiagSpec

MODES = ("shadow", "plain", "full", "warm")


def parse(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", required=True, choices=("rc", "libero"))
    ap.add_argument("--mode", required=True, choices=MODES)
    ap.add_argument("--env-id", required=True, choices=sorted(k for k in _envs.ENVS if k.startswith("groot")))
    ap.add_argument("--arm-id", required=True)
    ap.add_argument("--experiment-id", required=True)
    ap.add_argument("--diag-out", required=True)
    ap.add_argument("--exec-steps", type=int, default=None)
    ap.add_argument("--cache-config", default=None)
    ap.add_argument("--launch-id", default="")
    ap.add_argument("--n-primary", type=int, default=4)
    ap.add_argument("--n-dense-extra", type=int, default=28)
    if "--" in argv:
        i = argv.index("--")
        own, rest = argv[:i], argv[i + 1:]
    else:
        own, rest = argv, []
    args = ap.parse_args(own)
    env = _envs.resolve_env(args.env_id)
    if (args.benchmark == "rc") != (env.benchmark == "robocasa365"):
        ap.error("--benchmark and --env-id disagree")
    if args.n_primary != 4 or args.n_dense_extra != 28:
        ap.error("the frozen sampling contract is 4 primary + 28 extra dense samples")
    if args.benchmark == "libero" and args.mode != "shadow":
        ap.error("LIBERO GR00T serves only the shadow mode here (ladders use the production server)")
    if args.mode in ("shadow", "warm") and not args.cache_config:
        ap.error(f"--mode {args.mode} requires --cache-config")
    if args.mode == "full":
        args.exec_steps = env.k_full
    if args.mode == "plain" and args.exec_steps not in (*env.k_set, env.k_full):
        ap.error("--mode plain requires --exec-steps")
    if args.mode in ("plain", "full") and args.cache_config:
        ap.error("--cache-config is refused for plain / full arms")
    if args.cache_config:
        rest = [*rest, "--cache-config", args.cache_config]
    return args, rest


def build_recorder(args: argparse.Namespace, checkpoint: str = "") -> DiagRecorder:
    env = _envs.resolve_env(args.env_id)
    if args.benchmark == "rc":
        from exp.robocasa365.groot_keys import ACTION_KEYS, ACTION_DIMS
        if [ACTION_DIMS[k] for k in ACTION_KEYS] != [3, 3, 1, 4, 1]:
            raise ValueError("GR00T RoboCasa executed action mapping drifted")
    _envs.validate_arm(args.env_id, args.mode, args.arm_id, args.exec_steps, args.cache_config)
    out = pathlib.Path(args.diag_out)
    out.mkdir(parents=True, exist_ok=True)
    cfg_sha = _envs.sha256_file(args.cache_config) if args.cache_config else None
    identity = _envs.resource_identity(args.env_id, checkpoint, args.cache_config)
    manifest = _envs.RunManifest(
        experiment_id=args.experiment_id, env=env.to_json(), arm_id=args.arm_id, mode=args.mode,
        exec_steps=args.exec_steps if args.mode in ("plain", "full") else None, checkpoint=checkpoint,
        checkpoint_sha256=identity["checkpoint_sha256"], cache_config=args.cache_config, cache_config_sha256=cfg_sha,
        library=identity["library"], library_sha256=identity["library_sha256"], code_commit=_envs.git_commit(),
        extras={"n_primary": args.n_primary, "n_dense_extra": args.n_dense_extra, "h_exec": _envs.H_EXEC,
                "benchmark": args.benchmark, "runtime": identity["runtime"]},
    )
    config_sha = manifest.write(out / f"manifest_{args.arm_id}.json")
    spec = DiagSpec(
        experiment_id=args.experiment_id, env_id=env.env_id, arm_id=args.arm_id, mode=args.mode,
        k_full=env.k_full, k_set=tuple(env.k_set) if args.mode == "shadow" else (),
        warm_ts=tuple(env.warm_ts) if args.mode == "shadow" else (), n_primary=args.n_primary,
        n_dense_extra=args.n_dense_extra, action_shape=(env.action_horizon, env.action_dim),
        exec_steps=args.exec_steps, config_sha=config_sha,
    )
    return DiagRecorder(spec, out, launch_id=args.launch_id)


def _orchestrator_and_runner(policy, args_srv, cache_config_path: str | None):
    """Mirror of ``serve_groot_n15._build_served_policy``'s construction (config, guards, runner)."""
    from openpi.cache.groot.staged import GrootStagedRunner

    if not cache_config_path:
        return None, GrootStagedRunner(policy.model)
    from openpi.cache.config import build_cache_components, load_cache_config, validate_cache_config
    from openpi.cache.groot.load_guard import (
        live_num_inference_timesteps,
        validate_artifact_identity,
        validate_groot_cache_config,
    )
    from openpi.cache.orchestrator import CacheOrchestrator

    config = load_cache_config(cache_config_path)
    validate_cache_config(config)
    validate_groot_cache_config(config, num_inference_timesteps=live_num_inference_timesteps(policy),
                                allow_hysteresis_gate=True)
    components = build_cache_components(config)
    validate_artifact_identity(components["storage"], config)
    timer = components["timer"]
    orchestrator = CacheOrchestrator(
        storage=components["storage"], key_builder=components["key_builder"], gates=components["gates"],
        judges=components["judges"], search_strategies=components["search_strategies"], timer=timer,
        write_policy=components["write_policy"], offline_writers=components["offline_writers"],
        library_stats=components["library_stats"],
    )
    runner = GrootStagedRunner(policy.model, timer=timer, compile_vision=getattr(args_srv, "compile_stage1", False))
    return orchestrator, runner


def install_rc(args: argparse.Namespace, recorder: DiagRecorder) -> None:
    from openpi.cache.types import groot_n15_schedule

    from exp.robocasa365 import serve_groot_n15 as _srv
    from exp.step_diag.groot import GrootDiagPolicy, GrootEvidencePolicy

    env = _envs.resolve_env(args.env_id)
    if args.mode in ("plain", "full"):
        import gr00t.model.policy as policy_module

        k = int(args.exec_steps)
        base_cls = policy_module.Gr00tPolicy

        class _Pinned(base_cls):  # type: ignore[misc,valid-type]
            def __init__(self, *a, **kw):
                kw["denoising_steps"] = k
                super().__init__(*a, **kw)
                live = int(self.model.action_head.num_inference_timesteps)
                if live != k:
                    raise RuntimeError(f"requested {k} steps but the head runs {live}")
                print(f"STEP_DIAG num_inference_timesteps={live}", flush=True)

        policy_module.Gr00tPolicy = _Pinned

    def _build(policy, srv_args):
        shape = (int(policy.model.action_head.config.action_horizon), int(policy.model.action_head.config.action_dim))
        if shape != (env.action_horizon, env.action_dim):
            raise RuntimeError(f"live GR00T action shape {shape} differs from {env.env_id}")
        orchestrator, runner = _orchestrator_and_runner(policy, srv_args, args.cache_config)
        live = int(policy.model.action_head.num_inference_timesteps)
        # A one-step plain loop has no intermediate snapshot schedule. Do not construct
        # DenoiseSchedule here: its >=2 guard is deliberate for cache resume schedules.
        if args.mode in ("plain", "full"):
            served = GrootDiagPolicy(policy, runner, orchestrator=None, diag=recorder,
                                     schedule=None, shadow=False)
            return served, f"step_diag {args.mode} k={live}"
        if live != env.k_full:
            raise RuntimeError(f"live GR00T K={live}, expected {env.k_full}")
        schedule = groot_n15_schedule(live)
        if args.mode == "warm":
            from openpi.cache.groot.interceptor import GrootCacheInterceptor

            inner = GrootCacheInterceptor(policy, runner, orchestrator=orchestrator, timer=runner._timer)  # noqa: SLF001
            return GrootEvidencePolicy(inner, runner, recorder, schedule_id=schedule.schedule_id), f"step_diag warm -> {args.cache_config}"
        if args.mode == "shadow":
            served = GrootDiagPolicy(policy, runner, orchestrator=orchestrator, diag=recorder, schedule=schedule)
            return served, f"step_diag shadow (k_set={env.k_set}, warm_ts={env.warm_ts})"

    _srv._build_served_policy = _build  # noqa: SLF001 - process-local rebinding of the builder


def install_libero(args: argparse.Namespace, recorder: DiagRecorder) -> None:
    from openpi.cache.types import groot_n15_schedule

    from exp.libero_groot import serve_groot_libero as _srv
    from exp.step_diag.groot import GrootDiagPolicy

    env = _envs.resolve_env(args.env_id)

    def _factory(srv_args, config, shared_storage, lock):
        from openpi.cache.config import build_per_connection_components
        from openpi.cache.groot.staged import GrootStagedRunner
        from openpi.cache.orchestrator import CacheOrchestrator

        from exp.libero_groot.policy_adapter import GrootLiberoPolicyAdapter

        def shadow_factory(shared_base_policy, bundle_id: str = "default"):
            _srv._require_default_bundle(bundle_id)  # noqa: SLF001
            head_config = shared_base_policy.model.action_head.config
            if (int(head_config.action_horizon), int(head_config.action_dim)) != (env.action_horizon, env.action_dim):
                raise RuntimeError("LIBERO GR00T action shape differs from the frozen environment")
            components = build_per_connection_components(config, shared_storage, quiet=True)
            orchestrator = CacheOrchestrator(
                storage=components["storage"], key_builder=components["key_builder"], gates=components["gates"],
                judges=components["judges"], search_strategies=components["search_strategies"],
                timer=components["timer"], write_policy=components["write_policy"],
                offline_writers=components["offline_writers"], library_stats=components["library_stats"],
            )
            runner = GrootStagedRunner(shared_base_policy.model, timer=components["timer"],
                                       compile_vision=getattr(srv_args, "compile_stage1", False))
            if int(shared_base_policy.model.action_head.num_inference_timesteps) != env.k_full:
                raise RuntimeError("LIBERO live step count differs from the frozen environment")
            schedule = groot_n15_schedule(int(runner.live_schedule().num_steps))
            served = GrootDiagPolicy(shared_base_policy, runner, orchestrator=orchestrator, diag=recorder,
                                     schedule=schedule)
            return _srv._InferLockedPolicy(GrootLiberoPolicyAdapter(served), lock)  # noqa: SLF001

        return shadow_factory, f"step_diag shadow (k_set={env.k_set}, warm_ts={env.warm_ts}) conn={uuid.uuid4().hex[:6]}"

    _srv._build_shadow_factory = _factory  # noqa: SLF001


def main(argv: list[str] | None = None) -> None:
    args, rest = parse(sys.argv[1:] if argv is None else argv)
    ckpt = ""
    if "--checkpoint" in rest:
        ckpt = rest[rest.index("--checkpoint") + 1]
    for arg in rest:
        if arg.startswith("--checkpoint="):
            ckpt = arg.split("=", 1)[1]
    recorder = build_recorder(args, checkpoint=ckpt)
    print(f"STEP_DIAG arm={args.arm_id} mode={args.mode} env={args.env_id} exec_steps={args.exec_steps} "
          f"rows={recorder.rows_path}", flush=True)
    if args.benchmark == "rc":
        install_rc(args, recorder)
        from exp.robocasa365 import serve_groot_n15 as srv
    else:
        install_libero(args, recorder)
        from exp.libero_groot import serve_groot_libero as srv
        # the LIBERO shadow rides the production ``--rit-shadow-out`` switch to reach the factory
        if "--rit-shadow-out" not in rest:
            rest = [*rest, "--rit-shadow-out", str(pathlib.Path(args.diag_out) / "unused_rit_shadow.jsonl"),
                    "--rit-warm-ts", ",".join(str(t) for t in _envs.resolve_env(args.env_id).warm_ts)]
    sys.argv = [sys.argv[0], *rest]
    try:
        srv.main()
    finally:
        recorder.close()


if __name__ == "__main__":
    main()
