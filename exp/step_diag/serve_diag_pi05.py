"""Pi0.5 server of the step-vs-warm-start line: one arm per process, evidence for every decision.

Composes ``scripts.serve_policy`` (unchanged) with ``Pi05DiagInterceptor`` by rebinding
``openpi.cache.interceptor.InferenceInterceptor`` before the production server builds its policy
(the same class-level technique as ``exp.nfe_baseline.serve_pi05_ksweep``). Modes:

* ``shadow``  teacher executes the full loop; ``--cache-config`` names the read-only retrieval yaml
              (``judge.type: threshold`` with an unreachable threshold, ``write_policy: never``);
              the recorder samples the reduced / warm matrix on every decision;
* ``plain``   the executed loop runs ``--exec-steps`` Euler steps (``interceptor._NUM_STEPS``
              is pinned for the process; the cache path is not loaded);
* ``full``    ``plain`` with ``--exec-steps`` equal to the environment's full count;
* ``warm``    ``--cache-config`` names the forced warm-start yaml (``always_warm_start``).

Every mode serialises ``infer`` behind one process lock, stamps the arm into the handshake
metadata and writes ``manifest_<arm>.json`` (config sha referenced by every row).

usage::

    python -m exp.step_diag.serve_diag_pi05 --mode plain --exec-steps 2 --env-id pi05_rc \\
        --arm-id plain_k2 --experiment-id sdiag_rc_v1 --diag-out /data/step_diag/pi05_rc \\
        -- --cache --non-concurrent --port 23150 policy:checkpoint --policy.config <cfg> --policy.dir <dir>
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import threading

from exp.step_diag import envs as _envs
from exp.step_diag.recorder import DiagRecorder, DiagSpec

MODES = ("shadow", "plain", "full", "warm")


def parse(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", required=True, choices=MODES)
    ap.add_argument("--env-id", required=True, choices=sorted(k for k in _envs.ENVS if k.startswith("pi05")))
    ap.add_argument("--arm-id", required=True)
    ap.add_argument("--experiment-id", required=True)
    ap.add_argument("--diag-out", required=True)
    ap.add_argument("--exec-steps", type=int, default=None, help="plain: executed Euler steps")
    ap.add_argument("--cache-config", default=None, help="shadow / warm: cache yaml handed to serve_policy")
    ap.add_argument("--launch-id", default="")
    ap.add_argument("--n-primary", type=int, default=4)
    ap.add_argument("--n-dense-extra", type=int, default=28)
    ap.add_argument("--policy-dir", default="", help="checkpoint dir (for the manifest sha only)")
    if "--" in argv:
        i = argv.index("--")
        own, rest = argv[:i], argv[i + 1:]
    else:
        own, rest = argv, []
    args = ap.parse_args(own)
    if args.n_primary != 4 or args.n_dense_extra != 28:
        ap.error("the frozen sampling contract is 4 primary + 28 extra dense samples")
    if args.mode in ("shadow", "warm") and not args.cache_config:
        ap.error(f"--mode {args.mode} requires --cache-config")
    if args.mode in ("plain", "full") and args.cache_config:
        ap.error("--cache-config is refused for plain / full arms")
    env = _envs.resolve_env(args.env_id)
    if args.mode == "full":
        args.exec_steps = env.k_full
    if args.mode == "plain" and args.exec_steps not in (*env.k_set, env.k_full):
        ap.error("--mode plain requires --exec-steps")
    if args.mode in ("plain", "full") and "--cache" not in rest:
        rest = ["--cache", *rest]
    if args.mode in ("shadow", "warm"):
        rest = [f"--cache_config={args.cache_config}", *rest]
    if "--non-concurrent" not in rest:
        # Single-connection serving: the per-request thread runs every stage inline (exact Euler
        # step counts, shadow sampling under the production lock, no BatchingCoordinator).
        rest = ["--non-concurrent", *rest]
    # tyro: the top-level flags must precede the ``policy:checkpoint`` subcommand
    return args, rest


def build_recorder(args: argparse.Namespace) -> tuple[DiagRecorder, str]:
    """Recorder + manifest for this process; returns ``(recorder, config_sha)``."""
    env = _envs.resolve_env(args.env_id)
    out = pathlib.Path(args.diag_out)
    _envs.validate_arm(args.env_id, args.mode, args.arm_id, args.exec_steps, args.cache_config)
    out.mkdir(parents=True, exist_ok=True)
    ckpt = args.policy_dir or ""
    identity = _envs.resource_identity(args.env_id, ckpt, args.cache_config)
    cfg_sha = _envs.sha256_file(args.cache_config) if args.cache_config else None
    manifest = _envs.RunManifest(
        experiment_id=args.experiment_id, env=env.to_json(), arm_id=args.arm_id, mode=args.mode,
        exec_steps=args.exec_steps if args.mode in ("plain", "full") else None, checkpoint=ckpt,
        checkpoint_sha256=identity["checkpoint_sha256"], cache_config=args.cache_config, cache_config_sha256=cfg_sha,
        library=identity["library"], library_sha256=identity["library_sha256"], code_commit=_envs.git_commit(),
        extras={"n_primary": args.n_primary, "n_dense_extra": args.n_dense_extra, "h_exec": _envs.H_EXEC,
                "runtime": identity["runtime"]},
    )
    config_sha = manifest.write(out / f"manifest_{args.arm_id}.json")
    spec = DiagSpec(
        experiment_id=args.experiment_id, env_id=env.env_id, arm_id=args.arm_id, mode=args.mode,
        k_full=env.k_full, k_set=tuple(env.k_set) if args.mode == "shadow" else (),
        warm_ts=tuple(env.warm_ts) if args.mode == "shadow" else (), n_primary=args.n_primary,
        n_dense_extra=args.n_dense_extra, action_shape=(env.action_horizon, env.action_dim),
        exec_steps=args.exec_steps, config_sha=config_sha,
    )
    return DiagRecorder(spec, out, launch_id=args.launch_id), config_sha


def install(args: argparse.Namespace, recorder: DiagRecorder) -> None:
    """Rebind the interceptor class, pin the executed step count, lock infer, stamp metadata."""
    from openpi.cache import interceptor as _icpt
    from openpi.serving import websocket_policy_server as wps

    from exp.step_diag.pi05 import Pi05DiagInterceptor

    lock = threading.Lock()
    mode = args.mode

    exec_steps = int(args.exec_steps) if mode in ("plain", "full") else None

    class _Bound(Pi05DiagInterceptor):
        def __init__(self, *a, **kw):
            super().__init__(*a, diag=recorder, mode=mode, exec_steps=exec_steps, **kw)

        def infer(self, obs, *, noise=None):
            with lock:
                return super().infer(obs, noise=noise)

    _icpt.InferenceInterceptor = _Bound
    if mode in ("plain", "full"):
        # Process-wide pin for the interceptor's own num_steps users (warm-up, orchestrator MISS,
        # response metadata); the executed no-cache path is pinned per instance in the subclass.
        _icpt._NUM_STEPS = int(args.exec_steps)  # noqa: SLF001

    orig_init = wps.WebsocketPolicyServer.__init__

    def _init(self, *a, **kw):
        md = dict(kw.get("metadata") or {})
        md.update({"step_diag_arm": args.arm_id, "step_diag_mode": mode, "step_diag_env": args.env_id,
                   "nfe_num_steps": args.exec_steps if mode in ("plain", "full") else _envs.resolve_env(args.env_id).k_full})
        kw["metadata"] = md
        return orig_init(self, *a, **kw)

    wps.WebsocketPolicyServer.__init__ = _init


def main(argv: list[str] | None = None) -> None:
    args, rest = parse(sys.argv[1:] if argv is None else argv)
    # The served path is authoritative; a separate manifest-only path must agree.
    served_path = next((v.split("=", 1)[1] for v in rest if v.startswith("--policy.dir=")), None)
    if "--policy.dir" in rest:
        served_path = rest[rest.index("--policy.dir") + 1]
    if not served_path or (args.policy_dir and pathlib.Path(served_path).resolve() != pathlib.Path(args.policy_dir).resolve()):
        raise ValueError("the served --policy.dir must match --policy-dir")
    args.policy_dir = served_path
    served_config = next((v.split("=", 1)[1] for v in rest if v.startswith("--policy.config=")), None)
    if "--policy.config" in rest:
        served_config = rest[rest.index("--policy.config") + 1]
    expected_config = "pi05_robocasa" if args.env_id == "pi05_rc" else "pi05_libero"
    if served_config != expected_config:
        raise ValueError(f"expected --policy.config {expected_config}")
    recorder, config_sha = build_recorder(args)
    install(args, recorder)
    print(f"STEP_DIAG arm={args.arm_id} mode={args.mode} env={args.env_id} exec_steps={args.exec_steps} "
          f"config_sha={config_sha[:12]} rows={recorder.rows_path}", flush=True)
    import tyro

    from scripts import serve_policy

    sys.argv = [sys.argv[0], *rest]
    try:
        serve_policy.main(tyro.cli(serve_policy.Args))
    finally:
        recorder.close()


if __name__ == "__main__":
    main()
