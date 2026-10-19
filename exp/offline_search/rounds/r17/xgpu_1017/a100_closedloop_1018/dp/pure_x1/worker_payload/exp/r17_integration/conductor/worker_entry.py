"""Client-only Python 3.11 worker using the existing renderer and WorkerLoop."""
from __future__ import annotations

import argparse
import functools
import os
import socket
import subprocess

CLIENT_ONLY_FILES = (
    "exp/__init__.py", "exp/mw_finetune/__init__.py", "src/openpi/__init__.py",
    "exp/metaworld/__init__.py", "exp/metaworld/tasks.py", "exp/metaworld/env.py",
    "exp/metaworld/episode_runner.py", "exp/mw_finetune/rendering.py",
    "exp/r17_integration/benchmarks/spec.py", "exp/r17_integration/benchmarks/metaworld.py",
    "exp/r17_integration/conductor/manifest.py", "exp/libero_groot/emit_task_map.py",
    "exp/r17_integration/benchmarks/physical_mt50.py", "exp/r17_integration/benchmarks/execution.py",
    "exp/r17_integration/benchmarks/libero_bridge.py", "examples/libero/episode_runner.py",
    "examples/__init__.py", "examples/libero/__init__.py", "examples/libero/collect_util.py", "examples/libero/main.py", "examples/libero/worker_entry.py", "exp/r17_integration/conductor/worker_entry.py",
    "src/openpi/conductor/__init__.py", "src/openpi/conductor/task.py", "src/openpi/conductor/protocol.py",
    "src/openpi/conductor/worker.py",
    "src/openpi/conductor/agent.py", "src/openpi/conductor/driver.py", "src/openpi/conductor/scheduler.py",
    "src/openpi/conductor/strategy.py", "src/openpi/conductor/journal.py",
    "src/openpi/conductor/health.py", "src/openpi/conductor/monitor.py", "src/openpi/conductor/sharding.py",
    "exp/r17_integration/common/__init__.py", "exp/r17_integration/common/contracts.py",
)


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--worker-id", required=True)
    ap.add_argument("--server-key", required=True)
    ap.add_argument("--driver-host", required=True)
    ap.add_argument("--driver-port", required=True, type=int)
    ap.add_argument("--benchmark", choices=("metaworld", "libero"), default="metaworld")
    ap.add_argument("--benchmark-spec", help="sealed LIBERO benchmark JSON (all four native task maps)")
    ap.add_argument("--init-states-dir", default="", help="LIBERO suite=pool map; original indices")
    ap.add_argument("--renderer", choices=("egl", "osmesa"), default="egl")
    ap.add_argument("--egl-device", type=int, default=0)
    ap.add_argument("--import-probe", action="store_true")
    ap.add_argument("--debug-directory", help="explicit P6 physical capture directory, default off")
    return ap


def import_probe():
    import sys
    from exp.r17_integration.benchmarks.metaworld import MetaWorldBenchmarkAdapter
    from exp.r17_integration.benchmarks.libero_bridge import LiberoBenchmarkSpec
    from openpi.conductor.worker import WorkerLoop
    if any(k in sys.modules for k in ("torch", "jax", "metaworld", "mujoco", "libero")):
        raise RuntimeError("worker import unexpectedly loaded a model/simulator")
    return {"python": sys.version.split()[0], "client_only": True,
            "benchmark_protocol": MetaWorldBenchmarkAdapter().spec.identity}


def attest_worker(benchmark, renderer_stamp):
    """One owned environment validates camera/GL; no policy/model calls."""
    from exp.metaworld import env as kernel
    environment = benchmark.env_factory("reach-v3", 0, 7)
    try:
        kernel.reset_and_settle(environment)
        image = kernel.render_image(environment)
        if image.shape != (480, 480, 3) or str(image.dtype) != "uint8":
            raise ValueError("worker camera probe failed")
        task_data = benchmark.task_data_identity("reach-v3", 0)
        return benchmark.worker_probe(renderer_stamp(), task_data)
    finally:
        environment.close()



def validate_capture_identity(metadata, event):
    """A fixed-model worker must not write another checkpoint's physics binding."""
    from exp.r17_integration.common.contracts import digest as contract_digest
    spec = metadata["r17_model_spec"]
    expected = {"model_identity": spec["checkpoint_identity"], "norm_identity": spec["norm_identity"],
                "layout_identity": contract_digest(spec["layout"])}
    for name, value in expected.items():
        if event.get(name) is not None and event[name] != value:
            raise ValueError("worker physical capture model binding mismatch: " + name)

def main(argv=None):
    a = parser().parse_args(argv)
    if a.import_probe:
        import json
        print(json.dumps(import_probe(), sort_keys=True))
        return
    from openpi.conductor.worker import WorkerLoop
    if a.benchmark == "libero":
        if not a.benchmark_spec:
            raise ValueError("LIBERO worker requires a sealed --benchmark-spec")
        import json
        from dataclasses import replace
        from .manifest import load_benchmark_spec
        from exp.r17_integration.benchmarks.libero_bridge import LiberoBenchmarkAdapter, LiberoBenchmarkSpec, SUITES
        from examples.libero import main as native
        from examples.libero.worker_entry import _build_episode_setup, build_probe
        spec = load_benchmark_spec(a.benchmark_spec, register=False)
        if not isinstance(spec, LiberoBenchmarkSpec) or spec.identity != LiberoBenchmarkSpec.from_official(
                seed=spec.seed, seed_mode=spec.seed_mode, settle_steps=spec.settle_steps, replan_steps=spec.replan_steps,
                wire_size=spec.wire_size, max_steps_by_suite=spec.max_steps_by_suite,
                formal_indices=spec.formal_indices, library_indices=spec.library_indices, dev_indices=spec.dev_indices,
                protocol_id=spec.protocol_id).identity:
            raise ValueError("worker official LIBERO task map differs from sealed benchmark")
        benchmark = LiberoBenchmarkAdapter(spec)
        args = native.Args(task_suite_name=SUITES[0], seed=spec.seed, resize_size=spec.wire_size,
                           replan_steps=spec.replan_steps, num_steps_wait=spec.settle_steps)
        setup = _build_episode_setup(args, spec.seed, a.init_states_dir, "orig")
        def libero_setup(task):
            env, initial, prompt, _ = setup(replace(task, task_id=task.task_id % 10))
            return env, initial, prompt, spec.max_steps_by_suite[task.task_id // 10]
        runner = benchmark.make_episode_runner(args, libero_setup, framework_context=True)
        def attest():
            probe = benchmark.worker_probe(build_probe(args, a.init_states_dir))
            # Attest one original pool state with the same setup; no policy/render forwards.
            from openpi.conductor.task import EpisodeTask
            task = EpisodeTask("probe", "probe", "eval", SUITES[0], 0, 0, 0, "unused", 0, "probe")
            env, state, prompt, _ = libero_setup(task)
            try:
                import numpy as np
                import hashlib
                raw = state.detach().cpu().numpy() if hasattr(state, "detach") else np.asarray(state)
                probe["task_data"] = {"task_name": spec.tasks[0][0], "original_init_idx": 0,
                                      "state_sha256": hashlib.sha256(raw.tobytes()).hexdigest()}
            finally:
                env.close()
            return probe
    else:
        from exp.mw_finetune.rendering import configure, make_env, STAMP
        from exp.r17_integration.benchmarks.metaworld import MetaWorldBenchmarkAdapter
        configure(a.renderer, a.egl_device)
        from .manifest import load_benchmark_spec
        from exp.r17_integration.benchmarks.metaworld import FORMAL_SPEC
        spec = load_benchmark_spec(a.benchmark_spec, register=False) if a.benchmark_spec else FORMAL_SPEC
        benchmark = MetaWorldBenchmarkAdapter(spec=spec, env_factory=make_env)
        runner = benchmark.make_episode_runner(framework_context=True)
        attest = lambda: attest_worker(benchmark, lambda: dict(STAMP))
    bridge = None
    sink = None
    if a.debug_directory:
        if a.benchmark == "libero":
            raise ValueError("native LIBERO physical capture adapter remains unsupported; model capture is server-side")
        from openpi_client.websocket_client_policy import WebsocketClientPolicy
        from exp.r17_integration.debug.capture_bridge import PhysicalCaptureFactory
        host, port = a.server_key.rsplit(":", 1)
        ctl = WebsocketClientPolicy(host, int(port))
        try:
            metadata = ctl.get_server_metadata()
        finally:
            ctl.close()
        # All grids come from attested serving metadata, never guessed by K.
        def sampler_for(k):
            sampler = metadata.get("r17_samplers", {}).get(str(k))
            if sampler is None and k == metadata["r17_model_spec"]["default_sampler"]["num_steps"]:
                sampler = metadata["r17_model_spec"]["default_sampler"]
            if sampler is None:
                raise ValueError("endpoint did not attest the requested native sampler")
            return sampler
        bridge = PhysicalCaptureFactory(a.debug_directory, metadata, sampler_resolver=sampler_for)
        def capture(event):
            validate_capture_identity(metadata, event)
            bridge(event)
        runner.capture_sink = capture
        runner.execution_ack = lambda row: None
    try:
        probe = attest()
        WorkerLoop(a.worker_id, a.server_key, runner,
                   connect=lambda: socket.create_connection((a.driver_host, a.driver_port)), probe=probe).run_forever()
    finally:
        runner.close()
        if bridge:
            bridge.close()


def worker_command(spec, driver_host, driver_port, *, worker_python, benchmark="metaworld",
                   benchmark_spec=None, init_states_dir="", debug_directory=None):
    command = [worker_python, "-m", "exp.r17_integration.conductor.worker_entry", "--worker-id", spec.worker_id,
            "--server-key", spec.server_key, "--driver-host", driver_host, "--driver-port", str(driver_port),
            "--egl-device", str(spec.gpu_id)]
    if benchmark != "metaworld":
        command += ["--benchmark", benchmark]
    if benchmark == "libero" and benchmark_spec is None:
        raise ValueError("LIBERO worker requires a sealed benchmark spec")
    if benchmark_spec is not None:
        command += ["--benchmark-spec", str(benchmark_spec)]
    if init_states_dir:
        command += ["--init-states-dir", init_states_dir]
    if debug_directory is not None:
        command += ["--debug-directory", str(debug_directory)]
    return command


def spawn_worker(spec, driver_host, driver_port, *, worker_python, repo_root, **options):
    from exp.metaworld.warm_reset_env import worker_environ
    return subprocess.Popen(worker_command(spec, driver_host, driver_port, worker_python=worker_python, **options),
                            env=worker_environ(spec, repo_root=repo_root), start_new_session=True)


def make_worker_agent(specs, driver_host, driver_port, *, worker_python, repo_root, **options):
    from openpi.conductor.agent import WorkerAgent
    return WorkerAgent(specs, driver_host, driver_port,
                       spawn_fn=functools.partial(spawn_worker, worker_python=worker_python, repo_root=repo_root, **options))


if __name__ == "__main__":
    main()
