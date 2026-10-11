"""Resumable parallel CPU rollouts using the established conductor EpisodeRunner."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import fcntl
import importlib.metadata
import json
import multiprocessing
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import traceback

from exp.metaworld import tasks as T
from exp.mw_finetune.aggregate import aggregate
from exp.mw_finetune.journal import read_journal, write_json_atomic

_runner = None


def initialize(replan, renderer="osmesa", device_ids=(0,), counter=None):
    global _runner
    device = 0
    if counter is not None:
        with counter.get_lock():
            device = device_ids[counter.value % len(device_ids)]
            counter.value += 1
    from exp.mw_finetune.rendering import configure, make_env
    configure(renderer, device)
    from exp.metaworld.episode_runner import MetaworldEpisodeRunner
    from exp.mw_finetune.client import BoundedClient
    _runner = MetaworldEpisodeRunner(seed=7, replan_steps=replan, env_factory=make_env,
        client_factory=lambda server: BoundedClient(server.host, server.port))


def episode(job):
    from openpi.conductor.task import EpisodeTask, make_task_uid
    name, idx, model, host, port = job
    started = time.monotonic()
    task = EpisodeTask(task_uid=make_task_uid(model, "eval", T.task_id_of(name), idx),
                       yaml_id=model, phase="eval", experiment=T.BENCHMARK,
                       task_id=T.task_id_of(name), episode_idx=idx, orig_init_state_idx=idx,
                       server_host=host, server_port=port, bundle_id="default", extra={"task_name": name})
    row = {"task": name, "idx": idx, "seed": 7, "model": model, "task_uid": task.task_uid}
    try:
        result = _runner.run(task, lambda *args: None)
        row.update(success=bool(result.success), n_steps=result.n_steps, error=result.error,
                   per_step_rows=result.per_step_rows)
    except Exception:
        row.update(error=traceback.format_exc(), success=None)
    row["duration_s"] = time.monotonic() - started
    from exp.mw_finetune.rendering import STAMP
    row["worker"] = dict(STAMP)
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--model-id", required=True, help="unique checkpoint/run identifier")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8050)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--tasks", nargs="+", choices=T.TASK_NAMES)
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--replan-steps", type=int, choices=range(1, 6), default=5)
    ap.add_argument("--max-errors", type=int, default=10, help="stop after this many error attempts per invocation")
    ap.add_argument("--smoke", action="store_true", help="reach-v3 idx 0,1 (2 episodes)")
    ap.add_argument("--capacity-probe", action="store_true", help="explicitly permit zero policy; never a formal SR")
    node = socket.gethostname().split(".")[0]
    ap.add_argument("--renderer", choices=("egl", "osmesa"), default="egl" if node.startswith("timan") else "osmesa")
    ap.add_argument("--egl-devices", nargs="+", type=int, help="physical EGL indices, round-robin; default all NVIDIA GPUs")
    a = ap.parse_args()
    if a.workers < 1 or a.max_errors < 1 or not 1 <= a.trials <= 50:
        ap.error("workers must be positive; trials must be 1..50")
    if node.startswith("timan"):
        if a.workers > 30:
            ap.error("timan maximum is 30 workers (EGL context safety limit)")
        expected_host, port_min = ("ziyanglin.com", 23100) if node == "timan107" else ("149.165.153.233", 23200)
        if a.host != expected_host or not port_min <= a.port <= port_min + 99:
            ap.error(f"{node} lane requires {expected_host}:{port_min}..{port_min+99}")
    names = ["reach-v3"] if a.smoke else (a.tasks or list(T.TASK_NAMES))
    trials = 2 if a.smoke else a.trials
    if len(names) != len(set(names)):
        ap.error("duplicate tasks")
    a.run.mkdir(parents=True, exist_ok=True)
    lock = (a.run / ".eval.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    from exp.mw_finetune.client import BoundedClient
    client = BoundedClient(a.host, a.port)
    metadata = client.get_server_metadata()
    client.close()
    if metadata.get("pure_inference") is not True:
        ap.error("server must advertise pure_inference=true; use mw_finetune.serve")
    if metadata.get("test_policy") and not (a.smoke or a.capacity_probe):
        ap.error("test policies require --smoke or --capacity-probe")
    if a.capacity_probe and metadata.get("test_policy") is not True:
        ap.error("--capacity-probe requires a test policy")
    versions = {p: importlib.metadata.version(p) for p in ("metaworld", "gymnasium", "mujoco", "numpy", "msgpack", "websockets")}
    expected_versions = {"metaworld": "3.0.0", "gymnasium": "1.3.0", "mujoco": "3.14.0"}
    if any(versions[p] != v for p, v in expected_versions.items()):
        ap.error(f"simulator versions differ: {versions}; expected {expected_versions}")
    manifest = {"model_id": a.model_id, "server_metadata": metadata, "tasks": names, "trials": trials,
                "seed": 7, "replan_steps": a.replan_steps, "max_steps": T.MAX_POLICY_STEPS,
                "settle_steps": T.SETTLE_STEPS, "camera": T.CAMERA_NAME, "camera_pos": T.CAMERA_POS,
                "render_size": 480, "render_rotation_degrees": 180, "renderer": a.renderer, "versions": versions,
                "capacity_probe": a.capacity_probe}
    # JSON round-trip canonicalizes tuples for stable resume comparison.
    manifest = json.loads(json.dumps(manifest))
    mp = a.run / "manifest.json"
    if mp.exists() and json.loads(mp.read_text()) != manifest:
        ap.error("resume manifest differs (checkpoint/protocol/episode identities changed)")
    if not mp.exists():
        write_json_atomic(mp, manifest)
    journal = a.run / "episodes.jsonl"
    rows = read_journal(journal, recover_tail=True)
    aggregate(rows, names, trials, require_complete=False)  # detect duplicates and invalid identities
    done = {(r["task"], r["idx"]) for r in rows if not r.get("error")}
    jobs = [(name, idx, a.model_id, a.host, a.port) for name in names for idx in range(trials)
            if (name, idx) not in done]
    started = time.monotonic()
    new_rows = []
    pool = None
    error_count = 0
    device_ids = a.egl_devices or [0]
    if a.renderer == "egl" and a.egl_devices is None:
        device_ids = [int(x) for x in subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], text=True).split()]
        if not device_ids:
            ap.error("no NVIDIA devices for EGL")
    if min(device_ids) < 0 or len(set(device_ids)) != len(device_ids):
        ap.error("EGL devices must be unique nonnegative indices")
    ctx = multiprocessing.get_context("spawn")
    counter = ctx.Value("i", 0)

    def interrupted(signum, frame):
        raise KeyboardInterrupt("evaluation interrupted; completed episodes are durable")

    signal.signal(signal.SIGTERM, interrupted)
    with journal.open("a", buffering=1) as out:
        try:
            pool = ProcessPoolExecutor(max_workers=a.workers, mp_context=ctx,
                                       initializer=initialize, initargs=(a.replan_steps, a.renderer, device_ids, counter))
            futures = [pool.submit(episode, job) for job in jobs]
            for f in as_completed(futures):
                row = f.result()
                out.write(json.dumps(row) + "\n")
                out.flush()
                os.fsync(out.fileno())
                rows.append(row)
                new_rows.append(row)
                census = {f"{r['worker']['hostname']}:{r['worker']['pid']}": r["worker"]
                          for r in rows if r.get("worker")}
                write_json_atomic(a.run / "worker_metadata.json", census)
                print(f"{row['task']} idx={row['idx']} success={row['success']} error={bool(row['error'])}", flush=True)
                error_count += bool(row.get("error"))
                if error_count >= a.max_errors:
                    raise RuntimeError(f"{error_count} error attempts; stop and resume after fixing the server")
        except BaseException:
            if pool is not None:
                # Stop only this executor's own workers, leaving external sessions untouched.
                for process in list(pool._processes.values()):
                    if process.is_alive():
                        process.terminate()
                pool.shutdown(wait=True, cancel_futures=True)
            raise
        else:
            pool.shutdown(wait=True)
    elapsed = time.monotonic() - started
    invocation = {"workers": a.workers, "hostname": node, "endpoint": f"{a.host}:{a.port}", "renderer": a.renderer,
                  "egl_devices": device_ids if a.renderer == "egl" else [],
                  "seconds": elapsed, "new_completed_episodes": sum(not r.get("error") for r in new_rows),
                  "error_attempts": sum(bool(r.get("error")) for r in new_rows)}
    invocation["episodes_per_second"] = invocation["new_completed_episodes"] / max(elapsed, 1e-9)
    invocation["episodes_per_minute"] = 60 * invocation["episodes_per_second"]
    with (a.run / "invocations.jsonl").open("a") as out:
        out.write(json.dumps(invocation) + "\n")
    summary = aggregate(rows, names, trials, require_complete=False)
    if metadata.get("test_policy"):
        summary.update(formal_mt50=False, rate_scope="test policy capacity/smoke only; not a model success rate")
    summary["provenance"] = manifest
    summary["last_invocation"] = invocation
    write_json_atomic(a.run / "summary.json", summary)
    if not summary["complete"]:
        raise SystemExit("evaluation incomplete; rerun identical command to retry errors")
    print(json.dumps({"episodes": summary["completed_episodes"], "overall_per_task_mean": summary["overall_per_task_mean"]}))


if __name__ == "__main__":
    main()
