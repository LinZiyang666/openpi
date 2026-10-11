"""Probe-only render/horizon overrides, reusing A's episode function and MW kernel."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import functools
import importlib.metadata
import json
import multiprocessing
from pathlib import Path

import numpy as np

from exp.metaworld import env as E, tasks as T
from exp.metaworld.episode_runner import MetaworldEpisodeRunner
from exp.mw_finetune import evaluate as A
from exp.mw_finetune.aggregate import aggregate
from exp.mw_finetune.client import BoundedClient
from exp.mw_finetune.journal import write_json_atomic


class RawView:
    def __init__(self, env):
        self.env = env

    def __getattr__(self, name):
        return getattr(self.env, name)

    def render(self):
        # E.render_image rotates this once more, yielding the exact raw view.
        return np.ascontiguousarray(self.env.render()[::-1, ::-1])


def make_probe_env(task, idx, seed):
    benchmark = E._benchmark(task, seed)
    env = benchmark.train_classes[task](render_mode="rgb_array", camera_name="corner3", width=256, height=256)
    env.set_task(benchmark.train_tasks[idx])
    return RawView(env)


def initialize(replan, max_steps):
    A.initialize(replan)
    A._runner = MetaworldEpisodeRunner(
        seed=7, replan_steps=replan, env_factory=make_probe_env,
        client_factory=lambda s: BoundedClient(s.host, s.port),
        run_episode_fn=functools.partial(E.run_episode, max_steps=max_steps))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--task", choices=T.TASK_NAMES, required=True)
    ap.add_argument("--trials", type=int, required=True)
    ap.add_argument("--max-steps", type=int, default=160)
    a = ap.parse_args()
    a.run.mkdir(parents=True, exist_ok=False)
    client = BoundedClient("127.0.0.1", a.port)
    metadata = client.get_server_metadata()
    client.close()
    manifest = {"model_id": a.model_id, "server_metadata": metadata, "tasks": [a.task], "trials": a.trials,
                "seed": 7, "replan_steps": 5, "max_steps": a.max_steps, "settle_steps": 15,
                "camera": "corner3", "camera_pos": "unmodified model default", "render_size": 256,
                "render_rotation_degrees": 0, "renderer": "osmesa", "probe_only": True,
                "versions": {k: importlib.metadata.version(k) for k in ("metaworld", "gymnasium", "mujoco")}}
    write_json_atomic(a.run / "manifest.json", manifest)
    jobs = [(a.task, idx, a.model_id, "127.0.0.1", a.port) for idx in range(a.trials)]
    rows = []
    with (a.run / "episodes.jsonl").open("x", buffering=1) as journal:
        with ProcessPoolExecutor(max_workers=a.workers, mp_context=multiprocessing.get_context("spawn"),
                                 initializer=initialize, initargs=(5, a.max_steps)) as pool:
            for future in as_completed([pool.submit(A.episode, job) for job in jobs]):
                row = future.result()
                rows.append(row)
                journal.write(json.dumps(row) + "\n")
                print(f"{row['task']} idx={row['idx']} success={row['success']} error={bool(row.get('error'))}", flush=True)
    summary = aggregate(rows, [a.task], a.trials, require_complete=False)
    summary["provenance"] = manifest
    write_json_atomic(a.run / "summary.json", summary)
    if not summary["complete"]:
        raise SystemExit("probe incomplete; inspect errors")


if __name__ == "__main__":
    main()
