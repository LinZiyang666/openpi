"""A few full episodes through examples/libero/main.py's verified loop (sim island)."""
from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path
import time


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="149.165.153.233")
    ap.add_argument("--port", type=int, default=23254)
    ap.add_argument("--suites", nargs="+", default=["libero_spatial", "libero_object", "libero_goal", "libero_10"])
    ap.add_argument("--tasks", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--output", type=Path, default=Path("/data/mw_finetune/dp/libero_smoke.json"))
    a = ap.parse_args()
    from examples.libero import main as m
    from libero.libero import benchmark
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
    client = WebsocketClientPolicy(host=a.host, port=a.port)
    client.select_bundle("default")
    metadata = client.get_server_metadata()
    rows = []
    for suite_name in a.suites:
      suite = benchmark.get_benchmark_dict()[suite_name]()
      args = m.Args(resize_size=96, replan_steps=5, num_trials_per_task=a.trials, task_suite_name=suite_name)
      for task_id in a.tasks:
        task = suite.get_task(task_id)
        for idx in range(a.trials):
            env = None
            started = time.monotonic()
            try:
                with open("/tmp/rc365_env_build.lock", "a") as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX)
                    env, language = m._get_libero_env(task, m.LIBERO_ENV_RESOLUTION, 7)
                    env.reset()
                from OpenGL import GL
                renderer = {key: GL.glGetString(getattr(GL, key)).decode() for key in ("GL_RENDERER", "GL_VENDOR", "GL_VERSION")}
                initial = suite.get_task_init_states(task_id)[idx]
                client.episode_start(experiment=suite_name, task=task.name, episode_id=task_id * a.trials + idx)
                result = m._run_episode(env, client, initial, language, args, m._get_max_steps(suite_name))
                client.episode_end(success=bool(result[0]))
                rows.append({"suite": suite_name, "task_id": task_id, "task": task.name, "idx": idx,
                    "success": bool(result[0]), "final_env_timestep": result[4], "duration_s": time.monotonic()-started,
                    "renderer": renderer})
            finally:
                if env is not None:
                    with open("/tmp/rc365_env_build.lock", "a") as lock:
                        fcntl.flock(lock, fcntl.LOCK_EX); env.close()
            a.output.parent.mkdir(parents=True, exist_ok=True)
            a.output.write_text(json.dumps({"metadata": metadata, "episodes": rows, "smoke_only": True,
                "resize": 96, "replan_steps": 5, "seed": 7}, indent=2))
            print(json.dumps(rows[-1]), flush=True)
    client.close()


if __name__ == "__main__":
    main()
