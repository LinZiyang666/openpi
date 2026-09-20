"""Run the official Cosmos Policy LIBERO evaluation on a subset of tasks and write a results JSON.

The upstream script (``cosmos_policy.experiments.robot.libero.run_libero_eval``) evaluates
all tasks of a suite serially in one process; its only parallelism is best-of-N queries
across GPUs. To use one 48 GB card for several processes, this wrapper monkeypatches
``run_task`` so a process only evaluates ``--task-ids``, and records per-episode
successes and per-query latency so the shards can be joined afterwards. Everything else
(model loading, denoising, environment, seeding, logging) is the upstream code path;
the denoising step count is upstream's own ``--num_denoising_steps_action``.

Usage (inside the cosmos-policy uv env)::

    python -m exp.cosmos_nfe.run_libero_shard --task-ids 0,5 --results-json out.json \\
        --config cosmos_predict2_2b_480p_libero__inference_only \\
        --ckpt_path nvidia/Cosmos-Policy-LIBERO-Predict2-2B ... --num_denoising_steps_action 1

With ``--remote ws://host:port`` the model is not loaded locally: ``get_model`` is stubbed and
``get_action`` is answered by ``exp.cosmos_nfe.serve_cosmos`` on that host (same upstream
``get_action``, same seed and step count), so the simulator side runs on a client box.

Arguments other than ``--task-ids`` / ``--results-json`` / ``--remote`` are passed through to upstream.
"""

from __future__ import annotations

import dataclasses
import json
import os
import pathlib
import statistics
import sys
import time


def _take(argv: list[str], flag: str) -> str | None:
    """Remove ``flag value`` (or ``flag=value``) from argv and return the value."""
    for i, tok in enumerate(argv):
        if tok == flag and i + 1 < len(argv):
            val = argv[i + 1]
            del argv[i : i + 2]
            return val
        if tok.startswith(flag + "="):
            del argv[i]
            return tok.split("=", 1)[1]
    return None


def main() -> None:
    argv = sys.argv[1:]
    ids = _take(argv, "--task-ids")
    out = _take(argv, "--results-json")
    if ids is None or out is None:
        raise SystemExit("--task-ids and --results-json are required")
    remote = _take(argv, "--remote")
    task_ids = sorted({int(x) for x in ids.split(",") if x.strip()})
    out_path = pathlib.Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sys.argv = [sys.argv[0], *argv]

    from cosmos_policy.experiments.robot.libero import run_libero_eval as R

    record: dict = {"task_ids": task_ids, "tasks": {}, "latency_ms": [], "started": time.time()}
    state = {"cfg": None, "cur": None}

    orig_run_task = R.run_task
    orig_run_episode = R.run_episode
    orig_get_action = R.get_action

    def run_task(cfg, task_suite, task_id, model, planning_model, dataset_stats, worker_pool,
                 resize_size, total_episodes=0, total_successes=0, log_file=None):
        if task_id not in task_ids:
            return total_episodes, total_successes
        state["cfg"] = cfg
        desc = task_suite.get_task(task_id).language
        state["cur"] = {"task_id": task_id, "description": desc, "episodes": [], "episode_s": []}
        t0 = time.time()
        n_ep, n_ok = orig_run_task(cfg, task_suite, task_id, model, planning_model, dataset_stats,
                                   worker_pool, resize_size, total_episodes, total_successes, log_file)
        cur = state["cur"]
        cur["n"] = n_ep - total_episodes
        cur["successes"] = n_ok - total_successes
        cur["wall_s"] = time.time() - t0
        record["tasks"][str(task_id)] = cur
        state["cur"] = None
        _flush(record, out_path, state["cfg"], final=False)
        return n_ep, n_ok

    def run_episode(*a, **kw):
        t0 = time.time()
        res = orig_run_episode(*a, **kw)
        if state["cur"] is not None:
            state["cur"]["episodes"].append(bool(res[0]))
            state["cur"]["episode_s"].append(round(time.time() - t0, 2))
        return res

    def get_action(*a, **kw):
        t0 = time.perf_counter()
        res = orig_get_action(*a, **kw)
        record["latency_ms"].append(round((time.perf_counter() - t0) * 1000.0, 2))
        return res

    R.run_task = run_task
    R.run_episode = run_episode
    R.get_action = get_action
    if remote:
        _install_remote(R, remote, record)

    R.eval_libero()
    _flush(record, out_path, state["cfg"], final=True)


def _install_remote(R, url: str, record: dict) -> None:
    """Stub the local model and route ``get_action`` to a serve_cosmos websocket server."""
    import types

    import msgpack
    import msgpack_numpy
    import numpy as np
    from websockets.sync.client import connect

    msgpack_numpy.patch()
    conn = {"ws": None}

    def _ws():
        if conn["ws"] is None:
            conn["ws"] = connect(url, max_size=64 * 1024 * 1024, open_timeout=60)
            conn["ws"].send(msgpack.packb({"ping": 1}, use_bin_type=True))
            pong = msgpack.unpackb(conn["ws"].recv(), raw=False)
            print(f"REMOTE connected {url} server_default_k={pong.get('num_denoising_steps_action')} host={pong.get('host')}", flush=True)
            record["remote"] = {"url": url, **{k: v for k, v in pong.items() if k != "pong"}}
        return conn["ws"]

    def get_model(cfg):
        stub = types.SimpleNamespace(dataloader_train=types.SimpleNamespace(dataset=types.SimpleNamespace(chunk_size=cfg.chunk_size)))
        return None, stub

    orig_get_action = R.get_action  # the timing wrapper installed above

    def remote_call(cfg, model, dataset_stats, obs, task_label, seed=1, randomize_seed=False,
                    num_denoising_steps_action=5, generate_future_state_and_value_in_parallel=True, **kw):
        req = {
            "obs": {k: np.asarray(v) for k, v in obs.items()},
            "task_description": task_label,
            "seed": int(seed),
            "randomize_seed": bool(randomize_seed),
            "num_denoising_steps_action": int(num_denoising_steps_action),
            "generate_future_state_and_value_in_parallel": bool(generate_future_state_and_value_in_parallel),
            "deterministic": bool(cfg.deterministic),
        }
        for attempt in range(3):
            try:
                ws = _ws()
                ws.send(msgpack.packb(req, use_bin_type=True))
                resp = msgpack.unpackb(ws.recv(), raw=False)
                break
            except Exception as exc:  # reconnect once on a dropped socket
                print(f"REMOTE error ({attempt}): {exc}", flush=True)
                conn["ws"] = None
                if attempt == 2:
                    raise
        actions = np.asarray(resp["actions"])
        record.setdefault("server_ms", []).append(float(resp.get("server_ms", 0.0)))
        return {
            "actions": [actions[i] for i in range(len(actions))],
            "value_prediction": resp.get("value_prediction"),
            "future_image_predictions": resp.get("future_image_predictions"),
        }

    R.get_model = get_model
    R.init_t5_text_embeddings_cache = lambda *a, **kw: {}  # the server owns the T5 cache; keep the client GPU-free
    # keep the outer timing wrapper: it calls R.get_action's original, so rebind the original to the remote call
    R.get_action = lambda *a, **kw: _timed(orig_get_action, remote_call, record, *a, **kw)


def _timed(timing_wrapper, fn, record, *a, **kw):
    """Run ``fn`` through the same latency bookkeeping as the local path."""
    t0 = time.perf_counter()
    res = fn(*a, **kw)
    record["latency_ms"].append(round((time.perf_counter() - t0) * 1000.0, 2))
    return res


def _flush(record: dict, out_path: pathlib.Path, cfg, final: bool) -> None:
    lat = record["latency_ms"]
    summary = {
        "complete": final,
        "task_ids": record["task_ids"],
        "suite": getattr(cfg, "task_suite_name", None),
        "num_denoising_steps_action": getattr(cfg, "num_denoising_steps_action", None),
        "seed": getattr(cfg, "seed", None),
        "deterministic": getattr(cfg, "deterministic", None),
        "ckpt_path": getattr(cfg, "ckpt_path", None),
        "num_trials_per_task": getattr(cfg, "num_trials_per_task", None),
        "n_episodes": sum(t.get("n", 0) for t in record["tasks"].values()),
        "n_successes": sum(t.get("successes", 0) for t in record["tasks"].values()),
        "latency_ms": {
            "n": len(lat),
            "mean": round(statistics.fmean(lat), 2) if lat else None,
            "p50": round(statistics.median(lat), 2) if lat else None,
            "p90": round(sorted(lat)[int(0.9 * (len(lat) - 1))], 2) if lat else None,
        },
        "wall_s": round(time.time() - record["started"], 1),
        "hostname": os.uname().nodename,
        "remote": record.get("remote"),
        "server_ms_mean": round(statistics.fmean(record["server_ms"]), 2) if record.get("server_ms") else None,
        "gpu": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "tasks": record["tasks"],
        "config": dataclasses.asdict(cfg) if cfg is not None and dataclasses.is_dataclass(cfg) else None,
    }
    tmp = out_path.with_name(out_path.name + ".tmp")
    tmp.write_text(json.dumps(summary, indent=1, default=str))
    os.replace(tmp, out_path)


if __name__ == "__main__":
    main()
