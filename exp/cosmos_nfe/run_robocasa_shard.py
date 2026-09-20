"""Run the official Cosmos Policy RoboCasa (2024, 24 tasks) evaluation on one task and a trial range.

The upstream script (``cosmos_policy.experiments.robot.robocasa.run_robocasa_eval``) evaluates
one ``--task_name`` for ``--num_trials_per_task`` episodes in one process (trial ``i`` is a fresh
``robosuite.make`` with seed ``cfg.seed * i * 256`` and test scene ``(i // 10) % 5``). To spread a
task's 50 trials over many simulator workers, this wrapper replaces ``run_task`` with the same
per-trial loop restricted to ``--trials a-b`` (inclusive, upstream episode indices, so seeds and
scenes are unchanged) and skips the rollout-video writing; ``run_episode`` and ``get_action``
are the upstream functions, wrapped only for bookkeeping. The denoising step count is upstream's
own ``--num_denoising_steps_action``.

Usage (inside the cosmos-policy uv env, fork robocasa installed, kitchen assets present)::

    python -m exp.cosmos_nfe.run_robocasa_shard --trials 0-9 --results-json out.json \\
        --config cosmos_predict2_2b_480p_robocasa_50_demos_per_task__inference \\
        --ckpt_path nvidia/Cosmos-Policy-RoboCasa-Predict2-2B --task_name TurnOffMicrowave ... \\
        --num_denoising_steps_action 1

With ``--remote ws://host:port`` the model is not loaded locally: ``get_model`` is stubbed and
``get_action`` is answered by ``exp.cosmos_nfe.serve_cosmos --bench robocasa`` on that host.
Arguments other than ``--trials`` / ``--results-json`` / ``--remote`` are passed through to upstream.
"""

from __future__ import annotations

import dataclasses
import json
import os
import pathlib
import statistics
import sys
import time

from exp.cosmos_nfe.run_libero_shard import _install_remote, _take


def main() -> None:
    argv = sys.argv[1:]
    trials = _take(argv, "--trials")
    out = _take(argv, "--results-json")
    if trials is None or out is None:
        raise SystemExit("--trials a-b and --results-json are required")
    remote = _take(argv, "--remote")
    a, b = (int(x) for x in trials.split("-"))
    trial_ids = list(range(a, b + 1))
    out_path = pathlib.Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sys.argv = [sys.argv[0], *argv]

    from cosmos_policy.experiments.robot.robocasa import run_robocasa_eval as R
    from cosmos_policy.experiments.robot.robot_utils import log_message

    record: dict = {"trial_ids": trial_ids, "task": None, "latency_ms": [], "started": time.time()}
    state = {"cfg": None, "cur": None}
    orig_run_episode = R.run_episode
    orig_get_action = R.get_action

    def run_task(cfg, task_name, model, planning_model, dataset_stats, worker_pool, log_file=None):
        """Upstream ``run_task`` restricted to ``trial_ids``; identical env creation, reset and episode call."""
        state["cfg"] = cfg
        cur = {"task_name": task_name, "trials": {}, "descriptions": {}, "episodes": [], "episode_s": [], "lengths": []}
        state["cur"] = cur
        record["task"] = cur
        t0 = time.time()
        successes, lengths = [], []
        for episode_idx in trial_ids:
            log_message(f"Starting episode {episode_idx + 1}...", log_file)
            seed = cfg.seed * episode_idx * 256 if (cfg.deterministic or cfg.deterministic_reset) else None
            env, _ = R.create_robocasa_env(cfg, seed=seed, episode_idx=episode_idx)
            if cfg.deterministic_reset:
                reset_seed = cfg.deterministic_reset_seed if cfg.deterministic_reset_seed is not None else cfg.seed
                R.set_seed_everywhere(reset_seed)
            env.reset()
            task_description = env.get_ep_meta()["lang"]
            log_message(f"\nTask description: {task_description}", log_file)
            success, length, *_ = R.run_episode(
                cfg, env, task_description, model, planning_model, dataset_stats, worker_pool, episode_idx, log_file
            )
            env.close()
            successes.append(bool(success))
            lengths.append(int(length))
            cur["trials"][str(episode_idx)] = bool(success)
            cur["descriptions"][str(episode_idx)] = task_description
            cur["lengths"].append(int(length))
            log_message(f"Success: {success}  ({sum(successes)}/{len(successes)} so far)", log_file)
            _flush(record, out_path, cfg, final=False)
        cur["n"] = len(successes)
        cur["successes"] = sum(successes)
        cur["wall_s"] = time.time() - t0
        _flush(record, out_path, cfg, final=False)
        sr = float(sum(successes) / len(successes)) if successes else 0.0
        avg_len = float(sum(lengths) / len(lengths)) if lengths else 0.0
        log_message(f"Task {task_name} results: {sum(successes)}/{len(successes)} = {sr:.4f}", log_file)
        return sr, avg_len, successes

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

    R.eval_robocasa()
    _flush(record, out_path, state["cfg"], final=True)


def _flush(record: dict, out_path: pathlib.Path, cfg, final: bool) -> None:
    lat = record["latency_ms"]
    task = record.get("task") or {}
    summary = {
        "complete": final and task.get("n", 0) == len(record["trial_ids"]),
        "suite": "robocasa",
        "task_name": getattr(cfg, "task_name", None),
        "trial_ids": record["trial_ids"],
        "num_denoising_steps_action": getattr(cfg, "num_denoising_steps_action", None),
        "seed": getattr(cfg, "seed", None),
        "deterministic": getattr(cfg, "deterministic", None),
        "ckpt_path": getattr(cfg, "ckpt_path", None),
        "obj_instance_split": getattr(cfg, "obj_instance_split", None),
        "layout_and_style_ids": getattr(cfg, "layout_and_style_ids", None),
        "n_episodes": task.get("n", len(task.get("trials", {}))),
        "n_successes": task.get("successes", sum(task.get("trials", {}).values())),
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
        "task": task,
        "config": dataclasses.asdict(cfg) if cfg is not None and dataclasses.is_dataclass(cfg) else None,
    }
    tmp = out_path.with_name(out_path.name + ".tmp")
    tmp.write_text(json.dumps(summary, indent=1, default=str))
    os.replace(tmp, out_path)


if __name__ == "__main__":
    main()
