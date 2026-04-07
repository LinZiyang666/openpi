"""Experiment runner: sequentially execute cache experiment runs.

Tasks within each run are batched by --num-workers and executed concurrently
via main.py.  Progress is persisted after every batch, so interrupted runs
resume from the last completed batch — not from scratch.

Usage:
    uv run exp/run_cache_experiments.py \
        --yaml-dir configs/cache_runs/phase1 \
        --episodes-per-run 5 \
        --num-workers 5 \
        --host 155.98.36.13 --port 9000 \
        --task-suite libero_spatial \
        --seed 42 --conda-env libero_sim

    # Resume from checkpoint
    uv run exp/run_cache_experiments.py \
        --yaml-dir configs/cache_runs/phase1 \
        --episodes-per-run 5 --num-workers 5 \
        --host 155.98.36.13 --port 9000 \
        --task-suite libero_spatial \
        --seed 42 --conda-env libero_sim \
        --resume
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import tqdm

logger = logging.getLogger(__name__)

# LIBERO suite sizes (must match benchmark).
_SUITE_NUM_TASKS = {
    "libero_spatial": 10,
    "libero_object": 10,
    "libero_goal": 10,
    "libero_10": 10,
    "libero_90": 90,
}


@dataclass
class RunState:
    yaml_path: str
    run_id: str
    status: str = "pending"          # pending | running | done | failed
    episodes_per_task: int = 0       # trials per task (not total across all tasks)
    task_suite: str = ""             # e.g. "libero_spatial" — for resume validation
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    success_rate: Optional[float] = None
    log_path: Optional[str] = None
    # Per-task progress: {"0": "done", "1": "done", "2": "pending", ...}
    # Keys are stringified task IDs.  Values: "done" | "failed" | "pending"
    task_progress: dict[str, str] = field(default_factory=dict)
    # Per-task success counts: {"0": [successes, total], ...}
    task_results: dict[str, list[int]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Server control
# ---------------------------------------------------------------------------


def _send_cache_config(server_url: str, yaml_path: str) -> None:
    """Send load_cache_config control message with YAML content via WebSocket.

    Sends yaml_content (the file body) so the server doesn't need the local path.
    """
    import msgpack
    import websockets

    yaml_content = Path(yaml_path).read_text()

    async def _send():
        async with websockets.connect(server_url) as ws:
            _metadata = await ws.recv()
            msg = {"__ctrl__": "load_cache_config", "yaml_content": yaml_content}
            await ws.send(msgpack.packb(msg))
            resp = msgpack.unpackb(await ws.recv())
            if resp.get("__ack__") != "load_cache_config":
                raise RuntimeError(f"Config switch failed: {resp}")
            logger.debug("Server switched to bundle v%s: %s", resp.get("version"), yaml_path)

    asyncio.run(_send())


# ---------------------------------------------------------------------------
# Task batch execution
# ---------------------------------------------------------------------------


_TASK_TIMEOUT_PER_TASK = 3600  # 1 hour per task


def _execute_tasks(
    task_ids: list[int],
    episodes_per_task: int,
    num_workers: int,
    host: str,
    port: int,
    task_suite: str,
    log_path: Path,
    seed: int = 7,
    conda_env: str | None = None,
) -> dict:
    """Execute a batch of tasks concurrently via main.py.

    Passes multiple --task-ids so main.py distributes them across workers.
    Appends output to the run's log file.
    """
    task_id_strs = [str(t) for t in task_ids]
    main_args = [
        "examples/libero/main.py",
        "--host", host,
        "--port", str(port),
        "--task-suite-name", task_suite,
        "--num-trials-per-task", str(episodes_per_task),
        "--num-workers", str(num_workers),
        "--task-ids", *task_id_strs,
        "--seed", str(seed),
    ]
    env = None
    if conda_env:
        cmd = ["conda", "run", "--no-capture-output", "-n", conda_env, "python", *main_args]
        # Clean env: uv injects VIRTUAL_ENV / PYTHONPATH / PATH that override conda's paths.
        env = {k: v for k, v in os.environ.items()
               if k not in ("VIRTUAL_ENV", "PYTHONPATH", "PYTHONHOME")}
        # Remove .venv/bin from PATH so conda's python takes priority.
        venv_bin = os.environ.get("VIRTUAL_ENV", "")
        if venv_bin:
            venv_bin = os.path.join(venv_bin, "bin")
            env["PATH"] = os.pathsep.join(
                p for p in env.get("PATH", "").split(os.pathsep) if p != venv_bin
            )
        env["MUJOCO_GL"] = "egl"
    else:
        cmd = ["uv", "run", *main_args]

    batch_label = ",".join(task_id_strs)
    with open(log_path, "a") as log_file:
        log_file.write(f"\n{'='*60}\n")
        log_file.write(f"TASKS [{batch_label}] started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"CMD: {' '.join(cmd)}\n")
        log_file.write(f"{'='*60}\n")
        log_file.flush()

        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            env=env,
        )

        # Read stdout in a daemon thread so the main thread can enforce timeout
        # via proc.wait(timeout=...).  If the subprocess hangs silently (no
        # output), the reader blocks on the pipe but the main thread still kills
        # the process on deadline, which unblocks the reader.
        stdout_lines: list[str] = []

        def _drain():
            for line in proc.stdout:
                log_file.write(line)
                log_file.flush()
                stdout_lines.append(line)

        reader = threading.Thread(target=_drain, daemon=True)
        reader.start()

        timed_out = False
        # Tasks run concurrently in main.py, so timeout is per-task (not cumulative).
        # Add 5 min margin for env init / teardown overhead.
        batch_timeout = _TASK_TIMEOUT_PER_TASK + 300
        try:
            proc.wait(timeout=batch_timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            timed_out = True
            log_file.write(f"\nTIMEOUT: killed after {batch_timeout}s\n")
            log_file.flush()

        reader.join(timeout=5)

    stdout = "".join(stdout_lines)
    successes, total = _parse_task_result(stdout)

    return {
        "exit_code": proc.returncode,
        "stdout": stdout,
        "successes": successes,
        "total": total,
        "timed_out": timed_out,
    }


def _parse_task_result(stdout: str) -> tuple[int, int]:
    """Parse successes and total episodes from main.py output.

    Looks for "Total success rate: XX.X% (S/T)" pattern.
    Falls back to counting "Success: True/False" lines.
    """
    # Try structured output first
    m = re.search(r"Total success rate:.*?(\d+)/(\d+)", stdout)
    if m:
        return int(m.group(1)), int(m.group(2))

    # Fallback: count Success lines
    successes = len(re.findall(r"Success:\s*True", stdout, re.IGNORECASE))
    failures = len(re.findall(r"Success:\s*False", stdout, re.IGNORECASE))
    total = successes + failures
    return successes, total


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------


def _load_state(state_path: Path) -> list[RunState]:
    if state_path.exists():
        data = json.loads(state_path.read_text())
        states = []
        for d in data:
            filtered = {k: v for k, v in d.items() if k in RunState.__dataclass_fields__}
            states.append(RunState(**filtered))
        return states
    return []


def _save_state(state_path: Path, states: list[RunState]) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps([asdict(s) for s in states], indent=2))


def _parse_runs_filter(runs_str: str | None, total: int) -> set[int]:
    """Parse runs filter like '1-8' or '1,3,5' into 0-based index set."""
    if runs_str is None:
        return set(range(total))
    indices = set()
    for part in runs_str.split(","):
        if "-" in part:
            lo, hi = part.split("-", 1)
            for i in range(int(lo), int(hi) + 1):
                indices.add(i - 1)
        else:
            indices.add(int(part) - 1)
    return indices


def _init_task_progress(state: RunState, task_id_list: list[int]) -> None:
    """Initialize task_progress if empty (first run or migrated state)."""
    if not state.task_progress:
        state.task_progress = {str(i): "pending" for i in task_id_list}
        state.task_results = {}


def _remaining_tasks(state: RunState) -> list[int]:
    """Return task IDs that haven't completed yet."""
    return [int(tid) for tid, status in state.task_progress.items() if status != "done"]


def _compute_aggregate_success_rate(state: RunState) -> Optional[float]:
    """Compute overall success rate from completed tasks only."""
    total_s, total_t = 0, 0
    for tid, status in state.task_progress.items():
        if status == "done" and tid in state.task_results:
            total_s += state.task_results[tid][0]
            total_t += state.task_results[tid][1]
    if total_t == 0:
        return None
    return total_s / total_t


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def main():
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Run cache experiments")
    parser.add_argument("--yaml-dir", required=True, help="Directory with experiment YAMLs")
    parser.add_argument("--episodes-per-run", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--task-suite", default="libero_spatial")
    parser.add_argument("--runs", default=None, help="Run filter, e.g. '1-8' or '1,3,5'")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint (per-task granularity)")
    parser.add_argument("--state-path", default=None, help="State file path (default: yaml-dir/experiment_state.json)")
    parser.add_argument("--log-dir", default=None, help="Directory for run log files (default: alongside YAMLs)")
    parser.add_argument("--seed", type=int, default=7, help="Random seed passed to main.py (default: 7)")
    parser.add_argument("--task-ids", default=None, help="Only run these task IDs, e.g. '0' or '0,3,5' (default: all tasks in suite)")
    parser.add_argument("--conda-env", default=None, help="Use conda environment instead of uv to run main.py (e.g. 'libero')")
    args = parser.parse_args()

    num_tasks = _SUITE_NUM_TASKS.get(args.task_suite)
    if num_tasks is None:
        print(f"Unknown task suite: {args.task_suite}. Known: {list(_SUITE_NUM_TASKS)}")
        return

    # Determine which task IDs to run.
    if args.task_ids is not None:
        task_id_list = [int(x.strip()) for x in args.task_ids.split(",")]
        for tid in task_id_list:
            if tid < 0 or tid >= num_tasks:
                print(f"Invalid task ID {tid} for suite {args.task_suite} (valid: 0-{num_tasks-1})")
                return
    else:
        task_id_list = list(range(num_tasks))

    yaml_dir = Path(args.yaml_dir)
    yaml_files = sorted(yaml_dir.glob("*.yaml"))
    if not yaml_files:
        print(f"No YAML files found in {yaml_dir}")
        return

    state_path = Path(args.state_path) if args.state_path else yaml_dir / "experiment_state.json"

    if args.resume:
        states = _load_state(state_path)
        # Validate that resumed states are compatible with current CLI args.
        # Incomplete runs (running/failed/pending) will have new tasks mixed in,
        # so their saved parameters MUST match the current CLI.  Completed runs
        # are read-only and won't receive new task results, so they are exempt.
        for s in states:
            if s.status == "done":
                continue
            # Reject legacy state files that lack task_suite/episodes_per_task.
            # These were written by an older version and cannot be safely resumed
            # because we don't know what parameters produced the existing results.
            if not s.task_suite:
                print(
                    f"ERROR: run {s.run_id} has no task_suite in saved state "
                    f"(written by older version).\n"
                    f"Cannot safely resume — existing results may come from a "
                    f"different task suite.\n"
                    f"Fix: delete {state_path} and start fresh, or manually add "
                    f"\"task_suite\": \"{args.task_suite}\" to each entry after verification."
                )
                return
            if not s.episodes_per_task:
                print(
                    f"ERROR: run {s.run_id} has no episodes_per_task in saved state "
                    f"(written by older version).\n"
                    f"Cannot safely resume — existing results may use a different trial count.\n"
                    f"Fix: delete {state_path} and start fresh, or manually add "
                    f"\"episodes_per_task\": {args.episodes_per_run} to each entry after verification."
                )
                return
            if s.episodes_per_task != args.episodes_per_run:
                print(
                    f"ERROR: --resume conflict for run {s.run_id}: "
                    f"saved episodes_per_task={s.episodes_per_task}, "
                    f"current --episodes-per-run={args.episodes_per_run}.\n"
                    f"Mixing different trial counts would corrupt success_rate. "
                    f"Use the same --episodes-per-run or start fresh without --resume."
                )
                return
            if s.task_suite != args.task_suite:
                print(
                    f"ERROR: --resume conflict for run {s.run_id}: "
                    f"saved task_suite={s.task_suite!r}, "
                    f"current --task-suite={args.task_suite!r}.\n"
                    f"Mixing different task suites would corrupt results. "
                    f"Use the same --task-suite or start fresh without --resume."
                )
                return
        existing = {s.yaml_path for s in states}
        for yf in yaml_files:
            if str(yf) not in existing:
                states.append(RunState(
                    yaml_path=str(yf), run_id=yf.stem,
                    episodes_per_task=args.episodes_per_run,
                    task_suite=args.task_suite,
                ))
    else:
        states = [
            RunState(
                yaml_path=str(yf), run_id=yf.stem,
                episodes_per_task=args.episodes_per_run,
                task_suite=args.task_suite,
            )
            for yf in yaml_files
        ]

    run_filter = _parse_runs_filter(args.runs, len(states))
    runs_to_do = [s for i, s in enumerate(states) if i in run_filter]
    total_runs = len(runs_to_do)

    # --- Progress tracking ---
    total_episodes_all = 0  # across all runs
    total_successes_all = 0
    completed = 0
    failed = 0

    # Overall progress bar: one tick per completed run.
    pbar = tqdm.tqdm(total=total_runs, desc="Experiment", unit="run",
                     bar_format="{desc} |{bar}| {n}/{total} runs [{elapsed}<{remaining}] {postfix}")
    pbar.set_postfix_str("sr=N/A")

    for idx, state in enumerate(states):
        if idx not in run_filter:
            continue

        _init_task_progress(state, task_id_list)

        if args.resume and state.status == "done":
            # Count previously completed run's results into totals.
            for tid, counts in state.task_results.items():
                if state.task_progress.get(tid) == "done":
                    total_successes_all += counts[0]
                    total_episodes_all += counts[1]
            completed += 1
            pbar.update(1)
            agg_sr = f"{total_successes_all/total_episodes_all:.1%}" if total_episodes_all else "N/A"
            pbar.set_postfix_str(f"agg={agg_sr} ({total_successes_all}/{total_episodes_all}), done={completed}, fail={failed}")
            continue

        remaining = _remaining_tasks(state)
        if args.resume and not remaining:
            state.status = "done"
            state.success_rate = _compute_aggregate_success_rate(state)
            _save_state(state_path, states)
            completed += 1
            pbar.update(1)
            continue

        if args.resume and remaining:
            done_count = len(task_id_list) - len(remaining)
            tqdm.tqdm.write(f"Resuming {state.run_id}: {done_count}/{len(task_id_list)} tasks done")

        state.status = "running"
        if state.start_time is None:
            state.start_time = time.strftime("%Y-%m-%d %H:%M:%S")
        _save_state(state_path, states)

        # Determine log file path
        if args.log_dir:
            log_path = Path(args.log_dir) / f"{state.run_id}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            log_path = Path(state.yaml_path).with_suffix(".log")
        state.log_path = str(log_path)

        # Switch server to this run's YAML (once per run, not per task).
        pbar.set_description(f"Run {idx+1}/{len(states)}: {state.run_id}")
        try:
            server_url = f"ws://{args.host}:{args.port}"
            _send_cache_config(server_url, state.yaml_path)
        except Exception as e:
            state.status = "failed"
            failed += 1
            tqdm.tqdm.write(f"[FAIL] {state.run_id}: config switch failed: {e}")
            state.end_time = time.strftime("%Y-%m-%d %H:%M:%S")
            _save_state(state_path, states)
            pbar.update(1)
            agg_sr = f"{total_successes_all/total_episodes_all:.1%}" if total_episodes_all else "N/A"
            pbar.set_postfix_str(f"agg={agg_sr} ({total_successes_all}/{total_episodes_all}), done={completed}, fail={failed}")
            continue

        # Execute remaining tasks in batches of num_workers.
        # Initialize with previously completed task results (for resume).
        run_failed = False
        run_successes = 0
        run_total = 0
        for tid, status in state.task_progress.items():
            if status == "done" and tid in state.task_results:
                run_successes += state.task_results[tid][0]
                run_total += state.task_results[tid][1]
        batch_size = max(args.num_workers, 1)
        num_batches = (len(remaining) + batch_size - 1) // batch_size
        for batch_idx, batch_start in enumerate(range(0, len(remaining), batch_size)):
            batch = remaining[batch_start:batch_start + batch_size]
            pbar.set_description(
                f"Run {idx+1}/{len(states)}: {state.run_id} batch {batch_idx+1}/{num_batches}"
            )
            try:
                result = _execute_tasks(
                    task_ids=batch,
                    episodes_per_task=args.episodes_per_run,
                    num_workers=min(args.num_workers, len(batch)),
                    host=args.host,
                    port=args.port,
                    task_suite=args.task_suite,
                    log_path=log_path,
                    seed=args.seed,
                    conda_env=args.conda_env,
                )

                if result["exit_code"] == 0:
                    n = len(batch)
                    per_task_s = result["successes"] // n
                    per_task_t = result["total"] // n
                    remainder_s = result["successes"] - per_task_s * n
                    remainder_t = result["total"] - per_task_t * n
                    for i, task_id in enumerate(batch):
                        state.task_progress[str(task_id)] = "done"
                        s = per_task_s + (1 if i < remainder_s else 0)
                        t = per_task_t + (1 if i < remainder_t else 0)
                        state.task_results[str(task_id)] = [s, t]
                    run_successes += result["successes"]
                    run_total += result["total"]
                    total_successes_all += result["successes"]
                    total_episodes_all += result["total"]
                    run_sr = f"{run_successes/run_total:.1%}" if run_total else "N/A"
                    agg_sr = f"{total_successes_all/total_episodes_all:.1%}" if total_episodes_all else "N/A"
                    pbar.set_postfix_str(
                        f"run_sr={run_sr} ({run_successes}/{run_total}), "
                        f"agg={agg_sr}, done={completed}, fail={failed}"
                    )
                    tqdm.tqdm.write(
                        f"  [{state.run_id}] batch {batch_idx+1}/{num_batches}: "
                        f"{result['successes']}/{result['total']} successes"
                    )
                else:
                    for task_id in batch:
                        state.task_progress[str(task_id)] = "failed"
                    run_failed = True
                    tail = "\n".join(result["stdout"].splitlines()[-5:])
                    tqdm.tqdm.write(f"  [{state.run_id}] batch {batch_idx+1} FAILED (exit {result['exit_code']}): {tail}")
            except Exception as e:
                for task_id in batch:
                    state.task_progress[str(task_id)] = "failed"
                run_failed = True
                tqdm.tqdm.write(f"  [{state.run_id}] batch {batch_idx+1} exception: {e}")

            # Persist after every batch.
            state.success_rate = _compute_aggregate_success_rate(state)
            _save_state(state_path, states)

        # Determine final run status.
        all_done = all(v == "done" for v in state.task_progress.values())
        if all_done:
            state.status = "done"
            completed += 1
            run_sr = f"{run_successes/run_total:.1%}" if run_total else "N/A"
            tqdm.tqdm.write(f"[DONE] {state.run_id}: {run_sr} ({run_successes}/{run_total})")
        elif run_failed:
            state.status = "failed"
            failed += 1
            tqdm.tqdm.write(f"[FAIL] {state.run_id}")
        else:
            state.status = "done"
            completed += 1

        state.end_time = time.strftime("%Y-%m-%d %H:%M:%S")
        state.success_rate = _compute_aggregate_success_rate(state)
        _save_state(state_path, states)
        pbar.update(1)
        agg_sr = f"{total_successes_all/total_episodes_all:.1%}" if total_episodes_all else "N/A"
        pbar.set_postfix_str(f"agg={agg_sr} ({total_successes_all}/{total_episodes_all}), done={completed}, fail={failed}")

    pbar.close()
    print(f"\nDone: {completed} completed, {failed} failed, {total_runs - completed - failed} skipped")
    print(f"State saved to {state_path}")


if __name__ == "__main__":
    main()
