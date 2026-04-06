"""Experiment runner: sequentially execute cache experiment runs.

Per-task granularity: each run is split into individual task executions.
Progress is persisted after every task, so interrupted runs resume from
the last completed task — not from scratch.

Usage:
    # Run all Phase 1 (4 workers per run)
    uv run exp/run_cache_experiments.py \
        --yaml-dir configs/cache_runs/phase1 \
        --episodes-per-run 10 \
        --num-workers 4 \
        --host localhost --port 8000

    # Resume from checkpoint (skips done runs, resumes interrupted runs per-task)
    uv run exp/run_cache_experiments.py \
        --yaml-dir configs/cache_runs/phase1 \
        --episodes-per-run 10 \
        --num-workers 4 \
        --host localhost --port 8000 \
        --resume
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

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
            logger.info("Server switched to bundle v%s: %s", resp.get("version"), yaml_path)

    asyncio.run(_send())


# ---------------------------------------------------------------------------
# Single task execution
# ---------------------------------------------------------------------------


_TASK_TIMEOUT = 3600  # 1 hour per task


def _execute_task(
    task_id: int,
    episodes_per_task: int,
    num_workers: int,
    host: str,
    port: int,
    task_suite: str,
    log_path: Path,
    seed: int = 7,
    conda_env: str | None = None,
) -> dict:
    """Execute a single task within a run.

    Calls main.py with --task-ids to run only this task.
    Appends output to the run's log file.
    Kills subprocess after _TASK_TIMEOUT seconds (even if stdout is blocked).
    """
    main_args = [
        "examples/libero/main.py",
        "--host", host,
        "--port", str(port),
        "--task_suite_name", task_suite,
        "--num_trials_per_task", str(episodes_per_task),
        "--num_workers", str(num_workers),
        "--task-ids", str(task_id),
        "--seed", str(seed),
    ]
    if conda_env:
        cmd = ["conda", "run", "--no-capture-output", "-n", conda_env, "python", *main_args]
    else:
        cmd = ["uv", "run", *main_args]

    with open(log_path, "a") as log_file:
        log_file.write(f"\n{'='*60}\n")
        log_file.write(f"TASK {task_id} started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"CMD: {' '.join(cmd)}\n")
        log_file.write(f"{'='*60}\n")
        log_file.flush()

        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
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
        try:
            proc.wait(timeout=_TASK_TIMEOUT)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            timed_out = True
            log_file.write(f"\nTIMEOUT: killed after {_TASK_TIMEOUT}s\n")
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
    """Parse successes and total episodes from single-task main.py output.

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
    """Compute overall success rate from per-task results."""
    total_s, total_t = 0, 0
    for counts in state.task_results.values():
        total_s += counts[0]
        total_t += counts[1]
    if total_t == 0:
        return None
    return total_s / total_t


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

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

    completed = 0
    failed = 0
    for idx, state in enumerate(states):
        if idx not in run_filter:
            continue

        _init_task_progress(state, task_id_list)

        if args.resume and state.status == "done":
            logger.info("Skipping completed run %s", state.run_id)
            completed += 1
            continue

        remaining = _remaining_tasks(state)
        if args.resume and not remaining:
            state.status = "done"
            state.success_rate = _compute_aggregate_success_rate(state)
            _save_state(state_path, states)
            logger.info("Run %s already fully completed (all tasks done)", state.run_id)
            completed += 1
            continue

        if args.resume and remaining:
            done_count = num_tasks - len(remaining)
            logger.info("Resuming run %s: %d/%d tasks done, %d remaining",
                        state.run_id, done_count, num_tasks, len(remaining))

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
        logger.info("=== Run %d/%d: %s (%d tasks remaining) ===",
                     idx + 1, len(states), state.run_id, len(remaining))
        try:
            server_url = f"ws://{args.host}:{args.port}"
            _send_cache_config(server_url, state.yaml_path)
        except Exception as e:
            state.status = "failed"
            failed += 1
            logger.error("Run %s: config switch failed: %s", state.run_id, e)
            state.end_time = time.strftime("%Y-%m-%d %H:%M:%S")
            _save_state(state_path, states)
            continue

        # Execute remaining tasks one by one.
        run_failed = False
        for task_id in remaining:
            logger.info("  Task %d/%d for run %s", task_id, num_tasks - 1, state.run_id)
            try:
                result = _execute_task(
                    task_id=task_id,
                    episodes_per_task=args.episodes_per_run,
                    num_workers=args.num_workers,
                    host=args.host,
                    port=args.port,
                    task_suite=args.task_suite,
                    log_path=log_path,
                    seed=args.seed,
                    conda_env=args.conda_env,
                )
                state.task_results[str(task_id)] = [result["successes"], result["total"]]

                if result["exit_code"] == 0:
                    state.task_progress[str(task_id)] = "done"
                    logger.info("  Task %d done: %d/%d successes",
                                task_id, result["successes"], result["total"])
                else:
                    state.task_progress[str(task_id)] = "failed"
                    run_failed = True
                    logger.error("  Task %d failed (exit %d)", task_id, result["exit_code"])
            except Exception as e:
                state.task_progress[str(task_id)] = "failed"
                run_failed = True
                logger.error("  Task %d exception: %s", task_id, e)

            # Persist after every task — this is the core of per-task resume.
            state.success_rate = _compute_aggregate_success_rate(state)
            _save_state(state_path, states)

        # Determine final run status.
        all_done = all(v == "done" for v in state.task_progress.values())
        if all_done:
            state.status = "done"
            completed += 1
        elif run_failed:
            state.status = "failed"
            failed += 1
        else:
            state.status = "done"
            completed += 1

        state.end_time = time.strftime("%Y-%m-%d %H:%M:%S")
        state.success_rate = _compute_aggregate_success_rate(state)
        _save_state(state_path, states)

    print(f"\nDone: {completed} completed, {failed} failed, {len(states) - completed - failed} skipped")
    print(f"State saved to {state_path}")


if __name__ == "__main__":
    main()
