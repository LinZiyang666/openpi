"""Experiment runner: sequentially execute cache experiment runs.

Tasks within each run are batched by --num-workers and executed concurrently
via main.py.  Progress is persisted after every batch, so interrupted runs
resume from the last completed batch — not from scratch.

Usage:
    uv run exp/common/run_cache_experiments.py \
        --yaml-dir exp/phase1 \
        --episodes-per-run 5 \
        --num-workers 5 \
        --host 155.98.36.13 --port 9000 \
        --task-suite libero_spatial \
        --seed 42 --conda-env libero_sim

    # Resume from checkpoint
    uv run exp/common/run_cache_experiments.py \
        --yaml-dir exp/phase1 \
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
import shutil
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import tqdm

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

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
    episode_results_path: Path | None = None,
    cuda_visible_devices: str = "0",
) -> dict:
    """Execute a batch of tasks concurrently via main.py.

    Passes multiple --task-ids so main.py distributes them across workers.
    Appends output to the run's log file.

    If ``episode_results_path`` is given, forwards ``--save-episode-results
    --episode-results-path`` to ``main.py`` so per-episode success rows are
    written for later aggregation into ``cache_eval_results.json`` (plan §9.0
    改动 2 / §19.B5).
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
        "--cuda-visible-devices", cuda_visible_devices,
    ]
    if episode_results_path is not None:
        main_args += [
            "--save-episode-results",
            "--episode-results-path", str(episode_results_path),
        ]
    from exp.common._subprocess import build_subprocess_cmd

    cmd, env = build_subprocess_cmd(main_args, conda_env=conda_env)

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


@dataclass
class _RunStats:
    """Mutable aggregate counters shared across runs in one CLI invocation."""

    total_episodes_all: int = 0
    total_successes_all: int = 0
    completed: int = 0
    failed: int = 0

    def agg_postfix_str(self) -> str:
        if self.total_episodes_all:
            agg_sr = f"{self.total_successes_all/self.total_episodes_all:.1%}"
        else:
            agg_sr = "N/A"
        return (
            f"agg={agg_sr} "
            f"({self.total_successes_all}/{self.total_episodes_all}), "
            f"done={self.completed}, fail={self.failed}"
        )


def _build_arg_parser() -> argparse.ArgumentParser:
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
    parser.add_argument("--cuda", default="0", help="CUDA_VISIBLE_DEVICES value for all main.py workers (default: '0')")
    return parser


def _resolve_task_ids(args, num_tasks: int) -> Optional[list[int]]:
    """Return the task-id list requested by the CLI, or ``None`` on error
    (after printing a human-readable diagnostic)."""
    if args.task_ids is None:
        return list(range(num_tasks))
    task_id_list = [int(x.strip()) for x in args.task_ids.split(",")]
    for tid in task_id_list:
        if tid < 0 or tid >= num_tasks:
            print(
                f"Invalid task ID {tid} for suite {args.task_suite} "
                f"(valid: 0-{num_tasks-1})"
            )
            return None
    return task_id_list


def _validate_resumed_state(
    s: "RunState", args, state_path: Path
) -> bool:
    """Return True if ``s`` is safe to resume under ``args``. Prints an
    error and returns False otherwise. ``done`` runs are exempt — they
    won't receive new results, so their saved parameters can't influence
    anything."""
    if s.status == "done":
        return True
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
        return False
    if not s.episodes_per_task:
        print(
            f"ERROR: run {s.run_id} has no episodes_per_task in saved state "
            f"(written by older version).\n"
            f"Cannot safely resume — existing results may use a different trial count.\n"
            f"Fix: delete {state_path} and start fresh, or manually add "
            f"\"episodes_per_task\": {args.episodes_per_run} to each entry after verification."
        )
        return False
    if s.episodes_per_task != args.episodes_per_run:
        print(
            f"ERROR: --resume conflict for run {s.run_id}: "
            f"saved episodes_per_task={s.episodes_per_task}, "
            f"current --episodes-per-run={args.episodes_per_run}.\n"
            f"Mixing different trial counts would corrupt success_rate. "
            f"Use the same --episodes-per-run or start fresh without --resume."
        )
        return False
    if s.task_suite != args.task_suite:
        print(
            f"ERROR: --resume conflict for run {s.run_id}: "
            f"saved task_suite={s.task_suite!r}, "
            f"current --task-suite={args.task_suite!r}.\n"
            f"Mixing different task suites would corrupt results. "
            f"Use the same --task-suite or start fresh without --resume."
        )
        return False
    return True


def _load_or_init_states(
    args, yaml_files: list[Path], state_path: Path
) -> Optional[list["RunState"]]:
    """Build the RunState list either from a resumed state file or fresh
    from ``yaml_files``. Returns ``None`` on resume-validation failure
    after printing the diagnostic."""
    if not args.resume:
        return [
            RunState(
                yaml_path=str(yf), run_id=yf.stem,
                episodes_per_task=args.episodes_per_run,
                task_suite=args.task_suite,
            )
            for yf in yaml_files
        ]

    states = _load_state(state_path)
    for s in states:
        if not _validate_resumed_state(s, args, state_path):
            return None
    existing = {s.yaml_path for s in states}
    for yf in yaml_files:
        if str(yf) not in existing:
            states.append(RunState(
                yaml_path=str(yf), run_id=yf.stem,
                episodes_per_task=args.episodes_per_run,
                task_suite=args.task_suite,
            ))
    return states


def _resolve_log_path(args, state: "RunState") -> Path:
    if args.log_dir:
        log_path = Path(args.log_dir) / f"{state.run_id}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        log_path = Path(state.yaml_path).with_suffix(".log")
    return log_path


def _seed_run_totals_from_done_tasks(state: "RunState") -> tuple[int, int]:
    """Sum up success/total counts for tasks already marked ``done`` — the
    starting point for the per-run aggregate when resuming."""
    run_successes = 0
    run_total = 0
    for tid, status in state.task_progress.items():
        if status == "done" and tid in state.task_results:
            run_successes += state.task_results[tid][0]
            run_total += state.task_results[tid][1]
    return run_successes, run_total


def _execute_run_batches(
    *,
    state: "RunState",
    remaining: list[int],
    args,
    log_path: Path,
    state_path: Path,
    states: list["RunState"],
    idx: int,
    num_states: int,
    pbar: tqdm.tqdm,
    stats: _RunStats,
) -> tuple[int, int, bool]:
    """Run ``remaining`` task IDs in batches of ``--num-workers``. Returns
    ``(run_successes, run_total, run_failed)``. Mutates ``state`` and
    ``stats`` in-place and saves after every batch."""
    run_successes, run_total = _seed_run_totals_from_done_tasks(state)
    run_failed = False

    batch_size = max(args.num_workers, 1)
    num_batches = (len(remaining) + batch_size - 1) // batch_size
    for batch_idx, batch_start in enumerate(range(0, len(remaining), batch_size)):
        batch = remaining[batch_start:batch_start + batch_size]
        pbar.set_description(
            f"Run {idx+1}/{num_states}: {state.run_id} batch {batch_idx+1}/{num_batches}"
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
                cuda_visible_devices=args.cuda,
                episode_results_path=_episode_results_path_for(
                    log_path, batch_idx, num_batches
                ),
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
                stats.total_successes_all += result["successes"]
                stats.total_episodes_all += result["total"]
                run_sr = f"{run_successes/run_total:.1%}" if run_total else "N/A"
                if stats.total_episodes_all:
                    agg_sr = f"{stats.total_successes_all/stats.total_episodes_all:.1%}"
                else:
                    agg_sr = "N/A"
                pbar.set_postfix_str(
                    f"run_sr={run_sr} ({run_successes}/{run_total}), "
                    f"agg={agg_sr}, done={stats.completed}, fail={stats.failed}"
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

    return run_successes, run_total, run_failed


def _absorb_resumed_done_totals(state: "RunState", stats: _RunStats) -> None:
    """A ``done`` run on resume still contributes to the aggregate totals,
    even though we don't re-execute it."""
    for tid, counts in state.task_results.items():
        if state.task_progress.get(tid) == "done":
            stats.total_successes_all += counts[0]
            stats.total_episodes_all += counts[1]


def _switch_server_config(
    *, state: "RunState", args, states: list["RunState"],
    state_path: Path, pbar: tqdm.tqdm, stats: _RunStats,
) -> bool:
    """Tell the server to load ``state.yaml_path``. On failure, mark the
    run failed, save, advance the pbar, and return False."""
    try:
        server_url = f"ws://{args.host}:{args.port}"
        _send_cache_config(server_url, state.yaml_path)
        return True
    except Exception as e:
        state.status = "failed"
        stats.failed += 1
        tqdm.tqdm.write(f"[FAIL] {state.run_id}: config switch failed: {e}")
        state.end_time = time.strftime("%Y-%m-%d %H:%M:%S")
        _save_state(state_path, states)
        pbar.update(1)
        pbar.set_postfix_str(stats.agg_postfix_str())
        return False


def _execute_one_run(
    *,
    idx: int,
    state: "RunState",
    states: list["RunState"],
    args,
    task_id_list: list[int],
    state_path: Path,
    pbar: tqdm.tqdm,
    stats: _RunStats,
) -> None:
    """Execute one ``RunState``. Handles skip-if-done (resume), skip-if-
    already-complete, server config switch, batch loop, and final status
    bookkeeping. Mutates ``state`` + ``stats`` in place."""
    _init_task_progress(state, task_id_list)

    if args.resume and state.status == "done":
        _absorb_resumed_done_totals(state, stats)
        stats.completed += 1
        pbar.update(1)
        pbar.set_postfix_str(stats.agg_postfix_str())
        return

    remaining = _remaining_tasks(state)
    if args.resume and not remaining:
        state.status = "done"
        state.success_rate = _compute_aggregate_success_rate(state)
        _save_state(state_path, states)
        stats.completed += 1
        pbar.update(1)
        return

    if args.resume and remaining:
        done_count = len(task_id_list) - len(remaining)
        tqdm.tqdm.write(f"Resuming {state.run_id}: {done_count}/{len(task_id_list)} tasks done")

    state.status = "running"
    if state.start_time is None:
        state.start_time = time.strftime("%Y-%m-%d %H:%M:%S")
    _save_state(state_path, states)

    log_path = _resolve_log_path(args, state)
    state.log_path = str(log_path)

    pbar.set_description(f"Run {idx+1}/{len(states)}: {state.run_id}")
    if not _switch_server_config(
        state=state, args=args, states=states,
        state_path=state_path, pbar=pbar, stats=stats,
    ):
        return

    run_successes, run_total, run_failed = _execute_run_batches(
        state=state,
        remaining=remaining,
        args=args,
        log_path=log_path,
        state_path=state_path,
        states=states,
        idx=idx,
        num_states=len(states),
        pbar=pbar,
        stats=stats,
    )

    # Determine final run status.
    all_done = all(v == "done" for v in state.task_progress.values())
    if all_done:
        state.status = "done"
        stats.completed += 1
        run_sr = f"{run_successes/run_total:.1%}" if run_total else "N/A"
        tqdm.tqdm.write(f"[DONE] {state.run_id}: {run_sr} ({run_successes}/{run_total})")
    elif run_failed:
        state.status = "failed"
        stats.failed += 1
        tqdm.tqdm.write(f"[FAIL] {state.run_id}")
    else:
        state.status = "done"
        stats.completed += 1

    state.end_time = time.strftime("%Y-%m-%d %H:%M:%S")
    state.success_rate = _compute_aggregate_success_rate(state)
    _save_state(state_path, states)
    pbar.update(1)
    pbar.set_postfix_str(stats.agg_postfix_str())


def main():
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s: %(message)s")
    args = _build_arg_parser().parse_args()

    num_tasks = _SUITE_NUM_TASKS.get(args.task_suite)
    if num_tasks is None:
        print(f"Unknown task suite: {args.task_suite}. Known: {list(_SUITE_NUM_TASKS)}")
        return

    task_id_list = _resolve_task_ids(args, num_tasks)
    if task_id_list is None:
        return

    yaml_dir = Path(args.yaml_dir)
    yaml_files = sorted(yaml_dir.glob("*.yaml"))
    if not yaml_files:
        print(f"No YAML files found in {yaml_dir}")
        return

    state_path = Path(args.state_path) if args.state_path else yaml_dir / "experiment_state.json"

    states = _load_or_init_states(args, yaml_files, state_path)
    if states is None:
        return

    run_filter = _parse_runs_filter(args.runs, len(states))
    total_runs = sum(1 for i in range(len(states)) if i in run_filter)

    stats = _RunStats()
    pbar = tqdm.tqdm(total=total_runs, desc="Experiment", unit="run",
                     bar_format="{desc} |{bar}| {n}/{total} runs [{elapsed}<{remaining}] {postfix}")
    pbar.set_postfix_str("sr=N/A")

    for idx, state in enumerate(states):
        if idx not in run_filter:
            continue
        _execute_one_run(
            idx=idx,
            state=state,
            states=states,
            args=args,
            task_id_list=task_id_list,
            state_path=state_path,
            pbar=pbar,
            stats=stats,
        )

    pbar.close()
    skipped = total_runs - stats.completed - stats.failed
    print(f"\nDone: {stats.completed} completed, {stats.failed} failed, {skipped} skipped")
    print(f"State saved to {state_path}")

    if not args.resume:
        _retry_failed_runs(
            states=states,
            state_path=state_path,
            yaml_dir=yaml_dir,
            args=args,
            task_id_list=task_id_list,
        )

    # --- Post-run aggregate (plan §9.0 改动 2 + §19.B5) ---
    # Walk every *.episode_results.json produced by main.py (both first-pass
    # and retry attempts) and dedup by (config_id, task_id, init_state_idx,
    # seed), keeping the latest attempt. Output goes next to the YAMLs so
    # ``exp/trajectory_deviation/dump_step1a_failed_inits.py`` can consume it.
    aggregate_path = yaml_dir / "cache_eval_results.json"
    _aggregate_episode_results(yaml_dir, aggregate_path)
    print(f"Aggregated episode results → {aggregate_path}")


def _episode_results_path_for(
    log_path: Path, batch_idx: int, num_batches: int
) -> Path:
    """Pick a per-batch ``.episode_results.json`` path alongside ``log_path``.

    Two subtleties matter for the §19.B5 aggregate:

    - ``log_path`` may end in ``.log`` (first pass) or ``.retryN.log``
      (retry pass). ``Path.with_suffix`` strips only the trailing ``.log``
      so retry pass paths still carry ``.retryN`` as a stem component —
      exactly what the aggregator's ``count('/retry/')`` attempt detection
      wants.
    - When a run is split across multiple batches, each batch invocation of
      ``main.py`` overwrites ``--episode-results-path``; appending
      ``.batchNofM`` disambiguates so nothing is lost. Single-batch runs
      keep the plain ``.episode_results.json`` name for readability.
    """
    base = log_path.with_suffix(".episode_results.json")
    if num_batches <= 1:
        return base
    # Insert ``.batchK`` before the final suffix set.
    stem = base.name[: -len(".episode_results.json")]
    return base.with_name(f"{stem}.batch{batch_idx}of{num_batches}.episode_results.json")


def _infer_config_id_from_source(source: str) -> str:
    """Derive a stable config_id from the per-batch results file path.

    Removes the ``.episode_results.json`` tail, any ``.batchKofN`` batch
    marker, any ``.retryN`` retry marker, and any leading ``retry/`` prefix.
    The remainder is the YAML stem (``run_id``) — which is how every other
    part of the pipeline identifies a cache config.
    """
    name = Path(source).name
    if name.endswith(".episode_results.json"):
        name = name[: -len(".episode_results.json")]
    # Strip batch / retry markers in a loop — they can appear in either
    # order (``.batch0of2.retry1`` or ``.retry1.batch0of2``) depending on
    # whether a retry also needed to be split across batches.
    while True:
        new_name = re.sub(r"\.batch\d+of\d+$", "", name)
        new_name = re.sub(r"\.retry\d+$", "", new_name)
        if new_name == name:
            return name
        name = new_name


def _aggregate_episode_results(run_root: Path, out_path: Path) -> None:
    """Merge all per-batch ``*.episode_results.json`` into a single file.

    Dedup key (plan §19.B5): ``(config_id, task_id, init_state_idx, seed)``.
    Later attempts (higher ``/retry/`` depth in the path, then later batch)
    override earlier attempts so a retry-recovered episode correctly reads
    as a success.
    """
    if not run_root.exists():
        return
    rows: list[dict] = []
    for p in sorted(run_root.rglob("*.episode_results.json")):
        attempt = str(p).count("/retry/")
        source = p.relative_to(run_root).as_posix()
        try:
            payload = json.loads(p.read_text())
        except json.JSONDecodeError as e:
            logger.warning("Skipping malformed %s: %s", p, e)
            continue
        config_id = _infer_config_id_from_source(source)
        for r in payload:
            enriched = dict(r)
            enriched.setdefault("config_id", config_id)
            enriched["attempt"] = attempt
            enriched["source_path"] = source
            rows.append(enriched)

    def _key(r: dict) -> tuple:
        return (
            str(r.get("config_id", "")),
            int(r["task_id"]),
            int(r["init_state_idx"]),
            int(r.get("seed", -1)),
        )

    # Stable sort by (key, attempt) then overwrite: last write wins.
    rows.sort(key=lambda r: (_key(r), r["attempt"]))
    deduped: dict[tuple, dict] = {}
    for r in rows:
        deduped[_key(r)] = r
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(list(deduped.values()), indent=2))


def _retry_failed_runs(
    states: list[RunState],
    state_path: Path,
    yaml_dir: Path,
    args,
    task_id_list: list[int],
    max_retries: int = 2,
) -> None:
    """Retry failed runs: copy YAMLs to retry/, run up to max_retries times.

    On success: update original state JSON, delete YAML from retry/.
    On failure after max_retries: leave YAML in retry/ for manual inspection.
    """
    failed_states = [s for s in states if s.status == "failed"]
    if not failed_states:
        return

    retry_dir = yaml_dir / "retry"
    retry_dir.mkdir(parents=True, exist_ok=True)

    # Copy failed YAMLs to retry/
    for s in failed_states:
        src = Path(s.yaml_path)
        dst = retry_dir / src.name
        shutil.copy2(src, dst)

    print(f"\n{'='*60}")
    print(f"Retry phase: {len(failed_states)} failed runs copied to {retry_dir}")
    print(f"{'='*60}")

    for attempt in range(1, max_retries + 1):
        retry_yamls = sorted(retry_dir.glob("*.yaml"))
        if not retry_yamls:
            print(f"Retry attempt {attempt}: no remaining retries, all recovered!")
            break

        print(f"\nRetry attempt {attempt}/{max_retries}: {len(retry_yamls)} runs")
        pbar = tqdm.tqdm(
            total=len(retry_yamls), desc=f"Retry {attempt}/{max_retries}",
            unit="run", bar_format="{desc} |{bar}| {n}/{total} [{elapsed}<{remaining}]",
        )

        for yaml_path in retry_yamls:
            run_id = yaml_path.stem
            pbar.set_description(f"Retry {attempt}/{max_retries}: {run_id}")

            # Find original state
            orig_state = next((s for s in states if s.run_id == run_id), None)
            if orig_state is None:
                pbar.update(1)
                continue

            # Reset failed tasks to pending
            failed_tasks = [
                int(tid) for tid, status in orig_state.task_progress.items()
                if status == "failed"
            ]
            if not failed_tasks:
                # All tasks actually done, clean up
                orig_state.status = "done"
                yaml_path.unlink()
                _save_state(state_path, states)
                pbar.update(1)
                continue

            for tid in failed_tasks:
                orig_state.task_progress[str(tid)] = "pending"

            orig_state.status = "running"
            orig_state.start_time = time.strftime("%Y-%m-%d %H:%M:%S")
            _save_state(state_path, states)

            # Switch server config
            try:
                server_url = f"ws://{args.host}:{args.port}"
                _send_cache_config(server_url, str(yaml_path))
            except Exception as e:
                tqdm.tqdm.write(f"  [RETRY FAIL] {run_id}: config switch failed: {e}")
                orig_state.status = "failed"
                _save_state(state_path, states)
                pbar.update(1)
                continue

            # Execute failed tasks
            log_path = Path(args.log_dir) / f"{run_id}_retry{attempt}.log" if args.log_dir else yaml_path.with_suffix(f".retry{attempt}.log")
            log_path.parent.mkdir(parents=True, exist_ok=True)

            retry_failed = False
            batch_size = max(args.num_workers, 1)
            for batch_start in range(0, len(failed_tasks), batch_size):
                batch = failed_tasks[batch_start:batch_start + batch_size]
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
                        episode_results_path=_episode_results_path_for(
                            log_path,
                            batch_start // batch_size,
                            (len(failed_tasks) + batch_size - 1) // batch_size,
                        ),
                    )
                    if result["exit_code"] == 0:
                        n = len(batch)
                        per_task_s = result["successes"] // n
                        per_task_t = result["total"] // n
                        remainder_s = result["successes"] - per_task_s * n
                        remainder_t = result["total"] - per_task_t * n
                        for i, task_id in enumerate(batch):
                            orig_state.task_progress[str(task_id)] = "done"
                            s = per_task_s + (1 if i < remainder_s else 0)
                            t = per_task_t + (1 if i < remainder_t else 0)
                            orig_state.task_results[str(task_id)] = [s, t]
                    else:
                        for task_id in batch:
                            orig_state.task_progress[str(task_id)] = "failed"
                        retry_failed = True
                except Exception as e:
                    for task_id in batch:
                        orig_state.task_progress[str(task_id)] = "failed"
                    retry_failed = True
                    tqdm.tqdm.write(f"  [RETRY] {run_id} batch exception: {e}")

                orig_state.success_rate = _compute_aggregate_success_rate(orig_state)
                _save_state(state_path, states)

            # Determine retry result
            all_done = all(v == "done" for v in orig_state.task_progress.values())
            if all_done:
                orig_state.status = "done"
                orig_state.end_time = time.strftime("%Y-%m-%d %H:%M:%S")
                orig_state.success_rate = _compute_aggregate_success_rate(orig_state)
                yaml_path.unlink()  # Remove from retry/
                tqdm.tqdm.write(f"  [RETRY OK] {run_id}: recovered on attempt {attempt}")
            else:
                orig_state.status = "failed"
                orig_state.end_time = time.strftime("%Y-%m-%d %H:%M:%S")
                if attempt == max_retries:
                    tqdm.tqdm.write(f"  [RETRY GIVE UP] {run_id}: still failing after {max_retries} attempts")

            _save_state(state_path, states)
            pbar.update(1)

        pbar.close()

    # Summary
    remaining_retries = list(retry_dir.glob("*.yaml"))
    if remaining_retries:
        print(f"\nRetry complete: {len(remaining_retries)} runs still failed in {retry_dir}")
    else:
        print("\nRetry complete: all failed runs recovered!")
        # Clean up retry dir
        shutil.rmtree(retry_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
