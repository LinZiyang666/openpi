"""Step 1b GT-trajectory runner (plan §9.2 + §18.B4 + §19.B6).

This is a thin dispatcher on top of ``examples/libero/main.py``. It reads the
filter written by ``scripts/dump_step1a_failed_inits.py`` — which enumerates
every ``(task_id, orig_init_state_idx, subset_init_state_idx)`` failure from
Step 1a — and, for each such unit, spawns a subprocess ``main.py`` call with

    --task-ids <task_id>                       # single task per subprocess
    --num-trials-per-task <K_subset>           # covers the subset so the
                                               # filter can pick subset_idx
    --init-states-dir <inits_dir>              # subset .init loader
    --episode-filter <single_unit_filter.json> # only this (task, subset) pair
    --save-trajectory --save-trajectory-dir    # writes the GT HDF5
    --save-episode-results --episode-results-path
                                               # per-unit success JSON

Per plan §18.B4.3, Step 1b always uses ``inference_clip_w7_d4.yaml`` (the
"standard" AlwaysSkip bundle); Phase 2 and Spawn reuse the same GT.

Unit key format is ``f"{task_id}:{orig_init_state_idx}"`` so resume
semantics are stable against subset reordering by ``dump_step1a_failed_inits``
(the orig idx is what's unique, the subset pos is just a positional
assignment).

The user must start the inference server themselves before running this
driver — Step 1b does not manage the server process. Reference command:

    uv run scripts/serve_policy.py \\
        --config pi05_libero \\
        --cache_config configs/cache_runs/deviate_exp/inference_clip_w7_d4.yaml \\
        --collect --collect-dir data/deviation_experiment/collected
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from exp._cache_config_rpc import send_load_cache_config
from exp._run_state_base import BaseRunState, UnitState

logger = logging.getLogger(__name__)

# Default clip GT bundle (plan §18.B4.3 decision).
DEFAULT_INFERENCE_YAML = "configs/cache_runs/deviate_exp/inference_clip_w7_d4.yaml"

# Per-unit subprocess timeout. Mirrors run_cache_experiments.py's per-task
# budget — an individual GT episode is 1 task × 1 init, comfortably inside
# this window even on slow hardware.
_UNIT_TIMEOUT_SECONDS = 1800


# ---------------------------------------------------------------------------
# Unit key helpers (public so tests can lock the format)
# ---------------------------------------------------------------------------


def build_unit_key(task_id: int, orig_init_state_idx: int) -> str:
    """Canonical state-file key. Stable against subset reorderings.

    The subset position is encoded implicitly by the ``.init_map.json``
    loaded at runtime, not in the key. ``compute_global_episode_id`` uses
    the subset position but we do not — the state file must survive a
    regenerated filter that keeps the same set of failures but in a
    different subset order.
    """
    return f"{int(task_id)}:{int(orig_init_state_idx)}"


def parse_unit_key(key: str) -> Tuple[int, int]:
    task_str, orig_str = key.split(":")
    return int(task_str), int(orig_str)


# ---------------------------------------------------------------------------
# Filter-file loading
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FilterEntry:
    task_id: int
    orig_init_state_idx: int
    subset_init_state_idx: int


def load_filter_entries(path: Path) -> List[FilterEntry]:
    """Parse the §19.B6 filter JSON produced by dump_step1a_failed_inits.py."""
    data = json.loads(path.read_text())
    return [
        FilterEntry(
            task_id=int(d["task_id"]),
            orig_init_state_idx=int(d["orig_init_state_idx"]),
            subset_init_state_idx=int(d["subset_init_state_idx"]),
        )
        for d in data
    ]


def subset_size_per_task(entries: List[FilterEntry]) -> Dict[int, int]:
    """Max subset position + 1, per task — used for --num-trials-per-task
    so ``main.py``'s inner loop reaches the filter's largest subset index."""
    out: Dict[int, int] = {}
    for e in entries:
        out[e.task_id] = max(out.get(e.task_id, 0), e.subset_init_state_idx + 1)
    return out


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class Step1bRunner(BaseRunState):
    """Per-(task, orig_init) dispatcher that drives ``main.py`` subprocesses.

    Each unit runs one episode via ``main.py``. The server is kept pointed
    at ``DEFAULT_INFERENCE_YAML`` (set once at ``main()`` startup) so the
    cache bundle is fixed across units — §18.B4 decision.
    """

    def __init__(
        self,
        *,
        state_path: Path,
        inits_dir: Path,
        filter_entries: List[FilterEntry],
        out_dir: Path,
        per_unit_results_dir: Path,
        per_unit_filter_dir: Path,
        task_suite_name: str,
        host: str,
        port: int,
        seed: int,
        conda_env: Optional[str] = None,
        max_retries: int = 2,
    ) -> None:
        super().__init__(state_path=state_path, max_retries=max_retries)
        self.inits_dir = Path(inits_dir)
        self.filter_entries = filter_entries
        self.out_dir = Path(out_dir)
        self.per_unit_results_dir = Path(per_unit_results_dir)
        self.per_unit_filter_dir = Path(per_unit_filter_dir)
        self.task_suite_name = task_suite_name
        self.host = host
        self.port = port
        self.seed = seed
        self.conda_env = conda_env

        self._subset_size = subset_size_per_task(filter_entries)
        self._by_key: Dict[str, FilterEntry] = {
            build_unit_key(e.task_id, e.orig_init_state_idx): e
            for e in filter_entries
        }

    # ---- BaseRunState hooks ----

    def build_units(self) -> List[UnitState]:
        return [UnitState(unit_key=k) for k in sorted(self._by_key)]

    def execute_unit(self, unit: UnitState) -> dict:
        entry = self._by_key[unit.unit_key]
        task_id, orig_idx = entry.task_id, entry.orig_init_state_idx
        subset_idx = entry.subset_init_state_idx

        # Per-unit single-entry filter file. Keep one file per unit so two
        # workers (future parallel extension) never clobber each other.
        self.per_unit_filter_dir.mkdir(parents=True, exist_ok=True)
        filter_path = self.per_unit_filter_dir / f"unit_{task_id}_{orig_idx}.filter.json"
        filter_path.write_text(json.dumps([{
            "task_id": task_id,
            "orig_init_state_idx": orig_idx,
            "subset_init_state_idx": subset_idx,
        }]))

        self.per_unit_results_dir.mkdir(parents=True, exist_ok=True)
        results_path = self.per_unit_results_dir / f"unit_{task_id}_{orig_idx}.episode_results.json"
        # Wipe any stale result from a previous failed run so we can
        # confidently read-back.
        if results_path.exists():
            results_path.unlink()

        main_args = [
            "examples/libero/main.py",
            "--host", self.host,
            "--port", str(self.port),
            "--task-suite-name", self.task_suite_name,
            "--task-ids", str(task_id),
            "--num-trials-per-task", str(self._subset_size[task_id]),
            "--init-states-dir", str(self.inits_dir),
            "--episode-filter", str(filter_path),
            "--save-trajectory",
            "--save-trajectory-dir", str(self.out_dir),
            "--save-episode-results",
            "--episode-results-path", str(results_path),
            "--seed", str(self.seed),
        ]
        cmd, env = _build_subprocess_cmd(main_args, self.conda_env)

        proc = subprocess.run(
            cmd, env=env, capture_output=True, text=True, timeout=_UNIT_TIMEOUT_SECONDS,
        )
        if proc.returncode != 0:
            tail = "\n".join(proc.stdout.splitlines()[-20:])
            raise RuntimeError(
                f"main.py exit {proc.returncode} for {unit.unit_key}\n--- tail ---\n{tail}"
            )

        return _read_unit_result(
            results_path=results_path,
            out_dir=self.out_dir,
            task_id=task_id,
            subset_idx=subset_idx,
            orig_idx=orig_idx,
        )


# ---------------------------------------------------------------------------
# Subprocess / result-parsing helpers (public for unit tests)
# ---------------------------------------------------------------------------


def _build_subprocess_cmd(
    main_args: List[str], conda_env: Optional[str]
) -> Tuple[List[str], Optional[Dict[str, str]]]:
    """Mirror the env / command shape used in ``run_cache_experiments.py``."""
    if not conda_env:
        return ["uv", "run", *main_args], None

    cmd = ["conda", "run", "--no-capture-output", "-n", conda_env, "python", *main_args]
    env = {
        k: v for k, v in os.environ.items()
        if k not in ("VIRTUAL_ENV", "PYTHONPATH", "PYTHONHOME")
    }
    venv_bin = os.environ.get("VIRTUAL_ENV", "")
    if venv_bin:
        venv_bin = os.path.join(venv_bin, "bin")
        env["PATH"] = os.pathsep.join(
            p for p in env.get("PATH", "").split(os.pathsep) if p != venv_bin
        )
    env["MUJOCO_GL"] = "egl"
    return cmd, env


def _read_unit_result(
    *,
    results_path: Path,
    out_dir: Path,
    task_id: int,
    subset_idx: int,
    orig_idx: int,
) -> dict:
    """Read the per-episode JSON produced by ``main.py`` and return a
    runner-friendly summary.

    Contract (plan §9.2 pseudo-code):
    - ``success`` = the inner pi0.5 rollout succeeded.
    - ``hdf5_path`` = relative path under ``save_trajectory_dir`` to the GT
      HDF5 (``task_{id}/episode_{subset_idx}.h5``). main.py writes the HDF5
      even on failure (§20.R1.1 mid-chunk finalise) so Step 2/3 can still
      use partial trajectories if needed, but Step 1b flags failures so
      downstream code can elect to skip them.
    - ``inference_failed`` = True when no episode row was written at all
      (e.g. the subprocess crashed before reaching episode save). This is
      distinct from ``success=False``, which means the episode ran but did
      not reach the goal.
    """
    if not results_path.exists():
        return {
            "success": False,
            "inference_failed": True,
            "reason": "no episode_results.json emitted",
        }

    rows = json.loads(results_path.read_text())
    matching = [r for r in rows if int(r["init_state_idx"]) == subset_idx]
    if not matching:
        return {
            "success": False,
            "inference_failed": True,
            "reason": f"no row for subset_idx={subset_idx} in {results_path}",
            "rows": rows,
        }
    row = matching[-1]
    success = bool(row.get("success"))
    return {
        "success": success,
        "inference_failed": False,
        "hdf5_path": f"task_{task_id}/episode_{subset_idx}.h5",
        "orig_init_state_idx": orig_idx,
        "seed": int(row.get("seed", -1)),
    }


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inits-dir", required=True,
                    help="Directory from dump_step1a_failed_inits.py (contains .init + filter)")
    ap.add_argument("--out-dir", required=True, help="GT HDF5 output root (passed as --save-trajectory-dir to main.py)")
    ap.add_argument("--task-suite", required=True, help="LIBERO task suite name")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--conda-env", default=None,
                    help="Run main.py inside this conda env (mirrors run_cache_experiments.py)")
    ap.add_argument("--inference-yaml", default=DEFAULT_INFERENCE_YAML,
                    help="Cache bundle YAML loaded onto the server (plan §18.B4.3 default: clip_w7_d4)")
    ap.add_argument("--state-path", default=None, help="Runner state JSON (default: <inits-dir>/step1b_state.json)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--skip-config-switch", action="store_true",
                    help="Do not call load_cache_config — assume the server already runs the correct bundle")
    return ap.parse_args(argv)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    args = parse_args()

    inits_dir = Path(args.inits_dir)
    filter_path = inits_dir / "step1b_filter.json"
    if not filter_path.exists():
        raise FileNotFoundError(
            f"step1b_filter.json not found at {filter_path} — run "
            f"scripts/dump_step1a_failed_inits.py first."
        )
    entries = load_filter_entries(filter_path)
    if not entries:
        print("Filter is empty — nothing to collect.")
        return

    state_path = Path(args.state_path) if args.state_path else inits_dir / "step1b_state.json"
    out_dir = Path(args.out_dir)
    per_unit_filter_dir = state_path.parent / "per_unit_filters"
    per_unit_results_dir = state_path.parent / "per_unit_results"

    # Switch the server to the GT cache bundle once. All subsequent main.py
    # subprocesses inherit this configuration via their own WS connections.
    if not args.skip_config_switch:
        server_url = f"ws://{args.host}:{args.port}"
        version = send_load_cache_config(server_url, args.inference_yaml)
        logger.info("Server switched to bundle v%s for Step 1b", version)

    runner = Step1bRunner(
        state_path=state_path,
        inits_dir=inits_dir,
        filter_entries=entries,
        out_dir=out_dir,
        per_unit_results_dir=per_unit_results_dir,
        per_unit_filter_dir=per_unit_filter_dir,
        task_suite_name=args.task_suite,
        host=args.host,
        port=args.port,
        seed=args.seed,
        conda_env=args.conda_env,
    )
    start = time.monotonic()
    runner.run(resume=args.resume)
    elapsed = time.monotonic() - start

    done = [u for u in runner.units.values() if u.status == "done"]
    failed = [u for u in runner.units.values() if u.status == "failed"]
    print(
        f"Step 1b finished in {elapsed:.1f}s: {len(done)} done, {len(failed)} failed, "
        f"{len(runner.units)} total. State: {state_path}"
    )


if __name__ == "__main__":
    main()
