"""verdict_factor_judge — per-yaml warmup -> preload -> eval orchestrator.

This is the B2.5 runner shim. For every eval yaml under ``--phase-dir`` it
drives the seven-step protocol that the rest of B2 was built to support:

    1. Server-load ``<eval>__warmup.yaml``  (yaml_id = warmup_id)
    2. Spawn ``examples.libero.main`` workers running W warmup trials/task —
       the server-side ``DumpingJudge`` writes raw factor values to
       ``<warmup_dump_root>/<warmup_id>.jsonl``
    3. Fetch that dump file via ``WebsocketClientPolicy.fetch_dump``
    4. Aggregate raw values per-key into a ``{key: list[float]}`` buffer
       (NaNs are skipped so the eval normalizer sees only valid samples)
    5. Server-load ``<eval>.yaml`` (yaml_id = eval_id) and seed the
       per-yaml WarmupPool with ``preload_normalizer_buffer(eval_id, buffer)``
    6. Spawn ``examples.libero.main`` workers running E eval trials/task —
       per-step JSONL is written by the worker pool itself (B1.2)
    7. Call ``unload_warmup_buffer(eval_id)`` to free the pool entry and
       delete the warmup dump file

All wire helpers landed in B2.4 (``packages/openpi-client``) and the
server-side ctrl handlers landed in B2.3 (``websocket_policy_server.py``);
this script only orchestrates them. No new ws ctrls.

CLI invocation::

    uv run python -m exp.verdict_factor_judge.common.run_phase \\
        --phase-dir exp/verdict_factor_judge/config/clip/phase1 \\
        --host 155.98.36.13 --port 9000 \\
        --num-workers 5 --warmup-trials 2 --eval-trials 10 \\
        --task-suite libero_spatial \\
        --per-step-log-dir exp/verdict_factor_judge/data/phase1/clip/per_step \\
        --summary-out exp/verdict_factor_judge/data/phase1/clip/per_yaml_summary.jsonl

Failure-mode contract:

* Each eval yaml is wrapped in ``try / except`` so one yaml's crash does
  NOT abort the phase — the failure is logged with ``traceback.format_exc``
  and the loop continues with the next yaml.
* Inside a single yaml, ``_run_one_yaml`` re-raises any exception from
  warmup-subprocess / fetch_dump / preload / eval-subprocess so the caller
  can decide whether to skip vs retry; the helper itself never silently
  swallows a failure.
"""

from __future__ import annotations

import collections
import dataclasses
import json
import logging
import math
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Optional

import tyro

# B2.4 client SDK helpers — re-exported here so test code can monkeypatch
# ``run_phase.WebsocketClientPolicy`` without touching the openpi-client pkg.
from openpi_client.websocket_client_policy import WebsocketClientPolicy  # noqa: F401

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI args
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Args:
    """Runner CLI arguments (tyro-parsed).

    ``phase_dir`` is the only required argument; the others have production
    defaults that match the verdict_factor_judge runbook
    (``logs/verdict_factor_judge_phase0_phase1_run_commands.log.md``).
    """

    # Directory containing the Phase 1 eval yamls; sibling ``__warmup.yaml``
    # files are expected next to each eval yaml.
    phase_dir: str

    # Server endpoint. Default points at the public frpc relay used by the
    # remote inference server (per project memory: openpi front door is
    # 155.98.36.13:9000 → server 0.0.0.0:8000).
    host: str = "155.98.36.13"
    port: int = 9000

    # LIBERO task suite name forwarded to ``examples.libero.main``.
    task_suite: str = "libero_spatial"

    # Concurrency forwarded to ``examples.libero.main``. Default 5 matches
    # the production runbook (each worker holds its own normalizer).
    num_workers: int = 5

    # Trials per task for the warmup phase. 2 trials × 10 tasks × ~21 verdicts
    # ≈ 420 verdicts per yaml (plan §3.4 verdict-count math), enough to fill
    # a window_size=50 buffer per key with margin.
    warmup_trials: int = 2

    # Trials per task for the eval phase. 10 × 10 = 100 episodes / yaml
    # matching the existing Phase 1 evaluation budget.
    eval_trials: int = 10

    # Subset of task IDs to run (empty = all 10 libero_spatial tasks). Must
    # be the same set for warmup and eval so the buffer's distribution
    # matches the eval workload.
    task_ids: tuple[int, ...] = ()

    # Where ``examples.libero.main`` writes per-step JSONL (B1.2 schema). The
    # merged file ends up at ``<dir>/<yaml_id>.jsonl``. Empty disables.
    per_step_log_dir: str = ""

    # Per-yaml summary output (JSONL). One row per eval yaml; empty disables.
    summary_out: str = ""

    # Where ``examples.libero.main --save-episode-results`` writes the
    # per-episode success JSON. One file per eval yaml at
    # ``<dir>/<yaml_id>.json``. Empty disables SR forwarding (the summary's
    # ``success_rate`` field stays ``null`` like the legacy behaviour).
    # When ``--per-step-log-dir`` is set and this is empty, defaults to
    # ``<per_step_log_dir>/../episode_results``.
    episode_results_dir: str = ""

    # Resume mode: if ``--summary-out`` already exists, load completed
    # ``yaml_id`` set from it and skip those yamls (re-running them would
    # duplicate rows since each yaml is atomic — server warmup buffer is
    # in-memory and lost on restart, so per-yaml granularity is the right
    # checkpoint level). Without ``--resume`` the summary file is truncated
    # on start (legacy behaviour).
    resume: bool = False

    # Forwarded as-is to ``examples.libero.main`` when set.
    init_states_dir: str = ""
    episode_filter: str = ""

    # Pin the LIBERO sim / MuJoCo EGL renderer to a specific GPU. Forwarded
    # to ``examples.libero.main --cuda-visible-devices`` which sets
    # ``os.environ["CUDA_VISIBLE_DEVICES"]`` before importing CUDA-touching
    # libs. Empty preserves main.py's own default ("0").
    cuda_visible_devices: str = ""

    # Conda env that owns the LIBERO sim install (``libero``, ``robosuite``,
    # MuJoCo). Mirrors ``run_cache_experiments.py --conda-env``: when set,
    # the spawned ``examples.libero.main`` runs under
    # ``conda run -p/-n <env> python ...`` with VIRTUAL_ENV / PYTHONPATH /
    # PYTHONHOME stripped so the conda interpreter wins. Empty falls back
    # to ``uv run``, which only works when LIBERO is installed in the
    # current uv venv (typical only on the WSL2 client host).
    conda_env: str = ""

    # Debug shortcut — skip the warmup phase entirely (used while iterating
    # on the eval yaml without re-collecting in-distribution factor values).
    skip_warmup: bool = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_dump_to_buffer(content: bytes, max_per_key: int = 200) -> dict[str, list[float]]:
    """Aggregate the warmup JSONL dump into ``{factor_key: list[float]}``.

    Drops NaN values (the eval normalizer's percentile buffer can't use
    them anyway and they pollute the cap). ``max_per_key`` bounds the list
    length so a runaway warmup run can't push GBs of values across the wire
    on the next ``preload_normalizer_buffer``.

    The JSONL row schema is set by ``DumpingJudge._write_dump_row``
    (``src/openpi/cache/components/judge.py:631``); we only read
    ``row["factor_raw"]`` here.
    """
    buffer: dict[str, list[float]] = collections.defaultdict(list)
    text = content.decode("utf-8") if isinstance(content, (bytes, bytearray)) else str(content)
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            # Tolerate the (unlikely) case of a partially-flushed last line —
            # the worker uses ``buffering=1`` line-buffered so an in-flight
            # write is rare but a SIGKILL during a row write would leave one.
            logger.warning("Skipping malformed JSONL line in warmup dump")
            continue
        factor_raw = row.get("factor_raw") or {}
        for k, v in factor_raw.items():
            if v is None:
                continue
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if math.isnan(fv):
                continue
            if len(buffer[k]) >= max_per_key:
                # Cap reached: silently skip the remainder. Keeping FIFO is
                # not important here — the warmup distribution is iid w.r.t.
                # ordering since workers interleave non-deterministically.
                continue
            buffer[k].append(fv)
    return dict(buffer)


def _build_libero_argv(
    *,
    args: Args,
    yaml_id: str,
    phase: str,
    num_trials_per_task: int,
    episode_results_path: Optional[Path] = None,
) -> tuple[list[str], dict | None]:
    """Assemble the ``examples.libero.main`` ``(cmd, env)`` for one subprocess.

    Returns the same ``(cmd, env)`` tuple shape as
    ``exp.common._subprocess.build_subprocess_cmd`` so the caller can pass
    ``env`` straight into ``subprocess.run`` — this is the one place that
    knows whether to wrap the spawn in ``conda run`` (LIBERO host) or
    plain ``uv run`` (WSL2 client with libero in the uv venv).

    ``tyro`` accepts ``--task-ids 0 1 2`` for tuples and ``--task-ids ''``
    has no clean spelling, so when ``args.task_ids`` is empty we omit the
    flag entirely (the script's default ``()`` matches "all tasks").
    """
    main_args: list[str] = [
        "examples/libero/main.py",
        "--host", str(args.host),
        "--port", str(args.port),
        "--task-suite-name", str(args.task_suite),
        "--num-workers", str(args.num_workers),
        "--num-trials-per-task", str(num_trials_per_task),
        "--yaml-id", yaml_id,
        "--phase", phase,
    ]
    if args.per_step_log_dir:
        main_args += ["--per-step-log-dir", args.per_step_log_dir]
    if args.task_ids:
        main_args.append("--task-ids")
        main_args += [str(t) for t in args.task_ids]
    if args.init_states_dir:
        main_args += ["--init-states-dir", args.init_states_dir]
    if args.episode_filter:
        main_args += ["--episode-filter", args.episode_filter]
    if args.cuda_visible_devices:
        main_args += ["--cuda-visible-devices", args.cuda_visible_devices]
    # Forward per-episode SR JSON only on eval; warmup uses AlwaysWarmStart
    # so its "success rate" is not the metric of interest.
    if episode_results_path is not None and phase == "eval":
        main_args += [
            "--save-episode-results",
            "--episode-results-path", str(episode_results_path),
        ]

    from exp.common._subprocess import build_subprocess_cmd

    extra_env: Optional[dict] = None
    if args.conda_env:
        # ``build_subprocess_cmd`` strips PYTHONPATH so the conda interpreter
        # finds its own packages without uv-venv interference. main.py's
        # lazy ``from exp.verdict_factor_judge.common.per_step_log_writer import ...``
        # fires when --per-step-log-dir is set; for that import to resolve
        # we need the repo root back on PYTHONPATH (the conda interpreter
        # otherwise only sees ``examples/libero/`` from sys.argv[0]).
        repo_root = str(Path(__file__).resolve().parents[3])
        extra_env = {"PYTHONPATH": repo_root}

    return build_subprocess_cmd(
        main_args, conda_env=args.conda_env or None, extra_env=extra_env,
    )


def _resolve_episode_results_dir(args: Args) -> Optional[Path]:
    """Return the directory that holds per-yaml ``episode_results.json``
    files, or ``None`` when SR forwarding is disabled.

    Explicit ``--episode-results-dir`` wins. Otherwise falls back to
    ``<per_step_log_dir>/../episode_results`` so the SR files sit next to
    ``per_step/`` under the same cfg data dir. When neither is available we
    return ``None`` and ``success_rate`` stays ``null`` (legacy behaviour).
    """
    if args.episode_results_dir:
        return Path(args.episode_results_dir)
    if args.per_step_log_dir:
        return Path(args.per_step_log_dir).parent / "episode_results"
    return None


def _aggregate_sr_from_episode_json(path: Path) -> Optional[float]:
    """Compute success_rate from main.py's ``--save-episode-results`` JSON.

    The schema is a list of dicts with a ``success: bool`` field per
    episode (see ``examples/libero/main.py:_write_episode_results_json``).
    Returns ``None`` when the file is missing / empty / malformed so the
    caller can leave the summary's SR field null without crashing the run.
    """
    if not path.exists():
        return None
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        logger.warning("Failed to parse episode results JSON: %s", path)
        return None
    if not isinstance(rows, list) or not rows:
        return None
    successes = sum(1 for r in rows if isinstance(r, dict) and r.get("success") is True)
    return successes / len(rows)


def _load_done_yaml_ids(summary_path: Path) -> set[str]:
    """Read ``summary_out`` to find which yaml_ids already have a row.

    Used by ``--resume`` to skip already-completed yamls. A row counts as
    "done" purely by presence — partial / failed rows aren't written by the
    main loop (failures hit the ``except`` branch and skip the write), so
    every persisted row corresponds to a successful eval.
    """
    if not summary_path.exists():
        return set()
    done: set[str] = set()
    for line in summary_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        yid = row.get("yaml_id")
        if isinstance(yid, str):
            done.add(yid)
    return done


def _summarize_per_step_log(per_step_log_dir: str, eval_yaml_id: str) -> dict:
    """Tally ``hit_type`` counts in the eval phase of the per-step JSONL.

    Best-effort: returns zero counts if the file does not exist (e.g. the
    user did not pass ``--per-step-log-dir``). The caller decides whether
    a missing file is fatal.
    """
    counts = {
        "n_eval_verdicts": 0,
        "n_full_hit": 0,
        "n_warm_start": 0,
        "n_miss": 0,
    }
    if not per_step_log_dir:
        return counts
    path = Path(per_step_log_dir) / f"{eval_yaml_id}.jsonl"
    if not path.exists():
        return counts
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("phase") != "eval":
            continue
        counts["n_eval_verdicts"] += 1
        ht = (row.get("hit_type") or "").upper()
        if ht == "FULL_HIT":
            counts["n_full_hit"] += 1
        elif ht == "WARM_START":
            counts["n_warm_start"] += 1
        elif ht == "MISS":
            counts["n_miss"] += 1
    return counts


# ---------------------------------------------------------------------------
# Per-yaml orchestration
# ---------------------------------------------------------------------------


def _run_one_yaml(args: Args, eval_path: Path) -> dict:
    """Drive one eval yaml through the full 7-step protocol.

    Returns the per-yaml summary dict (the caller decides whether to write
    it to ``--summary-out``). Re-raises on any wire / subprocess failure so
    the outer loop can record + skip; this helper never silently consumes
    an error.
    """
    eval_yaml_id = eval_path.stem
    warmup_yaml_id = f"{eval_yaml_id}__warmup"
    warmup_path = eval_path.parent / f"{eval_yaml_id}__warmup.yaml"

    if not args.skip_warmup and not warmup_path.exists():
        raise FileNotFoundError(
            f"Sibling warmup yaml missing for {eval_path}: expected {warmup_path}"
        )

    n_warmup_keys: Optional[int] = None
    buffer: dict[str, list[float]] = {}

    if not args.skip_warmup:
        # ── 1. server-load warmup yaml ────────────────────────────────────
        logger.info("[%s] Step 1/7: load warmup yaml (%s)", eval_yaml_id, warmup_yaml_id)
        with WebsocketClientPolicy(host=args.host, port=args.port) as ctl:
            ctl.load_cache_config(
                yaml_content=warmup_path.read_text(encoding="utf-8"),
                yaml_id=warmup_yaml_id,
            )

        # ── 2. spawn workers for warmup ───────────────────────────────────
        logger.info("[%s] Step 2/7: warmup workers (%d trials/task)",
                    eval_yaml_id, args.warmup_trials)
        warmup_cmd, warmup_env = _build_libero_argv(
            args=args,
            yaml_id=warmup_yaml_id,
            phase="warmup",
            num_trials_per_task=args.warmup_trials,
        )
        # check=True → CalledProcessError surfaces on non-zero exit so the
        # outer loop counts this yaml as failed (rather than silently
        # producing a half-warmed buffer).
        subprocess.run(warmup_cmd, env=warmup_env, check=True)

        # ── 3. fetch the dump file ────────────────────────────────────────
        logger.info("[%s] Step 3/7: fetch warmup dump", eval_yaml_id)
        with WebsocketClientPolicy(host=args.host, port=args.port) as ctl:
            content = ctl.fetch_dump(warmup_yaml_id)

        # ── 4. aggregate per-key ─────────────────────────────────────────
        logger.info("[%s] Step 4/7: aggregate raw factor values", eval_yaml_id)
        buffer = _parse_dump_to_buffer(content, max_per_key=200)
        n_warmup_keys = len(buffer)
        logger.info("[%s] Aggregated %d keys (sample sizes: min=%d max=%d)",
                    eval_yaml_id, n_warmup_keys,
                    min((len(v) for v in buffer.values()), default=0),
                    max((len(v) for v in buffer.values()), default=0))

    # ── 5+6. load eval yaml + preload buffer ──────────────────────────────
    logger.info("[%s] Step 5/7: load eval yaml", eval_yaml_id)
    with WebsocketClientPolicy(host=args.host, port=args.port) as ctl:
        ctl.load_cache_config(
            yaml_content=eval_path.read_text(encoding="utf-8"),
            yaml_id=eval_yaml_id,
        )
        if buffer:
            logger.info("[%s] Step 6/7: preload normalizer buffer (%d keys)",
                        eval_yaml_id, len(buffer))
            ack = ctl.preload_normalizer_buffer(eval_yaml_id, buffer)
            # Server returns ``n_keys`` for round-trip sanity; keep ours
            # authoritative when the server doesn't echo it back.
            n_warmup_keys = ack.get("n_keys", n_warmup_keys)

    # ── 7. spawn workers for eval ────────────────────────────────────────
    logger.info("[%s] Step 7/7: eval workers (%d trials/task)",
                eval_yaml_id, args.eval_trials)
    episode_results_dir = _resolve_episode_results_dir(args)
    episode_results_path: Optional[Path] = None
    if episode_results_dir is not None:
        episode_results_dir.mkdir(parents=True, exist_ok=True)
        episode_results_path = episode_results_dir / f"{eval_yaml_id}.json"
    eval_cmd, eval_env = _build_libero_argv(
        args=args,
        yaml_id=eval_yaml_id,
        phase="eval",
        num_trials_per_task=args.eval_trials,
        episode_results_path=episode_results_path,
    )
    subprocess.run(eval_cmd, env=eval_env, check=True)

    # ── 8. unload pool + delete warmup file ──────────────────────────────
    if not args.skip_warmup:
        logger.info("[%s] Cleanup: unload warmup buffer", eval_yaml_id)
        with WebsocketClientPolicy(host=args.host, port=args.port) as ctl:
            ctl.unload_warmup_buffer(eval_yaml_id)

    # ── 9. summarize ─────────────────────────────────────────────────────
    counts = _summarize_per_step_log(args.per_step_log_dir, eval_yaml_id)
    success_rate: Optional[float] = None
    if episode_results_path is not None:
        success_rate = _aggregate_sr_from_episode_json(episode_results_path)
    summary = {
        "yaml_id": eval_yaml_id,
        "warmup_yaml_id": warmup_yaml_id if not args.skip_warmup else None,
        "n_warmup_buffer_keys": n_warmup_keys,
        "success_rate": success_rate,
        **counts,
    }
    return summary


def _iter_eval_yamls(phase_dir: Path) -> list[Path]:
    """Return the eval yamls in deterministic lex order, excluding siblings.

    The ``*__warmup.yaml`` suffix is reserved for the sibling warmup yamls
    emitted by ``phase1_spec.main()`` — they MUST be skipped here because
    the warmup yaml is loaded as part of running its eval sibling, not on
    its own.
    """
    out: list[Path] = []
    for p in sorted(phase_dir.iterdir()):
        if not p.is_file() or p.suffix != ".yaml":
            continue
        if p.stem.endswith("__warmup"):
            continue
        out.append(p)
    return out


def main(args: Args) -> int:
    """Run the full phase. Returns an exit code (0 = all yamls succeeded)."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
    phase_dir = Path(args.phase_dir)
    if not phase_dir.is_dir():
        logger.error("phase-dir does not exist or is not a directory: %s", phase_dir)
        return 2

    eval_yamls = _iter_eval_yamls(phase_dir)
    logger.info("Discovered %d eval yamls under %s", len(eval_yamls), phase_dir)

    summary_path: Optional[Path] = Path(args.summary_out) if args.summary_out else None
    done_ids: set[str] = set()
    if summary_path is not None:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        if args.resume:
            # Keep existing rows and skip yamls that already have a row.
            done_ids = _load_done_yaml_ids(summary_path)
            logger.info(
                "Resume mode: %d/%d yamls already complete in %s",
                len(done_ids), len(eval_yamls), summary_path,
            )
        else:
            # Legacy behaviour: truncate so re-runs don't duplicate rows.
            summary_path.write_text("", encoding="utf-8")

    n_ok = 0
    n_fail = 0
    n_skip = 0
    for eval_path in eval_yamls:
        if eval_path.stem in done_ids:
            n_skip += 1
            logger.info("[%s] SKIP — already in summary (resume)", eval_path.stem)
            continue
        try:
            summary = _run_one_yaml(args, eval_path)
            n_ok += 1
            if summary_path is not None:
                with summary_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(summary) + "\n")
        except Exception:  # noqa: BLE001
            # Per-yaml isolation: log + continue. We prefer a partial phase
            # to no phase at all, but the failure must be loud (full TB) so
            # operators don't miss it in the log.
            n_fail += 1
            logger.error(
                "[%s] FAILED — skipping this yaml:\n%s",
                eval_path.stem,
                traceback.format_exc(),
            )
            continue

    logger.info("Phase complete: %d ok, %d failed, %d skipped (out of %d)",
                n_ok, n_fail, n_skip, len(eval_yamls))
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main(tyro.cli(Args)))
