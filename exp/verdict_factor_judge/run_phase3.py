"""Phase 3 runner — per-recipe warmup -> threshold solve -> 16-cell eval.

Plan §7: orchestrates the 11 spatial16 gold-circle recipes through the
two-pass flow:

    Per recipe:
      1. Server-load <recipe>__warmup.yaml; spawn libero warmup workers
         (W trials/task; DumpingJudge writes raw factor JSONL).
      2. Fetch the dump, save as ``<phase3>/warmup/<recipe>__warmup.jsonl``.
      3. Aggregate the dump per-key (the eval calibration buffer needs
         this) and run ``solve_recipe(...)`` to derive 16 (FH_thr, WS_thr)
         pairs from the same factor_raw column.
      4. For each (fh_ratio, ws_ratio) cell:
           a. Build eval yaml with the cell's thresholds patched in.
           b. Server-load eval yaml; preload normalizer buffer with the
              aggregated warmup samples (same wire as run_phase.py).
           c. Spawn libero eval workers (E trials/task).
           d. Append a per-yaml summary row.
      5. Unload warmup buffer + delete dump.

Everything below the ``Args`` block reuses helpers from ``run_phase``
verbatim (``_parse_dump_to_buffer``, ``_build_libero_argv``,
``_summarize_per_step_log``, etc.) — runner shape is identical.

CLI invocation::

    uv run python -m exp.verdict_factor_judge.run_phase3 \\
        --cfg-id spatial16_w8_d4 \\
        --host 155.98.36.13 --port 9000 \\
        --num-workers 5 --warmup-trials 2 --eval-trials 10 \\
        --task-suite libero_spatial \\
        --per-step-log-dir exp/verdict_factor_judge/data/phase3/per_step \\
        --episode-results-dir exp/verdict_factor_judge/data/phase3/episode_results \\
        --summary-out exp/verdict_factor_judge/data/phase3/per_yaml_summary.jsonl \\
        --warmup-jsonl-dir exp/verdict_factor_judge/data/phase3/warmup \\
        --thresholds-dir exp/verdict_factor_judge/data/phase3/thresholds \\
        --eval-yaml-dir exp/verdict_factor_judge/config/spatial16/phase3/eval

Failure-mode contract: per-recipe try/except so one recipe's failure
does not abort the phase. Inside a recipe, any wire / subprocess error
re-raises to the recipe-level handler.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Optional

import tyro

from exp.verdict_factor_judge.generate_yamls import write_yaml
from exp.verdict_factor_judge.phase3_spec import (
    GRID,
    RECIPES,
    build_eval_yaml_for_cell,
    build_warmup_yaml_for_recipe,
    recipe_to_solver_recipe,
)
from exp.verdict_factor_judge.phase3_threshold_solver import (
    load_per_key_finite_history,
    solve_recipe,
)
from exp.verdict_factor_judge.run_phase import (
    _aggregate_sr_from_episode_json,
    _build_libero_argv,
    _summarize_per_step_log,
)
from openpi_client.websocket_client_policy import WebsocketClientPolicy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI args
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Args:
    """Phase 3 runner arguments (tyro-parsed)."""

    # cfg key into phase3_spec.CFG_SPECS. Phase 3 plan locks spatial16 only.
    cfg_id: str = "spatial16_w8_d4"

    # Subset of recipe ids to run (empty = all 11). Allows phase3 to be
    # sharded across multiple servers.
    recipe_ids: tuple[str, ...] = ()

    # Server endpoint (frpc default — see run_phase.Args).
    host: str = "155.98.36.13"
    port: int = 9000

    task_suite: str = "libero_spatial"
    num_workers: int = 5
    warmup_trials: int = 2     # 20 episodes / recipe (10 task * W=2)
    eval_trials: int = 10      # 100 episodes / cell (10 task * E=10)
    task_ids: tuple[int, ...] = ()

    # Logging / output
    per_step_log_dir: str = ""
    summary_out: str = ""
    episode_results_dir: str = ""

    # Phase 3-specific output dirs.
    warmup_jsonl_dir: str = ""    # where _save_dump_jsonl writes the raw dump
    thresholds_dir: str = ""      # where solve_recipe's JSON output goes
    eval_yaml_dir: str = ""       # where the patched eval yamls are written
    warmup_yaml_dir: str = ""     # where the spec module emitted warmups (read-only)

    # Resume / debug
    resume: bool = False
    skip_warmup: bool = False
    init_states_dir: str = ""
    episode_filter: str = ""
    cuda_visible_devices: str = ""
    conda_env: str = ""


# ---------------------------------------------------------------------------
# Helpers — phase3-specific
# ---------------------------------------------------------------------------


def _save_dump_jsonl(content: bytes, out_path: Path) -> Path:
    """Write fetched dump bytes to a JSONL file. Idempotent: overwrites."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    text = content.decode("utf-8") if isinstance(content, (bytes, bytearray)) else str(content)
    out_path.write_text(text)
    return out_path


def _write_na_summary_rows(
    summary_path: Optional[Path],
    *,
    recipe_id: str,
    warmup_yaml_id: str,
    warmup_eval_yaml_id: str,
    error: str,
    grid: list[tuple[float, float]],
) -> list[dict]:
    """Append one ``_NA`` summary row per cell on solver fail-fast.

    Plan §5.7 + G2 R1 B2: an insufficient-sample recipe is skipped with
    ``_NA`` markers that propagate to ``per_yaml_summary.jsonl`` so
    downstream aggregation does not silently lose 16 cells.
    """
    rows: list[dict] = []
    for fh_ratio, ws_ratio in grid:
        rows.append({
            "yaml_id": f"{warmup_eval_yaml_id}__fh{fh_ratio}_ws{ws_ratio}__NA",
            "recipe_id": recipe_id,
            "fh_ratio": fh_ratio,
            "ws_ratio": ws_ratio,
            "fh_thr": None,
            "ws_thr": None,
            "warmup_yaml_id": warmup_yaml_id,
            "n_warmup_buffer_keys": None,
            "success_rate": None,
            "n_eval_verdicts": 0,
            "n_full_hit": 0,
            "n_warm_start": 0,
            "n_miss": 0,
            "error": error,
        })
    if summary_path is not None:
        with summary_path.open("a") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
    return rows


def _resolve_episode_results_dir(args: Args) -> Optional[Path]:
    if args.episode_results_dir:
        return Path(args.episode_results_dir)
    if args.per_step_log_dir:
        return Path(args.per_step_log_dir).parent / "episode_results"
    return None


def _load_done_yaml_ids(summary_path: Path) -> set[str]:
    """Resume support: collect already-processed yaml_ids from a previous run."""
    if not summary_path.exists():
        return set()
    done: set[str] = set()
    for line in summary_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "yaml_id" in row:
            done.add(row["yaml_id"])
    return done


# ---------------------------------------------------------------------------
# Per-recipe driver
# ---------------------------------------------------------------------------


def _run_one_recipe(
    args: Args,
    recipe_id: str,
    *,
    warmup_yaml_dir: Path,
    warmup_jsonl_dir: Path,
    thresholds_dir: Path,
    eval_yaml_dir: Path,
    summary_path: Optional[Path],
    done_ids: set[str],
) -> list[dict]:
    """Run one recipe's warmup + 16 eval cells; return a list of per-cell summaries."""
    warmup_eval_yaml_id = f"{args.cfg_id}_phase3_{recipe_id}"
    warmup_yaml_id = f"{warmup_eval_yaml_id}__warmup"
    warmup_yaml_path = warmup_yaml_dir / f"{warmup_yaml_id}.yaml"
    warmup_jsonl_path = warmup_jsonl_dir / f"{warmup_yaml_id}.jsonl"

    if not warmup_yaml_path.exists():
        raise FileNotFoundError(
            f"Warmup yaml missing: {warmup_yaml_path}. "
            "Run `uv run python -m exp.verdict_factor_judge.phase3_spec` first."
        )

    buffer: dict[str, list[float]] = {}
    n_warmup_keys: Optional[int] = None

    # ── 1+2. warmup ──────────────────────────────────────────────────────
    if not args.skip_warmup:
        logger.info("[%s] warmup load + run (%d trials/task)",
                    recipe_id, args.warmup_trials)
        with WebsocketClientPolicy(host=args.host, port=args.port) as ctl:
            ctl.load_cache_config(
                yaml_content=warmup_yaml_path.read_text(encoding="utf-8"),
                yaml_id=warmup_yaml_id,
            )
        warmup_cmd, warmup_env = _build_libero_argv(
            args=args,
            yaml_id=warmup_yaml_id,
            phase="warmup",
            num_trials_per_task=args.warmup_trials,
        )
        subprocess.run(warmup_cmd, env=warmup_env, check=True)

        # ── 3. fetch dump and save JSONL ────────────────────────────────
        with WebsocketClientPolicy(host=args.host, port=args.port) as ctl:
            content = ctl.fetch_dump(warmup_yaml_id)
        _save_dump_jsonl(content, warmup_jsonl_path)

    if not warmup_jsonl_path.exists():
        raise FileNotFoundError(
            f"warmup JSONL missing: {warmup_jsonl_path} (use --skip-warmup only "
            "when a prior run already wrote this file)"
        )

    # G2 R1 B1 fix: load buffer from the same JSONL the solver scores
    # from, finite-only and uncapped. Server-side bind_keys then takes
    # the last 50 finite values per key — identical to the saturated
    # buffer reconstruct_scores uses, so eval-time calibration matches
    # the threshold-derivation distribution by construction.
    solver_recipe = recipe_to_solver_recipe(recipe_id)
    buffer = load_per_key_finite_history(
        warmup_jsonl_path, solver_recipe.declared_keys,
    )
    n_warmup_keys = len(buffer)
    logger.info(
        "[%s] aggregated %d keys (per-key sample size min=%d max=%d)",
        recipe_id, n_warmup_keys,
        min((len(v) for v in buffer.values()), default=0),
        max((len(v) for v in buffer.values()), default=0),
    )

    # ── 4. solve thresholds ──────────────────────────────────────────────
    thresholds_dir.mkdir(parents=True, exist_ok=True)
    thresholds_path = thresholds_dir / f"{recipe_id}__thresholds.json"
    threshold_doc = solve_recipe(
        warmup_jsonl_path,
        solver_recipe,
        list(GRID),
        warmup_yaml_id=warmup_yaml_id,
    )
    thresholds_path.write_text(json.dumps(threshold_doc, indent=2) + "\n")

    if "error" in threshold_doc:
        logger.error(
            "[%s] solver fail-fast: %s — recipe skipped (16 _NA cells written)",
            recipe_id, threshold_doc["error"],
        )
        na_rows = _write_na_summary_rows(
            summary_path,
            recipe_id=recipe_id,
            warmup_yaml_id=warmup_yaml_id,
            warmup_eval_yaml_id=warmup_eval_yaml_id,
            error=threshold_doc["error"],
            grid=list(GRID),
        )
        if not args.skip_warmup:
            with WebsocketClientPolicy(host=args.host, port=args.port) as ctl:
                ctl.unload_warmup_buffer(warmup_eval_yaml_id)
        return na_rows

    # ── 5+6+7. eval over 16 cells ────────────────────────────────────────
    cells_summaries: list[dict] = []
    eval_yaml_dir.mkdir(parents=True, exist_ok=True)
    episode_results_dir = _resolve_episode_results_dir(args)
    if episode_results_dir is not None:
        episode_results_dir.mkdir(parents=True, exist_ok=True)

    for fh_ratio, ws_ratio in GRID:
        cell_key = f"fh{fh_ratio}_ws{ws_ratio}"
        eval_yaml_id = f"{warmup_eval_yaml_id}__{cell_key}"
        if eval_yaml_id in done_ids:
            logger.info("[%s] resume: %s already done — skipping", recipe_id, eval_yaml_id)
            continue
        cell = threshold_doc["cells"][cell_key]
        eval_yaml_dict = build_eval_yaml_for_cell(
            args.cfg_id, recipe_id,
            fh_thr=cell["fh_thr"], ws_thr=cell["ws_thr"],
            fh_ratio=fh_ratio, ws_ratio=ws_ratio,
        )
        eval_yaml_path = eval_yaml_dir / f"{eval_yaml_id}.yaml"
        write_yaml(eval_yaml_path, eval_yaml_dict)

        # Phase 3 ordering: preload BEFORE load_cache_config — the eval
        # yaml's calibration uses ``samples_source.type=warmup`` which on
        # load_cache_config calls ``_load_calibration_samples(yaml_id=
        # eval_yaml_id)`` against the WarmupPool. WarmupPool is a module-
        # level dict keyed by eval_yaml_id; preload populates that entry,
        # so it MUST run before load_cache_config or load fails with
        # "WarmupPool has no entry for yaml_id=...".
        # (Differs from phase2's run_phase.py where the legacy yaml uses
        #  ``cold_start_strategy: force_miss`` and never reads WarmupPool.)
        with WebsocketClientPolicy(host=args.host, port=args.port) as ctl:
            ack = ctl.preload_normalizer_buffer(eval_yaml_id, buffer)
            n_warmup_keys = ack.get("n_keys", n_warmup_keys)
            ctl.load_cache_config(
                yaml_content=eval_yaml_path.read_text(encoding="utf-8"),
                yaml_id=eval_yaml_id,
            )

        # Eval workers.
        episode_results_path = (
            episode_results_dir / f"{eval_yaml_id}.json"
            if episode_results_dir is not None else None
        )
        eval_cmd, eval_env = _build_libero_argv(
            args=args,
            yaml_id=eval_yaml_id,
            phase="eval",
            num_trials_per_task=args.eval_trials,
            episode_results_path=episode_results_path,
        )
        subprocess.run(eval_cmd, env=eval_env, check=True)

        # Cell-end cleanup: unload this cell's WarmupPool entry, then RESET
        # the server bundle to the recipe's warmup yaml (AlwaysWarmStartJudge,
        # no WarmupPool dependency).
        #
        # Why the reset is required: a new client connection triggers
        # ``_connection_policy_factory(self._policy)`` on the server which
        # calls ``build_per_connection_components(_current_bundle)`` BEFORE
        # the client can send any ctrl command. If ``_current_bundle`` is
        # this cell's eval yaml (samples_source=warmup) and we just popped
        # its WarmupPool entry, that build fails with
        # "WarmupPool has no entry for yaml_id=...; the sibling warmup yaml
        # must run before this eval yaml loads." — and the next cell's
        # WebsocketClientPolicy(...) connect raises 1011 internal error
        # before it can preload + reload.
        #
        # Rolling the bundle back to the warmup yaml between cells keeps
        # every connection-time build_per_connection_components against an
        # AlwaysWarmStartJudge config, which has no calibration source and
        # therefore no WarmupPool lookup. The next cell then preloads its
        # buffer and load_cache_config(eval_yaml_id) switches the bundle
        # back to the cell's eval state for the actual eval rollout.
        with WebsocketClientPolicy(host=args.host, port=args.port) as ctl:
            ctl.unload_warmup_buffer(eval_yaml_id)
            ctl.load_cache_config(
                yaml_content=warmup_yaml_path.read_text(encoding="utf-8"),
                yaml_id=warmup_yaml_id,
            )

        # Summarize.
        counts = _summarize_per_step_log(args.per_step_log_dir, eval_yaml_id)
        success_rate: Optional[float] = None
        if episode_results_path is not None:
            success_rate = _aggregate_sr_from_episode_json(episode_results_path)
        summary = {
            "yaml_id": eval_yaml_id,
            "recipe_id": recipe_id,
            "fh_ratio": fh_ratio,
            "ws_ratio": ws_ratio,
            "fh_thr": cell["fh_thr"],
            "ws_thr": cell["ws_thr"],
            "warmup_yaml_id": warmup_yaml_id,
            "n_warmup_buffer_keys": n_warmup_keys,
            "success_rate": success_rate,
            **counts,
        }
        cells_summaries.append(summary)
        if summary_path is not None:
            with summary_path.open("a") as fh:
                fh.write(json.dumps(summary) + "\n")

    # G2 R1 B5 fix: recipe-level cleanup. The per-cell unload above only
    # frees each eval yaml's WarmupPool entry; the shared warmup yaml's
    # own pool entry + dump file are released here, after every cell of
    # this recipe has finished.
    if not args.skip_warmup:
        logger.info("[%s] recipe done: unload warmup buffer", recipe_id)
        try:
            with WebsocketClientPolicy(host=args.host, port=args.port) as ctl:
                ctl.unload_warmup_buffer(warmup_eval_yaml_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[%s] recipe-level unload_warmup_buffer raised: %r — leaving "
                "the warmup pool entry in place (server cleanup will catch it)",
                recipe_id, exc,
            )

    return cells_summaries


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _select_recipes(args: Args) -> list[str]:
    if args.recipe_ids:
        unknown = [r for r in args.recipe_ids if r not in RECIPES]
        if unknown:
            raise ValueError(f"unknown recipe ids: {unknown}")
        return list(args.recipe_ids)
    return list(RECIPES.keys())


def main(args: Args) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg_short = args.cfg_id.split("_")[0]
    base_data = Path(f"exp/verdict_factor_judge/data/phase3")
    base_cfg = Path(f"exp/verdict_factor_judge/config/{cfg_short}/phase3")

    warmup_yaml_dir = (
        Path(args.warmup_yaml_dir) if args.warmup_yaml_dir else base_cfg / "warmup"
    )
    warmup_jsonl_dir = (
        Path(args.warmup_jsonl_dir) if args.warmup_jsonl_dir else base_data / "warmup"
    )
    thresholds_dir = (
        Path(args.thresholds_dir) if args.thresholds_dir else base_data / "thresholds"
    )
    eval_yaml_dir = (
        Path(args.eval_yaml_dir) if args.eval_yaml_dir else base_cfg / "eval"
    )

    summary_path: Optional[Path] = None
    done_ids: set[str] = set()
    if args.summary_out:
        summary_path = Path(args.summary_out)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        if args.resume and summary_path.exists():
            done_ids = _load_done_yaml_ids(summary_path)
            logger.info("resume: %d yaml_ids already done", len(done_ids))
        elif not args.resume:
            summary_path.write_text("")    # truncate

    failures: list[tuple[str, str]] = []
    for recipe_id in _select_recipes(args):
        try:
            _run_one_recipe(
                args,
                recipe_id,
                warmup_yaml_dir=warmup_yaml_dir,
                warmup_jsonl_dir=warmup_jsonl_dir,
                thresholds_dir=thresholds_dir,
                eval_yaml_dir=eval_yaml_dir,
                summary_path=summary_path,
                done_ids=done_ids,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("[%s] recipe failed: %s", recipe_id, exc)
            failures.append((recipe_id, traceback.format_exc()))

    if failures:
        logger.error("phase3 finished with %d failed recipe(s):", len(failures))
        for rid, _ in failures:
            logger.error("  - %s", rid)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(tyro.cli(Args)))
