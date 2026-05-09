"""Phase 4 runner — 4-mode pipeline for the weight sweep over phase3 winners.

Plan §4.2: phase 4 runs in four explicit modes that progress strictly in
this order, with mandatory user intervention between rounds (decision
gate inspection):

    Per phase4 launch:
      mode 1  emit-warmup-yaml    (no server) write per-recipe warmup yaml
      mode 2  run-warmup          (server)    run warmup, dump factor_raw,
                                              extract finite-only jsonl
      mode 3  emit-eval-yamls     (no server) per round, build cell list,
                                              solve thresholds, write yaml
      mode 4  run-eval            (server)    preload buffer -> load eval
                                              yaml -> run libero workers

The two no-server modes do pure file I/O (yaml emission + threshold
solving from cached factor_raw); the two server modes need
``--host``/``--port`` and reuse the WarmupPool ordering established by
phase 3 (preload_normalizer_buffer BEFORE load_cache_config).

After each round's run-eval, ``_dump_decision_gate_table`` writes a
``decision_gate.json`` whose ``next_args_suggestion`` field contains the
exact CLI mapping the operator can paste into the next round's
``--alpha-star`` / ``--offline-pattern`` / ``--online-pattern``.

CLI invocation (R1 example, full pipeline)::

    # 1. Emit warmup yamls (one per phase4 recipe).
    uv run python -m exp.verdict_factor_judge.run_phase4 \\
        --mode emit-warmup-yaml --round 1

    # 2. Run warmup (writes data/phase4/warmup_factor_raw/<rid>.jsonl).
    uv run python -m exp.verdict_factor_judge.run_phase4 \\
        --mode run-warmup --round 1 \\
        --host 155.98.36.13 --port 9000

    # 3. Emit R1 eval yamls (14 cells: 2 recipe x 7 alpha).
    uv run python -m exp.verdict_factor_judge.run_phase4 \\
        --mode emit-eval-yamls --round 1

    # 4. Run R1 eval cells.
    uv run python -m exp.verdict_factor_judge.run_phase4 \\
        --mode run-eval --round 1 \\
        --host 155.98.36.13 --port 9000

    # 5. Inspect decision gate.
    cat exp/verdict_factor_judge/data/phase4/r1_alpha/decision_gate.json | jq .next_args_suggestion
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import math
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Optional

from exp.verdict_factor_judge.generate_yamls import write_yaml
from exp.verdict_factor_judge.phase3_threshold_solver import (
    derive_thresholds,
    load_per_key_finite_history,
    reconstruct_scores,
)
from exp.verdict_factor_judge.phase4_spec import (
    ANCHOR_SR,
    LOCKED_CELLS,
    R1_ALPHAS,
    R2_OFFLINE_PATTERNS,
    R3_ONLINE_PATTERNS,
    R4_WINDOW_PATTERNS,
    RECIPES_PHASE4,
    build_phase4_eval_yaml,
    build_phase4_warmup_yaml,
    cell_id_r1,
    cell_id_r2,
    cell_id_r3,
    cell_id_r4,
    generate_r1_weights,
    generate_r2_weights,
    generate_r3_weights,
    generate_r4_weights,
    recipe_to_solver_recipe,
    warmup_eval_yaml_id,
    warmup_yaml_id,
)
from exp.verdict_factor_judge.run_phase import (
    _aggregate_sr_from_episode_json,
    _build_libero_argv,
    _summarize_per_step_log,
)
from exp.verdict_factor_judge.run_phase3 import (
    _save_dump_jsonl,
    _load_done_yaml_ids,
)

try:
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
except ImportError:  # tests import this module without the runtime client
    WebsocketClientPolicy = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Args (argparse — phase 4 needs the per-recipe mapping flags which
# tyro's positional-style parsing handles less cleanly)
# ----------------------------------------------------------------------


VALID_MODES = ("emit-warmup-yaml", "run-warmup", "emit-eval-yamls", "run-eval")
WARM_COST = 0.75      # warm_start_t=0.5 -> cost = 1 - 0.5 * (1 - 0.5)


@dataclasses.dataclass
class Args:
    mode: str
    round: int
    alpha_star: Optional[str] = None
    offline_pattern: Optional[str] = None
    online_pattern: Optional[str] = None
    recipe: Optional[str] = None
    cell_ids: tuple[str, ...] = ()
    host: str = "155.98.36.13"
    port: int = 9000
    cfg_id: str = "spatial16_w8_d4"
    task_suite: str = "libero_spatial"
    num_workers: int = 5
    warmup_trials: int = 2
    eval_trials: int = 10
    task_ids: tuple[int, ...] = ()
    per_step_log_dir: str = ""
    summary_out: str = ""
    episode_results_dir: str = ""
    warmup_yaml_dir: str = ""
    warmup_jsonl_dir: str = ""
    eval_yaml_dir: str = ""
    init_states_dir: str = ""
    episode_filter: str = ""
    cuda_visible_devices: str = ""
    conda_env: str = ""


def _parse_args(argv: Optional[list[str]] = None) -> Args:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, choices=VALID_MODES)
    p.add_argument("--round", type=int, required=True, choices=[1, 2, 3, 4])
    p.add_argument("--alpha-star", type=str, default=None)
    p.add_argument("--offline-pattern", type=str, default=None)
    p.add_argument("--online-pattern", type=str, default=None)
    p.add_argument("--recipe", type=str, default=None)
    p.add_argument(
        "--cell-ids", nargs="*", default=[],
        help=(
            "Optional yaml_id substring allowlist for sharding cells across "
            "multiple parallel run-eval invocations (e.g. 6 GPU servers split "
            "the 14-cell R1 sweep). A cell is kept iff any of its substrings "
            "appears in the cell yaml_id. Empty = no filter."
        ),
    )
    # Both --host/--port (phase3 spelling) and --serve-host/--serve-port
    # (plan §5 / earlier docs) are accepted; the latter is an alias for
    # operator parity with the plan's published commands.
    p.add_argument(
        "--host", "--serve-host", dest="host",
        type=str, default="155.98.36.13",
    )
    p.add_argument(
        "--port", "--serve-port", dest="port",
        type=int, default=9000,
    )
    p.add_argument("--cfg-id", type=str, default="spatial16_w8_d4")
    p.add_argument("--task-suite", type=str, default="libero_spatial")
    p.add_argument("--num-workers", type=int, default=5)
    p.add_argument("--warmup-trials", type=int, default=2)
    p.add_argument("--eval-trials", type=int, default=10)
    p.add_argument("--task-ids", type=str, default="")
    p.add_argument("--per-step-log-dir", type=str, default="")
    p.add_argument("--summary-out", type=str, default="")
    p.add_argument("--episode-results-dir", type=str, default="")
    p.add_argument("--warmup-yaml-dir", type=str, default="")
    p.add_argument("--warmup-jsonl-dir", type=str, default="")
    p.add_argument("--eval-yaml-dir", type=str, default="")
    p.add_argument("--init-states-dir", type=str, default="")
    p.add_argument("--episode-filter", type=str, default="")
    p.add_argument("--cuda-visible-devices", type=str, default="")
    p.add_argument("--conda-env", type=str, default="")
    a = p.parse_args(argv)
    task_ids: tuple[int, ...] = tuple()
    if a.task_ids:
        task_ids = tuple(int(x) for x in a.task_ids.split(","))
    return Args(
        mode=a.mode, round=a.round,
        alpha_star=a.alpha_star, offline_pattern=a.offline_pattern,
        online_pattern=a.online_pattern, recipe=a.recipe,
        cell_ids=tuple(a.cell_ids or ()),
        host=a.host, port=a.port, cfg_id=a.cfg_id,
        task_suite=a.task_suite, num_workers=a.num_workers,
        warmup_trials=a.warmup_trials, eval_trials=a.eval_trials,
        task_ids=task_ids,
        per_step_log_dir=a.per_step_log_dir,
        summary_out=a.summary_out,
        episode_results_dir=a.episode_results_dir,
        warmup_yaml_dir=a.warmup_yaml_dir,
        warmup_jsonl_dir=a.warmup_jsonl_dir,
        eval_yaml_dir=a.eval_yaml_dir,
        init_states_dir=a.init_states_dir,
        episode_filter=a.episode_filter,
        cuda_visible_devices=a.cuda_visible_devices,
        conda_env=a.conda_env,
    )


# ----------------------------------------------------------------------
# Per-recipe mapping parser (CLI form: "p1=0.4,p2=0.6")
# ----------------------------------------------------------------------


def _try_float(s: str) -> Any:
    try:
        return float(s)
    except (TypeError, ValueError):
        return s


def parse_per_recipe_map(
    raw: Optional[str],
    *,
    valid_values: set,
    recipes: Iterable[str],
) -> dict[str, Any]:
    """Parse "p1_state_fut_online_act=jerk-heavy,p2_action_fut_online_act=uniform"
    into a dict {rid: value}, validating recipe ids and value membership.

    Returns {} when raw is None or empty (used for round 1 where no
    mapping is needed).
    """
    if raw is None or raw == "":
        return {}
    recipes_set = set(recipes)
    out: dict[str, Any] = {}
    for kv in raw.split(","):
        kv = kv.strip()
        if not kv:
            continue
        k, _, v = kv.partition("=")
        k = k.strip()
        v = v.strip()
        if k not in recipes_set:
            raise ValueError(
                f"unknown recipe id {k!r}; valid: {sorted(recipes_set)}"
            )
        v_parsed = _try_float(v)
        if v_parsed not in valid_values and v not in valid_values:
            raise ValueError(
                f"value {v!r} not in valid set {sorted(map(str, valid_values))}"
            )
        out[k] = v_parsed if v_parsed in valid_values else v
    return out


# ----------------------------------------------------------------------
# CLI invariants per mode (plan §4.2.1)
# ----------------------------------------------------------------------


def _validate_cli_invariants(args: Args) -> None:
    if args.mode == "emit-warmup-yaml":
        return
    if args.mode == "run-warmup":
        return
    # For emit-eval-yamls / run-eval, round-specific kwargs are required.
    if args.round >= 2 and not args.alpha_star:
        raise ValueError(
            f"--alpha-star required when round >= 2 (mode={args.mode}). "
            "Pass per-recipe mapping like 'p1_...=0.4,p2_...=0.6'."
        )
    if args.round >= 3 and not args.offline_pattern:
        raise ValueError(
            f"--offline-pattern required when round >= 3 (mode={args.mode})."
        )
    if args.round >= 4 and not args.online_pattern:
        raise ValueError(
            f"--online-pattern required when round == 4 (mode={args.mode})."
        )


def _apply_default_data_paths(args: Args) -> Args:
    """Fill in round-specific default data paths.

    Without this, ``--mode run-eval`` would default ``per_step_log_dir`` /
    ``episode_results_dir`` to empty, which makes ``_summarize_per_step_log``
    return zero verdict counts -> ``_compute_inf`` returns 0.0 for every
    cell -> R1 ``argmax(SR - 0.5*inf)`` collapses to ``argmax SR``. The
    decision gate would then silently disagree with plan §3.1 / §4.2.7.

    The defaults mirror the plan §4.2.2 layout under
    ``data/phase4/r{N}_*/`` so each round has its own per_step / episode
    sub-tree.
    """
    if args.mode != "run-eval":
        return args                                        # only run-eval needs these
    if not args.per_step_log_dir:
        args = dataclasses.replace(
            args, per_step_log_dir=str(_round_data_dir(args.round) / "per_step"),
        )
    if not args.episode_results_dir:
        args = dataclasses.replace(
            args, episode_results_dir=str(_round_data_dir(args.round) / "episode_results"),
        )
    return args


# ----------------------------------------------------------------------
# Cell construction (plan §4.2.4)
# ----------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Cell:
    yaml_id: str
    yaml_path: Path
    recipe_id: str
    round_id: int
    pattern_label: str
    weights: dict[str, float]
    fh_thr: Optional[float]
    ws_thr: Optional[float]
    fh_ratio: float
    ws_ratio: float
    alpha: float
    offline_pattern_name: Optional[str]
    online_pattern_name: Optional[str]
    window_pattern_name: Optional[str]


def _filter_recipes(args: Args) -> list[str]:
    """Honor --recipe restriction (used for R4 single-recipe trigger)."""
    if args.recipe:
        if args.recipe not in RECIPES_PHASE4:
            raise ValueError(
                f"--recipe {args.recipe!r} not in {list(RECIPES_PHASE4)}"
            )
        return [args.recipe]
    return list(RECIPES_PHASE4)


def _eval_yaml_dir(args: Args) -> Path:
    return Path(args.eval_yaml_dir) if args.eval_yaml_dir else (
        Path(f"exp/verdict_factor_judge/config/{args.cfg_id.split('_')[0]}/phase4/eval")
    )


def _warmup_yaml_dir(args: Args) -> Path:
    return Path(args.warmup_yaml_dir) if args.warmup_yaml_dir else (
        Path(f"exp/verdict_factor_judge/config/{args.cfg_id.split('_')[0]}/phase4/warmup")
    )


def _warmup_raw_dir() -> Path:
    return Path("exp/verdict_factor_judge/data/phase4/warmup_factor_raw")


def _round_data_dir(round_id: int) -> Path:
    name = {1: "r1_alpha", 2: "r2_offline_desc", 3: "r3_online_desc", 4: "r4_window"}[
        round_id
    ]
    return Path("exp/verdict_factor_judge/data/phase4") / name


def _summary_path(args: Args) -> Path:
    if args.summary_out:
        return Path(args.summary_out)
    return _round_data_dir(args.round) / "per_yaml_summary.jsonl"


def _build_cell_list(args: Args) -> list[Cell]:
    """Build the per-recipe cell list for the round, honoring per-recipe
    winner mappings parsed from --alpha-star / --offline-pattern /
    --online-pattern.
    """
    alpha_map = parse_per_recipe_map(
        args.alpha_star, valid_values=set(R1_ALPHAS), recipes=RECIPES_PHASE4,
    )
    off_map = parse_per_recipe_map(
        args.offline_pattern,
        valid_values=set(R2_OFFLINE_PATTERNS),
        recipes=RECIPES_PHASE4,
    )
    on_map = parse_per_recipe_map(
        args.online_pattern,
        valid_values=set(R3_ONLINE_PATTERNS),
        recipes=RECIPES_PHASE4,
    )
    eval_dir = _eval_yaml_dir(args)
    cells: list[Cell] = []
    for rid in _filter_recipes(args):
        fh, ws = LOCKED_CELLS[rid]
        if args.round == 1:
            for alpha in R1_ALPHAS:
                cid = cell_id_r1(args.cfg_id, rid, alpha)
                cells.append(Cell(
                    yaml_id=cid,
                    yaml_path=eval_dir / f"{cid}.yaml",
                    recipe_id=rid, round_id=1,
                    pattern_label=f"a{alpha:.1f}",
                    weights=generate_r1_weights(rid, alpha),
                    fh_thr=None, ws_thr=None,
                    fh_ratio=fh, ws_ratio=ws, alpha=alpha,
                    offline_pattern_name=None,
                    online_pattern_name=None,
                    window_pattern_name=None,
                ))
        elif args.round == 2:
            alpha_star = float(alpha_map[rid])
            for off_pat_name, off_pat in R2_OFFLINE_PATTERNS.items():
                cid = cell_id_r2(args.cfg_id, rid, alpha_star, off_pat_name)
                cells.append(Cell(
                    yaml_id=cid,
                    yaml_path=eval_dir / f"{cid}.yaml",
                    recipe_id=rid, round_id=2,
                    pattern_label=f"a{alpha_star:.1f}__off-{off_pat_name}",
                    weights=generate_r2_weights(rid, alpha_star, off_pat),
                    fh_thr=None, ws_thr=None,
                    fh_ratio=fh, ws_ratio=ws, alpha=alpha_star,
                    offline_pattern_name=off_pat_name,
                    online_pattern_name=None,
                    window_pattern_name=None,
                ))
        elif args.round == 3:
            alpha_star = float(alpha_map[rid])
            off_pat_name = off_map[rid]
            off_pat = R2_OFFLINE_PATTERNS[off_pat_name]
            for on_pat_name, on_pat in R3_ONLINE_PATTERNS.items():
                cid = cell_id_r3(
                    args.cfg_id, rid, alpha_star, off_pat_name, on_pat_name,
                )
                cells.append(Cell(
                    yaml_id=cid,
                    yaml_path=eval_dir / f"{cid}.yaml",
                    recipe_id=rid, round_id=3,
                    pattern_label=(
                        f"a{alpha_star:.1f}__off-{off_pat_name}__on-{on_pat_name}"
                    ),
                    weights=generate_r3_weights(rid, alpha_star, off_pat, on_pat),
                    fh_thr=None, ws_thr=None,
                    fh_ratio=fh, ws_ratio=ws, alpha=alpha_star,
                    offline_pattern_name=off_pat_name,
                    online_pattern_name=on_pat_name,
                    window_pattern_name=None,
                ))
        elif args.round == 4:
            alpha_star = float(alpha_map[rid])
            off_pat_name = off_map[rid]
            off_pat = R2_OFFLINE_PATTERNS[off_pat_name]
            on_pat_name = on_map[rid]
            on_pat = R3_ONLINE_PATTERNS[on_pat_name]
            for win_pat_name, win_pat in R4_WINDOW_PATTERNS.items():
                cid = cell_id_r4(
                    args.cfg_id, rid, alpha_star, off_pat_name,
                    on_pat_name, win_pat_name,
                )
                cells.append(Cell(
                    yaml_id=cid,
                    yaml_path=eval_dir / f"{cid}.yaml",
                    recipe_id=rid, round_id=4,
                    pattern_label=(
                        f"a{alpha_star:.1f}__off-{off_pat_name}"
                        f"__on-{on_pat_name}__win-{win_pat_name}"
                    ),
                    weights=generate_r4_weights(
                        rid, alpha_star, off_pat, on_pat, win_pat,
                    ),
                    fh_thr=None, ws_thr=None,
                    fh_ratio=fh, ws_ratio=ws, alpha=alpha_star,
                    offline_pattern_name=off_pat_name,
                    online_pattern_name=on_pat_name,
                    window_pattern_name=win_pat_name,
                ))
    if args.cell_ids:
        # Substring allowlist for multi-server sharding. A cell survives
        # iff any of args.cell_ids appears in its yaml_id (OR semantics).
        allow = tuple(args.cell_ids)
        cells = [c for c in cells if any(sub in c.yaml_id for sub in allow)]
    return cells


# ----------------------------------------------------------------------
# Solver call (plan §4.2.6) — pure offline, no server
# ----------------------------------------------------------------------


def _solve_thresholds(cell: Cell) -> tuple[float, float]:
    """For a cell's weight vector, derive (fh_thr, ws_thr) from the cached
    warmup factor_raw under the per-recipe locked (fh_ratio, ws_ratio).
    """
    raw_path = _warmup_raw_dir() / f"{cell.recipe_id}.jsonl"
    if not raw_path.exists():
        raise FileNotFoundError(
            f"warmup factor_raw not found: {raw_path}. "
            "Run `--mode run-warmup --round 1` first."
        )
    solver_recipe = recipe_to_solver_recipe(cell.recipe_id)
    scores = reconstruct_scores(
        raw_path, solver_recipe, composer_weights=cell.weights,
    )
    fh_thr, ws_thr = derive_thresholds(
        scores, fh_ratio=cell.fh_ratio, ws_ratio=cell.ws_ratio,
    )
    return fh_thr, ws_thr


# ----------------------------------------------------------------------
# Mode 1: emit-warmup-yaml
# ----------------------------------------------------------------------


def _mode_emit_warmup_yaml(args: Args) -> None:
    out_dir = _warmup_yaml_dir(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    for rid in _filter_recipes(args):
        wid = warmup_yaml_id(args.cfg_id, rid)
        yaml_dict = build_phase4_warmup_yaml(args.cfg_id, rid)
        path = out_dir / f"{wid}.yaml"
        write_yaml(path, yaml_dict)
        logger.info("[emit-warmup-yaml] %s -> %s", rid, path)


# ----------------------------------------------------------------------
# Mode 2: run-warmup
# ----------------------------------------------------------------------


def _extract_finite_factor_raw(
    src_per_step_jsonl: Path,
    dst: Path,
    declared_keys: list[str],
) -> None:
    """Project per-step jsonl rows down to ``{factor_raw: {k: v}}`` finite-only
    rows in row order. Same shape as phase3's warmup factor_raw cache.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src_per_step_jsonl.open() as src, dst.open("w") as out:
        for line in src:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            raw = row.get("factor_raw") or row.get("factor_outputs", {}).get("raw") or {}
            kept = {k: float(raw[k]) for k in declared_keys if k in raw}
            if not kept:
                continue
            if not any(math.isfinite(v) for v in kept.values()):
                continue
            out.write(json.dumps({"factor_raw": kept}) + "\n")


def _run_warmup_for_recipe(ctl, recipe_id: str, args: Args) -> None:
    """Run warmup yaml on the server and extract finite-only factor_raw to disk.
    Caller (the run-warmup mode) verified the warmup yaml file exists.
    """
    wid = warmup_yaml_id(args.cfg_id, recipe_id)
    wpath = _warmup_yaml_dir(args) / f"{wid}.yaml"
    raw_dst = _warmup_raw_dir() / f"{recipe_id}.jsonl"
    if raw_dst.exists() and raw_dst.stat().st_size > 0:
        logger.info("[run-warmup] %s: factor_raw cached -> %s, skipping",
                    recipe_id, raw_dst)
        return

    logger.info("[run-warmup] %s: load + run %d trials/task",
                recipe_id, args.warmup_trials)
    ctl.load_cache_config(
        yaml_content=wpath.read_text(encoding="utf-8"),
        yaml_id=wid,
    )
    cmd, env = _build_libero_argv(
        args=args, yaml_id=wid, phase="warmup",
        num_trials_per_task=args.warmup_trials,
    )
    subprocess.run(cmd, env=env, check=True)

    # Server-side dump fetch: phase4 reuses phase3's _save_dump_jsonl semantics
    # (write the raw bytes to a sibling jsonl), then extract finite-only
    # factor_raw rows for solver consumption.
    raw_jsonl_dir = (
        Path(args.warmup_jsonl_dir) if args.warmup_jsonl_dir
        else Path("exp/verdict_factor_judge/data/phase4/warmup_dump")
    )
    raw_jsonl = raw_jsonl_dir / f"{wid}.jsonl"
    content = ctl.fetch_dump(wid)
    _save_dump_jsonl(content, raw_jsonl)
    declared_keys = list(RECIPES_PHASE4[recipe_id]["declared_keys"])
    _extract_finite_factor_raw(raw_jsonl, raw_dst, declared_keys)
    logger.info("[run-warmup] %s: factor_raw -> %s", recipe_id, raw_dst)


def _mode_run_warmup(args: Args) -> None:
    if WebsocketClientPolicy is None:
        raise RuntimeError("openpi_client is not installed; cannot run warmup")
    out_dir = _warmup_yaml_dir(args)
    for rid in _filter_recipes(args):
        wpath = out_dir / f"{warmup_yaml_id(args.cfg_id, rid)}.yaml"
        if not wpath.exists():
            raise FileNotFoundError(
                f"warmup yaml missing: {wpath}. "
                "Run `--mode emit-warmup-yaml` first."
            )
    for rid in _filter_recipes(args):
        with WebsocketClientPolicy(host=args.host, port=args.port) as ctl:
            _run_warmup_for_recipe(ctl, rid, args)


# ----------------------------------------------------------------------
# Mode 3: emit-eval-yamls
# ----------------------------------------------------------------------


def _mode_emit_eval_yamls(args: Args) -> None:
    cells = _build_cell_list(args)
    eval_dir = _eval_yaml_dir(args)
    eval_dir.mkdir(parents=True, exist_ok=True)
    for cell in cells:
        fh_thr, ws_thr = _solve_thresholds(cell)
        cell = replace(cell, fh_thr=fh_thr, ws_thr=ws_thr)
        yaml_dict = build_phase4_eval_yaml(
            cfg_id=args.cfg_id, recipe_id=cell.recipe_id,
            fh_thr=fh_thr, ws_thr=ws_thr,
            fh_ratio=cell.fh_ratio, ws_ratio=cell.ws_ratio,
            composer_weights=cell.weights,
        )
        write_yaml(cell.yaml_path, yaml_dict)
        logger.info(
            "[emit-eval-yamls] r%d %s %s fh_thr=%.4f ws_thr=%.4f -> %s",
            cell.round_id, cell.recipe_id, cell.pattern_label,
            fh_thr, ws_thr, cell.yaml_path,
        )


# ----------------------------------------------------------------------
# Mode 4: run-eval (plan §4.2.3b — preload BEFORE load_cache_config)
# ----------------------------------------------------------------------


def _resolve_episode_results_dir(args: Args) -> Optional[Path]:
    if args.episode_results_dir:
        return Path(args.episode_results_dir)
    if args.per_step_log_dir:
        return Path(args.per_step_log_dir).parent / "episode_results"
    return _round_data_dir(args.round) / "episode_results"


def _read_thresholds_from_yaml(yaml_path: Path) -> tuple[float, float]:
    """Extract (fh_thr, ws_thr) baked into a phase4 eval yaml.

    ``--mode emit-eval-yamls`` solves thresholds from the cached warmup
    factor_raw and writes them into ``composer.tier_thresholds``. The
    ``--mode run-eval`` path then re-builds Cell objects through
    ``_build_cell_list``, which initializes ``cell.fh_thr`` /
    ``cell.ws_thr`` to ``None`` (no solver call). Without this helper,
    the per-yaml summary row would carry ``None`` for both threshold
    fields even though the actual run used the correct values from the
    yaml file. Reading the yaml here back-fills the summary so downstream
    analysis (Pareto plots, decision-gate diagnostics, R2/R3 reasoning)
    has the threshold trail intact.
    """
    import yaml as _yaml_mod
    cfg = _yaml_mod.safe_load(yaml_path.read_text(encoding="utf-8"))
    tt = cfg["checkpoints"]["cp1"]["judge"]["composer"]["tier_thresholds"]
    return float(tt["full_hit"]), float(tt["warm_start"])


def _run_one_cell(
    ctl, cell: Cell, args: Args, summary_path: Path,
) -> dict:
    """Run a single phase4 eval cell. Order is:
    1) preload_normalizer_buffer(cell.yaml_id, buffer)   (phase3 invariant)
    2) load_cache_config(eval_yaml_text, yaml_id)
    3) launch libero workers
    4) summarize + append summary row (incl. thr back-filled from yaml)
    """
    # Back-fill thresholds from the eval yaml so the summary row records
    # the same values that actually drove server-side gating. Without
    # this, `cell.fh_thr` / `cell.ws_thr` are still the None placeholders
    # set by `_build_cell_list` (the solver runs only in emit-eval-yamls
    # mode, not run-eval).
    if cell.fh_thr is None or cell.ws_thr is None:
        fh_thr, ws_thr = _read_thresholds_from_yaml(cell.yaml_path)
        cell = replace(cell, fh_thr=fh_thr, ws_thr=ws_thr)

    raw_path = _warmup_raw_dir() / f"{cell.recipe_id}.jsonl"
    declared_keys = list(RECIPES_PHASE4[cell.recipe_id]["declared_keys"])
    buffer = load_per_key_finite_history(raw_path, declared_keys)

    ack = ctl.preload_normalizer_buffer(cell.yaml_id, buffer)
    n_warmup_keys = ack.get("n_keys", len(buffer)) if isinstance(ack, dict) else len(buffer)
    ctl.load_cache_config(
        yaml_content=cell.yaml_path.read_text(encoding="utf-8"),
        yaml_id=cell.yaml_id,
    )

    episode_results_dir = _resolve_episode_results_dir(args)
    episode_results_path: Optional[Path] = None
    if episode_results_dir is not None:
        episode_results_dir.mkdir(parents=True, exist_ok=True)
        episode_results_path = episode_results_dir / f"{cell.yaml_id}.json"

    cmd, env = _build_libero_argv(
        args=args, yaml_id=cell.yaml_id, phase="eval",
        num_trials_per_task=args.eval_trials,
        episode_results_path=episode_results_path,
    )
    subprocess.run(cmd, env=env, check=True)

    counts = _summarize_per_step_log(args.per_step_log_dir, cell.yaml_id)
    success_rate = (
        _aggregate_sr_from_episode_json(episode_results_path)
        if episode_results_path is not None else None
    )
    summary = {
        "yaml_id": cell.yaml_id,
        "recipe_id": cell.recipe_id,
        "round_id": cell.round_id,
        "pattern_label": cell.pattern_label,
        "alpha": cell.alpha,
        "offline_pattern": cell.offline_pattern_name,
        "online_pattern": cell.online_pattern_name,
        "window_pattern": cell.window_pattern_name,
        "fh_ratio": cell.fh_ratio,
        "ws_ratio": cell.ws_ratio,
        "fh_thr": cell.fh_thr,
        "ws_thr": cell.ws_thr,
        "warmup_yaml_id": warmup_yaml_id(args.cfg_id, cell.recipe_id),
        "n_warmup_buffer_keys": n_warmup_keys,
        "success_rate": success_rate,
        **counts,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("a") as fh:
        fh.write(json.dumps(summary) + "\n")
    return summary


def _mode_run_eval(args: Args) -> None:
    if WebsocketClientPolicy is None:
        raise RuntimeError("openpi_client is not installed; cannot run eval")
    cells = _build_cell_list(args)
    for cell in cells:
        if not cell.yaml_path.exists():
            raise FileNotFoundError(
                f"eval yaml missing: {cell.yaml_path}. "
                "Run `--mode emit-eval-yamls` first for this round."
            )

    summary_path = _summary_path(args)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    done_ids = _load_done_yaml_ids(summary_path) if summary_path.exists() else set()
    pending = [c for c in cells if c.yaml_id not in done_ids]
    logger.info("[run-eval] r%d cells: %d total, %d pending (resume hits %d)",
                args.round, len(cells), len(pending), len(cells) - len(pending))

    for cell in pending:
        with WebsocketClientPolicy(host=args.host, port=args.port) as ctl:
            _run_one_cell(ctl, cell, args, summary_path)

    _dump_decision_gate_table(args.round, summary_path)


# ----------------------------------------------------------------------
# Decision gate (plan §4.2.7) — round-specific selection rules
# ----------------------------------------------------------------------


def _compute_inf(row: dict) -> float:
    n = row.get("n_eval_verdicts") or 0
    if not n:
        return 0.0
    return (
        row.get("n_full_hit", 0) * 0.0
        + row.get("n_warm_start", 0) * WARM_COST
        + row.get("n_miss", 0) * 1.0
    ) / n


def _r1_winner_per_recipe(rows: list[dict]) -> tuple[dict, dict]:
    """Return (winners_by_recipe, decisions_by_recipe). G_R1 rule."""
    by_recipe: dict[str, list[dict]] = {}
    for r in rows:
        by_recipe.setdefault(r["recipe_id"], []).append(r)
    winners, decisions = {}, {}
    for rid, cells in by_recipe.items():
        scored = sorted(
            cells,
            key=lambda r: (
                (r.get("success_rate") or 0.0) - 0.5 * _compute_inf(r)
            ),
            reverse=True,
        )
        w = scored[0]
        winners[rid] = w
        anchor = ANCHOR_SR.get(rid, 0.95)
        sr = w.get("success_rate") or 0.0
        cont = sr >= anchor - 0.02
        decisions[rid] = {
            "continue": cont,
            "reason": (
                f"argmax SR-0.5*inf -> alpha={w.get('alpha')} "
                f"SR={sr:.3f} vs anchor {anchor:.3f}"
            ),
        }
    return winners, decisions


def _r_argmax_with_uniform_baseline(
    rows: list[dict], uniform_label_suffix: str,
) -> tuple[dict, dict, dict]:
    """G_R2 / G_R3 selection rule:
    1. winner = argmax SR per recipe.
    2. uniform_baseline = the row whose pattern_label endswith
       ``uniform_label_suffix`` (e.g. "off-uniform" for R2 or
       "on-uniform" for R3).
    3. If winner.SR - uniform.SR > 2pp: keep argmax winner.
       Else: force the uniform row as winner (R2/R3 §3.2 clause).
    """
    by_recipe: dict[str, list[dict]] = {}
    for r in rows:
        by_recipe.setdefault(r["recipe_id"], []).append(r)
    winners, baselines, decisions = {}, {}, {}
    for rid, cells in by_recipe.items():
        try:
            uniform_row = next(
                c for c in cells
                if (c.get("pattern_label") or "").endswith(uniform_label_suffix)
            )
        except StopIteration:
            uniform_row = max(cells, key=lambda c: (c.get("success_rate") or 0.0))
        baselines[rid] = uniform_row
        argmax = max(cells, key=lambda c: (c.get("success_rate") or 0.0))
        sr_argmax = argmax.get("success_rate") or 0.0
        sr_uniform = uniform_row.get("success_rate") or 0.0
        delta = sr_argmax - sr_uniform
        if delta > 0.02:
            winners[rid] = argmax
            decisions[rid] = {
                "continue": True,
                "reason": (
                    f"argmax pat={argmax.get('pattern_label')} SR={sr_argmax:.3f} "
                    f"> uniform SR={sr_uniform:.3f} + 2pp"
                ),
            }
        else:
            winners[rid] = uniform_row
            decisions[rid] = {
                "continue": True,
                "reason": (
                    f"argmax SR delta {delta * 100:+.1f}pp <= 2pp; "
                    "forcing pattern*=uniform"
                ),
            }
    return winners, baselines, decisions


def _stringify_per_recipe(d: dict[str, Any], extractor) -> Optional[str]:
    parts = []
    for rid, row in d.items():
        v = extractor(row)
        if v is None:
            continue
        parts.append(f"{rid}={v}")
    return ",".join(parts) if parts else None


def _populate_recipe_restriction_fields(
    out: dict[str, Any],
    active_winners: dict[str, dict],
    *,
    next_round: int,
) -> None:
    """Add ``active_recipes`` (full list of continued recipes) and, when
    only one recipe is active, a ``recipe`` field plus a ready-to-paste
    ``cli_command`` line. R4 is conditional on per-recipe triggers, so
    single-recipe trips need an explicit ``--recipe <rid>`` argument.

    Mutates ``out["next_args_suggestion"]`` in place.
    """
    suggestion = out["next_args_suggestion"]
    suggestion["active_recipes"] = sorted(active_winners.keys())
    if len(active_winners) == 1:
        only_rid = next(iter(active_winners))
        suggestion["recipe"] = only_rid
    else:
        suggestion["recipe"] = None

    cli_parts: list[str] = [
        f"--mode emit-eval-yamls --round {next_round}",
    ]
    if suggestion.get("alpha-star"):
        cli_parts.append(f"--alpha-star '{suggestion['alpha-star']}'")
    if suggestion.get("offline-pattern"):
        cli_parts.append(f"--offline-pattern '{suggestion['offline-pattern']}'")
    if suggestion.get("online-pattern"):
        cli_parts.append(f"--online-pattern '{suggestion['online-pattern']}'")
    if suggestion.get("recipe"):
        cli_parts.append(f"--recipe {suggestion['recipe']}")
    suggestion["cli_command"] = " ".join(cli_parts) if active_winners else None


def _dump_decision_gate_table(round_id: int, summary_path: Path) -> dict:
    """Write decision_gate.json next to the round's per_yaml_summary.jsonl,
    with a round-specific selection rule that matches §3 of the plan.

    Output schema:
        {
            "round": int,
            "rule": str,
            "winners": {rid: row},
            "baselines": {rid: row} | {},
            "next_args_suggestion": {
                "alpha-star": str | None,
                "offline-pattern": str | None,
                "online-pattern": str | None,
            },
            "trigger_decisions": {rid: {"continue": bool, "reason": str}},
            "all": [row, ...],
        }
    """
    rows = []
    if summary_path.exists():
        for line in summary_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    # Fail-fast guard: if every row has n_eval_verdicts=0 (and rows exist),
    # the per_step log dir was empty/missing during run-eval. Without verdict
    # counts, _compute_inf returns 0.0 for every cell and R1's
    # argmax(SR - 0.5*inf) collapses to argmax(SR). Surface this so the
    # operator does not consume a silently-broken decision gate.
    if rows and all((r.get("n_eval_verdicts") or 0) == 0 for r in rows):
        raise RuntimeError(
            f"decision_gate({round_id}): every summary row has n_eval_verdicts=0. "
            f"This usually means --per-step-log-dir was empty during run-eval, "
            f"so verdict counts never made it to the summary. "
            f"Re-run run-eval with --per-step-log-dir set (or unset to use the "
            f"round-specific default in data/phase4/r{round_id}_*/per_step/)."
        )

    out: dict[str, Any] = {
        "round": round_id, "rule": "", "winners": {}, "baselines": {},
        "next_args_suggestion": {
            "alpha-star": None, "offline-pattern": None, "online-pattern": None,
        },
        "trigger_decisions": {}, "all": rows,
    }

    # next_args_suggestion is gate-safe: only recipes whose
    # trigger_decisions[rid]["continue"] == True appear in the suggested
    # CLI mappings. Aborted recipes (R1 anchor fail) and R3-only-skip
    # recipes are excluded so copy-pasting next_args_suggestion never
    # advances a recipe past its decision gate. In single-recipe-active
    # cases we also emit ``recipe`` and ``cli_command`` to make the next
    # invocation a complete round trip (plan §3.1 / §3.3).

    if round_id == 1:
        out["rule"] = (
            "argmax SR - 0.5*inf per recipe; "
            "continue iff SR(alpha*) >= recipe_anchor_SR - 2pp"
        )
        winners, decisions = _r1_winner_per_recipe(rows)
        out["winners"] = winners
        out["trigger_decisions"] = decisions
        active_winners = {
            rid: w for rid, w in winners.items() if decisions[rid]["continue"]
        }
        out["next_args_suggestion"]["alpha-star"] = _stringify_per_recipe(
            active_winners,
            extractor=lambda r: f"{r.get('alpha'):.1f}" if r.get('alpha') is not None else None,
        )
        _populate_recipe_restriction_fields(out, active_winners, next_round=2)

    elif round_id == 2:
        out["rule"] = (
            "argmax SR per recipe; if argmax SR - uniform SR <= 2pp, "
            "force pattern*=uniform"
        )
        winners, baselines, decisions = _r_argmax_with_uniform_baseline(
            rows, uniform_label_suffix="off-uniform",
        )
        out["winners"] = winners
        out["baselines"] = baselines
        out["trigger_decisions"] = decisions
        active_winners = {
            rid: w for rid, w in winners.items() if decisions[rid]["continue"]
        }
        out["next_args_suggestion"]["alpha-star"] = _stringify_per_recipe(
            active_winners,
            extractor=lambda r: f"{r.get('alpha'):.1f}" if r.get('alpha') is not None else None,
        )
        out["next_args_suggestion"]["offline-pattern"] = _stringify_per_recipe(
            active_winners, extractor=lambda r: r.get("offline_pattern"),
        )
        _populate_recipe_restriction_fields(out, active_winners, next_round=3)

    elif round_id == 3:
        out["rule"] = (
            "argmax SR among 5 online patterns per recipe; "
            "R4 trigger iff argmax SR > uniform_online SR + 2pp"
        )
        winners, baselines, decisions = _r_argmax_with_uniform_baseline(
            rows, uniform_label_suffix="on-uniform",
        )
        out["winners"] = winners
        out["baselines"] = baselines
        # Override "continue" semantics for R3: it represents R4 trigger.
        for rid, w in winners.items():
            base = baselines[rid]
            sr_w = w.get("success_rate") or 0.0
            sr_base = base.get("success_rate") or 0.0
            delta = sr_w - sr_base
            decisions[rid] = {
                "continue": delta > 0.02,
                "reason": (
                    f"R3 winner SR={sr_w:.3f}; uniform SR={sr_base:.3f}; "
                    f"delta {delta * 100:+.1f}pp"
                ),
            }
        out["trigger_decisions"] = decisions
        # R4 is conditional: only recipes that triggered get the full
        # alpha-star/offline-pattern/online-pattern triplet, plus a
        # `--recipe` restriction so single-recipe R4 runs end-to-end.
        triggered_winners = {
            rid: w for rid, w in winners.items() if decisions[rid]["continue"]
        }
        out["next_args_suggestion"]["alpha-star"] = _stringify_per_recipe(
            triggered_winners,
            extractor=lambda r: f"{r.get('alpha'):.1f}" if r.get('alpha') is not None else None,
        )
        out["next_args_suggestion"]["offline-pattern"] = _stringify_per_recipe(
            triggered_winners, extractor=lambda r: r.get("offline_pattern"),
        )
        out["next_args_suggestion"]["online-pattern"] = _stringify_per_recipe(
            triggered_winners, extractor=lambda r: r.get("online_pattern"),
        )
        _populate_recipe_restriction_fields(out, triggered_winners, next_round=4)

    elif round_id == 4:
        out["rule"] = "R4 final report (no further round)"
        winners, decisions = _r1_winner_per_recipe(rows)    # reuse argmax-with-anchor
        out["winners"] = winners
        out["trigger_decisions"] = decisions

    out_path = summary_path.parent / "decision_gate.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    logger.info("[decision_gate] r%d -> %s", round_id, out_path)
    return out


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------


def main(args: Optional[Args] = None) -> int:
    if args is None:
        args = _parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    _validate_cli_invariants(args)
    args = _apply_default_data_paths(args)
    if args.mode == "emit-warmup-yaml":
        _mode_emit_warmup_yaml(args)
    elif args.mode == "run-warmup":
        _mode_run_warmup(args)
    elif args.mode == "emit-eval-yamls":
        _mode_emit_eval_yamls(args)
    elif args.mode == "run-eval":
        _mode_run_eval(args)
    else:
        raise ValueError(f"unsupported mode: {args.mode!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
