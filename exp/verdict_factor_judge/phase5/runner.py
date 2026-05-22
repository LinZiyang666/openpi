"""Phase 5 runner — 4-mode pipeline for the systematic 240-cell sweep.

Modes (mirror phase4 structure but phase5-local — see plan §3.2):

    emit-warmup-yamls   (no server)  write per-cell / per-shared-group warmup yaml
    run-warmup          (server)     run warmup, dump factor_raw, extract finite-only
    emit-eval-yamls     (no server)  per-cell solve thresholds + write eval yaml
    run-eval            (server)     preload buffer -> load eval -> libero workers

Phase5-local guarantees (plan §3.3 / §5.2 R-1..R-5):

    - structure-mirrors phase4.runner._run_one_cell preload-before-load order
    - does NOT call phase4.runner._run_one_cell directly
    - does NOT read RECIPES_PHASE4 (cell.declared_keys / cell.weights / cell.factors
      come from the Cell object alone — see R-5 monkeypatch test)
    - threshold derivation: reconstruct_scores(..., composer_weights=cell.weights)
      + derive_thresholds(...) — NOT solve_recipe
    - raw source resolution: G5 dual-source (phase3 g6 + phase4 p1/p2),
      G1-G4 own dir (phase5_systematic/warmup_factor_raw/)
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

from exp.verdict_factor_judge.common.generate_yamls import write_yaml
from exp.verdict_factor_judge.common.run_phase import (
    _aggregate_sr_from_episode_json,
    _build_libero_argv,
    _summarize_per_step_log,
)
from exp.verdict_factor_judge.phase3.runner import (
    _save_dump_jsonl,
    _load_done_yaml_ids,
)
from exp.verdict_factor_judge.phase3.threshold_solver import (
    derive_thresholds,
    load_per_key_finite_history,
    reconstruct_scores,
)
from exp.verdict_factor_judge.phase5.spec import (
    CFG_ID_DEFAULT,
    Cell,
    allocate_to_servers,
    build_eval_yaml_for_cell,
    build_warmup_yaml_for_cell,
    cell_to_solver_recipe,
    generate_all_cells,
    warmup_raw_path_for_cell,
)

try:
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
except ImportError:
    WebsocketClientPolicy = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


VALID_MODES = ("emit-warmup-yamls", "run-warmup", "emit-eval-yamls", "run-eval")
VALID_GROUPS = ("g1", "g2", "g3", "g4", "g5")
WARM_COST = 0.75
PHASE5_DATA_ROOT = Path("exp/verdict_factor_judge/data/phase5_systematic")


# ----------------------------------------------------------------------
# Args + CLI
# ----------------------------------------------------------------------


@dataclasses.dataclass
class Args:
    mode: str
    groups: tuple[str, ...] = VALID_GROUPS
    cell_ids: tuple[str, ...] = ()
    cfg_id: str = CFG_ID_DEFAULT
    host: str = "155.98.36.32"
    port: int = 9000
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
    phase3_warmup_dir: str = ""
    phase4_warmup_raw_dir: str = ""
    eval_yaml_dir: str = ""
    init_states_dir: str = ""
    episode_filter: str = ""
    cuda_visible_devices: str = ""
    conda_env: str = ""
    preload_pkl_override: str = ""


def _parse_args(argv: Optional[list[str]] = None) -> Args:
    p = argparse.ArgumentParser(prog="phase5_runner")
    p.add_argument("--mode", required=True, choices=VALID_MODES)
    p.add_argument(
        "--groups", type=str, default=",".join(VALID_GROUPS),
        help="Comma-separated subset of g1,g2,g3,g4,g5 (default = all).",
    )
    p.add_argument(
        "--cell-ids", nargs="*", default=[],
        help="Substring allowlist for cell yaml_id. Empty = no filter.",
    )
    p.add_argument("--cfg-id", default=CFG_ID_DEFAULT)
    p.add_argument("--host", "--serve-host", dest="host", default="155.98.36.32")
    p.add_argument("--port", "--serve-port", dest="port", type=int, default=9000)
    p.add_argument("--task-suite", default="libero_spatial")
    p.add_argument("--num-workers", type=int, default=5)
    p.add_argument("--warmup-trials", type=int, default=2)
    p.add_argument("--eval-trials", type=int, default=10)
    p.add_argument("--task-ids", default="")
    p.add_argument("--per-step-log-dir", default="")
    p.add_argument("--summary-out", default="")
    p.add_argument("--episode-results-dir", default="")
    p.add_argument("--warmup-yaml-dir", default="")
    p.add_argument("--warmup-jsonl-dir", default="")
    p.add_argument(
        "--phase3-warmup-dir", default="",
        help="Path to phase3 warmup factor_raw jsonls (G5 g6 raw source).",
    )
    p.add_argument(
        "--phase4-warmup-raw-dir", default="",
        help="Path to phase4 warmup factor_raw jsonls (G5 p1/p2 raw source).",
    )
    p.add_argument("--eval-yaml-dir", default="")
    p.add_argument("--init-states-dir", default="")
    p.add_argument("--episode-filter", default="")
    p.add_argument("--cuda-visible-devices", default="")
    p.add_argument("--conda-env", default="")
    p.add_argument(
        "--preload-pkl-override", default="",
        help="Override backend.in_memory.preload_path in every emitted "
             "warmup/eval yaml (default: empty = use v2_spec CFG_SPECS "
             "default, currently libero_spatial pkl). Set this to a "
             "libero_10 pkl path when sweeping libero_10.",
    )
    a = p.parse_args(argv)

    groups = tuple(g.strip() for g in a.groups.split(",") if g.strip())
    for g in groups:
        if g not in VALID_GROUPS:
            raise SystemExit(f"--groups: unknown group {g!r} (valid: {VALID_GROUPS})")
    task_ids = tuple(int(x) for x in a.task_ids.split(",")) if a.task_ids else ()
    return Args(
        mode=a.mode, groups=groups, cell_ids=tuple(a.cell_ids),
        cfg_id=a.cfg_id, host=a.host, port=a.port,
        task_suite=a.task_suite, num_workers=a.num_workers,
        warmup_trials=a.warmup_trials, eval_trials=a.eval_trials,
        task_ids=task_ids,
        per_step_log_dir=a.per_step_log_dir, summary_out=a.summary_out,
        episode_results_dir=a.episode_results_dir,
        warmup_yaml_dir=a.warmup_yaml_dir, warmup_jsonl_dir=a.warmup_jsonl_dir,
        phase3_warmup_dir=a.phase3_warmup_dir,
        phase4_warmup_raw_dir=a.phase4_warmup_raw_dir,
        eval_yaml_dir=a.eval_yaml_dir, init_states_dir=a.init_states_dir,
        episode_filter=a.episode_filter,
        cuda_visible_devices=a.cuda_visible_devices, conda_env=a.conda_env,
        preload_pkl_override=a.preload_pkl_override,
    )


# ----------------------------------------------------------------------
# Path helpers
# ----------------------------------------------------------------------


def _cfg_short(cfg_id: str) -> str:
    return cfg_id.split("_")[0]


def _eval_yaml_dir(args: Args) -> Path:
    return Path(args.eval_yaml_dir) if args.eval_yaml_dir else (
        Path(f"exp/verdict_factor_judge/config/{_cfg_short(args.cfg_id)}/phase5/eval")
    )


def _warmup_yaml_dir(args: Args) -> Path:
    return Path(args.warmup_yaml_dir) if args.warmup_yaml_dir else (
        Path(f"exp/verdict_factor_judge/config/{_cfg_short(args.cfg_id)}/phase5/warmup")
    )


def _phase5_warmup_raw_dir(args: Args) -> Path:
    return Path(args.warmup_jsonl_dir) if args.warmup_jsonl_dir else (
        PHASE5_DATA_ROOT / "warmup_factor_raw"
    )


def _phase3_warmup_dir(args: Args) -> Path:
    return Path(args.phase3_warmup_dir) if args.phase3_warmup_dir else (
        Path("exp/verdict_factor_judge/data/phase3/warmup")
    )


def _phase4_warmup_raw_dir(args: Args) -> Path:
    return Path(args.phase4_warmup_raw_dir) if args.phase4_warmup_raw_dir else (
        Path("exp/verdict_factor_judge/data/phase4/warmup_factor_raw")
    )


def _summary_path(args: Args) -> Path:
    return Path(args.summary_out) if args.summary_out else (
        PHASE5_DATA_ROOT / "per_yaml_summary.jsonl"
    )


def _episode_results_dir(args: Args) -> Optional[Path]:
    return Path(args.episode_results_dir) if args.episode_results_dir else (
        PHASE5_DATA_ROOT / "episode_results"
    )


# ----------------------------------------------------------------------
# Cell list resolution (group + cell-ids filter)
# ----------------------------------------------------------------------


def _resolve_cells(args: Args) -> list[Cell]:
    """Generate full 240-cell list, then apply --groups + --cell-ids filters."""
    cells = generate_all_cells(cfg_id=args.cfg_id)
    if args.groups != VALID_GROUPS:
        cells = [c for c in cells if c.group in args.groups]
    if args.cell_ids:
        substrs = tuple(args.cell_ids)
        cells = [c for c in cells if any(s in c.yaml_id for s in substrs)]
    return cells


# ----------------------------------------------------------------------
# Threshold solver (phase4 pattern, phase5-local)
# ----------------------------------------------------------------------


def _solve_thresholds_phase5(cell: Cell, args: Args) -> tuple[float, float]:
    """Per-cell threshold derivation. Walks the path:

        raw jsonl -> reconstruct_scores(..., composer_weights=cell.weights)
                  -> derive_thresholds(scores, fh_ratio, ws_ratio)
    """
    raw_path = warmup_raw_path_for_cell(
        cell,
        phase5_warmup_dir=_phase5_warmup_raw_dir(args),
        phase3_warmup_dir=_phase3_warmup_dir(args),
        phase4_warmup_raw_dir=_phase4_warmup_raw_dir(args),
    )
    if not raw_path.exists():
        raise FileNotFoundError(
            f"warmup factor_raw not found for cell {cell.yaml_id!r}: {raw_path}. "
            "Run `--mode run-warmup` first (or pass --phase{3,4}-warmup-raw-dir for G5)."
        )
    solver_recipe = cell_to_solver_recipe(cell)
    scores = reconstruct_scores(
        raw_path, solver_recipe, composer_weights=cell.weights,
    )
    fh_thr, ws_thr = derive_thresholds(
        scores, fh_ratio=cell.fh_ratio, ws_ratio=cell.ws_ratio,
    )
    return fh_thr, ws_thr


# ----------------------------------------------------------------------
# Mode 1: emit-warmup-yamls
# ----------------------------------------------------------------------


def _mode_emit_warmup_yamls(args: Args) -> None:
    out_dir = _warmup_yaml_dir(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    cells = _resolve_cells(args)
    seen: set[str] = set()
    for cell in cells:
        if cell.group == "g5":
            continue   # G5 reuses phase3/phase4 warmup; no phase5 emission
        if cell.warmup_yaml_id in seen:
            continue
        seen.add(cell.warmup_yaml_id)
        yaml_dict = build_warmup_yaml_for_cell(
            args.cfg_id, cell,
            preload_pkl_override=args.preload_pkl_override or None,
        )
        path = out_dir / f"{cell.warmup_yaml_id}.yaml"
        write_yaml(path, yaml_dict)
        logger.info("[emit-warmup-yamls] %s", cell.warmup_yaml_id)
    print(f"[emit-warmup-yamls] {len(seen)} warmup yamls -> {out_dir}")


# ----------------------------------------------------------------------
# Mode 2: run-warmup
# ----------------------------------------------------------------------


def _extract_finite_factor_raw(
    src_per_step_jsonl: Path, dst: Path, declared_keys: list[str],
) -> None:
    """Project per-step jsonl to {factor_raw: {k: v}} finite-only rows."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src_per_step_jsonl.open() as src, dst.open("w") as out:
        for line in src:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            raw = row.get("factor_raw") or row.get("factor_outputs", {}).get("raw") or {}
            kept = {k: float(raw[k]) for k in declared_keys if k in raw}
            if not kept or not any(math.isfinite(v) for v in kept.values()):
                continue
            out.write(json.dumps({"factor_raw": kept}) + "\n")


def _run_one_warmup(ctl, cell: Cell, args: Args) -> None:
    wid = cell.warmup_yaml_id
    wpath = _warmup_yaml_dir(args) / f"{wid}.yaml"
    raw_dst = _phase5_warmup_raw_dir(args) / f"{wid}.jsonl"
    if raw_dst.exists() and raw_dst.stat().st_size > 0:
        logger.info("[run-warmup] %s: cached -> %s, skipping", wid, raw_dst)
        return

    if not wpath.exists():
        # Lazy emit: write the warmup yaml on demand so callers don't
        # have to gate on a separate `--mode emit-warmup-yamls` run.
        yaml_dict = build_warmup_yaml_for_cell(
            args.cfg_id, cell,
            preload_pkl_override=args.preload_pkl_override or None,
        )
        wpath.parent.mkdir(parents=True, exist_ok=True)
        write_yaml(wpath, yaml_dict)
        logger.info("[run-warmup] lazy-emit warmup yaml %s", wid)

    logger.info("[run-warmup] %s", wid)
    ctl.load_cache_config(yaml_content=wpath.read_text(encoding="utf-8"), yaml_id=wid)
    cmd, env = _build_libero_argv(
        args=args, yaml_id=wid, phase="warmup",
        num_trials_per_task=args.warmup_trials,
    )
    subprocess.run(cmd, env=env, check=True)

    raw_jsonl_dir = _phase5_warmup_raw_dir(args) / "_raw_dump"
    raw_jsonl_dir.mkdir(parents=True, exist_ok=True)
    raw_jsonl = raw_jsonl_dir / f"{wid}.jsonl"
    content = ctl.fetch_dump(wid)
    _save_dump_jsonl(content, raw_jsonl)
    _extract_finite_factor_raw(raw_jsonl, raw_dst, list(cell.declared_keys))
    logger.info("[run-warmup] %s -> %s", wid, raw_dst)


def _mode_run_warmup(args: Args) -> None:
    if WebsocketClientPolicy is None:
        raise RuntimeError("openpi_client is not installed; cannot run warmup")
    cells = _resolve_cells(args)
    seen: set[str] = set()
    unique_warmup_cells: list[Cell] = []
    for cell in cells:
        if cell.group == "g5":
            continue
        if cell.warmup_yaml_id in seen:
            continue
        seen.add(cell.warmup_yaml_id)
        unique_warmup_cells.append(cell)
    for cell in unique_warmup_cells:
        with WebsocketClientPolicy(host=args.host, port=args.port) as ctl:
            _run_one_warmup(ctl, cell, args)


# ----------------------------------------------------------------------
# Mode 3: emit-eval-yamls (per-cell solve thresholds + write yaml)
# ----------------------------------------------------------------------


def _mode_emit_eval_yamls(args: Args) -> None:
    out_dir = _eval_yaml_dir(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    cells = _resolve_cells(args)
    n_ok = 0
    n_skip = 0
    for cell in cells:
        path = out_dir / f"{cell.yaml_id}.yaml"
        try:
            fh_thr, ws_thr = _solve_thresholds_phase5(cell, args)
        except FileNotFoundError as e:
            logger.warning("[emit-eval-yamls] skip %s: %s", cell.yaml_id, e)
            n_skip += 1
            continue
        except (ValueError, RuntimeError) as e:
            logger.warning("[emit-eval-yamls] skip %s (solver error): %s",
                           cell.yaml_id, e)
            n_skip += 1
            continue
        yaml_dict = build_eval_yaml_for_cell(
            args.cfg_id, cell, fh_thr, ws_thr,
            preload_pkl_override=args.preload_pkl_override or None,
        )
        write_yaml(path, yaml_dict)
        n_ok += 1
    print(f"[emit-eval-yamls] {n_ok} ok, {n_skip} skipped -> {out_dir}")


# ----------------------------------------------------------------------
# Mode 4: run-eval — phase5-local _run_one_cell_phase5
# ----------------------------------------------------------------------


def _read_thresholds_from_yaml(yaml_path: Path) -> tuple[float, float]:
    """Re-parse the eval yaml to recover the FH/WS thresholds the server
    will see. Used to back-fill the summary jsonl when run-eval is
    invoked without the solver having run in this process.
    """
    import yaml as _yaml   # noqa: PLC0415
    data = _yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    composer = data["checkpoints"]["cp1"]["judge"]["composer"]
    th = composer["tier_thresholds"]
    return float(th["full_hit"]), float(th["warm_start"])


def _run_one_cell_phase5(
    ctl, cell: Cell, args: Args, summary_path: Path,
) -> dict:
    """Run a single phase5 eval cell (mirror phase4 preload-before-load
    order, but phase5-local — no RECIPES_PHASE4 lookup, no phase4 paths).
    """
    eval_yaml_path = _eval_yaml_dir(args) / f"{cell.yaml_id}.yaml"
    fh_thr, ws_thr = _read_thresholds_from_yaml(eval_yaml_path)

    raw_path = warmup_raw_path_for_cell(
        cell,
        phase5_warmup_dir=_phase5_warmup_raw_dir(args),
        phase3_warmup_dir=_phase3_warmup_dir(args),
        phase4_warmup_raw_dir=_phase4_warmup_raw_dir(args),
    )
    declared_keys = list(cell.declared_keys)
    buffer = load_per_key_finite_history(raw_path, declared_keys)

    ack = ctl.preload_normalizer_buffer(cell.yaml_id, buffer)
    n_warmup_keys = ack.get("n_keys", len(buffer)) if isinstance(ack, dict) else len(buffer)
    ctl.load_cache_config(
        yaml_content=eval_yaml_path.read_text(encoding="utf-8"),
        yaml_id=cell.yaml_id,
    )

    ep_dir = _episode_results_dir(args)
    episode_results_path: Optional[Path] = None
    if ep_dir is not None:
        ep_dir.mkdir(parents=True, exist_ok=True)
        episode_results_path = ep_dir / f"{cell.yaml_id}.json"

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
        "group": cell.group,
        "base_recipe": cell.base_recipe,
        "axis_tag": cell.axis_tag,
        "fh_ratio": cell.fh_ratio,
        "ws_ratio": cell.ws_ratio,
        "fh_thr": fh_thr,
        "ws_thr": ws_thr,
        "warmup_yaml_id": cell.warmup_yaml_id,
        "n_warmup_buffer_keys": n_warmup_keys,
        "success_rate": success_rate,
        **counts,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("a") as fh:
        fh.write(json.dumps(summary) + "\n")
    return summary


def _ensure_warmup_for_cell(ctl, cell: Cell, args: Args) -> None:
    """Lazy: run the cell's warmup yaml if its factor_raw jsonl is missing.

    G5 cells are skipped — their raw is historical (phase3 g6 / phase4
    p1+p2) and not produced by phase5 own warmup. If a G5 raw is missing
    when the eval needs it, the downstream solver call surfaces a clear
    FileNotFoundError; we don't try to re-create historical data here.
    """
    if cell.group == "g5":
        return
    raw_path = warmup_raw_path_for_cell(
        cell,
        phase5_warmup_dir=_phase5_warmup_raw_dir(args),
        phase3_warmup_dir=_phase3_warmup_dir(args),
        phase4_warmup_raw_dir=_phase4_warmup_raw_dir(args),
    )
    if raw_path.exists() and raw_path.stat().st_size > 0:
        return   # already cached
    _run_one_warmup(ctl, cell, args)


def _ensure_eval_yaml_for_cell(cell: Cell, args: Args) -> None:
    """Lazy: solve thresholds + write eval yaml if missing on disk."""
    eval_yaml_path = _eval_yaml_dir(args) / f"{cell.yaml_id}.yaml"
    if eval_yaml_path.exists() and eval_yaml_path.stat().st_size > 0:
        return
    fh_thr, ws_thr = _solve_thresholds_phase5(cell, args)
    yaml_dict = build_eval_yaml_for_cell(
        args.cfg_id, cell, fh_thr, ws_thr,
        preload_pkl_override=args.preload_pkl_override or None,
    )
    eval_yaml_path.parent.mkdir(parents=True, exist_ok=True)
    write_yaml(eval_yaml_path, yaml_dict)
    logger.info("[run-eval] lazy-emit eval yaml %s (fh=%.4f ws=%.4f)",
                cell.yaml_id, fh_thr, ws_thr)


def _mode_run_eval(args: Args) -> None:
    """Per-cell pipeline: ensure warmup raw → solve+emit eval yaml → eval.

    Each cell is fully self-contained: the server runs whatever warmup it
    needs (skipping if already cached), then solves thresholds, then runs
    the LIBERO eval workers. No separate `run-warmup` / `emit-eval-yamls`
    step is required up-front for G1-G4 — those modes remain available
    as standalone helpers but `run-eval` will lazy-trigger them per cell.
    """
    if WebsocketClientPolicy is None:
        raise RuntimeError("openpi_client is not installed; cannot run eval")
    cells = _resolve_cells(args)
    summary_path = _summary_path(args)
    done = _load_done_yaml_ids(summary_path) if summary_path.exists() else set()
    pending = [c for c in cells if c.yaml_id not in done]
    logger.info("[run-eval] %d cells total, %d already done, %d to run",
                len(cells), len(done), len(pending))
    for cell in pending:
        with WebsocketClientPolicy(host=args.host, port=args.port) as ctl:
            try:
                _ensure_warmup_for_cell(ctl, cell, args)
                _ensure_eval_yaml_for_cell(cell, args)
                _run_one_cell_phase5(ctl, cell, args, summary_path)
            except Exception:   # noqa: BLE001 — log + continue
                import traceback
                logger.error("[run-eval] %s failed:\n%s", cell.yaml_id,
                             traceback.format_exc())


# ----------------------------------------------------------------------
# Decision gate (per-group dispatch)
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


def _winner_rule_5pp(rows: list[dict], group_key) -> dict[str, Any]:
    """G1-G4 rule: per group_key bucket, top-1 SR is winner only if
    SR_top1 - SR_top2 ≥ 0.05 — else inconclusive.
    """
    buckets: dict[Any, list[dict]] = {}
    for r in rows:
        buckets.setdefault(group_key(r), []).append(r)
    out: dict[str, Any] = {}
    for k, lst in buckets.items():
        ranked = sorted(lst, key=lambda r: -(r.get("success_rate") or 0.0))
        if len(ranked) < 2:
            out[str(k)] = {"winner": ranked[0]["yaml_id"] if ranked else None,
                           "reason": "single-cell"}
            continue
        top1 = ranked[0].get("success_rate") or 0.0
        top2 = ranked[1].get("success_rate") or 0.0
        # Float-fuzz tolerance: 0.96 - 0.91 may be 0.0499...96 in IEEE754.
        if top1 - top2 >= 0.05 - 1e-9:
            out[str(k)] = {
                "winner": ranked[0]["yaml_id"],
                "sr_top1": top1, "sr_top2": top2, "delta": top1 - top2,
            }
        else:
            out[str(k)] = {
                "winner": None,
                "reason": f"inconclusive (Δ={top1 - top2:.3f} < 0.05)",
                "top1_yaml": ranked[0]["yaml_id"],
                "top2_yaml": ranked[1]["yaml_id"],
            }
    return out


def _pareto_frontier(rows: list[dict]) -> list[dict]:
    """Upper-left Pareto front on (inf, sr)."""
    pts = sorted(
        [(_compute_inf(r), r.get("success_rate") or 0.0, r) for r in rows],
        key=lambda x: x[0],
    )
    front = []
    best = -1.0
    for inf, sr, r in pts:
        if sr > best:
            front.append({"yaml_id": r["yaml_id"], "inf": inf, "sr": sr})
            best = sr
    return front


def _dump_decision_gate_table_phase5(summary_path: Path) -> dict:
    """Write 5 per-group decision files next to summary_path (plan §4):

    - g1_decision.json — bucket by (base_recipe, channel, desc); 5pp rule.
    - g2_decision.json — bucket by (base_recipe, channel, desc); 5pp rule.
    - g3_decision.json — bucket by (base_recipe, channel); 5pp over 12 patterns.
    - g4_decision.json — bucket by (base_recipe, channel); 5pp over 12 subsets.
    - g5_decision.json — per recipe: Pareto frontier + best_sr +
      cheapest_above_0.85 (no winner ranking).

    Returns the aggregate dict (keys "g1".."g5") for callers/tests; the
    on-disk artifacts are five separate JSON files per plan §4.1..§4.5.
    """
    if not summary_path.exists():
        return {}
    rows = [json.loads(l) for l in summary_path.read_text().splitlines() if l.strip()]
    gate: dict[str, Any] = {}

    def _axis_split(axis_tag: str) -> list[str]:
        return axis_tag.split("__")[0].split("_")

    g1 = [r for r in rows if r.get("group") == "g1"]
    if g1:
        gate["g1"] = _winner_rule_5pp(
            g1, lambda r: tuple(_axis_split(r["axis_tag"])[:3]),
        )
    g2 = [r for r in rows if r.get("group") == "g2"]
    if g2:
        gate["g2"] = _winner_rule_5pp(
            g2, lambda r: tuple(_axis_split(r["axis_tag"])[:3]),
        )
    g3 = [r for r in rows if r.get("group") == "g3"]
    if g3:
        gate["g3"] = _winner_rule_5pp(
            g3, lambda r: tuple(_axis_split(r["axis_tag"])[:2]),
        )
    g4 = [r for r in rows if r.get("group") == "g4"]
    if g4:
        gate["g4"] = _winner_rule_5pp(
            g4, lambda r: tuple(_axis_split(r["axis_tag"])[:2]),
        )
    g5 = [r for r in rows if r.get("group") == "g5"]
    if g5:
        per_recipe: dict[str, Any] = {}
        for recipe_short in ("p1", "p2", "g6"):
            sub = [r for r in g5 if r.get("base_recipe") == recipe_short]
            if not sub:
                continue
            ranked = sorted(sub, key=lambda r: -(r.get("success_rate") or 0.0))
            best_sr = ranked[0]
            cheap_above = sorted(
                [r for r in sub if (r.get("success_rate") or 0.0) >= 0.85],
                key=_compute_inf,
            )
            per_recipe[recipe_short] = {
                "best_sr_yaml": best_sr["yaml_id"],
                "best_sr": best_sr.get("success_rate"),
                "cheapest_above_0.85_yaml": cheap_above[0]["yaml_id"] if cheap_above else None,
                "frontier": _pareto_frontier(sub),
            }
        gate["g5"] = per_recipe

    # Plan §4.1..§4.5: write 5 per-group decision files.
    for g in ("g1", "g2", "g3", "g4", "g5"):
        out_path = summary_path.parent / f"{g}_decision.json"
        payload = gate.get(g, {})
        out_path.write_text(json.dumps(payload, indent=2) + "\n")
    return gate


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args(argv)

    if args.mode == "emit-warmup-yamls":
        _mode_emit_warmup_yamls(args)
    elif args.mode == "run-warmup":
        _mode_run_warmup(args)
    elif args.mode == "emit-eval-yamls":
        _mode_emit_eval_yamls(args)
    elif args.mode == "run-eval":
        _mode_run_eval(args)
        # decision gate at end of run-eval
        _dump_decision_gate_table_phase5(_summary_path(args))
    else:
        raise SystemExit(f"unknown --mode {args.mode!r}")
    return 0


__all__ = [
    "Args",
    "VALID_GROUPS",
    "VALID_MODES",
    "WARM_COST",
    "_dump_decision_gate_table_phase5",
    "_ensure_eval_yaml_for_cell",
    "_ensure_warmup_for_cell",
    "_mode_emit_eval_yamls",
    "_mode_emit_warmup_yamls",
    "_mode_run_eval",
    "_mode_run_warmup",
    "_parse_args",
    "_resolve_cells",
    "_run_one_cell_phase5",
    "_solve_thresholds_phase5",
]


if __name__ == "__main__":
    sys.exit(main())
