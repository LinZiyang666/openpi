"""CLI entrypoint for kinematic phase5 replication.

8 modes (mirror phase5/runner.py + always-warm self-ceiling + summary aggregator):

  emit-warmup        Write super_warmup.yaml (no server).
  run-warmup         Run super warmup on a server, fetch_dump, extract finite raw.
  verify-raw         Hard-gate 7-check verifier on the super warmup raw.
  emit-eval-yamls    Per-cell: reconstruct_scores + derive_thresholds + write yaml.
                     Wraps each cell in try/except so one bad cell does not
                     abort the whole emission (G1 R1 R3 mandate).
  run-always-warm    3 always-warm yamls on d1 base (start_t ∈ {0.3, 0.5, 0.7})
                     for self-ceiling anchor (replaces phase5 d4 rrf reused values).
  run-eval           Dispatch to exp/weighted_sum/run_phase2.py with
                     --strategy=kinematic over the 237 emitted cells.
  aggregate-summary  Rebuild per_yaml_summary.jsonl + always_warm_results.json
                     from raw journal + per_step + cell registry. weighted_sum's
                     run_phase2 writes journal+per_step but not the phase5-style
                     per_yaml_summary that ``analyze`` consumes — this mode
                     bridges the gap so the analyze pipeline is reproducible
                     end-to-end without inline scripting.
  analyze            decision-gate dump + 4-frontier Pareto overlay.

Coupling map:
  DEPENDS ON:  exp.weighted_sum.kinematic.{spec, super_warmup, strategy},
               exp.verdict_factor_judge.phase3.threshold_solver,
               exp.verdict_factor_judge.phase5.{spec, runner},
               openpi_client.websocket_client_policy.
  CONSUMED BY: CLI / agentchat-driven Stage 0..7 (operator).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from exp.weighted_sum.kinematic.spec import (
    SUPER_WARMUP_RAW_RELPATH,
    build_eval_yaml_for_cell,
    cell_to_solver_recipe,
    generate_all_cells,
)
from exp.weighted_sum.kinematic.super_warmup import (
    DEFAULT_RAW_PATH,
    DEFAULT_YAML_PATH,
    build_super_warmup_yaml,
    run_super_warmup,
    verify,
)


logger = logging.getLogger(__name__)


VALID_MODES = (
    "emit-warmup",
    "run-warmup",
    "verify-raw",
    "emit-eval-yamls",
    "run-always-warm",
    "run-eval",
    "aggregate-summary",
    "analyze",
)


# ----------------------------------------------------------------------
# Default paths
# ----------------------------------------------------------------------


DEFAULT_EVAL_YAML_DIR = Path("exp/weighted_sum/config/kinematic_phase5/libero_spatial/d1/eval")
DEFAULT_ALWAYS_WARM_YAML_DIR = Path(
    "exp/weighted_sum/config/kinematic_phase5/libero_spatial/d1/always_warm"
)
DEFAULT_DATA_DIR = Path("exp/weighted_sum/data/libero_spatial/kinematic_phase5/d1")
DEFAULT_THRESHOLDS_DIR = DEFAULT_DATA_DIR / "thresholds"


# ----------------------------------------------------------------------
# emit-warmup
# ----------------------------------------------------------------------


def _mode_emit_warmup(args: argparse.Namespace) -> None:
    from exp.verdict_factor_judge.common.generate_yamls import write_yaml

    out_path = Path(args.warmup_yaml) if args.warmup_yaml else DEFAULT_YAML_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_dict = build_super_warmup_yaml(
        preload_pkl_override=args.preload_pkl_override or None,
        cfg_id=args.cfg_id,
    )
    write_yaml(out_path, yaml_dict)
    n_factors = len(yaml_dict["checkpoints"]["cp1"]["judge"]["dump"]["factors"])
    logger.info(
        "[emit-warmup] wrote super warmup yaml -> %s (%d dump factor blocks)",
        out_path,
        n_factors,
    )


# ----------------------------------------------------------------------
# run-warmup
# ----------------------------------------------------------------------


def _parse_task_ids(raw: str) -> tuple[int, ...]:
    """Parse ``"0,1,2,..."`` CSV string into a ``tuple[int, ...]``.

    Required because ``exp/verdict_factor_judge/common/run_phase._build_libero_argv``
    iterates ``args.task_ids`` and stringifies each element (run_phase.py:244).
    Passing a raw ``str`` would emit per-character argv tokens
    (``--task-ids 0 , 1 , 2 ...``) — see G2 R1 B2.

    Mirrors ``phase5/runner._parse_args`` line 156-157.
    """
    return tuple(int(x) for x in raw.split(",") if x.strip()) if raw else ()


def _build_libero_args_from_cli(args: argparse.Namespace) -> dict[str, Any]:
    """Translate CLI args into the kwargs phase5._build_libero_argv expects.

    phase5/runner.py has a hand-rolled ``_build_libero_argv(args, yaml_id,
    phase, num_trials_per_task)`` that pulls fields off a dataclass; here
    we adapt the argparse Namespace into a shape that satisfies the same
    field access pattern. Critically, ``task_ids`` is parsed from the
    CSV string into ``tuple[int, ...]`` so the downstream argv loop
    (run_phase.py:244) emits well-formed ``--task-ids 0 1 2 ...`` rather
    than per-character tokens (G2 R1 B2 fix).
    """
    # Avoid in-place mutation of argparse Namespace (subtle aliasing bug
    # if caller re-uses ``args`` after this call) — write to a private
    # SimpleNamespace clone that exposes the fields run_phase.py reads.
    import types

    adapted = types.SimpleNamespace(
        host=args.host,
        port=args.port,
        task_suite=args.task_suite,
        num_workers=args.num_workers,
        # The B2 fix — convert "0,1,...,9" to (0, 1, ..., 9).
        task_ids=_parse_task_ids(args.task_ids),
        per_step_log_dir=args.per_step_log_dir,
        init_states_dir=getattr(args, "init_states_dir", ""),
        episode_filter=args.episode_filter,
        cuda_visible_devices=args.cuda_visible_devices,
        conda_env=args.conda_env,
        preload_pkl_override=args.preload_pkl_override,
    )
    return {"args": adapted}


def _mode_run_warmup(args: argparse.Namespace) -> None:
    raw_path = run_super_warmup(
        host=args.host,
        port=args.port,
        cfg_id=args.cfg_id,
        trials_per_task=args.trials_per_task,
        yaml_path=Path(args.warmup_yaml) if args.warmup_yaml else DEFAULT_YAML_PATH,
        raw_path=Path(args.super_raw) if args.super_raw else DEFAULT_RAW_PATH,
        libero_args=_build_libero_args_from_cli(args),
        preload_pkl_override=args.preload_pkl_override or None,
    )
    logger.info("[run-warmup] super warmup raw at %s", raw_path)


# ----------------------------------------------------------------------
# verify-raw
# ----------------------------------------------------------------------


def _mode_verify_raw(args: argparse.Namespace) -> None:
    raw_path = Path(args.super_raw) if args.super_raw else DEFAULT_RAW_PATH
    if not raw_path.exists():
        raise SystemExit(f"super warmup raw not found: {raw_path}")
    verify(raw_path)
    logger.info("[verify-raw] HARD GATE PASSED — Stage 4 may proceed")


# ----------------------------------------------------------------------
# emit-eval-yamls (G1 R1 R3 + R4: per-cell try/except, skip-not-crash)
# ----------------------------------------------------------------------


def _mode_emit_eval_yamls(args: argparse.Namespace) -> None:
    from exp.verdict_factor_judge.common.generate_yamls import write_yaml
    from exp.verdict_factor_judge.phase3.threshold_solver import (
        derive_thresholds,
        reconstruct_scores,
    )

    super_raw = Path(args.super_raw) if args.super_raw else DEFAULT_RAW_PATH
    if not super_raw.exists():
        raise SystemExit(f"super warmup raw not found: {super_raw}")
    eval_dir = Path(args.eval_dir) if args.eval_dir else DEFAULT_EVAL_YAML_DIR
    thresholds_dir = (
        Path(args.thresholds_dir) if args.thresholds_dir else DEFAULT_THRESHOLDS_DIR
    )
    eval_dir.mkdir(parents=True, exist_ok=True)
    thresholds_dir.mkdir(parents=True, exist_ok=True)

    n_ok = 0
    n_skip = 0
    fails: list[tuple[str, str]] = []
    for cell in generate_all_cells():
        try:
            recipe = cell_to_solver_recipe(cell)
            scores = reconstruct_scores(
                super_raw, recipe, composer_weights=cell.weights
            )
            fh_thr, ws_thr = derive_thresholds(scores, cell.fh_ratio, cell.ws_ratio)
        except (ValueError, RuntimeError, KeyError) as e:
            # G1 R1 R3 R1A B4 mandate: skip-not-crash on per-cell solver failure.
            fails.append((cell.yaml_id, repr(e)))
            n_skip += 1
            (thresholds_dir / f"{cell.yaml_id}__SKIP.json").write_text(
                json.dumps({"yaml_id": cell.yaml_id, "error": str(e)}) + "\n",
                encoding="utf-8",
            )
            continue

        # Tie-break corner: T_fh == T_ws would be rejected server-side by
        # ThresholdJudge's strict warm_tier < judge.threshold check.
        if not (fh_thr > ws_thr):
            fails.append(
                (cell.yaml_id, f"threshold tie: fh_thr={fh_thr}, ws_thr={ws_thr}")
            )
            n_skip += 1
            (thresholds_dir / f"{cell.yaml_id}__SKIP.json").write_text(
                json.dumps(
                    {
                        "yaml_id": cell.yaml_id,
                        "error": "threshold tie",
                        "fh_thr": fh_thr,
                        "ws_thr": ws_thr,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            continue

        yaml_dict = build_eval_yaml_for_cell(
            cell,
            fh_thr,
            ws_thr,
            super_raw_relpath=args.super_raw_relpath,
            cfg_id=args.cfg_id,
            preload_pkl_override=args.preload_pkl_override or None,
        )
        write_yaml(eval_dir / f"{cell.yaml_id}.yaml", yaml_dict)

        n_finite = sum(1 for s in scores if not (s is None or math.isnan(s)))
        (thresholds_dir / f"{cell.yaml_id}.json").write_text(
            json.dumps(
                {
                    "yaml_id": cell.yaml_id,
                    "group": cell.group,
                    "base_recipe": cell.base_recipe,
                    "fh_ratio": cell.fh_ratio,
                    "ws_ratio": cell.ws_ratio,
                    "fh_thr": fh_thr,
                    "ws_thr": ws_thr,
                    "n_finite_scores": n_finite,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        n_ok += 1

    logger.info(
        "[emit-eval] %d ok, %d skip; first fails: %s",
        n_ok,
        n_skip,
        fails[:5],
    )


# ----------------------------------------------------------------------
# run-always-warm (3 cells × 100 ep self-ceiling)
# ----------------------------------------------------------------------


def _build_always_warm_yaml(
    start_t: float,
    *,
    preload_pkl_override: str | None = None,
    cfg_id: str = "spatial16_ws_d1_best",
) -> dict[str, Any]:
    """Drop-in always-warm yaml on d1 base for self-measured SR ceiling."""
    from exp.verdict_factor_judge.common.v2_spec import CFG_SPECS

    cfg = CFG_SPECS[cfg_id]
    preload_path = (
        preload_pkl_override if preload_pkl_override is not None else cfg["preload_pkl"]
    )
    return {
        "enabled": True,
        "timer": {"enabled": False, "buffer_size": 10000, "output_csv_dir": None},
        "keys": dict(cfg["keys"]),
        "key_builder": {"type": cfg["key_builder_type"]},
        "checkpoints": {
            "cp1": {
                "enabled": True,
                "gate": {"type": "always_search"},
                "judge": {
                    "type": "always_warm_start",
                    "start_t": float(start_t),
                },
                "search_strategy": dict(cfg["search_strategy"]),
            },
        },
        "backend": {
            "type": "in_memory",
            "vector_dims": dict(cfg["vector_dims"]),
            "in_memory": {
                "preload_path": preload_path,
                "index_type": "brute_force",
            },
        },
        "write_policy": {"type": "never"},
    }


def _mode_run_always_warm(args: argparse.Namespace) -> None:
    """Emit + run 3 always-warm yamls. Operator delegates the actual
    LIBERO drive to ``run_phase2.py --strategy=weight`` against this dir.
    """
    from exp.verdict_factor_judge.common.generate_yamls import write_yaml

    out_dir = (
        Path(args.always_warm_dir)
        if args.always_warm_dir
        else DEFAULT_ALWAYS_WARM_YAML_DIR
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    for start_t in (0.3, 0.5, 0.7):
        yid = f"ws_d1_kin_always_warm_t{start_t}"
        y = _build_always_warm_yaml(
            start_t,
            preload_pkl_override=args.preload_pkl_override or None,
            cfg_id=args.cfg_id,
        )
        write_yaml(out_dir / f"{yid}.yaml", y)
        logger.info("[always-warm] emitted %s (start_t=%.2f)", yid, start_t)
    logger.info(
        "[always-warm] now run: python exp/weighted_sum/run_phase2.py "
        "--strategy weight --yaml-dir %s --eval-trials 10 ...",
        out_dir,
    )


# ----------------------------------------------------------------------
# run-eval — delegate to run_phase2
# ----------------------------------------------------------------------


def _mode_run_eval(args: argparse.Namespace) -> None:
    """Echo the run_phase2.py command line; operator runs it from timan107.

    We do NOT subprocess.run it here because the conductor + workers need
    to live on the client machine (timan107) with EGL slots + conda env,
    not on this driver host.
    """
    # Owner override (2026-05-31, WA line 7): Stage 4 eval runs on DUAL servers
    # (e.g. 40+40 workers), consistent with Stage 1/2/3 which produced the
    # winners on dual-server Pareto comparisons. The earlier single-server guard
    # (plan §2.3) is retired by owner decision; --servers / --server-workers are
    # threaded through verbatim so the operator picks the topology.
    eval_dir = Path(args.eval_dir) if args.eval_dir else DEFAULT_EVAL_YAML_DIR
    data_dir = Path(args.data_dir) if args.data_dir else DEFAULT_DATA_DIR
    init_map = f"exp/common/data/db/libero_cache/{args.task_suite}_init_map.json"
    print("# Run from client (timan107).")
    print("# NOTE: dual-server eval (owner 2026-05-31). --servers host:p1,host:p2")
    print("#   and --server-workers n1,n2 (e.g. 40,40); consistent with Stage 1/2/3.")
    print(
        "PYTHONPATH=. /shared/nas/data/m1/zixuans8/miniconda3/bin/uv run "
        f"exp/weighted_sum/run_phase2.py \\\n"
        f"  --strategy kinematic \\\n"
        f"  --yaml-dir {eval_dir} \\\n"
        f"  --init-map {init_map} \\\n"
        f"  --task-suite {args.task_suite} \\\n"
        f"  --journal {data_dir}/journal.jsonl \\\n"
        f"  --servers {args.servers} \\\n"
        f"  --task-ids 0-9 --eval-trials 10 \\\n"
        f"  --workers {args.num_workers} --server-workers {args.server_workers} \\\n"
        f"  --gpus 8 --conda-env /scratch/zixuans8/libero_sim \\\n"
        f"  --eval-concurrency 2 \\\n"
        f"  --per-step-out {data_dir}/per_step.jsonl"
    )


# ----------------------------------------------------------------------
# aggregate-summary — rebuild per_yaml_summary.jsonl + always_warm_results.json
# ----------------------------------------------------------------------
#
# weighted_sum/run_phase2.py writes ``journal.jsonl`` (one row per episode
# terminal state) and ``per_step/<yaml_id>.jsonl`` (one row per cache verdict).
# It does NOT write the phase5-style ``per_yaml_summary.jsonl`` (one row per
# yaml) that ``_mode_analyze`` consumes. This mode bridges the gap by reading
# the two raw artifacts + the cell registry from spec.py and producing both
# ``per_yaml_summary.jsonl`` (237 rows) and ``always_warm_results.json``
# (3 anchor entries) at server-side relative paths matching the analyze
# default. The status filter ``{done, failed}`` covers both terminal cases so
# the denominator is the full episode count rather than only successful ones.


def _aggregate_per_yaml_summary(
    cells: dict[str, Any],
    journal_path: Path,
    per_step_dir: Path,
) -> list[dict[str, Any]]:
    """Build per-yaml summary rows from raw journal + per_step.

    Each row carries the schema ``_mode_analyze`` and the plot script expect:
    ``yaml_id, group, base_recipe, axis_tag, fh_ratio, ws_ratio,
    n_episodes, n_success, success_rate, n_eval_verdicts,
    n_full_hit, n_warm_start, n_miss``.
    """
    from collections import Counter as _Counter

    ok: _Counter = _Counter()
    total: _Counter = _Counter()
    with journal_path.open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("phase") != "eval":
                continue
            if r.get("status") not in ("done", "failed"):
                continue
            yid = r["yaml_id"]
            total[yid] += 1
            if r.get("success"):
                ok[yid] += 1

    rows: list[dict[str, Any]] = []
    for yid, cell in cells.items():
        t = total.get(yid, 0)
        if t == 0:
            continue
        counts: _Counter = _Counter()
        per_step = per_step_dir / f"{yid}.jsonl"
        if per_step.exists():
            with per_step.open(encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    counts[r.get("hit_type") or "UNKNOWN"] += 1
        n_verdicts = sum(counts.values())
        rows.append(
            {
                "yaml_id": yid,
                "group": cell.group,
                "base_recipe": cell.base_recipe,
                "axis_tag": cell.axis_tag,
                "fh_ratio": cell.fh_ratio,
                "ws_ratio": cell.ws_ratio,
                "n_episodes": t,
                "n_success": ok.get(yid, 0),
                "success_rate": ok.get(yid, 0) / t,
                "n_eval_verdicts": n_verdicts,
                "n_full_hit": counts.get("FULL_HIT", 0),
                "n_warm_start": counts.get("WARM_START", 0),
                "n_miss": counts.get("MISS", 0),
            }
        )
    return rows


def _aggregate_always_warm(journal_path: Path) -> dict[str, dict[str, Any]]:
    """Build {start_t_<t>: {success_rate, inf, n_episodes, n_success}}.

    ``always_warm_start`` judge marks every step as ``WARM_START``, so the
    per-episode inference ratio is the **per-step warm cost**
    ``cost_WS(t) = 1 - 0.5*(1 - t) = 0.5 + 0.5*t`` (see
    ``exp/weighted_sum/summarize_inf_ratio.py:_warm_cost``). Earlier
    revisions used the constant 0.75 here — that is only correct at
    ``start_t = 0.5`` and produced two coincident anchors on the Pareto plot
    (``t0.5`` and ``t0.7`` both falling at ``(0.75, 0.99)``). The three
    anchors should be at ``inf ∈ {0.65, 0.75, 0.85}`` for
    ``start_t ∈ {0.3, 0.5, 0.7}`` respectively, giving the three distinct
    points the ceiling sweep was designed to produce.

    There is no per_step file for the Stage 6 anchors — Stage 6 uses
    ``--strategy weight`` (per_step_writer=None), and the
    ``always_warm_start`` judge has no per-step branching anyway, so the
    closed-form ``_warm_cost(start_t)`` is the exact value.

    Schema matches what ``plot_pareto_overlay._load_self_always_warm``
    expects.
    """
    from collections import Counter as _Counter

    def _warm_cost(start_t: float) -> float:
        return 0.5 + 0.5 * start_t

    ok: _Counter = _Counter()
    total: _Counter = _Counter()
    with journal_path.open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("phase") != "eval":
                continue
            if r.get("status") not in ("done", "failed"):
                continue
            yid = r["yaml_id"]
            total[yid] += 1
            if r.get("success"):
                ok[yid] += 1

    out: dict[str, dict[str, Any]] = {}
    for yid in (
        "ws_d1_kin_always_warm_t0.3",
        "ws_d1_kin_always_warm_t0.5",
        "ws_d1_kin_always_warm_t0.7",
    ):
        t = total.get(yid, 0)
        start_t = float(yid.split("_t")[-1])
        out[f"start_t_{start_t}"] = {
            "success_rate": ok.get(yid, 0) / t if t else 0.0,
            "inf": _warm_cost(start_t),
            "n_episodes": t,
            "n_success": ok.get(yid, 0),
        }
    return out


def _mode_aggregate_summary(args: argparse.Namespace) -> None:
    """Rebuild per_yaml_summary.jsonl + always_warm_results.json."""
    data_dir = Path(args.data_dir) if args.data_dir else DEFAULT_DATA_DIR
    journal_path = Path(args.journal) if args.journal else data_dir / "journal.jsonl"
    per_step_dir = (
        Path(args.per_step_dir) if args.per_step_dir else data_dir / "per_step"
    )
    aw_journal_path = (
        Path(args.always_warm_journal)
        if args.always_warm_journal
        else data_dir / "always_warm_journal.jsonl"
    )
    summary_out = (
        Path(args.summary) if args.summary else data_dir / "per_yaml_summary.jsonl"
    )
    aw_out_path = data_dir / "always_warm_results.json"

    cells = {c.yaml_id: c for c in generate_all_cells()}
    logger.info("[aggregate] loaded %d cells from registry", len(cells))

    if not journal_path.exists():
        raise SystemExit(f"[aggregate] journal not found: {journal_path}")
    rows = _aggregate_per_yaml_summary(cells, journal_path, per_step_dir)
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )
    logger.info("[aggregate] wrote %s (%d rows)", summary_out, len(rows))

    if aw_journal_path.exists():
        aw = _aggregate_always_warm(aw_journal_path)
        aw_out_path.write_text(json.dumps(aw, indent=2) + "\n", encoding="utf-8")
        logger.info("[aggregate] wrote %s (%d anchors)", aw_out_path, len(aw))
    else:
        logger.warning(
            "[aggregate] skipped always_warm_results — journal missing at %s",
            aw_journal_path,
        )


# ----------------------------------------------------------------------
# analyze — decision-gate + 4-frontier overlay
# ----------------------------------------------------------------------


WARM_COST = 0.75


def _compute_inf(n_full_hit: int, n_warm: int, n_miss: int) -> float:
    n = n_full_hit + n_warm + n_miss
    if not n:
        return 0.0
    return (n_full_hit * 0.0 + n_warm * WARM_COST + n_miss * 1.0) / n


def _winner_rule_5pp(
    rows: list[dict[str, Any]],
    group_key,
) -> dict[str, Any]:
    """Mirror phase5/runner._winner_rule_5pp: 5pp Δ-SR top1-top2."""
    buckets: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        buckets[group_key(r)].append(r)
    out: dict[str, Any] = {}
    for k, lst in buckets.items():
        ranked = sorted(lst, key=lambda r: -(r.get("success_rate") or 0.0))
        if len(ranked) < 2:
            out[str(k)] = {
                "winner": ranked[0]["yaml_id"] if ranked else None,
                "reason": "single-cell",
            }
            continue
        top1 = ranked[0].get("success_rate") or 0.0
        top2 = ranked[1].get("success_rate") or 0.0
        if top1 - top2 >= 0.05 - 1e-9:
            out[str(k)] = {
                "winner": ranked[0]["yaml_id"],
                "sr_top1": top1,
                "sr_top2": top2,
                "delta": top1 - top2,
            }
        else:
            out[str(k)] = {
                "winner": None,
                "reason": f"inconclusive (Δ={top1 - top2:.3f} < 0.05)",
                "top1_yaml": ranked[0]["yaml_id"],
                "top2_yaml": ranked[1]["yaml_id"],
            }
    return out


def _pareto_frontier(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Upper-left Pareto front on (inf, sr)."""
    pts = sorted(
        [
            (
                _compute_inf(
                    r.get("n_full_hit", 0),
                    r.get("n_warm_start", 0),
                    r.get("n_miss", 0),
                ),
                r.get("success_rate") or 0.0,
                r,
            )
            for r in rows
        ],
        key=lambda x: x[0],
    )
    front: list[dict[str, Any]] = []
    best = -1.0
    for inf, sr, r in pts:
        if sr > best:
            front.append({"yaml_id": r["yaml_id"], "inf": inf, "sr": sr})
            best = sr
    return front


def _mode_analyze(args: argparse.Namespace) -> None:
    """Read per_yaml_summary.jsonl, write 5 decision JSONs + Pareto figure."""
    summary_path = (
        Path(args.summary)
        if args.summary
        else DEFAULT_DATA_DIR / "per_yaml_summary.jsonl"
    )
    if not summary_path.exists():
        raise SystemExit(f"per_yaml_summary not found: {summary_path}")

    rows = [
        json.loads(line)
        for line in summary_path.read_text().splitlines()
        if line.strip()
    ]

    # ----- decision gate per phase5 rules ---------------------------
    def _axis_split(axis_tag: str) -> list[str]:
        return axis_tag.split("__")[0].split("_")

    out_dir = summary_path.parent
    gate: dict[str, Any] = {}
    g1 = [r for r in rows if r.get("group") == "g1"]
    if g1:
        gate["g1"] = _winner_rule_5pp(
            g1, lambda r: tuple(_axis_split(r["axis_tag"])[:3])
        )
    g2 = [r for r in rows if r.get("group") == "g2"]
    if g2:
        gate["g2"] = _winner_rule_5pp(
            g2, lambda r: tuple(_axis_split(r["axis_tag"])[:3])
        )
    g3 = [r for r in rows if r.get("group") == "g3"]
    if g3:
        gate["g3"] = _winner_rule_5pp(
            g3, lambda r: tuple(_axis_split(r["axis_tag"])[:2])
        )
    g4 = [r for r in rows if r.get("group") == "g4"]
    if g4:
        gate["g4"] = _winner_rule_5pp(
            g4, lambda r: tuple(_axis_split(r["axis_tag"])[:2])
        )
    g5 = [r for r in rows if r.get("group") == "g5"]
    if g5:
        per_recipe: dict[str, Any] = {}
        for recipe_short in ("p1", "p2", "g6"):
            sub = [r for r in g5 if r.get("base_recipe") == recipe_short]
            if not sub:
                continue
            ranked = sorted(sub, key=lambda r: -(r.get("success_rate") or 0.0))
            cheap_above = sorted(
                [r for r in sub if (r.get("success_rate") or 0.0) >= 0.85],
                key=lambda r: _compute_inf(
                    r.get("n_full_hit", 0),
                    r.get("n_warm_start", 0),
                    r.get("n_miss", 0),
                ),
            )
            per_recipe[recipe_short] = {
                "best_sr_yaml": ranked[0]["yaml_id"],
                "best_sr": ranked[0].get("success_rate"),
                "cheapest_above_0.85_yaml": cheap_above[0]["yaml_id"]
                if cheap_above
                else None,
                "frontier": _pareto_frontier(sub),
            }
        gate["g5"] = per_recipe

    for g in ("g1", "g2", "g3", "g4", "g5"):
        out_path = out_dir / f"{g}_decision.json"
        payload = gate.get(g, {})
        out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        logger.info("[analyze] wrote %s", out_path)

    # ----- Pareto overlay figure ------------------------------------
    try:
        from exp.weighted_sum.kinematic.analysis.plot_pareto_overlay import (
            main as plot_main,
        )

        plot_main(summary_path=summary_path, out_dir=out_dir)
    except ImportError as e:
        logger.warning("[analyze] skipped Pareto plot (matplotlib missing?): %s", e)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def _parse(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="kinematic_runner")
    p.add_argument("--mode", required=True, choices=VALID_MODES)
    p.add_argument(
        "--host", default="weiland.top", help="server host (default: weiland.top)"
    )
    p.add_argument(
        "--port",
        type=int,
        default=14000,
        help="server port (default: 14000 → ziyang10)",
    )
    p.add_argument(
        "--trials-per-task",
        type=int,
        default=15,
        help="super warmup trials per task (15 → ~150 ep ≈ 3250 verdict)",
    )
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument(
        "--server-workers",
        default="",
        help="run-eval per-server worker split, e.g. 40,40 for dual server",
    )
    p.add_argument("--task-suite", default="libero_spatial")
    p.add_argument("--warmup-trials", type=int, default=15)
    p.add_argument("--eval-trials", type=int, default=10)
    p.add_argument("--task-ids", default="0,1,2,3,4,5,6,7,8,9")
    p.add_argument("--per-step-log-dir", default="")
    p.add_argument("--episode-results-dir", default="")
    p.add_argument("--episode-filter", default="")
    p.add_argument("--cuda-visible-devices", default="")
    p.add_argument("--conda-env", default="")
    p.add_argument("--preload-pkl-override", default="")
    p.add_argument(
        "--cfg-id",
        default="spatial16_ws_d1_best",
        help="v2_spec CFG_SPECS key for the kinematic base "
        "(libero_10 Option B run: spatial16_ws_d1_best_libero10)",
    )

    p.add_argument("--warmup-yaml", default="", help="super warmup yaml path")
    p.add_argument("--super-raw", default="", help="super warmup raw jsonl path")
    p.add_argument(
        "--super-raw-relpath",
        default=SUPER_WARMUP_RAW_RELPATH,
        help="server-side relative path for offline calibration (default: kinematic_phase5/super_warmup_raw.jsonl)",
    )
    p.add_argument("--eval-dir", default="")
    p.add_argument("--always-warm-dir", default="")
    p.add_argument("--thresholds-dir", default="")
    p.add_argument("--summary", default="")
    p.add_argument(
        "--servers",
        default="weiland.top:14000,weiland.top:14001",
        help="comma-separated host:port for run-eval delegate command",
    )
    p.add_argument(
        "--data-dir",
        default="",
        help="aggregate-summary: data dir holding journal + per_step "
        "(default: exp/weighted_sum/data/libero_spatial/kinematic_phase5/d1)",
    )
    p.add_argument(
        "--journal",
        default="",
        help="aggregate-summary: per-episode eval journal path (default: <data_dir>/journal.jsonl)",
    )
    p.add_argument(
        "--per-step-dir",
        default="",
        help="aggregate-summary: per-yaml verdict log dir (default: <data_dir>/per_step)",
    )
    p.add_argument(
        "--always-warm-journal",
        default="",
        help="aggregate-summary: Stage 6 always-warm journal (default: <data_dir>/always_warm_journal.jsonl)",
    )
    return p.parse_args(argv)


_DISPATCH = {
    "emit-warmup": _mode_emit_warmup,
    "run-warmup": _mode_run_warmup,
    "verify-raw": _mode_verify_raw,
    "emit-eval-yamls": _mode_emit_eval_yamls,
    "run-always-warm": _mode_run_always_warm,
    "run-eval": _mode_run_eval,
    "aggregate-summary": _mode_aggregate_summary,
    "analyze": _mode_analyze,
}


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    args = _parse(argv)
    _DISPATCH[args.mode](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
