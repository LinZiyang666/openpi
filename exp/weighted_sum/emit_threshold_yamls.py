"""Emit warmup + eval ThresholdJudge YAMLs for the SR x inference_ratio sweep.

Takes an already-validated weighted_sum base YAML (the chosen best config, e.g.
the wsweep winner) and swaps ONLY ``checkpoints.cp1.judge`` — everything else
(keys, weights, per-field zscore normalizers, trajectory_depth, preload,
write_policy) is inherited verbatim from the base.

- ``warmup``: judge = threshold@2.0 (no warm tiers) -> forces MISS so the robot
  follows the real policy trajectory while search still computes cp1_score
  (collected via run_phase2 --per-step-out).
- ``eval``: for each (fh_ratio, ws_ratio) cell, judge =
  {type: threshold, threshold: T_fh, warm_tiers: [{threshold: T_ws, start_t: 0.5}]}.
  Degenerate cells (solve returned None) are skipped.
- ``anchor``: a single threshold@2.0 eval yaml = all-MISS raw-policy inf_ratio=1 point.

Every emitted YAML is round-tripped through ``load_cache_config`` (strict
warm-tier ordering + judge.threshold correctness) before it is accepted.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import yaml

from openpi.cache.config import load_cache_config

from exp.weighted_sum.solve_thresholds import grid_cells, load_cp1_scores, solve_quantile, solve_zscore

_FORCE_MISS = 2.0  # > [0,1] score range -> threshold judge never FULL_HIT/WARM


def _with_judge(base_cfg: dict, judge: dict) -> dict:
    cfg = copy.deepcopy(base_cfg)
    cfg["checkpoints"]["cp1"]["judge"] = judge
    return cfg


def _write(cfg: dict, out_dir: Path, yaml_id: str) -> None:
    path = out_dir / f"{yaml_id}.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    load_cache_config(path)  # strict schema self-check (warm-tier ordering etc.)


def main():
    repo = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description="Emit threshold warmup/eval/anchor YAMLs")
    ap.add_argument("--base-yaml", required=True, help="chosen weighted_sum base config (judge swapped out)")
    ap.add_argument("--mode", choices=["warmup", "eval", "anchor"], required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--warmup-scores", help="[eval] per-step JSONL from warmup (for threshold solve)")
    ap.add_argument("--method", choices=["quantile", "zscore"], default="quantile")
    ap.add_argument("--start-t", type=float, default=0.5)
    ap.add_argument(
        "--fh-ratios", default="0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.6,0.7,0.8",
        help="[eval] comma f_FH (FULL_HIT) target fractions; denser at the low end for frontier "
        "detail, extends past 0.5 for aggressive (low inference_ratio) operating points.",
    )
    ap.add_argument(
        "--ws-ratios", default="0.05,0.1,0.2,0.3,0.4",
        help="[eval] comma f_WS (WARM_START) target fractions (secondary axis).",
    )
    ap.add_argument(
        "--max-total", type=float, default=0.9,
        help="[eval] keep only cells with f_FH+f_WS <= max-total — enforces feasibility "
        "(>1 impossible) and leaves a MISS margin so high-f_FH cells don't collapse into the "
        "degenerate bottom tail. Makes the grid triangular past f_FH~0.5.",
    )
    args = ap.parse_args()

    base_cfg = yaml.safe_load(Path(args.base_yaml).read_text(encoding="utf-8"))
    stem = Path(args.base_yaml).stem
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "warmup":
        _write(_with_judge(base_cfg, {"type": "threshold", "threshold": _FORCE_MISS}), out_dir, f"{stem}__warmup")
        print(f"wrote 1 warmup YAML (force-MISS @ {_FORCE_MISS}) to {out_dir}")
        return

    if args.mode == "anchor":
        _write(_with_judge(base_cfg, {"type": "threshold", "threshold": _FORCE_MISS}), out_dir, f"{stem}__anchor_allmiss")
        print(f"wrote 1 all-MISS anchor YAML (inf_ratio=1 raw-policy) to {out_dir}")
        return

    # eval: solve thresholds from warmup scores, emit one yaml per non-degenerate cell
    if not args.warmup_scores:
        raise SystemExit("eval mode requires --warmup-scores")
    scores = load_cp1_scores(args.warmup_scores)
    if len(scores) < 10:
        raise SystemExit(f"only {len(scores)} warmup cp1_scores — too few to solve thresholds")
    fh_ratios = tuple(float(x) for x in args.fh_ratios.split(","))
    ws_ratios = tuple(float(x) for x in args.ws_ratios.split(","))
    cells = grid_cells(fh_ratios, ws_ratios, args.max_total)
    n_written, n_skipped = 0, 0
    for fh, ws in cells:
        solved = (
            solve_quantile(scores, fh, ws) if args.method == "quantile"
            # zscore: map ratios to k via simple symmetric offsets (k_fh>k_ws)
            else solve_zscore(scores, k_fh=(0.5 - fh), k_ws=(0.5 - fh - ws))
        )
        if solved is None:
            n_skipped += 1
            print(f"  skip degenerate cell fh={fh} ws={ws} (T_fh-T_ws<eps)")
            continue
        t_fh, t_ws = solved
        judge = {"type": "threshold", "threshold": round(t_fh, 6),
                 "warm_tiers": [{"threshold": round(t_ws, 6), "start_t": args.start_t}]}
        _write(_with_judge(base_cfg, judge), out_dir, f"{stem}__fh{int(fh*100)}_ws{int(ws*100)}_{args.method}")
        n_written += 1
    print(f"wrote {n_written} eval YAMLs ({n_skipped} degenerate cells skipped, method={args.method}) to {out_dir}")


if __name__ == "__main__":
    main()
