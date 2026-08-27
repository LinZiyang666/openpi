"""Teacher noise-sensitivity diagnostic (preliminary; does not drive the pipeline).

For a stratified sample of fit-split steps, repeat full generation M times at
the same observation with independent, seed-logged initial noises and measure
the pairwise teacher-teacher deviation; contrast it with the warm-teacher
deviation (tau=7 completion from the retrieval winner vs one teacher sample).
Reported as the preliminary diagnostic promised in the dispatch note; the
calibration ref-mode is fixed to fresh regardless of this outcome.

Usage:
  uv run python -m exp.dispatch_surface.noise_sensitivity \
      --table exp/dispatch_surface/data/dispatch_table_fresh.jsonl \
      --query-h5-dir exp/dispatch_surface/data/query_cohort \
      --library-pkl <pkl> --cache-yaml <yaml> \
      --config-name pi05_libero --checkpoint-dir <ckpt> \
      --n-steps 50 --m-samples 8 \
      --out-dir exp/dispatch_surface/analysis/noise_sensitivity
"""

from __future__ import annotations

import argparse
import itertools
import json
import pathlib
import random

import h5py
import numpy as np
import torch

from openpi.cache.components.surface_judge import weighted_chunk_deviation


def _sample_rows(table_path: str, n_steps: int, seed: int) -> list[dict]:
    """Task-stratified sample of fit-split rows from the calibration table."""
    rows = [json.loads(line) for line in open(table_path)]
    fit_rows = [r for r in rows if r["split"] == "fit"]
    by_task: dict[int, list[dict]] = {}
    for r in fit_rows:
        by_task.setdefault(r["task_id"], []).append(r)
    rng = random.Random(seed)
    per_task = max(1, n_steps // max(1, len(by_task)))
    sampled: list[dict] = []
    for task_id in sorted(by_task):
        pool = by_task[task_id]
        sampled.extend(rng.sample(pool, min(per_task, len(pool))))
    return sampled[:n_steps]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--table", required=True, help="fresh-mode calibration table JSONL")
    ap.add_argument("--query-h5-dir", required=True)
    ap.add_argument("--library-pkl", required=True)
    ap.add_argument("--cache-yaml", required=True)
    ap.add_argument("--config-name", default="pi05_libero")
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--weights-npz", required=True, help="W/active_mask from build_dispatch_table")
    ap.add_argument("--n-steps", type=int, default=50)
    ap.add_argument("--m-samples", type=int, default=8)
    ap.add_argument("--h-exec", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20260827)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    from exp.common.build_in_memory_cache_artifact import (
        _build_fake_stage1_with_masks,
        _load_pi05_for_llm_extract,
    )
    from openpi.cache.shadow_teacher import stable_seed

    weights = np.load(args.weights_npz)
    w = torch.from_numpy(weights["w"])
    active = torch.from_numpy(weights["active_mask"])

    sampled = _sample_rows(args.table, args.n_steps, args.seed)
    model, tokenizer = _load_pi05_for_llm_extract(args.checkpoint_dir, args.config_name, args.device)
    dev = torch.device(args.device)

    h5_dir = pathlib.Path(args.query_h5_dir)
    h5_by_stem = {p.stem: p for p in h5_dir.rglob("*.h5")}

    tt_devs: list[float] = []      # teacher-teacher pairwise deviations
    wt_devs: list[float] = []      # warm-teacher deviations (from the table)
    per_step_records: list[dict] = []
    for row in sampled:
        h5_path = h5_by_stem.get(row["episode_id"])
        if h5_path is None:
            raise SystemExit(f"episode {row['episode_id']} not found under {h5_dir}")
        with h5py.File(h5_path, "r") as h5:
            task_str = h5.attrs.get("prompt") or h5.attrs.get("task", "")
            group = h5[f"step_{row['step_idx']:04d}"]
            stage1 = _build_fake_stage1_with_masks(group, str(task_str), tokenizer, model, dev)
            with torch.no_grad():
                stage2 = model.run_stage2(stage1)
                samples = []
                horizon_dim = np.array(group["clean_action"]).shape
                for m in range(args.m_samples):
                    gen = torch.Generator(device=dev)
                    gen.manual_seed(stable_seed(row["episode_id"], m, row["step_idx"]))
                    noise = model.sample_noise((1, *horizon_dim), dev, generator=gen)
                    samples.append(model.run_stage3(stage2, noise=noise).action_chunk[0].float().cpu())
        step_tt = [
            weighted_chunk_deviation(a, b, w, active, args.h_exec)
            for a, b in itertools.combinations(samples, 2)
        ]
        tt_devs.extend(step_tt)
        wt_devs.append(row["y_tau7"])
        per_step_records.append({
            "episode_id": row["episode_id"], "step_idx": row["step_idx"],
            "tt_median": float(np.median(step_tt)), "wt": row["y_tau7"],
        })

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "n_steps": len(sampled),
        "m_samples": args.m_samples,
        "tt_median": float(np.median(tt_devs)),
        "tt_p95": float(np.percentile(tt_devs, 95)),
        "wt_median": float(np.median(wt_devs)),
        "wt_p95": float(np.percentile(wt_devs, 95)),
        "ratio_median": float(np.median(tt_devs) / max(np.median(wt_devs), 1e-12)),
        "note": (
            "Preliminary diagnostic only. The calibration ref-mode is fresh "
            "regardless of this ratio; uncoupled/tau1 are sensitivity columns."
        ),
    }
    (out_dir / "noise_sensitivity.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "per_step.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in per_step_records)
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
