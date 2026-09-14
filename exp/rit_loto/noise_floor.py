"""Two-seed noise floor of the shadow deviation on the W13 corpus.

For a stratified sample of stored decisions, run the full flow-matching loop
twice from two independently seeded noises under the reconstructed observation
and measure ``D(ref1, ref2)`` with the same W / mask / H_exec the shadow table
uses. Its median is the scale the parity gate of ``build_loto_table.py``
compares against. It describes the magnitude of the teacher's own randomness
under the reconstructed observations of this sample; it is not a mathematical
lower bound on ``delta`` for every observation or candidate, and the plan does
not read it as one. ``D(ref1, clean_action)`` is recorded alongside as a third,
online-drawn sample; the two are reported separately and never pooled.

Seeds are row-stable: ``derive_seed(root, {suite, trajectory_id, decision_id,
tag})`` for ``tag in (noise_floor_ref1, noise_floor_ref2)``.

Environment: the GR00T island. Usage:
  python -m exp.rit_loto.noise_floor --suite libero_10 --corpus-dir ... \
      --library-pkl ... --template-yaml ... --checkpoint ... --out-dir <data>/libero_10
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
import time

import h5py
import numpy as np
import torch

from openpi.cache.components.surface_judge import weighted_chunk_deviation
from openpi.cache.types import groot_n15_schedule
from openpi.collect.h5_intermediates import episode_schedule

from exp.rit_loto.build_loto_table import (
    DEFAULT_DENOISING_STEPS,
    DEFAULT_H_EXEC,
    DEFAULT_WARM_TS,
    NOISE_CONTRACT,
    NUM_TASKS,
    SUITES,
    ModelSide,
    build_identity,
    checkpoint_identity,
    corpus_files,
    corpus_manifest,
    derive_seed,
    enabled_fields,
    h5_chunk,
    load_library,
    load_template,
    make_noise,
    stage2_in_session,
    stratified_decisions,
    to_chunk,
    write_jsonl,
)
from exp.robocasa365.rit_shadow import library_action_weights

logger = logging.getLogger("rit_loto.noise_floor")
DEFAULT_PER_TASK = 50
TAGS = ("noise_floor_ref1", "noise_floor_ref2")


def summarize(values: list[float]) -> dict:
    """n / median / p90 / p95 / mean / min / max of a finite sample; empty or non-finite input is an error."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0 or not np.isfinite(arr).all():
        raise SystemExit("noise floor values are empty or non-finite")
    return {"n": int(arr.size), "median": float(np.quantile(arr, 0.5)), "p90": float(np.quantile(arr, 0.9)),
            "p95": float(np.quantile(arr, 0.95)), "mean": float(arr.mean()), "min": float(arr.min()),
            "max": float(arr.max())}


def floor_row(runner, templates, task: str, group, *, seeds: tuple[int, int], w, mask, h_exec: int) -> dict:
    """``D(ref1, ref2)`` and ``D(ref1, clean_action)`` for one stored decision."""
    clean = h5_chunk(group, "clean_action")
    z1, z2 = make_noise(seeds[0], (1,) + tuple(clean.shape)), make_noise(seeds[1], (1,) + tuple(clean.shape))
    with runner.session():
        _, stage2 = stage2_in_session(runner, templates, task, group)
        ref1 = runner.run_stage3(stage2, noise=z1).action_pred
        ref2 = runner.run_stage3(stage2, noise=z2).action_pred
    a, b = to_chunk(ref1), to_chunk(ref2)
    return {"d_ref1_ref2": weighted_chunk_deviation(a, b, w, mask, h_exec),
            "d_ref1_clean": weighted_chunk_deviation(a, clean, w, mask, h_exec)}


def main() -> None:
    """CLI: stratified two-seed noise floor of one suite, written with the full input identity."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", required=True, choices=SUITES)
    ap.add_argument("--corpus-dir", required=True)
    ap.add_argument("--library-pkl", required=True)
    ap.add_argument("--template-yaml", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--denoising-steps", type=int, default=DEFAULT_DENOISING_STEPS)
    ap.add_argument("--h-exec", type=int, default=DEFAULT_H_EXEC)
    ap.add_argument("--per-task", type=int, default=DEFAULT_PER_TASK)
    ap.add_argument("--sample-seed", type=int, default=20260914)
    ap.add_argument("--root-seed", type=int, default=20260914)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    schedule = groot_n15_schedule(args.denoising_steps)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg, _ = load_template(args.template_yaml, args.library_pkl, out_dir)
    library = load_library(args.library_pkl, expected_schedule=schedule, warm_ts=DEFAULT_WARM_TS)
    w, mask = library_action_weights(args.library_pkl)
    w = torch.as_tensor(np.asarray(w), dtype=torch.float32)
    mask = torch.as_tensor(np.asarray(mask), dtype=torch.bool)
    enabled, weights = enabled_fields(cfg)
    files = corpus_files(args.corpus_dir)
    identity = build_identity(
        suite=args.suite, library=library, library_path=args.library_pkl, template_path=args.template_yaml,
        ckpt=checkpoint_identity(args.checkpoint), corpus=corpus_manifest(files), corpus_dir=args.corpus_dir,
        w=w, mask=mask, h_exec=args.h_exec, schedule=schedule, warm_ts=DEFAULT_WARM_TS, enabled=enabled,
        weights=weights,
    )
    model = ModelSide(args.checkpoint, denoising_steps=args.denoising_steps, device=args.device)
    if model.schedule != schedule:
        raise SystemExit(f"live schedule {model.schedule.schedule_id} != requested {schedule.schedule_id}")

    sample = stratified_decisions(files, args.per_task, args.sample_seed, num_tasks=NUM_TASKS)
    by_stem: dict[str, list[dict]] = {}
    for d in sample:
        by_stem.setdefault(d["trajectory_id"], []).append(d)
    rows: list[dict] = []
    t0 = time.time()
    for stem, picks in by_stem.items():
        with h5py.File(pathlib.Path(args.corpus_dir) / f"{stem}.h5", "r") as f:
            if episode_schedule(f) != schedule:
                raise SystemExit(f"{stem}: file schedule differs from {schedule.schedule_id}")
            for d in picks:
                ident = {"suite": args.suite, "trajectory_id": stem, "decision_id": int(d["decision_id"])}
                seeds = tuple(derive_seed(args.root_seed, {**ident, "tag": tag}) for tag in TAGS)
                g = f[f"step_{d['decision_id']:04d}"]
                rows.append({**d, "suite": args.suite, "seed_ref1": seeds[0], "seed_ref2": seeds[1],
                             **floor_row(model.runner, model.templates, d["task"], g, seeds=seeds, w=w, mask=mask,
                                         h_exec=args.h_exec)})
    write_jsonl(out_dir / "noise_floor_rows.jsonl", rows)
    record = {
        "protocol": "rit_loto_noise_floor_v1",
        "identity": identity,
        "noise_contract": {"version": NOISE_CONTRACT, "root_seed": int(args.root_seed), "tags": list(TAGS),
                           "shape": [1, 16, 32], "torch": torch.__version__,
                           "derivation": "sha256(canonical_json({root_seed, suite, trajectory_id, decision_id, tag}))[:8] big-endian"},
        "sample": {"per_task": args.per_task, "sample_seed": args.sample_seed, "n_rows": len(rows)},
        "n_active_dims": int(mask.sum()),
        "d_ref1_ref2": summarize([r["d_ref1_ref2"] for r in rows]),
        "d_ref1_clean": summarize([r["d_ref1_clean"] for r in rows]),
        "rows_path": str(out_dir / "noise_floor_rows.jsonl"),
        "elapsed_s": time.time() - t0,
    }
    (out_dir / "noise_floor.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    logger.info("noise floor %s: D(ref1,ref2) median %.4f p90 %.4f; D(ref1,clean) median %.4f (n=%d, %.0fs)",
                args.suite, record["d_ref1_ref2"]["median"], record["d_ref1_ref2"]["p90"],
                record["d_ref1_clean"]["median"], len(rows), time.time() - t0)


if __name__ == "__main__":
    main()
