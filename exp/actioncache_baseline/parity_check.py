"""Offline parity of the CP2 key: H5-rebuilt Stage 1 vs. the real Stage 1 (plan §3.8).

Two paths to the same key for a sampled step:
  (i)  offline library path — Stage 1 re-assembled from the stored
       ``vision_*`` / ``prompt_emb`` / ``robot_state`` tensors
       (``_build_fake_stage1_with_masks``), then ``run_stage2_capture`` and
       the CP2 builder;
  (ii) online-equivalent path — the raw ``input_images`` (the transformed
       224x224 uint8 frames the model saw online), the task string and the
       stored state are turned back into a model ``Observation`` and pushed
       through the real ``run_stage1``, then the same capture + builder.

The only difference between the paths is the fp16 storage of the Stage-1
tensors, so the keys must agree to ``cosine >= --min-cosine`` (default
0.999). The max deviation and the per-step cosines are written out; the
script exits non-zero on the first violation (this is the build-time gate of
plan §8 step 2; a failure means the library must be rebuilt through path (ii)).

Usage:
  uv run python -m exp.actioncache_baseline.parity_check \\
      --h5-root <collection dir> --config-name pi05_libero --checkpoint-dir <ckpt> \\
      --expect-weights-digest <artifact model.weights_digest> \\
      --seed 20260904 --samples 200 --out <parity.json>
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random

import h5py
import torch

from exp.actioncache_baseline import libs
from exp.common.build_in_memory_cache_artifact import (
    _build_fake_stage1_with_masks,
    _load_pi05_for_llm_extract,
    _self_check_tokenizer_consistency,
)
from exp.actioncache_baseline.stage1_paths import observation_from_h5
from openpi.cache.components.cp2_vlm_key_builder import CP2VlmTernaryKeyBuilder
from openpi.cache.types import CheckpointID


def run(args: argparse.Namespace) -> dict:
    if not getattr(args, "expect_weights_digest", ""):
        raise SystemExit("--expect-weights-digest is required for fail-closed artifact/model binding")
    # ``run`` is also imported by experiment tooling, so enforce the binding
    # here as well as in argparse before loading the model or reading samples.
    libs.assert_model_binding({"weights_digest": args.expect_weights_digest}, args.checkpoint_dir)
    device = torch.device(args.device)
    model, tokenizer = _load_pi05_for_llm_extract(args.checkpoint_dir, args.config_name, args.device)
    model.eval()
    builder = CP2VlmTernaryKeyBuilder(seed=args.seed, d=args.d, p=args.p, input_dim=args.input_dim)
    files = sorted(pathlib.Path(args.h5_root).rglob("*.h5"))
    if not files:
        raise SystemExit(f"no H5 under {args.h5_root}")
    rng = random.Random(args.sample_seed)
    steps: list[tuple[pathlib.Path, str]] = []
    for p in files:
        with h5py.File(p, "r") as f:
            for _idx, g in libs.iter_steps(f):
                if "input_images" in g:
                    steps.append((p, g.name))
    if not steps:
        raise SystemExit("no step with input_images found; cannot run path (ii)")
    sample = rng.sample(steps, min(args.samples, len(steps)))
    results = []
    worst = 1.0
    with torch.no_grad():
        _self_check_tokenizer_consistency(sample[0][0], model, tokenizer, device)
        for p, gname in sample:
            with h5py.File(p, "r") as f:
                task = str(f.attrs.get("task", ""))
                g = f[gname]
                fake = _build_fake_stage1_with_masks(g, task_str=task, tokenizer=tokenizer, model=model, device=device)
                k_off = _key(builder, model.run_stage2_capture(fake))
                obs = observation_from_h5(g, task, tokenizer, model, device)
                k_on = _key(builder, model.run_stage2_capture(model.run_stage1(obs)))
            cos = float(torch.nn.functional.cosine_similarity(k_off, k_on, dim=0))
            max_abs = float((k_off - k_on).abs().max())
            worst = min(worst, cos)
            results.append({"h5": str(p), "step": gname, "cosine": cos, "max_abs": max_abs})
            if cos < args.min_cosine:
                rec = _record(args, results, worst, ok=False)
                libs.dump_json(args.out, rec)
                raise SystemExit(f"parity violation at {p}:{gname}: cosine={cos:.6f} < {args.min_cosine}")
    rec = _record(args, results, worst, ok=True)
    libs.dump_json(args.out, rec)
    return rec


def _key(builder: CP2VlmTernaryKeyBuilder, stage2) -> torch.Tensor:
    builder.collect(CheckpointID.CP2, stage2=stage2)
    k = builder.build(CheckpointID.CP2)[libs.FIELD]
    builder.clear()
    return k


def _record(args, results, worst, *, ok) -> dict:
    return {"protocol": libs.PROTOCOL, "h5_root": str(pathlib.Path(args.h5_root).resolve()),
            "config_name": args.config_name, "checkpoint_dir": str(pathlib.Path(args.checkpoint_dir).resolve()),
            "model": libs.weights_digest(args.checkpoint_dir),
            "expect_weights_digest": args.expect_weights_digest,
            "projection": {"seed": args.seed, "d": args.d, "p": args.p, "input_dim": args.input_dim},
            "samples": len(results), "min_cosine_required": args.min_cosine, "min_cosine_observed": worst,
            "max_abs_observed": max((r["max_abs"] for r in results), default=None),
            "ok": ok, "results": results, "git_commit": libs.git_commit()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h5-root", required=True)
    ap.add_argument("--config-name", default="pi05_libero")
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--d", type=int, default=libs.ProjectionArgs.d)
    ap.add_argument("--p", type=float, default=libs.ProjectionArgs.p)
    ap.add_argument("--input-dim", type=int, default=libs.ProjectionArgs.input_dim)
    ap.add_argument("--samples", type=int, default=200)
    ap.add_argument("--sample-seed", type=int, default=0)
    ap.add_argument("--min-cosine", type=float, default=0.999)
    ap.add_argument("--expect-weights-digest", required=True,
                    help="artifact model.weights_digest; the checkpoint must re-derive it")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rec = run(args)
    print(json.dumps({k: rec[k] for k in ("samples", "min_cosine_observed", "max_abs_observed", "ok")}))


if __name__ == "__main__":
    main()
