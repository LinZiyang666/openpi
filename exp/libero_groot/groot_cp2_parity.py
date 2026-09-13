"""Parity gate for the GR00T CP2 key (plan §3.7) -- manual, on the GR00T island with the real model.

Four checks, each recorded and each fail-closed (a failure stops the line
before any full library is built):

1. Synthetic observation (random images, random non-zero state, a real task
   string): path (A) ``run_stage1 -> run_stage2_llm -> run_cp2_key_source``
   versus path (B) the collector's slicing of the same stage-1 sequence
   (``slice_groot_cp1_fields``, fp16 round trip into an in-memory HDF5 group)
   -> ``cp2_reconstruct.reconstruct_stage1`` -> the same chain. The rebuilt
   ``input_embeds`` and the valid state must be bit-identical, the encoded
   state segment of the flattened key source bit-identical, the encoded VLM
   segment is recorded bit-equal or not (attention kernels may pick a
   different reduction order for different strides), and the projected keys
   must agree to cosine >= 0.999.
2. Negatives: same images / text, a different state -> both reconstructions
   read their own state back, and their state segments differ; a step group
   reconstructs its own state, not its neighbour's; a group without
   ``robot_state`` raises.
3. Real library steps (``--h5-root``, ``--samples`` steps): the text-position
   and state read-back assertions of the reconstruction, plus a direct
   comparison of ``run_cp2_key_source`` against the head's own encoder calls
   (``process_backbone_output`` / ``state_encoder``) -- shape, dtype, values --
   with the live head config and the weights digest recorded.
4. Helper purity and path equivalence: ``run_cp2_key_source`` leaves the
   stage-2 features, the action inputs and the CPU / CUDA RNG state
   unchanged; with the same seed a MISS through ``run_stage3`` after the
   helper equals the plain ``run_stage2`` teacher action bit for bit; with the
   same snapshot a WARM_START@0.875 through ``run_stage3_from`` after the
   helper equals the resume path without it.

The instruction strings come from the benchmark task map (``emit_task_map.py``,
``task.language``) and are cross-checked against the ``task`` attr of the
sampled H5 files -- never from the shadow manifest's canonical ``task_name``.

Usage:
  python -m exp.libero_groot.groot_cp2_parity --suite libero_spatial --checkpoint <ckpt> \\
      --task-map <task_map.json> --h5-root /archive/libero_cache/build_spatial_w13/libero_spatial \\
      --samples 20 --out <parity.json>
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
from typing import Any

import h5py
import numpy as np
import torch

from openpi.cache.groot.cp2_key_builder import GrootCP2TernaryKeyBuilder, flatten_source
from openpi.cache.groot.key_builder import slice_groot_cp1_fields
from openpi.cache.groot.staged import GrootStagedRunner, _batch_feature
from openpi.cache.types import PROMPT_EMB, ROBOT_STATE, CheckpointID

from exp.actioncache_baseline import libs
from exp.libero_groot import libero_keys as K
from exp.libero_groot.emit_task_map import load_task_map
from exp.libero_groot.cp2_reconstruct import (
    LIBERO_VISION_FIELDS,
    TemplateCache,
    h5_task,
    load_groot_libero_policy,
    normalized_input_for,
    reconstruct_stage1,
)

PROFILE = libs.GROOT_LIBERO
MIN_COSINE = 0.999


class ParityError(RuntimeError):
    """One of the four parity checks failed; the record is written with the reason before exiting."""


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.nn.functional.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0))


def random_wire_observation(rng: np.random.Generator, task: str) -> dict[str, Any]:
    """A legal wire observation with random images and a random non-zero state carrying ``task``."""
    res = K.WIRE_IMAGE_RESOLUTION
    return {
        K.WIRE_IMAGE: rng.integers(0, 256, (res, res, 3), dtype=np.uint8),
        K.WIRE_WRIST: rng.integers(0, 256, (res, res, 3), dtype=np.uint8),
        K.WIRE_STATE: rng.uniform(-0.5, 0.5, K.WIRE_STATE_DIM).astype(np.float64) + 0.05,
        K.WIRE_PROMPT: task,
    }


def slice_to_memory_group(stage1, h5_file: h5py.File, name: str) -> h5py.Group:
    """Store one step the way ``GrootCacheCollector`` does (fp16 embeddings, fp32 state)."""
    raw = slice_groot_cp1_fields(
        stage1.input_embeds, stage1.image_token_mask, stage1.state, stage1.state_mask,
        None, vision_fields=LIBERO_VISION_FIELDS,
    )
    g = h5_file.create_group(name)
    for field in LIBERO_VISION_FIELDS:
        g.create_dataset(field, data=raw[field].cpu().to(torch.float16).numpy())
    g.create_dataset(PROMPT_EMB, data=raw[PROMPT_EMB].cpu().to(torch.float16).numpy())
    g.create_dataset(ROBOT_STATE, data=raw[ROBOT_STATE].cpu().float().numpy())
    return g


def key_chain(runner, builder, stage1):
    """stage-1 output -> ``(stage2, key source, flattened source, projected key)``, inside the session."""
    stage2 = runner.run_stage2_llm(stage1)
    src = runner.run_cp2_key_source(stage2)
    h = flatten_source(src.vl_encoded, src.state_encoded, token_len=builder.token_len,
                       feature_dim=builder.feature_dim, state_feat_dim=builder.state_feat_dim)
    builder.collect(CheckpointID.CP2, cp2_source=src, stage2=stage2)
    key = builder.build(CheckpointID.CP2)[libs.FIELD]
    builder.clear()
    return stage2, src, h, key


def check_synthetic(policy, runner, builder, templates, tasks: list[str], rng, n: int) -> dict:
    """Item 1: path A (online stage 1) vs path B (collector slicing -> fp16 -> reconstruction)."""
    results = []
    worst = 1.0
    with runner.session():
        for i in range(n):
            task = tasks[i % len(tasks)]
            obs = random_wire_observation(rng, task)
            stage1_a = runner.run_stage1(normalized_input_for(policy, obs))
            _, src_a, h_a, key_a = key_chain(runner, builder, stage1_a)
            with h5py.File(f"parity_{i}.h5", "w", driver="core", backing_store=False) as mem:
                g = slice_to_memory_group(stage1_a, mem, "step_0000")
                stage1_b = reconstruct_stage1(templates.get(task), g)
                _, src_b, h_b, key_b = key_chain(runner, builder, stage1_b)
            embeds_equal = torch.equal(stage1_a.input_embeds, stage1_b.input_embeds)
            valid = stage1_a.state_mask[0, -1]
            state_equal = torch.equal(stage1_a.state[0, -1][valid], stage1_b.state[0, -1][valid])
            state_dim = builder.state_feat_dim
            state_seg_equal = torch.equal(h_a[-state_dim:], h_b[-state_dim:])
            vl_equal = torch.equal(h_a[:-state_dim], h_b[:-state_dim])
            cos = _cos(key_a, key_b)
            worst = min(worst, cos)
            results.append({"task": task, "input_embeds_equal": embeds_equal, "state_equal": state_equal,
                            "state_segment_equal": state_seg_equal, "vl_segment_equal": vl_equal,
                            "vl_segment_max_abs": float((h_a[:-state_dim] - h_b[:-state_dim]).abs().max()),
                            "key_cosine": cos})
            if not (embeds_equal and state_equal and state_seg_equal):
                raise ParityError(f"synthetic parity failed: {results[-1]}")
            if cos < MIN_COSINE:
                raise ParityError(f"synthetic key cosine {cos:.6f} < {MIN_COSINE}")
    return {"n": n, "min_cosine": worst, "all_vl_segments_bitwise": all(r["vl_segment_equal"] for r in results),
            "results": results}


def check_negatives(policy, runner, builder, templates, task: str, rng) -> dict:
    """Item 2: a different state must change the state segment; a missing state must raise."""
    out: dict[str, Any] = {}
    with runner.session():
        obs1 = random_wire_observation(rng, task)
        obs2 = dict(obs1)
        obs2[K.WIRE_STATE] = obs1[K.WIRE_STATE] + 0.1  # same images / text, different state
        s1 = runner.run_stage1(normalized_input_for(policy, obs1))
        s2 = runner.run_stage1(normalized_input_for(policy, obs2))
        with h5py.File("parity_neg.h5", "w", driver="core", backing_store=False) as mem:
            g1 = slice_to_memory_group(s1, mem, "step_0000")
            g2 = slice_to_memory_group(s2, mem, "step_0001")
            template = templates.get(task)
            r1 = reconstruct_stage1(template, g1)
            r2 = reconstruct_stage1(template, g2)
            valid = r1.state_mask[0, -1]
            st1 = r1.state[0, -1][valid].float().cpu().numpy()
            st2 = r2.state[0, -1][valid].float().cpu().numpy()
            out["state_read_back_own_group"] = bool(np.array_equal(st1, np.asarray(g1[ROBOT_STATE])) and
                                                    np.array_equal(st2, np.asarray(g2[ROBOT_STATE])))
            out["state_differs_between_groups"] = not np.array_equal(st1, st2)
            out["state_not_neighbours"] = not np.array_equal(st1, np.asarray(g2[ROBOT_STATE]))
            _, _, h1, _ = key_chain(runner, builder, r1)
            _, _, h2, _ = key_chain(runner, builder, r2)
            d = builder.state_feat_dim
            out["state_segment_differs"] = not torch.equal(h1[-d:], h2[-d:])
            out["vl_segment_same_images"] = torch.equal(h1[:-d], h2[:-d])
            g3 = mem.create_group("step_0002")
            for name in (*LIBERO_VISION_FIELDS, PROMPT_EMB):
                g3.create_dataset(name, data=np.asarray(g1[name]))
            try:
                reconstruct_stage1(template, g3)
                out["missing_state_raises"] = False
            except RuntimeError:
                out["missing_state_raises"] = True
    ok = all(out[k] for k in ("state_read_back_own_group", "state_differs_between_groups",
                              "state_not_neighbours", "state_segment_differs", "missing_state_raises"))
    if not ok:
        raise ParityError(f"negative cases failed: {out}")
    out["ok"] = ok
    return out


def check_library_steps(policy, runner, builder, templates, h5_root: str, n: int, rng: random.Random,
                        languages: set[str]) -> dict:
    """Item 3: real library steps reconstruct (text / state asserts) and the helper equals the head's encoders."""
    files = sorted(pathlib.Path(h5_root).rglob("*.h5"))
    if not files:
        raise ParityError(f"no H5 under {h5_root}")
    picks: list[tuple[pathlib.Path, str]] = []
    for path in rng.sample(files, min(len(files), n)):
        with h5py.File(path, "r") as f:
            steps = [g.name for _, g in libs.iter_steps(f)]
            picks.append((path, rng.choice(steps)))
    head = policy.model.action_head
    results = []
    with runner.session():
        for path, gname in picks:
            with h5py.File(path, "r") as f:
                task = h5_task(f)
                if task not in languages:
                    raise ParityError(f"{path}: instruction {task!r} is not a benchmark task.language of this suite")
                stage1 = reconstruct_stage1(templates.get(task), f[gname])  # asserts text + state
                stage2 = runner.run_stage2_llm(stage1)
                src = runner.run_cp2_key_source(stage2)
                # Direct comparison against the head's own encoder entry points.
                processed = head.process_backbone_output(_batch_feature({
                    "backbone_features": stage2.backbone_features, "backbone_attention_mask": stage2.attention_mask}))
                vl_direct = processed["backbone_features"][0]
                st_direct = head.state_encoder(stage2.action_inputs["state"], stage2.action_inputs["embodiment_id"])[0, -1]
                results.append({
                    "h5": str(path), "step": gname, "n_tokens": int(stage1.input_embeds.shape[1]),
                    "vl_shape": list(src.vl_encoded.shape), "vl_dtype": str(src.vl_encoded.dtype),
                    "state_shape": list(src.state_encoded.shape), "state_dtype": str(src.state_encoded.dtype),
                    "vl_equal_direct": torch.equal(src.vl_encoded, vl_direct),
                    "vl_cos_direct": _cos(src.vl_encoded, vl_direct),
                    "state_equal_direct": torch.equal(src.state_encoded, st_direct),
                })
                if not results[-1]["state_equal_direct"] or results[-1]["vl_cos_direct"] < MIN_COSINE:
                    raise ParityError(f"helper disagrees with the head's encoders: {results[-1]}")
    return {"n": len(results), "results": results,
            "all_vl_equal_direct": all(r["vl_equal_direct"] for r in results),
            "feature_dim": builder.feature_dim, "state_feat_dim": builder.state_feat_dim}


def check_helper_purity_and_paths(policy, runner, task: str, rng, warm_t: float) -> dict:
    """Item 4: the helper leaves stage 2 / inputs / RNG untouched; MISS and WARM paths are unchanged by it."""
    out: dict[str, Any] = {}
    obs = random_wire_observation(rng, task)
    with runner.session():
        stage1 = runner.run_stage1(normalized_input_for(policy, obs))
        stage2 = runner.run_stage2_llm(stage1)
        feats_before = stage2.backbone_features.clone()
        state_before = stage2.action_inputs["state"].clone()
        cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
        runner.run_cp2_key_source(stage2)
        out["features_unchanged"] = torch.equal(feats_before, stage2.backbone_features)
        out["state_unchanged"] = torch.equal(state_before, stage2.action_inputs["state"])
        out["cpu_rng_unchanged"] = torch.equal(cpu_rng, torch.get_rng_state())
        out["cuda_rng_unchanged"] = torch.equal(cuda_rng, torch.cuda.get_rng_state())
        # MISS equivalence: same seed, plain teacher vs helper-then-run_stage3.
        torch.manual_seed(1234)
        plain = runner.run_stage2(stage1).action_pred.clone()
        torch.manual_seed(1234)
        runner.run_cp2_key_source(stage2)
        with_helper = runner.run_stage3(stage2).action_pred.clone()
        out["miss_equal"] = torch.equal(plain, with_helper)
        # WARM equivalence: same snapshot with / without the helper before.
        schedule = runner.live_schedule()
        horizon = plain.shape[1]
        start_x = torch.randn(1, horizon, plain.shape[2], device=plain.device, dtype=plain.dtype)
        a = runner.run_stage3_from(stage2, start_x, warm_t, schedule=schedule).action_pred.clone()
        runner.run_cp2_key_source(stage2)
        b = runner.run_stage3_from(stage2, start_x, warm_t, schedule=schedule).action_pred.clone()
        out["warm_equal"] = torch.equal(a, b)
        out["warm_steps_run"] = schedule.remaining_steps(warm_t)
    ok = all(out[k] for k in ("features_unchanged", "state_unchanged", "cpu_rng_unchanged", "cuda_rng_unchanged",
                              "miss_equal", "warm_equal"))
    if not ok:
        raise ParityError(f"helper purity / path equivalence failed: {out}")
    out["ok"] = ok
    return out


def run(args: argparse.Namespace) -> dict:
    """Run the four checks and write the parity record; ``SystemExit`` on the first failure."""
    task_map = load_task_map(args.task_map, args.suite)
    tasks = [task_map[t]["language"] for t in sorted(task_map)]
    model_identity = libs.weights_digest(args.checkpoint)
    if args.expect_weights_digest and model_identity["weights_digest"] != args.expect_weights_digest:
        raise SystemExit("checkpoint weights digest != --expect-weights-digest")
    policy = load_groot_libero_policy(args.checkpoint, denoising_steps=args.denoising_steps, device=args.device)
    runner = GrootStagedRunner(policy.model)
    builder = GrootCP2TernaryKeyBuilder(seed=args.seed, d=args.d, p=args.p, token_len=args.token_len,
                                        feature_dim=args.feature_dim, state_feat_dim=args.state_feat_dim)
    templates = TemplateCache(policy, runner)
    np_rng = np.random.default_rng(args.sample_seed)
    py_rng = random.Random(args.sample_seed)
    record: dict[str, Any] = {
        "protocol": libs.PROTOCOL, "teacher": PROFILE.name, "suite": args.suite,
        "checkpoint": str(pathlib.Path(args.checkpoint).resolve()), "model": model_identity,
        "task_map": str(pathlib.Path(args.task_map).resolve()), "task_map_sha256": libs.sha256_file(args.task_map),
        "instructions": tasks,
        "denoising_steps": args.denoising_steps, "schedule_id": runner.live_schedule().schedule_id,
        "head_config": {k: getattr(policy.model.action_head.config, k, None)
                        for k in ("backbone_embedding_dim", "input_embedding_dim", "hidden_size", "max_state_dim",
                                  "num_inference_timesteps", "use_vlln", "add_pos_embed")},
        "projection": builder.projection_meta(), "min_cosine_required": MIN_COSINE,
        "git_commit": libs.git_commit(), "ok": False,
    }
    try:
        record["synthetic"] = check_synthetic(policy, runner, builder, templates, tasks, np_rng, args.synthetic)
        record["negatives"] = check_negatives(policy, runner, builder, templates, tasks[0], np_rng)
        record["library_steps"] = check_library_steps(policy, runner, builder, templates, args.h5_root, args.samples, py_rng,
                                                      set(tasks))
        record["helper"] = check_helper_purity_and_paths(policy, runner, tasks[1 % len(tasks)], np_rng, PROFILE.warm_start_t)
        record["ok"] = True
    except ParityError as exc:
        record["error"] = str(exc)
        libs.dump_json(args.out, record)
        raise SystemExit(f"PARITY FAILED: {exc}") from exc
    libs.dump_json(args.out, record)
    print(json.dumps({"ok": True, "synthetic_min_cos": record["synthetic"]["min_cosine"],
                      "library_steps": record["library_steps"]["n"],
                      "all_vl_bitwise": record["synthetic"]["all_vl_segments_bitwise"]}))
    return record


def main() -> None:
    """CLI entry: see the module docstring."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", required=True, choices=sorted(libs.SUITE_TAGS))
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--task-map", required=True, help="emit_task_map.py output: the ten instructions (task.language)")
    ap.add_argument("--h5-root", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--denoising-steps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260904)
    ap.add_argument("--d", type=int, default=500)
    ap.add_argument("--p", type=float, default=0.01)
    ap.add_argument("--token-len", type=int, default=640)
    ap.add_argument("--feature-dim", type=int, default=2048)
    ap.add_argument("--state-feat-dim", type=int, default=1536)
    ap.add_argument("--synthetic", type=int, default=10)
    ap.add_argument("--samples", type=int, default=20)
    ap.add_argument("--sample-seed", type=int, default=0)
    ap.add_argument("--expect-weights-digest", default="")
    ap.add_argument("--out", required=True)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
