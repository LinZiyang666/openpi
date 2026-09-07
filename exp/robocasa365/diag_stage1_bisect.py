"""Bisect the stage-1 compiled-vs-eager divergence (plan §7.4-1 / §7.4-2).

The first diagnosis showed the same divergence under ``default`` and
``reduce-overhead`` (cos_min 0.9719, relF 7.9e-2), so CUDA graphs are not the
cause and the worst token is not a low-norm one. This script localises the
divergence and calibrates its scale:

* eager vs eager (determinism floor);
* eager with autocast on vs off (the known LayerNorm fp32/bf16 effect, for scale);
* compiled ``extract_feature`` alone, compiled embedding lookup alone, and
  the full compiled stage 1 -- each against eager, ``default`` mode;
* the quantity that actually matters downstream: per-camera 4x4-pooled image
  keys (the cp1_groot_spatial_pool_16 geometry) of eager vs compiled stage 1.

Diagnostic only; nothing here touches production code.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import torch

from exp.robocasa365 import bench_groot_stages as bench
from openpi.cache.groot.staged import GrootStagedRunner


def _pooled_keys(input_embeds: torch.Tensor, image_token_mask: torch.Tensor) -> torch.Tensor:
    """[cams, 16*C] per-camera 4x4 average pool of the 16x16 image-token grid."""
    tokens = input_embeds[0][image_token_mask[0]].float()  # [cams*256, C]
    cams = tokens.shape[0] // 256
    grid = tokens.reshape(cams, 16, 16, -1).permute(0, 3, 1, 2)  # [cams, C, 16, 16]
    pooled = torch.nn.functional.avg_pool2d(grid, 4)  # [cams, C, 4, 4]
    return pooled.permute(0, 2, 3, 1).reshape(cams, -1)


def _key_cos(a: torch.Tensor, b: torch.Tensor) -> list[float]:
    return torch.nn.functional.cosine_similarity(a, b, dim=1).tolist()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--prompt-index", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    checkpoint = pathlib.Path(args.checkpoint)
    torch.cuda.set_device(0)
    policy = bench.load_policy(checkpoint, device="cuda:0")
    runner = GrootStagedRunner(policy.model, compile_vision=False)
    normalized = bench.build_production_input(policy, checkpoint, bench.PROMPTS[args.prompt_index])
    eagle = runner._eagle  # noqa: SLF001
    out: dict = {"prompt_index": args.prompt_index, "torch": torch.__version__}

    with runner.session():
        e1 = runner.run_stage1(normalized)
        e2 = runner.run_stage1(normalized)
        out["eager_vs_eager"] = bench.tensor_stats(e2.input_embeds, e1.input_embeds)
        eagle_tensors, action_inputs = bench.prepare_stage1_inputs(runner, normalized)
        input_ids, attention_mask, pixel_values = eagle_tensors
        vit_eager = eagle.extract_feature(pixel_values)
        emb_eager = eagle.language_model.get_input_embeddings()(input_ids)
        out["dtypes"] = {
            "vit_eager": str(vit_eager.dtype), "emb_eager": str(emb_eager.dtype),
            "input_embeds": str(e1.input_embeds.dtype), "pixel_values": str(pixel_values.dtype),
        }

    # Scale reference: the same eager path with autocast off (bf16 params, so
    # LayerNorm etc. run in bf16 instead of autocast's fp32 promotion).
    with torch.inference_mode():
        vit_noac = eagle.extract_feature(pixel_values)
    out["vit_eager_autocast_off_vs_on"] = bench.tensor_stats(vit_noac.float(), vit_eager.float())

    with runner.session():
        vit_c = torch.compile(eagle.extract_feature, dynamic=False, fullgraph=False)(pixel_values)
        out["vit_compiled_default_vs_eager"] = bench.tensor_stats(vit_c.float(), vit_eager.float())
        emb_fn = eagle.language_model.get_input_embeddings()
        emb_c = torch.compile(emb_fn, dynamic=False)(input_ids)
        out["emb_compiled_default_vs_eager"] = bench.tensor_stats(emb_c.float(), emb_eager.float())

        positions = torch.nonzero(e1.image_token_mask.reshape(-1), as_tuple=False).flatten()
        stage1_full, _ = bench._stage_callables(runner, positions)  # noqa: SLF001
        s1_c = torch.compile(stage1_full, dynamic=False, fullgraph=True)(*eagle_tensors)
        out["stage1_compiled_default_vs_eager"] = bench.tensor_stats(s1_c[0], e1.input_embeds)
        # Where inside stage 1 does the compiled output differ? image tokens vs text tokens.
        mask = e1.image_token_mask[0]
        diff = (s1_c[0][0].float() - e1.input_embeds[0].float()).abs().max(dim=1).values
        out["max_abs_by_token_class"] = {
            "image_tokens": float(diff[mask].max()), "text_tokens": float(diff[~mask].max()),
        }
        # The downstream quantity: pooled per-camera keys.
        k_e = _pooled_keys(e1.input_embeds, e1.image_token_mask)
        k_c = _pooled_keys(s1_c[0], e1.image_token_mask)
        out["pooled_key_cos_eager_vs_compiled"] = _key_cos(k_e, k_c)
        # And the same pooled keys for the eager-autocast-off tower, as a scale reference.
        out["pooled_key_cos_eager_vs_eager"] = _key_cos(k_e, _pooled_keys(e2.input_embeds, e2.image_token_mask))

    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text(json.dumps(out, indent=1, sort_keys=True, default=str))
    for key, value in out.items():
        print(f"[bisect] {key}: {value}", flush=True)


if __name__ == "__main__":
    main()
