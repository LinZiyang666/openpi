"""Third bisect: is the compiled vision tower wrong, or just rounded differently?

Bisect 2 ruled out SDPA backend choice and eager-style precision emulation.
What is left is a discriminator that needs a truth reference:

* run the tower in **fp32** (deep-copied, no autocast): the closest thing to
  ground truth for the same weights;
* compare eager-bf16 and compiled-bf16 against it -- if both sit at similar
  distance from fp32, the compiled path is merely a different bf16 rounding
  order; if compiled is far and eager is close, inductor changed the math;
* compile the **fp32** tower too: if compiled-fp32 matches eager-fp32 to
  ~1e-6, inductor's codegen is sound for this graph and the bf16 gap is
  accumulation-order sensitivity; if compiled-fp32 also diverges, it is a
  codegen defect;
* localise the worst tokens on the 16x16 grid.

Diagnostic only.
"""

from __future__ import annotations

import argparse
import copy
import json
import pathlib

import torch

from exp.robocasa365 import bench_groot_stages as bench
from openpi.cache.groot.staged import GrootStagedRunner


def _pooled(tokens: torch.Tensor) -> torch.Tensor:
    cams, n, c = tokens.shape
    grid = tokens.float().reshape(cams, 16, 16, c).permute(0, 3, 1, 2)
    return torch.nn.functional.avg_pool2d(grid, 4).permute(0, 2, 3, 1).reshape(cams, -1)


def _report(name: str, got: torch.Tensor, ref: torch.Tensor, out: dict) -> dict:
    stats = bench.tensor_stats(got.float(), ref.float())
    stats["pooled_key_cos"] = torch.nn.functional.cosine_similarity(_pooled(got), _pooled(ref), dim=1).tolist()
    out[name] = stats
    print(f"[bisect3] {name}: cos_min={stats['cos_min']:.6f} relF={stats['rel_frobenius']:.3e} "
          f"max|d|={stats['max_abs_delta']:.3e} pooled_cos={[round(x, 6) for x in stats['pooled_key_cos']]}", flush=True)
    return stats


def _tower_copy(eagle, *, dtype: torch.dtype, attn: str):
    """Twin of extract_feature (select_layer == -1, no pixel shuffle) with a
    chosen dtype and attention implementation.

    The production tower runs flash-attention, which only supports fp16/bf16,
    so an fp32 reference has to switch to SDPA. Older transformers pick the
    attention class at construction, newer ones dispatch on the config at
    forward time; try the cheap config flip first and rebuild from config only
    if the flip is ignored.
    """
    assert eagle.select_layer == -1 and not eagle.use_pixel_shuffle, (eagle.select_layer, eagle.use_pixel_shuffle)
    vision = copy.deepcopy(eagle.vision_model)
    for module in vision.modules():
        cfg = getattr(module, "config", None)
        if cfg is not None and hasattr(cfg, "_attn_implementation"):
            cfg._attn_implementation = attn  # noqa: SLF001
    vision = vision.to(dtype).eval()
    mlp = copy.deepcopy(eagle.mlp1).to(dtype).eval()

    def run(pixel_values):
        h = vision(pixel_values=pixel_values.to(dtype), output_hidden_states=False, return_dict=True)
        h = h.last_hidden_state if hasattr(h, "last_hidden_state") else h
        return mlp(h)

    return run, vision


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--prompt-index", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import torch._dynamo

    checkpoint = pathlib.Path(args.checkpoint)
    torch.cuda.set_device(0)
    policy = bench.load_policy(checkpoint, device="cuda:0")
    runner = GrootStagedRunner(policy.model, compile_vision=False)
    normalized = bench.build_production_input(policy, checkpoint, bench.PROMPTS[args.prompt_index])
    eagle = runner._eagle  # noqa: SLF001
    out: dict = {"prompt_index": args.prompt_index, "torch": torch.__version__}

    with runner.session():
        (_, _, pixel_values), _ = bench.prepare_stage1_inputs(runner, normalized)
        eager16 = eagle.extract_feature(pixel_values)
        torch._dynamo.reset()
        comp16 = torch.compile(eagle.extract_feature, dynamic=False)(pixel_values).clone()

    out["production_attn_implementation"] = str(
        getattr(getattr(eagle.vision_model, "config", None), "_attn_implementation", "?")
    )
    print(f"[bisect3] production vision attn = {out['production_attn_implementation']}", flush=True)

    # bf16 + SDPA twins (eager and compiled): isolates the attention-kernel path.
    tower16_sdpa, vision16 = _tower_copy(eagle, dtype=torch.bfloat16, attn="sdpa")
    with torch.inference_mode(), torch.autocast("cuda", torch.bfloat16):
        eager16_sdpa = tower16_sdpa(pixel_values)
        torch._dynamo.reset()
        comp16_sdpa = torch.compile(tower16_sdpa, dynamic=False)(pixel_values).clone()
    out["sdpa_twin_attn_implementation"] = str(vision16.config._attn_implementation)  # noqa: SLF001
    del vision16

    # fp32 + SDPA reference (no autocast), eager and compiled.
    tower32, vision32 = _tower_copy(eagle, dtype=torch.float32, attn="sdpa")
    with torch.inference_mode():
        ref32 = tower32(pixel_values)
        torch._dynamo.reset()
        comp32 = torch.compile(tower32, dynamic=False)(pixel_values).clone()
    del vision32

    _report("compiled_bf16_vs_eager_bf16 (production flash path)", comp16, eager16, out)
    _report("eager_bf16_sdpa_vs_eager_bf16_flash", eager16_sdpa, eager16, out)
    _report("compiled_bf16_sdpa_vs_eager_bf16_sdpa", comp16_sdpa, eager16_sdpa, out)
    _report("eager_bf16_flash_vs_fp32", eager16, ref32, out)
    _report("compiled_bf16_flash_vs_fp32", comp16, ref32, out)
    _report("eager_bf16_sdpa_vs_fp32", eager16_sdpa, ref32, out)
    _report("compiled_bf16_sdpa_vs_fp32", comp16_sdpa, ref32, out)
    _report("compiled_fp32_vs_eager_fp32", comp32, ref32, out)

    # Spatial localisation of the bf16 compiled-vs-eager gap on the token grid.
    d = (comp16.float() - eager16.float()).abs().max(dim=-1).values  # [cams, 256]
    norms = eager16.float().norm(dim=-1)
    cams = d.shape[0]
    worst = torch.topk(d.reshape(-1), 10)
    out["worst_tokens"] = [
        {"cam": int(i // 256), "row": int((i % 256) // 16), "col": int(i % 16),
         "max_abs": float(v), "norm": float(norms.reshape(-1)[i])}
        for v, i in zip(worst.values.tolist(), worst.indices.tolist())
    ]
    row_mean = d.mean(dim=(0, 2)) if False else d.reshape(cams, 16, 16).mean(dim=(0, 2))
    col_mean = d.reshape(cams, 16, 16).mean(dim=(0, 1))
    out["gap_by_row"] = [round(float(x), 4) for x in row_mean]
    out["gap_by_col"] = [round(float(x), 4) for x in col_mean]
    out["gap_vs_norm_corr"] = float(torch.corrcoef(torch.stack([d.reshape(-1), norms.reshape(-1)]))[0, 1])
    print("[bisect3] worst_tokens:", out["worst_tokens"][:5], flush=True)
    print("[bisect3] gap_by_row:", out["gap_by_row"], flush=True)
    print("[bisect3] gap_by_col:", out["gap_by_col"], flush=True)
    print(f"[bisect3] corr(gap, token_norm)={out['gap_vs_norm_corr']:.3f}", flush=True)

    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text(json.dumps(out, indent=1, sort_keys=True, default=str))
    print("[bisect3] wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
