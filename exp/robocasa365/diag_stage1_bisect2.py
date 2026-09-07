"""Second bisect: which inductor behaviour makes the compiled vision tower drift.

Bisect 1 localised the whole stage-1 divergence to ``extract_feature`` under
inductor (cos_min 0.9719, relF 7.9e-2, identical in default and
reduce-overhead; text tokens and the embedding lookup are bit-exact). Two
candidate mechanisms, each with a switch that needs no production change:

* inductor keeps fused intermediates in fp32 where eager rounds to bf16 each
  op -- ``torch._inductor.config.emulate_precision_casts`` makes it round the
  way eager does;
* inductor picks a different SDPA kernel -- forcing the MATH backend on both
  sides removes that degree of freedom.

Every variant reports the raw-token stats and the downstream pooled-key
cosine. Diagnostic only.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import torch

from exp.robocasa365 import bench_groot_stages as bench
from openpi.cache.groot.staged import GrootStagedRunner


def _pooled(tokens: torch.Tensor) -> torch.Tensor:
    """[cams, 16*C] 4x4-pooled image keys from [cams, 256, C] tower output."""
    cams, n, c = tokens.shape
    grid = tokens.float().reshape(cams, 16, 16, c).permute(0, 3, 1, 2)
    return torch.nn.functional.avg_pool2d(grid, 4).permute(0, 2, 3, 1).reshape(cams, -1)


def _report(name: str, got: torch.Tensor, ref: torch.Tensor, out: dict) -> None:
    stats = bench.tensor_stats(got.float(), ref.float())
    stats["pooled_key_cos"] = torch.nn.functional.cosine_similarity(
        _pooled(got), _pooled(ref), dim=1
    ).tolist()
    out[name] = stats
    print(f"[bisect2] {name}: cos_min={stats['cos_min']:.6f} relF={stats['rel_frobenius']:.3e} "
          f"max|d|={stats['max_abs_delta']:.3e} pooled_cos={[round(x, 6) for x in stats['pooled_key_cos']]}",
          flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--prompt-index", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import torch._dynamo
    import torch._inductor.config as icfg

    checkpoint = pathlib.Path(args.checkpoint)
    torch.cuda.set_device(0)
    policy = bench.load_policy(checkpoint, device="cuda:0")
    runner = GrootStagedRunner(policy.model, compile_vision=False)
    normalized = bench.build_production_input(policy, checkpoint, bench.PROMPTS[args.prompt_index])
    eagle = runner._eagle  # noqa: SLF001
    out: dict = {
        "prompt_index": args.prompt_index,
        "torch": torch.__version__,
        "has_emulate_precision_casts": hasattr(icfg, "emulate_precision_casts"),
    }
    print(f"[bisect2] torch={torch.__version__} emulate_flag={out['has_emulate_precision_casts']}", flush=True)

    with runner.session():
        (_, _, pixel_values), _ = bench.prepare_stage1_inputs(runner, normalized)
        ref = eagle.extract_feature(pixel_values)

    def compiled_tower(mode: str | None):
        torch._dynamo.reset()
        kw = {} if mode is None else {"mode": mode}
        return torch.compile(eagle.extract_feature, dynamic=False, **kw)

    # 1. baseline again (default), then with eager-style precision emulation.
    with runner.session():
        _report("default", compiled_tower(None)(pixel_values), ref, out)
    if out["has_emulate_precision_casts"]:
        icfg.emulate_precision_casts = True
        with runner.session():
            _report("default+emulate_precision_casts", compiled_tower(None)(pixel_values), ref, out)
            _report("reduce-overhead+emulate_precision_casts", compiled_tower("reduce-overhead")(pixel_values), ref, out)
        icfg.emulate_precision_casts = False

    # 2. SDPA backend pinned to MATH on both sides.
    from torch.nn.attention import SDPBackend, sdpa_kernel

    with runner.session(), sdpa_kernel(SDPBackend.MATH):
        ref_math = eagle.extract_feature(pixel_values)
        _report("eager_math_sdpa_vs_eager_default_sdpa", ref_math, ref, out)
        _report("default_compiled_math_sdpa_vs_eager_math_sdpa", compiled_tower(None)(pixel_values), ref_math, out)
    if out["has_emulate_precision_casts"]:
        icfg.emulate_precision_casts = True
        with runner.session(), sdpa_kernel(SDPBackend.MATH):
            _report("default+emulate+math_sdpa_vs_eager_math_sdpa", compiled_tower(None)(pixel_values), ref_math, out)
        icfg.emulate_precision_casts = False

    # 3. Which side is closer to an fp32 reference of the tower? (upcast copy)
    try:
        import copy

        tower32 = copy.deepcopy(eagle.vision_model).float()
        mlp32 = copy.deepcopy(eagle.mlp1).float()
        out["fp32_reference_available"] = True
    except Exception as exc:  # noqa: BLE001
        out["fp32_reference_available"] = f"no: {exc!r}"
        tower32 = None
    if tower32 is not None:
        import inspect

        out["extract_feature_source"] = inspect.getsource(type(eagle).extract_feature)
        print("[bisect2] extract_feature source recorded for the fp32 re-implementation check", flush=True)

    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text(json.dumps(out, indent=1, sort_keys=True, default=str))
    print("[bisect2] wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
