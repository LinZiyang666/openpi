"""Count Dynamo graph breaks in each stage boundary, under flash-attn and SDPA.

Three W2 smoke runs died at three different fullgraph boundaries. Rather than
patch them one by one, ask Dynamo directly: ``torch._dynamo.explain`` lists
every graph break and its reason for the vision tower, the LLM call and one
denoise step -- first with the production attention implementation, then with
the model's attention switched to SDPA on a copy of the config. Diagnostic only.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import torch

from exp.robocasa365 import bench_groot_stages as bench
from openpi.cache.groot.staged import GrootStagedRunner, denoise_step


def _explain(name: str, fn, *args, out: dict) -> None:
    import torch._dynamo

    torch._dynamo.reset()
    try:
        exp = torch._dynamo.explain(fn)(*args)
        reasons = []
        for br in exp.break_reasons:
            frame = br.user_stack[-1] if br.user_stack else None
            reasons.append(
                {
                    "reason": str(br.reason)[:200],
                    "at": f"{frame.filename.split('/')[-1]}:{frame.lineno}"
                    if frame
                    else "?",
                }
            )
        rec = {
            "graph_count": exp.graph_count,
            "graph_break_count": exp.graph_break_count,
            "reasons": reasons[:12],
        }
    except Exception as exc:  # noqa: BLE001
        rec = {"error": f"{type(exc).__name__}: {str(exc)[:300]}"}
    out[name] = rec
    print(f"[explain] {name}: {json.dumps(rec)[:600]}", flush=True)


def _set_attn(module: torch.nn.Module, impl: str) -> list[str]:
    seen = set()
    for m in module.modules():
        cfg = getattr(m, "config", None)
        if cfg is not None and hasattr(cfg, "_attn_implementation"):
            seen.add(str(cfg._attn_implementation))  # noqa: SLF001
            cfg._attn_implementation = impl  # noqa: SLF001
    return sorted(seen)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    checkpoint = pathlib.Path(args.checkpoint)
    torch.cuda.set_device(0)
    policy = bench.load_policy(checkpoint, device="cuda:0")
    runner = GrootStagedRunner(policy.model, compile_vision=False)
    normalized = bench.build_production_input(policy, checkpoint, bench.PROMPTS[0])
    eagle = runner._eagle  # noqa: SLF001
    head = policy.model.action_head
    out: dict = {"torch": torch.__version__}
    out["attn_impl_vision"] = str(
        getattr(eagle.vision_model.config, "_attn_implementation", "?")
    )
    out["attn_impl_llm"] = str(
        getattr(eagle.language_model.config, "_attn_implementation", "?")
    )
    try:
        import flash_attn

        out["flash_attn_version"] = getattr(flash_attn, "__version__", "?")
    except Exception as exc:  # noqa: BLE001
        out["flash_attn_version"] = f"import failed: {exc!r}"
    print(
        f"[explain] attn vision={out['attn_impl_vision']} llm={out['attn_impl_llm']} flash_attn={out['flash_attn_version']}",
        flush=True,
    )

    with runner.session():
        s1 = runner.run_stage1(normalized)
        (input_ids, attention_mask, pixel_values), action_inputs = (
            bench.prepare_stage1_inputs(runner, normalized)
        )
        positions = torch.nonzero(
            s1.image_token_mask.reshape(-1), as_tuple=False
        ).flatten()
        stage1_full, stage2_llm = bench._stage_callables(runner, positions)  # noqa: SLF001
        s2 = runner.run_stage2_llm(s1)
        backbone_outputs = runner._head_inputs(s2)  # noqa: SLF001
        processed = head.process_backbone_output(backbone_outputs)
        vl = processed.backbone_features
        emb = action_inputs["embodiment_id"]
        state_features = head.state_encoder(action_inputs["state"], emb)
        noise = torch.randn(
            1,
            head.config.action_horizon,
            head.config.action_dim,
            device=vl.device,
            dtype=vl.dtype,
        )
        tsteps = torch.full((1,), 0, device=vl.device)

        for label in ("production", "sdpa"):
            if label == "sdpa":
                out["switched_from"] = {
                    "vision": _set_attn(eagle.vision_model, "sdpa"),
                    "llm": _set_attn(eagle.language_model, "sdpa"),
                    "head": _set_attn(head, "sdpa"),
                }
            _explain(
                f"{label}/stage1_full",
                stage1_full,
                input_ids,
                attention_mask,
                pixel_values,
                out=out,
            )
            _explain(f"{label}/stage2_llm", stage2_llm, s1.input_embeds, out=out)
            _explain(
                f"{label}/denoise_step",
                denoise_step,
                head,
                vl,
                state_features,
                emb,
                noise,
                tsteps,
                0.25,
                out=out,
            )

    pathlib.Path(args.out).write_text(
        json.dumps(out, indent=1, sort_keys=True, default=str)
    )
    print("[explain] wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
