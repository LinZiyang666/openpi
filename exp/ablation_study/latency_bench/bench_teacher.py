"""pi0.5 staged latency, eager vs torch.compile (weilandserver, openpi venv).

The production staged path times each stage separately with a cuda-sync
(policy.py:101-131), so compiling the three stage callables individually keeps
that breakdown intact: Stage1 = vision + token prep (the part every cache-HIT
step must pay), Stage2 = LLM prefix KV, Stage3 = action expert denoising.
"""
from __future__ import annotations

import argparse, json, pathlib, statistics, subprocess, sys, threading, time
import numpy as np, torch

sys.path.insert(0, "/home/weiland/openpi")
from openpi.policies import policy_config as pc          # noqa: E402
from openpi.training import config as train_config       # noqa: E402

CKPT = "/home/weiland/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch"
PROMPT = "pick up the black bowl between the plate and the ramekin and place it on the plate"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", required=True,
                    choices=["none", "default", "reduce-overhead",
                             "max-autotune", "max-autotune-no-cudagraphs"])
    ap.add_argument("--mark-step", action="store_true")
    ap.add_argument("--fused", action="store_true",
                    help="compile the whole inference as ONE graph (model.sample_actions, "
                         "which is stage1->stage2->stage3 verbatim) instead of three "
                         "separately-compiled stages; per-stage timing is then unavailable")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--device", default="cuda:0", help="cuda:0 or cpu")
    ap.add_argument("--no-builtin-compile", action="store_true",
                    help="clear config.pytorch_compile_mode (default 'max-autotune') so the "
                         "model does NOT compile sample_actions itself (pi0_pytorch.py:234). "
                         "Required for a clean fused measurement: compiling an already-compiled "
                         "callable stacks two compilations and max-autotune alone takes hours.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = train_config.get_config("pi05_libero")
    if args.no_builtin_compile:
        import dataclasses
        cfg = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, pytorch_compile_mode=None))
        print(f"[builtin] pytorch_compile_mode -> {cfg.model.pytorch_compile_mode}", flush=True)
    policy = pc.create_trained_policy(cfg, CKPT, pytorch_device=args.device)
    _sa = policy._model.sample_actions  # noqa: SLF001 - verify what we are measuring
    print(f"[builtin] sample_actions compiled = "
          f"{hasattr(_sa, '_torchdynamo_orig_callable') or 'OptimizedModule' in type(_sa).__name__}",
          flush=True)
    model = policy._model  # noqa: SLF001 - benchmark probe

    what = "eager (no compile)"
    if args.fused:
        # One graph for the whole forward: policy.infer's non-staged branch
        # calls self._sample_actions, i.e. model.sample_actions, whose body is
        # stage1 -> stage2 -> stage3 verbatim (pi0_pytorch.py:764-773).
        policy._staged_inference = False  # noqa: SLF001 - benchmark probe
        if args.mode != "none":
            kw = {} if args.mode == "default" else {"mode": args.mode}
            policy._sample_actions = torch.compile(model.sample_actions, **kw)  # noqa: SLF001
            what = f"torch.compile(model.sample_actions [FUSED], mode={args.mode})"
        else:
            what = "eager, fused path (no per-stage sync)"
    elif args.mode != "none":
        kw = {} if args.mode == "default" else {"mode": args.mode}
        # Compile each stage callable separately so policy.infer's per-stage
        # timing still measures the same three boundaries.
        model._stage1_token_prep = torch.compile(model._stage1_token_prep, **kw)
        model._stage2_llm_backbone = torch.compile(model._stage2_llm_backbone, **kw)
        model._stage3_action_expert = torch.compile(model._stage3_action_expert, **kw)
        what = f"torch.compile(stage1|stage2|stage3, mode={args.mode})"
    print(f"[compile] {what}")

    rng = np.random.RandomState(0)
    def obs():
        return {"observation/image": rng.randint(0, 255, (224, 224, 3), dtype=np.uint8),
                "observation/wrist_image": rng.randint(0, 255, (224, 224, 3), dtype=np.uint8),
                "observation/state": rng.randn(8).astype(np.float32),
                "prompt": PROMPT}

    def call():
        if args.mark_step:
            torch.compiler.cudagraph_mark_step_begin()
        # policy.py wraps each STAGE in torch.no_grad() but leaves the
        # non-staged branch ungraded; wrap here so fused and staged runs share
        # the same autograd context and stay comparable.
        with torch.no_grad():
            return policy.infer(obs())

    cuda = args.device.startswith("cuda")
    sync = (lambda: torch.cuda.synchronize()) if cuda else (lambda: None)
    t_w = time.perf_counter()
    for _ in range(args.warmup):
        call()
    sync()
    print(f"[warmup] {args.warmup} calls in {time.perf_counter()-t_w:.1f}s (includes compilation)")

    samples, stop = [], threading.Event()
    def sampler():
        while not stop.is_set() and cuda:
            out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,clocks.sm,power.draw",
                                  "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip()
            if out:
                samples.append([float(x) for x in out.split(",")])
            time.sleep(0.05)
    threading.Thread(target=sampler, daemon=True).start()

    wall, tot, s1, s2, s3 = [], [], [], [], []
    for _ in range(args.iters):
        sync(); t0 = time.perf_counter()
        out = call()
        sync(); wall.append((time.perf_counter() - t0) * 1000)
        st = out.get("stage_timing")
        if st is None:  # fused path: no per-stage boundaries to time
            tot.append(out["policy_timing"]["infer_ms"])
            s1.append(float("nan")); s2.append(float("nan")); s3.append(float("nan"))
        else:
            tot.append(st["total_ms"]); s1.append(st["token_prep_ms"])
            s2.append(st["llm_backbone_ms"]); s3.append(st["action_expert_ms"])
    stop.set(); time.sleep(0.15)

    m = statistics.median
    rec = {"model": "pi05_libero", "compile_mode": args.mode, "mark_step": args.mark_step,
           "fused": args.fused,
           "no_grad_wrapped": True,
           "builtin_compile_disabled": args.no_builtin_compile,
           "compiled": what, "n": args.iters,
           "wall_ms": m(wall), "model_total_ms": m(tot),
           "stage1_token_prep_ms": m(s1), "stage2_llm_backbone_ms": m(s2),
           "stage3_action_expert_ms": m(s3),
           "stage2plus3_ms": m(s2) + m(s3),
           "device": args.device,
           "gpu_util_median_pct": m([x[0] for x in samples]) if samples else None,
           "power_median_w": m([x[2] for x in samples]) if samples else None,
           "torch": torch.__version__}
    pathlib.Path(args.out).write_text(json.dumps(rec, indent=2))
    print(f"\n== pi0.5 @ 4090 | compile={args.mode} | n={args.iters} ==")
    print(f"  wall (incl. transforms) : {rec['wall_ms']:8.2f} ms")
    print(f"  model total             : {rec['model_total_ms']:8.2f} ms")
    print(f"    Stage1 vision+tokens  : {rec['stage1_token_prep_ms']:8.2f} ms   <-- paid by every cache-HIT step")
    print(f"    Stage2 LLM prefix     : {rec['stage2_llm_backbone_ms']:8.2f} ms")
    print(f"    Stage3 denoise        : {rec['stage3_action_expert_ms']:8.2f} ms")
    print(f"    Stage2+3 (skippable)  : {rec['stage2plus3_ms']:8.2f} ms")
    print(f"  GPU util {rec['gpu_util_median_pct']}% | power {rec['power_median_w']} W")


if __name__ == "__main__":
    main()
