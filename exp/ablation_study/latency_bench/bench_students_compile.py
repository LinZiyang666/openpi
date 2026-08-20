"""Eager vs torch.compile latency for the student policies (weilandserver).

Question: are these models launch-bound (host/CPU limited) rather than compute
limited? If so, compiling with mode="reduce-overhead" (CUDA Graph capture,
which replays the whole kernel sequence with one launch) must cut latency
sharply and raise GPU utilisation.

Same timing contract as bench_wls.py: obs dict -> lerobot batch ->
predict_action_chunk -> .cpu().numpy(), i.e. the sidecar's `forward_ms`.
"""
from __future__ import annotations

import argparse, json, pathlib, statistics, subprocess, sys, threading, time
import numpy as np
import torch

REPO = "/home/weiland/openpi"
sys.path.insert(0, REPO)
from exp.ablation_study.sidecar_server import _obs_to_lerobot_batch  # noqa: E402

ACT_CKPT = ("/data/openpi/ablation_study/executor_substitution/checkpoints/"
            "libero_spatial/act_selected/task_0/pretrained_model")
SMOLVLA_CKPT = ("/data/openpi/ablation_study/executor_substitution/checkpoints/"
                "libero_spatial/smolvla/checkpoints/020000/pretrained_model")
PROMPT = "pick up the black bowl between the plate and the ramekin and place it on the plate"


def load(policy_kind: str, device: str):
    if policy_kind == "act":
        from lerobot.policies.act.modeling_act import ACTPolicy
        p = ACTPolicy.from_pretrained(ACT_CKPT)
    else:
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        p = SmolVLAPolicy.from_pretrained(SMOLVLA_CKPT)
    p.config.device = device
    return p.to(device).eval()


def apply_compile(policy, kind: str, mode: str) -> str:
    """Compile the hot module; returns a description of what was compiled."""
    if mode == "none":
        return "eager (no compile)"
    kw = {} if mode == "default" else {"mode": mode}
    if kind == "act":
        policy.model = torch.compile(policy.model, **kw)
        return f"torch.compile(policy.model, mode={mode})"
    # SmolVLA: the hot path is vlm_with_expert.forward — called once for the
    # prefix and once per denoise step.
    policy.model.vlm_with_expert = torch.compile(policy.model.vlm_with_expert, **kw)
    return f"torch.compile(policy.model.vlm_with_expert, mode={mode})"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--policy", required=True, choices=["act", "smolvla"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--mode", required=True, choices=["none", "default", "reduce-overhead"])
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--mark-step", action="store_true",
                    help="call torch.compiler.cudagraph_mark_step_begin() before each "
                         "invocation (required when CUDA Graphs alias a KV cache)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    policy = load(args.policy, args.device)
    what = apply_compile(policy, args.policy, args.mode)
    print(f"[compile] {what}")

    rng = np.random.default_rng(20260819)
    def obs():
        return {"observation/image": rng.integers(0, 256, (224, 224, 3), dtype=np.uint8),
                "observation/wrist_image": rng.integers(0, 256, (224, 224, 3), dtype=np.uint8),
                "observation/state": rng.standard_normal(8).astype(np.float32),
                "prompt": PROMPT}

    def call(o):
        if args.mark_step:
            torch.compiler.cudagraph_mark_step_begin()
        with torch.no_grad():
            return policy.predict_action_chunk(_obs_to_lerobot_batch(o, args.device))[0].cpu().numpy()

    t_w = time.perf_counter()
    for _ in range(args.warmup):
        call(obs())
    torch.cuda.synchronize()
    warm_s = time.perf_counter() - t_w
    print(f"[warmup] {args.warmup} calls in {warm_s:.1f}s (includes compilation)")

    samples, stop = [], threading.Event()
    def sampler():
        while not stop.is_set():
            out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,clocks.sm,power.draw",
                                  "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip()
            if out:
                samples.append([float(x) for x in out.split(",")])
            time.sleep(0.05)
    threading.Thread(target=sampler, daemon=True).start()

    lat = []
    for _ in range(args.iters):
        o = obs()
        torch.cuda.synchronize(); t0 = time.perf_counter()
        out = call(o)
        torch.cuda.synchronize(); lat.append((time.perf_counter() - t0) * 1000)
        assert np.asarray(out).shape == (10, 7)
    stop.set(); time.sleep(0.15)

    s = sorted(lat)
    q = lambda p: s[min(len(s) - 1, int(round(p * (len(s) - 1))))]  # noqa: E731
    m = statistics.median
    rec = {"policy": args.policy, "device": args.device, "compile_mode": args.mode,
           "mark_step": args.mark_step,
           "compiled": what, "warmup_seconds": round(warm_s, 1),
           "median_ms": m(s), "p90_ms": q(0.9), "min_ms": s[0], "max_ms": s[-1],
           "mean_ms": statistics.fmean(s), "n": len(s),
           "gpu_util_median_pct": m([x[0] for x in samples]) if samples else None,
           "sm_clock_median_mhz": m([x[1] for x in samples]) if samples else None,
           "power_median_w": m([x[2] for x in samples]) if samples else None,
           "torch": torch.__version__}
    pathlib.Path(args.out).write_text(json.dumps(rec, indent=2))
    print(f"\n== {args.policy} @ {args.device} | compile={args.mode} | n={rec['n']} ==")
    print(f"  median {rec['median_ms']:8.2f} ms | p90 {rec['p90_ms']:8.2f} | min {rec['min_ms']:8.2f}")
    print(f"  GPU util {rec['gpu_util_median_pct']}% | clock {rec['sm_clock_median_mhz']} MHz | power {rec['power_median_w']} W")


if __name__ == "__main__":
    main()
