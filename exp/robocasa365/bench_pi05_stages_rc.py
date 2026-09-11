"""pi0.5 staged latency on RoboCasa geometry, with the stage-3 step ladder.

The ledger's pi0.5 stage costs were measured on LIBERO: two cameras and an
eight-dimensional state. RoboCasa runs three cameras and a thirty-two
dimensional state, so stage 1 in particular is a different number, and the
inference ratio is a ratio of these three -- addressing an IR grid with the
LIBERO constants would place every arm at a budget nobody asked for.

The step ladder is the second reason this exists. A warm rung re-runs part of
stage 3, and stage 3 is not proportional to its step count: there is a fixed
head that every call pays. Timing only the full ten-step loop and dividing
would under-price a short rung exactly as it did on the GR00T side.

Usage:
  uv run python -m exp.robocasa365.bench_pi05_stages_rc \\
      --checkpoint /home/weiland/ckpt_pi05_robocasa_pytorch \\
      --mode reduce-overhead --out <json>
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import time

import jax
import numpy as np
import torch

from openpi.models import model as _model
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config

#: A RoboCasa instruction of typical length; prompt length moves stage 1 and 2,
#: so the cell records it rather than leaving it implicit.
PROMPT = "pick the mug from the counter and place it in the cabinet"
STEP_LADDER = (1, 3, 5, 10)


def observation(rng: np.random.RandomState, state_dim: int) -> dict:
    """Three cameras and a padded state -- the RoboCasa input shape."""
    img = lambda: rng.randint(0, 255, (224, 224, 3), dtype=np.uint8)  # noqa: E731
    return {
        "observation/image": img(),
        "observation/wrist_image": img(),
        "observation/right_image": img(),
        "observation/state": rng.randn(state_dim).astype(np.float32),
        "prompt": PROMPT,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config-name", default="pi05_robocasa")
    ap.add_argument("--mode", default="reduce-overhead",
                    choices=["none", "default", "reduce-overhead"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--state-dim", type=int, default=32)
    ap.add_argument("--warmup", type=int, default=15)
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = _config.get_config(args.config_name)
    policy = _policy_config.create_trained_policy(cfg, args.checkpoint, pytorch_device=args.device)
    model = policy._model  # noqa: SLF001 - benchmark probe

    if args.mode != "none":
        kw = {} if args.mode == "default" else {"mode": args.mode}
        model._stage1_token_prep = torch.compile(model._stage1_token_prep, **kw)  # noqa: SLF001
        model._stage2_llm_backbone = torch.compile(model._stage2_llm_backbone, **kw)  # noqa: SLF001
        model._stage3_action_expert = torch.compile(model._stage3_action_expert, **kw)  # noqa: SLF001

    rng = np.random.RandomState(0)
    sync = torch.cuda.synchronize if args.device.startswith("cuda") else (lambda: None)

    def call():
        torch.compiler.cudagraph_mark_step_begin()
        with torch.no_grad():
            return policy.infer(observation(rng, args.state_dim))

    for _ in range(args.warmup):
        call()
    sync()

    s1, s2, s3, tot = [], [], [], []
    for _ in range(args.iters):
        sync()
        out = call()
        sync()
        st = out["stage_timing"]
        s1.append(st["token_prep_ms"])
        s2.append(st["llm_backbone_ms"])
        s3.append(st["action_expert_ms"])
        tot.append(st["total_ms"])

    # Step ladder. Each sample runs the same three staged calls the serving
    # path runs -- stage 1, stage 2, then the action expert at k steps -- and
    # times only stage 3. Timing stage 3 alone against one reused stage 2 is
    # not available here: under reduce-overhead the compiled stages return
    # tensors owned by a CUDA graph pool that is recycled at every step
    # boundary, so a stage 2 carried across one is invalidated; and without the
    # boundary the expert falls back to eager and times an execution the server
    # never runs (3.8x slower when measured that way).
    obs_dev = _model.Observation.from_dict(
        jax.tree.map(
            lambda x: torch.from_numpy(np.asarray(x)).to(args.device)[None, ...],
            policy._input_transform(observation(rng, args.state_dim)),  # noqa: SLF001
        )
    )
    ladder = {}
    with torch.no_grad():
        for k in STEP_LADDER:

            def one_sample(k: int = k) -> float:
                torch.compiler.cudagraph_mark_step_begin()
                state, embs, pad, att, pos = model._stage1_token_prep(obs_dev)  # noqa: SLF001
                kv = model._stage2_llm_backbone(embs, pad, att, pos)  # noqa: SLF001
                noise = model.sample_noise(
                    (1, model.config.action_horizon, model.config.action_dim),
                    torch.device(args.device),
                )
                sync()
                t0 = time.perf_counter()
                model._stage3_action_expert(state, pad, kv, noise, k)  # noqa: SLF001
                sync()
                return (time.perf_counter() - t0) * 1000

            for _ in range(5):
                one_sample()
            ladder[k] = statistics.median(one_sample() for _ in range(args.iters))

    ks = np.array(sorted(ladder), dtype=float)
    ys = np.array([ladder[int(k)] for k in ks])
    b, a = np.polyfit(ks, ys, 1)
    pred = a + b * ks
    r2 = 1.0 - float(((ys - pred) ** 2).sum()) / float(((ys - ys.mean()) ** 2).sum())

    med = statistics.median
    # The ladder and the infer path must agree on the one point they share --
    # the full loop. They are measured by different code, so a disagreement
    # means one of them is not running what the server runs, and the tier costs
    # built from the ladder would be fiction.
    ladder_at_full = a + b * 10
    infer_full = med(s3)
    agreement = abs(ladder_at_full - infer_full) / infer_full
    rec = {
        "teacher": "pi05", "schedule_id": "pi05_v1", "num_steps": 10,
        "stage1_ms": med(s1), "stage2_ms": med(s2),
        "stage3_head_ms": float(a), "stage3_step_ms": float(b), "stage3_fit_r2": r2,
        "stage3_full_loop_ms": infer_full, "stage3_measured": {str(int(k)): ladder[int(k)] for k in ks},
        "stage3_ladder_vs_infer_rel": agreement,
        "total_ms": med(tot), "linear_stage3": False, "mode": args.mode,
        "prompt": PROMPT, "state_dim": args.state_dim, "iters": args.iters,
        "gpu_name": torch.cuda.get_device_name(0) if args.device.startswith("cuda") else "cpu",
        "torch": torch.__version__, "config_name": args.config_name, "checkpoint": args.checkpoint,
        "provenance": "RoboCasa-geometry pi0.5 stage split, three cameras, "
                      "stage-3 step ladder fitted as a + b*k",
    }
    if agreement > 0.10:
        raise SystemExit(
            f"ladder s3(10)={ladder_at_full:.3f} ms disagrees with the infer path's "
            f"stage 3 ({infer_full:.3f} ms) by {100 * agreement:.1f}%. One of the two "
            "is not the execution the server runs -- refusing to write a cost record "
            "that would address the IR grid at the wrong budget."
        )
    pathlib.Path(args.out).write_text(json.dumps(rec, indent=1))
    miss = rec["stage1_ms"] + rec["stage2_ms"] + a + b * 10
    print(json.dumps({k: v for k, v in rec.items() if k != "stage3_measured"}, indent=1))
    print("\ntier costs (ms) / IR share of MISS:")
    for name, c in (("FULL_HIT", rec["stage1_ms"]),
                    ("WARM@0.3 (3 steps)", rec["stage1_ms"] + rec["stage2_ms"] + a + 3 * b),
                    ("WARM@0.5 (5 steps)", rec["stage1_ms"] + rec["stage2_ms"] + a + 5 * b),
                    ("MISS", miss)):
        print("  %-20s %7.3f  %6.2f%%" % (name, c, 100 * c / miss))


if __name__ == "__main__":
    main()
