"""Diagnose where SmolVLA's per-call latency goes (weilandserver).

Splits one policy call into: obs prep -> VLM prefix forward (KV cache fill) ->
N denoise steps, and records the attention implementation, dtypes and the GPU
clock/power state during the run. Read-only instrumentation: the timed code path
is the production one, wrapped by monkeypatched timers.
"""
from __future__ import annotations

import json
import statistics
import subprocess
import sys
import time

import numpy as np
import torch

REPO = "/home/weiland/openpi"
sys.path.insert(0, REPO)
CKPT = ("/data/openpi/ablation_study/executor_substitution/checkpoints/"
        "libero_spatial/smolvla/checkpoints/020000/pretrained_model")
PROMPT = "pick up the black bowl between the plate and the ramekin and place it on the plate"

from exp.ablation_study.sidecar_server import _obs_to_lerobot_batch  # noqa: E402
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy  # noqa: E402

policy = SmolVLAPolicy.from_pretrained(CKPT)
policy.to("cuda").eval()
model = policy.model

# --- static facts -------------------------------------------------------
vlm = model.vlm_with_expert
print("== static ==")
print("num_steps (denoise):", policy.config.num_steps)
print("resize_imgs_with_padding:", policy.config.resize_imgs_with_padding)
print("num_vlm_layers:", policy.config.num_vlm_layers, "| attention_mode:", policy.config.attention_mode)
print("use_cache:", policy.config.use_cache)
for attr in ("attention_implementation", "_attn_implementation"):
    if hasattr(vlm, attr):
        print(f"vlm.{attr}:", getattr(vlm, attr))
inner = getattr(vlm, "vlm", None)
if inner is not None and hasattr(inner, "config"):
    print("hf config._attn_implementation:", getattr(inner.config, "_attn_implementation", "?"))
print("param dtypes:", {str(p.dtype) for p in policy.parameters()})

# --- instrument ---------------------------------------------------------
prefix_ms, denoise_ms, denoise_calls = [], [], []
orig_denoise = model.denoise_step
orig_vlm_fwd = vlm.forward
_state = {"first_fwd": None, "n_denoise": 0}


def timed_vlm_forward(*a, **kw):
    fill = kw.get("fill_kv_cache", False)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = orig_vlm_fwd(*a, **kw)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) * 1000.0
    if fill:
        _state["first_fwd"] = dt
    return out


def timed_denoise(*a, **kw):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = orig_denoise(*a, **kw)
    torch.cuda.synchronize()
    denoise_ms.append((time.perf_counter() - t0) * 1000.0)
    _state["n_denoise"] += 1
    return out


vlm.forward = timed_vlm_forward
model.denoise_step = timed_denoise

rng = np.random.default_rng(20260819)
def obs():
    return {"observation/image": rng.integers(0, 256, (224, 224, 3), dtype=np.uint8),
            "observation/wrist_image": rng.integers(0, 256, (224, 224, 3), dtype=np.uint8),
            "observation/state": rng.standard_normal(8).astype(np.float32),
            "prompt": PROMPT}

for _ in range(3):  # warmup
    with torch.no_grad():
        policy.predict_action_chunk(_obs_to_lerobot_batch(obs(), "cuda"))

prefix_ms.clear(); denoise_ms.clear(); _state["n_denoise"] = 0
totals = []
clocks = []
for i in range(10):
    b = _obs_to_lerobot_batch(obs(), "cuda")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        policy.predict_action_chunk(b)
    torch.cuda.synchronize()
    totals.append((time.perf_counter() - t0) * 1000.0)
    prefix_ms.append(_state["first_fwd"])
    if i == 5:  # sample GPU state mid-run
        clocks = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.sm,clocks.max.sm,power.draw,power.limit,temperature.gpu,utilization.gpu",
             "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()

med = statistics.median
print("\n== breakdown (n=10 calls) ==")
print(f"total per call      : {med(totals):8.2f} ms")
print(f"  VLM prefix (1x)   : {med(prefix_ms):8.2f} ms")
print(f"  denoise steps     : {_state['n_denoise']/10:.1f} calls/call, median {med(denoise_ms):7.2f} ms each,"
      f" sum {med(denoise_ms)*_state['n_denoise']/10:8.2f} ms")
print(f"GPU state mid-run   : {clocks}")
