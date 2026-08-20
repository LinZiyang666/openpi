"""Sample the GPU's actual work state while ACT runs (weilandserver)."""
from __future__ import annotations
import subprocess, sys, threading, time, statistics
import numpy as np, torch

REPO = "/home/weiland/openpi"; sys.path.insert(0, REPO)
from exp.ablation_study.sidecar_server import make_act_policy  # noqa: E402

fn = make_act_policy("/home/weiland/bench_latency/act_manifest_task0.json", "cuda")
PROMPT = "pick up the black bowl between the plate and the ramekin and place it on the plate"
rng = np.random.default_rng(20260819)
obs = lambda: {"observation/image": rng.integers(0,256,(224,224,3),dtype=np.uint8),
               "observation/wrist_image": rng.integers(0,256,(224,224,3),dtype=np.uint8),
               "observation/state": rng.standard_normal(8).astype(np.float32),
               "prompt": PROMPT}
for _ in range(20): fn(obs())

samples = []
stop = threading.Event()
def sampler():
    while not stop.is_set():
        out = subprocess.run(["nvidia-smi","--query-gpu=utilization.gpu,clocks.sm,power.draw",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip()
        if out: samples.append([float(x) for x in out.split(",")])
        time.sleep(0.05)
t = threading.Thread(target=sampler, daemon=True); t.start()
lat = []
t_end = time.time() + 8.0
while time.time() < t_end:
    torch.cuda.synchronize(); t0 = time.perf_counter(); fn(obs()); torch.cuda.synchronize()
    lat.append((time.perf_counter()-t0)*1000)
stop.set(); t.join(timeout=2)
u = [s[0] for s in samples]; c = [s[1] for s in samples]; p = [s[2] for s in samples]
print(f"ACT back-to-back: n={len(lat)} median {statistics.median(lat):.2f} ms")
print(f"GPU util median {statistics.median(u):.0f}% | sm clock median {statistics.median(c):.0f} MHz | power median {statistics.median(p):.0f} W")
