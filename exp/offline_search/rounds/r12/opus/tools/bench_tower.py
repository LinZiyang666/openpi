"""R12 opus: indicative batch-size latency of the pi0.5 vision tower + projector (local 4090, eager bf16).
NOT the serving path; the coordinator should measure stage 1 on the per-request path. Tower-only.
"""
import json, time
import numpy as np
import torch
from exp.offline_search.rounds.r12.opus.tools.lowres import build
from exp.offline_search.rounds.r12.opus.tools import common as cm

torch.set_num_threads(4)
tower, proj = build()
out = {}
with torch.inference_mode():
    for b in (1, 2, 3, 4, 8, 16):
        x = torch.randn(b, 3, 224, 224, device="cuda:0").clamp(-1, 1).to(torch.bfloat16)
        for _ in range(20):
            proj(tower(pixel_values=x).last_hidden_state)
        torch.cuda.synchronize()
        ts = []
        for _ in range(100):
            t0 = time.perf_counter()
            proj(tower(pixel_values=x).last_hidden_state)
            torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1e3)
        out[f"b{b}"] = dict(median_ms=float(np.median(ts)), p10=float(np.percentile(ts, 10)), p90=float(np.percentile(ts, 90)))
        print(b, out[f"b{b}"], flush=True)
    # sequential per-camera calls (the stock pattern) for 2 and 3 cameras
    x1 = torch.randn(1, 3, 224, 224, device="cuda:0").to(torch.bfloat16)
    for ncam in (2, 3):
        ts = []
        for _ in range(100):
            t0 = time.perf_counter()
            for _ in range(ncam):
                proj(tower(pixel_values=x1).last_hidden_state)
            torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1e3)
        out[f"seq{ncam}"] = dict(median_ms=float(np.median(ts)))
        print("sequential", ncam, out[f"seq{ncam}"], flush=True)
out["note"] = "HF SiglipVisionModel+projector, bf16 eager, 4 CPU threads, local RTX 4090 idle; tower-only microbenchmark"
cm.save_json(out, cm.OUT / "bench_tower.json")
