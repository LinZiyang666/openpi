"""Measured latency of the per-request camera paths on π0.5 (R7 completion check for SELECTION rule 3's price).

One GPU process at batch size 1, eager model (same serving mode as the R7 servers). For each of 12 recorded
observations (the C2 parity rows) time, with CUDA synchronisation: stock two-camera stage 1, per-request full stage 1,
per-request wrist-only stage 1, completion of the missing camera after a wrist-only look, and stage 2 + stage 3 (K10)
for scale. Prices are proportional to the owner basis: wrist look = .152 × t_wrist / t_full, completion =
.152 × t_completion / t_full (the same proportional-latency convention as R4, now on the actual R7 path).
"""
import argparse
import dataclasses
import json
import statistics
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/weiland/trace_runs/offline_search_store")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    a = ap.parse_args()
    import torch
    import numpy as np
    import jax
    from openpi.training import config
    from openpi.policies import policy_config
    from openpi.models.model import Observation
    from openpi.serving.batching_coordinator import Pi05StageBatcher
    from exp.offline_search.harness.store import QueryCell
    from exp.offline_search.closed_loop.stage_overrides import CameraRequest, install_pi05

    torch.set_num_threads(1)
    device = "cuda:0"
    cfg = config.get_config("pi05_libero")
    cfg = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, pytorch_compile_mode=None))
    checkpoint = "/home/weiland/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch"
    policy = policy_config.create_trained_policy(cfg, checkpoint, pytorch_device=device)
    model = policy._model.eval()
    stock = model.run_stage1
    override = install_pi05("per_request")
    batcher = Pi05StageBatcher(model, torch.device(device))

    def timed(fn):
        for _ in range(a.warmup):
            fn()
        torch.cuda.synchronize()
        out = []
        for _ in range(a.reps):
            t = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            out.append((time.perf_counter() - t) * 1000)
        return statistics.median(out), sorted(out)[int(.9 * len(out)) - 1]

    rows = []
    with torch.inference_mode():
        for suite in ("l10", "spatial"):
            qc = QueryCell(a.root, f"pi05_{suite}_cache")
            episodes = [ep for ep in qc.episodes if qc.tok_index[ep["start"]] >= 0]
            for ep in (episodes[0], episodes[len(episodes) // 2]):
                for row in (ep["start"], ep["start"] + 1, ep["end"] - 1):
                    ti = int(qc.tok_index[row])
                    raw = {"observation/image": np.asarray(qc.tok("img0")[ti]),
                           "observation/wrist_image": np.asarray(qc.tok("img1")[ti]),
                           "observation/state": np.asarray(qc.raw_state[row], np.float64), "prompt": ep["task"]}
                    inp = policy._input_transform(raw)
                    obs = Observation.from_dict(jax.tree.map(lambda x: torch.as_tensor(np.array(x), device=device)[None], inp))
                    t_stock = timed(lambda: stock(obs))
                    t_full = timed(lambda: batcher.run_stage1_batch([CameraRequest(inp, "full")]))
                    t_wrist = timed(lambda: batcher.run_stage1_batch([CameraRequest(inp, "wrist_only", {})]))
                    wrist_out = batcher.run_stage1_batch([CameraRequest(inp, "wrist_only", {})])[0]
                    t_comp = timed(lambda: override.complete(model, wrist_out))
                    full_out = stock(obs)
                    t_s23 = timed(lambda: model.run_stage3(model.run_stage2(full_out), num_steps=10))
                    rec = dict(suite=suite, row=int(row), stock_full_ms=t_stock, per_request_full_ms=t_full,
                               wrist_only_ms=t_wrist, completion_ms=t_comp, stage23_k10_ms=t_s23)
                    rows.append(rec)
                    print(json.dumps(rec), flush=True)
    med = lambda k: statistics.median(r[k][0] for r in rows)
    full, wrist, comp, s23 = med("per_request_full_ms"), med("wrist_only_ms"), med("completion_ms"), med("stage23_k10_ms")
    report = dict(rows=rows, median_ms=dict(stock_full=med("stock_full_ms"), per_request_full=full, wrist_only=wrist,
                                            completion=comp, stage23_k10=s23),
                  eager_vision_share=full / (full + s23),
                  measured_prices_owner_basis=dict(wrist_look=.152 * wrist / full, completion=.152 * comp / full),
                  assumed_prices=dict(wrist_look=.055198, completion=.049890),
                  note="eager batch-1 latency on the local RTX 4090; prices scale the owner .152 full-look share by the "
                       "measured wrist/full and completion/full ratios (proportional-latency convention)")
    a.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("median_ms", "eager_vision_share", "measured_prices_owner_basis")}, indent=1))


if __name__ == "__main__":
    main()
