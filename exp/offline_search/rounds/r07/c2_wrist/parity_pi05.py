"""One guarded GPU process: real keys, completed policy inputs and K10 outputs.

No server, simulator, worker or GPU benchmarking. Admission happens immediately
before model loading; failure stops the process without any GPU work/retries.
"""
import argparse
import dataclasses
import json
import subprocess
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/weiland/trace_runs/offline_search_store")
    ap.add_argument("--out", type=Path, default=Path("/tmp/r7_C2/gpu_parity.json"))
    a = ap.parse_args()
    admission = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free,memory.total", "--format=csv,noheader,nounits"],
                               text=True, capture_output=True)
    if admission.returncode:
        raise SystemExit("NOT_ADMITTED: nvidia-smi failed: " + admission.stderr.strip())
    devices = [list(map(int, line.split(","))) for line in admission.stdout.strip().splitlines()]
    allowed = [(i, free, total) for i, free, total in devices if free >= 12288]
    if not allowed:
        raise SystemExit("NOT_ADMITTED: no GPU has >= 12288 MiB free: " + admission.stdout.strip())
    gpu, free, total = allowed[0]
    import torch
    import numpy as np
    import jax
    from openpi.training import config
    from openpi.policies import policy_config
    from openpi.models_pytorch.pi0_pytorch import Stage1Output
    from openpi.models.model import Observation
    from openpi.serving import stage_io
    from openpi.serving.batching_coordinator import Pi05StageBatcher
    from openpi.cache.components.key_builder import CP1SpatialPool16KeyBuilder
    from openpi.cache.types import CheckpointID
    from exp.offline_search.harness.store import QueryCell
    from exp.offline_search.closed_loop.stage_overrides import CameraRequest, install_pi05

    torch.set_num_threads(1)
    device = f"cuda:{gpu}"
    torch.cuda.set_per_process_memory_fraction(min(9.5*1024/total, 1.), device=device)
    cfg = config.get_config("pi05_libero")
    cfg = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, pytorch_compile_mode=None))
    checkpoint = "/home/weiland/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch"
    policy = policy_config.create_trained_policy(cfg, checkpoint, pytorch_device=device)
    model = policy._model.eval()
    stock = model.run_stage1
    override = install_pi05("per_request")
    batcher = Pi05StageBatcher(model, torch.device(device))
    checks = []

    def key(out):
        kb = CP1SpatialPool16KeyBuilder(enabled_fields=["vision_0", "vision_1", "robot_state"])
        kb.collect(CheckpointID.CP1, stage1=out)
        return kb.build(CheckpointID.CP1)

    with torch.inference_mode():
        for suite in ("l10", "spatial"):
            qc = QueryCell(a.root, f"pi05_{suite}_cache")
            episodes = [ep for ep in qc.episodes if qc.tok_index[ep["start"]] >= 0]
            for ep in (episodes[0], episodes[len(episodes)//2]):
                for row in (ep["start"], ep["start"]+1, ep["end"]-1):
                    ti = int(qc.tok_index[row])
                    if ti < 0:
                        raise ValueError("chosen parity observation lacks stored images")
                    raw = {"observation/image": np.asarray(qc.tok("img0")[ti]),
                           "observation/wrist_image": np.asarray(qc.tok("img1")[ti]),
                           "observation/state": np.asarray(qc.raw_state[row], np.float64), "prompt": ep["task"]}
                    inp = policy._input_transform(raw)
                    obs = Observation.from_dict(jax.tree.map(lambda x: torch.as_tensor(np.array(x), device=device)[None], inp))
                    full = stock(obs)
                    live_full = batcher.run_stage1_batch([CameraRequest(inp, "full")])[0]
                    audit = {}
                    wrist = batcher.run_stage1_batch([CameraRequest(inp, "wrist_only", audit)])[0]
                    # Exercise mixed/reordered outputs and .to() without a second
                    # encoder pass. MISS completion restores every input field.
                    mixed = stage_io.stack_stage1_output([wrist, live_full])
                    shards = stage_io.split_stage1_output(mixed.to(device), 2)
                    complete = override.complete(model, shards[0])
                    inputs_equal = all(torch.equal(getattr(full, f.name), getattr(complete, f.name)) for f in dataclasses.fields(Stage1Output))
                    full_equal = all(torch.equal(getattr(full, f.name), getattr(live_full, f.name)) for f in dataclasses.fields(Stage1Output))
                    noise = torch.randn((1, model.config.action_horizon, model.config.action_dim), device=device,
                                        generator=torch.Generator(device=device).manual_seed(17+row))
                    ref = model.run_stage3(model.run_stage2(full), noise=noise, num_steps=10).action_chunk
                    got = model.run_stage3(model.run_stage2(wrist), noise=noise, num_steps=10).action_chunk
                    record = dict(suite=suite, row=int(row), wrist_key_equal=torch.equal(key(full)["vision_1"], key(wrist)["vision_1"]),
                                  full_path_equal=full_equal, completed_policy_inputs_equal=inputs_equal,
                                  policy_K10_equal=torch.equal(ref, got), policy_max_abs=float((ref-got).abs().max()),
                                  audit=audit.copy())
                    checks.append(record)
                    print(json.dumps(record), flush=True)
                    del ref, got, full, wrist, mixed, shards, complete
    report = dict(admission=admission.stdout.strip(), checkpoint=checkpoint, checks=checks,
                  PASS=all(r[k] for r in checks for k in ("wrist_key_equal", "full_path_equal", "completed_policy_inputs_equal", "policy_K10_equal")),
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(device))
    a.out.write_text(json.dumps(report, indent=2) + "\n")
    if not report["PASS"]:
        raise SystemExit("PARITY FAILED: do not profile SW")


if __name__ == "__main__":
    main()
