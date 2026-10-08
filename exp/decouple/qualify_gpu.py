"""Coordinator-only numerical qualification; one real model copy, no listeners.

Replay explicit per-row noise through B=1 baseline stages and the split stage
service at selected batch width. Report raw-bit equality and absolute/relative
error for pooled keys and actions. B=1 requires exact equality; wider batches
require an explicit numerical bound, and still require identical retrieval
donor ordering on the frozen B library. This is a stage gate, not a closed-loop
decision replay or a substitute for GPU testing of the complete launch path.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import socket
import threading


class Harness:
    def __init__(self, service):
        from openpi.serving.split_service import StageClient
        from openpi.serving.split_transport import Channel
        self.channels, self.threads = [], []
        def factory():
            left, right = socket.socketpair()
            thread = threading.Thread(target=service.serve_channel, args=(right,))
            thread.start()
            channel = Channel(left)
            self.threads.append(thread)
            self.channels.append(channel)
            return channel
        self.client = StageClient(None, channel_factory=factory)

    def close(self):
        for channel in self.channels:
            channel.close()
        for thread in self.threads:
            thread.join(10)
            if thread.is_alive():
                raise RuntimeError("stage service failed to stop")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("pi05", "groot"), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="pi05_libero")
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--rows", type=int, default=16)
    parser.add_argument("--max-abs", type=float, help="explicit permitted wider-batch tensor error")
    parser.add_argument("--max-rel", type=float, help="explicit permitted wider-batch tensor error")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if not os.environ.get("CUDA_VISIBLE_DEVICES", ""):
        parser.error("coordinator-only: select the single 4090 with CUDA_VISIBLE_DEVICES")
    if min(args.k, args.rows, args.batch_size) < 1:
        parser.error("k, rows and batch-size must be positive")
    if args.batch_size > 1 and (args.max_abs is None or args.max_rel is None):
        parser.error("wider batches require explicit --max-abs and --max-rel bounds")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    os.environ.update(HF_MODULES_CACHE=str(Path(args.out).resolve().parent / "hf_modules"),
                      NO_ALBUMENTATIONS_UPDATE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    import numpy as np
    import torch
    from exp.decouple.recorded import load, observation
    from openpi.cache.types import CheckpointID
    from openpi.serving.split_service import StageService, RemoteCoordinator, RemoteGrootRunner, RemoteKeyBuilder, make_groot_coordinator
    data = load(args.model, limit=args.rows)
    if "4090" not in torch.cuda.get_device_name(0):
        raise ValueError("qualification requires the requested single RTX 4090")
    if args.model == "pi05":
        import jax
        from openpi.serving.batching_coordinator import BatchingCoordinator, Stage3MissPayload  # pi05 only: stage_io needs jax, absent in the GR00T venv
        from openpi.training import config
        import dataclasses
        from openpi.policies.policy_config import create_trained_policy
        from openpi.cache.components.key_builder import CP1SpatialPool16KeyBuilder
        train = config.get_config(args.config)
        train = dataclasses.replace(train, model=dataclasses.replace(train.model, pytorch_compile_mode=None))
        policy = create_trained_policy(train, args.checkpoint, pytorch_device="cuda")
        model, runner = policy._model, None
        def inputs(index):
            transformed = policy._input_transform(observation(data, index))
            return jax.tree.map(lambda value: torch.from_numpy(np.array(value)), transformed)
        def coordinator(width):
            core = BatchingCoordinator(model, max_batch_size=width, max_wait_ms=2 if width > 1 else 0)
            core.start()
            return core
        def builder():
            return CP1SpatialPool16KeyBuilder(enabled_fields=["vision_0", "vision_1", "robot_state"])
        shape = (1, model.config.action_horizon, model.config.action_dim)
    else:
        from gr00t.model.policy import Gr00tPolicy
        from custom_data_config import LiberoDataConfig
        from exp.libero_groot.policy_adapter import build_groot_observation
        from openpi.cache.groot.staged import GrootStagedRunner
        from openpi.cache.groot.key_builder import GrootLiberoCP1SpatialPool16KeyBuilder
        import cv2
        cfg = LiberoDataConfig()
        policy = Gr00tPolicy(args.checkpoint, "new_embodiment", cfg.modality_config(), cfg.transform(),
                             denoising_steps=args.k, device="cuda")
        model = policy.model
        runner = GrootStagedRunner(model)
        def inputs(index):
            obs = observation(data, index)
            for name in ("observation/image", "observation/wrist_image"):
                obs[name] = cv2.resize(obs[name], (256, 256))
            batched = {k: np.asarray(v)[None] for k, v in build_groot_observation(obs).items()}
            return policy.apply_transforms(batched)
        def coordinator(width):
            return make_groot_coordinator(runner, max_batch_size=width, max_wait_ms=2 if width > 1 else 0)
        def builder():
            return GrootLiberoCP1SpatialPool16KeyBuilder(enabled_fields=["vision_0", "vision_1", "robot_state"])
        shape = (1, model.action_head.config.action_horizon, model.action_head.config.action_dim)
    prepared = [inputs(i) for i in range(len(data.rows))]
    noises = [torch.randn(shape, generator=torch.Generator().manual_seed(700+i)) for i in range(len(prepared))]
    def evaluate(index, core=None, client=None):
        key = builder()
        if client is not None:
            key = RemoteKeyBuilder(key, client)
        try:
            if args.model == "pi05":
                stage = RemoteCoordinator(client) if client else core
                s1 = stage.submit_to_stage(1, "qualify", prepared[index])
                key.collect(CheckpointID.CP1, stage1=s1)
                keys = key.build(CheckpointID.CP1)
                s2 = stage.submit_to_stage(2, "qualify", s1)
                noise = noises[index] if client else noises[index].cuda()
                action = stage.submit_to_stage(3, "qualify", Stage3MissPayload(s2, noise[0], args.k)).action_chunk
            else:
                stage = RemoteGrootRunner(client, model) if client else runner
                with stage.session():
                    s1 = stage.run_stage1(prepared[index])
                    key.collect(CheckpointID.CP1, stage1=s1)
                    keys = key.build(CheckpointID.CP1)
                    s2 = stage.run_stage2_llm(s1)
                    noise = noises[index] if client else noises[index].cuda()
                    action = stage.run_stage3(s2, noise=noise).action_pred
            return {**{f"key:{k}": v.detach().cpu() for k, v in keys.items()}, "action": action.detach().cpu()}
        finally:
            key.clear()
            if client:
                key.close()
    baseline_core = coordinator(1) if args.model == "pi05" else None
    try:
        baseline = [evaluate(i, core=baseline_core) for i in range(len(prepared))]
    finally:
        if baseline_core:
            baseline_core.stop()
    split_core = coordinator(args.batch_size)
    harness = Harness(StageService(model, runner=runner, coordinator=split_core))
    def split(index):
        with harness.client.decision():
            return evaluate(index, client=harness.client)
    try:
        with ThreadPoolExecutor(max_workers=args.batch_size) as pool:
            candidate = list(pool.map(split, range(len(prepared))))
    finally:
        harness.close()
        split_core.stop()
    fields, passed = {}, True
    for name in baseline[0]:
        pairs = [(old[name], new[name]) for old, new in zip(baseline, candidate)]
        exact = all(torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)) for a, b in pairs)
        absolute = max(float((a.float()-b.float()).abs().max()) for a, b in pairs)
        relative = max(float(((a.float()-b.float()).abs()/a.float().abs().clamp_min(1e-6)).max()) for a, b in pairs)
        fields[name] = dict(bit_exact=exact, max_abs=absolute, max_rel=relative)
        passed &= exact if args.batch_size == 1 else absolute <= args.max_abs and relative <= args.max_rel
    # Key perturbations must also preserve the nearest donor ranking. Uses the
    # recorded B key arrays, not held-out A outcomes; full method replay below
    # remains a separate mandatory closed-loop gate.
    donor_order_equal = True
    library = {"vision_0": data.arrays["key_v0"], "vision_1": data.arrays["key_v1"], "robot_state": data.arrays["rs"]}
    for old, new in zip(baseline, candidate):
        for name, values in library.items():
            a, b = old[f"key:{name}"].float().numpy().reshape(-1), new[f"key:{name}"].float().numpy().reshape(-1)
            if values.shape[1] != len(a):
                raise ValueError(f"B library key dimension mismatch for {name}")
            # Stable top-8 ordering per key field. This is an additional stage
            # diagnostic; it does not claim the method's composite metric.
            scores_a = np.sum((values - a)**2, axis=1)
            scores_b = np.sum((values - b)**2, axis=1)
            donor_order_equal &= np.array_equal(np.argsort(scores_a, kind="stable")[:8], np.argsort(scores_b, kind="stable")[:8])
    passed &= donor_order_equal
    record = dict(model=args.model, checkpoint=args.checkpoint, k=args.k, batch_size=args.batch_size,
                  rows=len(prepared), bounds=dict(max_abs=args.max_abs, max_rel=args.max_rel), fields=fields,
                  donor_order_equal=bool(donor_order_equal), passed=bool(passed), torch=torch.__version__,
                  gpu=torch.cuda.get_device_name(0), image_source=str(data.image_source),
                  limitations="stage qualification only; GR00T uses adapted Pi0.5 B image fixtures; complete launcher/decision replay still required")
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2)+"\n")
    print(json.dumps(record, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
