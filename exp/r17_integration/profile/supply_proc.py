"""One formal MT50 resident, measuring native hit/call supply without queues."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time


DESCRIPTORS = {
    "pi05": "pimk1_ful",
    "groot": "grmk1_red",
    "dp": "csdp_1504",
    "fastwam": "csfw_k3",
}
OBSERVATION = Path(__file__).parent / "inputs_mt50/assembly_seed8_idx0.npz"


def sizes_arg(value):
    sizes = [int(v) for v in value.replace(",", " ").split()]
    if not sizes or len(set(sizes)) != len(sizes) or any(not 1 <= b <= 32 for b in sizes):
        raise argparse.ArgumentTypeError("sizes must be distinct integers in 1..32")
    return sizes


def positive_seconds(value):
    seconds = float(value)
    if not math.isfinite(seconds) or seconds <= 0:
        raise argparse.ArgumentTypeError("seconds must be finite and positive")
    return seconds


def load(args):
    """Reuse the formal loader with a temporary, host-adjusted descriptor."""
    from exp.r17_integration.qualification.model_runtime import load as formal_load

    descriptor = args.descriptor or Path("/data/mw_finetune/eval") / DESCRIPTORS[args.model] / "assets/load.json"
    descriptor = descriptor.resolve()
    desc = json.loads(descriptor.read_text())
    if args.checkpoint:
        desc["checkpoint"] = str(args.checkpoint.resolve())
    if args.model == "groot":
        if args.vendor:
            vendor = args.vendor.resolve()
            sys.path.insert(0, str(vendor))
            for key in ("code_files", "tokenizer_files"):
                desc[key] = [str(vendor / p.split("/gr00t_n15/", 1)[1])
                             if "/gr00t_n15/" in p else p for p in desc[key]]
        if args.steps is not None:
            desc["live_k"] = args.steps  # The native action-head schedule is load-time bound.
    # model_runtime.load expects assets/load.json and the adjacent model_spec.json.
    # Neither original file is changed; this directory disappears after loading.
    with tempfile.TemporaryDirectory(prefix="r17-supply-load-") as directory:
        assets = Path(directory)
        (assets / "load.json").write_text(json.dumps(desc))
        (assets / "model_spec.json").symlink_to(descriptor.with_name("model_spec.json"))
        adapter, provider = formal_load(dict(model=args.model, domain="metaworld", assets=str(assets)), args.device)
    steps = args.steps if args.steps is not None else adapter.metadata().default_sampler.num_steps
    sampler = provider.sampler_for_steps(steps)
    if args.model in ("dp", "fastwam"):
        # Select the existing vector execution path, without a runtime or ledger.
        if args.model == "dp":
            from exp.r17_integration.models.dp.batch_runtime import BatchConfig, FrozenLanguageMemo
            if not args.dp_uncached_text:
                adapter.language = FrozenLanguageMemo(adapter.language)
        else:
            from exp.r17_integration.models.fastwam.batch_runtime import BatchConfig
        adapter.batch_config = BatchConfig(max(args.sizes), 0)
    return adapter, sampler


class SupplyLoop:
    """Batch orchestration stays CPU-testable; only payload construction imports torch."""

    def __init__(self, model, adapter, sampler, observation, *, synchronize,
                 payload_factory=None, finish_call=None):
        self.model, self.adapter, self.sampler, self.observation = model, adapter, sampler, observation
        self.batcher = adapter.make_stage_batcher()
        self.synchronize = synchronize
        self.payload_factory = payload_factory or self.miss_payload
        self.finish_call = finish_call or self.copy_actions
        self.sequence = 0

    def miss_payload(self, stage2, request):
        import torch
        from exp.r17_integration.common.contracts import CaptureSpec
        from exp.r17_integration.common.request_rng import RequestRNG
        from openpi.serving.batching_core import Stage3MissPayload

        rng = RequestRNG(request)
        if self.model == "dp":
            # This also preserves DP's generator AFTER the initial noise draw.
            return self.batcher.miss_payload(stage2, self.sampler, rng, CaptureSpec())
        if self.model == "pi05":
            noise = torch.from_numpy(rng.normal((self.adapter.metadata().layout.horizon, 32)))
            noise = noise.to(stage2.data.stage1.state.device)
            conditioning, points = self.adapter.stage3_batch_input(stage2), ()
        elif self.model == "groot":
            device = stage2.data.backbone_features.device
            noise = self.adapter.runner.sample_noise(stage2.data, generator=rng.torch_generator(device=str(device)))[0]
            conditioning, points = self.adapter.stage3_batch_input(stage2), None
        else:
            spec = self.adapter.metadata()
            noise = torch.randn((spec.layout.horizon, spec.layout.action_width),
                                generator=rng.torch_generator(device="cpu"), dtype=torch.float32)
            noise = noise.to(self.adapter.model.device, self.adapter.model.torch_dtype)
            conditioning, points = stage2, None
        return Stage3MissPayload(conditioning, noise, self.sampler.num_steps, points)

    @staticmethod
    def copy_actions(outputs):
        for out in outputs:
            action = getattr(out, "action_chunk", None)
            if action is None:
                action = out.action
            action.detach().float().cpu().clone()

    def run_batch(self, size, phase):
        from exp.r17_integration.common.contracts import RequestContext, Stage2

        if phase not in ("hit", "call"):
            raise ValueError(phase)
        protocol = self.adapter.metadata().allowed_benchmark_protocols[0]
        requests = [RequestContext("gpu_supply", protocol, "assembly-v3", 0, self.sequence + i,
                                   task_id=0, explicit_seed=195 + self.sequence + i) for i in range(size)]
        self.sequence += size
        prepared = [self.adapter.prepare(self.observation, request) for request in requests]
        wrapped = self.model in ("pi05", "groot")
        inputs = [self.adapter.stage1_batch_input(p) for p in prepared] if wrapped else prepared
        native1 = self.batcher.run_stage1_batch(inputs)
        stage1 = [self.adapter.finish_stage1(out, p) for out, p in zip(native1, prepared, strict=True)] if wrapped else native1
        for out in stage1:
            self.adapter.retrieval_fields(out)
        if phase == "call":
            native2 = self.batcher.run_stage2_batch(native1, capture=False)
            stage2 = [Stage2(out, s1) for out, s1 in zip(native2, stage1, strict=True)] if wrapped else native2
            payloads = [self.payload_factory(s2, request) for s2, request in zip(stage2, requests, strict=True)]
            points = () if self.model == "pi05" else None
            outputs = self.batcher.run_stage3_miss(payloads, num_steps=self.sampler.num_steps,
                                                 save_timesteps_per_request=[points] * size)
            self.finish_call(outputs)
        self.synchronize()


def run_phases(loop, sizes, plan, phase_seconds, *, model, memory, reset_memory,
               clock=time.time, monotonic=time.monotonic, sleep=time.sleep):
    """Common absolute slots; complete the last batch and include its overrun."""
    for slot, (size, phase) in enumerate((b, p) for b in sizes for p in ("hit", "call")):
        scheduled = plan["t0"] + slot * (phase_seconds + plan["gap_s"])
        deadline = scheduled + phase_seconds
        while clock() < scheduled:
            sleep(min(0.05, max(0, scheduled - clock())))
        begin = clock()
        if begin >= deadline:
            raise RuntimeError("phase missed its START slot; increase --gap-seconds or --phase-seconds")
        reset_memory()
        begin = clock()
        start = monotonic()
        batches = 0
        while clock() < deadline:
            loop.run_batch(size, phase)
            batches += 1
        wall = monotonic() - start
        if clock() > deadline + plan["gap_s"]:
            raise RuntimeError("batch overran the phase gap; increase --gap-seconds")
        yield dict(model=model, B=size, phase=phase, requests=batches * size, wall_s=wall,
                   begin=begin, scheduled_begin=scheduled, phase_s=phase_seconds,
                   mem_gib=memory(), pid=os.getpid(), steps=loop.sampler.num_steps, batches=batches)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=tuple(DESCRIPTORS), required=True)
    parser.add_argument("--descriptor", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--vendor", "--vendor-root", type=Path, help="GR00T checkout; relocate descriptor code/tokenizer paths")
    parser.add_argument("--observation", type=Path, default=OBSERVATION)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, help="MISS K; defaults to the formal descriptor/adapter sampler")
    parser.add_argument("--sizes", type=sizes_arg, default=sizes_arg("1,2,4,8,16"))
    parser.add_argument("--phase-seconds", type=positive_seconds, default=12)
    parser.add_argument("--warmup", type=int, default=1, help="untimed hit/call batches per B before READY")
    parser.add_argument("--dp-uncached-text", action="store_true", help="disable the batched DP server's exact-prompt CPU memo")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ready", type=Path, required=True)
    parser.add_argument("--start", type=Path, required=True, help='JSON {"t0": epoch seconds, "gap_s": seconds}')
    args = parser.parse_args()
    if args.warmup < 1 or (args.steps is not None and args.steps < 1):
        parser.error("warmup and steps must be positive")
    import numpy as np
    import torch

    adapter, sampler = load(args)
    with np.load(args.observation, allow_pickle=False) as data:
        observation = {name: data[name][0].copy() for name in ("observation/image", "observation/state")}
        observation["prompt"] = str(data["prompt"][0])
    loop = SupplyLoop(args.model, adapter, sampler, observation,
                      synchronize=lambda: torch.cuda.synchronize(args.device))
    with torch.no_grad():
        for size in args.sizes:
            for phase in ("hit", "call"):
                for _ in range(args.warmup):
                    loop.run_batch(size, phase)
        args.ready.write_text(str(os.getpid()))
        while not args.start.exists():
            time.sleep(0.1)
        plan = json.loads(args.start.read_text())
        if not math.isfinite(plan["t0"]) or not math.isfinite(plan["gap_s"]) or plan["gap_s"] < 0:
            raise ValueError("START needs finite t0 and nonnegative gap_s")
        with args.out.open("a") as stream:
            for row in run_phases(loop, args.sizes, plan, args.phase_seconds, model=args.model,
                                  memory=lambda: torch.cuda.max_memory_reserved(args.device) / 2**30,
                                  reset_memory=lambda: torch.cuda.reset_peak_memory_stats(args.device)):
                row["dp_text_memo"] = args.model == "dp" and not args.dp_uncached_text
                stream.write(json.dumps(row) + "\n")
                stream.flush()
                print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
