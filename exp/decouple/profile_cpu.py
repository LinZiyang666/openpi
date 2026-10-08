"""CPU-only recorded B-pool component profiler, writes only under --out.

    python -m exp.decouple.profile_cpu --out exp/decouple/results --rows 32

The two model profiles run in fresh main-interpreter subprocesses. GR00T adds
its island dependencies to sys.path, because the main environment cannot load
its CPU transforms with the main transformers version. No model is constructed.
"""
from __future__ import annotations

import argparse
import cProfile
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def profile(args):
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    # Prevent HF's dynamic module copying and update checks from writing or
    # accessing anything beyond this output directory during CPU profiling.
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      HF_MODULES_CACHE=str(out / "hf_modules"), NO_ALBUMENTATIONS_UPDATE="1")
    if args.model == "groot":
        from exp.decouple.cpu_transforms import ISLAND_PACKAGES
        sys.path.insert(0, str(ISLAND_PACKAGES))
    import numpy as np
    import torch
    from openpi_client import msgpack_numpy
    from exp.decouple.recorded import load, observation
    from exp.decouple import cpu_transforms
    from exp.libero_groot.policy_adapter import build_groot_observation
    data = load(args.model, limit=args.rows)
    inputs, output_transform = getattr(cpu_transforms, args.model)()
    if args.model == "groot":
        from types import SimpleNamespace
        from gr00t.model.gr00t_n1 import GR00T_N1_5
        from transformers.feature_extraction_utils import BatchFeature
        # Call the actual preparation body without constructing any modules.
        # CPU placement excludes CUDA transfer/launch costs; record that limit.
        prepare_self = SimpleNamespace(device="cpu", action_horizon=16, action_dim=32,
                                       backbone=SimpleNamespace(prepare_input=lambda value: BatchFeature(data=value)),
                                       action_head=SimpleNamespace(dtype=torch.bfloat16,
                                           prepare_input=lambda value: BatchFeature(data=value)))
        prepare_self.validate_inputs = lambda value: GR00T_N1_5.validate_inputs(prepare_self, value)
    samples = {}
    subcalls = {}
    if args.model == "groot":
        processor = inputs[-1].eagle_processor
        for label, instance in (("tokenizer_only_ms", processor.tokenizer),
                                ("image_processor_only_ms", processor.image_processor)):
            cls = type(instance)
            original = cls.__call__
            def timed_call(self, *positional, _original=original, _label=label, **kwargs):
                start = time.perf_counter_ns()
                value = _original(self, *positional, **kwargs)
                subcalls[_label] = subcalls.get(_label, 0.) + (time.perf_counter_ns() - start) / 1e6
                return value
            cls.__call__ = timed_call
    profiler = cProfile.Profile()
    def measure(name, fn):
        start = time.perf_counter_ns()
        result = fn()
        samples.setdefault(name, []).append((time.perf_counter_ns() - start) / 1e6)
        return result

    def transforms(index):
        subcalls.clear()
        obs = observation(data, index)
        if args.model == "groot":
            # The available B image archive contains post-resize 224px Pi0.5
            # frames, not raw 256px GR00T frames. Shape adaptation is explicit
            # profiling input construction, outside timed serving transforms.
            import cv2
            for name in ("observation/image", "observation/wrist_image"):
                obs[name] = cv2.resize(obs[name], (256, 256), interpolation=cv2.INTER_LINEAR)
            obs = {k: np.expand_dims(np.asarray(v), 0)
                   for k, v in build_groot_observation(obs).items()}
        token_ms = 0.
        total = time.perf_counter_ns()
        for transform in inputs:
            start = time.perf_counter_ns()
            obs = transform(obs)
            if type(transform).__name__ in ("TokenizePrompt", "GR00TTransform"):
                token_ms += (time.perf_counter_ns() - start) / 1e6
        elapsed = (time.perf_counter_ns() - total) / 1e6
        samples.setdefault("transforms_excluding_token_prep_ms", []).append(elapsed - token_ms)
        samples.setdefault("token_prep_ms", []).append(token_ms)
        for name, value in subcalls.items():
            samples.setdefault(name, []).append(value)
        return obs

    # Prime lazy CPU tokenizer/processor imports; keep startup outside samples.
    transforms(0)
    samples.clear()
    profiler.enable()
    for i in range(len(data.rows)):
        obs = observation(data, i)
        packed = measure("websocket_pack_ms", lambda: msgpack_numpy.Packer().pack(obs))
        decoded = measure("websocket_unpack_ms", lambda: msgpack_numpy.unpackb(packed))
        transformed = measure("input_total_ms", lambda: transforms(i))
        def stage_glue():
            if args.model == "pi05":
                import jax
                # Matches the interceptor's unbatched CPU conversion.
                return jax.tree.map(lambda value: torch.from_numpy(np.array(value))
                                    if not torch.is_tensor(value) else value, transformed)
            return GR00T_N1_5.prepare_input(prepare_self, transformed)
        measure("model_stage_cpu_glue_ms", stage_glue)
        action = torch.from_numpy(np.array(data.arrays["action"][int(data.rows[i])]))
        def output():
            if args.model == "pi05":
                value = {"actions": action.numpy(), "state": np.array(data.arrays["rs"][int(data.rows[i])])}
                for transform in output_transform:
                    value = transform(value)
                return value
            return output_transform.unapply({"action": action[None]})
        measure("output_transforms_ms", output)
    profiler.disable()
    profiler.dump_stats(str(out / f"{args.model}_transforms.prof"))

    # Real frozen retrieval/corrector and decision log measurement. Run in the
    # main package environment separately so no test stubs import island torch.
    record = dict(model=args.model, rows=len(data.rows), source=str(data.path), pool="B",
                  image_source=str(data.image_source),
                  image_note="recorded model-input images; state keys are normalized, not reconstructed simulator state",
                  groot_image_adaptation=("224px B frames resized to 256px outside timed transforms" if args.model == "groot" else None),
                  python=sys.executable, torch=torch.__version__, numpy=np.__version__,
                  components={k: dict(mean_ms=float(np.mean(v)), p50_ms=float(np.percentile(v, 50)),
                                      p99_ms=float(np.percentile(v, 99))) for k, v in samples.items()},
                  glue_note=("original recursive interceptor CPU tensor conversion" if args.model == "pi05" else
                             "actual GR00T prepare_input body with CPU device and bf16 casts; CUDA transfers/launches excluded"),
                  token_note=("TokenizePrompt includes SentencePiece and state prompt construction" if args.model == "pi05"
                              else "GR00TTransform includes tokenizer, image processor, padding and collate"))
    (out / f"{args.model}_transforms.json").write_text(json.dumps(record, indent=2) + "\n")


def methods(args):
    import numpy as np
    from exp.decouple.recorded import load
    from exp.decouple.replay import frozen_runtime, make_connections, tape
    from openpi.serving.batching_coordinator import BatchingCoordinator
    from tests.serving.decouple.stubs import PiModel
    model = PiModel()
    core = BatchingCoordinator(model, max_batch_size=1, max_wait_ms=0)
    core.start()
    out = Path(args.out).resolve()
    rt, fit = frozen_runtime(args.model, out / f"{args.model}_logs")
    data = load(args.model, limit=2)
    connections = make_connections(rt, data, core, count=2)
    timings, queries, logs = [], [], []
    originals = []
    for _, session, _ in connections:
        original = session.on_search
        def timed(ctx, original=original):
            start = time.perf_counter_ns()
            result = original(ctx)
            queries.append((time.perf_counter_ns() - start) / 1e6)
            return result
        session.on_search = timed
    emit = rt.emit
    def timed_emit(row):
        start = time.perf_counter_ns()
        result = emit(row)
        logs.append((time.perf_counter_ns() - start) / 1e6)
        return result
    rt.emit = timed_emit
    profiler = cProfile.Profile()
    profiler.enable()
    try:
        for ci, start, obs in tape(data, steps=args.rows):
            conn, session, inner = connections[ci]
            if session.pending:
                conn.on_episode_start(**start)
            t = time.perf_counter_ns()
            conn.infer(obs)
            timings.append((time.perf_counter_ns() - t) / 1e6)
        for conn, _, _ in connections:
            conn.on_episode_end(success=False)
    finally:
        profiler.disable()
        core.stop()
    profiler.dump_stats(str(out / f"{args.model}_methods.prof"))
    def stats(values):
        return dict(n=len(values), mean_ms=float(np.mean(values)), p50_ms=float(np.percentile(values, 50)),
                    p99_ms=float(np.percentile(values, 99)))
    result = dict(model=args.model, fit=str(fit), source=str(data.path), pool="B",
                  decisions=stats(timings), retrieval_corrector_including_plugin_glue=stats(queries),
                  logging_write=stats(logs), stage1_calls=sum(inner.encodes for _, _, inner in connections),
                  policy_calls=sum(inner.policy_calls for _, _, inner in connections))
    (out / f"{args.model}_methods.json").write_text(json.dumps(result, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument("--model", choices=("pi05", "groot"))
    parser.add_argument("--methods", action="store_true")
    args = parser.parse_args()
    if args.model:
        (methods if args.methods else profile)(args)
        return
    for model in ("pi05", "groot"):
        for extra in ([], ["--methods"]):
            subprocess.run([sys.executable, "-m", "exp.decouple.profile_cpu", "--out", args.out,
                            "--rows", str(args.rows), "--model", model, *extra], check=True)


if __name__ == "__main__":
    main()
