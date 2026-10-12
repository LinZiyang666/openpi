"""Batch-one R17 stage profile. Import/help/input inspection never initializes CUDA.

Run only later on an exclusive device. Native samplers stay in Python: only
stage tensor programs and the single network evaluation are compiled. The full
sample timings include scheduler arithmetic, RNG, validation and output copies.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
import subprocess
import time
from types import SimpleNamespace
import uuid

ROOT = Path(__file__).resolve().parents[3]
STEPS = {"dp_libero": (100, 10), "fastwam_libero": (10, 1), "cosmos_libero": (5, 2)}
SUITES = ("libero_spatial", "libero_10")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def summary(samples):
    import numpy as np
    values = np.asarray(samples, dtype=float)
    if not values.size or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("timing samples must be nonempty, finite and nonnegative")
    return dict(zip(("p10", "median", "p90"), map(float, np.percentile(values, [10, 50, 90]))))


def pricing(stage1, stage2, full, reduced):
    import math
    if not all(math.isfinite(v) and v >= 0 for v in (stage1, stage2, full, reduced)):
        raise ValueError("invalid stage costs")
    denominator = stage1 + stage2 + full
    if denominator <= 0:
        raise ValueError("zero full-step denominator")
    return {"vision": stage1 / denominator, "full_miss": (stage2 + full) / denominator,
            "reduced_miss": (stage2 + reduced) / denominator, "denominator_ms": denominator}


def actual_nfe(model, k):
    if k not in STEPS[model]:
        raise ValueError("unregistered profile step budget")
    return 1 if model == "cosmos_libero" and k == 2 else k


def certificate(expected, observed, captures, skips, compiles_after_warmup, compile_count):
    checks = {"launches_match": observed is not None and observed == expected and expected > 0,
              "no_new_captures": captures == 0, "no_skips": skips == 0,
              "no_new_compiles": compiles_after_warmup == 0, "three_compiled_stages": compile_count == 3}
    return {"certified": all(checks.values()), "checks": checks}


def read_input(model, suite, pool, row=0, observations=None):
    """Read a real row, including both camera bytes; no synthetic observations."""
    import numpy as np
    if observations:
        with np.load(observations, allow_pickle=False) as data:
            obs = {k: str(data[k][row]) if k == "prompt" else data[k][row].copy() for k in data.files}
        info = {"source": str(observations), "source_sha256": sha(observations), "row": row,
                "benchmark_protocol": None}
    else:
        ready = Path(pool) / "READY.json"
        manifest = json.loads(ready.read_text())
        record = manifest["rows"][row]
        if not record["task_name"].startswith(suite + "/"):
            raise ValueError("selected row belongs to a different suite")
        tasks = json.loads((ROOT / "exp/r17_integration/qualification/specs/libero_benchmark_spec.json").read_text())["tasks"]
        prompt = next(language for name, language, _ in tasks if name == record["task_name"])
        def array(name):
            return np.load(Path(pool) / manifest["arrays"][name]["file"], mmap_mode="r", allow_pickle=False)[row].copy()
        obs = {"observation/image": array("image_0"), "observation/wrist_image": array("image_1"),
               "observation/state": array("state"), "prompt": prompt}
        if model == "cosmos_libero":
            # The library stores the recorded native gripper/xyz/quaternion in
            # float32. Preserve these fields directly; never reconstruct a quat.
            obs["observation/cosmos_proprio"] = obs.pop("observation/state")
        info = {"source": str(ready), "source_sha256": sha(ready), "row": row,
                "row_id": record["row_id"], "task_name": record["task_name"],
                "campaign": record["campaign"], "benchmark_protocol": manifest["binding"]["benchmark_protocol"],
                "state_precision": ("library recorded float32; native quaternion fields, no reconstruction" if model == "cosmos_libero"
                                    else "library recorded float32"),
                "descriptor": str(Path(pool).parent / "campaigns" / record["campaign"] / "assets/load.json")}
    for name in ("observation/image", "observation/wrist_image"):
        value = obs[name]
        allowed = ((96, 96, 3), (256, 256, 3)) if model == "dp_libero" else ((256, 256, 3),)
        if value.dtype != np.uint8 or value.shape not in allowed:
            raise ValueError(f"invalid serving camera geometry: {name}: {value.shape}/{value.dtype}")
    state_key = "observation/cosmos_proprio" if model == "cosmos_libero" else "observation/state"
    state = obs[state_key]
    if state.shape != ((9,) if model == "cosmos_libero" else (8,)) or not np.isfinite(state).all():
        raise ValueError("invalid serving state")
    if not isinstance(obs["prompt"], str) or not obs["prompt"].strip():
        raise ValueError("real task prompt required")
    info.update(prompt=obs["prompt"], prompt_sha256=hashlib.sha256(obs["prompt"].encode()).hexdigest(),
                geometry={k: list(v.shape) for k, v in obs.items() if hasattr(v, "shape")},
                observation_sha256={k: hashlib.sha256(v.tobytes()).hexdigest()
                                    for k, v in obs.items() if hasattr(v, "shape")})
    return obs, info


def load_adapter(model, descriptor, device, vendor_root=None):
    desc = json.loads(Path(descriptor).read_text())
    if model == "dp_libero":
        from exp.r17_integration.models.dp.loader import load_adapter as load
        desc["device"] = device
        adapter = load(desc)
    elif model == "fastwam_libero":
        from exp.r17_integration.models.fastwam.loader import LoadSpec, load_adapter as load
        adapter = load(LoadSpec(**desc), device=device)
    else:
        from exp.r17_integration.models.cosmos.loader import load_adapter as load
        if vendor_root:
            desc["vendor_root"] = str(vendor_root)
        elif not Path(desc["vendor_root"]).exists():
            desc["vendor_root"] = "/home/weiland/projects/openpi_ext/lines/cosmos-policy"
        adapter = load(desc, device=device)
    return adapter, desc


def tensor_copy(value, *, cpu=False):
    """Copy graph outputs before the next replay overwrites their storage."""
    import torch
    from torch.utils._pytree import tree_map
    def own(x):
        if not isinstance(x, torch.Tensor):
            return x
        x = x.detach()
        return x.cpu().clone() if cpu else x.clone()
    return tree_map(own, value)


def parity(got, expected, rtol=0.02, atol=0.02):
    import numpy as np
    from torch.utils._pytree import tree_flatten
    a, spec_a = tree_flatten(tensor_copy(got, cpu=True))
    b, spec_b = tree_flatten(tensor_copy(expected, cpu=True))
    if spec_a != spec_b:
        return {"passed": False, "reason": "different output structure"}
    errors, relatives, ok = [], [], True
    for x, y in zip(a, b, strict=True):
        if hasattr(x, "numpy"):
            x, y = x.float().numpy(), y.float().numpy()
        x, y = np.asarray(x), np.asarray(y)
        if x.shape != y.shape:
            return {"passed": False, "reason": "different tensor shape"}
        diff = x.astype(float) - y.astype(float)
        errors.append(float(np.max(np.abs(diff))) if diff.size else 0.)
        relatives.append(float(np.linalg.norm(diff.ravel()) / max(np.linalg.norm(y.astype(float).ravel()), 1e-12)))
        ok &= bool(np.isfinite(x).all() and np.allclose(x, y, rtol=rtol, atol=atol))
    return {"passed": ok, "max_abs_delta": max(errors, default=0.),
            "rel_frobenius": max(relatives, default=0.), "rtol": rtol, "atol": atol}


def compile_tensor_program(fn, inputs):
    """Freeze Python structure, not tensor values, then compile one tensor stage.

    Native VAE caches and Cosmos condition objects are Python structures. Real
    FX tracing records their tensor work and specializes serving geometry and
    protocol branches. Eager-chain parity checks that specialization later.
    No adapter or vendor source is edited, and no graph-break fallback is used.
    """
    import torch
    from torch.fx.experimental.proxy_tensor import make_fx
    graph = make_fx(fn, tracing_mode="real", _error_on_data_dependent_ops=False)(*inputs)
    graph.graph.eliminate_dead_code()
    graph.recompile()
    # A scalar branch is legal only when it has been resolved during tracing.
    if any(node.target == torch.ops.aten._local_scalar_dense.default for node in graph.graph.nodes):
        raise RuntimeError("unresolved tensor-dependent branch in stage tensor program")
    return torch.compile(graph, mode="reduce-overhead", fullgraph=True, dynamic=False)


class GraphCounts:
    """Fallback observes actual torch CUDA graph methods, never assumes replay."""
    def __init__(self, torch):
        self.cls = torch.cuda.CUDAGraph
        self.launches = self.captures = 0
    def __enter__(self):
        self.original_replay, self.original_capture = self.cls.replay, self.cls.capture_begin
        owner = self
        def replay(graph, *args, **kwargs):
            result = owner.original_replay(graph, *args, **kwargs)
            owner.launches += 1
            return result
        def capture(graph, *args, **kwargs):
            result = owner.original_capture(graph, *args, **kwargs)
            owner.captures += 1
            return result
        self.cls.replay, self.cls.capture_begin = replay, capture
        return self
    def snapshot(self):
        return self.launches, self.captures
    def __exit__(self, *exc):
        self.cls.replay, self.cls.capture_begin = self.original_replay, self.original_capture


def counters():
    from torch._dynamo.utils import counters as source
    return {f"{group}.{name}": int(value) for group, values in source.items()
            for name, value in values.items() if isinstance(value, int)}


def skip_count(values):
    return sum(v for k, v in values.items() if "cudagraph_skip" in k or "skipped_cudagraph" in k)


class StepProbe:
    """Time network evaluations and the native update between evaluations."""
    def __init__(self, torch, fn):
        self.torch, self.fn, self.enabled = torch, fn, False
        self.intervals, self.networks = [], []
    def event(self):
        return self.torch.cuda.Event(enable_timing=True)
    def reset(self):
        self.intervals, self.networks = [], []
    def __call__(self, *args):
        if self.enabled:
            start = self.event()
            start.record()
            if self.intervals:
                self.intervals[-1][1] = start
        value = tensor_copy(self.fn(*args))
        if self.enabled:
            end = self.event()
            end.record()
            self.intervals.append([start, None])
            self.networks.append((start, end))
        return value
    def finish(self):
        if self.enabled and self.intervals:
            end = self.event()
            end.record()
            self.intervals[-1][1] = end
    def milliseconds(self):
        return ([a.elapsed_time(b) for a, b in self.intervals],
                [a.elapsed_time(b) for a, b in self.networks])


def build_stages(model, adapter, obs, request, *, precompute_text=False):
    """Profiling-only tensor equivalents of prepare+stage1, stage2 and sample."""
    import torch
    from exp.r17_integration.common.contracts import CaptureSpec, Stage1, Stage2
    from exp.r17_integration.common.request_rng import RequestRNG
    prepared = adapter.prepare(obs, request)
    eager1 = adapter.run_stage1(prepared)
    eager2 = adapter.run_stage2(eager1)
    native_sample = adapter.sample

    if model == "dp_libero":
        from exp.mw_finetune.dp.model import crop
        core = adapter.model._policy
        def first(inputs):
            normalized = {k: core.normalizer.normalize(k, v) for k, v in inputs.items()}
            features = {k: torch.relu(net(crop(normalized[k], False), normalized["lang_emb"]))
                        for k, net in core.obs_encoder.obs_nets.items()}
            return {**features, "state": normalized["state"], "lang_emb": normalized["lang_emb"]}
        c1 = compile_tensor_program(first, (prepared.data["inputs"],))
        def stage1(p):
            features = tensor_copy(c1(p.data["inputs"]))
            fields = {"vision_0": features["base_0_rgb"][0].float().cpu().numpy().copy(),
                      "vision_1": features["left_wrist_0_rgb"][0].float().cpu().numpy().copy(),
                      "robot_state": p.state.copy()}
            return Stage1({"prepared": p, "features": features}, fields, p.state)
        def second(features):
            return torch.cat([features[k] for k in core.obs_encoder.feature_order], -1).unsqueeze(1)
        c2 = compile_tensor_program(second, (eager1.data["features"],))
        def stage2(s):
            return Stage2({"conditioning": tensor_copy(c2(s.data["features"]))}, s)
        original = core.model
        cond = eager2.data["conditioning"]
        noise = torch.zeros((1, 16, 7), device=adapter.device, dtype=cond.dtype)
        c3 = compile_tensor_program(lambda x, t, c: original(x, t, c),
                                    (noise, torch.tensor([99], device=adapter.device), cond))
        probe = StepProbe(torch, c3)
        class Network(torch.nn.Module):
            def forward(self, x, t, c):
                return probe(x, t.to(device=x.device).reshape(1), c)
        core.model = Network().eval()
        restore = lambda: setattr(core, "model", original)

    elif model == "fastwam_libero":
        from exp.r17_integration.models.fastwam.staged import _ReusePatch
        m = adapter.model
        def first(pixels):
            first = m.vae.encode([(p * 2 - 1).unsqueeze(1) for p in pixels],
                                 device=m.device, tiled=False, tile_size=(30, 52), tile_stride=(15, 26))
            if isinstance(first, (tuple, list)):
                first = torch.stack(list(first))
            first = first.to(dtype=m.video_expert.patch_embedding.weight.dtype)
            patches = m.video_expert.patchify(first)
            tokens = patches.flatten(2).transpose(1, 2).contiguous()
            spatial = tokens.float().reshape(1, *patches.shape[3:], -1).permute(0, 3, 1, 2)
            pooled = torch.nn.functional.adaptive_avg_pool2d(spatial, (4, 4)).permute(0, 2, 3, 1).reshape(-1)
            return first, patches, tokens, pooled
        pixels = prepared.data["input_image"].to(m.device, m.torch_dtype)
        c1 = compile_tensor_program(first, (pixels,))
        def stage1(p):
            first, patches, tokens, pooled = tensor_copy(c1(p.data["input_image"].to(m.device, m.torch_dtype)))
            f, h, w = patches.shape[2:]
            data = dict(eager1.data)
            data.update(first_latents=first, patches=patches, video_seq_len=f*h*w,
                        proprio=p.data["proprio"].clone(), prompt=p.data["prompt"],
                        conditioning_source=(p.data["prompt"], p.data["input_image"], p.data["proprio"]),
                        video_pre={"tokens": tokens, "meta": {"grid_sizes": torch.tensor([[f, h, w]], device=m.device),
                        "batch_size": 1, "tokens_per_frame": h*w}})
            return Stage1(data, {"vision_0": pooled.float().cpu().numpy().copy(), "robot_state": p.state.copy()}, p.state.copy())
        ids, token_mask = m.tokenizer([prepared.data["prompt"]], return_mask=True, add_special_tokens=True)
        ids, token_mask = ids.to(m.device), token_mask.to(m.device, dtype=torch.bool)
        cached_context, cached_mask = m.encode_prompt([prepared.data["prompt"]]) if precompute_text else (None, None)
        if precompute_text:
            m.text_encoder.to("cpu")
            torch.cuda.empty_cache()
        def second(first, patches, proprio, ids, token_mask):
            if precompute_text:
                context, mask = cached_context, cached_mask
            else:
                context = m.text_encoder(ids, token_mask)
                lengths = token_mask.gt(0).sum(dim=1).long()
                context = context.masked_fill(torch.arange(context.shape[1], device=context.device)[None, :, None]
                                              >= lengths[:, None, None], 0)
                mask = torch.ones_like(token_mask)
            context, mask = m._append_proprio_to_context(context, mask, proprio)
            proxy = _ReusePatch(m.video_expert, patches)
            pre = m.video_expert.pre_dit.__func__(proxy, x=first,
                timestep=torch.zeros(1, device=m.device, dtype=first.dtype), context=context, context_mask=mask,
                action=None, fuse_vae_embedding_in_latents=bool(m.video_expert.fuse_vae_embedding_in_latents))
            seq = eager1.data["video_seq_len"]
            attn = m._build_mot_attention_mask(video_seq_len=seq, action_seq_len=adapter.metadata().layout.horizon,
                video_tokens_per_frame=int(pre["meta"]["tokens_per_frame"]), device=first.device)
            kv = m.mot.prefill_video_cache(video_tokens=pre["tokens"], video_freqs=pre["freqs"], video_t_mod=pre["t_mod"],
                video_context_payload={"context": pre["context"], "mask": pre["context_mask"]}, video_attention_mask=attn[:seq, :seq])
            return context, mask, attn, kv
        c2 = compile_tensor_program(second, (eager1.data["first_latents"], eager1.data["patches"],
            eager1.data["proprio"].to(m.device, m.torch_dtype), ids, token_mask))
        def stage2(s):
            # Tokenization remains in the measured stage2, as in the adapter.
            ti, tm = m.tokenizer([s.data["prompt"]], return_mask=True, add_special_tokens=True)
            context, mask, attn, kv = tensor_copy(c2(s.data["first_latents"], s.data["patches"],
                s.data["proprio"].to(m.device, m.torch_dtype), ti.to(m.device), tm.to(m.device, dtype=torch.bool)))
            data = {k: s.data[k] for k in ("video_seq_len", "conditioning_source", "model_identity")}
            data.update(context=context, context_mask=mask, attention_mask=attn, video_kv_cache=kv, capture_cp2_enabled=False)
            return Stage2(data, s)
        original = m._predict_action_noise_with_cache
        def network(x, t, context, mask, kv, attn):
            return original(latents_action=x, timestep_action=t, context=context, context_mask=mask,
                video_kv_cache=kv, attention_mask=attn, video_seq_len=eager1.data["video_seq_len"])
        d = eager2.data
        c3 = compile_tensor_program(network, (torch.zeros((1, 32, 7), device=m.device, dtype=m.torch_dtype),
            torch.ones(1, device=m.device, dtype=m.torch_dtype), d["context"], d["context_mask"], d["video_kv_cache"], d["attention_mask"]))
        probe = StepProbe(torch, c3)
        def wrapped(**kw):
            return probe(*(kw[k] for k in ("latents_action", "timestep_action", "context", "context_mask", "video_kv_cache", "attention_mask")))
        m._predict_action_noise_with_cache = wrapped
        restore = lambda: setattr(m, "_predict_action_noise_with_cache", original)

    else:
        from exp.r17_integration.models.cosmos.staged import make_data_batch
        from cosmos_policy.utils.utils import set_seed_everywhere
        m = adapter.model
        def first(batch):
            batch = dict(batch)
            m._normalize_video_databatch_inplace(batch)
            m._augment_image_dim_inplace(batch)
            latent = m.encode(batch["video"]).contiguous().float()
            keys = tuple(torch.nn.functional.adaptive_avg_pool2d(latent[:, :, i], (4, 4)).flatten() for i in (3, 2))
            return batch, latent, keys
        def batch_for(p):
            return make_data_batch(p.data["native"], adapter.embeddings[p.data["prompt"]], adapter.cfg,
                                   adapter.stats, m, adapter.cu, torch)
        set_seed_everywhere(0)
        c1 = compile_tensor_program(first, (batch_for(prepared),))
        def stage1(p):
            set_seed_everywhere(0)
            batch, latent, keys = tensor_copy(c1(batch_for(p)))
            fields = {"vision_0": keys[0].cpu().numpy().copy(), "vision_1": keys[1].cpu().numpy().copy(),
                      "robot_state": p.state.copy()}
            return Stage1({"batch": batch, "latent": latent, "model_identity": adapter.metadata().identity}, fields, p.state.copy())
        template = inspect.getclosurevars(eager2.data["x0_fn"]).nonlocals
        if template.get("uncondition") is not None or m.config.use_flowunipc_scheduler:
            raise ValueError("this profile expects the formal always-conditional Cosmos EDM path")
        condition = template["condition"]
        names = tuple(k for k, v in vars(condition).items() if isinstance(v, torch.Tensor))
        def second(batch, latent):
            fn = m.get_x0_fn_from_batch(dict(batch), 1.5, is_negative_prompt=False,
                    skip_vae_encoding=True, previous_generated_latent=latent)
            cond = inspect.getclosurevars(fn).nonlocals["condition"]
            return tuple(getattr(cond, k) for k in names)
        c2 = compile_tensor_program(second, (eager1.data["batch"], eager1.data["latent"]))
        def network(x, sigma, values):
            cond = copy.copy(condition)
            for k, v in zip(names, values, strict=True):
                setattr(cond, k, v)
            return m.denoise(x, sigma, cond).x0
        values = tuple(getattr(condition, k) for k in names)
        c3 = compile_tensor_program(network, (torch.zeros_like(eager1.data["latent"]),
                    torch.tensor([80.], device=m.tensor_kwargs["device"]), values))
        probe = StepProbe(torch, c3)
        def stage2(s):
            values = tensor_copy(c2(s.data["batch"], s.data["latent"]))
            return Stage2({"x0_fn": lambda x, sigma: probe(x, sigma, values),
                           "condition_tensors": values, "latent_shape": tuple(s.data["latent"].shape),
                           "model_identity": adapter.metadata().identity}, s)
        restore = lambda: None

    def sample(s, sampler, rng_request=None):
        probe.reset()
        result = native_sample(s, sampler, RequestRNG(rng_request or request), CaptureSpec())
        probe.finish()
        return result
    return SimpleNamespace(stage1=stage1, stage2=stage2, sample=sample, probe=probe, restore=restore)


def stage_payload(model, stage, which):
    if which == 1:
        return stage.fields
    if model == "dp_libero":
        return stage.data["conditioning"]
    if model == "fastwam_libero":
        return {k: stage.data[k] for k in ("context", "context_mask", "attention_mask", "video_kv_cache")}
    if "condition_tensors" in stage.data:
        return stage.data["condition_tensors"]
    import torch
    cond = inspect.getclosurevars(stage.data["x0_fn"]).nonlocals["condition"]
    return tuple(v for v in vars(cond).values() if isinstance(v, torch.Tensor))


def measure_iterations(torch, runtime, adapter, obs, request, sampler, iters):
    rows = {name: [] for name in ("stage1_ms", "stage2_ms", "stage3_ms", "total_ms", "stage3_step_ms", "stage3_network_ms", "wall_ms")}
    nfe = []
    runtime.probe.enabled = True
    try:
        for _ in range(iters):
            torch.compiler.cudagraph_mark_step_begin()
            events = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
            tick = time.perf_counter()
            events[0].record()
            prepared = adapter.prepare(obs, request)
            s1 = runtime.stage1(prepared)
            events[1].record()
            s2 = runtime.stage2(s1)
            events[2].record()
            result = runtime.sample(s2, sampler)
            events[3].record()
            events[3].synchronize()
            rows["wall_ms"].append((time.perf_counter() - tick) * 1000)
            for i, name in enumerate(("stage1_ms", "stage2_ms", "stage3_ms")):
                rows[name].append(events[i].elapsed_time(events[i+1]))
            rows["total_ms"].append(events[0].elapsed_time(events[3]))
            steps, networks = runtime.probe.milliseconds()
            if len(steps) != result.actual_nfe:
                raise RuntimeError("observed network calls differ from adapter NFE receipt")
            rows["stage3_step_ms"].extend(steps)
            rows["stage3_network_ms"].extend(networks)
            nfe.append(result.actual_nfe)
    finally:
        runtime.probe.enabled = False
    return rows, nfe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=STEPS, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--suite", choices=SUITES, default=SUITES[0])
    parser.add_argument("--pool", type=Path)
    parser.add_argument("--row", type=int, default=0)
    parser.add_argument("--observations", type=Path, help="real wire-observation NPZ, arrays with a leading row axis")
    parser.add_argument("--descriptor", type=Path, help="formal load.json; inferred from B row if omitted")
    parser.add_argument("--vendor-root", type=Path, help="local Cosmos checkout path, recorded in receipt")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--seed", type=int, help="default: Cosmos195, other0")
    parser.add_argument("--precompute-text", action="store_true", help="Fast-WAM only; separately labelled cached-text cost")
    parser.add_argument("--inspect-input", action="store_true", help="CPU only: write observation provenance without loading weights")
    args = parser.parse_args()
    if args.warmup < 3 or args.iters < 1 or args.row < 0:
        parser.error("warmup >=3, iters >=1, row >=0 required")
    if args.precompute_text and args.model != "fastwam_libero":
        parser.error("--precompute-text is Fast-WAM only")
    pool = args.pool or Path("/data/r17_library") / (args.model + "_" + args.suite) / "bpool"
    obs, source = read_input(args.model, args.suite, pool, args.row, args.observations)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.inspect_input:
        args.out.write_text(json.dumps(source, indent=2) + "\n")
        return
    descriptor = args.descriptor or source.get("descriptor")
    if descriptor is None:
        parser.error("--observations requires --descriptor")
    import torch
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        parser.error("measurement requires CUDA; CPU work uses --inspect-input or stub tests")
    torch.cuda.set_device(args.device)
    started = time.monotonic()
    adapter, loaded_descriptor = load_adapter(args.model, descriptor, args.device, args.vendor_root)
    from exp.r17_integration.common.contracts import CaptureSpec, RequestContext
    from exp.r17_integration.common.request_rng import RequestRNG
    protocol = source["benchmark_protocol"] or adapter.metadata().allowed_benchmark_protocols[0]
    seed = args.seed if args.seed is not None else (195 if args.model == "cosmos_libero" else 0)
    request = RequestContext("latency", protocol, source.get("task_name", args.suite), 0, 0, explicit_seed=seed)
    def sampler(k):
        if args.model == "dp_libero":
            from exp.r17_integration.models.dp.sampler import SamplerProfile
            return SamplerProfile("auto", k, 0.).to_spec(adapter.model.cfg)
        return adapter.sampler_for_steps(k)
    samplers = {k: sampler(k) for k in STEPS[args.model]}
    # Reference from the untouched formal adapter, with its actual initial and
    # posterior RNG stream. Save owned CPU copies before any compiled replay.
    reference = {}
    for k, spec in samplers.items():
        p = adapter.prepare(obs, request)
        s1 = adapter.run_stage1(p)
        s2 = adapter.run_stage2(s1)
        result = adapter.sample(s2, spec, RequestRNG(request), CaptureSpec())
        reference[k] = (tensor_copy(stage_payload(args.model, s1, 1), cpu=True),
                        tensor_copy(stage_payload(args.model, s2, 2), cpu=True),
                        tensor_copy(result.normalized_chunk, cpu=True), adapter.to_wire(result.normalized_chunk, p).actions.copy())
    before = counters()
    with torch.inference_mode():
        runtime = build_stages(args.model, adapter, obs, request, precompute_text=args.precompute_text)
    marker = "r17_latency_" + uuid.uuid4().hex
    cells = {}
    try:
        with torch.inference_mode(), GraphCounts(torch) as graphs:
            for k, spec in samplers.items():
                for _ in range(args.warmup):
                    torch.compiler.cudagraph_mark_step_begin()
                    p = adapter.prepare(obs, request)
                    runtime.sample(runtime.stage2(runtime.stage1(p)), spec)
                torch.cuda.synchronize()
                p = adapter.prepare(obs, request)
                torch.compiler.cudagraph_mark_step_begin()
                s1 = runtime.stage1(p)
                s2 = runtime.stage2(s1)
                result = runtime.sample(s2, spec)
                a, b, c, wire = reference[k]
                parity_row = {"stage1": parity(stage_payload(args.model, s1, 1), a),
                              "stage2": parity(stage_payload(args.model, s2, 2), b),
                              "normalized_actions": parity(result.normalized_chunk, c),
                              "wire_actions": parity(adapter.to_wire(result.normalized_chunk, p).actions, wire)}
                replay = runtime.sample(s2, spec)
                fixed_equal = bool(torch.equal(result.normalized_chunk, replay.normalized_chunk))
                fresh = runtime.sample(s2, spec, replace(request, explicit_seed=seed+1))
                noise_distinct = not bool(torch.equal(result.normalized_chunk, fresh.normalized_chunk))
                cells[str(k)] = {"k": k, "actual_nfe": result.actual_nfe, "sampler": asdict(spec),
                                 "eager_vs_compiled_parity": parity_row, "fixed_noise_replay_equal": fixed_equal,
                                 "sampled_noise_distinct": noise_distinct}
            torch.cuda.synchronize()
            warm_counters = counters()
            warm_graphs = graphs.snapshot()
            warmup_peak = {"allocated_gib": torch.cuda.max_memory_allocated()/2**30,
                           "reserved_gib": torch.cuda.max_memory_reserved()/2**30}
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.profiler.start()
            torch.cuda.nvtx.range_push(marker)
            try:
                for k, spec in samplers.items():
                    tick_graphs, tick_counters = graphs.snapshot(), counters()
                    samples, nfe = measure_iterations(torch, runtime, adapter, obs, request, spec, args.iters)
                    after_graphs, after_counters = graphs.snapshot(), counters()
                    cell = cells[str(k)]
                    cell.update(samples)
                    cell.update(summary_ms={name: summary(values) for name, values in samples.items()},
                        expected_actual_nfe=actual_nfe(args.model, k), actual_nfe_by_iteration=nfe,
                        expected_cudagraph_launch_count=args.iters * (2 + actual_nfe(args.model, k)),
                        cudagraph_launch_count=after_graphs[0] - tick_graphs[0],
                        capture_calls_after_warmup=after_graphs[1] - tick_graphs[1],
                        compiles_after_warmup=after_counters.get("stats.unique_graphs", 0) - tick_counters.get("stats.unique_graphs", 0))
            finally:
                torch.cuda.nvtx.range_pop()
                torch.cuda.profiler.stop()
            end_counters = counters()
            counts = graphs.snapshot()
            graph_count = end_counters.get("stats.unique_graphs", 0) - before.get("stats.unique_graphs", 0)
            skips = skip_count(end_counters) - skip_count(before)
            new_compiles = end_counters.get("stats.unique_graphs", 0) - warm_counters.get("stats.unique_graphs", 0)
            for cell in cells.values():
                cell["cudagraph_skips"] = skips
                cell.update(certificate(cell["expected_cudagraph_launch_count"], cell["cudagraph_launch_count"],
                    cell["capture_calls_after_warmup"], skips, cell["compiles_after_warmup"], graph_count))
                cell["valid"] = (cell["certified"] and cell["fixed_noise_replay_equal"] and cell["sampled_noise_distinct"]
                    and all(p["passed"] for p in cell["eager_vs_compiled_parity"].values())
                    and all(n == cell["expected_actual_nfe"] for n in cell["actual_nfe_by_iteration"]))
            full, reduced = (cells[str(k)]["summary_ms"] for k in STEPS[args.model])
            costs = pricing(full["stage1_ms"]["median"], full["stage2_ms"]["median"],
                            full["stage3_ms"]["median"], reduced["stage3_ms"]["median"])
            props = torch.cuda.get_device_properties(args.device)
            driver_rows = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,driver_version", "--format=csv,noheader"],
                                         text=True, capture_output=True, check=False).stdout.strip()
            gpu_uuid = str(getattr(props, "uuid", "unavailable"))
            visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
            physical = visible[torch.cuda.current_device()] if visible != [""] else str(torch.cuda.current_device())
            driver = "unavailable"
            for line in driver_rows.splitlines():
                fields = [v.strip() for v in line.split(",")]
                if len(fields) == 3 and (fields[1] == gpu_uuid or fields[0] == physical or fields[1] == physical):
                    _, gpu_uuid, driver = fields
                    break
            expected = sum(c["expected_cudagraph_launch_count"] for c in cells.values())
            content = getattr(adapter, "asset_digests", getattr(getattr(adapter, "manifest", None), "content_digests", {}))
            weight_sha = getattr(adapter, "weight_identity", None) or next((v for k, v in content.items()
                if k == "base_weights" or k.endswith("Cosmos-Policy-LIBERO-Predict2-2B.pt")), None)
            if weight_sha:
                weight_sha = weight_sha.removeprefix("sha256:")
            git_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                                        capture_output=True, check=False).stdout.strip()
            receipt = dict(schema="r17.batch1_latency.v1", mode_run="measure", profile="libero", model=args.model,
                suite=args.suite, batch_size=1, mode="reduce-overhead", fullgraph=True, warmup=args.warmup,
                iters=args.iters, seed=seed, cells=cells, coefficients=costs,
                stage1_ms=full["stage1_ms"]["median"], stage2_ms=full["stage2_ms"]["median"],
                stage3_full_loop_ms=full["stage3_ms"]["median"], stage3_reduced_loop_ms=reduced["stage3_ms"]["median"],
                stage3_step_ms=full["stage3_step_ms"]["median"], stage3_reduced_step_ms=reduced["stage3_step_ms"]["median"],
                num_steps=STEPS[args.model][0],
                compile_count=graph_count, compiles_after_warmup=new_compiles,
                capture_calls_after_warmup=counts[1]-warm_graphs[1], cudagraph_skips=skips,
                expected_cudagraph_launch_count=expected, cudagraph_launch_count=counts[0]-warm_graphs[0],
                certification_source="torch.cuda.CUDAGraph.replay/capture_begin + Dynamo counters",
                trace_marker=marker, cuda_trace_path=None, inductor_counters=end_counters,
                **certificate(expected, counts[0]-warm_graphs[0], counts[1]-warm_graphs[1], skips, new_compiles, graph_count),
                valid=all(c["valid"] for c in cells.values()), coefficient_source_k=STEPS[args.model][0],
                full_steps=STEPS[args.model][0], reduced_steps=STEPS[args.model][1],
                coefficient_rounding="none; reduced loop measured directly; full-step denominator shared",
                timing_scope="CUDA events around prepare+run_stage1, run_stage2, sample; includes host dispatch gaps; excludes to_wire/search/transport",
                step_timing_scope="per NFE: network through native update until next network; last interval includes sample output conversion; RNG/setup before first network is only in full loop",
                input=source, descriptor=str(descriptor), descriptor_sha256=sha(descriptor), loaded_descriptor=loaded_descriptor,
                checkpoint_identity=adapter.metadata().checkpoint_identity,
                ckpt_sha256=weight_sha, model_spec=asdict(adapter.metadata()), content_digests=content,
                git_commit=git_commit,
                source_sha256={"profile": sha(__file__), "adapter": sha(inspect.getfile(type(adapter)))},
                torch=torch.__version__, cuda=torch.version.cuda, gpu_name=props.name,
                gpu_uuid=gpu_uuid, driver=driver,
                device=args.device, device_index=torch.cuda.current_device(), gpu_total_gib=props.total_memory/2**30,
                peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                load_compile_warmup_peak=warmup_peak,
                precompute_text=args.precompute_text, host=platform.node(), python=platform.python_version(),
                ts=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic()-started)
            args.out.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
            print(json.dumps({"out": str(args.out), "valid": receipt["valid"], "coefficients": costs,
                              "elapsed_seconds": receipt["elapsed_seconds"]}))
    finally:
        runtime.restore()


if __name__ == "__main__":
    main()
