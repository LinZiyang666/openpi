"""Native Fast-WAM cuts, request-local flow sampler and strict suffix resume.

The loaded vendor model is read-only. No scheduler/head/global seed is mutated.
Production is B=1. Explicit CPU fixture vector hooks retain the underlying
VAE/pre-DiT/prefill/head calls, but confer no GPU batching certificate.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import threading

from exp.r17_integration.common.contracts import (
    CPUModelBlueprint, CaptureSpec, NativeSnapshot, SampleResult, Stage1, Stage2,
)
from .key_builder import FastWAMKeyBuilder
from .manifest import make_sampler


def _cpu_array(value):
    return value.detach().float().cpu().numpy().copy()


def _scope_digest(prompt, input_image, proprio):
    digest = hashlib.sha256(b"r17.fastwam.conditioning.v2\0")
    prompt = prompt.encode("utf-8")
    digest.update(len(prompt).to_bytes(8, "little")); digest.update(prompt)
    for name, tensor in (("input_image", input_image), ("proprio", proprio)):
        value = tensor.detach().contiguous()
        if value.device.type != "cpu":
            raise ValueError("conditioning identity requires CPU-prepared wire inputs")
        metadata = json.dumps([name, str(value.dtype), list(value.shape)], separators=(",", ":")).encode()
        digest.update(len(metadata).to_bytes(8, "little")); digest.update(metadata)
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def conditioning_scope(prepared):
    """Bind resume to effective model inputs before transferring them to GPU.

    Wire inputs are CPU tensors. Hash their values, typed geometry and prompt;
    never run text encoding or copy latents/KV from GPU for this identity.
    """
    return _scope_digest(prepared.data["prompt"], prepared.data["input_image"], prepared.data["proprio"])


def stage_conditioning_scope(data):
    """Lazily compute (once per stage) the native snapshot/resume scope.

    Hashing the CPU-prepared image costs milliseconds per request, so it is
    paid only when a snapshot is actually created or a resume is validated,
    never on HIT/key-only or capture-off MISS decisions.
    """
    scope = data.get("conditioning_scope")
    if scope is None:
        scope = _scope_digest(*data["conditioning_source"])
        data["conditioning_scope"] = scope
    return scope


class _ReusePatch:
    """Call unchanged native pre_dit with request-owned patch embeddings.

    No resident method is replaced. All other attributes/methods resolve to the
    actual video expert; only the already executed convolution is reused.
    """
    def __init__(self, expert, patches):
        self.expert, self.patches = expert, patches

    def __getattr__(self, name):
        return getattr(self.expert, name)

    def patchify(self, x):
        if x.shape[0] != self.patches.shape[0]:
            raise ValueError("native pre_dit changed request batch")
        return self.patches


class FastWAMAdapter:
    def __init__(self, model, manifest, wire, config, *, stage1_lock=None,
                 allow_unqualified_batch=False, phase_observer=None, batch_config=None, batch_ledger=None):
        self.model, self.manifest, self.wire, self.config = model, manifest, wire, dict(config)
        self._key_builder = FastWAMKeyBuilder(manifest.spec.feature_spec)
        # Native raw Wan VAE encode clears mutable temporal caches. Serialize
        # that stage across connection wrappers; vector batches still share one
        # VAE call. Prefill/head operate on request-owned tensors.
        self.stage1_lock = stage1_lock or threading.RLock()
        # Test-only vector hooks remain available; production loaders never opt
        # in until real BF16 batch-composition parity has been certified.
        self.allow_unqualified_batch = allow_unqualified_batch
        self.phase_observer = phase_observer
        self.batch_config, self.batch_ledger = batch_config, batch_ledger

    def _check_batch(self, count):
        if self.batch_config is not None:
            if self.metadata().model_id not in ("fastwam_libero", "fastwam_mt50") or not 0 <= count <= self.batch_config.max_batch:
                raise ValueError("Fast-WAM batch exceeds explicit execution study limit")
            return
        if count > 1 and not self.allow_unqualified_batch:
            raise ValueError("Fast-WAM B>1 disabled: GPU batch-composition parity unqualified")

    def _phase(self, name, operation):
        return operation() if self.phase_observer is None else self.phase_observer(name, operation)

    def metadata(self):
        return self.manifest.spec

    def prepare(self, wire_obs, request):
        return self.wire.prepare(wire_obs, request)

    def extract_state_only(self, wire_obs):
        return self.wire.extract_state_only(wire_obs)

    def to_wire(self, normalized_chunk, prepared):
        return self.wire.to_wire(normalized_chunk, prepared)

    def sampler_for_steps(self, k):
        if self.wire.protocol.fixed_seed is not None and k != self.wire.protocol.default_k:
            raise ValueError("released LIBERO protocol requires K10; use system profile for reduced steps")
        return make_sampler(self.model.infer_action_scheduler, k,
            dtype=str(self.model.torch_dtype).removeprefix("torch."))

    def _validate_sampler(self, sampler):
        expected = self.sampler_for_steps(sampler.num_steps)
        if sampler != expected:
            raise ValueError("Fast-WAM native sampler/grid/shift/dtype identity mismatch")

    def run_stage1(self, prepared):
        return self.run_stage1_batch([prepared])[0]

    def run_stage1_batch(self, prepared):
        import torch
        if not prepared:
            return []
        self._check_batch(len(prepared))
        model = self.model
        if getattr(model.video_expert, "video_attention_mask_mode", "") != "first_frame_causal":
            raise ValueError("Fast-WAM requires first_frame_causal video attention")
        shape = prepared[0].data["input_image"].shape
        if any(p.data["input_image"].shape != shape for p in prepared):
            raise ValueError("stage1 batch must have one camera geometry")
        for p in prepared:
            if any(p.data[name].device.type != "cpu" for name in ("input_image", "proprio")):
                raise ValueError("conditioning identity requires CPU-prepared wire inputs")
        with self.stage1_lock, torch.inference_mode():
            pixels = torch.cat([p.data["input_image"] for p in prepared]).to(model.device, model.torch_dtype)
            if pixels.ndim != 4 or pixels.shape[1] != 3 or any(n % 16 for n in pixels.shape[-2:]):
                raise ValueError("invalid native Fast-WAM image shape")
            # Same encode boundary as _encode_input_image_latents_tensor; retain B.
            first = self._phase("vae", lambda: model.vae.encode([(p * 2 - 1).unsqueeze(1) for p in pixels],
                device=model.device, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)))
            if isinstance(first, (tuple, list)):
                first = torch.stack(list(first))
            if first.shape[0] != len(prepared):
                raise ValueError("VAE dropped the batch dimension")
            # Native pre_dit casts to the patch convolution's dtype first. The
            # key depends only on VAE + this convolution, never T5/proprio.
            patch_dtype = model.video_expert.patch_embedding.weight.dtype
            first = first.to(dtype=patch_dtype)
            patches = self._phase("patchify", lambda: model.video_expert.patchify(first))
            f, h, w = patches.shape[2:]
            tokens = patches.flatten(2).transpose(1, 2).contiguous()
            grids = torch.tensor([[f,h,w]] * len(prepared), device=tokens.device)
            outputs = []
            for i, p in enumerate(prepared):
                video_pre = {"tokens": tokens[i:i+1].clone(), "meta": {
                    "grid_sizes": grids[i:i+1].clone(), "batch_size": 1, "tokens_per_frame": h*w}}
                fields = self._phase("key_pool", lambda: self._key_builder.fields_from_tokens(video_pre, p.state))
                outputs.append(Stage1({"video_pre": video_pre, "first_latents": first[i:i+1].clone(),
                    "patches": patches[i:i+1].clone(), "prompt": p.data["prompt"],
                    "proprio": p.data["proprio"].clone(), "video_seq_len": f*h*w,
                    # Request-owned CPU inputs; the scope digest is computed lazily.
                    "conditioning_source": (p.data["prompt"], p.data["input_image"], p.data["proprio"]),
                    "model_identity": self.metadata().identity}, fields, p.state.copy()))
            return outputs

    def retrieval_fields(self, stage1):
        return {k: v.copy() for k,v in stage1.fields.items()}

    def run_stage2(self, stage1, *, capture=False, capture_cp2=False):
        return self.run_stage2_batch([stage1], capture=capture, capture_cp2=capture_cp2)[0]

    def run_stage2_batch(self, stages, *, capture=False, capture_cp2=False):
        import torch
        if not stages:
            return []
        self._check_batch(len(stages))
        with torch.inference_mode():
            first = stages[0].data
            for s in stages:
                if s.data.get("model_identity") != self.metadata().identity:
                    raise ValueError("stage1 belongs to a different resident model")
                if s.data["video_seq_len"] != first["video_seq_len"]:
                    raise ValueError("stage2 native geometry mismatch")
            model = self.model
            proprio = torch.cat([s.data["proprio"] for s in stages]).to(model.device, model.torch_dtype)
            context, mask = self._phase("text_context", lambda: model._prepare_infer_context(
                [s.data["prompt"] for s in stages], None, None, proprio))
            latents = torch.cat([s.data["first_latents"] for s in stages])
            patches = torch.cat([s.data["patches"] for s in stages])
            proxy = _ReusePatch(model.video_expert, patches)
            pre = self._phase("pre_dit_conditioning", lambda: model.video_expert.pre_dit.__func__(proxy,
                x=latents, timestep=torch.zeros(len(stages), device=model.device, dtype=latents.dtype),
                context=context, context_mask=mask, action=None,
                fuse_vae_embedding_in_latents=bool(model.video_expert.fuse_vae_embedding_in_latents)))
            attn = model._build_mot_attention_mask(video_seq_len=first["video_seq_len"],
                action_seq_len=self.metadata().layout.horizon,
                video_tokens_per_frame=int(pre["meta"]["tokens_per_frame"]), device=latents.device)
            kv = self._phase("video_prefill", lambda: model.mot.prefill_video_cache(
                video_tokens=pre["tokens"], video_freqs=pre["freqs"], video_t_mod=pre["t_mod"],
                video_context_payload={"context": pre["context"], "mask": pre["context_mask"]},
                video_attention_mask=attn[:first["video_seq_len"], :first["video_seq_len"]]))
            outputs = []
            for i, s in enumerate(stages):
                caches = [{name: layer[name][i:i+1].clone() for name in ("k", "v")} for layer in kv]
                if not caches or any(c["k"].shape[0] != 1 or c["v"].shape[0] != 1 for c in caches):
                    raise ValueError("prefill dropped batch rows")
                data = {k: s.data[k] for k in ("video_seq_len", "conditioning_source", "model_identity")}
                if s.data.get("conditioning_scope") is not None:
                    data["conditioning_scope"] = s.data["conditioning_scope"]
                data.update(context=context[i:i+1].clone(), context_mask=mask[i:i+1].clone(), attention_mask=attn.clone())
                data["video_kv_cache"] = caches
                data["capture_cp2_enabled"] = bool(capture_cp2)
                if capture_cp2:
                    # CP2: each layer's K and V averaged over sequence, flattened
                    # head/width preserved. Never pretend it is an independent camera.
                    data["cp2_fields"] = {"vlm_out": torch.cat([
                        c[name].float().mean(dim=1).reshape(-1) for c in caches for name in ("k", "v")
                    ]).detach().cpu().numpy().copy()}
                outputs.append(Stage2(data, s))
            return outputs

    def sample(self, stage2, sampler, rng, capture=CaptureSpec()):
        import torch
        self._validate_sampler(sampler)
        noise = torch.randn((1, self.metadata().layout.horizon, self.metadata().layout.action_width),
            generator=rng.torch_generator(device="cpu"), device="cpu", dtype=torch.float32)
        noise = noise.to(self.model.device, self.model.torch_dtype)
        return self._sample_batch([stage2], sampler, noise, 0, [capture], [rng.receipt()])[0]

    def resume(self, stage2, snapshot, sampler, rng, capture=CaptureSpec()):
        import torch
        self._validate_sampler(sampler)
        snapshot.validate(self.metadata(), sampler, conditioning_scope=stage_conditioning_scope(stage2.data))
        if str(snapshot.x_t.dtype) != str(self.model.torch_dtype) or not torch.isfinite(snapshot.x_t).all():
            raise ValueError("native snapshot dtype/finiteness mismatch")
        x = snapshot.x_t.clone().unsqueeze(0).to(self.model.device)
        return self._sample_batch([stage2], sampler, x, snapshot.loop_index,
            [capture], [rng.receipt()])[0]

    def reset_snapshot(self, stage2, clean, noise, sampler, loop_index):
        """Native add_noise at an existing saved grid point; no time guessing."""
        import torch
        self._validate_sampler(sampler)
        if type(loop_index) is not int or not 0 < loop_index < sampler.num_steps:
            raise ValueError("warm reset requires a recoverable native index")
        shape = (self.metadata().layout.horizon, self.metadata().layout.action_width)
        if tuple(clean.shape) != shape or tuple(noise.shape) != shape:
            raise ValueError("warm reset action/noise shape mismatch")
        clean, noise = (v.to(self.model.device, self.model.torch_dtype) for v in (clean, noise))
        if not torch.isfinite(clean).all() or not torch.isfinite(noise).all():
            raise ValueError("warm reset requires finite native tensors")
        t = torch.tensor(sampler.native_timesteps[loop_index], device=self.model.device, dtype=self.model.torch_dtype)
        x = self.model.infer_action_scheduler.add_noise(clean, noise, t)
        return self._snapshot(stage2, sampler, loop_index, x)

    def _snapshot(self, stage2, sampler, index, x):
        s = self.metadata()
        return NativeSnapshot(s.checkpoint_identity, s.norm_identity, s.layout.identity,
            sampler.identity, sampler.schedule_digest, index, sampler.native_timesteps[index],
            x.detach().cpu().clone(), {"algorithm": "deterministic_flow; no posterior draws"},
            stage_conditioning_scope(stage2.data))

    def _sample_batch(self, stages, sampler, latents, start_index, captures, receipts):
        import torch
        self._validate_sampler(sampler)
        self._check_batch(len(stages))
        if not stages or any(s.data.get("model_identity") != self.metadata().identity for s in stages):
            raise ValueError("stage2 belongs to a different resident model")
        if not 0 <= start_index < sampler.num_steps:
            raise ValueError("invalid native flow start index")
        first = stages[0].data
        contexts = torch.cat([s.data["context"] for s in stages])
        masks = torch.cat([s.data["context_mask"] for s in stages])
        kv = [{name: torch.cat([s.data["video_kv_cache"][layer][name] for s in stages])
               for name in ("k", "v")} for layer in range(len(first["video_kv_cache"]))]
        snapshots, traces = [[] for _ in stages], [[] for _ in stages]
        times, deltas = self.model.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=sampler.num_steps, device=self.model.device, dtype=latents.dtype)
        with torch.inference_mode():
            for index in range(start_index, sampler.num_steps):
                before = latents
                for j, (s, c) in enumerate(zip(stages, captures, strict=True)):
                    if c.enabled and c.snapshots and index > 0:
                        snapshots[j].append(self._snapshot(s, sampler, index, before[j]))
                pred = self._phase("action_head", lambda: self.model._predict_action_noise_with_cache(latents_action=latents,
                    timestep_action=times[index].expand(len(stages)), context=contexts, context_mask=masks,
                    video_kv_cache=kv, attention_mask=first["attention_mask"], video_seq_len=first["video_seq_len"]))
                latents = self.model.infer_action_scheduler.step(pred, deltas[index], latents)
                for j, c in enumerate(captures):
                    if c.enabled:
                        traces[j].append({"loop_index": index, "native_t": sampler.native_timesteps[index],
                            "native_delta": sampler.native_deltas[index], "x_t": _cpu_array(before[j]),
                            "prediction": _cpu_array(pred[j]), "x_next": _cpu_array(latents[j])})
            outputs = []
            for j, c in enumerate(captures):
                outputs.append(SampleResult(latents[j].detach().float().cpu().clone(),
                    sampler.num_steps-start_index, sampler, tuple(snapshots[j]),
                    {"native_steps": traces[j], "native_dtype": str(self.model.torch_dtype),
                     "start_loop_index": start_index, "capture_enabled": True,
                     "pricing_eligible": False} if c.enabled else
                    {"capture_enabled": False, "pricing_eligible": not first.get("capture_cp2_enabled", False)}, dict(receipts[j])))
            return outputs

    def make_stage_batcher(self):
        from .batcher import FastWAMStageBatcher
        return FastWAMStageBatcher(self)

    def make_trace_adapter(self):
        from .trace import FastWAMTraceAdapter
        return FastWAMTraceAdapter(self)

    def cpu_blueprint(self):
        return CPUModelBlueprint(self.metadata(), {"schema": "r17.fastwam.blueprint.v1",
            "manifest": asdict(self.manifest), "config": self.config,
            "benchmark": self.wire.benchmark, "stats": {
                k: {name: v.tolist() for name,v in bounds.items()} for k,bounds in self.wire.stats.items()},
            "execution_protocol": asdict(self.wire.protocol),
            "split_status": "unsupported; no transport/GPU qualification"})
