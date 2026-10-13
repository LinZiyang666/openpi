"""Current-observation VAE -> native conditioning -> native EDM sampler.

The official skip_vae_encoding hook reuses this request's VAE output only.
It never carries an observation/latent/prediction across calls or connections.
"""
from __future__ import annotations

from dataclasses import asdict
import threading

import numpy as np

from exp.r17_integration.common.contracts import CPUModelBlueprint, CaptureSpec, SampleResult, Stage1, Stage2
from .key_builder import CosmosKeyBuilder
from .manifest import make_sampler


def make_data_batch(native, embedding, cfg, stats, model, cu, torch):
    # Same 19 fields and raw frame sequence as cosmos_utils.get_action B1.
    images = cu.prepare_images_for_model([native["wrist_image"], native["primary_image"]], cfg)
    blank = np.zeros_like(images[0])
    latent_frames = [blank, blank, images[0], images[1], blank, blank, images[0], images[1], blank]
    frames = [latent_frames[0][None]] + [np.repeat(frame[None], 4, axis=0) for frame in latent_frames[1:]]
    video = np.concatenate(frames).transpose(3, 0, 1, 2)[None]
    device = model.tensor_kwargs["device"]
    proprio = cu.rescale_proprio(native["proprio"], stats, non_negative_only=False, scale_multiplier=1.0)
    batch = {"dataset_name": "video_data", "video": torch.from_numpy(video).to(device=device, dtype=torch.uint8),
             "t5_text_embeddings": embedding.repeat(1, 1, 1).to(device=device, dtype=torch.bfloat16),
             "fps": torch.tensor([16], device=device, dtype=torch.bfloat16),
             "padding_mask": torch.zeros((1, 1, 224, 224), device=device, dtype=torch.bfloat16),
             "num_conditional_frames": model.config.min_num_conditional_frames,
             "proprio": torch.from_numpy(proprio).reshape(1, -1).to(device=device, dtype=torch.bfloat16)}
    indices = {"current_proprio": 1, "current_wrist_image": 2, "current_wrist_image2": -1,
               "current_image": 3, "current_image2": -1, "action": 4,
               "future_proprio": 5, "future_wrist_image": 6, "future_wrist_image2": -1,
               "future_image": 7, "future_image2": -1, "value": 8}
    batch.update({name + "_latent_idx": torch.tensor([value], device=device, dtype=torch.int64)
                  for name, value in indices.items()})
    return batch


class CosmosAdapter:
    @staticmethod
    def data_batch_builder(*args):
        return make_data_batch(*args)

    action_latent_idx = 4
    action_shape = (16, 7)
    teacher_seed = 195

    def __init__(self, model, manifest, cfg, stats, embeddings, cu, *, lock=None):
        self.model, self.manifest, self.cfg, self.stats = model, manifest, cfg, stats
        self.embeddings, self.cu = embeddings, cu
        self.lock = lock or threading.RLock()
        self.wire = self.make_wire_policy(stats, cu)
        self.keys = self.make_key_builder(manifest.spec.feature_spec)

    @staticmethod
    def make_wire_policy(stats, cu):
        from .wire_policy import CosmosWirePolicy
        return CosmosWirePolicy(stats, cu)

    @staticmethod
    def make_key_builder(feature_spec):
        return CosmosKeyBuilder(feature_spec)

    def metadata(self):
        return self.manifest.spec

    def prepare(self, obs, request):
        if request.benchmark_protocol not in self.metadata().allowed_benchmark_protocols:
            raise ValueError("Cosmos request is not bound to its benchmark protocol")
        prepared = self.wire.prepare(obs, request)
        if prepared.data["prompt"] not in self.embeddings:
            raise ValueError("prompt absent from the sealed official T5 cache; no encoder fallback")
        return prepared

    def extract_state_only(self, obs):
        return self.wire.extract_state_only(obs)

    def to_wire(self, normalized_chunk, prepared):
        return self.wire.to_wire(normalized_chunk, prepared)

    def sampler_for_steps(self, k):
        return make_sampler(k)

    def run_stage1(self, prepared):
        import torch
        with self.lock, torch.inference_mode():
            from cosmos_policy.utils.utils import set_seed_everywhere
            # run_libero_eval.py:396-400 resets process RNG0 at each control.
            # Our B1 resident resets it at each decision; the native initial
            # diffusion noise separately uses the explicit request seed195.
            set_seed_everywhere(0)
            batch = self.data_batch_builder(prepared.data["native"], self.embeddings[prepared.data["prompt"]],
                                    self.cfg, self.stats, self.model, self.cu, torch)
            self.model._normalize_video_databatch_inplace(batch)
            self.model._augment_image_dim_inplace(batch)
            latent = self.model.encode(batch["video"]).contiguous().float()
            fields = self.keys.fields_from_latent(latent, prepared.state)
            return Stage1({"batch": batch, "latent": latent, "model_identity": self.metadata().identity},
                          fields, prepared.state.copy())

    def retrieval_fields(self, stage1):
        return {name: value.copy() for name, value in stage1.fields.items()}

    def run_stage2(self, stage1, *, capture=False):
        if stage1.data["model_identity"] != self.metadata().identity:
            pass
        import torch
        with self.lock, torch.inference_mode():
            x0_fn = self.model.get_x0_fn_from_batch(stage1.data["batch"], 1.5, is_negative_prompt=False,
                skip_vae_encoding=True, previous_generated_latent=stage1.data["latent"])
        return Stage2({"x0_fn": x0_fn, "latent_shape": tuple(stage1.data["latent"].shape),
                       "model_identity": self.metadata().identity}, stage1)

    def sample(self, stage2, sampler, rng, capture=CaptureSpec()):
        if sampler != make_sampler(sampler.num_steps):
            raise ValueError("mismatched Cosmos native sampler")
        if capture.enabled and (capture.snapshots or capture.fields):
            raise ValueError("Cosmos capture supports only the minimal teacher profile; no snapshots or extra fields")
        if capture.enabled and sampler != self.metadata().default_sampler:
            raise ValueError("Cosmos teacher capture requires the default K5 sampler")
        if stage2.data["model_identity"] != self.metadata().identity:
            pass
        import torch
        from cosmos_policy._src.imaginaire.utils import misc
        device = self.model.tensor_kwargs["device"]
        seed = rng.seed_for()
        if capture.enabled and self.teacher_seed is not None and seed != self.teacher_seed:
            raise ValueError(f"Cosmos teacher capture requires seed{self.teacher_seed}")
        calls = 0
        def denoise(x, sigma):
            nonlocal calls
            calls += 1
            return stage2.data["x0_fn"](x, sigma)
        with self.lock, torch.inference_mode():
            noise = misc.arch_invariant_rand(stage2.data["latent_shape"], torch.float32, device, seed) * self.model.sde.sigma_max
            generated = self.model.sampler(denoise, noise, num_steps=sampler.num_steps,
                sigma_max=self.model.sde.sigma_max, sigma_min=self.model.sde.sigma_min, solver_option="2ab")
            indices = torch.tensor([self.action_latent_idx], device=device, dtype=torch.int64)
            actions = self.cu.extract_action_chunk_from_latent_sequence(generated, self.action_shape, indices).float().cpu()[0]
        if calls != sampler.config.get("expected_nfe", sampler.num_steps):
            raise RuntimeError("Cosmos actual denoiser NFE differs from attested sampler")
        return SampleResult(actions, calls, sampler, capture={"capture_enabled": capture.enabled,
                            "pricing_eligible": not capture.enabled},
                            rng_receipt={**rng.receipt(), "native_seed": seed, "native_rng": "arch_invariant_rand"})

    def resume(self, *args, **kwargs):
        raise ValueError("Cosmos native warm resume is unsupported")

    def make_stage_batcher(self):
        raise ValueError("Cosmos B>1 stage batching is unsupported; use B1 resident replicas")

    def make_trace_adapter(self):
        from .capture import CosmosTeacherCaptureAdapter
        return CosmosTeacherCaptureAdapter()

    def cpu_blueprint(self):
        return CPUModelBlueprint(self.metadata(), {"schema": "r17.cosmos.blueprint.v1",
            "manifest": asdict(self.manifest), "split_status": "unsupported", "current_observation_only": True})
