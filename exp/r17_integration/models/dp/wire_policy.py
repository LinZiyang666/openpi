"""Plain policy for shared WS metadata; cache execution belongs to P1."""
from dataclasses import asdict
from exp.r17_integration.common.contracts import CaptureSpec, RequestContext

from .request_noise import RequestNoise


class WirePolicy:
    def __init__(self, adapter):
        self.adapter = adapter

    def get_metadata(self):
        spec = self.adapter.metadata()
        batch = self.adapter.batch_config
        from .sampler import SamplerProfile
        samplers = {str(k): asdict(SamplerProfile("auto" if spec.default_sampler.family == "ddpm" else "ddim",
                       k, spec.default_sampler.config.get("eta", 0.0)).to_spec(self.adapter.model.cfg))
                    for k in sorted({1, 2, 4, 10, 100, spec.default_sampler.num_steps} | ({5} if batch else set()))}
        return {"r17_contract": "r17.contracts.v1", "model_id": spec.model_id,
                "model_identity": spec.checkpoint_identity, "norm_identity": spec.norm_identity,
                "layout_identity": spec.layout.identity, "feature_identity": spec.feature_spec.identity,
                "sampler_identity": spec.default_sampler.identity, "horizon": spec.layout.horizon,
                "action_dim": spec.layout.valid_action_dim, "diagnostic": self.adapter.diagnostic,
                "r17_model_spec": asdict(spec),
                "r17_batch_policy": {"production_max_batch_size": batch.max_batch if batch else 1,
                    "status": "enabled_matched_study" if batch else "unsupported",
                    "reason": "Owner permits measured numerical drift with matched control/arms" if batch else "GPU batch composition changes outputs",
                    "diagnostic_label": self.adapter.batch_diagnostic_label},
                **({"r17_batching": batch.metadata(), "batch_runtime_capabilities": {f"batch_stage{i}": "enabled_matched_study" for i in (1, 2, 3)}} if batch else {}),
                "r17_sampler_specs": samplers,
                "r17_sampler_admission": {"schema": "r17.dp.sampler_admission.v1",
                    "default_steps": 100, "cache_miss_min_steps": 2, "teacher_min_steps": 2,
                    "recipe_min_steps": 2, "descent_steps": [2, 4, 10, 100],
                    "K1": {"quality_status": "lossy", "allowed_role": "plain_step_reduction_study",
                           "required_binding": {"allow_lossy_sampler": True,
                                                "sampler_study_label": "nonempty study label"}}},
                "benchmark_protocols": list(spec.allowed_benchmark_protocols),
                "libero_wire_image_sizes": [96, 256] if self.adapter.domain == "libero" else None,
                "libero_wire_protocols": dict(self.adapter.libero_wire_protocols),
                "reduced_sampler": "trailing DDIM; K1 lossy, benchmark quality unqualified",
                "reconstruction": "E checkpoint-compatible reconstructed wrapper"}

    def infer(self, obs):
        # Legacy E conformance uses the explicitly supplied seed, default 0.
        request = RequestContext("diagnostic", "diagnostic", str(obs.get("prompt", "")), 0, 0)
        prepared = self.adapter.prepare({k: v for k, v in obs.items() if not k.startswith("__")}, request)
        s1 = self.adapter.run_stage1(prepared)
        s2 = self.adapter.run_stage2(s1)
        result = self.adapter.sample(s2, self.adapter.metadata().default_sampler,
                                     RequestNoise(int(obs.get("seed", 0)), self.adapter.device), CaptureSpec())
        return {"actions": self.adapter.to_wire(result.normalized_chunk, prepared).actions}

    def reset(self):
        pass
