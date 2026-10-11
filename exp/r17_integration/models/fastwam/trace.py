"""Observe-only capture and explicit, costed native augmentation variants."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
import numpy as np

from exp.r17_integration.common.contracts import CaptureSpec


def numeric_copy(value):
    if hasattr(value, "detach"):
        return value.detach().float().cpu().numpy().copy()
    if isinstance(value, np.ndarray):
        return value.copy()
    if is_dataclass(value):
        return {k: numeric_copy(v) for k,v in asdict(value).items()}
    if isinstance(value, dict):
        return {k: numeric_copy(v) for k,v in value.items()}
    if isinstance(value, (tuple, list)):
        return [numeric_copy(v) for v in value]
    return value


class FastWAMTraceAdapter:
    def __init__(self, adapter):
        self.adapter = adapter

    def capture(self, decision):
        """Only copies already computed data. No forward or RNG entry point."""
        spec = self.adapter.metadata()
        if tuple(decision.normalized_chunk.shape) != (32, spec.layout.action_width):
            raise ValueError("Fast-WAM capture normalized shape mismatch")
        native = decision.model_capture.get("native_grid")
        if native is None:
            times = decision.sampler_receipt.get("native_timesteps", spec.default_sampler.native_timesteps)
            sampler = self.adapter.sampler_for_steps(len(times))
            if decision.sampler_receipt.get("identity", sampler.identity) != sampler.identity:
                raise ValueError("capture needs the actual bound native sampler receipt")
            native = asdict(sampler)
        return {"schema": "r17.fastwam.capture.v1", "model_spec": asdict(spec),
            "capture_enabled": True, "pricing_eligible": False,
            "decision": numeric_copy(decision), "native_grid": numeric_copy(native),
            "real_cameras": list(spec.feature_spec.real_cameras),
            "vision_semantics": "fused_real_views" if len(spec.feature_spec.real_cameras) > 1 else "one_real_view",
            "native_dtype": str(self.adapter.model.torch_dtype),
            "capability_statuses": {k: asdict(v) for k,v in spec.capabilities.features.items()}}

    def sample_fields(self, stage1, stage2, result):
        return {"model_spec": asdict(self.adapter.metadata()),
            "capture_enabled": True, "pricing_eligible": False,
            "capture_cp2_enabled": stage2.data.get("capture_cp2_enabled", False),
            "retrieval_fields": numeric_copy(stage1.fields), "robot_state": numeric_copy(stage1.state),
            "tokens": numeric_copy(stage1.data["video_pre"]["tokens"]),
            "token_grid": numeric_copy(stage1.data["video_pre"]["meta"]["grid_sizes"]),
            "cp2_fields": numeric_copy(stage2.data.get("cp2_fields", {})),
            "normalized_actions": numeric_copy(result.normalized_chunk),
            "native_grid": asdict(result.sampler), "native_snapshots": numeric_copy(result.snapshots),
            "native_steps": numeric_copy(result.capture.get("native_steps", [])),
            "rng_receipt": dict(result.rng_receipt), "actual_nfe": result.actual_nfe}

    def variants(self, stage2, rng, *, steps=(10, 2, 1), snapshot=None):
        """Explicit offline/shadow work with matched initial seed, billed separately."""
        results = {}
        for k in steps:
            sampler = self.adapter.sampler_for_steps(k)
            result = self.adapter.sample(stage2, sampler, rng, CaptureSpec(True, True))
            results[f"native_k{k}"] = {"result": result, "extra_stage_calls": (0, 0, 1), "extra_nfe": result.actual_nfe}
        if snapshot is not None:
            sampler = self.adapter.sampler_for_steps(len(self.adapter.metadata().default_sampler.native_timesteps))
            result = self.adapter.resume(stage2, snapshot, sampler, rng, CaptureSpec(True, True))
            results["native_resume"] = {"result": result, "extra_stage_calls": (0, 0, 1), "extra_nfe": result.actual_nfe}
        return results


def make_trace_adapter(adapter):
    return FastWAMTraceAdapter(adapter)
