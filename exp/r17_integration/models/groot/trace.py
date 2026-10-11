"""Capture existing decision tensors for P6; stock GR00T trace writer remains shared."""
from __future__ import annotations

from dataclasses import asdict
import numpy as np
from .manifest import library_metadata


def _array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.ascontiguousarray(value).copy()


class GrootMetaWorldTraceAdapter:
    def __init__(self, manifest):
        self.manifest = manifest

    def capture(self, decision):
        normalized = _array(decision.normalized_chunk)
        wire = _array(decision.wire_actions)
        if normalized.shape != (16, 32) or wire.ndim != 2 or wire.shape[1] != 4 or not 1 <= wire.shape[0] <= 16:
            raise ValueError("GR00T-MW capture must preserve normalized H16×32 and wire4")
        if not np.isfinite(normalized).all() or not np.isfinite(wire).all():
            raise ValueError("non-finite capture")
        fields = {name: _array(value) for name, value in decision.fields.items()}
        if "vision_1" in fields or "vision_2" in fields:
            raise ValueError("capture contains fictitious wrist")
        if "vision_0" in fields and fields["vision_0"].shape != (16 * self.manifest.hidden_width,):
            raise ValueError("capture vision width differs")
        if "robot_state" in fields and fields["robot_state"].shape != (4,):
            raise ValueError("capture state width differs")
        snapshots = []
        for snap in getattr(decision, "native_snapshots", ()):
            snap.validate(self.manifest.model_spec(), self.manifest.model_spec().default_sampler)
            snapshots.append({**{key: value for key, value in vars(snap).items() if key != "x_t"},
                              "x_t": _array(snap.x_t)})
        return {"schema": "r17.decision_capture.v1", "request": asdict(decision.request),
                "source": decision.source, "cursor": decision.cursor, "executed_count": decision.executed_count,
                "stage_calls": decision.stage_calls, "actual_nfe": decision.actual_nfe,
                "normalized_chunk": normalized, "wire_actions": wire, "fields": fields,
                "sampler_receipt": dict(decision.sampler_receipt), "diagnostics": dict(decision.diagnostics),
                "library_metadata": library_metadata(self.manifest), "native_sampler": self.manifest.sampler_spec,
                "native_snapshots": snapshots}


def make_trace_adapter(manifest):
    return GrootMetaWorldTraceAdapter(manifest)


def make_stock_trace_runtime(trace_config, **kwargs):
    from dataclasses import replace
    from openpi.cache.trace.runtime import build_groot_trace_runtime
    if trace_config is not None:
        trace_config = replace(trace_config, raw_image_keys=["video.image"])
    return build_groot_trace_runtime(trace_config, **kwargs)
