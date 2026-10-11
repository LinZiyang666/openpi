"""Capture only existing tensors; never evaluate a model or consume randomness."""
from dataclasses import asdict

import numpy as np


def _array(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.ascontiguousarray(x).copy()


class DPTraceAdapter:
    def __init__(self, adapter):
        self.adapter = adapter

    def sample_fields(self, s1, s2, result):
        """P1's default-off post-MISS hook; all values were already computed."""
        fields = {"conditioning": _array(s2.data["conditioning"]),
                  "native_grid": np.asarray(result.sampler.native_timesteps, np.int64)}
        fields.update(policy_wrist_feature=_array(s1.data["features"]["left_wrist_0_rgb"]),
                      normalized_state=_array(s1.data["features"]["state"]),
                      language_embedding=_array(s1.data["features"]["lang_emb"]))
        for name, value in result.capture.items():
            if value is not None:
                fields[name] = _array(value)
        return fields

    def sample_variants(self, s2, samplers, rng, capture):
        """Explicit offline/twin hook; caller accounts for every extra NFE.

        Each variant starts from the same private initial stream. It continues
        its own DDPM/DDIM stream on its own native grid; no flow interpolation.
        Ordinary debug capture never calls this method.
        """
        from .request_noise import RequestNoise
        outputs = {}
        for sampler in samplers:
            private = rng.fresh(self.adapter.device) if isinstance(rng, RequestNoise) else rng
            result = self.adapter.sample(s2, sampler, private, capture)
            if result.sampler.identity in outputs:
                raise ValueError("duplicate DP trace variant sampler")
            outputs[result.sampler.identity] = result
        return outputs

    def capture(self, decision):
        """P1 ModelTraceAdapter interface, including HIT/tail status."""
        arrays = {f"key_{k}": _array(v) for k, v in decision.fields.items()}
        arrays.update(normalized_actions=_array(decision.normalized_chunk),
                      wire_actions=_array(decision.wire_actions),
                      native_grid=np.asarray(decision.sampler_receipt.get("native_timesteps", ()), dtype=np.int64))
        if "robot_state" in decision.fields:
            arrays["state"] = _array(decision.fields["robot_state"])
        snapshots = []
        native_snapshots = getattr(decision, "native_snapshots", ())
        if native_snapshots:
            arrays.update(self._snapshot_arrays(native_snapshots))
        for snap in native_snapshots:
            snapshots.append({k: v for k, v in asdict(snap).items() if k not in ("x_t", "noise_state")})
        for key, value in getattr(decision, "model_capture", {}).items():
            if value is not None:
                arrays[key] = _array(value)
        return {"metadata": {"schema": "r17.dp.capture.v1", "model_spec": asdict(self.adapter.metadata()),
                             "request": asdict(decision.request), "source": decision.source,
                             "cursor": decision.cursor, "executed_count": decision.executed_count,
                             "stage_calls": decision.stage_calls, "actual_nfe": decision.actual_nfe,
                             "sampler_receipt": dict(decision.sampler_receipt),
                             "snapshots": snapshots,
                             "rng_receipt": decision.diagnostics.get("rng_receipt", {}),
                             "capability_status": "available" if snapshots else "no native snapshot on HIT/tail or K1"},
                "arrays": arrays}

    def from_stages(self, prepared, s1, result, wire):
        arrays = {f"key_{k}": _array(v) for k, v in self.adapter.retrieval_fields(s1).items()}
        arrays.update(state=_array(prepared.state), normalized_actions=_array(result.normalized_chunk),
                      wire_actions=_array(wire.actions),
                      native_grid=np.asarray(result.sampler.native_timesteps, dtype=np.int64),
                      policy_wrist_feature=_array(s1.data["features"]["left_wrist_0_rgb"]),
                      normalized_state=_array(s1.data["features"]["state"]),
                      language_embedding=_array(s1.data["features"]["lang_emb"]))
        snapshot_meta = []
        if result.snapshots:
            arrays.update(self._snapshot_arrays(result.snapshots))
        for snapshot in result.snapshots:
            snapshot_meta.append({k: v for k, v in asdict(snapshot).items() if k not in ("x_t", "noise_state")})
        for key, value in result.capture.items():
            if value is not None:
                arrays[key] = _array(value)
        return {"metadata": {"schema": "r17.dp.capture.v1", "model_spec": asdict(self.adapter.metadata()),
                             "request": asdict(prepared.request), "prompt": prepared.data["prompt"],
                             "sampler": asdict(result.sampler), "rng_receipt": dict(result.rng_receipt),
                             "actual_nfe": result.actual_nfe, "snapshots": snapshot_meta},
                "arrays": arrays}

    @staticmethod
    def _snapshot_arrays(snapshots):
        return {"snapshot_x": np.stack([_array(s.x_t) for s in snapshots]),
                "snapshot_loop_index": np.asarray([s.loop_index for s in snapshots], np.int64),
                "snapshot_native_timestep": np.asarray([s.native_timestep for s in snapshots], np.int64),
                "snapshot_rng_state": np.stack([_array(s.noise_state) for s in snapshots])}

    @staticmethod
    def save(capture, output):
        """Portable P6 fixture: strict JSON + numeric NPZ, no pickle."""
        import json
        from pathlib import Path
        root = Path(output)
        root.mkdir(parents=True, exist_ok=True)
        (root / "capture.json").write_text(json.dumps(capture["metadata"], indent=2, allow_nan=False) + "\n")
        np.savez(root / "capture.npz", **capture["arrays"])
