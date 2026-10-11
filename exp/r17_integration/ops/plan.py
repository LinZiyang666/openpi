"""Qualification plans only; no model loading, process launch or CUDA calls."""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path


def cpu_blueprint(value):
    def walk(item):
        if hasattr(item, "device"):
            device = str(item.device)
            if device not in ("cpu", "None"):
                raise ValueError("blueprint contains non-CPU tensor")
            # Even CPU weights do not belong in a config-only split blueprint.
            raise ValueError("blueprint contains tensor/array instead of configuration")
        if dataclasses.is_dataclass(item):
            return walk(dataclasses.asdict(item))
        if isinstance(item, dict):
            return {str(k): walk(v) for k, v in item.items()}
        if isinstance(item, (list, tuple)):
            return [walk(v) for v in item]
        if item is None or isinstance(item, (str, int, float, bool)):
            return item
        raise ValueError("blueprint contains non-JSON object")
    result = walk(value)
    json.dumps(result, allow_nan=False)
    return result


def unix_socket_path(root, run_digest):
    path = Path(root) / ("r17_" + run_digest[:12] + ".sock")
    if len(str(path).encode()) >= 104:
        raise ValueError("Unix socket path exceeds portable 103-byte limit")
    return str(path)


def qualification_plan(model_id, checkpoint_digest):
    if not model_id or not checkpoint_digest:
        raise ValueError("immutable checkpoint identity required")
    return dict(schema="r17.qualification.v1", model=model_id, checkpoint=checkpoint_digest,
                deadline_seconds=900, executed=False, gpu_concurrency=1,
                phases=[dict(until=180, action="load qualified immutable model; 3 network warmups"),
                        dict(until=480, action="2 mechanical episodes full/retrieval; capture and cursor"),
                        dict(until=660, action="1-2 saved observations explicit shadow/native resume; extra cost"),
                        dict(until=780, action="closed H5->tiny artifact->new bundle generation->forced HIT"),
                        dict(until=900, action="normal owned shutdown; SHA snapshot/join/analysis receipt")],
                gates=["coordinator allocates exclusive GPU lock after training", "model G2 receipt passed",
                       "client Python3.11 payload attested", "reuse existing EGL lock in P2 worker"],
                excluded=["training", "formal500", "full pricing", "replica/batch/split certification"])
