"""Bounded mechanical framework probe. Real model mode is explicit, never default.

No environment, training, worker, process supervisor or formal quality run.
The full helper is staged; upstream equivalence remains the model owner's G2
certificate. The synthetic donor here only qualifies framework dispatch.
"""
from __future__ import annotations

from dataclasses import asdict
import time


def config_for_probe(provider, k, commit):
    from exp.r17_integration.common.config import (BackendConfig, CacheConfig, CheckpointConfig,
                                    JudgeConfig, KeyBuilderConfig, KeysConfig,
                                    SearchStrategyConfig, WritePolicyConfig)
    from .provider import binding_for
    keys = KeysConfig()
    dims = dict(provider.model_spec.feature_spec.fields)
    for name, value in vars(keys).items():
        value.enabled = name in dims
    return CacheConfig(enabled=True, keys=keys, key_builder=KeyBuilderConfig(type="adapter_native"),
                        backend=BackendConfig(type="in_memory", vector_dims=dims),
                        write_policy=WritePolicyConfig(type="never"),
                        checkpoints={"cp1": CheckpointConfig(judge=JudgeConfig(type="always_hit"),
                                      search_strategy=SearchStrategyConfig(type="weighted_rrf_knn", top_k=1))},
                        adapter_binding=binding_for(provider, provider.model_spec.allowed_benchmark_protocols[0],
                                                    num_steps=k, commit_horizon=commit))


def run_probe(provider, observations, *, steps=None, deadline_seconds=840):
    import numpy as np
    import torch
    from exp.r17_integration.common.config import build_shared_storage
    from openpi.cache.cache_storage import CacheStorage
    from openpi.cache.backends.in_memory_backend import InMemoryBackend
    from openpi.cache.storage_types import CacheEntry, CachePayload
    from openpi.cache.types import CheckpointID
    from .contracts import RequestContext
    from .provider import artifact_binding
    from .serving import _array, _wire_array, full_call, make_connection_factory

    begin = time.monotonic()
    rows = []
    layout = provider.model_spec.layout
    commit = max(layout.replan_steps, min(10, layout.horizon) // layout.replan_steps * layout.replan_steps)
    steps = steps or ((100, 10, 4, 2) if provider.model_spec.default_sampler.family in {"ddpm", "ddim"}
                      else (provider.model_spec.default_sampler.num_steps, 2, 1))
    def check_deadline():
        if time.monotonic() - begin >= deadline_seconds:
            raise TimeoutError("qualification deadline reached; incomplete gates remain unqualified")
    for k in dict.fromkeys(steps):
        check_deadline()
        cfg = config_for_probe(provider, k, commit)
        storage = build_shared_storage(cfg)
        captures = []
        plain = make_connection_factory(provider, cfg, storage)()
        debug = make_connection_factory(provider, cfg, storage, capture_sink=captures.append)()
        reference = provider.load_adapter()
        plain.on_task_begin(); debug.on_task_begin()
        initial = RequestContext("p1_probe", cfg.adapter_binding["benchmark_protocol"], "probe", 0, 0, explicit_seed=7)
        episode = dict(asdict(initial), ack_protocol="r17.execution.v1")
        plain.on_episode_start(episode_id=0, extra_metadata=episode)
        debug.on_episode_start(episode_id=0, extra_metadata=episode)
        try:
            for i, obs in enumerate(observations):
                check_deadline()
                request = RequestContext("p1_probe", cfg.adapter_binding["benchmark_protocol"], "probe", 0, i, explicit_seed=7)
                normalized, expected_wire = full_call(reference, obs, request, sampler=provider.sampler_for_steps(k))
                sent = dict(obs, __request__=asdict(request))
                off = plain.infer(sent); on = debug.infer(sent)
                expected = _wire_array(expected_wire.actions)[:commit]
                if off["actions"].tobytes() != expected.tobytes() or on["actions"].tobytes() != expected.tobytes():
                    raise AssertionError("staged helper / stock MISS / capture bytes mismatch")
                if off["__hit_meta__"]["stage_calls"] != (1, 1, k):
                    raise AssertionError("actual native MISS NFE differs from bound K")
                rows.append({"k": k, "observation": i, "stage_calls": (1,1,k), "debug_bytes_equal": True})
                # Fixed observations request fresh plans; ack the actual
                # executed prefix without assuming the remainder ran.
                for policy in (plain, debug):
                    policy.acknowledge_execution({"decision_seq": i, "executed_count": layout.replan_steps, "terminal": True})
            # Synthetic normalized donor: no claim about retrieval quality.
            backend = InMemoryBackend(dict(cfg.backend.vector_dims))
            backend.insert(CacheEntry("p1_synthetic_donor", CheckpointID.CP1,
                {name: torch.zeros(dim) for name, dim in cfg.backend.vector_dims.items()},
                CachePayload(torch.from_numpy(_array(normalized.normalized_chunk)), task_key="probe")))
            backend.artifact_meta = {"r17_binding": artifact_binding(cfg), "synthetic": True}
            backend.freeze()
            hit = make_connection_factory(provider, cfg, CacheStorage(backend))()
            hit.on_task_begin()
            hit.on_episode_start(episode_id=1, extra_metadata=episode)
            try:
                first = hit.infer(observations[-1])
                if first["__hit_meta__"]["stage_calls"] != (1,0,0):
                    raise AssertionError("FULL_HIT did not skip stage2/sampler")
                tail_status = "not_applicable: commit equals replan/horizon"
                if commit > layout.replan_steps:
                    tail = hit.infer(dict(observations[-1], __execution__={"decision_seq":0, "executed_count":layout.replan_steps}))
                    if tail["__hit_meta__"]["stage_calls"] != (0,0,0) or tail["actions"].tobytes() != first["actions"][layout.replan_steps:].tobytes():
                        raise AssertionError("tail cursor/stage/byte mismatch")
                    tail_status = "passed"
                rows.append({"k": k, "synthetic_full_hit": (1,0,0), "tail": tail_status})
            finally:
                hit.on_task_end()
        finally:
            plain.on_task_end(); debug.on_task_end()
    check_deadline()
    return {"schema":"r17.framework_qualification.v1", "status":"passed", "model":asdict(provider.model_spec),
            "rows":rows, "elapsed_seconds":time.monotonic()-begin,
            "limits":["staged helper reference; upstream parity requires model G2 receipt", "synthetic donor; no quality SR",
                      "direct factories only; CPU WebSocket/proxy receipts are separate", "B2 and real benchmark remain coordinator slots"]}


def main(argv=None):
    import argparse
    import importlib
    import json
    from pathlib import Path
    import numpy as np
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fake", action="store_true")
    parser.add_argument("--adapter-module", default="exp.r17_integration.models.pi05.adapter")
    parser.add_argument("--provider-module", default="exp.r17_integration.models.pi05.provider")
    parser.add_argument("--spec-json", type=Path)
    parser.add_argument("--observations", type=Path, help="no-pickle NPZ, arrays with leading observation axis")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--deadline-seconds", type=float, default=840)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("receipt output already exists; use a new owned run directory")
    if args.fake:
        from .fake_adapter import fake_provider
        provider = fake_provider("p1.qualification.fake")
        observations = [{"state":np.arange(4,dtype=np.float32)} for _ in range(2)]
    else:
        if args.spec_json is None or args.observations is None:
            parser.error("real mode requires explicit immutable model descriptor and fixed NPZ observations")
        adapter = importlib.import_module(args.adapter_module).load_adapter(json.loads(args.spec_json.read_text()))
        provider = importlib.import_module(args.provider_module).register_provider(adapter)
        with np.load(args.observations,allow_pickle=False) as data:
            count = min(8,len(data[data.files[0]]))
            observations = [{name:data[name][i].item() if data[name][i].ndim == 0 else data[name][i].copy()
                             for name in data.files} for i in range(count)]
    if not observations:
        parser.error("qualification requires at least one observation")
    try:
        receipt = run_probe(provider, observations, deadline_seconds=args.deadline_seconds)
    except Exception as exc:
        receipt = {"schema":"r17.framework_qualification.v1", "status":"failed", "error":str(exc)}
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(receipt,indent=2)+"\n")
        raise
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(receipt,indent=2)+"\n")


if __name__ == "__main__":
    main()
