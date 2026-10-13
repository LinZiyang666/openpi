"""DP formal resident: shared connection path + S07 batches, no resident lock."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import signal
import threading
import time


def cache_study_admission(campaign, cache, batch, provider):
    """Explicit owner-matched cache mode; ordinary startup admission is unchanged."""
    if not campaign.get('cache_study'):
        if cache.enabled or cache.checkpoints or cache.recipe.enabled:
            raise ValueError('DP formal batch baseline requires cache/checkpoints/recipe disabled')
        return False
    if (campaign.get('replicas') != 2 or batch.max_batch != 8 or batch.batch_wait_ms != 5 or
            not campaign.get('mps') or campaign.get('server_node') != 'local'):
        pass
    active = provider.sampler_for_steps(cache.adapter_binding['num_steps'])
    if active.family != 'ddpm' or active.num_steps != 100 or cache.checkpoints:
        raise ValueError('DP cache study stall MISS requires native DDPM100 and no CP checkpoints')
    if cache.recipe.enabled and cache.recipe.miss_mode not in ('never','stall_only'):
        raise ValueError('DP cache study permits only never or stall_only MISS mode')
    provider.config_validator(cache)
    return True


def serve(campaign, port, directory, cuda_graph=True):
    if campaign["model"] != "dp" or campaign["domain"] not in ("libero", "metaworld"):
        raise ValueError("DP batched formal server requires LIBERO or MetaWorld")
    from .batch_runtime import BatchConfig, create_runtime
    from .provider import make_connection_factory
    from .wire_policy import WirePolicy
    from exp.r17_integration.qualification.model_runtime import load
    from exp.r17_integration.common.config import load_cache_config, build_shared_storage
    from exp.r17_integration.common.serving import make_bundle_validator
    from exp.r17_integration.common.websocket_policy_server import WebsocketPolicyServer
    from exp.r17_integration.ops.formal_process import write_json
    config_doc = campaign.get("dp_batching")
    if not config_doc:
        pass
    config = BatchConfig(config_doc["max_batch"], config_doc["batch_wait_ms"], config_doc["numerics_policy"])
    if config.metadata() != config_doc:
        pass
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if not (directory.parent / "manifest.json").exists():
        write_json(directory.parent / "manifest.json", campaign)
    started = time.monotonic()
    adapter, provider = load(dict(campaign, benchmark_spec=str(Path(campaign["assets"]) / "benchmark.json")), campaign["device"])
    cache = load_cache_config(Path(campaign["assets"]) / "full.yaml")
    study = cache_study_admission(campaign, cache, config, provider)
    active = provider.sampler_for_steps(cache.adapter_binding["num_steps"])
    if active.num_steps not in (100, 10, 5) or active.config.get("eta", 0) != 0:
        raise ValueError("DP batch arms are DDPM100 or trailing eta0 DDIM10/5")
    import torch
    total = torch.cuda.get_device_properties(campaign["device"]).total_memory
    torch.cuda.set_per_process_memory_fraction(campaign["memory_budget_gib"] * 2**30 / campaign["replicas"] / total, campaign["device"])
    graph_report = None
    if cuda_graph:
        from .batch_graph import check_graph, warm_batch_signatures
        from .probe import fixed_observation
        observations_path = Path(campaign.get("graph_probe_observations",
            str(Path(campaign["assets"]) / "wire_observations.npz")))
        if not observations_path.is_file() and campaign['domain'] == 'libero':
            observations_path = Path("/data/mw_finetune/qualification/assets/dp_libero_gpu_fix1_v1/wire_observations.npz")
        if observations_path.is_file():
            observation, _ = fixed_observation(observations_path, campaign['domain'])
        elif campaign['domain'] == 'libero':
            from .fixtures import libero_observation
            observation, _ = libero_observation()
            benchmark = next(iter(adapter.libero_wire_protocols.values()))
            if observation["observation/image"].shape[0] != benchmark:
                pass
        else:
            from exp.mw_finetune.dp.cuda_graph import check_observations
            observation = next(iter(check_observations('metaworld', adapter.legacy_policy.stats)))
        graph_report = check_graph(adapter, [observation], steps=(active.num_steps,))
        graph = graph_report.pop("graph")
        from exp.r17_integration.common.contracts import RequestContext
        from .sampler import SamplerProfile
        context = RequestContext("dp_graph_prewarm", adapter.metadata().allowed_benchmark_protocols[0], "fixed_real_frame", 0, 0, explicit_seed=18)
        s2 = adapter.run_stage2(adapter.run_stage1(adapter.prepare(observation, context)))
        timestep = SamplerProfile(num_steps=active.num_steps).scheduler(adapter.model.cfg).timesteps[0]
        graph_report["batch_prewarm"] = warm_batch_signatures(adapter, graph, s2, config.max_batch, timestep)
    runtime = create_runtime(adapter, config.max_batch, config.batch_wait_ms, directory / "batching.jsonl", force=True)
    captures = None
    try:
        capture_kwargs = {}
        if campaign.get('library'):
            from exp.r17_integration.debug.library_capture import campaign_capture_factory
            from exp.r17_integration.common.contracts import CaptureSpec
            captures = campaign_capture_factory(campaign, directory / 'capture')
            capture_kwargs = dict(capture_sink_factory=captures,
                                  capture_spec=CaptureSpec(enabled=True, snapshots=False))
        base = make_connection_factory(adapter, provider=provider, default_config=cache,
            default_storage=build_shared_storage(cache), stage_core=runtime.core, **capture_kwargs)
        base = runtime.wrap_connection_factory(base)
        evidence_lock = threading.Lock()
        def factory(*args, **kwargs):
            connection = base(*args, **kwargs)
            original = connection.infer
            def infer(obs):
                begin = time.monotonic()
                if captures is not None:
                    connection.capture_sink.bind_observation(obs)
                result = original(obs)
                hit = result["__hit_meta__"]
                row = dict(replica_port=port, task_uid=connection._episode.get("task_uid"),
                    task_name=connection._episode.get("task_name"), decision_seq=hit["decision_seq"],
                    source=hit["source"], nfe=hit["actual_nfe"], elapsed_seconds=time.monotonic()-begin,
                    batching=hit["batching"], time=time.time(),
                    timing_scope="end_to_end_including_stage_queue_wait")
                with evidence_lock, (directory / "serving.jsonl").open("a") as stream:
                    stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                return result
            connection.infer = infer
            return connection
        metadata = WirePolicy(adapter).get_metadata()
        metadata.update(formal_pure_inference=True, cache_enabled=False, debug_capture=False,
            cuda_graph=cuda_graph, cuda_graph_self_check=graph_report,
            resident_execution="S07 dynamic native batches; one worker per stage; no resident inference lock",
            execution_sampler=asdict(active), execution_commit_horizon=cache.adapter_binding["commit_horizon"],
            replica_port=port)
        if study:
            metadata.update(cache_study_admitted=True,
                cache_study_startup='plain full.yaml; model stays resident across validated YAML switches',
                cache_study_groups=['c','cr','ch','chr','cg','crg','chg','chrg'],
                cache_study_miss_modes=['never','stall_only'],
                cache_study_teacher_compatibility='r17.teacher_compatibility.v1; exact producer/asset proof',
                cache_study_native_stall_nfe=100)
        if captures is not None:
            metadata.update(formal_pure_inference=False, library_teacher=True, debug_capture=True,
                            capture_profile='B-full-native-library-allowlist; received_wire_uint8')
        server = WebsocketPolicyServer(factory(), host="0.0.0.0", port=port, metadata=metadata, concurrent=True,
            connection_policy_factory=factory, bundle_validator=make_bundle_validator(provider),
            ready_callback=lambda: write_json(directory / "ready.json", dict(metadata, pid=os.getpid(),
                elapsed_load_seconds=time.monotonic()-started)))
        signal.signal(signal.SIGTERM, lambda *unused: server.request_stop())
        server.serve_forever(stop_on_request=True)
    finally:
        runtime.close()
        if captures is not None:
            write_json(directory / 'CAPTURE_CLOSED.json', captures.close())
        write_json(directory / "closed.json", dict(status="CLOSED", pid=os.getpid(),
            elapsed_seconds=time.monotonic()-started, peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
            peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20, batching=config.metadata()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--cuda-graph", action="store_true")
    args = parser.parse_args()
    serve(json.loads(args.campaign.read_text()), args.port, args.directory, args.cuda_graph)


if __name__ == "__main__":
    main()
