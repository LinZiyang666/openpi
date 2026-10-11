"""Opt-in denoiser-only CUDA replay. RNG/scheduler/connection state stay eager."""
import torch


def enable_graph(adapter):
    from exp.mw_finetune.dp.cuda_graph import DenoiserGraph
    if adapter.device.type != "cuda":
        raise ValueError("DP CUDA graph requires an explicit CUDA adapter")
    original = adapter.model._policy.model
    graph = DenoiserGraph(original)
    class GraphDenoiser(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.original = original
        def forward(self, trajectory, timestep, cond):
            if getattr(graph, "r17_capture_sealed", False):
                key = tuple((tuple(x.shape), x.dtype, x.stride()) for x in (trajectory, timestep, cond)) + (trajectory.device,)
                if key not in graph.graphs:
                    raise RuntimeError("DP graph signature not prewarmed; live capture is forbidden during concurrent stages")
            return graph(trajectory, timestep, cond)
    wrapped = GraphDenoiser().eval()
    adapter.model._policy.model = wrapped
    return graph, original


@torch.inference_mode()
def warm_batch_signatures(adapter, graph, stage2, maximum, timestep):
    """Capture every legal B before starting stage threads (global capture mode).

    E's graph capture is global. A different stage's GPU-to-CPU readback during
    a lazy capture is illegal, even on another stream. Serving may replay only.
    """
    cond = stage2.data["conditioning"]
    if not 1 <= maximum <= 32 or len(cond) != 1:
        raise ValueError("DP graph prewarm requires a B1 conditioning and max1..32")
    for size in range(1, maximum+1):
        conditioning = torch.cat([cond] * size, 0)
        trajectory = torch.zeros((size, 16, adapter.model.cfg["action_dim"]), dtype=cond.dtype, device=cond.device)
        graph(trajectory, timestep, conditioning)
    graph.r17_capture_sealed = True
    return dict(batch_sizes=list(range(1, maximum+1)), live_capture_allowed=False,
                scope="denoiser graph signatures; completed before any S07 stage thread starts")


@torch.inference_mode()
def check_graph(adapter, observations, *, steps=(100, 10, 5)):
    """B1 wire equality for every arm before accepting an opt-in graph server."""
    from exp.r17_integration.common.contracts import CaptureSpec, RequestContext
    from .request_noise import RequestNoise
    from .sampler import SamplerProfile
    graph, original = enable_graph(adapter)
    wrapped = adapter.model._policy.model
    checked = []
    try:
        for i, observation in enumerate(observations):
            context = RequestContext("dp_batch_graph_check", adapter.metadata().allowed_benchmark_protocols[0],
                "fixed_real_frame", i, 0, explicit_seed=18+i*19)
            s2 = adapter.run_stage2(adapter.run_stage1(adapter.prepare(observation, context)))
            for k in steps:
                profile = SamplerProfile(num_steps=k)
                adapter.model._policy.model = original
                eager = adapter.sample(s2, profile, RequestNoise(context.explicit_seed, adapter.device), CaptureSpec())
                adapter.model._policy.model = wrapped
                actual = adapter.sample(s2, profile, RequestNoise(context.explicit_seed, adapter.device), CaptureSpec())
                expected_wire = adapter.to_wire(eager.normalized_chunk, None).actions
                actual_wire = adapter.to_wire(actual.normalized_chunk, None).actions
                if expected_wire.tobytes() != actual_wire.tobytes() or eager.rng_receipt != actual.rng_receipt:
                    raise RuntimeError(f"DP graph B1 self-check failed for row{i}/K{k}")
                checked.append({"row": i, "K": k, "wire_bytes_exact": True, "rng_exact": True})
    except BaseException:
        adapter.model._policy.model = original
        raise
    return {"passed": True, "implementation": "E DenoiserGraph; one S07 stage3 worker owns signature workspaces",
            "checks": checked, "graph": graph}
