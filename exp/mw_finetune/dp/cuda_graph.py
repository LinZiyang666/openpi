"""Inference-only denoiser replay; DDPM sampling and private RNG stay eager."""
from contextlib import contextmanager
import threading

import numpy as np
import torch


@contextmanager
def _causal_hint(model, horizon):
    # TransformerDecoder otherwise reads a CUDA mask back to Python on every
    # forward to detect causality. Compute that same hint once outside capture;
    # leave the actual attention mask and all attention arithmetic unchanged.
    decoder = getattr(model, "decoder", None)
    hook = None
    if isinstance(decoder, torch.nn.TransformerDecoder):
        from torch.nn.modules.transformer import _detect_is_causal_mask
        hint = _detect_is_causal_mask(model.mask, size=horizon)

        def supply_hint(module, args, kwargs):
            return args, {**kwargs, "tgt_is_causal": hint}

        hook = decoder.register_forward_pre_hook(supply_hint, with_kwargs=True)
    try:
        yield
    finally:
        if hook is not None:
            hook.remove()


class DenoiserGraph:
    """One graph per tensor signature, with no cached observation or RNG state.

    Output aliases graph workspace until the next replay. The sampler consumes
    it immediately on the same stream. Policy servers already serialize calls.
    """

    def __init__(self, model):
        self.model = model
        self.graphs = {}
        self._lock = threading.Lock()

    @torch.inference_mode()
    def __call__(self, trajectory, timestep, cond):
        if trajectory.device.type != "cuda" or self.model.training:
            raise ValueError("CUDA graph denoiser requires a CUDA model in eval mode")
        # Scheduler timesteps remain CPU scalars. Only their denoiser copies
        # live on CUDA; scheduler.step continues to receive the original t.
        key = tuple((tuple(x.shape), x.dtype, x.stride())
                    for x in (trajectory, timestep, cond)) + (trajectory.device,)
        if key not in self.graphs:
            static_trajectory = trajectory.clone()
            static_timestep = timestep.to(trajectory.device).clone()
            static_cond = cond.clone()
            current = torch.cuda.current_stream(trajectory.device)
            warmup = torch.cuda.Stream(device=trajectory.device)
            warmup.wait_stream(current)
            with _causal_hint(self.model, trajectory.shape[1]):
                with torch.cuda.stream(warmup):
                    for _ in range(3):
                        self.model(static_trajectory, static_timestep, static_cond)
                current.wait_stream(warmup)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=warmup):
                    output = self.model(static_trajectory, static_timestep, static_cond)
            self.graphs[key] = (graph, static_trajectory, static_timestep, static_cond, output)
        graph, static_trajectory, static_timestep, static_cond, output = self.graphs[key]
        static_trajectory.copy_(trajectory)
        static_timestep.copy_(timestep)
        static_cond.copy_(cond)
        graph.replay()
        return output

    @torch.inference_mode()
    def sample(self, policy, obs, generator=None, steps=None):
        # Serialize workspace access even when called outside the server lock.
        with self._lock:
            return self._sample(policy, obs, generator, steps)

    def _sample(self, policy, obs, generator, steps):
        # Mirrors model.sample verbatim except for the denoiser dispatch.
        core = policy._policy
        cond = policy.condition(obs)
        trajectory = torch.randn((len(cond), 16, policy.cfg["action_dim"]), dtype=cond.dtype,
                                 device=cond.device, generator=generator)
        scheduler = policy.scheduler(steps)
        for t in scheduler.timesteps:
            noise = self(trajectory, t, cond)
            trajectory = scheduler.step(noise, t, trajectory, generator=generator).prev_sample
        return core.normalizer.unnormalize("action", trajectory)


def check_observations(domain, stats):
    prompts = ("pick up the black bowl and place it on the plate",
               "open the middle drawer", "put the white mug on the left plate")
    if domain == "metaworld":
        prompts = ("reach the target", "press the button", "close the door")
    for seed, prompt in zip((11, 37, 101), prompts):
        rng = np.random.default_rng(seed)
        image = rng.integers(0, 256, (96, 96, 3), dtype=np.uint8)
        low = np.asarray(stats["state"]["q01"], np.float32)
        high = np.asarray(stats["state"]["q99"], np.float32)
        state = (low + rng.uniform(.15, .85, low.shape) * (high - low)).astype(np.float32)
        obs = {"prompt": prompt, "seed": seed}
        if domain == "libero":
            obs.update({"observation/image": image, "observation/state": state,
                        "observation/wrist_image": rng.integers(0, 256, image.shape, dtype=np.uint8)})
        else:
            obs.update(image=image, state=state)
        yield obs


def require_bit_exact(observations, eager, graph):
    checked = []
    for obs in observations:
        expected = torch.as_tensor(eager(obs)["actions"])
        actual = torch.as_tensor(graph(obs)["actions"])
        if not torch.equal(expected, actual):
            raise RuntimeError(f"CUDA graph self-check failed for seed={obs['seed']}, prompt={obs['prompt']!r}; refusing to serve")
        checked.append({"seed": obs["seed"], "prompt": obs["prompt"]})
    if len(checked) < 3 or len({o["seed"] for o in checked}) < 3 or len({o["prompt"] for o in checked}) < 3:
        raise RuntimeError("CUDA graph self-check requires at least three distinct seeds and prompts")
    return {"passed": True, "comparison": "torch.equal(final physical actions)", "observations": checked}
