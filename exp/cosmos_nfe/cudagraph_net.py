"""CUDA-graph replay for the Cosmos Policy DiT forward (``model.net``), one graph per input signature.

Every denoising step of ``CosmosPolicySampler`` calls ``model.denoise`` -> ``model.net(x, timesteps, **condition)``
with tensors of fixed shape (the latent sequence, the T5 embedding, the masks) and a few non-tensor flags.
``GraphedNet`` runs the first ``warmup`` calls of a signature eagerly, then captures one ``torch.cuda.CUDAGraph``
on static input buffers and afterwards only copies the inputs in and replays. Nothing about the numerics changes
(same kernels, same order); the win is the launch overhead of a 2B-parameter transformer at batch size 1.

Usage::

    from exp.cosmos_nfe.cudagraph_net import GraphedNet
    model.net = GraphedNet(model.net)          # after get_model(); capture happens on the 3rd call per signature
"""

from __future__ import annotations

import time

import torch


def _sig(args, kwargs) -> tuple:
    def one(v):
        if torch.is_tensor(v):
            return ("T", tuple(v.shape), str(v.dtype), str(v.device))
        if isinstance(v, (list, tuple)):
            return (type(v).__name__, tuple(one(x) for x in v))
        if isinstance(v, dict):
            return ("D", tuple((k, one(x)) for k, x in sorted(v.items())))
        return ("V", repr(v))

    return (tuple(one(a) for a in args), tuple((k, one(v)) for k, v in sorted(kwargs.items())))


def _clone_static(v):
    if torch.is_tensor(v):
        return v.detach().clone()
    if isinstance(v, list):
        return [_clone_static(x) for x in v]
    if isinstance(v, tuple):
        return tuple(_clone_static(x) for x in v)
    if isinstance(v, dict):
        return {k: _clone_static(x) for k, x in v.items()}
    return v


def _copy_into(static, v) -> None:
    if torch.is_tensor(v):
        static.copy_(v, non_blocking=True)
    elif isinstance(v, (list, tuple)):
        for s, x in zip(static, v):
            _copy_into(s, x)
    elif isinstance(v, dict):
        for k, x in v.items():
            _copy_into(static[k], x)


class GraphedNet(torch.nn.Module):
    def __init__(self, net: torch.nn.Module, warmup: int = 2, verbose: bool = True):
        super().__init__()
        self.net = net
        self.warmup = warmup
        self.verbose = verbose
        self.graphs: dict = {}
        self.counts: dict = {}
        self.stats = {"eager": 0, "replay": 0, "captures": 0}

    def __getattr__(self, name):  # forward attribute access (config, dtype helpers) to the wrapped net
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.net, name)

    def forward(self, *args, **kwargs):
        key = _sig(args, kwargs)
        entry = self.graphs.get(key)
        if entry is None:
            n = self.counts.get(key, 0) + 1
            self.counts[key] = n
            if n <= self.warmup:
                self.stats["eager"] += 1
                return self.net(*args, **kwargs)
            entry = self._capture(key, args, kwargs)
        graph, s_args, s_kwargs, s_out = entry
        _copy_into(s_args, args)
        _copy_into(s_kwargs, kwargs)
        graph.replay()
        self.stats["replay"] += 1
        return s_out.clone() if torch.is_tensor(s_out) else s_out

    def _capture(self, key, args, kwargs):
        t0 = time.perf_counter()
        s_args = _clone_static(args)
        s_kwargs = _clone_static(kwargs)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):  # side-stream warm-up as torch.cuda.graph docs require (allocator, lazy inits)
                self.net(*s_args, **s_kwargs)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            s_out = self.net(*s_args, **s_kwargs)
        torch.cuda.synchronize()
        self.graphs[key] = (graph, s_args, s_kwargs, s_out)
        self.stats["captures"] += 1
        if self.verbose:
            shapes = [tuple(a.shape) for a in args if torch.is_tensor(a)][:2]
            print(f"CUDAGRAPH captured #{self.stats['captures']} x={shapes} in {time.perf_counter() - t0:.1f}s", flush=True)
        return self.graphs[key]
