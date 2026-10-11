"""Small native-stage stubs. Scheduler and reference loop are vendor source."""
from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from exp.r17_integration.common.contracts import RequestContext
from exp.r17_integration.models.fastwam.manifest import make_manifest, make_sampler
from exp.r17_integration.models.fastwam.staged import FastWAMAdapter
from exp.r17_integration.models.fastwam.wire_policy import FastWAMWirePolicy

FIXTURES = Path(__file__).parent / "fixtures"


def native_defs(name, env=None):
    # Execute exact checked-in source definitions, rather than a reimplemented
    # sampler. No importing heavyweight model/vendor dependency trees.
    source = (FIXTURES / name).read_text()
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            node.col_offset = 0
    namespace = {"torch": torch, "np": np, "Any": Any, **(env or {})}
    exec(compile(tree, str(FIXTURES / name), "exec"), namespace)
    return namespace


NATIVE = native_defs("video_dit_native.py.txt")
MODULAR = native_defs("modular_native.py.txt")
Scheduler = NATIVE["WanContinuousFlowMatchScheduler"]


class ToyVAE:
    def __init__(self, model):
        self.model = model

    def encode(self, images, **kwargs):
        self.model.calls["vae"].append(len(images))
        self.model.last_pixels = torch.stack(images).clone()
        return [torch.nn.functional.avg_pool2d(image[:, 0], 16).unsqueeze(1) for image in images]


class ToyVideo:
    video_attention_mask_mode = "first_frame_causal"
    fuse_vae_embedding_in_latents = True

    def __init__(self, model):
        self.model = model
        self.patch_embedding = SimpleNamespace(weight=torch.empty(0,dtype=model.torch_dtype))

    def patchify(self, x):
        self.model.calls["patch"].append(len(x))
        base = torch.nn.functional.avg_pool2d(x[:, :, 0].mean(dim=1, keepdim=True), 2)
        return (base.repeat(1,8,1,1) * torch.arange(1,9).reshape(1,8,1,1)).unsqueeze(2)

    def pre_dit(self, x, timestep, context, context_mask, **kwargs):
        b = len(x)
        self.model.calls["pre"].append(b)
        patches = self.patchify(x)
        _, h, w = patches.shape[2:]
        tokens = patches.flatten(2).transpose(1,2).contiguous()
        grids = torch.tensor([[1,h,w]]*b)
        return {"tokens": tokens, "t": torch.zeros(b,h*w,8), "t_mod": torch.zeros(b,h*w,6,8),
            "context": context.clone(), "context_mask": context_mask.unsqueeze(1).expand(-1,h*w,-1).clone(),
            "freqs": {"grid_sizes": grids.clone(), "freqs": torch.arange(5)},
            "meta": {"grid_sizes": grids, "tokens_per_frame": h*w, "batch_size":b}}


class ToyMoT:
    def __init__(self, model):
        self.model = model

    def prefill_video_cache(self, video_tokens, video_context_payload, **kwargs):
        self.model.calls["prefill"].append(len(video_tokens))
        ctx = video_context_payload["context"].mean((1,2)).reshape(-1,1,1)
        return [{"k":video_tokens.clone()+ctx, "v":video_tokens.clone()*2+ctx},
                {"k":video_tokens.clone()+ctx+1, "v":video_tokens.clone()*2+ctx-1}]


class ToyModel:
    def __init__(self, action_dim, state_dim, *, dtype=torch.float32):
        self.device, self.torch_dtype, self.proprio_dim = torch.device("cpu"), dtype, state_dim
        self.action_expert = SimpleNamespace(action_dim=action_dim)
        self.infer_action_scheduler = Scheduler(1000, 5)
        self.calls = {k:[] for k in ("vae", "patch", "pre", "prefill", "head", "context")}
        self.vae, self.video_expert, self.mot = ToyVAE(self), ToyVideo(self), ToyMoT(self)

    def eval(self):
        return self

    def encode_prompt(self, prompt):
        prompts = [prompt] if isinstance(prompt,str) else prompt
        self.calls["context"].append(len(prompts))
        features = torch.tensor([sum(p.encode())/1000 for p in prompts],dtype=self.torch_dtype)
        return features.reshape(-1,1,1).expand(-1,3,8).clone(), torch.ones(len(prompts),3,dtype=torch.bool)

    def _append_proprio_to_context(self, context, context_mask, proprio):
        token = proprio.mean(1).reshape(-1,1,1).expand(-1,1,8)
        return torch.cat([context,token],1), torch.cat([context_mask,torch.ones(len(context),1,dtype=torch.bool)],1)

    def _build_mot_attention_mask(self, video_seq_len, action_seq_len, **kwargs):
        return torch.ones(video_seq_len+action_seq_len,video_seq_len+action_seq_len,dtype=torch.bool)

    def _predict_action_noise_with_cache(self, latents_action, timestep_action, context, video_kv_cache, **kwargs):
        self.calls["head"].append(len(latents_action))
        ctx = context.mean((1,2)).reshape(-1,1,1)
        kv = sum(c["k"].mean((1,2)) + c["v"].mean((1,2)) for c in video_kv_cache).reshape(-1,1,1)
        return latents_action*.15 + ctx*.01 + kv*.01 + timestep_action.reshape(-1,1,1)*.00001


for method in ("infer_action", "_normalize_infer_input_image", "_normalize_infer_proprio",
               "_prepare_infer_context", "_normalize_context_tensors", "_make_action_latents",
               "_encode_input_image_latents_tensor"):
    setattr(ToyModel, method, MODULAR[method])


def make_adapter(benchmark="metaworld", *, allow_unqualified_batch=True, protocol="system"):
    config = json.loads((FIXTURES/"libero_config.json").read_text())
    a,s = (7,8) if benchmark == "libero" else (4,4)
    if benchmark == "metaworld":
        config.update(action_dim=4,proprio_dim=4,image_size=[224,224],toggle_action_dimensions=[])
    stats = {"action":{"min":[-2.]*(a-1)+[0.], "max":[2.]*(a-1)+[1.]},
             "observation.state":{"min":[-1.]*s,"max":[1.]*s}}
    model = ToyModel(a,s)
    sampler = make_sampler(model.infer_action_scheduler, 10, dtype="float32")
    manifest = make_manifest(benchmark,config,stats,sampler,{"base":"fixture_only_not_real_weights"},hidden_width=8,
        execution_protocol=protocol, benchmark_protocols=(f"fake_fastwam_{benchmark}_fixed_observations_v1",))
    return FastWAMAdapter(model,manifest,FastWAMWirePolicy(benchmark,stats,execution_protocol=protocol),config,
        allow_unqualified_batch=allow_unqualified_batch)


@pytest.fixture
def adapter():
    return make_adapter()


def observation(benchmark="metaworld", variant=0):
    s = 8 if benchmark == "libero" else 4
    size = 256 if benchmark == "libero" else 480
    obs = {"image":np.full((size,size,3), 50+variant*70,np.uint8),
           "state":np.arange(s,dtype=np.float32)/10+variant/20,"prompt":f"reach target {variant}"}
    if benchmark == "libero":
        obs["image2"] = np.full((256,256,3),220-variant*30,np.uint8)
    return obs


def request(seq=0, *, explicit_seed=None):
    kwargs = {} if explicit_seed is None else {"explicit_seed":explicit_seed}
    return RequestContext("p3_cpu", "metaworld_mt50_r17_v1", "reach-v3", 0, seq, **kwargs)


def fixture_benchmark_spec(benchmark):
    """Complete P2 task table; LIBERO names are explicitly CPU fixtures only."""
    if benchmark == "libero":
        from exp.r17_integration.benchmarks.libero_bridge import LiberoBenchmarkSpec, SUITES
        return LiberoBenchmarkSpec(tuple((f"{suite}/cpu_task_{i}",f"CPU task {i}",suite)
            for suite in SUITES for i in range(10)))
    from exp.r17_integration.benchmarks.spec import BenchmarkSpec
    from exp.metaworld.tasks import BENCHMARK, MT50_TASKS
    return BenchmarkSpec(BENCHMARK,"mt50_seed7_corner2_r17_v1",MT50_TASKS)
