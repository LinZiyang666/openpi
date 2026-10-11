"""Explicit CPU topology audit and serial GPU raw/released-loader parity.

The CPU audit uses real constructors on meta tensors and substitutes ONLY frozen
sidecars and tensor body loading. It proves config/trainable header equivalence,
not frozen VAE/T5 weights or output equivalence. No import-time model loading.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
import time


def audit_loader_headers(checkpoint, wan_root):
    import torch
    from unittest.mock import patch
    from lerobot.policies.fastwam.configuration_fastwam import FastWAMConfig
    from lerobot.policies.fastwam.modeling_fastwam import FastWAMPolicy
    import lerobot.policies.fastwam.modeling_fastwam as implementation
    from lerobot.policies.fastwam.wan.video_dit import WanContinuousFlowMatchScheduler
    from exp.mw_finetune.fastwam.model import canonical_key, skip_overwritten_initialization
    from .loader import LoadSpec, _load_libero, _raw_loaders, inspect_config, safetensor_header
    from .manifest import make_sampler
    spec = LoadSpec("libero", str(checkpoint), str(wan_root))
    config = inspect_config(spec)
    header = safetensor_header(Path(checkpoint)/"model.safetensors")
    expected = {canonical_key(k):tuple(v["shape"]) for k,v in header.items()}
    if len(expected) != len(header):
        raise ValueError("checkpoint contains duplicate canonical aliases")
    calls = []
    class FrozenStub(torch.nn.Module):
        pass
    def frozen(**kwargs):
        calls.append({k:str(v) for k,v in kwargs.items()})
        return FrozenStub()
    @contextmanager
    def frozen_substitutes():
        with ExitStack() as stack:
            for name in ("load_pretrained_wan_vae","load_pretrained_wan_text_encoder","build_wan_tokenizer"):
                stack.enter_context(patch.object(implementation,name,frozen))
            yield
    def shapes(policy):
        return {canonical_key(k):tuple(p.shape) for k,p in policy.named_parameters()}
    config_fields = ("action_dim","proprio_dim","action_horizon","n_action_steps","n_obs_steps",
        "image_size","num_video_frames","action_video_freq_ratio","prompt_template","tokenizer_max_len",
        "num_inference_steps","inference_seed","rand_device","text_cfg_scale","negative_prompt",
        "sigma_shift","tiled","compile_action_infer","toggle_action_dimensions","normalization_mapping",
        "video_scheduler","action_scheduler","video_dit_config","action_dit_config")
    def inference_config(policy):
        return json.dumps({k:getattr(policy.config,k) for k in config_fields},sort_keys=True,default=str)
    def released_header_load(cls, policy, model_file, map_location, strict):
        from .loader import audit_parameters
        if not strict:
            raise ValueError("released comparison requires strict=True")
        audit_parameters(expected, shapes(policy))
        return policy
    native_scheduler_init = WanContinuousFlowMatchScheduler.__init__
    def cpu_scheduler_init(self,*args,**kwargs):
        # Native schedule scalar precomputation uses .item(); its small grids
        # stay on CPU while the multi-billion-parameter modules are on meta.
        with torch.device("cpu"):
            native_scheduler_init(self,*args,**kwargs)
    with patch("safetensors.torch.load_file",lambda *args,**kwargs:
               {k:torch.empty(v["shape"],device="meta",dtype=torch.bfloat16) for k,v in header.items()}), \
         patch.object(FastWAMPolicy,"_load_as_safetensor",classmethod(released_header_load)), \
         patch.object(WanContinuousFlowMatchScheduler,"__init__",cpu_scheduler_init), \
         torch.device("meta"), skip_overwritten_initialization():
        # Actual P3 raw-loader dispatch and parameter-copy path. Installation
        # only imports code/checks asset existence; no frozen weight body reads.
        original_path = tuple(sys.path)
        original_loaders = {n:getattr(implementation,n) for n in
            ("load_pretrained_wan_vae","load_pretrained_wan_text_encoder")}
        with _raw_loaders(spec.wan_root):
            selected_loaders = {name:getattr(implementation,name).__qualname__ for name in
                ("load_pretrained_wan_vae","load_pretrained_wan_text_encoder")}
            with frozen_substitutes():
                raw = _load_libero(spec,"meta")
        if tuple(sys.path) != original_path or any(getattr(implementation,n) is not f for n,f in original_loaders.items()):
            raise ValueError("raw loader scope did not restore paths/loader functions")
        # Actual released from_pretrained constructor/load dispatch, strict.
        cfg = FastWAMConfig.from_pretrained(str(checkpoint),local_files_only=True)
        cfg.device, cfg.dtype = "meta", torch.bfloat16
        with frozen_substitutes():
            released = FastWAMPolicy.from_pretrained(str(checkpoint),config=cfg,strict=True,local_files_only=True)
        from .loader import audit_parameters
        raw_shapes, released_shapes = shapes(raw), shapes(released)
        native_state_shapes = {k:tuple(v.shape) for k,v in released.state_dict().items()}
        audit_parameters(expected, raw_shapes)
        audit_parameters(raw_shapes, released_shapes)
        if inference_config(raw) != inference_config(released):
            raise ValueError("raw/released inference config mismatch")
        if {k:str(p.dtype) for k,p in raw.named_parameters()} != {k:str(p.dtype) for k,p in released.named_parameters()}:
            raise ValueError("raw/released parameter dtype mismatch")
        original_shapes = {k:tuple(v["shape"]) for k,v in header.items()}
        # Current vendor reparents blocks onto MoT layers; the published export
        # uses their older mixtures.*.blocks aliases. Audit this explicitly.
        audit_parameters(expected,native_state_shapes)
    raw_sampler = make_sampler(raw.model.infer_action_scheduler,10)
    released_sampler = make_sampler(released.model.infer_action_scheduler,10)
    if raw_sampler != released_sampler:
        raise ValueError("raw/released scheduler config mismatch")
    return {"schema":"r17.fastwam.loader_header_equivalence.v1", "status":"passed",
        "checkpoint":str(checkpoint), "device":"meta", "tensor_body_reads":0,
        "matched_unique_parameters":len(expected), "matched_shapes":True,
        "matched_inference_config_fields":list(config_fields), "matched_parameter_dtypes":True,
        "released_serialization":{
            "original_key_sets_equal":set(original_shapes)==set(native_state_shapes),
            "renamed_checkpoint_keys":sum(k!=canonical_key(k) for k in header),
            "canonical_state_dict_shapes_match":True,
            "GPU_comparison_shim":"canonical alias rename only; strict current native load_state_dict"},
        "published_geometry":{k:config[k] for k in ("image_size","action_dim","proprio_dim","action_horizon")},
        "sampler":asdict(raw_sampler), "selected_raw_loaders":selected_loaders,
        "raw_loader_scope_restored":True,"frozen_loader_calls":calls,
        "limits":["VAE/T5/tokenizer substituted; frozen sidecar numerical parity NOT established",
                  "meta shapes and strict canonical header mapping only; trained tensor values NOT loaded",
                  "unmodified released serialization loader is NOT certified; published aliases need normalization"]}


def _load_released_canonical(cls, policy, model_file, map_location, strict):
    """Qualification-only serialization shim; unchanged released forward math.

    Normalize published aliases to the current vendor's registered names, then
    use native strict load_state_dict. Never drop missing/unexpected tensors.
    """
    from safetensors.torch import load_file
    from exp.mw_finetune.fastwam.model import canonical_key
    from .loader import audit_parameters
    if not strict:
        raise ValueError("released parity requires strict loading")
    source = load_file(model_file,device="cpu")
    state = {canonical_key(k):v for k,v in source.items()}
    if len(state) != len(source):
        raise ValueError("duplicate canonical checkpoint aliases")
    audit_parameters({k:tuple(v.shape) for k,v in policy.state_dict().items()},
                     {k:tuple(v.shape) for k,v in state.items()})
    policy.load_state_dict(state,strict=True)
    return policy


def compare_outputs(raw, released, *, rtol, atol):
    import math
    import numpy as np
    if any(not math.isfinite(t) or t < 0 for t in (rtol,atol)):
        raise ValueError("parity tolerances must be finite and nonnegative")
    if set(raw) != set(released) or not raw:
        raise ValueError("loader outputs must have identical nonempty field sets")
    rows = []
    for key in sorted(raw):
        a,b = np.asarray(raw[key]),np.asarray(released[key])
        if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError("loader output geometry/finiteness mismatch: " + key)
        error = float(np.max(np.abs(a.astype(np.float64)-b.astype(np.float64))))
        exact = key.rsplit(":",1)[-1] in ("input_ids","tokenizer_mask","context_mask")
        rows.append({"field":key,"shape":list(a.shape),"max_abs_error":error,"exact":exact,
                     "passed":bool(np.array_equal(a,b) if exact else np.allclose(a,b,rtol=rtol,atol=atol))})
    return {"status":"passed" if all(r["passed"] for r in rows) else "failed",
            "rtol":rtol,"atol":atol,"rows":rows}


def _gpu_worker(args):
    import numpy as np
    import torch
    from .loader import LoadSpec, load_adapter, read_stats
    from .wire_policy import FastWAMWirePolicy
    from exp.r17_integration.common.contracts import RequestContext
    from .protocols import validate_benchmark_specs
    spec = LoadSpec(**json.loads(Path(args.descriptor).read_text()))
    if spec.benchmark != "libero" or spec.execution_protocol != "released_libero":
        raise ValueError("parity requires LIBERO released_libero profile and corresponding P2 spec")
    validate_benchmark_specs(spec)
    if args.worker == "raw":
        adapter = load_adapter(spec,device=args.device)
        model, wire = adapter.model, adapter.wire
    else:
        # Use unchanged published HF/Diffusers component functions with explicit
        # local assets. Offline env prevents implicit network fetches.
        from lerobot.policies.fastwam.configuration_fastwam import FastWAMConfig
        from lerobot.policies.fastwam.modeling_fastwam import FastWAMPolicy
        from lerobot.policies.fastwam.wan import components
        from unittest.mock import patch
        from exp.mw_finetune.fastwam.model import skip_overwritten_initialization
        from .loader import audit_parameters, safetensor_header
        from exp.mw_finetune.fastwam.model import canonical_key
        root = Path(args.released_sidecars).resolve(strict=True)
        if not all((root/p).is_dir() for p in ("vae","text_encoder")):
            raise ValueError("local released Diffusers vae/text_encoder directories required")
        components.WAN22_DIFFUSERS_MODEL_ID = str(root)
        cfg = FastWAMConfig.from_pretrained(spec.checkpoint,local_files_only=True)
        cfg.device, cfg.dtype = args.device, torch.bfloat16
        cfg.text_encoder_model_id = str(root)
        tokenizer = Path(args.released_tokenizer).resolve(strict=True)
        if not (tokenizer/"tokenizer_config.json").is_file():
            raise ValueError("complete local released UMT5 tokenizer required")
        cfg.tokenizer_model_id = str(tokenizer)
        cfg.use_gradient_checkpointing = cfg.mot_checkpoint_mixed_attn = False
        with skip_overwritten_initialization(), \
             patch.object(FastWAMPolicy,"_load_as_safetensor",classmethod(_load_released_canonical)):
            policy = FastWAMPolicy.from_pretrained(spec.checkpoint,config=cfg,strict=True,local_files_only=True)
        header = safetensor_header(Path(spec.checkpoint)/"model.safetensors")
        audit_parameters({canonical_key(k):tuple(p.shape) for k,p in policy.named_parameters()},
                         {canonical_key(k):v["shape"] for k,v in header.items()})
        model = policy.model
        wire = FastWAMWirePolicy("libero",read_stats(spec),prompt_template=cfg.prompt_template,
                                 execution_protocol="released_libero")
    with np.load(args.observations,allow_pickle=False) as data:
        if len(data["state"]) < 2:
            raise ValueError("at least two real paired LIBERO observations required")
        observations = [{k:str(data[k][i]) if k=="prompt" else data[k][i].copy()
                         for k in data.files} for i in range(2)]
    outputs = {}
    with torch.inference_mode():
        for i,obs in enumerate(observations):
            req = RequestContext("loader_parity",spec.benchmark_protocols[0],"fixed_observation",i,0,explicit_seed=42)
            prepared = wire.prepare(obs,req)
            input_ids,tokenizer_mask = model.tokenizer([prepared.data["prompt"]],return_mask=True)
            pixels = prepared.data["input_image"].to(model.device,model.torch_dtype)
            latents = model._encode_input_image_latents_tensor(pixels,tiled=False,
                tile_size=(30,52),tile_stride=(15,26))
            context,mask = model._prepare_infer_context([prepared.data["prompt"]],None,None,
                prepared.data["proprio"].to(model.device,model.torch_dtype))
            native = model.infer_action(prompt=prepared.data["prompt"],input_image=prepared.data["input_image"],
                proprio=prepared.data["proprio"],action_horizon=32,num_inference_steps=10,seed=42,
                rand_device="cpu",text_cfg_scale=1.0,compile_action_infer=False)["action"]
            for name,value in {"input_ids":input_ids,"tokenizer_mask":tokenizer_mask,
                               "latents":latents,"context":context,"context_mask":mask,
                               "normalized_actions":native}.items():
                outputs[f"{i}:{name}"] = value.detach().cpu().numpy() if name=="input_ids" else value.detach().float().cpu().numpy()
            outputs[f"{i}:wire_actions"] = wire.to_wire(native,prepared).actions
    np.savez(args.worker_output,**outputs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--header-only",action="store_true")
    parser.add_argument("--checkpoint")
    parser.add_argument("--wan-root")
    parser.add_argument("--descriptor")
    parser.add_argument("--released-sidecars")
    parser.add_argument("--released-tokenizer")
    parser.add_argument("--observations")
    parser.add_argument("--device",default="cpu")
    parser.add_argument("--receipt",required=True)
    parser.add_argument("--deadline-seconds",type=float,default=840)
    parser.add_argument("--rtol",type=float,default=1e-3)
    parser.add_argument("--atol",type=float,default=1e-3)
    parser.add_argument("--worker",choices=("raw","released"),help=argparse.SUPPRESS)
    parser.add_argument("--worker-output",help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        _gpu_worker(args)
        return
    started = time.monotonic()
    receipt = {"schema":"r17.fastwam.loader_parity.v1","status":"failed"}
    try:
        if args.header_only:
            receipt = audit_loader_headers(args.checkpoint,args.wan_root)
        else:
            if not all((args.descriptor,args.released_sidecars,args.released_tokenizer,args.observations)) or not args.device.startswith("cuda"):
                raise ValueError("GPU parity requires sealed descriptor, local released sidecars/tokenizer, real observations and cuda device")
            import os
            import numpy as np
            from .manifest import content_identity
            from .loader import LoadSpec
            from .protocols import validate_benchmark_specs
            spec = LoadSpec(**json.loads(Path(args.descriptor).read_text()))
            validate_benchmark_specs(spec)
            if spec.benchmark != "libero" or spec.execution_protocol != "released_libero":
                raise ValueError("loader parity requires LIBERO released_libero profile")
            receipt.update(device=args.device,execution_protocol=spec.execution_protocol,
                benchmark_protocols=spec.benchmark_protocols,observation_count=2,k=10,seed=42,
                timing_scope="qualification_with_capture; not pricing",
                released_serialization="published canonical alias rename + native strict load_state_dict",
                frozen_component_path="unchanged released HF/Diffusers loader functions, local offline assets")
            # Bind both encoder exports and observations, before GPU workers.
            sources = {"descriptor":Path(args.descriptor),"observations":Path(args.observations)}
            root = Path(args.released_sidecars).resolve(strict=True)
            for component in ("vae","text_encoder"):
                directory = root/component
                if not directory.is_dir():
                    raise ValueError("missing local released sidecar: " + component)
                sources.update({"released:"+str(p.relative_to(root)):p
                    for p in directory.rglob("*") if p.is_file()})
            tokenizer = Path(args.released_tokenizer).resolve(strict=True)
            if not (tokenizer/"tokenizer_config.json").is_file():
                raise ValueError("missing local released tokenizer_config.json")
            sources.update({"released_tokenizer:"+p.name:p for p in tokenizer.iterdir() if p.is_file()})
            receipt["source_digests"] = content_identity(sources)
            results = []
            for worker in ("raw","released"):
                output = Path(args.receipt).with_suffix(f".{worker}.npz")
                command = [sys.executable,"-m",__package__+".loader_parity","--worker",worker,
                    "--descriptor",args.descriptor,"--released-sidecars",args.released_sidecars,
                    "--released-tokenizer",args.released_tokenizer,
                    "--observations",args.observations,"--device",args.device,
                    "--receipt",args.receipt,"--worker-output",str(output)]
                remaining = args.deadline_seconds-(time.monotonic()-started)
                if remaining <= 0:
                    raise TimeoutError("loader parity deadline exhausted")
                proc = subprocess.run(command,env={**os.environ,"HF_HUB_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1"},
                    capture_output=True,text=True,timeout=remaining)
                if proc.returncode:
                    raise RuntimeError(f"{worker} worker exited {proc.returncode}: {proc.stderr[-3000:]}")
                with np.load(output,allow_pickle=False) as data:
                    results.append({k:data[k].copy() for k in data.files})
            receipt.update(compare_outputs(*results,rtol=args.rtol,atol=args.atol))
            receipt["limits"] = ["two fixed B1 observations/K10/seed42 only; no SR or throughput certificate",
                                  "raw and released workers exit sequentially; no simultaneous GPU residency"]
    except Exception as exc:
        receipt.update(status="failed",error=f"{type(exc).__name__}: {exc}")
    receipt["elapsed_seconds"] = time.monotonic()-started
    Path(args.receipt).write_text(json.dumps(receipt,indent=2,allow_nan=False)+"\n")
    if receipt["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
