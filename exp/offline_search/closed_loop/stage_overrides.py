"""Opt-in pi05 LIBERO cost overrides; no src or plugin changes.

Install before constructing the model/coordinator. No patches are made for full.
Wrist stage outputs OWN their deferred input, including across coordinator splits,
rebatches, and .to(). No connection-global pending image or episode state exists.
The concurrent server is eager SDPA (no CUDA graphs); these overrides deliberately
validate the dummy at each eager boundary. Packing drops only the 256 permanently
masked dummy slots, preserving prompt padding and batch-compatible shapes.
"""
from __future__ import annotations
import argparse
import dataclasses
import functools
import math
import pathlib
import threading


def parse_flags(argv):
    p = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    p.add_argument("--os-stage1-mode", choices=("full", "dummy_cached", "wrist_only"), default="full")
    p.add_argument("--os-pack-prefix", action="store_true")
    return p.parse_known_args(argv)


def miss_steps_from_yaml(path, model, live_steps=None):
    import yaml
    data = yaml.safe_load(pathlib.Path(path).read_text()) or {}
    miss = data.get("miss")
    steps = (10 if model == "pi05" else 8) if live_steps is None else live_steps
    if miss is not None:
        steps = miss.get("num_steps")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("miss.num_steps / denoising steps must be a positive integer")
    if model == "groot" and live_steps is not None and steps != live_steps:
        raise ValueError(f"GR00T bundle MISS K={steps} != live head K={live_steps}; set GROOT_DENOISING_STEPS")
    return steps


def install_startup_hook(plugin, *, stage1_mode, miss_steps, prefix_packing=False):
    """Own the documented startup fields without modifying plugin.py."""
    original = plugin.PluginRuntime.emit
    @functools.wraps(original)
    def emit(self, row):
        if row.get("ev") == "startup":
            row = dict(row, stage1_mode=stage1_mode, miss_steps=miss_steps,
                       prefix_packing=bool(prefix_packing))
        return original(self, row)
    plugin.PluginRuntime.emit = emit


def validate_method(opts, mode):
    if mode != "wrist_only":
        return
    if not opts.os_no_shadow_native or opts.os_tokens != "off":
        raise ValueError("wrist_only requires --os-no-shadow-native --os-tokens off")
    from exp.offline_search.closed_loop.plugin import load_method_class
    cls, _ = load_method_class(opts.os_method)
    if getattr(cls, "camera_mode", None) != "wrist_only":
        raise ValueError("wrist_only requires a method declaring camera_mode='wrist_only'")


def _types():
    # Lazy torch import: reduced-step startup logging does not need model imports.
    from openpi.models_pytorch.pi0_pytorch import Stage1Output
    @dataclasses.dataclass
    class DeferredStage1(Stage1Output):
        deferred_base: object = None

        def to(self, device):
            base = super().to(device)
            return DeferredStage1(**vars(base), deferred_base=self.deferred_base.to(device))
    return DeferredStage1


class Pi05Override:
    def __init__(self, mode="dummy_cached", pack_prefix=False):
        if mode not in ("full", "dummy_cached", "wrist_only"):
            raise ValueError(mode)
        self.mode, self.pack_prefix = mode, bool(pack_prefix)
        self.DeferredStage1 = _types()
        self._lock = threading.Lock()

    def _dummy(self, model, image, mask):
        import torch
        if bool(mask.any()) or not bool(torch.all(image == -1)):
            raise ValueError("dummy cache requires an all-masked, exactly -1 preprocessed third image")
        tower = model.paligemma_with_expert
        # Model-owned cache: never shared across weights, dtype/device changes,
        # or batch sizes (GEMM kernels may depend on B).
        params = list(tower.paligemma.vision_tower.parameters()) + list(
            tower.paligemma.multi_modal_projector.parameters())
        signature = tuple((p.data_ptr(), p._version, p.dtype, p.device) for p in params)
        shape_key = (tuple(image.shape), image.dtype, image.device)
        with self._lock:
            cached = getattr(model, "_r4_dummy_embedding", None)
            if cached is None or cached[0] != signature:
                cached = (signature, {})
                model._r4_dummy_embedding = cached
            if shape_key not in cached[1]:
                cached[1][shape_key] = tower.embed_image(image).detach()
            return cached[1][shape_key]

    def stage1(self, model, observation):
        import torch
        from openpi.models_pytorch.pi0_pytorch import Stage1Output, make_att_2d_masks
        if not model.pi05:
            raise ValueError("R4 overrides require pi05")
        images, masks, tokens, lang_mask, state = model._preprocess_observation(observation, train=False)
        if len(images) != 3 or list(observation.images) != ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]:
            raise ValueError("R4 overrides require the canonical LIBERO camera order")
        dummy = self._dummy(model, images[2], masks[2])
        if dummy.shape[1] != 256:
            raise ValueError("R4 overrides require 256 tokens per camera")
        tower = model.paligemma_with_expert
        wrist = tower.embed_image(images[1])
        base = torch.zeros_like(wrist) if self.mode == "wrist_only" else tower.embed_image(images[0])
        lang = tower.embed_language_tokens(tokens)
        lang = lang * math.sqrt(lang.shape[-1])
        prefix = torch.cat((base, wrist, dummy, lang), dim=1)
        pad = torch.cat([m[:, None].expand(image.shape[:2]) for m, image in zip(masks, (base, wrist, dummy))]
                        + [lang_mask], dim=1)
        att = make_att_2d_masks(pad, torch.zeros_like(pad))
        fields = dict(state=state, prefix_embs=prefix, prefix_pad_masks=pad,
                      prefix_att_2d_masks_4d=model._prepare_attention_masks_4d(att),
                      prefix_position_ids=torch.cumsum(pad, dim=1) - 1)
        if self.mode == "wrist_only":
            return self.DeferredStage1(**fields, deferred_base=images[0].detach().clone())
        return Stage1Output(**fields)

    def complete(self, model, stage1):
        import torch
        from openpi.models_pytorch.pi0_pytorch import Stage1Output
        deferred = getattr(stage1, "deferred_base", None)
        if deferred is None:
            return stage1
        base = model.paligemma_with_expert.embed_image(deferred)
        fields = {f.name: getattr(stage1, f.name) for f in dataclasses.fields(Stage1Output)}
        fields["prefix_embs"] = torch.cat((base, stage1.prefix_embs[:, 256:]), dim=1)
        return Stage1Output(**fields)

    @staticmethod
    def pack(stage1):
        import torch
        from openpi.models_pytorch.pi0_pytorch import Stage1Output
        if bool(stage1.prefix_pad_masks[:, 512:768].any()):
            raise ValueError("cannot pack an unmasked dummy camera")
        n = stage1.prefix_embs.shape[1]
        keep = torch.cat((torch.arange(512, device=stage1.state.device),
                          torch.arange(768, n, device=stage1.state.device)))
        return Stage1Output(state=stage1.state, prefix_embs=stage1.prefix_embs.index_select(1, keep),
                            prefix_pad_masks=stage1.prefix_pad_masks.index_select(1, keep),
                            prefix_att_2d_masks_4d=stage1.prefix_att_2d_masks_4d.index_select(2, keep).index_select(3, keep),
                            prefix_position_ids=stage1.prefix_position_ids.index_select(1, keep))

    def stage2_input(self, model, stage1):
        full = self.complete(model, stage1)
        return self.pack(full) if self.pack_prefix else full

    def install(self):
        from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
        from openpi.serving import stage_io
        if getattr(PI0Pytorch, "_r4_override", None) is not None:
            raise RuntimeError("R4 stage overrides already installed")
        PI0Pytorch._r4_override = self
        original1, original2, original_capture = PI0Pytorch.run_stage1, PI0Pytorch.run_stage2, PI0Pytorch.run_stage2_capture
        override = self
        def s1(model, obs):
            return original1(model, obs) if override.mode == "full" else override.stage1(model, obs)
        def s2(model, out):
            return original2(model, override.stage2_input(model, out))
        def capture(model, out):
            return original_capture(model, override.stage2_input(model, out))
        PI0Pytorch.run_stage1, PI0Pytorch.run_stage2, PI0Pytorch.run_stage2_capture = s1, s2, capture
        if self.mode == "wrist_only":
            old_split, old_stack = stage_io.split_stage1_output, stage_io.stack_stage1_output
            def split(out, n):
                shards = old_split(out, n)
                deferred = getattr(out, "deferred_base", None)
                if deferred is None:
                    return shards
                return [override.DeferredStage1(**vars(s), deferred_base=deferred[i:i+1].clone())
                        for i, s in enumerate(shards)]
            def stack(outputs):
                base = old_stack(outputs)
                ds = [getattr(o, "deferred_base", None) for o in outputs]
                if all(d is None for d in ds):
                    return base
                if any(d is None for d in ds):
                    raise ValueError("mixed completed and deferred stage1 batch")
                import torch
                return override.DeferredStage1(**vars(base), deferred_base=torch.cat(ds, dim=0))
            stage_io.split_stage1_output, stage_io.stack_stage1_output = split, stack
        return self


def install_pi05(mode="full", pack_prefix=False):
    if pack_prefix:
        raise ValueError("prefix packing is benchmark-only: bf16 same-noise action parity failed; do not deploy")
    if mode == "full" and not pack_prefix:
        return None
    return Pi05Override(mode, pack_prefix).install()
