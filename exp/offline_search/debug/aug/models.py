"""Model adapters using the same transforms/stages as closed_loop/serve_*.

No server/coordinator thread is started. Noise is explicit and comes from one
private generator per decision/draw. Offline GR00T uses eager, shape-bucketed
stages 1/2/3 on one model thread; it never shares serving CUDA-graph buffers.
"""
import contextlib
import dataclasses
import hashlib
import json
from pathlib import Path
import threading
import time

import numpy as np


def input_sha(observation):
    digest = hashlib.sha256()
    for key in sorted(observation):
        digest.update(key.encode() + b"\0")
        value = observation[key]
        if isinstance(value, str):
            raw = value.encode("utf-8")
            digest.update(str(len(raw)).encode() + b":" + raw)
        else:
            array = np.ascontiguousarray(value)
            digest.update(json.dumps([array.dtype.str, array.shape]).encode() + b"\0" + array.tobytes())
    return digest.hexdigest()


def _array(tensor):
    return tensor.detach().cpu().float().numpy().copy()


class Profile:
    """CPU wall intervals and optional CUDA events; totals may overlap."""
    def __init__(self):
        self.wall, self.calls, self.events = {}, {}, []
        self.lock = threading.Lock()

    @contextlib.contextmanager
    def measure(self, name, device=None):
        events = None
        if device is not None and str(device).startswith("cuda"):
            import torch
            with torch.cuda.device(device):
                events = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                events[0].record()
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            if events:
                events[1].record()
            with self.lock:
                self.wall[name] = self.wall.get(name, 0.) + elapsed
                self.calls[name] = self.calls.get(name, 0) + 1
                if events:
                    self.events.append((name, events))

    def report(self):
        # Caller synchronizes the model once before reading these events.
        gpu = {}
        for name, (start, end) in self.events:
            gpu[name] = gpu.get(name, 0.) + start.elapsed_time(end) / 1000.
        return dict(wall_seconds=dict(self.wall), calls=dict(self.calls), gpu_seconds=gpu,
                    note="Wall intervals include enqueue/transfer waits; pipeline intervals overlap. CUDA events measure stream time, including contention.")


def measure(model, name, gpu=False):
    profile = getattr(model, "profile", None)
    return profile.measure(name, getattr(model, "device", None) if gpu else None) if profile else contextlib.nullcontext()


def _keys(model, outputs, builder_class):
    """Production slicing/reduction, with three batch transfers instead of 3B."""
    import torch
    from openpi.cache.types import CheckpointID
    fields = {"v0": [], "v1": [], "state": []}
    with measure(model, "key_build", gpu=True), torch.inference_mode():
        for out in outputs:
            builder = builder_class(enabled_fields=["vision_0", "vision_1", "robot_state"])
            builder.collect(CheckpointID.CP1, stage1=out)
            raw = builder._slice()
            fields["v0"].append(builder._reduce_vision(raw["vision_0"]))
            fields["v1"].append(builder._reduce_vision(raw["vision_1"]))
            fields["state"].append(raw["robot_state"].reshape(-1))
        return {name: _array(torch.stack(values)) for name, values in fields.items()}


class FakeModel:
    """CPU fixture adapter with private RNG and observable batch dispatches."""
    def __init__(self, model="pi05", horizon=None, action_dim=32, key_dim=128):
        self.model = model
        self.H = horizon or (10 if model == "pi05" else 16)
        self.action_dim = action_dim
        self.key_dim = key_dim
        self.encode_batches, self.draw_batches = [], []

    def encode(self, observations):
        self.encode_batches.append(len(observations))
        keys0, keys1, states = [], [], []
        for obs in observations:
            seed = int(input_sha(obs)[:16], 16)
            rng = np.random.default_rng(seed)
            keys0.append(rng.standard_normal(self.key_dim).astype(np.float32))
            keys1.append(rng.standard_normal(self.key_dim).astype(np.float32))
            states.append(np.asarray(obs["observation/state"], np.float32))
        return dict(v0=np.stack(keys0), v1=np.stack(keys1), state=np.stack(states), observations=observations)

    def prepare(self, observations):
        return observations

    def encode_prepared(self, prepared):
        with measure(self, "stage1", gpu=True):
            return self.encode(prepared)

    def select(self, encoded, positions):
        return {k: encoded[k][positions] for k in ("v0", "v1", "state")}

    def draw(self, encoded, seeds):
        self.draw_batches.append(len(seeds))
        with measure(self, "stage3." + getattr(self, "kind", "policy_shadow"), gpu=True):
            return np.stack([np.random.default_rng(int(seed)).standard_normal((self.H, self.action_dim)).astype(np.float32)
                             + encoded["state"][i].mean() for i, seed in enumerate(seeds)])

    def synchronize(self):
        pass


class Pi05Model:
    model = "pi05"
    def __init__(self, checkpoint, device="cuda:0", steps=10):
        if not Path(checkpoint).is_dir():
            raise ValueError("local checkpoint directory required")
        import torch
        from openpi.training import config
        from openpi.policies import policy_config
        from openpi.serving.batching_coordinator import Pi05StageBatcher
        cfg = config.get_config("pi05_libero")
        cfg = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, pytorch_compile_mode=None))
        self.policy = policy_config.create_trained_policy(cfg, str(checkpoint), pytorch_device=device)
        self._model = self.policy._model.eval()
        self.device, self.steps = torch.device(device), steps
        self.batcher = Pi05StageBatcher(self._model, self.device)
        self.H, self.action_dim = self._model.config.action_horizon, self._model.config.action_dim

    def prepare(self, observations):
        with measure(self, "transform"):
            return [self.policy._input_transform(dict(obs)) for obs in observations]

    def encode(self, observations):
        return self.encode_prepared(self.prepare(observations))

    def encode_prepared(self, payloads):
        import torch
        from openpi.cache.components.key_builder import CP1SpatialPool16KeyBuilder
        with measure(self, "stage1", gpu=True), torch.inference_mode():
            stage1 = self.batcher.run_stage1_batch(payloads)
        return dict(_keys(self, stage1, CP1SpatialPool16KeyBuilder), stage1=stage1, stage2=None)

    def condition(self, encoded):
        import torch
        from openpi.serving import stage_io
        if encoded["stage2"] is None:
            with measure(self, "stage2", gpu=True), torch.inference_mode():
                encoded["stage2"] = self._model.run_stage2(stage_io.stack_stage1_output(encoded["stage1"]))

    def select(self, encoded, positions):
        from openpi.serving import stage_io
        self.condition(encoded)
        if len(positions) == len(encoded["stage1"]) and positions == list(range(len(positions))):
            return encoded
        if "stage2_shards" not in encoded:
            with measure(self, "conditioning_select", gpu=True):
                encoded["stage2_shards"] = stage_io.split_stage2_output(encoded["stage2"], len(encoded["stage1"]))
        chosen = stage_io.stack_stage2_output([encoded["stage2_shards"][i] for i in positions])
        return dict(stage1=[encoded["stage1"][i] for i in positions], stage2=chosen)

    def draw(self, encoded, seeds):
        import torch
        self.condition(encoded)
        with torch.inference_mode():
            noise = torch.cat([self._model.sample_noise((1, self.H, self.action_dim), self.device,
                                                      generator=torch.Generator(device=self.device).manual_seed(int(seed)))
                               for seed in seeds], dim=0)
            with measure(self, "stage3." + getattr(self, "kind", "policy_shadow"), gpu=True):
                output = self._model.run_stage3(encoded["stage2"], noise=noise, num_steps=self.steps)
        return _array(output.action_chunk if hasattr(output, "action_chunk") else output)

    def synchronize(self):
        import torch
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)


class GrootModel:
    model = "groot"
    def __init__(self, checkpoint, device="cuda:0", steps=8, stage_mode="batched"):
        if not Path(checkpoint).is_dir():
            raise ValueError("local checkpoint directory required")
        from gr00t.model.policy import Gr00tPolicy
        from custom_data_config import LiberoDataConfig
        from exp.libero_groot.serve_groot_libero import EMBODIMENT_TAG
        from openpi.cache.groot.staged import GrootStagedRunner
        data = LiberoDataConfig()
        self.policy = Gr00tPolicy(model_path=str(checkpoint), embodiment_tag=EMBODIMENT_TAG,
                                  modality_config=data.modality_config(), modality_transform=data.transform(),
                                  denoising_steps=steps, device=device)
        self.policy.model.eval()
        self.runner = GrootStagedRunner(self.policy.model)
        self.device = device
        self.stage_mode = stage_mode
        head = self.policy.model.action_head
        from openpi.cache.groot.staged import _action_shape
        self.H, self.action_dim = _action_shape(head)

    def prepare(self, observations):
        from exp.libero_groot.policy_adapter import build_groot_observation
        from openpi.cache.groot.interceptor import _unsqueeze_values
        result = []
        with measure(self, "transform"):
            for observation in observations:
                inputs = _unsqueeze_values(build_groot_observation(observation))
                inputs = {k: v if isinstance(v, np.ndarray) else np.array(v) for k, v in inputs.items()}
                result.append(self.policy.apply_transforms(inputs))
        return result

    def encode(self, observations):
        return self.encode_prepared(self.prepare(observations))

    def encode_prepared(self, normalized):
        from openpi.cache.groot.key_builder import GrootLiberoCP1SpatialPool16KeyBuilder
        with measure(self, "stage1", gpu=True), self.runner.session():
            if getattr(self, "stage_mode", "batched") == "serial":
                stage1 = [self.runner.run_stage1(norm) for norm in normalized]
            else:
                stage1 = self._stage1_batch(normalized)
        return dict(_keys(self, stage1, GrootLiberoCP1SpatialPool16KeyBuilder), stage1=stage1, stage2=None)

    def _bucket(self, stage1):
        """Reuse the production conditioning bucket, before the LLM projection."""
        from openpi.cache.groot.batcher import stage3_input, GrootStageBatcher
        from openpi.cache.groot.staged import GrootStage2Output
        from openpi.serving.batching_core import Stage3MissPayload
        proxy = GrootStage2Output(backbone_features=stage1.input_embeds, attention_mask=stage1.attention_mask,
                                 action_inputs=stage1.action_inputs)
        cond = stage3_input(self.runner, proxy)
        # bucket_key only inspects the noise dtype; no noise is sampled here.
        return GrootStageBatcher.bucket_key(Stage3MissPayload(stage2_out=cond, noise=stage1.input_embeds,
                                                            num_steps=self.runner.live_schedule().num_steps))

    def _stage1_batch(self, normalized):
        """Pinned eager run_stage1 transcription, no tokenizer padding.

        prepare_input remains per row. Pixel groups are concatenated in row
        order; selected image-token counts are verified independently per row.
        Production runner verifies upstream source on construction.
        """
        import torch
        from openpi.cache.groot.staged import GrootStage1Output, _EAGLE_INPUT_KEYS, _batch_feature
        if self.runner._compiled_entry is not None:
            raise RuntimeError("offline stage1 batching requires eager vision")
        groups, result = {}, [None] * len(normalized)
        for i, norm in enumerate(normalized):
            backbone, action = self.runner._model.prepare_input(norm)
            eagle = {k[len("eagle_"):]: v for k, v in backbone.items() if k.startswith("eagle_")}
            eagle.pop("image_sizes", None)
            if set(eagle) != _EAGLE_INPUT_KEYS or eagle["input_ids"].shape[0] != 1:
                raise RuntimeError("unexpected Eagle inputs or non-unit transformed row")
            ids = eagle["input_ids"]
            # Shape-only stand-in for embeddings; avoids running the encoder
            # merely to choose a bucket. All rows share one model width.
            proxy = GrootStage1Output(input_embeds=ids.unsqueeze(-1), attention_mask=eagle["attention_mask"],
                                     image_token_mask=ids == self.runner._eagle.image_token_index, action_inputs=action)
            tensors = tuple((k, tuple(v.shape), str(v.dtype), str(v.device)) for k, v in sorted(eagle.items()))
            action_shapes = tuple((k, tuple(v.shape[1:]), str(v.dtype), str(v.device)) if torch.is_tensor(v)
                                  else (k, repr(v)) for k, v in sorted(action.items()))
            key = (self._bucket(proxy), tensors, action_shapes)
            groups.setdefault(key, []).append((i, eagle, action))
        for items in groups.values():
            inputs = {k: torch.cat([eagle[k] for _, eagle, _ in items], dim=0) for k in _EAGLE_INPUT_KEYS}
            input_embeds = self.runner._eagle.language_model.get_input_embeddings()(inputs["input_ids"])
            vit = self.runner._eagle.extract_feature(inputs["pixel_values"])
            b, n, c = input_embeds.shape
            selected = inputs["input_ids"] == self.runner._eagle.image_token_index
            # Each unbatched normalized row has the same pixel shape. Reject
            # a changed token layout rather than letting rows share features.
            tokens_per_row = (vit.shape[0] // b) * vit.shape[1]
            if vit.shape[0] % b or not torch.all(selected.sum(dim=1) == tokens_per_row):
                raise RuntimeError("image-token count differs from vision features per row")
            flat = input_embeds.reshape(b * n, c)
            mask = selected.reshape(-1)
            flat[mask] = flat[mask] * 0.0 + vit.reshape(-1, c)
            input_embeds = flat.reshape(b, n, c)
            for j, (i, _, action) in enumerate(items):
                result[i] = GrootStage1Output(input_embeds=input_embeds[j:j + 1],
                    attention_mask=inputs["attention_mask"][j:j + 1], image_token_mask=selected[j:j + 1],
                    action_inputs=_batch_feature(action))
        return result

    def condition(self, encoded):
        import torch
        from openpi.cache.groot.batcher import cat_stage2, stage3_input
        from openpi.cache.groot.staged import GrootStage1Output, GrootStage2Output, _batch_feature
        if encoded["stage2"] is not None:
            return
        groups, outputs = {}, [None] * len(encoded["stage1"])
        for i, out in enumerate(encoded["stage1"]):
            key = i if getattr(self, "stage_mode", "batched") == "serial" else self._bucket(out)
            groups.setdefault(key, []).append((i, out))
        with measure(self, "stage2", gpu=True), self.runner.session():
            for items in groups.values():
                proxies = [stage3_input(self.runner, GrootStage2Output(backbone_features=o.input_embeds,
                            attention_mask=o.attention_mask, action_inputs=o.action_inputs)) for _, o in items]
                combined = cat_stage2(proxies)
                s1 = GrootStage1Output(input_embeds=combined.backbone_features, attention_mask=combined.attention_mask,
                                      image_token_mask=torch.cat([o.image_token_mask for _, o in items]),
                                      action_inputs=combined.action_inputs)
                out = self.runner.run_stage2_llm(s1)
                for j, (i, _) in enumerate(items):
                    action = {k: v[j:j + 1] if torch.is_tensor(v) else v for k, v in out.action_inputs.items()}
                    outputs[i] = GrootStage2Output(backbone_features=out.backbone_features[j:j + 1],
                        attention_mask=out.attention_mask[j:j + 1], action_inputs=_batch_feature(action))
        encoded["stage2"] = outputs

    def select(self, encoded, positions):
        self.condition(encoded)
        self._conditioning_inputs(encoded)
        return dict(stage1=[encoded["stage1"][i] for i in positions],
                    stage2=[encoded["stage2"][i] for i in positions],
                    stage3_inputs=[encoded["stage3_inputs"][i] for i in positions])

    def _conditioning_inputs(self, encoded):
        from openpi.cache.groot.batcher import stage3_input
        if "stage3_inputs" not in encoded:
            encoded["stage3_inputs"] = [stage3_input(self.runner, out) for out in encoded["stage2"]]

    def draw(self, encoded, seeds):
        import torch
        from openpi.cache.groot.batcher import cat_stage2, GrootStageBatcher
        from openpi.cache.groot.staged import _processed_features
        from openpi.serving.batching_core import Stage3MissPayload
        self.condition(encoded)
        self._conditioning_inputs(encoded)
        groups = {}
        for i, out in enumerate(encoded["stage2"]):
            cond = encoded["stage3_inputs"][i]
            with self.runner.session():
                if hasattr(self.runner, "_noise_dtype"):
                    dtype = self.runner._noise_dtype(out)
                else:  # CPU protocol fake
                    head = self.policy.model.action_head
                    dtype = _processed_features(head.process_backbone_output(self.runner._head_inputs(out))).dtype
                device = out.backbone_features.device
                noise = torch.randn((1, self.H, self.action_dim), device=device, dtype=dtype,
                                    generator=torch.Generator(device=device).manual_seed(int(seeds[i])))
            # Production shape/embodiment bucket; never pad language features.
            key = GrootStageBatcher.bucket_key(Stage3MissPayload(stage2_out=cond, noise=noise,
                                                                num_steps=self.runner.live_schedule().num_steps))
            groups.setdefault(key, []).append((i, cond, noise))
        result = np.empty((len(seeds), self.H, self.action_dim), np.float32)
        for items in groups.values():
            combined = cat_stage2([cond for _, cond, _ in items])
            with measure(self, "stage3." + getattr(self, "kind", "policy_shadow"), gpu=True), self.runner.session():
                noise = torch.cat([noise for _, _, noise in items], dim=0)
                output = self.runner.run_stage3(combined, noise=noise)
            chunks = _array(output.action_pred)
            for j, (i, _, _) in enumerate(items):
                result[i] = chunks[j]
        return result

    def synchronize(self):
        import torch
        if str(self.device).startswith("cuda"):
            torch.cuda.synchronize(self.device)
