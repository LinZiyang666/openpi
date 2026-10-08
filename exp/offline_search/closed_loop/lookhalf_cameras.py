"""Request-owned genuine single-camera acquisition and idempotent completion."""
from __future__ import annotations
import dataclasses
import numpy as np

from .stage_overrides import CameraRequest


class GrootOverride:
    def __init__(self):
        self.original1 = None

    @staticmethod
    def prepare(runner, observation):
        runner._require_session('lookhalf stage1')
        backbone, action = runner._model.prepare_input(observation)
        inputs = {k.removeprefix('eagle_'): v for k, v in backbone.items() if k.startswith('eagle_')}
        inputs.pop('image_sizes', None)
        if set(inputs) != {'input_ids', 'attention_mask', 'pixel_values'}:
            raise ValueError('unexpected GR00T encoder inputs')
        if inputs['input_ids'].shape[0] != 1 or inputs['pixel_values'].shape[0] != 2:
            raise ValueError('R12 GR00T requires LIBERO two-camera B=1 layout')
        return inputs, action

    def partial(self, runner, observation, audit=None):
        import torch
        from openpi.cache.groot.staged import GrootStage1Output
        inputs, action = self.prepare(runner, observation)
        eagle = runner._eagle
        embeds = eagle.language_model.get_input_embeddings()(inputs['input_ids'])
        mask = inputs['input_ids'] == eagle.image_token_index
        positions = torch.nonzero(mask.reshape(-1)).reshape(-1)
        from openpi.cache.groot.key_builder import _contiguous_runs
        if len(positions) != 512 or [n for _, n in _contiguous_runs(mask[0])] != [256, 256]:
            raise ValueError('unexpected GR00T camera token layout')
        pixels = inputs['pixel_values']
        wrist = eagle.extract_feature(pixels[1:2])
        if wrist.shape != (1, 256, embeds.shape[-1]):
            raise ValueError('unexpected wrist embedding shape')
        flat = embeds.reshape(-1, embeds.shape[-1])
        flat[positions] = 0
        flat[positions[256:]] = wrist.reshape(-1, embeds.shape[-1])
        out = GrootStage1Output(embeds, inputs['attention_mask'], mask, action)
        # Fields belong to this request, never to the shared runner/model.
        out.deferred_base = pixels[:1].detach().clone()
        out.deferred_positions = positions[:256].clone()
        out.camera_audit = audit
        if audit is not None:
            audit['looks'] = audit.get('looks', 0)+1
            audit['initial_cameras'] = 1
            audit['availability'] = [False, True]
        return out

    def complete(self, runner, out):
        pixels = getattr(out, 'deferred_base', None)
        if pixels is None:
            return out
        runner._require_session('lookhalf completion')
        tokens = runner._eagle.extract_feature(pixels)
        # Update the very object owned by the interceptor; stage 2 sees the
        # completed prefix even when retrieval requested the second camera.
        out.input_embeds.reshape(-1, out.input_embeds.shape[-1])[out.deferred_positions] = tokens.reshape(-1, tokens.shape[-1])
        out.deferred_base = None
        audit = getattr(out, 'camera_audit', None)
        if audit is not None:
            audit['completions'] = audit.get('completions', 0)+1
            audit['availability'] = [True, True]
        return out

    def install(self):
        from openpi.cache.groot.staged import GrootStagedRunner
        if getattr(GrootStagedRunner, '_r12_override', None) is not None:
            raise RuntimeError('GR00T R12 already installed')
        GrootStagedRunner._r12_override = self
        self.original1 = GrootStagedRunner.run_stage1
        original2 = GrootStagedRunner.run_stage2_llm
        override = self
        def stage1(runner, request):
            if not isinstance(request, CameraRequest):
                return override.original1(runner, request)
            if request.mode == 'wrist_only':
                return override.partial(runner, request.observation, request.audit)
            out = override.original1(runner, request.observation)
            if request.audit is not None:
                request.audit.update(looks=1, initial_cameras=2, availability=[True, True])
            return out
        def stage2(runner, out):
            return original2(runner, override.complete(runner, out))
        GrootStagedRunner.run_stage1 = stage1
        GrootStagedRunner.run_stage2_llm = stage2
        return self


def complete_session(session, ctx=None, *, reason='call'):
    """Complete once, refresh the live key builder and existing history row."""
    import torch
    from openpi.cache.types import CheckpointID
    adapter = session._camera_adapter
    out = session._camera_stage1
    if out is None:
        raise RuntimeError('completion without request-owned stage1')
    with torch.inference_mode():
        if session.rt.model == 'pi05':
            model = adapter.policy._model
            override = model._r4_override
            completed = override.complete(model, out)
            if completed is not out:
                # The coordinator and interceptor still reference this object.
                for field in dataclasses.fields(completed):
                    setattr(out, field.name, getattr(completed, field.name))
                out.deferred_base = None
                out.deferred_mask = None
        else:
            runner = adapter.policy._runner
            with runner.session():
                runner._r12_override.complete(runner, out)
    session._camera_work['availability'] = [True, True]
    session._camera_work.setdefault('completion_reason', reason)
    session._look_stage1 = out
    session.kb.collect(CheckpointID.CP1, stage1=out)
    keys = session.kb.build(CheckpointID.CP1)
    step = session.step
    # Called before on_search advances the step; MISS completion supplies the
    # just-committed decision index explicitly through the decision object.
    if ctx is None and session._dec is not None:
        step = session._dec['step']
    for field, buf in (('vision_0', session.b_v0), ('vision_1', session.b_v1)):
        value = keys[field].detach().float().cpu().numpy() if hasattr(keys[field], 'detach') else keys[field]
        buf.a[step] = np.asarray(value, np.float32).reshape(-1)
    if ctx is not None:
        ctx.query_keys.update(keys)
    return keys


def ledger(model, *, vision, wrist_first, completions, miss):
    if completions not in (0, 1) or (completions and (not vision or not wrist_first)):
        raise ValueError('at most one missing-camera completion per look')
    if miss and (not vision or (wrist_first and completions != 1)):
        raise ValueError('a policy call must have both cameras')
    c = .076 if model == 'pi05' else .074
    policy = .848 if model == 'pi05' else .852
    initial = (1 if wrist_first else 2) if vision else 0
    completed = completions if vision else 0
    look = c*initial
    completion = c*completed
    remainder = policy if vision and miss else 0.
    # Legacy R4 pi05 ledger is retained for comparability, without calling
    # its request-latency prices the owner's compute-share price.
    legacy_look = (.055198 if wrist_first else .152) if model == 'pi05' else (c if wrist_first else 2*c)
    legacy_completion = .049890 if model == 'pi05' else c
    legacy = (legacy_look+legacy_completion*completed+remainder) if vision else 0.
    return dict(camera_initial_count=initial, camera_completion_count=completed,
                camera_encoded_count=initial+completed, look_cost=look, completion_cost=completion,
                policy_cost=remainder, compute_share_work=look+completion+remainder,
                legacy_work=legacy)
