"""Ablation-only stage one: encode the requested cameras; never complete."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class CameraRequest:
    observation: object
    mode: str
    audit: object

    def __post_init__(self):
        if self.mode not in ('full', 'wrist_only', 'agent_only'):
            raise ValueError(self.mode)


def receipt(request):
    available = [request.mode != 'wrist_only', request.mode != 'agent_only']
    request.audit.update(looks=1, initial_cameras=sum(available), availability=available,
                         completions=0)


def pi05_partial(model, request):
    import torch
    from openpi.models_pytorch.pi0_pytorch import Stage1Output, make_att_2d_masks
    obs = request.observation
    images, masks, tokens, lang_mask, state = model._preprocess_observation(obs, train=False)
    if not model.pi05 or list(obs.images) != ['base_0_rgb', 'left_wrist_0_rgb', 'right_wrist_0_rgb']:
        raise ValueError('ablation requires canonical pi05 LIBERO camera order')
    if len(images) != 3 or bool(masks[2].any()) or not bool(torch.all(images[2] == -1)):
        raise ValueError('ablation requires the masked LIBERO dummy slot')
    camera = 1 if request.mode == 'wrist_only' else 0
    tower = model.paligemma_with_expert
    encoded = tower.embed_image(images[camera])
    if encoded.shape[0] != 1 or encoded.shape[1] != 256:
        raise ValueError('ablation requires B=1, 256 tokens per camera')
    zeros = torch.zeros_like(encoded)
    # Dummy slots never enter retrieval or policy and need no tower invocation.
    views = [zeros, zeros, zeros]
    views[camera] = encoded
    lang = tower.embed_language_tokens(tokens)
    prefix = torch.cat([*views, lang*math.sqrt(lang.shape[-1])], dim=1)
    pad = torch.cat([m[:, None].expand(v.shape[:2]) for m, v in zip(masks, views)] + [lang_mask], dim=1)
    att = make_att_2d_masks(pad, torch.zeros_like(pad))
    out = Stage1Output(state=state, prefix_embs=prefix, prefix_pad_masks=pad,
                      prefix_att_2d_masks_4d=model._prepare_attention_masks_4d(att),
                      prefix_position_ids=torch.cumsum(pad, dim=1)-1)
    receipt(request)
    return out


def groot_partial(runner, request):
    import torch
    from .lookhalf_cameras import GrootOverride
    from openpi.cache.groot.staged import GrootStage1Output
    from openpi.cache.groot.key_builder import _contiguous_runs
    inputs, action = GrootOverride.prepare(runner, request.observation)
    eagle = runner._eagle
    embeds = eagle.language_model.get_input_embeddings()(inputs['input_ids'])
    mask = inputs['input_ids'] == eagle.image_token_index
    positions = torch.nonzero(mask.reshape(-1)).reshape(-1)
    if len(positions) != 512 or [n for _, n in _contiguous_runs(mask[0])] != [256, 256]:
        raise ValueError('unexpected GR00T camera token layout')
    camera = 1 if request.mode == 'wrist_only' else 0
    encoded = eagle.extract_feature(inputs['pixel_values'][camera:camera+1])
    if encoded.shape != (1, 256, embeds.shape[-1]):
        raise ValueError('unexpected GR00T single-camera shape')
    flat = embeds.reshape(-1, embeds.shape[-1])
    flat[positions] = 0
    flat[positions[camera*256:(camera+1)*256]] = encoded.reshape(-1, embeds.shape[-1])
    receipt(request)
    return GrootStage1Output(embeds, inputs['attention_mask'], mask, action)


def install(model):
    """Installed exclusively by --os-cache-ablate; legacy overrides are untouched."""
    if model == 'pi05':
        from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
        from openpi.serving.batching_coordinator import Pi05StageBatcher
        from openpi.serving import stage_io
        if getattr(PI0Pytorch, '_r4_override', None) is not None:
            raise RuntimeError('ablation cannot share a legacy camera override')
        original = PI0Pytorch.run_stage1
        def stage1(model, request):
            if not isinstance(request, CameraRequest):
                return original(model, request)
            if request.mode != 'full':
                return pi05_partial(model, request)
            out = original(model, request.observation)
            receipt(request)
            return out
        PI0Pytorch.run_stage1 = stage1
        old_batch = Pi05StageBatcher.run_stage1_batch
        def batch(batcher, payloads):
            outputs = []
            for request in payloads:
                if not isinstance(request, CameraRequest):
                    outputs.extend(old_batch(batcher, [request]))
                elif request.mode == 'full':
                    outputs.extend(old_batch(batcher, [request.observation]))
                    receipt(request)
                else:
                    from openpi.models.model import Observation
                    obs = stage_io.stack_observation([request.observation], device=batcher._device)
                    if isinstance(obs, dict) and 'image' in obs:
                        obs = Observation.from_dict(obs)
                    outputs.append(pi05_partial(batcher._model, CameraRequest(obs, request.mode, request.audit)))
            return outputs
        Pi05StageBatcher.run_stage1_batch = batch
    else:
        from openpi.cache.groot.staged import GrootStagedRunner
        if getattr(GrootStagedRunner, '_r12_override', None) is not None:
            raise RuntimeError('ablation cannot share a look-half override')
        original = GrootStagedRunner.run_stage1
        def stage1(runner, request):
            if not isinstance(request, CameraRequest):
                return original(runner, request)
            if request.mode != 'full':
                return groot_partial(runner, request)
            out = original(runner, request.observation)
            receipt(request)
            return out
        GrootStagedRunner.run_stage1 = stage1


def ledger(model, selected, *, vision, work, hit):
    if not hit:
        raise RuntimeError('PURE_CACHE_ABLATION_POLICY_CALL_REFUSED')
    count = 2 if selected == 'both' else 1
    availability = ([True, True] if selected == 'both' else
                    [False, True] if selected.startswith('wrist') else [True, False])
    if vision and (work.get('looks') != 1 or work.get('initial_cameras') != count
                   or work.get('availability') != availability or work.get('completions') != 0):
        raise RuntimeError('PURE_CACHE_ABLATION_CAMERA_RECEIPT_REQUIRED')
    cost = (.076 if model == 'pi05' else .074)*count if vision else 0.
    return dict(camera_initial_count=count if vision else 0, camera_encoded_count=count if vision else 0,
                camera_completion_count=0, look_cost=cost, completion_cost=0., policy_cost=0.,
                compute_share_work=cost)
