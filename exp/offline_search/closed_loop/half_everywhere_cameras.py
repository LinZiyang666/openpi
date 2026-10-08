"""Cold wrist-first stage one: one real tower call, request-owned completion."""
from __future__ import annotations
import math
from .stage_overrides import Pi05Override, CameraRequest
from .lookhalf_cameras import GrootOverride


def receipt(request):
    request.audit.update(looks=1, initial_cameras=1 if request.mode=='wrist_only' else 2,
                         availability=[request.mode!='wrist_only',True], completions=0)


class HalfPi05Override(Pi05Override):
    def __init__(self):
        super().__init__('per_request')

    def stage1(self, model, observation, *, mode=None):
        import torch
        from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
        if mode != 'wrist_only':
            return super().stage1(model,observation,mode=mode)
        images,masks,tokens,lang_mask,state=model._preprocess_observation(observation,train=False)
        if (not model.pi05 or list(observation.images)!=['base_0_rgb','left_wrist_0_rgb','right_wrist_0_rgb']
                or len(images)!=3 or bool(masks[2].any()) or not bool(torch.all(images[2]==-1))):
            raise ValueError('half-everywhere requires canonical LIBERO masked dummy layout')
        tower=model.paligemma_with_expert
        wrist=tower.embed_image(images[1])
        if wrist.shape[0]!=1 or wrist.shape[1]!=256:
            raise ValueError('half-everywhere requires B=1, 256 camera tokens')
        # Permanently masked dummy slots are zero as in CacheAblate. This also
        # works at step zero before any full request could seed a dummy cache.
        views=[torch.zeros_like(wrist),wrist,torch.zeros_like(wrist)]
        lang=tower.embed_language_tokens(tokens)*math.sqrt(wrist.shape[-1])
        prefix=torch.cat([*views,lang],dim=1)
        pad=torch.cat([m[:,None].expand(v.shape[:2]) for m,v in zip(masks,views)]+[lang_mask],dim=1)
        att=make_att_2d_masks(pad,torch.zeros_like(pad))
        return self.DeferredStage1(state=state,prefix_embs=prefix,prefix_pad_masks=pad,
            prefix_att_2d_masks_4d=model._prepare_attention_masks_4d(att),
            prefix_position_ids=torch.cumsum(pad,dim=1)-1,deferred_base=images[0].detach().clone())

    def install(self):
        super().install()
        from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
        from openpi.serving.batching_coordinator import Pi05StageBatcher
        eager,batch=PI0Pytorch.run_stage1,Pi05StageBatcher.run_stage1_batch
        def stage1(model,request):
            out=eager(model,request)
            if isinstance(request,CameraRequest) and request.audit is not None:
                receipt(request)
            return out
        def stage1_batch(batcher,requests):
            outputs=batch(batcher,requests)
            for request in requests:
                if isinstance(request,CameraRequest) and request.audit is not None:
                    receipt(request)
            return outputs
        PI0Pytorch.run_stage1=stage1
        Pi05StageBatcher.run_stage1_batch=stage1_batch
        return self


def install(model):
    return HalfPi05Override().install() if model=='pi05' else GrootOverride().install()
