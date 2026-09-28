"""Standalone capture-safe copy of pi05 stage1, without patching model methods.

Stock embed_prefix constructs an all-zero GPU tensor from a Python list, which
raises 'operation not permitted when stream is capturing'. Here that constant
is zeros_like(pad). Image towers, language embedding, arithmetic and all output
fields follow PI0Pytorch.run_stage1 exactly. No dummy-camera optimization.
"""
import math
import torch
from torch import nn
from openpi.models_pytorch.pi0_pytorch import Stage1Output,make_att_2d_masks

class Stage1Adapter(nn.Module):
    def __init__(self,model):
        super().__init__();self.model=model

    def forward(self,observation):
        m=self.model
        images,masks,lang,lang_mask,state=m._preprocess_observation(observation,train=False)
        embs=[];pads=[]
        for img,mask in zip(images,masks,strict=True):
            e=m.paligemma_with_expert.embed_image(img)
            embs.append(e);pads.append(mask[:,None].expand(e.shape[0],e.shape[1]))
        e=m.paligemma_with_expert.embed_language_tokens(lang)
        embs.append(e*math.sqrt(e.shape[-1]));pads.append(lang_mask)
        prefix=torch.cat(embs,1);pad=torch.cat(pads,1)
        att=torch.zeros_like(pad)
        att4=m._prepare_attention_masks_4d(make_att_2d_masks(pad,att))
        pos=torch.cumsum(pad,1)-1
        return Stage1Output(state,prefix,pad,att4,pos)
