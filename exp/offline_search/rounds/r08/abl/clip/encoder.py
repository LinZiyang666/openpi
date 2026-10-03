"""OpenAI CLIP ViT-B/32 image tower using only torch, PIL and safetensors.

OpenAI uses QuickGELU, not the ordinary GELU in open_clip's untrained
ViT-B-32 defaults. No package installation or network download is needed.
"""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import os
from pathlib import Path
import threading

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F

WEIGHT_SHA256 = 'e6d1bd7789aa45192b3bf90570a789b478bae1b74ebcce7eddd908e83a2b7c31'
MEAN = (0.48145466, 0.4578275, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)
RECIPE = 'openai_vitb32_quickgelu_f32_pil_bicubic_short224_centercrop_l2_v1'


def weights_path():
    explicit = os.environ.get('R8_CLIP_WEIGHTS')
    if explicit:
        return Path(explicit)
    hub = Path.home()/'.cache/huggingface/hub/models--timm--vit_base_patch32_clip_224.openai'
    found = sorted(hub.glob('snapshots/*/open_clip_model.safetensors'))
    if not found:
        raise FileNotFoundError('Set R8_CLIP_WEIGHTS to the cached OpenAI open_clip_model.safetensors')
    return found[0]


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def preprocess(image):
    """Exactly torchvision PIL Resize(224, BICUBIC), CenterCrop, ToTensor, Normalize."""
    a = np.asarray(image)
    if a.dtype != np.uint8 or a.ndim != 3 or a.shape[-1] != 3:
        raise ValueError(f'expected HWC RGB uint8, got {a.shape}, {a.dtype}')
    im = Image.fromarray(a).convert('RGB')
    w, h = im.size
    if min(w, h) != 224:
        size = (224, int(224*h/w)) if w <= h else (int(224*w/h), 224)
        im = im.resize(size, resample=Image.Resampling.BICUBIC)
    w, h = im.size
    left, top = int(round((w-224)/2)), int(round((h-224)/2))
    im = im.crop((left, top, left+224, top+224))
    x = torch.from_numpy(np.array(im, copy=True)).permute(2,0,1).contiguous().float().div(255)
    return (x - torch.tensor(MEAN,dtype=torch.float32)[:,None,None]) / torch.tensor(STD,dtype=torch.float32)[:,None,None]


class QuickGELU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.702*x)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln_1 = nn.LayerNorm(768)
        self.attn = nn.MultiheadAttention(768, 12, batch_first=True)
        self.ln_2 = nn.LayerNorm(768)
        self.mlp = nn.Sequential(OrderedDict([
            ('c_fc', nn.Linear(768,3072)), ('gelu', QuickGELU()), ('c_proj', nn.Linear(3072,768))]))

    def forward(self, x):
        y = self.ln_1(x)
        x = x + self.attn(y,y,y,need_weights=False)[0]
        return x + self.mlp(self.ln_2(x))


class VisionTower(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,768,32,stride=32,bias=False)
        self.class_embedding = nn.Parameter(torch.empty(768))
        self.positional_embedding = nn.Parameter(torch.empty(50,768))
        self.ln_pre = nn.LayerNorm(768)
        self.transformer = nn.Module()
        self.transformer.resblocks = nn.ModuleList([Block() for _ in range(12)])
        self.ln_post = nn.LayerNorm(768)
        self.proj = nn.Parameter(torch.empty(768,512))

    def forward(self, x):
        x = self.conv1(x).reshape(len(x),768,-1).permute(0,2,1)
        cls = self.class_embedding.reshape(1,1,-1).expand(len(x),1,-1)
        x = self.ln_pre(torch.cat([cls,x],1) + self.positional_embedding)
        for b in self.transformer.resblocks:
            x = b(x)
        # open_clip's default pools after ln_post; LayerNorm is token-local.
        return F.normalize(self.ln_post(x)[:,0] @ self.proj, dim=-1)


class Encoder:
    def __init__(self, device='cpu', path=None):
        from safetensors import safe_open
        self.device = torch.device(device)
        path = Path(path) if path else weights_path()
        got = sha256(path)
        if got != WEIGHT_SHA256:
            raise ValueError(f'CLIP checkpoint hash mismatch: {got}')
        self.model = VisionTower().float().eval()
        with safe_open(str(path), framework='pt', device='cpu') as f:
            state = {k[len('visual.'):]:f.get_tensor(k) for k in f.keys() if k.startswith('visual.')}
        self.model.load_state_dict(state, strict=True)
        self.model.requires_grad_(False).to(self.device)
        self.lock = threading.RLock()

    def encode(self, images):
        # A caller's autocast/TF32 settings must not change retrieval keys.
        # Restore backend flags after the encoder; never alter serving defaults.
        with self.lock, torch.inference_mode(), torch.autocast(device_type=self.device.type, enabled=False):
            old_matmul = torch.backends.cuda.matmul.allow_tf32
            old_cudnn = torch.backends.cudnn.allow_tf32
            try:
                if self.device.type == 'cuda':
                    torch.backends.cuda.matmul.allow_tf32 = False
                    torch.backends.cudnn.allow_tf32 = False
                x = torch.stack([preprocess(a) for a in images]).to(self.device)
                return self.model(x).cpu().float().numpy()
            finally:
                if self.device.type == 'cuda':
                    torch.backends.cuda.matmul.allow_tf32 = old_matmul
                    torch.backends.cudnn.allow_tf32 = old_cudnn


_ENCODERS = {}
_LOCK = threading.RLock()


def shared_encoder(device=None):
    """Process singleton: connection copies never clone a GPU model or a lock."""
    device = device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
    key = (str(device), str(weights_path()))
    with _LOCK:
        if key not in _ENCODERS:
            _ENCODERS[key] = Encoder(device, key[1])
        return _ENCODERS[key]


def guard_local_gpu():
    """Check immediately before a permitted local encoding/latency job; budget <=3 GiB."""
    import subprocess
    free = int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free',
                                      '--format=csv,noheader,nounits'], text=True).splitlines()[0])
    if free < 8000:
        raise RuntimeError(f'local GPU free memory {free} MiB < 8000 MiB')
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(3*2**30/total, .99), 0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return {'free_MiB':free, 'allocator_cap_bytes':3*2**30}
