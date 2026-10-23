"""Frozen CPU student; factories consume the complete online look interface."""
from __future__ import annotations
import hashlib
import importlib
import os
from pathlib import Path
import threading
import numpy as np
import torch
from torch import nn


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while b := f.read(8 << 20):
            h.update(b)
    return h.hexdigest()


class TinySpatial(nn.Module):
    def __init__(self, pm, ps, rm, rs):
        super().__init__()
        for k, v in [('pm', pm), ('ps', ps), ('rm', rm), ('rs', rs)]:
            self.register_buffer(k, torch.as_tensor(v, dtype=torch.float32))
        self.conv = nn.Sequential(nn.Conv2d(3, 16, 5, 2, 2), nn.SiLU(),
            nn.Conv2d(16, 24, 3, 2, 1), nn.SiLU(), nn.Conv2d(24, 32, 3, 2, 1), nn.SiLU())
        self.task = nn.Embedding(10, 8)
        self.fc = nn.Sequential(nn.Linear(32 * 7 * 7 * 2 + 16, 256), nn.SiLU(), nn.Linear(256, 128))
        self.action = nn.Linear(128, 35)

    def forward(self, im, rs, task):
        n = im.shape[0]
        z = self.conv(im.reshape(n * 2, 3, 56, 56) * 2 - 1).reshape(n, -1)
        p = self.fc(torch.cat([z, torch.clamp((rs - self.rm) / self.rs, -8, 8), self.task(task)], 1))
        return p * self.ps + self.pm, self.action(p)


class FeatureExport(nn.Module):
    """Deployment export omits the training-only action auxiliary head."""
    def __init__(self, student):
        super().__init__()
        self.student = student
    def forward(self, images, state, task_id):
        return self.student(images,state,task_id)[0]


def inputs(images, state, task):
    im = np.asarray(images)
    if im.dtype != np.uint8 or im.shape != (2, 224, 224, 3):
        raise ValueError('student requires agentview/wrist RGB uint8 [2,224,224,3]')
    rs = np.asarray(state, np.float32)
    if rs.ndim != 1 or len(rs) < 8 or not np.isfinite(rs).all() or not 0 <= int(task) < 10:
        raise ValueError('invalid normalized student state/task')
    return (torch.from_numpy(im[:, ::4, ::4].transpose(0, 3, 1, 2).copy())[None].float() / 255.,
            torch.from_numpy(rs[:8].copy())[None], torch.tensor([int(task)], dtype=torch.long))


class CNNStudent:
    def __init__(self, model_path, model_sha256, output_index=0, **unused):
        self.model = torch.jit.load(str(model_path), map_location='cpu').eval()
        self.output_index = output_index

    def encode(self, look):
        with torch.inference_mode():
            out = self.model(*inputs(np.stack([look.img0, look.img1]), look.rs, look.task_id))
            if self.output_index is not None:
                out = out[self.output_index]
            return out[0].numpy().reshape(2, 64)


class FrozenEncoder:
    def __init__(self, spec):
        if spec.get('device') != 'cpu':
            raise ValueError('only CPU light feature exports are supported')
        threads = int(spec.get('threads', 1))
        if threads < 1:
            raise ValueError('serving CPU encoder needs positive Torch threads')
        torch.set_num_threads(threads)
        if torch.get_num_interop_threads() != 1:
            torch.set_num_interop_threads(1)
        module, cls = spec['factory'].split(':')
        self.student = getattr(importlib.import_module(module), cls)(**spec['kwargs'])
        self.lock = threading.RLock()
        self.spec = spec
        self.projection = None
        if spec.get('projection_path'):
            self.projection = dict(np.load(spec['projection_path'],allow_pickle=False))

    def encode_raw(self, look):
        with self.lock, torch.inference_mode():
            before = os.sched_getaffinity(0)
            chosen = before & set(self.spec.get('cpu_affinity', before))
            if not chosen:
                raise ValueError('configured light-encoder CPU affinity is unavailable')
            os.sched_setaffinity(0, chosen)
            try:
                out = np.asarray(self.student.encode(look), np.float32)
            finally:
                os.sched_setaffinity(0, before)
        shape = tuple(self.spec.get('raw_feature_shape',self.spec['feature_shape']))
        if out.shape != shape or not np.isfinite(out).all():
            raise ValueError('frozen encoder returned invalid light features')
        return out.copy()

    def encode(self, look):
        out = self.encode_raw(look)
        if self.projection is not None:
            out = (out.ravel() - self.projection['mean']) @ self.projection['basis']
        if out.size != 128:
            raise ValueError('student requires a fitted library-only 128-D projection')
        return out.reshape(2,64).copy()
