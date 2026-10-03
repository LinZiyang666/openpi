"""A's unchanged fit/query/blind code with CLIP keys and PCA caches substituted."""
from __future__ import annotations

import json
from pathlib import Path
import time
from types import FunctionType

import numpy as np

from exp.offline_search.harness import api
from exp.offline_search.rounds.r02.g1_awm import awm
from exp.offline_search.rounds.r04.k1_blind.blind_awm import BlindAWM

DERIVED = Path('/home/weiland/trace_runs/offline_search_store/derived/clip_vitb32')


class _ClipQuery:
    def __init__(self, q, keys, previous):
        self.q = q
        self.key_v0, self.key_v1 = keys
        self.previous_available = previous is not None and previous[0] == q.episode.uid and previous[1] == q.step-1
        self.hist_key_v0 = (previous[2] if self.previous_available else self.key_v0)[None]
        self.hist_key_v1 = (previous[3] if self.previous_available else self.key_v1)[None]

    def __getattr__(self, k):
        return getattr(self.q, k)


class ClipAWM(BlindAWM):
    family = 'r8_clip_key'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.fit_data != 'same' or self.features != 'joint' or not self.early or self.codes:
            raise ValueError('CLIP ablation preserves A joint/full-rank/early/fit_data=same')
        self.name = 'CLIP_ViTB32__' + self.name
        self._previous_clip = None

    def fit(self, lib, ctx):
        root = Path(ctx.root)/'derived/clip_vitb32'
        deployed = lib if self.lib == 'current' else ctx.open_library(awm.BIG[ctx.model])
        def current(L, key, camera):
            d = root/key/L.dir.name/camera
            return tuple(np.array(np.load(d/f'{n}.npy'),np.float32,copy=True) for n in ('mean','basis','proj'))
        from .encoder import RECIPE, WEIGHT_SHA256
        import hashlib
        for v in ('v0','v1'):
            path = root/ctx.lib_key/deployed.dir.name/v/'meta.json'
            if not path.exists():
                raise api.ContractError(f'CLIP fit unavailable for {ctx.lib_key}/{deployed.dir.name}: '
                                        'requires exact source images and prepared CLIP PCA cache; see CLIP.md')
            meta = json.loads(path.read_text())
            if (meta['n'] != deployed.L or meta['encoder_recipe'] != RECIPE
                    or meta['weight_sha256'] != WEIGHT_SHA256
                    or meta['ids_sha256'] != hashlib.sha256((deployed.dir/'ids.json').read_bytes()).hexdigest()):
                raise api.ContractError('CLIP PCA cache identity mismatch')
        fit = FunctionType(awm.AWM.fit.__code__, {**awm.AWM.fit.__globals__, 'PCA_BIG':root, 'pca_current':current},
                           argdefs=awm.AWM.fit.__defaults__, closure=awm.AWM.fit.__closure__)
        fit(self,lib,ctx)
        self._fit_blind(deployed)

    def reset(self, episode):
        super().reset(episode)
        self._previous_clip = None

    def query(self, q):
        from .encoder import shared_encoder
        encoder = shared_encoder()
        # Host wall time includes preprocessing, both cameras, GPU completion and
        # D2H. Encoder lookup/model loading is a one-time setup cost, excluded.
        t = time.perf_counter()
        keys = encoder.encode([q.img0,q.img1])
        encode_ms = (time.perf_counter()-t)*1e3
        cq = _ClipQuery(q,keys,self._previous_clip)
        res = super().query(cq)
        if not cq.previous_available:
            res.extras.pop('still',None)
        # Insert first: the plugin scalar cap must never drop the latency field.
        res.extras = {'os_clip_encode_ms':encode_ms, **res.extras}
        self._previous_clip = (q.episode.uid,int(q.step),keys[0],keys[1])
        return res
