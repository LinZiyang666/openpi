"""A-only B-val recording adapter: exclude the matching acquisition episode.

The deployed metric/kernel stays frozen. This class is for calibration recording
only and never appears in a validation C spec.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from exp.offline_search.rounds.r04.k1_blind.blind_awm import BlindAWM
from .common import load_base, read_bank, sha


class ExcludedRecordingA(BlindAWM):
    def __init__(self, bank_path, exclusions_path, exclusions_sha256):
        super().__init__(serving='anchor_tail',budget=1,gates='budget_only')
        self.recording_bank_path=str(bank_path)
        self.recording_exclusions_path=str(exclusions_path)
        self.recording_exclusions_sha256=exclusions_sha256

    def fit(self, lib, ctx):
        meta=json.loads(Path(self.recording_bank_path).read_text())
        base,_=load_base(meta['source'])
        own={k:v for k,v in self.__dict__.items() if k.startswith('recording_')}
        self.__dict__.update(base.__dict__);self.__dict__.update(own)
        read_bank(self.recording_bank_path,self)
        self.fit_info={**getattr(self,'fit_info',{}), 'q1_r_bank_sha256':sha(self.recording_bank_path),
            'q1_retrieval_fingerprint':meta['retrieval_fingerprint']}
        if sha(self.recording_exclusions_path)!=self.recording_exclusions_sha256:
            raise ValueError('recording exclusion map hash mismatch')
        self.recording_exclusions=json.loads(Path(self.recording_exclusions_path).read_text())
        if self.recording_exclusions['role']!='NONTEST_BVAL':
            raise ValueError('recording exclusion map must be B-val')
        if self.recording_exclusions['cell'].rsplit('_',1)[0]+'_cache'!=ctx.cell:
            raise ValueError('recording library cell mismatch')

    def _dist(self,q):
        result=list(super()._dist(q))
        task,init=int(q.task_id),int(q.episode.init)
        record=self.recording_exclusions['tasks'][str(task)]
        if init!=record['init']:
            raise ValueError('recording tried an initialization outside its frozen B-val selection')
        excluded=record['source_episodes']
        if excluded:
            mask=np.isin(self.lib_ep[result[0].rows],excluded)
            d=np.array(result[7],copy=True);dt=np.array(result[-1],copy=True)
            d[mask]=np.inf;dt[mask]=np.inf
            if np.isfinite(dt).sum()<self.k:raise ValueError('source exclusion leaves insufficient candidates')
            result[7]=d;result[8]=float(np.median(d[~mask]))+1e-12;result[-1]=dt
        return tuple(result)
