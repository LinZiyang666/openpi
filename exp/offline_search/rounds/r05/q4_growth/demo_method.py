"""Demo-bank size ablation: normal AWM refit or frozen deployed-50 fit.

The demo corpus is a different collection from current. Thus frozen50 projects
all demo rows; only the 100/200/300/500 candidate sets are nested with each other.
"""
import copy
import numpy as np
from exp.offline_search.harness import api
from exp.offline_search.rounds.r02.g1_awm.awm import AWM, _Task
from exp.offline_search.rounds.r04.k1_blind.blind_awm import _BlindMixin
from exp.offline_search.rounds.r05.q4_growth.common import (
    load_base, base_artifact, readonly, sha256,
)


def stable_linear(x, matrix, shift):
    """Rowwise GEMV: independent of bank size, batch position, and row order."""
    out=np.empty((len(x),matrix.shape[1]),np.float32)
    transposed=np.ascontiguousarray(matrix.T)
    for i,row in enumerate(x):
        out[i]=transposed@np.ascontiguousarray(row)-shift
    return out


def features(method,obj,rows=None):
    """Use the same scalar PCA GEMV as AWM.query, independent of row order."""
    rows=np.arange(len(obj.rs)) if rows is None else np.asarray(rows)
    out=np.empty((len(rows),136),np.float32)
    for i,r in enumerate(rows):
        out[i,:64]=method.B0T@np.asarray(obj.key_v0[r])-method.muB0
        out[i,64:128]=method.B1T@np.asarray(obj.key_v1[r])-method.muB1
        out[i,128:]=obj.rs[r,:8]
    return out


class DemoAWM(AWM):
    family = 'q4_demo_curve'

    def __init__(self, library='demo200', variant='refit', kref=5):
        if library not in ('current', 'demo100', 'demo200', 'demo300', 'bpool_cs'):
            raise ValueError('unsupported demo library')
        if variant not in ('refit', 'frozen50') or int(kref) != 5:
            raise ValueError('variant=refit|frozen50; this curve holds kref=5 fixed')
        super().__init__(lib='big' if library == 'bpool_cs' else 'current', kref=5)
        self.library, self.variant = library, variant
        self.name = f'AWM_demo_{library}_{variant}_kr5'

    def fit(self, lib, ctx):
        selected = ctx.open_library(self.library)
        if self.library == 'current':
            self._load_frozen(ctx)
            self._add_visual_aux(lib, features(self, lib))
        elif self.variant == 'refit':
            super().fit(selected, ctx)
        else:
            self._load_frozen(ctx)
            self._replace_candidates(lib, selected, ctx)
        self.cand_name = self.os_library = self.library
        self.os_fit_library = 'current' if self.variant == 'frozen50' else self.library
        self.demo_manifest_sha256 = sha256(selected.dir/'manifest.json')
        self.name = f'AWM_demo_{self.library}_{self.variant}_kr5'
        self.prof = api.NULL_PROFILER
        readonly(self)

    def _load_frozen(self, ctx):
        base = load_base(ctx.suite)
        assert np.array_equal(base.act, ctx.open_library('current').action)
        # Do not overwrite K1 mixin settings or its per-episode state.
        library, variant, name = self.library, self.variant, self.name
        self.__dict__.update(copy.deepcopy(vars(base)))
        self.library, self.variant, self.name = library, variant, name
        for task, old in list(self.tasks.items()):
            self.tasks[task] = _Task()
            self.tasks[task].__dict__.update(vars(old))
        self.frozen_artifact = str(base_artifact(ctx.suite))
        self.frozen_artifact_sha256 = sha256(self.frozen_artifact)

    def _add_visual_aux(self, lib, x, centers=None):
        for task,T in self.tasks.items():
            for fi, sl in ((0,slice(0,64)), (1,slice(64,128))):
                center = (x[T.rows,sl].astype(np.float64).mean(0).astype(np.float32)
                          if centers is None else centers[task,fi])
                setattr(T,f'Vm{fi}',center)
                v = x[T.rows,sl].astype(np.float64)-center
                v /= np.maximum(np.linalg.norm(v,axis=1,keepdims=True),1e-12)
                setattr(T,f'V{fi}',v.astype(np.float32))

    def _replace_candidates(self, current, selected, ctx):
        xb=features(self,current)
        centers={(task,fi):xb[T.rows,sl].astype(np.float64).mean(0).astype(np.float32)
                 for task,T in self.tasks.items()
                 for fi,sl in ((0,slice(0,64)),(1,slice(64,128)))}
        x=features(self,selected)
        sig=np.asarray(ctx.action_sigma,np.float64)
        for task,T in self.tasks.items():
            T.rows=np.asarray(selected.rows_of_task(task),np.int64)
            rr=T.rows
            T.Z=stable_linear(x[rr],T.Wf,T.shift)
            T.z2=(T.Z.astype(np.float64)**2).sum(1).astype(np.float32)
            hd=(np.asarray(selected.action[rr,:5,:7],np.float64)/sig).reshape(-1,35)
            T.HD=hd.astype(np.float32);T.h2=(hd**2).sum(1).astype(np.float32)
            rs=np.asarray(selected.rs[rr,:8],np.float64)
            T.RS=rs.astype(np.float32);T.rs2=(rs**2).sum(1).astype(np.float32)
            assert T.Z0 is None and T.As0 is None
            y=T.Z.astype(np.float64)@T.A0.astype(np.float64)
            T.n20=(y*y).sum(1).astype(np.float32)
        self._add_visual_aux(selected,x,centers)
        self.act=np.array(selected.action,copy=True)
        self.lib_ep=np.array(selected.episode,dtype=np.int32,copy=True)
        self.lib_step=np.array(selected.step,dtype=np.int32,copy=True)
        self.n_cand={t:len(T.rows) for t,T in self.tasks.items()}


class DemoBlindAWM(_BlindMixin, DemoAWM):
    """The unchanged K1 BlindAWM mixin over DemoAWM; only anchor-tail is exposed."""
    def __init__(self, library='demo200', variant='refit', kref=5,
                 serving='anchor_tail', budget=1, gates='budget_only'):
        if (serving,budget,gates) != ('anchor_tail',1,'budget_only'):
            raise ValueError('Q4 demo arms require anchor_tail, budget=1, gates=budget_only')
        super().__init__(library=library,variant=variant,kref=kref,serving=serving,budget=budget,gates=gates)

    def fit(self, lib, ctx):
        super().fit(lib,ctx)
        if self.variant=='frozen50':
            # These are diagnostics with budget_only; preserve base-50 scale
            # and motion calibration as well, without changing K1 serving.
            selected_blind={k:getattr(self,k) for k in ('blind_next','blind_rs','blind_event','blind_terminal')}
            self._fit_blind(lib)
            self.__dict__.update(selected_blind)
        self.name=f'BL_anchor_tail_B1_budget_only__AWM_demo_{self.library}_{self.variant}_kr5'
