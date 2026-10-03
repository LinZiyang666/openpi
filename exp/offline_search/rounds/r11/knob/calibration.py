"""Build-time library-only calibration. Explorer imports never enter a fit graph.

Frozen B-only held-out tables are a SHA-bound intermediate, not parameter input.
Settings and deployable pooled heads are recomputed, then compared to the freeze.
"""
from __future__ import annotations
from functools import lru_cache
import json
from pathlib import Path
import numpy as np
from exp.offline_search.rounds.r10.data import HERE as R10, sha, SubsetLibrary, STORE
from .controller import COSTS

HERE = Path(__file__).resolve().parent
R11 = HERE.parent
CELLS = [(m,s,n) for m in ('pi05','groot') for s,ns in (('l10',(50,200,500)),('spatial',(50,))) for n in ns]


def bound(path):
    path = Path(path).resolve()
    if 'os_closed_loop' in path.parts or ('offline_search_store' in path.parts and 'library' not in path.parts):
        raise ValueError(f'non-library calibration input refused: {path}')
    return path


def checked_npz(path):
    path = bound(path)
    audit = json.loads((HERE/'input_audit.json').read_text())
    expected = next(r['sha256'] for r in audit if r['path'] == str(path))
    if sha(path) != expected:
        raise ValueError(f'frozen B table changed: {path}')
    with np.load(path, allow_pickle=False) as z:
        return {k:np.array(z[k]) for k in z.files}


class ScheduleReplay:
    """Vectorized count-only port of opus's exact keyed eight-replicate solve."""
    def __init__(self, model, episodes, reps=8, runtime=False):
        from exp.offline_search.rounds.r11.opus.ir_model import _u
        from .recipe import uniform
        identities = {}
        if runtime:
            from exp.offline_search.closed_loop.devset import library_episodes, PARENT
            _, metadata, _ = library_episodes(model, episodes[0].extra['suite'], PARENT[model])
            identities = {i: int(e['orig_init_state_idx']) for i,e in enumerate(metadata)}
        self.model, self.reps, self.eps = model, reps, episodes
        H, E = max(len(e.steps) for e in episodes), len(episodes)
        self.alive = np.zeros((H,E),bool)
        self.guard = np.zeros((H,E),bool)
        self.random = np.ones((H,reps,E))
        self.gap = np.ones((H+1,reps,E))
        for j,e in enumerate(episodes):
            L=len(e.steps); self.alive[:L,j]=True; self.guard[:L,j]=e.guard
            for r in range(reps):
                for i in range(L):
                    self.random[i,r,j] = (uniform('R11-random-v1',0,e.task,identities[e.ep],int(e.steps[i]),'knob-anchor')
                        if runtime else _u('r11-random',0,r,e.ep,int(e.steps[i])))
                for i in range(-1,L):
                    self.gap[i+1,r,j] = (uniform('R11-gap-v1',0,e.task,identities[e.ep],-1 if i<0 else int(e.steps[i]),'cap')
                        if runtime else _u('r11-gap',0,r,e.ep,i))
        self.N=sum(e.n_slots for e in episodes)*reps
        self.V=sum(len(e.steps) for e in episodes)*reps
        self.G=sum(e.guard.sum() for e in episodes)*reps

    def counts(self, method, x):
        if method=='off': return int(self.G)
        if method=='random':
            return int(self.G+((self.random<x)&(~self.guard[:,None,:])&self.alive[:,None,:]).sum())
        R,E=self.reps,len(self.eps)
        run=np.zeros((R,E),int); tail=np.zeros((R,E),bool)
        cap=np.floor(x).astype(int)+(self.gap[0] < x-np.floor(x)) if method.startswith('periodic') else None
        M=0
        for i in range(len(self.alive)):
            g=self.guard[i][None]; alive=self.alive[i][None]
            if method.startswith('periodic'):
                k=(~g)&((run>=cap)|tail)
                call=g|k
                run=np.where(call,0,run+1)
                newcap=int(np.floor(x))+(self.gap[i+1]<x-np.floor(x))
                cap=np.where(call,newcap,cap)
                tail=np.broadcast_to(g,(R,E)).copy() if method=='periodic_pgt1' else np.zeros((R,E),bool)
            else:
                k=(~g)&(tail|(self.random[i]<x))
                call=g|k
                tail=k&(~tail)  # a trigger's single follow-up; guard ends segment
            M+=int((call&alive).sum())
        return M

    def ir(self, method, x):
        a,b=COSTS[self.model]
        return (a*self.V+b*self.counts(method,x))/self.N

    def solve(self, method, target):
        lo,hi=(0.,40.) if method.startswith('periodic') else (0.,1.)
        a,b=self.ir(method,lo),self.ir(method,hi)
        if b<a: lo,hi,a,b=hi,lo,b,a
        if target<=a: return lo,a,'floor'
        if target>=b: return hi,b,'ceiling'
        for _ in range(26):
            mid=(lo+hi)/2
            if self.ir(method,mid)<target: lo=mid
            else: hi=mid
        x=(lo+hi)/2
        return x,self.ir(method,x),'ok'


@lru_cache(maxsize=16)
def tables(model,suite,size):
    from exp.offline_search.rounds.r11.opus import ir_model as M
    # Validate cached rows against the selected B episode view and provenance.
    cell=f'{model}_{suite}_{size}'
    a=checked_npz(R11/'astra/data'/cell/'signals.npz')
    pack=checked_npz(R11/'astra/data'/cell/'pack.npz')
    lib=SubsetLibrary(STORE,f'{model}_{suite}',size)
    np.testing.assert_array_equal(a['row'],lib.rows)
    for f in range(5):
        tr=set(a['ep'][a['fold']!=f]); va=set(a['ep'][a['fold']==f])
        assert not tr&va
    provenance=json.loads((R11/'astra/data'/cell/'provenance.json').read_text())
    assert sha(R10/'subsets'/f'{cell}.npy')==provenance['subset_sha256']
    assert sha(R10/'artifacts'/f'r10_{cell}_G.pkl')==provenance['source_fit_sha256']
    assert sha(R10/'astra/cv'/cell/'rows.npz')==provenance['r10_cv_sha256']
    assert sha(R11/'astra/data'/cell/'signals.npz')==provenance['signals_sha256']
    for split in provenance['folds']:
        assert not set(split['train_episodes'])&set(split['heldout_episodes'])
    for f in range(5):
        checked_npz(R11/'opus/out/anchor'/f'{cell}_f{f}.npz')
    eps=M.load_episodes(model,suite,size)
    for e in eps: e.extra['suite']=suite
    return a,pack,eps,ScheduleReplay(model,eps)


def fit_predictor(a):
    from exp.offline_search.rounds.r11.astra.experiment import predictor_fit
    return predictor_fit(a['X'],a['m0'],a['ep'])


def calibrate(model,suite,size,method,target):
    from exp.offline_search.rounds.r11.astra.signals import calibrate as solve, dag
    from exp.offline_search.rounds.r11.astra.analyze import adaptive_calibrate
    a,pack,eps,sched=tables(model,suite,size)
    cell=f'{model}_{suite}_{size}'
    source=R11/('opus/out/arm_grid.json' if method in ('off','random','periodic','periodic_pgt1','random_tail2') else 'astra/freeze.json')
    base_ir=sched.ir('off',0)
    common=dict(library_only=True,cell=cell,target=target,source_sha256=sha(source))
    if method=='off':
        return dict(common,pred_IR_lib=base_ir,v=sched.V/sched.N,g=sched.G/sched.V)
    if method in ('random','periodic','periodic_pgt1','random_tail2'):
        x,ir,status=sched.solve(method,target)
        result=dict(common,setting=round(x,4),unrounded_setting=x,solve_status=status,
                    pred_IR_lib=ir,pred_IR_rounded=sched.ir(method,round(x,4)),v=sched.V/sched.N,g=sched.G/sched.V)
        frozen=[r for r in json.loads(source.read_text()) if (r['cell'],r['method'],r['target'])==(cell,method,target)]
        if frozen: assert result['setting']==frozen[0]['setting'], (cell,method,target,result,frozen[0])
    else:
        if method=='adaptive_error_hybrid':
            if size!=50 or target not in (.32,.40): raise ValueError('adaptive grid limited to size50 .32/.40')
            c=adaptive_calibrate(pack,a['predicted_error'],target,model)
            result=dict(common,dose=c['dose'],threshold=None,tie_probability=None,beta=.5,eta=.2,
                feasible=c['feasible'],pred_IR_lib=c['validation']['owner_ir_mean'],
                score_reference=np.sort(a['predicted_error']),calibration=c)
        else:
            name='predicted_error' if method=='error_hybrid' else method
            c=solve(pack,a[name],target,model,.5 if method=='error_hybrid' else 1.)
            result=dict(common,**{k:c[k] for k in ('dose','threshold','tie_probability','beta','feasible')},
                eta=0.,pred_IR_lib=c['owner_ir'],v=c['v'],g=c['guard_per_look'])
        if not result['feasible']: raise ValueError('target below mandatory guard floor or above ceiling')
        if 'error' in method:
            head=fit_predictor(a)
            frozen_head=json.loads((R11/'astra/data'/cell/'predictor.json').read_text())
            for key in head:
                np.testing.assert_array_equal(head[key],np.array(frozen_head[key]))
            result['predictor']=head
        frozen=[r for r in json.loads(source.read_text())['arms'] if (r['cell'],r['method'],r['target_ir'])==(cell,method,target)]
        if frozen:
            for k in ('dose','threshold','tie_probability','beta','eta'):
                assert result[k]==frozen[0][k],(cell,method,target,k,result[k],frozen[0][k])
    result['frozen_exact']=bool(frozen)
    return result


def rebuild(model,suite,size):
    """Regenerate both explorers' B-only feature/anchor caches into our directory.

    References are used at build time with every output root redirected. Never
    invoke their CLIs, which own protected directories or install global hooks.
    """
    from unittest.mock import patch
    from exp.offline_search.rounds.r11.astra import experiment, analyze
    from exp.offline_search.rounds.r11.opus import loeo_anchor_table
    dest=HERE/'rebuilt/astra'; dest.mkdir(parents=True,exist_ok=True)
    with patch.object(experiment,'HERE',dest): experiment.run_cell(model,suite,size)
    cell=f'{model}_{suite}_{size}'
    for name in ('signals.npz','predictor.json'):
        fresh=dest/'data'/cell/name; old=R11/'astra/data'/cell/name
        if name.endswith('npz'):
            with np.load(fresh) as z, np.load(old) as o:
                for k in z.files: np.testing.assert_array_equal(z[k],o[k])
        else:
            assert json.loads(fresh.read_text())==json.loads(old.read_text())
    with patch.object(loeo_anchor_table,'OUT',HERE/'rebuilt/opus'):
        for f in range(5):
            p,_=loeo_anchor_table.run_job((model,suite,size,f))
            with np.load(p) as z, np.load(R11/'opus/out/anchor'/Path(p).name) as o:
                for k in z.files:
                    if k!='meta_json': np.testing.assert_array_equal(z[k],o[k])
    print('REBUILD_EXACT',cell,flush=True)
