"""Same-input B/Bmech replay, including blind LOOK and policy-tail lifecycle.

Recorded keys/states and same-observation recorded policy chunks drive one common
history using the specified controller's source decisions. Judges share that history,
not claimed to generate the same future closed-loop trajectory after a changed
call. No policy model, simulator, plugin installation, network, or git is used.
"""
import argparse
import copy
import json
from pathlib import Path
import pickle
from types import SimpleNamespace as NS

import numpy as np

from exp.offline_search.closed_loop.plugin import load_method_class
from exp.offline_search.closed_loop.blind import BlindResult, LookReason
from exp.offline_search.harness import api, store
from .fit_models import sha

HERE=Path(__file__).resolve().parent
ROOT=Path('/home/weiland/trace_runs/offline_search_store')


def equal(a,b):
    if isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:equal(a[k],b[k])
    elif isinstance(a,np.ndarray):
        assert a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal(x,y)
    elif isinstance(a,(float,np.floating)):
        assert np.float64(a).tobytes()==np.float64(b).tobytes(),(a,b)
    else:assert a==b,(a,b)


def result_equal(a,b):
    for k in ('topk','scores','confidence','action','library','extras'):equal(getattr(a,k),getattr(b,k))


def blind_equal(a,b):
    assert type(a)==type(b)
    for k in (('action','rows','weights','library','extras') if isinstance(a,BlindResult) else ('code','name')):
        equal(getattr(a,k),getattr(b,k))


class History:
    def __init__(self,array,rows):self.array,self.rows=array,rows
    def __getitem__(self,index):
        rows=self.rows[index]
        return self.array[rows] if not isinstance(rows,list) else self.array[np.array(rows,dtype=int)]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--name',required=True)
    ap.add_argument('--driver',choices=['B','Bmech'],default='Bmech');args=ap.parse_args()
    note=next(r for r in json.loads((HERE/'bmech_artifacts.json').read_text()) if r['name']==args.name)
    spec=next(r for r in json.loads((HERE/'emit_arms_bmech.json').read_text()) if r['name']==args.name)
    cls,_=load_method_class(spec['method']);cls(**spec['kwargs'])
    old=note['source_spec'];source_cls,_=load_method_class(old['method']);source_cls(**old['kwargs'])
    with open(note['source_fit'],'rb') as f:base=pickle.load(f)['method']
    with open(note['artifact'],'rb') as f:masked=pickle.load(f)['method']
    unmasked=copy.deepcopy(masked);unmasked._q3_mask_no_progress=False
    for method in (base,masked,unmasked):method.prof=api.NULL_PROFILER
    qc=store.QueryCell(ROOT,f'{spec["model"]}_{spec["suite"]}_cache')
    lib=store.LibraryView(ROOT,store.lib_key(qc.cell),base.base.cand_name)
    R=int(lib.meta['exec_steps'])
    chosen=[(i,e) for i,e in enumerate(qc.episodes) if int(e['init']) in (0,49)]
    assert len(chosen)==20
    count=dict(episodes=0,decisions=0,vision_queries=0,blind_comparisons=0,policy_tail_comparisons=0,
               noprog_look_vetoes=0,noprog_flagged=0,masked_miss_decisions=0,multiple_flag_masks=0)
    for ei,e in chosen:
        episode=api.EpisodeView(e['uid'],e['task'],e['task_id'],e['init'],ei,0)
        for method in (base,masked,unmasked):method.reset(episode)
        n=e['end']-e['start'];actions=np.empty((n,*qc.a_inf.shape[1:]),dtype=qc.a_inf.dtype)
        hits=np.zeros(n,np.int8);vision=np.zeros(n,bool);keyrows=[];age=0
        for step,row in enumerate(range(e['start'],e['end'])):
            q=NS(task_id=e['task_id'],step=step,episode=episode,model=spec['model'],
                rs=qc.rs[row],raw_state=qc.raw_state[row],key_v0=qc.key_v0[row],key_v1=qc.key_v1[row],
                prev_hit=bool(hits[step-1]) if step else None,prev_a_exec=actions[step-1] if step else None,
                hist_a_exec=actions[:step],hist_hit=hits[:step],hist_has_vision=vision[:step],
                hist_rs=qc.rs[e['start']:row],hist_key_v0=History(qc.key_v0,keyrows),
                hist_key_v1=History(qc.key_v1,keyrows),blind_age=age,executed_steps=R)
            blind=None
            if step and not q.prev_hit:
                triplet=[m.policy_tail_step(q) for m in (base,masked,unmasked)]
                blind_equal(triplet[0],triplet[1]);blind_equal(triplet[0],triplet[2])
                count['policy_tail_comparisons']+=1
                if isinstance(triplet[0],BlindResult):blind=triplet[0]
            if step and blind is None:
                triplet=[m.blind_step(q) for m in (base,masked,unmasked)]
                blind_equal(triplet[0],triplet[1]);blind_equal(triplet[0],triplet[2])
                count['blind_comparisons']+=1
                if isinstance(triplet[0],BlindResult):blind=triplet[0]
                elif triplet[0].name=='noprog_span':count['noprog_look_vetoes']+=1
            if blind is not None:
                actions[step]=blind.action;hits[step]=1;vision[step]=False
                keyrows.append(keyrows[-1]);age+=1
            else:
                ref=base.query(q);got=masked.query(q);identity=unmasked.query(q)
                result_equal(ref,identity)
                flags=int(ref.extras['os_flags']);new=flags&~8
                expected=copy.copy(ref)
                if new!=flags:
                    expected.extras={**ref.extras,'os_flags':float(new),'os_reason':float((new&-new).bit_length()),
                                     'os_force_miss':float(bool(new))}
                    count['noprog_flagged']+=1
                    count['masked_miss_decisions']+=int(new==0)
                    count['multiple_flag_masks']+=int(new!=0)
                result_equal(expected,got)
                for other in (masked,unmasked):
                    equal(base._s,other._s)
                    equal(base._vision_progress,other._vision_progress)
                    equal(base._noprog_span,other._noprog_span)
                    equal(base.base._anchor,other.base._anchor)
                driving=got if args.driver=='Bmech' else ref
                miss=bool(driving.extras['os_force_miss'])
                actions[step]=qc.a_inf[row] if miss else ref.action
                hits[step]=not miss;vision[step]=True;keyrows.append(row);age=0
                count['vision_queries']+=1
            count['decisions']+=1
        count['episodes']+=1
    assert count['noprog_flagged']>0 and count['masked_miss_decisions']>0
    assert count['policy_tail_comparisons']>0
    if args.driver=='Bmech':assert count['noprog_look_vetoes']>0
    result=dict(name=args.name,driver=args.driver,assertions='PASS',**count,source_fit_sha256=sha(note['source_fit']),
        bmech_fit_sha256=sha(note['artifact']),scope=f'recorded observations; common {args.driver}-driven history; no SR/counterfactual claim',
        checks='unmasked exact B; masked only three verdict fields; diagnostic histories/cache action/LOOK/tail bit-identical')
    (HERE/('bmech_replay_'+args.name+'_'+args.driver+'.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
