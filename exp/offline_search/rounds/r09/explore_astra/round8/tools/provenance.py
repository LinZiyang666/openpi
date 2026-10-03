"""Freeze local provenance; never hash or deserialize mixed-population logs whole."""
import hashlib
import json
import numpy as np
from .safe import HERE,RUNS,STORE,admitted,dump
from .analyze import recipe,ROOTS,clean_json
from exp.offline_search.rounds.r06.ideation_Q1.method_c.common import FitUnpickler


def sha(path):
    h=hashlib.sha256()
    with admitted(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def compare(a,b):
    seen=set();diff=[];count=[0]
    def walk(a,b,path):
        pair=(id(a),id(b))
        if pair in seen:return
        seen.add(pair)
        if isinstance(a,np.ndarray):
            count[0]+=1
            if not isinstance(b,np.ndarray) or not np.array_equal(a,b,equal_nan=a.dtype.kind in 'fc'):
                diff.append((path,'array_mismatch'))
        elif type(a)!=type(b):diff.append((path,str(type(a)),str(type(b))))
        elif isinstance(a,dict):
            for k in sorted(set(a)|set(b),key=str):
                if k not in a or k not in b:diff.append((path+'.'+str(k),'missing'))
                else:walk(a[k],b[k],path+'.'+str(k))
        elif isinstance(a,(list,tuple)):
            if len(a)!=len(b):diff.append((path,'length'))
            else:
                for k,(x,y) in enumerate(zip(a,b)):walk(x,y,path+'.'+str(k))
        elif hasattr(a,'__dict__'):walk(vars(a),vars(b),path)
        elif isinstance(a,(str,int,float,bool,type(None),np.generic)):
            if a!=b and not(isinstance(a,float) and np.isnan(a) and np.isnan(b)):diff.append((path,a,b))
    walk(a,b,'method')
    return dict(arrays_compared=count[0],differences=diff)


def main():
    artifacts=[];sources=[]
    for model in ROOTS:
        p=HERE.parents[1]/'recipe/artifacts'/f'r9eq_{model}_l10_50.pkl'
        q=RUNS/ROOTS[model]/'fits'/p.name
        with admitted(q).open('rb') as f:run=FitUnpickler(f).load()['method']
        cmp=compare(recipe(model),run)
        optional={f'method.{prefix}{key}' for prefix in ('','fit_info.') for key in ('stage_fit','wrist_fit','follow_kwargs','pace_lag')}
        assert all(path in optional and detail==['missing'] for path,*detail in cmp['differences'])
        assert cmp['arrays_compared']==301
        dump(HERE/'results'/f'artifact_compare_{model}.json',clean_json(cmp))
        artifacts.append(dict(model=model,recipe_artifact=str(p),run_artifact=str(q),sha256=sha(p),run_sha256=sha(q),
            numerical_state_equal=True,optional_nonstack_fields_absent=True))
        for root,arm in [(ROOTS[model],f'r9eq_{model}_l10_50'),('r08_main',f'r8_{model}_l10_P10')]:
            base=RUNS/root/'runs'/arm
            paths=[base/'client/journal.jsonl']
            if root!='r08_main':paths+=sorted(base.glob('server_*/decisions*.jsonl'))
            for x in paths:
                st=admitted(x).stat()
                sources.append(dict(path=str(x),size=st.st_size,mtime_ns=st.st_mtime_ns,
                    reader='safe.admitted_jsonl; only init 0..29 payloads decoded'))
        for variant in ('A','CU','IP','P10'):
            arm=f'r8_{model}_l10_'+(variant if variant=='P10' else '50_'+variant)
            p=STORE/'derived/r09_astra/compact'/f'{arm}.npz'
            with np.load(admitted(p),allow_pickle=False) as z:
                t,i=z['task'],z['init']
                assert ((t>=0)&(t<10)&(i>=0)&(i<30)).all()
            st=p.stat();sources.append(dict(path=str(p),size=st.st_size,mtime_ns=st.st_mtime_ns,
                admitted_rows=len(t),identity_sha256=hashlib.sha256(t.tobytes()+i.tobytes()).hexdigest(),
                reader='analyze.compact; entire payload admitted only after exclusive 0..29 identity check'))
    dump(HERE/'results/artifact_provenance.json',artifacts)
    dump(HERE/'PROVENANCE.json',dict(sources=sources,artifacts=artifacts,
        fit_inits=list(range(20)),eval_inits=list(range(20,30)),arms_frozen=0,
        serving_changes=False,new_task_indexed_logic=False,whole_mixed_log_hashes=False))
    print('Verified 602 arrays and all shared serving state; only eight optional non-stack fields differ per model.')


if __name__=='__main__':main()
