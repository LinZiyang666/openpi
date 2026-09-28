"""Independent admission audit and edge/admission counterexamples."""
import json
import numpy as np
from exp.offline_search.harness import store
from exp.offline_search.closed_loop.ops.remote.run_gtp_subset import load_manifest,check_manifest
from exp.offline_search.rounds.r05.q4_growth.common import OUT,SHM,load_base,features,fingerprint
from exp.offline_search.rounds.r05.q4_growth.build import topology


def main():
    x=np.zeros(136,np.float32);a=np.zeros((10,32),np.float32)
    assert fingerprint(0,x,a)==fingerprint(0,x.copy(),a.copy())
    b=a.copy();b[9,6]=1
    assert fingerprint(0,x,a)!=fingerprint(0,x,b)  # identical head, distinct full valid action
    b=a.copy();b[:,7:]=999
    assert fingerprint(0,x,a)==fingerprint(0,x,b)  # padding is irrelevant
    assert fingerprint(0,x,a)!=fingerprint(1,x,a)
    y=x.copy();y[128]=1
    assert fingerprint(0,x,a)!=fingerprint(0,y,a)
    top=topology(np.array([0,0,0,1]),np.array([0,2,3,0]),np.array([4,4,4,1]))
    assert top['next'].tolist()==[-1,2,-1,-1] and top['prev'].tolist()==[-1,-1,1,-1]
    assert top['is_terminal'].tolist()==[False,False,True,True]
    man=load_manifest(OUT/'evaluation_pairs.json')
    assert set(man['selected'])=={(t,i) for t in range(10) for i in range(25,50)}
    for suite in ('l10','spatial'):check_manifest(man,'pi05',suite)
    audit=[]
    for suite in ('l10','spatial'):
        key=f'pi05_{suite}';M=load_base(suite);L=store.LibraryView(SHM,key);G=store.LibraryView(SHM,key,'grow250');Q=store.QueryCell(SHM,key+'_inf')
        eps=sorted([e for e in Q.episodes if 0<=e['init']<25 and e['success']],key=lambda e:(e['init'],e['task_id']))
        rows=np.concatenate([np.arange(e['start'],e['end']) for e in eps])
        bx=features(M,L);sx=features(M,Q,rows)
        # Audit in the whitened frozen-code representation as well. It gives
        # the same admission set as exact frozen PCA/state equality here.
        bt=np.asarray(L.task_id);st=np.array([Q.episodes[e]['task_id'] for e in Q.ep[rows]])
        bz=np.empty_like(bx);sz=np.empty_like(sx)
        for t,T in M.tasks.items():
            bz[bt==t]=bx[bt==t]@T.Wf-T.shift
            sz[st==t]=sx[st==t]@T.Wf-T.shift
        seen=set()
        for r in range(L.L):seen.add(fingerprint(bt[r],np.r_[bz[r],bx[r,128:]],L.action[r]))
        kept=[];dup=0
        for i,r in enumerate(rows):
            fp=fingerprint(st[i],np.r_[sz[i],sx[i,128:]],Q.a_inf[r])
            if fp in seen:dup+=1
            else:seen.add(fp);kept.append(r)
        assert np.array_equal(G.source_query_row[L.L:],kept)
        audit.append(dict(suite=suite,successful_source_rows=len(rows),whitened_exact_duplicates=dup,
                          admission_equal=True,per_task=[dict(task=t,base_rows=int((L.task_id==t).sum()),
                            admitted_rows=int((G.task_id[L.L:]==t).sum()),
                            successful_source_episodes=sum(e['task_id']==t for e in eps)) for t in range(10)]))
    result=dict(pass_all=True,full_tail_counterexample=True,padding_counterexample=True,task_state_counterexamples=True,
                missing_edge_counterexample=True,manifest_pairs=len(man['selected']),manifest_selection_sha256=man['sha256'],audit=audit)
    (OUT/'results'/'semantic_checks.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))

if __name__=='__main__':main()
