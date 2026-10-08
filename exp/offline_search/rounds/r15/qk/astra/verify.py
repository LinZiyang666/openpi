"""Focused verification of dataset boundaries, raw image alignment and math."""
import ast
import json
import hashlib
from pathlib import Path
import h5py
import numpy as np
from exp.offline_search.rounds.r15.qk.astra.prepare import HERE,DERIVED,load_cell,dump
from exp.offline_search.rounds.r15.qk.astra.prototype import teacher_metric,reference_codes

def main():
    evidence={'syntax':[],'cells':{},'source_alignment':[]}
    for f in HERE.glob('*.py'):ast.parse(f.read_text());evidence['syntax'].append(f.name)
    for suite,size in [('l10',50),('l10',200),('spatial',50)]:
        cell=f'pi05_{suite}_{size}';tr=np.load(DERIVED/cell/'train.npz');m,b,p,rows,_=load_cell(suite,size)
        assert np.array_equal(tr['rows'],rows)
        assert set(tr['episode'].tolist())==set(sum(m.episode_subset.values(),[]))
        assert m.fit_info['library_only']
        sigma=np.maximum(tr['action'][:,:5,:7].std((0,1)),1e-6)
        assert np.allclose(sigma,b.sig,rtol=1e-5,atol=1e-6)
        maxerr=0.
        for t,T in b.tasks.items():
            r=np.flatnonzero(tr['task_id']==t);X=np.c_[tr['P0'][r],tr['P1'][r],tr['rs'][r,:8]]
            z=X@T.Wf-T.shift
            delta=float(np.max(np.abs(z-T.Z)));maxerr=max(maxerr,delta)
            assert np.allclose(z,T.Z,rtol=1e-3,atol=5e-3)
        root='r12_devlooks_50L' if suite=='l10' else 'r12_devlooks_50S_local';dp=DERIVED/f'dev_pi05_{suite}_{root}';dev=np.load(dp/'dev.npz');mask=np.load(dp/f'unseen_{size}.npy')
        eps=json.loads((p/'episodes.json').read_text());ep=np.load(p/'episode.npy');ids=set(sum(m.episode_subset.values(),[]));served={(e['task_id'],e['orig_init_state_idx']) for e in eps if int(ep[e['start']]) in ids}
        assert all((int(t),int(i)) not in served for t,i in zip(dev['task_id'][mask],dev['init'][mask]))
        evidence['cells'][cell]={'rows':len(rows),'episodes':len(ids),'sigma_max_abs':float(np.max(np.abs(sigma-b.sig))),'teacher_library_code_max_abs':maxerr,'unseen_looks':int(mask.sum()),'train_npz_sha256':hashlib.sha256((DERIVED/cell/'train.npz').read_bytes()).hexdigest()}
        if size==50:
            # One selected episode per task, both cameras, first row. Read B
            # source inputs only and verify exact stored downsampling.
            for task in range(10):
                i=int(np.flatnonzero(tr['task_id']==task)[0]);pe=int(tr['episode'][i]);e=eps[pe];st=int(tr['step'][i])
                with h5py.File(e['file'],'r') as f:
                    g=f[f'step_{st:04d}']
                    for c,key in enumerate(('base_0_rgb','left_wrist_0_rgb')):
                        im=g['input_images/'+key][()];assert np.array_equal(im[::4,::4],tr[f'img{c}'][i])
                    assert np.array_equal(g['robot_state'][()],tr['rs'][i])
                    assert np.array_equal(g['clean_action'][()],tr['action'][i])
                evidence['source_alignment'].append({'cell':cell,'task':task,'source':e['file'],'row':int(rows[i]),'cameras_exact':2,'state_exact':True,'action_exact':True})
    cost=json.loads((HERE/'cost_counts.json').read_text());assert all(a['ledger_reconciled'] for a in cost['arms'])
    evidence['cost_arms_reconciled']=len(cost['arms']);evidence['passed']=True
    dump(HERE/'verification.json',evidence);print(json.dumps({'passed':True,'cells':evidence['cells'],'source_episodes_checked':len(evidence['source_alignment']),'cost_arms':len(cost['arms'])},indent=2))

if __name__=='__main__':main()
