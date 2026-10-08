"""Meaningful boundary, causality, parity, and accounting checks for D2."""
from .d2_screen import *

def run():
    frozen=json.loads((HERE/'FINAL_MANIFEST.json').read_text())['files']['PREDICTION.md']
    archive=D2/'original_prediction.md'
    if not archive.exists():
        assert sha(HERE/'PREDICTION.md')==frozen
        archive.write_bytes((HERE/'PREDICTION.md').read_bytes())
    assert sha(archive)==frozen and (HERE/'PREDICTION.md').read_bytes().startswith(archive.read_bytes())
    records=[];checks=0
    for p in sorted(D2.glob('r12_devlooks_*/*/screen.json')):
        if p.parent.parent.name not in VALID_ROOTS:continue
        info=json.loads(p.read_text());root,cell,size=(info[k] for k in ('root','cell','size'))
        src=DEST/root/f'{cell}_{size}';dest=p.parent
        A=dict(np.load(src/'looks.npz',allow_pickle=False));Q=dict(np.load(dest/'candidates.npz',allow_pickle=False))
        meta=json.loads((src/'extract.json').read_text());dec=json.loads((src/'decisions.json').read_text())
        C,lib=load(cell,size);table=dict(np.load(HERE/'final_atlas'/f'{cell}_{size}'/'table.npz',allow_pickle=False));C['sigma']=table['sigma']
        for manifest in meta['manifests']:
            assert sha(manifest['path'])==manifest['sha256']
            data=json.loads(Path(manifest['path']).read_text());contract=data['contract']
            assert contract['dev'] and contract['init_pool']=='B'
            assert not any(i in contract['subset']['excluded_inits_by_task'][str(t)] for t,i in contract['pairs'])
        baseline=replay_gate(A,Q,C,lib,dec,dict(witnesses=0,safe_mass=2.),meta.get('periodic_setting'))
        assert not baseline['half'].any()
        np.testing.assert_array_equal(baseline['spans'],A['span'])
        np.testing.assert_array_equal(baseline['calls'],~A['hit'])
        np.testing.assert_array_equal(baseline['corrected'],A['full'])
        assert all(x['first'] is None for x in baseline['firsts'])
        if meta.get('periodic_setting') is not None:
            for i in range(len(A['step'])):
                assert baseline['due'][i]==due_before(A,i,dec[str(A['uid'][i])],meta['periodic_setting'])==A['periodic_due'][i]
        cfg=next(x['config'] for x in info['results'] if x['config']['name']=='consensus2_r0.1')
        R=replay_gate(A,Q,C,lib,dec,cfg,meta.get('periodic_setting'))
        for i in np.flatnonzero(R['half']):
            j=Q['reference'][i]
            assert j>=0 and A['uid'][i]==A['uid'][j] and A['step'][i]-A['step'][j]==2 and not R['half'][j]
            assert not R['due'][i] and R['spans'][i-1]==0 and R['lifecycle'][i]
            assert Q['witness_rms'][i,:2].max()<=.1 and not Q['witness_grip'][i,:2].any()
            np.testing.assert_array_equal(Q['raw'][i,:,6],Q['corrected'][i,:,6])
        with (DEST/'fits'/f'{cell}_{size}'/'half.pkl').open('rb') as f:fit=pickle.load(f)
        with Path(meta['fit_path']).open('rb') as f:base=pickle.load(f)['method'].inner.base
        wrist=fit_wrist(cell,size);past=lag(C,2)
        for i in np.flatnonzero(R['half'])[:3]:
            j=Q['reference'][i];poison=A['P'][i].copy();poison[:64]=np.nan
            c=candidate(saved_p0=A['P'][j,:64],current_wrist=poison[64:],current_rs=A['rs'][i],saved_rs=A['rs'][j],
                step=int(A['step'][i]),task=int(A['task'][i]),C=C,table=table,fit=fit,base=base,wrist=wrist,past=past)
            for k in ('rows','corrected','witness_rows','witness_rms','witness_grip'):np.testing.assert_array_equal(c[k],Q[k][i])
            checks+=1
        price=.076 if cell.startswith('pi05') else .074
        baseline_work=2*price*meta['V']+(1-2*price)*meta['M']
        H=int(R['half'].sum());completion=int((R['half']&~A['hit']).sum())
        mixed_work=2*price*(meta['V']-H)+price*(H+completion)+(1-2*price)*meta['M']
        s=short(A,Q,R,C,meta,dec)
        assert abs(s['saving_fixed_calls']-(baseline_work-mixed_work)/meta['N'])<1e-12
        assert abs((price+price+(1-2*price))-1)<1e-12
        records.append(dict(root=root,cell=cell,size=size,looks=meta['V'],episodes=meta['episodes'],halves=H,
            full_control_parity=True,periodic_prelook_parity=True,source_manifests_unchanged=True,
            B_disjoint=True,legal_half_memory=True,completion_ledger_parity=True,
            full_recording_parity=meta['full_parity'],looks_sha256=sha(src/'looks.npz'),candidate_sha256=sha(dest/'candidates.npz')))
    result=dict(timestamp=now(),frozen_prediction_sha256=frozen,frozen_prediction_bytes=len(archive.read_bytes()),
        frozen_prediction_prefix_unchanged=True,checks=records,poisoned_missing_camera_checks=checks,
        looks=sum(x['looks'] for x in records),episodes=sum(x['episodes'] for x in records),
        affinity=sorted(os.sched_getaffinity(0)),code={p.name:sha(p) for p in HERE.glob('d2_*.py')},reads=sorted(READS))
    dump(D2/'validation.json',result)
    print({k:v for k,v in result.items() if k not in ('checks','reads','code')},flush=True)

if __name__=='__main__':run()
