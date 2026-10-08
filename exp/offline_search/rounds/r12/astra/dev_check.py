"""Independent invariants for the recorded-state numerical analysis."""
from .dev_assess import *

checks=[]
frozen=json.loads((HERE/'FINAL_MANIFEST.json').read_text())
assert sha(HERE/'PREDICTION.md')==frozen['files']['PREDICTION.md']
for assessment in sorted(DEST.glob('r12_devlooks_*/*/assessment.json')):
    info_assessment=json.loads(assessment.read_text());cell=info_assessment['cell'];size=info_assessment['size'];dest=assessment.parent
    A=dict(np.load(dest/'looks.npz',allow_pickle=False));R=dict(np.load(dest/'propagated.npz',allow_pickle=False))
    C,lib=load(cell,size);tablepath=HERE/'final_atlas'/f'{cell}_{size}'/'table.npz'
    info=json.loads(tablepath.with_name('fit.json').read_text());assert sha(tablepath)==info['table_sha256']
    table=dict(np.load(tablepath,allow_pickle=False));C['sigma']=table['sigma']
    with (DEST/'fits'/f'{cell}_{size}'/'half.pkl').open('rb') as f:fit=pickle.load(f)
    meta=json.loads((dest/'extract.json').read_text());decisions=json.loads((dest/'decisions.json').read_text())
    for p in meta['manifests']:
        manifest=json.loads(Path(p['path']).read_text());assert sha(p['path'])==p['sha256']
        c=manifest['contract'];pairs={tuple(x) for x in c['pairs']}
        assert set(zip(A['task'].tolist(),A['init'].tolist()))==pairs
        assert not any(i in c['subset']['excluded_inits_by_task'][str(t)] for t,i in pairs)
    next2=table['next2'];valid=np.flatnonzero(next2>=0)
    assert np.all(C['ep'][valid]==C['ep'][next2[valid]])
    assert np.all(C['task'][valid]==C['task'][next2[valid]])
    assert np.all(C['step'][next2[valid]]-C['step'][valid]==2)
    cf=replay(A,C,lib,table,fit,decisions,config=AtlasConfig(required_mass=2.))
    assert not cf['half'].any()
    np.testing.assert_array_equal(cf['spans'],A['span'])
    np.testing.assert_array_equal(cf['corrected'],A['full'])
    assert all(x['first'] is None for x in cf['firsts'])
    for i in np.flatnonzero(R['half']):
        j=R['reference'][i];assert j>=0 and not R['half'][j]
        assert A['uid'][i]==A['uid'][j] and A['task'][i]==A['task'][j] and A['step'][i]-A['step'][j]==2
        assert A['step'][i]>=4 and R['spans'][i-1]==0 and R['lifecycle'][i] and not A['periodic'][i]
        assert R['safe_mass'][i]>=.75 and R['support'][i]>=.99 and R['state_error'][i]<=.2
        np.testing.assert_array_equal(R['raw'][i,:,6],R['corrected'][i,:,6])
        # Poison current v0, keeping only the legal previous full v0 and
        # current wrist in the half-query construction.
        poisoned=A['P'][i].copy();poisoned[:64]=np.nan
        x=half_features(last_full_v0=A['P'][j,:64],current_wrist=poisoned[64:],current_rs=A['rs'][i],last_full_rs=A['rs'][j])
        rr,ww,dd,aa,cc,_=half_query(table,fit,C,int(A['task'][i]),x,int(A['step'][i]))
        np.testing.assert_array_equal(rr,R['rows'][i]);np.testing.assert_array_equal(cc,R['corrected'][i])
    np.testing.assert_array_equal(R['corrected'][~R['half']],A['full'][~R['half']])
    checks.append(dict(cell=cell,size=size,root=info_assessment['root'],looks=len(A['step']),halves=int(R['half'].sum()),
        full_control_identical=True,poisoned_v0_identical=True,half_memory_and_gate_valid=True,
        successor_edges_checked=len(valid),B_pairs_disjoint=True,atlas_hash_unchanged=True))
result=dict(timestamp=now(),checks=checks,prediction_sha256=sha(HERE/'PREDICTION.md'),prediction_unchanged=True,
    affinity=sorted(os.sched_getaffinity(0)),gpu_used=False,
    scripts={p.name:sha(p) for p in HERE.glob('dev_*.py')},reads=sorted(READS))
dump(HERE/'dev_checks.json',result)
print(json.dumps({k:v for k,v in result.items() if k!='reads'},indent=2))
