"""Final immutable-payload/data-boundary audit; does not fit or launch anything."""
from common import *
fence()
import numpy as np
from controller import DistanceFeedback, Ledger, replay
from replay_study import trace_pack, take

def main():
    frozen=read(HERE/'freeze.json');configs=read(HERE/'configs.json')
    assert frozen['no_A_inputs'] and frozen['no_new_variant_closed_loop_results_accessed']
    for name,h in frozen['sources_sha256'].items(): assert sha(HERE/name)==h,(name,'changed')
    grid=frozen['grid'];names=[r['arm'] for r in grid];assert len(names)==len(set(names))
    ids={r['config_key'] for r in configs}
    for r in grid:
        assert r['episodes']==500 and r['init_pool']=='A'
        if r['method'] in ('periodic','distance_feedback'):assert r['config_key'] in ids
        assert r['host']==('local_4090' if r['model']=='groot' else 'h100')
    predictions={r['arm']:r for r in frozen['predictions']};assert set(names)==set(predictions)
    for cell in CELLS:
        for target,reps in ((.25,2),(.32,4),(.4,4),(.45,2)):
            rr=[r for r in grid if r['cell']==cell and r['target']==target]
            assert len(rr)==2*reps
            for rep in range(reps):
                pp=[r for r in rr if r['replicate']==rep]
                assert {r['method'] for r in pp}=={'periodic','distance_feedback'}
                assert len({r['pair_group'] for r in pp})==len({r['host'] for r in pp})==1
                assert abs(pp[0]['global_order']-pp[1]['global_order'])==1
    # Reference replay must match the scalar runtime implementation on every
    # accepted B validation trajectory at the selected frozen settings.
    matched=0;fits={r['config_key']:r for r in configs};source_stamps=0
    for mp in sorted((HERE/'data').glob('*.json')):
        meta=read(mp)
        for s in meta['log_sources']+meta['look_manifests']:
            assert sha(s['path'])==s['sha256'];source_stamps+=1
        if meta['cell'] not in CELLS:continue
        p=trace_pack(meta)
        if p is None:continue
        p=take(p,~p['cal'])
        for target in (.25,.32,.4,.45):
            fit=fits[f'{meta["cell"]}:distance_feedback:{target}']
            cfg={k:fit[k] for k in ('model','target','lo','hi','width','bias')}
            r=replay(p,**cfg)
            for j in range(len(p['N'])):
                c=DistanceFeedback(**cfg); V=M=0
                for i in np.flatnonzero(p['alive'][:,j]):
                    step=int(p['step'][i,j]);out=c.propose(step=step,distance=p['distance'][i,j],
                        guard=bool(p['guard'][i,j]),ledger=Ledger(step,V,M))
                    assert out['miss']==r['calls'][i,j]
                    V+=1;M+=out['miss'];matched+=1
    # The fence must refuse representative forbidden access before filesystem I/O.
    rejected=[]
    for p,mode in [(RUNS/'r11_knob_1/arms.json','r'),
                   (Path('/home/weiland/trace_runs/offline_search_store/queries/pi05_l10_cache/key_v0.npy'),'rb'),
                   (HERE.parent/'outside_write_probe','w')]:
        try:
            with p.open(mode): pass
        except PermissionError:rejected.append(str(p))
        else:raise AssertionError(f'Boundary failed for {p}')
    required=['SPEC.md','PREDICTION.md','OFFLINE.md','PLAN.md','REQUESTS.md','HANDBACK.md','STATUS.md','TEST_DESCRIPTIVE.md']
    for name in required:assert (HERE/name).stat().st_size>100
    counts=read(HERE/'grid_counts.json')
    assert len(grid)==counts['arms'] and sum(r['episodes'] for r in grid)==counts['episodes']
    assert len(frozen['pilot'])==counts['pilot_arms']
    assert all(r['episodes']==100 and r['init_pool']=='B' for r in frozen['pilot'])
    validation=read(HERE/'validation_selection.json');used=read(HERE/'used_b_identities.json')
    assert sha(HERE/'used_b_identities.json')==validation['used_identities_sha256']
    for r in frozen['pilot']:
        pairs=validation['pairs_by_cell'][r['cell']]
        assert len(pairs)==len({tuple(x) for x in pairs})==r['episodes']
        assert all(sum(x[0]==t for x in pairs)==10 for t in range(10))
        suite=r['cell'].split('_')[1]
        seen={tuple(x) for m in COSTS for x in used[f'{m}_{suite}']}
        assert not {tuple(x) for x in pairs}&seen
    out=dict(timestamp=now(),PASS=True,frozen_at=frozen['timestamp'],arms=len(grid),
        configs=len(configs),predictions=len(predictions),runtime_replay_proposals=matched,
        unchanged_source_log_and_manifest_hashes=source_stamps,boundary_refusals=len(rejected),
        independent_B_pairs_verified=True,B_pilot_run=False,A_grid_run=False,closed_loop_target_control_proven=False,
        note='Implementation-ready reference and frozen design; live/plugin verification remains external.')
    dump(HERE/'final_audit.json',out);provenance('audit')
    paths=sorted(p for p in HERE.rglob('*') if p.is_file() and p.name!='MANIFEST.sha256')
    (HERE/'MANIFEST.sha256').write_text(''.join(f'{sha(p)}  {p.relative_to(HERE)}\n' for p in paths))
    print(json.dumps(out,indent=2))

if __name__=='__main__':main()
