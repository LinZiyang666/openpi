"""Outcome-blind B-val selection, state overlap audit, and recording arm specs."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import yaml
import torch
from .common import HERE,OUT,STORE,CONTROLLER_VERSION,sources,sha,write_json


def load(path):
    return np.asarray(torch.load(path,weights_only=False))


def main():
    root=HERE.parents[5]
    specs=[];audit={};schedules=[]
    from exp.offline_search.rounds.r06.p3_profiling.campaign import make_kwargs
    for suite,full in [('l10','libero_10'),('spatial','libero_spatial')]:
        split_path=root/f'exp/ablation_study/config/common/split_{full}.yaml'
        split=yaml.safe_load(split_path.read_text())
        pool=root/f'exp/common/data/db_init/libero/{full}'
        apool=root/f'exp/common/data/db_init/libero/{full}_apool'
        cache_pool=root/f'exp/common/data/db_init/libero_cache/{full}'
        if list(pool.glob('*.pruned_init')):raise ValueError('B pool shadowed by test init extension')
        digests={p.stem:sha(p) for p in sorted(pool.glob('*.init'))}
        rollup=__import__('hashlib').sha256(''.join(f'{k}:{digests[k]}' for k in sorted(digests)).encode()).hexdigest()
        tasks={}
        for key,value in sorted(split.items()):
            task=int(key.split('_')[-1]);init=min(value['val']);stem=value['init_stem']
            b=load(pool/(stem+'.init'));a=load(apool/(stem+'.init'));cache=load(cache_pool/(stem+'.init'))
            canonical=lambda rows:{np.ascontiguousarray(row,dtype=np.float64).tobytes() for row in rows}
            if canonical(b)&canonical(a):raise ValueError('B pool intersects official test states')
            protected=[]
            for c in cache:
                matches=np.flatnonzero(np.all(np.isclose(b,c,atol=1e-7),axis=1))
                if len(matches)!=1:raise ValueError('historical cache init does not identify one B state')
                protected.append(int(matches[0]))
            if sorted(protected)!=sorted(value['protected_in_train']) or init in protected:
                raise ValueError('B-val selection intersects historical cache initialization')
            tasks[task]=dict(task=task,init=init,init_stem=stem,protected=sorted(protected),
                state_sha256=__import__('hashlib').sha256(np.ascontiguousarray(b[init],dtype=np.float64).tobytes()).hexdigest(),
                pool_count=len(b),official_test_count=len(a),shared_states=0,cache_init_count=len(cache))
        record=dict(schema='r6.q1.non_test_pool.v1',role='NONTEST_BVAL',suite=full,total_inits=sum(t['pool_count'] for t in tasks.values()),
            apool_dir=f'/scratch/zixuans8/openpi_trace/exp/common/data/db_init/libero/{full}',
            per_task_digests=digests,rollup_sha256=rollup,
            note='Legacy driver flag is --apool-record; this bound pool is B, CALIBRATION ONLY, not official test.',
            split_sha256=sha(split_path),disjoint_from_official_test=True)
        pool_record=HERE/f'bval_pool_{suite}.yaml'
        pool_record.write_text(yaml.safe_dump(record,sort_keys=False))
        audit[suite]=dict(split=str(split_path),split_sha256=sha(split_path),pool=str(pool),tasks=tasks)
        for cell,source in sources().items():
            if cell.rsplit('_',1)[0].split('_',1)[1]!=suite:continue
            model,_,size=cell.split('_')
            bank=json.loads((OUT/cell/'r_bank.json').read_text())
            entries=json.loads((Path(bank['library_directory'])/'episodes.json').read_text())
            exclusion={}
            provenance={}
            if model=='groot':
                # The collector's documented global episode identity is
                # task*trials+original init. Lane result files are partial after
                # resharding; use the collector contract, validating every
                # surviving explicit result and every bank task identity.
                acquisition=Path(entries[0]['file']).parent.parent
                explicit={}
                evidence_paths=[root/'exp/libero_groot/launch_collection.sh',
                    root/'exp/libero_groot/report_collection.py',root/'examples/libero/collect_util.py']
                for p in sorted(acquisition.glob('results_lane*.json')):
                    evidence_paths.append(p)
                    for r in json.loads(p.read_text()):
                        key=(int(r['task_id']),int(r['episode_id']))
                        original=int(r['orig_init_state_idx'])
                        assert r['episode_id']==r['task_id']*50+original
                        if key in explicit:assert explicit[key]==original
                        explicit[key]=original
                recovered=[]
                for e in entries:
                    task=int(e['task_id']);eid=int(e['episode_id'])
                    assert eid//50==task
                    init=eid%50
                    if e.get('orig_init_state_idx') is not None:assert e['orig_init_state_idx']==init
                    if (task,eid) in explicit:assert explicit[task,eid]==init
                    recovered.append(init)
                provenance=dict(contract='episode_id = task_id * num_trials_per_task + episode_idx; collection trials=50, B pool, no remapping',
                    evidence={str(p):sha(p) for p in evidence_paths},explicit_result_rows=len(explicit),
                    bank_rows_checked=len(entries),orig_init_indices=recovered)
            for task,t in tasks.items():
                if size=='500' or model=='groot':
                    matches=[i for i,e in enumerate(entries) if e['task_id']==task and
                        (recovered[i] if model=='groot' else e.get('orig_init_state_idx'))==t['init']]
                    if size=='500' and len(matches)!=1:raise ValueError(f'{cell} task {task}: expected exactly one B acquisition episode')
                    evidence='collector identity contract, validated against explicit lane results' if model=='groot' else 'episodes.json: explicit task/orig_init_state_idx; B acquisition library provenance'
                else:
                    matches=[]
                    evidence='historical cache .init states matched B protected_in_train, disjoint from chosen B-val state'
                exclusion[str(task)]=dict(init=t['init'],source_episodes=matches,
                    stems=[entries[i]['stem'] for i in matches],evidence=evidence)
            exc=dict(role='NONTEST_BVAL',cell=cell,tasks=exclusion,library_episodes_sha256=bank['library_hashes']['episodes.json'],init_provenance=provenance)
            audit[suite].setdefault('libraries',{})[cell]=dict(overlapping_tasks=[int(t) for t,v in exclusion.items() if v['source_episodes']],exclusions=exclusion,init_provenance=provenance)
            write_json(OUT/cell/'recording_exclusions.json',exc)
            manifest=dict(role='NONTEST_BVAL',cell=cell,model=model,suite=suite,pool_sha256=rollup,
                pool_record=str(pool_record),pool_directory=record['apool_dir'],split_sha256=sha(split_path),
                exclusions_sha256=sha(OUT/cell/'recording_exclusions.json'),
                selected=[dict(task=t,init=r['init'],state_sha256=r['state_sha256']) for t,r in tasks.items()])
            mf=HERE/f'calibration_manifest_{cell}.json';write_json(mf,manifest)
            name=f'r6c_cal_{cell}'
            short=('sp' if suite=='spatial' else suite)+'_'+size
            kwargs=make_kwargs(model,short,p=0.,replicate=0,baseline='A')
            kwargs.update(base_spec='exp.offline_search.rounds.r06.ideation_Q1.method_c.recording:ExcludedRecordingA',
                base_fit='',base_kwargs=dict(bank_path=f'<CAL>/{cell}/r_bank.json',
                    exclusions_path=f'<CAL>/{cell}/recording_exclusions.json',exclusions_sha256=manifest['exclusions_sha256']),
                design=dict(cohort='A',cohort_probability=1.),blind_shadow=True,resample_p=0.,resample_draws=1,
                provenance=dict(q1_controller_version=CONTROLLER_VERSION,q1_recording_role='NONTEST_BVAL',q1_pool_sha256=rollup,q1_selection_sha256=sha(mf),
                    population='one lowest-numbered B-val initial state/task; outcome-blind',block=0),calibration_path='')
            specs.append(dict(name=name,model=model,suite=suite,mode='plugin',
                method='exp.offline_search.rounds.r06.p3_profiling.v2:Profile',kwargs=kwargs,
                full_model=True,cost_ledger=True,client_overrides=dict(replan_steps=5,resize_size=256 if model=='groot' else 224),
                manifest=f'<RUN>/manifests/calibration_manifest_{cell}.json',
                plugin_args=['--os-root',str(STORE),'--os-blind','--os-policy-tail','--os-policy-tail-blocks','1',
                    '--os-judge','guard_only','--os-no-shadow-native','--os-log-inputs','--os-fit-artifact',f'<RUN>/fits/{name}.pkl']))
            schedules.append(dict(arm=name,manifest=f'<RUN>/manifests/calibration_manifest_{cell}.json',
                client_launcher='exp/offline_search/rounds/r06/p3_profiling/run_arm_v2.sh',
                client_env=dict(P3_ENV_SEED='1603'),
                extra_driver_args=['--apool-record',f'<REMOTE_RUN>/bval_pool_{suite}.yaml','--apool-dir',record['apool_dir']],
                note='extra args are last on stock launcher CLI and override its A-pool defaults; hash binding remains enabled'))
    write_json(HERE/'bval_membership_audit.json',audit)
    write_json(HERE/'emit_calibration_recordings.json',specs)
    write_json(HERE/'calibration_schedule.json',schedules)
    print(json.dumps(dict(cells=len(specs),episodes=10*len(specs),patch_required=False)))


if __name__=='__main__':main()
