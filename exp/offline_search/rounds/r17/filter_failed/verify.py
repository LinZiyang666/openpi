"""CPU audit of written artifacts and exact task/init/topology coverage."""
from __future__ import annotations

from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml

from .audit_counts import HERE, read
from .build import cell_id, write
from .prove_retrieval import compare
from .row_filter import load_r10, R17_CANDIDATES, R10_CANDIDATES
from exp.r17_integration.ops.immutable_release import sha


def equal(left,right):
    records=compare(left,right)
    assert all(r['equal'] for r in records),[r for r in records if not r['equal']][:2]
    return len(records)


@lru_cache(2)
def plan_tasks(source):
    root=Path(source)
    path=root/('FULL_EPISODE_PLAN.json' if (root/'FULL_EPISODE_PLAN.json').exists() else 'EPISODE_PLAN.json')
    return {r['task_uid']:r for r in read(path)['tasks']}


def artifacts(cells):
    from exp.r17_integration.search.fit import load_fitted_recipe
    report=[]
    for cell in cells:
        print('Verify frozen parameters '+cell_id(cell),flush=True)
        kept=set(cell['kept_episode_ids'])
        for source,artifact in cell['artifacts'].items():
            checks=0
            if cell['source']=='R17':
                before=read(Path(artifact['original'])/'READY.json')
                assert sha(Path(artifact['original'])/'READY.json')==artifact['original_manifest_sha256']
                assert sha(Path(artifact['original'])/'fit.npz')==artifact['original_sha256']
                old=load_fitted_recipe(artifact['original'],expected_binding=before['receipt']['binding'])
                new=load_fitted_recipe(artifact['path'],expected_binding=before['receipt']['binding'])
                checks+=equal(read(Path(artifact['path'])/'READY.json')['receipt'],before['receipt'])
                for field in ('action','state','task','episode','step','fields','pca','projected','sigma','heads'):
                    checks+=equal(getattr(old,field),getattr(new,field))
                if hasattr(old,'distance_scale'):checks+=equal(old.distance_scale,new.distance_scale)
                episodes=old.episode
                fields=R17_CANDIDATES
            else:
                assert sha(artifact['original'])==artifact['original_sha256']
                old_blob=load_r10(artifact['original']);new_blob=load_r10(artifact['path'])
                old=old_blob['method'].inner;new=new_blob['method'].inner;episodes=old.lib_ep
                for key,value in vars(old_blob['method']).items():
                    if key!='inner':checks+=equal(value,getattr(new_blob['method'],key))
                for key,value in vars(old).items():
                    if key!='tasks':checks+=equal(value,getattr(new,key))
                fields=R10_CANDIDATES
            assert set(new.tasks)=={t for t,T in old.tasks.items() if any(int(episodes[i]) in kept for i in T.rows)}
            for task,T in new.tasks.items():
                original=old.tasks[task]
                positions=np.flatnonzero([int(episodes[i]) in kept for i in original.rows])
                np.testing.assert_array_equal(T.rows,original.rows[positions])
                assert set(map(int,episodes[T.rows]))<=kept
                for name,value in vars(original).items():
                    got=getattr(T,name)
                    if name=='early' and value is not None:
                        for k,v in vars(value).items():checks+=equal(v[positions] if k=='n20' else v,getattr(got,k))
                    else:checks+=equal(value[positions] if name in fields and value is not None else value,got)
            report.append(dict(cell=cell_id(cell),source=source,status='PASS',comparisons=checks))
    return report


def coverage(cells,jobs,baseline):
    expected={cell_id(c):{(str(r['task']),r['init']):r for r in baseline[cell_id(c)]
                          if str(r['task']) not in set(map(str,c['empty_tasks']))} for c in cells}
    seen=defaultdict(Counter)
    for job in jobs:
        if job['domain']=='r10_libero':
            original=read(Path('/home/weiland/trace_runs/os_closed_loop')/f"r10_size_{job['model']}/arms.json")
            for arm in read(Path(job['run'])/'arms.json'):
                size=arm['kwargs']['size'];suite=arm['suite'];short=arm['suite_short']
                old=next(a for a in original if a['arm']==f"r10_{job['model']}_{short}_{size}_A")
                assert sha(arm['yaml'])==sha(old['yaml'])
                assert arm['kwargs']==old['kwargs'] and arm['client_overrides']==old['client_overrides']
                for task,init in read(arm['manifest']):seen[f"{job['model']}/{suite}/{size}"][str(task),init]+=1
            continue
        source=read(Path(job['source'])/'campaign.json');new=read(Path(job['run'])/'campaign.json')
        for field in ('replicas','dp_batching','fastwam_batching','stage_batching','mps','checkpoint','runtime','runtime_identity',
                      'worker_egl_devices','worker_egl_devices_by_node'):
            assert source.get(field)==new.get(field),(job['name'],field)
        assert source.get('execution_worker_nodes',source['worker_nodes'])==new.get('execution_worker_nodes',new['worker_nodes'])
        for file in ('runtime.tar','workers.tar'):
            assert sha(Path(job['run'])/file)==sha(Path(job['source'])/file)
        for t in read(Path(job['run'])/'EPISODE_PLAN.json')['tasks']:
            ex=t['extra'];suite='mt50' if t['experiment']=='metaworld_mt50' else t['experiment']
            key=f"{job['model']}/{suite}/{ex['library_size']}";coordinate=str(ex['task_name']),t['orig_init_state_idx']
            old=expected[key][coordinate]
            assert old['source']==job['source']
            before=plan_tasks(job['source'])[old['uid']]
            for field in ('seed','wire_size','expected_controls','num_trials_per_task','protocol_id','task_name','num_steps',
                          'commit_horizon','model_identity','norm_identity','layout_identity','task_table_identity'):
                assert ex.get(field)==before['extra'].get(field),(job['name'],coordinate,field)
            assert ex['cache_group']=='c' and not ex['corrector'] and ex['miss_mode']=='never'
            seen[key][coordinate]+=1
    for key,coordinates in expected.items():
        assert set(seen[key])==set(coordinates),(key,len(seen[key]),len(coordinates))
        assert set(seen[key].values())=={1}
    return dict(status='PASS',cells=len(expected),simulated_episodes=sum(len(v) for v in expected.values()),
                automatic_failures=sum(50*len(c['empty_tasks']) for c in cells),
                sources_preserved=True,duplicate_or_unplanned_episodes=0)


if __name__=='__main__':
    cells=read(HERE/'CELLS.json')['cells'];jobs=read(HERE/'CAMPAIGNS.json')['jobs']
    checked=artifacts(cells)
    write(HERE/'ARTIFACT_VERIFICATION.json',dict(status='PASS',artifacts=checked))
    report=dict(status='PASS',artifacts=checked,coverage=coverage(cells,jobs,read(HERE/'BASELINE.json')))
    write(HERE/'VERIFICATION.json',report)
    print(report['coverage'])
