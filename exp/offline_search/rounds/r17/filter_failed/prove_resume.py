"""CPU continuation coverage/integrity proof plus a clearly synthetic merge test."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import tempfile

from .audit_counts import HERE,read
from .build import write
from .analyze import accepted,collect,continuations
from .sources import r17_runs
from exp.r17_integration.ops.immutable_release import sha


def prove(root,output):
    root=Path(root).resolve(strict=True)
    c=read(root/'campaign.json');selection=c['cache_resume_selection'];source=Path(selection['source'])
    old=read(source/'campaign.json')
    full=read(root/'FULL_EPISODE_PLAN.json');segment=read(root/'EPISODE_PLAN.json')
    tasks={t['task_uid']:t for t in full['tasks']}
    retained=set(selection['retained_task_uids']);missing=set(selection['missing_task_uids'])
    assert len(tasks)==len(full['tasks'])==selection['expected_episodes']
    assert not retained&missing and retained|missing==set(tasks)
    assert len(retained)==len(selection['retained_task_uids'])
    assert len(missing)==len(selection['missing_task_uids'])==len(segment['tasks'])
    assert {t['task_uid'] for t in segment['tasks']}==missing
    assert all(t==tasks[t['task_uid']] for t in segment['tasks'])
    assert full==read(source/('FULL_EPISODE_PLAN.json' if (source/'FULL_EPISODE_PLAN.json').exists() else 'EPISODE_PLAN.json'))
    stable=('filter_failed','runtime','runtime_identity','server_python','checkpoint','fastwam_batching',
            'fastwam_batch_treatment_identity','mps','replicas','driver_runtime','orchestration','cache_plan_name')
    assert all(c.get(k)==old.get(k) for k in stable)
    hashes={}
    def pin(path):
        path=Path(path)
        if path not in hashes:hashes[path]=sha(path)
        return hashes[path]
    assert all((not Path(p).exists()) if s is None else pin(p)==s for p,s in selection['source_pins'].items())
    files=[*('assets/'+str(p.relative_to(root/'assets')) for p in (root/'assets').rglob('*') if p.is_file()),
           'runtime.tar','workers.tar','ORCHESTRATION.json']
    assert all(pin(root/p)==pin(source/p) for p in files)
    job=dict(run=str(root),domain=c['domain'],model=c['model'])
    parent=dict(job,run=str(source))
    observed=collect([job]);actual_n=sum(map(len,observed.values()))
    assert len(retained)<=actual_n<=len(tasks)
    assert str(root) in continuations([parent])
    # Never invent real run outcomes. Only this disposable private fixture gets
    # hypothetical remaining results and copied ancestor rows, to test dedup.
    terminal=accepted(source/'driver/journal.jsonl')
    copied=[terminal[u] for u in retained]
    with tempfile.TemporaryDirectory(prefix='ff_cache_resume_merge_') as tmp:
        fixture=Path(tmp)
        write(fixture/'campaign.json',c)
        write(fixture/'FULL_EPISODE_PLAN.json',full)
        write(fixture/'EPISODE_PLAN.json',segment)
        synthetic=[dict(task_uid=u,accepted=True,status='done',error=None,success=bool(i%2),
                        attempt=1,ts=10**12+i) for i,u in enumerate(sorted(missing))]
        (fixture/'driver').mkdir()
        (fixture/'driver/journal.jsonl').write_text('\n'.join(map(json.dumps,[*copied,*synthetic]))+'\n')
        merged=collect([dict(job,run=str(fixture))])
        assert sum(map(len,merged.values()))==len(tasks)
        assert collect([parent],[fixture])==merged
        expected=Counter(f"{c['model']}/"+('mt50' if t['experiment']=='metaworld_mt50' else t['experiment'])+
                         f"/{t['extra']['library_size']}" for t in tasks.values())
        assert {k:len(v) for k,v in merged.items()}==dict(expected)
        r17_runs.cache_clear()
    result=dict(status='PASS',run=str(root),source=str(source),retained_real_outcomes=len(retained),
        continuation_planned_episodes=len(missing),unique_planned_task_uids=len(tasks),
        retained_missing_overlap=0,current_real_merged_outcomes=actual_n,
        synthetic_merge_fixture_unique_outcomes=len(tasks),synthetic_remaining_outcomes=len(synthetic),
        copied_ancestor_rows_deduplicated=len(copied),per_cell_fixture_counts=dict(expected),
        source_pins_unchanged=True,scientific_campaign_fields_unchanged=list(stable),
        byte_identical_artifacts=len(files),worker_nodes=c.get('execution_worker_nodes',c['worker_nodes']),
        remote_writes=False,server_worker_driver_started=False)
    write(output,result);print(json.dumps(result,indent=2));return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    prove(args.run,args.output or HERE/'continuations'/(args.run.name+'_PROOF.json'))
